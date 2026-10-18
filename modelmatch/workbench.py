"""Bounded, session-local SQL and an interpreted subset of pandas syntax.

Python syntax is inspected and interpreted, never passed to eval/exec. These
functions deliberately do not offer a general server-side Python runtime.
"""
import ast
import io
import json
import operator
import sqlite3
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

MAX_RESULT_ROWS=5000
MAX_CODE_ROWS=10000
AGGREGATES={'sum','mean','median','min','max','count','size','nunique','std','var','first','last'}
METHODS=AGGREGATES|{'head','tail','describe','groupby','agg','value_counts','sort_values','sort_index','reset_index','rename','fillna','dropna','round','isna','notna','isin','between','abs','clip'}

@dataclass
class CodeResult:
    frame: pd.DataFrame
    truncated: bool=False
    duration_ms: float=0

def as_frame(result):
    if isinstance(result,pd.DataFrame):return result
    if isinstance(result,pd.Series):
        name=result.name if result.name is not None else 'value'
        return result.rename(name).reset_index()
    if isinstance(result,(int,float,str,bool,np.generic)) or result is None:return pd.DataFrame({'value':[result]})
    raise ValueError('Return a table, a Series or a scalar. Finish grouped expressions with an aggregation.')

def _check_size(df,code):
    if len(df)>MAX_CODE_ROWS:raise ValueError('Code workbench supports up to 10,000 records. Upload a deliberate subset to continue.')
    if len(code)>10000:raise ValueError('Code limit: 10,000 characters.')

def run_sql(df,sql):
    _check_size(df,sql);start=time.monotonic()
    conn=sqlite3.connect(':memory:')
    try:
        df.to_sql('dataset',conn,index=False,if_exists='replace')
        conn.enable_load_extension(False)
        conn.execute('PRAGMA query_only=ON')
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,1_000_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH,10000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH,50)
        conn.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT,5)
        allowed_functions={'count','sum','avg','min','max','round','abs','coalesce','ifnull','nullif','lower','upper','length','substr','substring','date','datetime','strftime','julianday','unixepoch','typeof','trim','ltrim','rtrim','total','like','glob','replace','instr'}
        def authorizer(action,a,b,database,source):
            if action==sqlite3.SQLITE_SELECT:return sqlite3.SQLITE_OK
            if action==sqlite3.SQLITE_READ:return sqlite3.SQLITE_OK if a=='dataset' else sqlite3.SQLITE_DENY
            if action==sqlite3.SQLITE_FUNCTION:return sqlite3.SQLITE_OK if (b or a or '').lower() in allowed_functions else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_DENY
        conn.set_authorizer(authorizer)
        counter=[0]
        def progress():
            counter[0]+=1
            return int(counter[0]>5000 or time.monotonic()-start>3)
        conn.set_progress_handler(progress,1000)
        cursor=conn.execute(sql)
        if cursor.description is None:raise ValueError('Use a SELECT query returning columns.')
        rows=cursor.fetchmany(MAX_RESULT_ROWS+1);names=[c[0] for c in cursor.description]
        if len(names)!=len(set(names)):raise ValueError('Output column names must be unique; add SQL aliases.')
        return CodeResult(pd.DataFrame(rows[:MAX_RESULT_ROWS],columns=names),len(rows)>MAX_RESULT_ROWS,round((time.monotonic()-start)*1000,1))
    except sqlite3.Error as e:raise ValueError(f'SQL could not run: {e}. Use a read-only SELECT against dataset; recursive queries, writes and extensions are unavailable.') from e
    finally:conn.close()

def _valid_aggregation(value):
    if isinstance(value,str):return value in AGGREGATES
    if isinstance(value,(list,tuple)):return all(_valid_aggregation(x) for x in value)
    if isinstance(value,dict):return all(_valid_aggregation(x) for x in value.values())
    return False

class PandasInterpreter:
    def __init__(self,df):self.df=df.copy(deep=True);self.start=time.monotonic()
    def read(self,node):
        if time.monotonic()-self.start>3:raise ValueError('Expression exceeded the time budget.')
        if isinstance(node,ast.Expression):return self.read(node.body)
        if isinstance(node,ast.Name):
            if node.id=='df':return self.df
            raise ValueError('Only df is available. Imports, files, external APIs and custom functions are not supported.')
        if isinstance(node,ast.Constant):
            if not isinstance(node.value,(str,int,float,bool,type(None))):raise ValueError('Unsupported value.')
            if isinstance(node.value,(int,float)) and abs(node.value)>1_000_000:raise ValueError('Numeric literals are limited to one million.')
            if isinstance(node.value,str) and len(node.value)>1000:raise ValueError('String literal is too long.')
            return node.value
        if isinstance(node,(ast.List,ast.Tuple)):return [self.read(x) for x in node.elts]
        if isinstance(node,ast.Dict):return {self.read(k):self.read(v) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.Subscript):
            obj=self.read(node.value)
            if not isinstance(obj,(pd.DataFrame,pd.Series,DataFrameGroupBy,SeriesGroupBy)):raise ValueError('Select columns or filter rows from df.')
            return obj[self.read(node.slice)]
        if isinstance(node,ast.BinOp):
            operations={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Mod:operator.mod,ast.BitAnd:operator.and_,ast.BitOr:operator.or_}
            op=operations.get(type(node.op))
            if op is None:raise ValueError('Supported arithmetic: +, -, *, /, %, &, |.')
            a,b=self.read(node.left),self.read(node.right)
            # Numeric vectors only; prevent unbounded string multiplication/concatenation.
            for v in (a,b):
                if isinstance(v,(str,list,dict)):raise ValueError('Arithmetic requires numeric values or columns.')
                if isinstance(v,pd.Series) and not pd.api.types.is_numeric_dtype(v):raise ValueError('Arithmetic requires numeric columns.')
                if isinstance(v,pd.DataFrame) and not all(pd.api.types.is_numeric_dtype(t) for t in v.dtypes):raise ValueError('Arithmetic requires numeric columns.')
            return op(a,b)
        if isinstance(node,ast.Compare):
            if len(node.ops)!=1:raise ValueError('Combine comparisons with parentheses and & or |.')
            operations={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.Lt:operator.lt,ast.LtE:operator.le}
            op=operations.get(type(node.ops[0]))
            if op is None:raise ValueError('Unsupported comparison.')
            return op(self.read(node.left),self.read(node.comparators[0]))
        if isinstance(node,ast.UnaryOp):
            op={ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Invert:operator.invert}.get(type(node.op))
            if not op:raise ValueError('Unsupported unary operation.')
            return op(self.read(node.operand))
        if isinstance(node,ast.Call):
            if not isinstance(node.func,ast.Attribute) or node.func.attr not in METHODS:raise ValueError('This function is outside the supported pandas subset.')
            obj=self.read(node.func.value)
            if not isinstance(obj,(pd.DataFrame,pd.Series,DataFrameGroupBy,SeriesGroupBy)):raise ValueError('Methods must operate on df, selected columns or groups.')
            name=node.func.attr;args=[self.read(a) for a in node.args]
            if any(k.arg is None for k in node.keywords):raise ValueError('Keyword unpacking is unsupported.')
            kwargs={k.arg:self.read(k.value) for k in node.keywords}
            if any(k in kwargs for k in ['inplace','engine','engine_kwargs']):raise ValueError('In-place operations and custom engines are disabled.')
            if name=='agg':
                if len(args)!=1 or kwargs or not _valid_aggregation(args[0]):raise ValueError('agg supports one function name, list or column-to-function mapping from the documented aggregate list.')
            # Callable strings passed to pandas aggregation methods are not permitted elsewhere.
            if name in AGGREGATES and any(isinstance(v,str) for v in args):raise ValueError('Aggregation positional strings are unsupported. Select the column first.')
            if name in ['head','tail'] and args and (not isinstance(args[0],int) or not 0<=args[0]<=MAX_RESULT_ROWS):raise ValueError('head/tail must use 0–5,000 rows.')
            if name=='groupby' and kwargs.get('observed') is None:kwargs['observed']=True
            return getattr(obj,name)(*args,**kwargs)
        raise ValueError('Use one supported pandas expression; statements, imports, assignments and arbitrary Python are not executed.')

def run_pandas(df,code):
    _check_size(df,code);start=time.monotonic()
    try:tree=ast.parse(code.strip(),mode='eval')
    except SyntaxError as e:raise ValueError('Enter one pandas expression starting with df. For full scripts, use the Python scratchpad export.') from e
    if len(list(ast.walk(tree)))>250:raise ValueError('Expression is too complex (250 syntax nodes maximum).')
    result=as_frame(PandasInterpreter(df).read(tree))
    if isinstance(result.columns,pd.MultiIndex):result.columns=['_'.join(map(str,c)).strip('_') for c in result.columns]
    result.columns=result.columns.map(str)
    if not result.columns.is_unique:raise ValueError('Result columns must have unique names; rename them in your expression.')
    return CodeResult(result.head(MAX_RESULT_ROWS).copy(),len(result)>MAX_RESULT_ROWS,round((time.monotonic()-start)*1000,1))

def scratchpad_notebook(code):
    return json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':['# ModelMatch • Python scratchpad\nCreated by Blessing Sekeramavi. Review this user-authored code before running it locally. Supply dataset.csv separately.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]},indent=2)
