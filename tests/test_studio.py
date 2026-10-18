import json
import pandas as pd
import pytest
from modelmatch.workbench import run_sql,run_pandas,scratchpad_notebook

@pytest.fixture
def data():return pd.DataFrame({'region':['North','South','North'],'sales':[10.,20.,30.],'units':[1,2,3]})

def test_sql_group_result(data):
 out=run_sql(data,'SELECT region, SUM(sales) AS revenue FROM dataset GROUP BY region ORDER BY revenue DESC')
 assert out.frame.to_dict('records')==[{'region':'North','revenue':40.},{'region':'South','revenue':20.}]
 assert not out.truncated
 assert run_sql(data,'WITH x AS (SELECT * FROM dataset) SELECT COUNT(*) AS n FROM x').frame.iloc[0,0]==3

@pytest.mark.parametrize('query',["DROP TABLE dataset", "SELECT load_extension('/tmp/x')", "ATTACH DATABASE '/tmp/x' AS other", "PRAGMA database_list", "SELECT * FROM sqlite_master", "SELECT 1; SELECT 2", "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT * FROM x"])
def test_sql_blocked(data,query):
 with pytest.raises(ValueError):run_sql(data,query)
 assert len(data)==3

def test_pandas_results_and_preservation(data):
 assert run_pandas(data,'df.groupby("region")["sales"].sum().reset_index()').frame.sales.tolist()==[40.,20.]
 assert run_pandas(data,'df[df["sales"] > 15]').frame.sales.tolist()==[20.,30.]
 assert run_pandas(data,'df["sales"] * df["units"]').frame.iloc[:,-1].tolist()==[10.,40.,90.]
 assert run_pandas(data,'df.groupby("region").agg({"sales": ["sum", "mean"]})').frame.columns.tolist()==['sales_sum','sales_mean']
 assert run_pandas(data,'df["sales"].mean()').frame.iloc[0,0]==20.
 assert len(data)==3

@pytest.mark.parametrize('expr',["__import__('os').system('id')", "df.__class__", "df.to_csv('/tmp/unwanted')", "df['sales'].agg('to_csv', '/tmp/unwanted')", "df['sales'].agg('eval')", "df.query('@__import__(\"os\")')", "df.apply('eval')", "df.pipe('eval')", "[x for x in df]", "df['region'] * 1000000", "10 ** 1000000", "df.groupby('region')['sales'].agg('to_pickle')"])
def test_pandas_restricted(data,expr):
 with pytest.raises(ValueError):run_pandas(data,expr)

def test_limits_and_notebook(data):
 with pytest.raises(ValueError):run_sql(pd.concat([data]*4000),'SELECT * FROM dataset')
 with pytest.raises(ValueError):run_pandas(data,'df.head(999999)')
 nb=json.loads(scratchpad_notebook('print(1)'));assert nb['cells'][1]['source']==['print(1)']

def test_studio_ui():
 from streamlit.testing.v1 import AppTest
 from pathlib import Path
 app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
 app.sidebar.radio[0].set_value('BI Studio').run(timeout=30);assert not app.exception
 next(b for b in app.button if b.label=='Run query').click().run(timeout=30);assert not app.exception
 assert len(app.session_state['bi_result'])==25
 next(b for b in app.button if b.label=='Pin result to report').click().run(timeout=30);assert not app.exception
 assert len(app.session_state['bi_tiles'])==1
 app.sidebar.radio[0].set_value('Overview').run(timeout=30)
 next(b for b in app.button if b.label=='Open dashboard').click().run(timeout=30);assert not app.exception
 assert app.sidebar.radio[0].value=='Dashboard'
