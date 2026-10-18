"""Deterministic workflow and standalone dashboard packages."""
import io,json,zipfile,html,platform
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn,streamlit,plotly,joblib

ROOT=Path(__file__).resolve().parents[1]

def versions():
    return {'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit-learn':sklearn.__version__,'streamlit':streamlit.__version__,'plotly':plotly.__version__}

def script(config,cleaning,dashboard):
    return '''"""ModelMatch reproducible workflow. Created by Blessing Sekeramavi.
Run from the extracted project root: python workflow.py --data examples/health_regression.csv
Install: pip install -r requirements.txt
Only load model files from trusted sources.
"""
import argparse, json
from pathlib import Path
from modelmatch.data import load_data, clean
from modelmatch.modeling import train, unsupervised
from modelmatch.forecasting import forecast
from modelmatch.prediction import predict
from modelmatch.dashboard import chart
import joblib

CONFIG = json.loads('''+repr(json.dumps(config))+''')
CLEANING = json.loads('''+repr(json.dumps(cleaning))+''')
DASHBOARD = json.loads('''+repr(json.dumps(dashboard))+''')
parser=argparse.ArgumentParser()
parser.add_argument('--data',required=True)
parser.add_argument('--sheet',default='0')
parser.add_argument('--predict')
parser.add_argument('--output',default='outputs')
args=parser.parse_args()
path=Path(args.data)
sheet=int(args.sheet) if args.sheet.isdigit() else args.sheet
original=load_data(path.read_bytes(),path.name,sheet)
df,log=clean(original,CLEANING)
out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
df.to_csv(out/'cleaned.csv',index=False)
(out/'transformations.json').write_text(json.dumps(log,indent=2))
for i,c in enumerate(DASHBOARD.get('charts',[])):
    chart(df,c).write_html(out/f'chart_{i}.html')
if CONFIG and CONFIG.get('goal') not in [None,'Explore']:
    goal=CONFIG['goal']
    result=forecast(df,CONFIG) if goal=='Forecasting' else train(df,CONFIG) if goal in ['Regression','Classification'] else unsupervised(df,CONFIG)
    if 'comparison' in result: result['comparison'].to_csv(out/'validation_metrics.csv',index=False)
    if 'test_metrics' in result: (out/'test_metrics.json').write_text(json.dumps(result['test_metrics'],indent=2))
    if goal=='Forecasting': result['future'].to_csv(out/'forecast.csv')
    else:
        joblib.dump(result,out/'model.joblib')
        if args.predict:
            p=Path(args.predict); frame=load_data(p.read_bytes(),p.name)
            predictions,warnings=predict(frame,result)
            predictions.to_csv(out/'predictions.csv',index=False)
            print(warnings)
print('Completed',out)
'''

def notebook(code):
    import ast
    config_line=next(line for line in code.splitlines() if line.startswith('CONFIG = json.loads('))
    cfg=json.loads(ast.literal_eval(config_line[len('CONFIG = json.loads('):-1]))
    example={'Classification':'iris_classification.csv','Forecasting':'daily_sales.csv'}.get(cfg.get('goal'),'health_regression.csv')
    return json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':['# ModelMatch\nCreated by Blessing Sekeramavi. Install requirements and keep the modelmatch package beside this notebook. Replace the data path below with the original dataset used in the app (and worksheet if applicable). The selected workflow may require different columns from the default health example.']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':('import sys\nsys.argv = ["workflow", "--data", "examples/'+example+'"]\n'+code).splitlines(True)}]},indent=2)

def report(df,figures,result,insights,log):
    parts=['<!doctype html><html><meta charset="utf-8"><title>ModelMatch analysis</title><body style="font-family:system-ui;max-width:1100px;margin:40px auto;color:#152238">','<h1>ModelMatch • Analysis report</h1><p>Created by Blessing Sekeramavi</p>',f'<p>{len(df):,} displayed records • {len(df.columns)} columns. This is a static interactive-chart report, not a live dashboard.</p>','<h2>Calculated findings</h2><p>'+html.escape(insights).replace('\n','<br>')+'</p>','<h2>Transformations</h2><pre>'+html.escape('\n'.join(log))+'</pre>']
    for i,fig in enumerate(figures): parts.append(fig.to_html(full_html=False,include_plotlyjs=True if i==0 else False))
    if result:
        parts.append('<h2>Model evaluation</h2><p>'+html.escape(result['name'])+'</p>')
        if 'comparison' in result: parts.append('<h3>Validation comparison</h3>'+result['comparison'].to_html(index=False))
        if 'test_metrics' in result: parts.append('<h3>Final held-out test</h3><pre>'+html.escape(json.dumps(result['test_metrics'],indent=2))+'</pre>')
    parts.append('<h2>Limitations</h2><p>Predictions are estimates. Correlations, feature importance and what-if changes do not establish causation. Clusters and anomaly flags are exploratory. No calibrated confidence or uncertainty intervals are provided. Dashboard filters do not change model training data.</p></body></html>')
    return '\n'.join(parts)

STANDALONE='''import json
from pathlib import Path
import streamlit as st
import pandas as pd
from modelmatch.dashboard import chart, insight
from modelmatch.data import load_data, filter_data, infer_roles
st.set_page_config(page_title="ModelMatch Dashboard",layout="wide")
st.title("ModelMatch Dashboard")
st.caption("Created by Blessing Sekeramavi • Supply the cleaned dataset used to build this dashboard.")
config=json.loads(Path('dashboard.json').read_text())
u=st.file_uploader('Cleaned CSV or Excel',type=['csv','xlsx'])
if not u: st.stop()
sheet=st.selectbox('Sheet',pd.ExcelFile(u).sheet_names) if u.name.endswith('.xlsx') else 0
df=load_data(u.getvalue(),u.name,sheet)
filters={}
roles=infer_roles(df)
st.header(config.get('title','My dashboard'))
for c in df:
 if roles[c]=='date':
  dates=pd.to_datetime(df[c],errors='coerce').dropna()
  if len(dates):
   chosen=st.sidebar.date_input(c,(dates.min().date(),dates.max().date()))
   if len(chosen)==2: filters[c]={'kind':'date','values':[str(chosen[0]),str(chosen[1])+' 23:59:59.999999']}
 elif df[c].nunique()<=40:
  values=sorted(df[c].astype(str).unique()); chosen=st.sidebar.multiselect(c,values,default=values); filters[c]={'kind':'category','values':chosen}
 elif pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique()>1:
  lo,hi=float(df[c].min()),float(df[c].max()); chosen=st.sidebar.slider(c,lo,hi,(lo,hi)); filters[c]={'kind':'numeric','values':chosen}
view=filter_data(df,filters)
st.metric('Filtered records',len(view))
for c in config.get('kpis',[]):
 if c in view and pd.api.types.is_numeric_dtype(view[c]):
  agg=config.get('aggregation','mean'); value=getattr(view[c],agg)();st.metric(agg+' · '+c,f'{value:,.4g}')
st.write(insight(view,[c for c in view.select_dtypes('number') if roles.get(c)!='identifier']))
for c in config.get('charts',[]):
 try: st.plotly_chart(chart(view,c),use_container_width=True)
 except Exception as e: st.warning(str(e))
st.dataframe(view,use_container_width=True)
st.download_button('Download filtered records',view.to_csv(index=False),'filtered.csv')
'''

def bundle(code=None, nb=None, dashboard=None, standalone=False):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['modelmatch','examples']:
            for f in (ROOT/folder).glob('*'):
                if f.is_file() and f.suffix in ['.py','.csv','.xlsx']: z.write(f,str(f.relative_to(ROOT)))
        z.write(ROOT/'requirements.txt','requirements.txt')
        if standalone:
            z.writestr('app.py',STANDALONE);z.writestr('dashboard.json',json.dumps(dashboard,indent=2));z.writestr('README.md','Run `pip install -r requirements.txt` then `streamlit run app.py`. Upload the cleaned dataset. No user data is included.')
        else:
            z.writestr('workflow.py',code);z.writestr('workflow.ipynb',nb);z.writestr('dashboard.json',json.dumps(dashboard,indent=2))
    return b.getvalue()
