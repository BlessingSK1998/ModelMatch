import io,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from modelmatch.data import load_data,infer_roles,clean,quality,filter_data
from modelmatch.modeling import train,unsupervised
from modelmatch.forecasting import forecast
from modelmatch.prediction import predict
from modelmatch.exports import script,notebook,bundle,report
from modelmatch.dashboard import chart
ROOT=Path(__file__).resolve().parents[1]
def example(n):return pd.read_csv(ROOT/'examples'/n)
def config(d,g,t=None):return {'goal':g,'target':t,'features':[c for c in d if c!=t],'roles':infer_roles(d),'group':None,'split_date':None}
def test_loading_quality_filters():
 assert len(load_data(b'a,b\n1,x\n2,y\n','a.csv'))==2
 assert len(load_data((ROOT/'examples/examples.xlsx').read_bytes(),'a.xlsx','Classification'))==150
 for data,name in [(b'','a.csv'),(b'hi','a.txt'),(b'a,b\n','a.csv'),(b'not a zip','a.xlsx')]:
  with pytest.raises(ValueError):load_data(data,name)
 raw=pd.DataFrame({'a':['1','2','oops','1'],'x':['q']*4});assert 'mixed' in quality(raw).iloc[0]['flags']
 out,log=clean(raw,{'roles':{'a':'numeric'},'duplicates':True});assert raw.a.dtype==object;assert len(out)==3;assert out.a.isna().sum()==1
 assert len(filter_data(out,{'a':{'kind':'numeric','values':[1.5,3]}}))==1
 assert len(filter_data(out,{'x':{'kind':'category','values':[]}}))==0
 assert chart(out,{'kind':'Histogram','x':'a'}).data
@pytest.fixture(scope='module')
def regression():
 d=example('health_regression.csv');return train(d,config(d,'Regression','disease_progression'))
@pytest.fixture(scope='module')
def classification():
 d=example('iris_classification.csv');d['site']=np.where(np.arange(len(d))%2,'a','b');return train(d,config(d,'Classification','target'))
def test_regression(regression):
 r=regression;assert len(r['comparison'])==4;assert 'MAE' in r['test_metrics']
 s=r['splits'];assert not set(s['test'])&set(s['development']);assert not set(s['fit'])&set(s['validation'])
 x=r['schema_data'].head(3).copy();x['id']=['a','b','c'];p,w=predict(x,r);assert list(p.id)==['a','b','c'];assert 'modelmatch_prediction' in p
 with pytest.raises(ValueError):predict(x.drop(columns=[r['config']['features'][0]]),r)
 bad=x.copy();bad[bad.columns[0]]=bad.iloc[:,0].astype(object);bad.iloc[0,0]='bad'
 with pytest.raises(ValueError):predict(bad,r)
 high=x.copy();high.iloc[0,0]=1e8;assert any('outside' in v for v in predict(high,r)[1])
def test_classification_unseen_and_imputation(classification):
 r=classification;assert 'F1 macro' in r['test_metrics'];assert 'ROC-AUC' in r['test_metrics']
 x=r['schema_data'].head(2).copy();x['site']='never-seen';p,w=predict(x,r);assert any('unfamiliar' in v for v in w);assert len([c for c in p if 'probability_' in c])==3
 imputer=r['model'].named_steps['prepare'].named_transformers_['numeric'].named_steps['impute'];nums=[c for c in r['config']['features'] if r['config']['roles'][c]=='numeric'];np.testing.assert_allclose(imputer.statistics_,r['schema_data'][nums].median())
def test_group_and_time():
 d=example('health_regression.csv');d['entity']=np.arange(len(d))//5;c=config(d,'Regression','disease_progression');c['group']='entity';c['features'].remove('entity');r=train(d,c);s=r['splits'];assert not set(d.iloc[s['test']].entity)&set(d.iloc[s['development']].entity)
 d['day']=pd.date_range('2025-01-01',periods=len(d));c['group']=None;c['split_date']='day';r=train(d,c);s=r['splits'];assert d.iloc[s['development']].day.max()<d.iloc[s['test']].day.min()
def test_forecast():
 d=example('daily_sales.csv');c={'goal':'Forecasting','date':'date','target':'sales','frequency':'D','aggregation':'mean','horizon':7,'season':7};r=forecast(d,c);assert {'Naive baseline','Seasonal naive','Lag Ridge'}==set(r['comparison'].model);assert len(r['future'])==7;assert r['future'].index.min()>r['history'].index.max()
 changed=d.copy();changed.loc[changed.index[-7:],'sales']=1e9;r2=forecast(changed,c);assert r['name']==r2['name'];np.testing.assert_allclose(r['backtest'],r2['backtest'])
@pytest.mark.parametrize('goal',['Clustering','Anomaly detection'])
def test_unsupervised(goal):
 d=example('health_regression.csv');c=config(d,goal);c.update(k=3,contamination=.05);r=unsupervised(d,c);assert len(predict(d.head(2),r)[0])==2
def test_exports_execute(tmp_path):
 d=example('health_regression.csv');cfg=config(d,'Regression','disease_progression');cleaning={'roles':infer_roles(d)};dash={'charts':[{'kind':'Histogram','x':'age'}]};code=script(cfg,cleaning,dash);nb=notebook(code);assert json.loads(nb)['nbformat']==4
 with zipfile.ZipFile(io.BytesIO(bundle(code,nb,dash))) as z:z.extractall(tmp_path)
 p=subprocess.run([sys.executable,'workflow.py','--data','examples/health_regression.csv'],cwd=tmp_path,capture_output=True,text=True,timeout=90);assert p.returncode==0,p.stderr
 assert (tmp_path/'outputs/model.joblib').exists();assert (tmp_path/'outputs/test_metrics.json').exists();assert 'plotly' in report(d,[chart(d,dash['charts'][0])],None,'Calculated',[])
 with zipfile.ZipFile(io.BytesIO(bundle(dashboard=dash,standalone=True))) as z:assert 'app.py' in z.namelist()
def test_app_smoke():
 from streamlit.testing.v1 import AppTest
 at=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30);assert not at.exception
 for name in ['Data Quality','Dashboard','Model Lab','Prediction Studio','Exports']:
  at.sidebar.radio[0].set_value(name).run(timeout=30);assert not at.exception,f'{name}: {at.exception}'

def test_notebook_and_classification_forecast_exports(tmp_path):
 cases=[('iris_classification.csv','Classification','target'),('daily_sales.csv','Forecasting','sales')]
 for filename,goal,target in cases:
  d=example(filename);cfg=config(d,goal,target)
  if goal=='Forecasting':cfg.update(date='date',frequency='D',aggregation='mean',horizon=7,season=7)
  cleaning={'roles':infer_roles(d)};code=script(cfg,cleaning,{'charts':[]});folder=tmp_path/goal;folder.mkdir()
  with zipfile.ZipFile(io.BytesIO(bundle(code,notebook(code),{'charts':[]}))) as z:z.extractall(folder)
  p=subprocess.run([sys.executable,'workflow.py','--data','examples/'+filename],cwd=folder,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
 # Execute actual generated notebook cell with its matching included regression example.
 d=example('health_regression.csv');code=script(config(d,'Regression','disease_progression'),{'roles':infer_roles(d)},{'charts':[]});folder=tmp_path/'nb';folder.mkdir()
 with zipfile.ZipFile(io.BytesIO(bundle(code,notebook(code),{'charts':[]}))) as z:z.extractall(folder)
 nb=json.loads(notebook(code));(folder/'run_notebook.py').write_text(''.join(nb['cells'][1]['source']))
 p=subprocess.run([sys.executable,'run_notebook.py'],cwd=folder,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr

def test_trained_ui_prediction(classification):
 from streamlit.testing.v1 import AppTest
 at=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30);at.sidebar.selectbox[0].set_value('Iris • Classification').run();at.session_state['result']=classification
 for name in ['Model Lab','Prediction Studio','Exports']:
  at.sidebar.radio[0].set_value(name).run(timeout=30);assert not at.exception
 at.sidebar.radio[0].set_value('Prediction Studio').run()
 next(b for b in at.button if b.label=='Predict one record').click().run(timeout=30);assert not at.exception


def test_boolean_categories_and_missing_targets():
 d=example('health_regression.csv');d['has_access']=np.arange(len(d))%2==0;d.loc[0,'disease_progression']=np.nan
 c=config(d,'Regression','disease_progression');r=train(d,c);assert r['dropped_targets']==1
 x=r['schema_data'].head(2).copy();p,w=predict(x,r);assert len(p)==2
 encoder=r['model'].named_steps['prepare'].named_transformers_['category'].named_steps['encode'];assert set(encoder.categories_[0])=={'True','False'}
