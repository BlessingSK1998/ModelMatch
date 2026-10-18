from pathlib import Path
import pandas as pd
import pytest
from modelmatch.automatic import chart_plan,prepare_chart,suggest_targets,default_features
from modelmatch.data import infer_roles
ROOT=Path(__file__).resolve().parents[1]

def test_automatic_charts_cover_fields():
 d=pd.read_csv(ROOT/'examples/daily_sales.csv');roles=infer_roles(d);plan=chart_plan(d,roles)
 assert {'Histogram','Box','Category frequencies','Time trend'}<=set(p['kind'] for p in plan)
 for s in plan:assert len(prepare_chart(d,s,roles).data)>0
 assert suggest_targets(d,roles)[0]=='sales'

def test_regression_and_classification_target_suggestions():
 for file,target in [('health_regression.csv','disease_progression'),('iris_classification.csv','target')]:
  d=pd.read_csv(ROOT/'examples'/file);roles=infer_roles(d);assert suggest_targets(d,roles)[0]==target
  d['customer_id']=range(len(d));roles=infer_roles(d);assert 'customer_id' not in default_features(d,roles,target)

def test_correlation_excludes_ids_and_maps_require_coords():
 d=pd.DataFrame({'id':range(50),'a':range(50),'b':[x*x for x in range(50)],'latitude':[999]*50,'longitude':[2]*50})
 roles=infer_roles(d);plan=chart_plan(d,roles);corr=next(s for s in plan if s['kind']=='Correlation')
 assert 'id' not in corr['columns'];assert not any(s['kind']=='Map' for s in plan)
 d['latitude']=list(range(50));plan=chart_plan(d,infer_roles(d));assert any(s['kind']=='Map' for s in plan)

def test_time_chart_preserves_missing_periods():
 d=pd.DataFrame({'date':['2025-01-01','2025-01-03'],'value':[1.,2.]});roles={'date':'date','value':'numeric'}
 spec=next(s for s in chart_plan(d,roles) if s['kind']=='Time trend');fig=prepare_chart(d,spec,roles)
 assert len(fig.data[0].x)==3;assert pd.isna(fig.data[0].y[1]);assert not fig.data[0].connectgaps

def test_automatic_ui_and_model_predictions():
 from streamlit.testing.v1 import AppTest
 app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60);assert not app.exception
 assert len(app.session_state['auto_plan'])>20
 app.checkbox(key='auto_confirm_disease_progression_Regression').check().run(timeout=60)
 app.button(key='auto_train').click().run(timeout=90);assert not app.exception
 r=app.session_state['result'];p=app.session_state['auto_predictions']
 assert len(p)==len(r['splits']['test']);assert 'modelmatch_prediction' in p
 app.button(key='auto_make_dashboard').click().run(timeout=60);assert not app.exception
 assert app.sidebar.radio[0].value=='Dashboard';assert len(app.session_state['dashboard']['charts'])==12

def test_automatic_report_export():
 from streamlit.testing.v1 import AppTest
 app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
 app.button(key='auto_build_report').click().run(timeout=90);assert not app.exception
 assert 'plotly' in app.session_state['auto_html'];assert 'Analysis report' in app.session_state['auto_html']


def test_monthly_trend_keeps_values():
 d=pd.DataFrame({'date':['2024-01-02','2025-12-03'],'value':[1.,2.]});roles={'date':'date','value':'numeric'}
 spec=next(s for s in chart_plan(d,roles) if s['kind']=='Time trend');fig=prepare_chart(d,spec,roles)
 assert fig.data[0].y[0]==1.;assert fig.data[0].y[-1]==2.;assert len(fig.data[0].x)==24
