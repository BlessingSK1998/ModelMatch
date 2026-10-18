"""Univariate regular-series forecasts, fixed-origin validation and test."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .modeling import metrics

def series_from(df, config):
    d=pd.to_datetime(df[config['date']],errors='coerce'); y=pd.to_numeric(df[config['target']],errors='coerce')
    if d.isna().any() or y.isna().any(): raise ValueError('Forecast dates and values must be complete and valid. Clean explicitly first.')
    s=pd.Series(y.to_numpy(),index=d).sort_index().resample(config['frequency']).agg(config.get('aggregation','mean'))
    if s.isna().any(): raise ValueError('There are empty periods. Choose a coarser frequency or supply missing observations; no future-based filling is applied.')
    if len(s)<40 or len(s)>10000: raise ValueError('Forecasting requires 40–10,000 regular aggregated periods.')
    return s

def _predict(history,horizon,method,season):
    values=list(np.asarray(history,dtype=float))
    lag=min(12,max(3,len(values)//6))
    est=None
    if method=='Lag Ridge':
        x=np.array([values[i-lag:i] for i in range(lag,len(values))]); y=np.array(values[lag:])
        est=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x,y)
    result=[]
    for _ in range(horizon):
        p=float(values[-1] if method=='Naive baseline' else values[-season] if method=='Seasonal naive' else est.predict(np.array(values[-lag:]).reshape(1,-1))[0])
        values.append(p); result.append(p)
    return np.array(result)

def forecast(df,config):
    s=series_from(df,config); h=int(config['horizon']); season=int(config.get('season',7))
    if h<1 or h>min(90,len(s)//5): raise ValueError('Horizon must be 1–90 and at most one fifth of the history.')
    fit=s.iloc[:-2*h]; val=s.iloc[-2*h:-h]; test=s.iloc[-h:]
    methods=['Naive baseline','Lag Ridge']
    if season>1 and len(fit)>=3*season: methods.insert(1,'Seasonal naive')
    rows=[]
    for method in methods:
        p=_predict(fit,h,method,season); rows.append({'model':method,**metrics(val,p,'Regression')})
    comparison=pd.DataFrame(rows).sort_values('MAE'); name=comparison.iloc[0]['model']
    p=_predict(s.iloc[:-h],h,name,season); baseline=_predict(s.iloc[:-h],h,'Naive baseline',season)
    idx=pd.date_range(s.index[-1],periods=h+1,freq=config['frequency'])[1:]
    return {'kind':'Forecasting','name':name,'config':config,'history':s,'comparison':comparison,'test_metrics':metrics(test,p,'Regression'),'baseline_metrics':metrics(test,baseline,'Regression'),'backtest':pd.Series(p,index=test.index),'validation':pd.Series(_predict(fit,h,name,season),index=val.index),'future':pd.Series(_predict(s,h,name,season),index=idx),'actual':test.to_numpy(),'predicted':p}
