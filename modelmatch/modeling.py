"""Bounded candidate comparison with a held-out final test partition."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, IsolationForest
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_val_score, StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support, accuracy_score, roc_auc_score, silhouette_score
from sklearn.inspection import permutation_importance
from .preprocessing import preprocessor

SEED=42

def candidates(goal):
    if goal=='Regression': return {'Dummy baseline':DummyRegressor(), 'Ridge':Ridge(alpha=10), 'Random forest':RandomForestRegressor(n_estimators=80,max_depth=10,min_samples_leaf=3,n_jobs=1,random_state=SEED), 'Gradient boosting':HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,random_state=SEED)}
    return {'Dummy baseline':DummyClassifier(strategy='prior'), 'Logistic regression':LogisticRegression(max_iter=500,class_weight='balanced',random_state=SEED), 'Random forest':RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=3,class_weight='balanced',n_jobs=1,random_state=SEED), 'Gradient boosting':HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,random_state=SEED)}

def recommendations(goal,df,features):
    names=list(candidates(goal)) if goal in ['Regression','Classification'] else {'Clustering':['K-means','Mini-batch K-means'],'Anomaly detection':['Isolation Forest'],'Forecasting':['Naive baseline','Seasonal naive','Lag Ridge']} .get(goal,[])
    rows=[]
    for name in names:
        if 'baseline' in name.lower() or 'naive' in name.lower(): why='Reference performance; exposes whether added complexity helps'; limit='Cannot learn complex feature relationships'
        elif name in ['Ridge','Logistic regression']: why='Fast, regularized, interpretable starting point'; limit='Linear relationships; sensitive to feature design'
        elif 'forest' in name.lower() or 'boosting' in name.lower(): why='Captures nonlinear relationships and interactions'; limit='Less interpretable; can overfit small datasets'
        elif 'means' in name.lower(): why='Efficient distance-based grouping'; limit='Assumes compact clusters; sensitive to scale and chosen k'
        elif name=='Isolation Forest': why='Scalable unsupervised unusual-record scoring'; limit='Flags are not verified errors; no accuracy without labels'
        else: why='Learns from past lag values with chronological backtesting'; limit='Needs regular history; no external predictors or uncertainty intervals'
        rows.append({'method':name,'why':why,'preparation':'Train-only imputation, scaling, bounded one-hot encoding; date calendar features' if goal!='Forecasting' else 'Regular time aggregation; strictly past lags','limitations / avoid':limit})
    return pd.DataFrame(rows)

def metrics(y,p,goal,proba=None,classes=None):
    if goal=='Regression': return {'MAE':mean_absolute_error(y,p),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R²':r2_score(y,p)}
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average='weighted',zero_division=0)
    macro=precision_recall_fscore_support(y,p,average='macro',zero_division=0)[2]
    out={'Accuracy':accuracy_score(y,p),'Precision weighted':precision,'Recall weighted':recall,'F1 weighted':f1,'F1 macro':macro}
    if proba is not None:
        try: out['ROC-AUC']=roc_auc_score(y,proba[:,1] if len(classes)==2 else proba,labels=classes,multi_class='ovr',average='weighted')
        except ValueError: pass
    return out

def train(df, config, progress=None):
    goal=config['goal']; target=config['target']; features=config['features']; group=config.get('group'); date=config.get('split_date')
    data=df.dropna(subset=[target]).copy(); dropped=len(df)-len(data)
    if len(data)>10000: raise ValueError('Training limit: 10,000 labeled rows. Select a smaller dataset explicitly.')
    if len(data)<40: raise ValueError('At least 40 labeled records are required for supervised comparison.')
    if not features: raise ValueError('Choose at least one prediction-time feature.')
    if len(features)>40: raise ValueError('Training limit: 40 features. Exclude identifiers and low-value columns.')
    if target in features or (group and group in features): raise ValueError('Target and grouping columns must be excluded from features.')
    if goal=='Regression':
        data[target]=pd.to_numeric(data[target],errors='raise')
        if not np.isfinite(data[target]).all(): raise ValueError('Regression target must contain finite numbers.')
    else:
        data[target]=data[target].astype(str)
        counts=data[target].value_counts()
        if len(counts)<2 or len(counts)>30 or counts.min()<5: raise ValueError('Classification requires 2–30 classes and at least 5 records per class.')
    if date:
        data[date]=pd.to_datetime(data[date],errors='coerce')
        if data[date].isna().any(): raise ValueError('Chronological split date contains missing/invalid dates.')
        data=data.sort_values(date,kind='stable')
    data=data.reset_index(drop=True); ids=np.arange(len(data))
    def split(ix, fraction):
        if date:
            # Keep equal timestamps in the same partition.
            times=data.iloc[ix][date]; unique=times.drop_duplicates().to_numpy()
            if len(unique)<6: raise ValueError('Need at least six distinct timestamps for time splitting.')
            cutoff=unique[max(1,int(len(unique)*(1-fraction)))]; mask=times<cutoff
            return ix[mask.to_numpy()],ix[~mask.to_numpy()]
        if group:
            if data[group].isna().any() or data[group].nunique()<6: raise ValueError('Grouping needs at least six nonmissing entities.')
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=SEED).split(ix,groups=data.iloc[ix][group])); return ix[a],ix[b]
        strat=data.iloc[ix][target] if goal=='Classification' else None
        return train_test_split(ix,test_size=fraction,random_state=SEED,stratify=strat)
    dev,test=split(ids,.2); fit,val=split(dev,.25)
    X=data[features]; y=data[target]
    if goal=='Classification' and (set(y.iloc[fit])!=set(y) or set(y.iloc[val])!=set(y) or set(y.iloc[test])!=set(y)):
        raise ValueError('Every split must contain every class. Add data or change split strategy.')
    rows=[]; fitted={}; failures=[]
    score='neg_mean_absolute_error' if goal=='Regression' else 'f1_macro'
    for i,(name,est) in enumerate(candidates(goal).items()):
        if progress: progress(i/4,f'Fitting {name}')
        try:
            pipe=Pipeline([('prepare',preprocessor(X,config['roles'])),('model',est)])
            pipe.fit(X.iloc[fit],y.iloc[fit]); pred=pipe.predict(X.iloc[val]); m=metrics(y.iloc[val],pred,goal)
            row={'model':name,**m}; row['Training MAE' if goal=='Regression' else 'Training F1 macro']=metrics(y.iloc[fit],pipe.predict(X.iloc[fit]),goal)['MAE' if goal=='Regression' else 'F1 macro']
            try:
                cv=TimeSeriesSplit(n_splits=3) if date else GroupKFold(n_splits=3) if group else StratifiedKFold(3,shuffle=True,random_state=SEED) if goal=='Classification' else KFold(3,shuffle=True,random_state=SEED)
                scores=cross_val_score(pipe,X.iloc[fit],y.iloc[fit],cv=cv,groups=data.iloc[fit][group] if group else None,scoring=score,error_score='raise',n_jobs=1)
                row['CV MAE' if goal=='Regression' else 'CV F1 macro']=float(-scores.mean() if goal=='Regression' else scores.mean())
            except ValueError as e: failures.append(f'{name}: CV unavailable ({e})')
            rows.append(row); fitted[name]=pipe
        except Exception as e: failures.append(f'{name}: {e}')
    if not rows: raise ValueError('All candidates failed: '+'; '.join(failures))
    comparison=pd.DataFrame(rows); metric='MAE' if goal=='Regression' else 'F1 macro'; ascending=goal=='Regression'
    comparison=comparison.sort_values(metric,ascending=ascending); name=comparison.iloc[0]['model']
    final=fitted[name]; final.fit(X.iloc[dev],y.iloc[dev]); pred=final.predict(X.iloc[test])
    proba=final.predict_proba(X.iloc[test]) if hasattr(final,'predict_proba') else None
    test_metrics=metrics(y.iloc[test],pred,goal,proba,getattr(final,'classes_',None))
    baseline=fitted.get('Dummy baseline'); baseline_metrics=None
    if baseline is not None:
        baseline.fit(X.iloc[dev],y.iloc[dev]); baseline_metrics=metrics(y.iloc[test],baseline.predict(X.iloc[test]),goal)
    # Importance uses validation and its fit-only estimator, never test outcomes.
    importance=None
    try:
        selected=candidates(goal)[name]; vp=Pipeline([('prepare',preprocessor(X,config['roles'])),('model',selected)]).fit(X.iloc[fit],y.iloc[fit])
        imp=permutation_importance(vp,X.iloc[val].head(500),y.iloc[val].head(500),scoring=score,n_repeats=3,random_state=SEED,n_jobs=1)
        importance=pd.DataFrame({'feature':features,'importance':imp.importances_mean}).sort_values('importance',ascending=False)
    except Exception: pass
    if progress: progress(1.,'Complete')
    return {'kind':goal,'model':final,'name':name,'comparison':comparison,'test_metrics':test_metrics,'baseline_metrics':baseline_metrics,'actual':y.iloc[test].to_numpy(),'predicted':pred,'proba':proba,'classes':getattr(final,'classes_',None),'importance':importance,'config':config,'dropped_targets':dropped,'failures':failures,'splits':{'fit':fit.tolist(),'validation':val.tolist(),'test':test.tolist(),'development':dev.tolist()},'schema_data':X.iloc[dev].copy(),'split_data':data[[c for c in [group,date] if c]].copy()}

def unsupervised(df,config):
    X=df[config['features']].copy()
    if not len(X.columns) or len(X.columns)>40: raise ValueError('Choose between 1 and 40 features.')
    if len(X)<10 or len(X)>10000: raise ValueError('Use between 10 and 10,000 records.')
    prep=preprocessor(X,config['roles']); z=prep.fit_transform(X)
    if config['goal']=='Clustering':
        k=config.get('k',3)
        est=(MiniBatchKMeans(n_clusters=k,random_state=SEED,n_init=10,batch_size=256) if config.get('method')=='Mini-batch K-means' else KMeans(n_clusters=k,random_state=SEED,n_init=10))
        labels=est.fit_predict(z); score=silhouette_score(z,labels,sample_size=min(1500,len(z)),random_state=SEED) if 1<len(np.unique(labels))<len(z) else None
        table=df.copy(); table['assigned_cluster']=labels
        profiles=table.groupby('assigned_cluster')[X.select_dtypes('number').columns].mean()
        details={'silhouette':score,'profiles':profiles,'sizes':table.assigned_cluster.value_counts()}; name=config.get('method','K-means')
    else:
        est=IsolationForest(n_estimators=100,contamination=config.get('contamination',.05),random_state=SEED,n_jobs=1).fit(z)
        table=df.copy(); table['anomaly_score']=-est.score_samples(z); table['flagged']=est.predict(z)==-1
        details={}; name='Isolation Forest'
    return {'kind':config['goal'],'name':name,'model':Pipeline([('prepare',prep),('model',est)]),'config':config,'schema_data':X,'table':table,**details}
