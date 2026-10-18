import numpy as np
import pandas as pd

def validate_inputs(frame,result):
    features=result['config']['features']; roles=result['config']['roles']; schema=result['schema_data']; warnings=[]
    missing=[c for c in features if c not in frame]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    out=frame[features].copy()
    for c in features:
        role=roles.get(c,'category'); before=out[c].notna()
        if role=='numeric':
            out[c]=pd.to_numeric(out[c],errors='coerce')
            if ((out[c].isna()&before)|np.isinf(out[c].to_numpy(dtype=float,na_value=np.nan))).any(): raise ValueError(f'{c}: incompatible numeric values.')
            low,high=pd.to_numeric(schema[c],errors='coerce').agg(['min','max'])
            if ((out[c]<low)|(out[c]>high)).any(): warnings.append(f'{c}: values outside training range [{low:g}, {high:g}].')
        elif role=='date':
            out[c]=pd.to_datetime(out[c],errors='coerce',format='mixed')
            if (out[c].isna()&before).any(): raise ValueError(f'{c}: invalid dates.')
        else:
            out[c]=out[c].map(lambda v:str(v) if pd.notna(v) else np.nan)
            unseen=set(out[c].dropna())-set(schema[c].dropna().astype(str))
            if unseen: warnings.append(f'{c}: unfamiliar categories ({len(unseen)}); encoded as unknown.')
        if out[c].isna().any(): warnings.append(f'{c}: missing values use the trained imputer.')
    return out,warnings

def predict(frame,result):
    X,warnings=validate_inputs(frame,result); model=result['model']; out=frame.copy(); kind=result['kind']
    prefix='modelmatch_'
    if any(str(c).startswith(prefix) for c in frame): raise ValueError('Input columns beginning modelmatch_ are reserved; rename them before predicting.')
    p=model.predict(X)
    if kind=='Clustering': out[prefix+'assigned_cluster']=p
    elif kind=='Anomaly detection':
        z=model.named_steps['prepare'].transform(X); est=model.named_steps['model']
        out[prefix+'anomaly_score']=-est.score_samples(z); out[prefix+'flagged']=p==-1
    else:
        out[prefix+'prediction']=p
        if hasattr(model,'predict_proba'):
            for c,prob in zip(model.classes_,model.predict_proba(X).T): out[prefix+'probability_'+str(c)]=prob
    return out,warnings
