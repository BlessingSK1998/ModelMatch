import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

class DateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        out=[]
        for c in X:
            s=pd.to_datetime(X[c],errors='coerce',format='mixed')
            out.extend([s.dt.year.to_numpy(),s.dt.month.to_numpy(),s.dt.dayofweek.to_numpy(),s.dt.day.to_numpy()])
        return np.array(out,dtype=float).T

def category_strings(X):
    return X.apply(lambda col: col.map(lambda value: str(value) if pd.notna(value) else np.nan))

def preprocessor(df, roles):
    numeric=[c for c in df if roles.get(c)=='numeric']
    dates=[c for c in df if roles.get(c)=='date']
    cats=[c for c in df if c not in numeric+dates]
    parts=[]
    impute=lambda strategy: SimpleImputer(strategy=strategy,keep_empty_features=True)
    if numeric: parts.append(('numeric',Pipeline([('impute',impute('median')),('scale',StandardScaler())]),numeric))
    if dates: parts.append(('date',Pipeline([('features',DateFeatures()),('impute',impute('median')),('scale',StandardScaler())]),dates))
    if cats: parts.append(('category',Pipeline([('strings',FunctionTransformer(category_strings)),('impute',impute('most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False,max_categories=25))]),cats))
    return ColumnTransformer(parts,remainder='drop')
