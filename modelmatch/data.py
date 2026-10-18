"""Session-safe loading, explicit cleaning, and descriptive quality checks."""
import io
import numpy as np
import pandas as pd

MAX_MB = 25
MAX_ROWS = 30000
MAX_COLS = 150

def load_data(content, name, sheet=0, max_mb=MAX_MB):
    if len(content) > max_mb * 1024**2:
        raise ValueError(f'File exceeds {max_mb} MB limit.')
    if not content:
        raise ValueError('The file is empty.')
    try:
        if name.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif name.lower().endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet, engine='openpyxl')
        else:
            raise ValueError('Supported formats: CSV and XLSX.')
    except Exception as e:
        raise ValueError(f'Could not read dataset: {e}') from e
    if df.empty or len(df.columns) == 0:
        raise ValueError('Dataset has no records.')
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLS:
        raise ValueError(f'Limit: {MAX_ROWS:,} rows and {MAX_COLS} columns. Sample locally before uploading.')
    df.columns = df.columns.map(str)
    if not df.columns.is_unique:
        raise ValueError('Column names must be unique.')
    return df

def infer_roles(df):
    roles = {}
    for c in df:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s): role = 'date'
        elif c.lower() in ['id','index','uuid'] or c.lower().endswith('_id'): role = 'identifier'
        elif pd.api.types.is_bool_dtype(s): role = 'category'
        elif pd.api.types.is_numeric_dtype(s): role = 'numeric'
        elif pd.to_datetime(s.dropna().astype(str).head(100), errors='coerce', format='mixed').notna().mean() > .9 and s.dropna().astype(str).str.contains(r'[-/:]').mean() > .8: role = 'date'
        else: role = 'category'
        roles[c] = role
    return roles

def clean(df, config):
    out = df.copy(deep=True)
    log = []
    for c, role in config.get('roles', {}).items():
        if c not in out: continue
        before = int(out[c].isna().sum())
        if role == 'numeric': out[c] = pd.to_numeric(out[c], errors='coerce').replace([np.inf,-np.inf],np.nan)
        elif role == 'date': out[c] = pd.to_datetime(out[c], errors='coerce', format='mixed')
        else: out[c] = out[c].map(lambda x: str(x) if pd.notna(x) else np.nan)
        log.append(f'{c}: confirmed {role}; {int(out[c].isna().sum())-before} values became missing')
    if config.get('duplicates'):
        n = len(out); out = out.drop_duplicates(); log.append(f'Removed {n-len(out)} exact duplicate rows')
    exclude = [c for c in config.get('exclude',[]) if c in out]
    out = out.drop(columns=exclude)
    if exclude: log.append('Excluded columns: '+', '.join(exclude))
    if config.get('missing') == 'Drop incomplete rows':
        n = len(out); out = out.dropna(); log.append(f'Removed {n-len(out)} incomplete rows')
    else: log.append('Missing features retained; model imputers fit on training data only')
    return out.reset_index(drop=True), log

def quality(df):
    rows=[]
    for c in df:
        s=df[c]; n=s.nunique(dropna=True); top=s.value_counts(normalize=True).max() if n else 0
        flags=[]
        if n<=1: flags.append('constant / empty')
        elif top>.95: flags.append('near constant (>95%)')
        if n/max(len(s),1)>.95: flags.append('possible identifier (review)')
        if not pd.api.types.is_numeric_dtype(s) and n>50: flags.append('high cardinality')
        numeric=pd.to_numeric(s, errors='coerce')
        valid=s.notna()
        if 0 < numeric.notna().sum() < valid.sum(): flags.append('mixed numeric/text values')
        outliers=0
        if pd.api.types.is_numeric_dtype(s):
            q1,q3=s.quantile([.25,.75]); iqr=q3-q1
            outliers=int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()) if iqr>0 else 0
            if np.isinf(s.to_numpy(dtype=float,na_value=np.nan)).any(): flags.append('infinite values')
        rows.append({'column':c,'type':str(s.dtype),'missing':int(s.isna().sum()),'missing_%':round(s.isna().mean()*100,2),'unique':n,'IQR_outliers':outliers,'flags':'; '.join(flags)})
    return pd.DataFrame(rows)

def filter_data(df, filters):
    out=df.copy()
    for c, spec in filters.items():
        if c not in out: continue
        if spec['kind']=='category': out=out[out[c].astype(str).isin(spec['values'])]
        elif spec['kind']=='search': out=out[out[c].astype(str).str.contains(spec['value'],case=False,regex=False,na=False)]
        else:
            values=pd.to_datetime(out[c],errors='coerce') if spec['kind']=='date' else pd.to_numeric(out[c],errors='coerce')
            lo,hi=spec['values']; lo,hi=(pd.Timestamp(lo),pd.Timestamp(hi)) if spec['kind']=='date' else (lo,hi)
            out=out[values.between(lo,hi)]
    return out
