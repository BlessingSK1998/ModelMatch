import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .ui import polish

def chart(df,config):
    kind=config.get('kind','Histogram'); x=config.get('x'); y=config.get('y'); agg=config.get('aggregation','mean'); title=config.get('title',kind)
    if df.empty: return go.Figure().update_layout(title='No records match these filters')
    if kind=='Histogram': fig=px.histogram(df,x=x,nbins=35)
    elif kind=='Box': fig=px.box(df,x=x if x!=y else None,y=y,points=False)
    elif kind=='Scatter': fig=px.scatter(df.head(5000),x=x,y=y,opacity=.65)
    elif kind=='Correlation':
        corr=df.select_dtypes('number').corr().iloc[:30,:30]; fig=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',text_auto='.2f')
    elif kind=='Map':
        # Only explicit validated latitude and longitude; no geocoding guesses.
        lat=pd.to_numeric(df[x],errors='coerce'); lon=pd.to_numeric(df[y],errors='coerce'); valid=lat.between(-90,90)&lon.between(-180,180)
        fig=go.Figure(go.Scattergeo(lat=lat[valid].head(5000),lon=lon[valid].head(5000),mode='markers'))
    else:
        d=df.copy()
        if kind=='Time trend':
            d[x]=pd.to_datetime(d[x],errors='coerce'); d=d.dropna(subset=[x]); d[x]=d[x].dt.to_period(config.get('granularity','M')).dt.to_timestamp()
        g=d.groupby(x,dropna=False).size().rename('value').reset_index() if agg=='count' else d.groupby(x,dropna=False)[y].agg(agg).rename('value').reset_index()
        g=g.sort_values(x) if kind=='Time trend' else g.sort_values('value',ascending=False).head(30)
        fig=px.line(g,x=x,y='value',markers=True) if kind=='Time trend' else px.bar(g,x=x,y='value')
    fig.update_layout(template='plotly_white',title=title,colorway=['#2563eb','#14b8a6','#f59e0b'],margin=dict(l=20,r=20,t=55,b=30),height=440)
    return polish(fig)

def insight(df,numeric):
    if df.empty:return 'No records match the current filters.'
    lines=[f'The displayed data contains {len(df):,} records.']
    for c in numeric[:3]:
        s=pd.to_numeric(df[c],errors='coerce').dropna()
        if len(s): lines.append(f'{c}: mean {s.mean():,.3g}, median {s.median():,.3g}, range {s.min():,.3g}–{s.max():,.3g} ({len(s):,} observed).')
    if len(numeric)>1:
        c=df[numeric[:25]].corr().abs(); import numpy as np
        c=c.where(np.triu(np.ones(c.shape),k=1).astype(bool)).stack()
        if len(c):
            a,b=c.idxmax(); r=df[[a,b]].corr().iloc[0,1]; lines.append(f'Strongest displayed numeric correlation: {a} and {b}, r={r:.3f}. Association does not establish causation.')
    return '\n\n'.join(lines)
