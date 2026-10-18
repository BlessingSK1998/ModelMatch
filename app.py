import os,io,json,hashlib,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve
from modelmatch.studio import render as render_studio
from modelmatch.ui import style,brand,header,polish
from modelmatch.data import load_data,infer_roles,clean,quality,filter_data
from modelmatch.dashboard import chart,insight
from modelmatch.modeling import train,unsupervised,recommendations
from modelmatch.forecasting import forecast
from modelmatch.prediction import predict
from modelmatch.exports import script,notebook,bundle,report,versions

st.set_page_config(page_title='ModelMatch | Data to decisions',page_icon='◈',layout='wide')
style()
S=st.session_state
with st.sidebar:
    brand()
    st.markdown('<div class="sidebar-label">WORKSPACE</div>',unsafe_allow_html=True)
    page=st.radio('Workspace',['Overview','Data Quality','Explore','Dashboard','BI Studio','Model Lab','Prediction Studio','Exports'],key='workspace_page',label_visibility='collapsed')
    st.divider()
    source=st.selectbox('Dataset source',['Health • Regression','Iris • Classification','Daily sales • Forecasting','Upload your dataset'])
    names={'Health • Regression':'health_regression.csv','Iris • Classification':'iris_classification.csv','Daily sales • Forecasting':'daily_sales.csv'}
    content=None; name='';sheet=0
    if source=='Upload your dataset':
        upload=st.file_uploader('CSV or Excel',type=['csv','xlsx'])
        if upload:
            content=upload.getvalue();name=upload.name
            if name.endswith('.xlsx'):
                try: sheet=st.selectbox('Worksheet',pd.ExcelFile(io.BytesIO(content)).sheet_names)
                except Exception as e:st.error(f'Invalid workbook: {e}');st.stop()
    else:
        name=names[source];content=(Path(__file__).parent/'examples'/name).read_bytes()
    st.caption('25 MB / 30,000 rows / 150 columns. Model training: 10,000 rows. Data stays in this session; server memory is temporary.')
    if st.button('Clear data and reset',use_container_width=True):
        S.clear();st.rerun()
    st.divider();st.caption('Created by **Blessing Sekeramavi**')
if content is None:
    st.title('Turn a dataset into your next decision.')
    st.info('Upload a CSV or Excel workbook in the sidebar, or choose an example dataset.');st.stop()
key=hashlib.sha256(content+str(sheet).encode()).hexdigest()
if S.get('source_key')!=key:
    try: original=load_data(content,name,sheet,max_mb=int(os.getenv('MODELMATCH_MAX_UPLOAD_MB','25')))
    except Exception as e:st.error(str(e));st.stop()
    for k in ['result','predictions','model_config','dashboard','figures','filters']:
        S.pop(k,None)
    S.source_key=key;S.original=original;S.data=original.copy(deep=True);S.roles=infer_roles(original);S.cleaning={'roles':S.roles.copy(),'duplicates':False,'exclude':[],'missing':'Keep for train-only imputation'};S.log=['Original data loaded; no cleaning applied.']
df=S.data;roles=S.roles;numeric=[c for c in df.select_dtypes('number') if roles.get(c)!='identifier'];cats=[c for c in df if roles.get(c)=='category'];dates=[c for c in df if roles.get(c)=='date']
header(page,name,len(df),len(df.columns))

def go_to(destination):
    S.workspace_page=destination

def download(label,data,name,mime=None,key=None):
    st.download_button(label,data,file_name=name,mime=mime,key=key)

def show_chart(fig,key):
    polish(fig)
    st.plotly_chart(fig,use_container_width=True,key=key)
    download('Download chart HTML',fig.to_html(include_plotlyjs=True),f'{key}.html','text/html',key=key+'_dl')

def ml_view(r):
    kind=r['kind'];st.subheader(f"Model results · {r['name']}")
    if 'comparison' in r:
        st.caption('Candidate selection: validation MAE (lower) or classification macro F1 (higher). CV uses fit partition only; final test is excluded from selection.')
        st.dataframe(r['comparison'],hide_index=True,use_container_width=True)
        metric='F1 macro' if kind=='Classification' else 'MAE'
        show_chart(px.bar(r['comparison'],x='model',y=metric,title='Validation comparison'),'model_comparison')
    if 'test_metrics' in r:
        st.markdown('**Final held-out test evaluation**')
        cols=st.columns(min(4,len(r['test_metrics'])))
        for i,(k,v) in enumerate(r['test_metrics'].items()):cols[i%len(cols)].metric(k,f'{v:.4g}')
        if r.get('baseline_metrics'):st.write('Same test partition — naive/dummy baseline:',r['baseline_metrics'])
        st.caption('A weak or negative R², or no improvement over the baseline, limits usefulness. Large training/validation gaps suggest overfitting. No estimate is guaranteed on new data.')
    if kind=='Regression':
        p=pd.DataFrame({'Actual':r['actual'],'Predicted':r['predicted']});show_chart(px.scatter(p,x='Actual',y='Predicted',title='Final test: actual vs predicted'),'actual_predicted')
        p['Residual']=p.Actual-p.Predicted;show_chart(px.scatter(p,x='Predicted',y='Residual',title='Final test residuals'),'residuals')
    elif kind=='Classification':
        labels=r['classes'];cm=confusion_matrix(r['actual'],r['predicted'],labels=labels)
        show_chart(px.imshow(cm,x=labels,y=labels,text_auto=True,labels={'x':'Predicted','y':'Actual'},title='Final test confusion matrix'),'confusion')
        if len(labels)==2 and r['proba'] is not None:
            binary=np.asarray(r['actual'])==labels[1];fpr,tpr,_=roc_curve(binary,r['proba'][:,1]);prec,rec,_=precision_recall_curve(binary,r['proba'][:,1]);a,b=st.columns(2)
            a.plotly_chart(px.line(x=fpr,y=tpr,labels={'x':'False positive rate','y':'True positive rate'},title='Final test ROC'),use_container_width=True)
            b.plotly_chart(px.line(x=rec,y=prec,labels={'x':'Recall','y':'Precision'},title='Final test precision–recall'),use_container_width=True)
        else:st.caption('Multiclass ROC-AUC is shown when computable; multiclass curves are not implemented.')
    elif kind=='Forecasting':
        f=go.Figure()
        for label,s in [('Observed',r['history']),('Validation predictions',r['validation']),('Final test backtest',r['backtest']),('Future forecast',r['future'])]:f.add_trace(go.Scatter(x=s.index,y=s.values,name=label,mode='lines',line={'dash':'dash' if label=='Future forecast' else 'solid'}))
        f.update_layout(title='Observed history, fixed-origin backtests and future forecasts');show_chart(f,'forecast')
        st.warning('Univariate forecasts assume historical patterns continue. Lag Ridge uses recursive predictions beyond one step. No uncertainty intervals or external predictors are implemented.')
    elif kind=='Clustering':
        st.metric('Silhouette (in-sample, exploratory)',f"{r['silhouette']:.3f}" if r['silhouette'] is not None else 'Undefined');st.write('Cluster sizes',r['sizes']);st.dataframe(r['profiles']);st.dataframe(r['table'])
        if len(numeric)>=2:show_chart(px.scatter(r['table'],x=numeric[0],y=numeric[1],color=r['table'].assigned_cluster.astype(str),title='Exploratory cluster assignment'),'clusters')
        st.caption('Clusters are exploratory, not verified ground truth; silhouette does not measure predictive accuracy.')
    else:
        show_chart(px.histogram(r['table'],x='anomaly_score',color='flagged',title='In-sample unusual-record scores'),'anomalies');st.dataframe(r['table'].sort_values('anomaly_score',ascending=False))
        st.caption('Higher scores mean more unusual. Flags depend on the chosen contamination rate. No detection accuracy can be claimed without verified labels.')
    if r.get('importance') is not None:
        show_chart(px.bar(r['importance'],x='importance',y='feature',orientation='h',title='Held-out validation permutation importance'),'importance')
        st.caption('Importance was computed on validation records, not the final test. Importance is associative; correlated predictors can share importance.')
    for failure in r.get('failures',[]):st.warning(failure)
    if 'dropped_targets' in r:st.caption(f"Excluded {r['dropped_targets']} records with missing targets. Final model fitted to development data, with final test kept separate.")

if page=='Overview':
    cols=st.columns(4)
    for col,label,value in zip(cols,['Records','Columns','Missing cells','Memory'],[f'{len(df):,}',len(df.columns),f'{df.isna().sum().sum():,}',f'{df.memory_usage(deep=True).sum()/1024**2:.2f} MB']):col.metric(label,value)
    st.markdown('<div class="section-kicker">THE NEXT STEP</div>',unsafe_allow_html=True)
    st.subheader('A clear path from upload to insight')
    steps=[('01 / UNDERSTAND','Start with the right data','Review missing values, types and quality signals.','Data Quality','Review data'),('02 / DISCOVER','Make the patterns visible','Explore interactive charts and build your dashboard.','Dashboard','Open dashboard'),('03 / PREDICT','Put your question to the test','Compare suitable models and evaluate their predictions.','Model Lab','Enter Model Lab')]
    for col,(step,title,copy,destination,label) in zip(st.columns(3),steps):
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="step-label">{step}</div><div class="step-title">{title}</div><div class="step-copy">{copy}</div>',unsafe_allow_html=True)
                st.button(label,key='overview_'+destination,on_click=go_to,args=(destination,),use_container_width=True)
    st.markdown('<div class="section-kicker">FIRST LOOK</div>',unsafe_allow_html=True)
    st.subheader('Your dataset, in focus')
    if numeric:
        left,right=st.columns([1.65,1])
        with left:
            focus=st.selectbox('Explore a numeric field',numeric,index=len(numeric)-1)
            show_chart(chart(df,{'kind':'Histogram','x':focus,'title':focus.replace('_',' ').title()+' · distribution'}),'overview_distribution')
        with right:
            with st.container(border=True):
                st.markdown('<div class="insight-head">CALCULATED SNAPSHOT</div>',unsafe_allow_html=True)
                st.write(insight(df,[focus]+[c for c in numeric if c!=focus][:2]))
                st.caption('Statistics use the active analysis dataset. Association does not establish causation.')
    preview_tab,schema_tab=st.tabs(['Data preview','Column guide'])
    with preview_tab:
        st.caption('First 100 records · use column headers to sort or the table toolbar to search.')
        st.dataframe(df.head(100),use_container_width=True,height=340)
    with schema_tab:
        st.dataframe(pd.DataFrame({'Column':df.columns,'Data type':df.dtypes.astype(str).values,'Confirmed role':[roles[c] for c in df]}),hide_index=True,use_container_width=True)
    with st.expander('About the example datasets'):
        st.caption('Health regression uses the scikit-learn diabetes dataset with standardized measurements, not raw clinical units. Iris contains species observations. Daily sales is a synthetic series for demonstration.')
elif page=='Data Quality':
    st.subheader('Inspect before changing')
    st.metric('Exact duplicate records',int(S.original.duplicated().sum()));st.dataframe(quality(S.original),hide_index=True,use_container_width=True)
    st.caption('IQR outliers are review flags, not errors; nothing is deleted automatically. Unique values may be meaningful measurements rather than IDs.')
    with st.form('cleaning'):
        st.markdown('**Confirm semantic roles**')
        role_frame=pd.DataFrame({'column':S.original.columns,'role':[roles.get(c,'category') for c in S.original]})
        edited=st.data_editor(role_frame,disabled=['column'],hide_index=True,column_config={'role':st.column_config.SelectboxColumn(options=['numeric','category','date','identifier','group'])},use_container_width=True)
        dup=st.checkbox('Remove exact duplicates',value=S.cleaning.get('duplicates',False))
        excluded=st.multiselect('Exclude columns',list(S.original),default=S.cleaning.get('exclude',[]))
        missing=st.selectbox('Missing values',['Keep for train-only imputation','Drop incomplete rows'],index=0 if S.cleaning.get('missing')!='Drop incomplete rows' else 1)
        preview=st.form_submit_button('Preview proposed changes')
    if preview:
        config={'roles':dict(zip(edited.column,edited.role)),'duplicates':dup,'exclude':excluded,'missing':missing}
        try:
            proposed,log=clean(S.original,config);S.proposal=(proposed,log,config)
        except Exception as e:st.error(str(e))
    if S.get('proposal'):
        proposed,log,config=S.proposal;st.write(log);st.write(f'{len(S.original):,} → {len(proposed):,} rows; {len(proposed.columns)} columns');st.dataframe(proposed.head(30))
        if st.button('Apply these changes',type='primary'):
            if proposed.empty or not len(proposed.columns):st.error('Cleaning would remove all data. Change your settings.')
            else:
                S.data=proposed;S.log=log;S.cleaning=config;S.roles=config['roles'];S.pop('result',None);S.pop('proposal',None);S.pop('dashboard',None);st.rerun()
    st.subheader('Transformation log');st.write(S.log)
    for c in dates:
        d=pd.to_datetime(df[c],errors='coerce').dropna().sort_values().drop_duplicates();interval=d.diff().dropna()
        st.write(f'{c}: {d.min()} to {d.max()}; {len(d)} unique dates; {interval.nunique()} distinct intervals. '+('Irregular intervals: aggregate deliberately before forecasting.' if interval.nunique()>1 else 'Regular intervals among observed dates.'))
elif page in ['Explore','Dashboard']:
    title=st.text_input('Dashboard title',S.get('dashboard',{}).get('title','My data, at a glance'),key='dash_title')
    with st.expander('Filters · update all charts, metrics and records',expanded=False):
        filters={};fc=st.columns(3)
        # Bounded, user-selected filters avoid hundreds of widgets.
        filter_cols=st.multiselect('Filter columns',list(df),default=[c for c in S.get('dashboard',{}).get('filters',{}) if c in df],key='filter_columns')
        for i,c in enumerate(filter_cols):
            with fc[i%3]:
                saved_filter=S.get('dashboard',{}).get('filters',{}).get(c,{})
                if c in dates:
                    d=pd.to_datetime(df[c],errors='coerce').dropna()
                    if len(d):
                        date_default=saved_filter.get('values',[d.min().date(),d.max().date()]);val=st.date_input(c,tuple(pd.Timestamp(v).date() for v in date_default),key='filter_date_'+c)
                        if len(val)==2:filters[c]={'kind':'date','values':[str(val[0]),str(val[1])+' 23:59:59.999999']}
                elif c in numeric and df[c].nunique()>1:
                    low,high=float(df[c].min()),float(df[c].max())
                    if np.isfinite(low) and np.isfinite(high):filters[c]={'kind':'numeric','values':st.slider(c,low,high,tuple(saved_filter.get('values',[low,high])),key='filter_num_'+c)}
                elif df[c].nunique()<=100:
                    values=sorted(df[c].astype(str).unique());filters[c]={'kind':'category','values':st.multiselect(c,values,default=[v for v in saved_filter.get('values',values) if v in values],key='filter_cat_'+c)}
                else:filters[c]={'kind':'search','value':st.text_input('Search '+c,value=saved_filter.get('value',''),key='filter_search_'+c)}
        if st.button('Reset filters'):
            for k in list(S):
                if k.startswith('filter_'):del S[k]
            if 'dashboard' in S:S.dashboard['filters']={}
            st.rerun()
    view=filter_data(df,filters);S.filtered=view
    kpi_cols=st.multiselect('KPI fields (identifiers excluded)',numeric,default=[c for c in S.get('dashboard',{}).get('kpis',numeric[:2]) if c in numeric],key='dash_kpis');agg=st.selectbox('KPI aggregation',['mean','median','sum','count'],index=['mean','median','sum','count'].index(S.get('dashboard',{}).get('aggregation','mean')),key='dash_agg')
    cols=st.columns(1+min(3,len(kpi_cols)));cols[0].metric('Filtered records',f'{len(view):,}')
    for c,col in zip(kpi_cols[:3],cols[1:]):
        val=getattr(view[c],agg)();col.metric(f'{agg.title()} · {c}',f'{val:,.4g}' if pd.notna(val) else '—')
    st.caption('Count = nonmissing values; mean = arithmetic average; median = midpoint; sum = total. Units are those of the source data. Confirm numeric category codes as categories in Data Quality.')
    if 'dashboard' not in S:S.dashboard={'title':title,'charts':[]}
    saved=st.file_uploader('Reload dashboard configuration',type=['json'])
    if saved and st.button('Load settings'):
        try:
            cfg=json.loads(saved.getvalue());assert isinstance(cfg.get('charts'),list) and len(cfg['charts'])<=12
            for c in cfg['charts']:assert c.get('x') in df and (not c.get('y') or c['y'] in df)
            S.dashboard=cfg
            for widget in list(S):
                if widget in ['dash_title','dash_kpis','dash_agg'] or widget.startswith('filter_'):S.pop(widget,None)
            st.rerun()
        except Exception:st.error('Invalid configuration or incompatible columns.')
    with st.expander('Chart builder · add a visual to your dashboard',expanded=False):
        kind=st.selectbox('Chart type',['Histogram','Category frequencies','Box','Scatter','Correlation','Group comparison','Time trend','Map'])
        x=st.selectbox('X / category / latitude',list(df));y=st.selectbox('Y / metric / longitude',numeric or list(df));a=st.selectbox('Aggregation',['count','mean','median','sum']);gran=st.selectbox('Time granularity',['D','W','M','Q','Y']);ct=st.text_input('Chart title',kind)
        if kind=='Map':st.caption('Select actual latitude for X and longitude for Y. Coordinates are range-validated; geographic names are not geocoded.')
        if st.button('Add chart',type='primary'):
            if len(S.dashboard['charts'])>=12:st.error('Maximum 12 charts.')
            elif kind=='Time trend' and x not in dates:st.error('Confirm X as a date column first.')
            elif kind in ['Group comparison','Category frequencies'] and x in numeric:st.error('Choose a category for X; confirm category codes in Data Quality.')
            elif kind=='Histogram' and x not in numeric:st.error('Choose a numeric X field.')
            else:S.dashboard['charts'].append({'kind':kind,'x':x,'y':y,'aggregation':a,'granularity':gran,'title':ct})
    if not S.dashboard['charts'] and numeric:S.dashboard['charts']=[{'kind':'Histogram','x':numeric[0],'y':numeric[0],'title':numeric[0]+' distribution'}]
    S.dashboard.update(title=title,kpis=kpi_cols,aggregation=agg,filters=filters)
    st.subheader(title);st.write(insight(view,numeric));figures=[]
    for i,c in enumerate(S.dashboard['charts']):
        try:
            fig=chart(view,c);figures.append(fig);show_chart(fig,'dashboard_'+str(i))
            if st.button('Remove chart',key='remove'+str(i)):S.dashboard['charts'].pop(i);st.rerun()
        except Exception as e:st.warning(f'Chart needs different fields: {e}')
    S.figures=figures
    search=st.text_input('Search all displayed fields')
    table=view[view.astype(str).apply(lambda c:c.str.contains(search,case=False,regex=False)).any(axis=1)] if search else view
    st.caption(f'{len(view):,} filtered records; {len(table):,} search matches. Click a table header to sort. Charts use at most 5,000 points for scatter/map.')
    st.dataframe(table,use_container_width=True);download('Download filtered data',view.to_csv(index=False),'filtered.csv');download('Save dashboard settings',json.dumps(S.dashboard,indent=2),'dashboard.json')
elif page=='BI Studio':
    render_studio(df,roles,S.source_key+json.dumps(S.cleaning,sort_keys=True))
elif page=='Model Lab':
    st.subheader('Choose the question before the algorithm')
    question=st.text_input('What would you like to understand or predict?',S.get('model_config',{}).get('question',''))
    st.caption('Possible questions: which fields are associated with an outcome? Can we predict an unknown number or category? Which records form similar groups? How might a time series continue?')
    goal=st.selectbox('Goal',['Explore','Regression','Classification','Clustering','Anomaly detection','Forecasting'])
    st.info(f'Training uses the full active analysis dataset: {len(df):,} rows. Dashboard filters are not used. To change training data, upload a deliberate subset or apply confirmed cleaning.')
    cfg={'goal':goal,'question':question,'roles':roles.copy()}
    if goal=='Explore':st.write(insight(df,numeric))
    elif goal=='Forecasting':
        if not dates or not numeric:st.warning('Confirm at least one date field and numeric value field in Data Quality.');st.stop()
        cfg.update(date=st.selectbox('Date column',dates),target=st.selectbox('Value to forecast',numeric),frequency=st.selectbox('Regular frequency',['D','W','MS','QS','YS']),aggregation=st.selectbox('Period aggregation',['mean','sum','median']),horizon=int(st.number_input('Future periods',1,90,7)),season=int(st.number_input('Season length (periods)',2,365,7)))
        st.caption('Forecasting aggregates the full series across entities. Filter entities before uploading when separate forecasts are needed. Missing periods cause a clear error; no interpolation across future dates.')
    else:
        target=st.selectbox('Target (confirm meaning, including numeric category codes)',list(df),index=len(df.columns)-1) if goal in ['Regression','Classification'] else None
        group=st.selectbox('Repeated entity / grouping column',['None']+list(df)) if target else 'None'
        split_date=st.selectbox('Chronological split date',['None']+dates) if target and group=='None' else 'None'
        excluded=[target,group]+[c for c in df if roles.get(c) in ['identifier','group']]
        defaults=[c for c in df if c not in excluded and df[c].nunique()>1 and not (roles.get(c)=='category' and df[c].nunique()>100)]
        features=st.multiselect('Features available at prediction time', [c for c in df if c not in excluded],default=defaults)
        cfg.update(target=target,features=features,group=None if group=='None' else group,split_date=None if split_date=='None' else split_date)
        suspects=[c for c in features if any(word in c.lower() for word in ['outcome','result','future','after','final','target','label'])]
        if suspects:st.warning('Potential leakage fields to review: '+', '.join(suspects))
        st.caption('Leakage review: remove fields recorded after the outcome, target proxies, duplicate target measures and raw identifiers. Automated names cannot detect every leakage source.')
        if target and goal=='Classification':
            counts=df[target].value_counts(dropna=False);st.write('Target class counts',counts)
            if len(counts)>1 and counts.min()/counts.sum()<.1:st.warning('Class imbalance: compare macro F1 and per-class confusion, not accuracy alone.')
        if goal=='Clustering':cfg.update(k=int(st.number_input('Number of clusters',2,min(12,max(2,len(df)-1)),3)),method=st.selectbox('Method',['K-means','Mini-batch K-means']))
        if goal=='Anomaly detection':cfg['contamination']=st.slider('Fraction flagged (assumption, not measured prevalence)',.01,.25,.05)
    if goal!='Explore':
        st.subheader('Suitable methods');st.dataframe(recommendations(goal,df,cfg.get('features',[])),hide_index=True,use_container_width=True)
        st.caption(f'Dataset: {len(df):,} records, {len(cfg.get("features",[]))} selected features; {df.isna().mean().mean():.1%} missing cells. Candidates use fixed bounded parameters, at most 3-fold CV, and no exhaustive tuning. High-cardinality features are bounded to 25 encoded categories.')
        confirmed=st.checkbox('I confirm the goal, target, prediction-time features and entity/time split are appropriate.')
        if st.button('Train and evaluate',type='primary',disabled=not confirmed):
            progress=st.progress(0.,text='Preparing data')
            try:
                if goal in ['Regression','Classification']:r=train(df,cfg,lambda p,t:progress.progress(p,text=t))
                elif goal=='Forecasting':r=forecast(df,cfg)
                else:r=unsupervised(df,cfg)
                S.result=r;S.model_config=cfg;S.pop('predictions',None);progress.progress(1.,text='Complete');st.success('Training complete. Results below; predictions are ready in Prediction Studio.')
            except Exception as e:st.error(f'Training could not complete: {e}')
    if S.get('result'):ml_view(S.result)
elif page=='Prediction Studio':
    r=S.get('result')
    if not r:st.info('Train a model in Model Lab to activate Prediction Studio.');st.stop()
    st.subheader(f"Prediction Studio · {r['name']}")
    if r.get('test_metrics'):st.write('Relevant held-out test results:',r['test_metrics'])
    st.caption('Predictions are estimates. Probability outputs, where available, are not necessarily calibrated. Source units apply unless you specify a unit below.')
    if r['kind']=='Forecasting':
        ml_view(r);st.dataframe(r['future'].rename('forecast'));download('Download future forecasts',r['future'].rename('forecast').to_csv(),'forecast.csv');st.stop()
    features=r['config']['features'];schema=r['schema_data'];tab1,tab2,tab3=st.tabs(['Single prediction','Batch predictions','What-if explorer'])
    def input_fields(prefix,defaults=None):
        values={};cols=st.columns(2)
        for i,c in enumerate(features):
            s=schema[c].dropna();role=r['config']['roles'].get(c);default=defaults[c] if defaults else None
            with cols[i%2]:
                if role=='numeric':
                    v=float(default) if default is not None and pd.notna(default) else float(pd.to_numeric(s,errors='coerce').median()) if len(s) else 0.
                    values[c]=st.number_input(c,value=v if np.isfinite(v) else 0.,key=prefix+c)
                elif role=='date':
                    d=pd.to_datetime(s,errors='coerce').dropna();v=pd.Timestamp(default) if default is not None and pd.notna(default) else d.iloc[0] if len(d) else pd.Timestamp('2025-01-01')
                    values[c]=pd.Timestamp(st.date_input(c,v.date(),key=prefix+c))
                else:
                    opts=sorted(s.astype(str).unique().tolist()) or ['Unknown'];v=str(default) if default is not None else opts[0]
                    values[c]=st.selectbox(c,opts,index=opts.index(v) if v in opts else 0,key=prefix+c)
        return pd.DataFrame([values])
    with tab1:
        single=input_fields('single_');unit=st.text_input('Output unit (optional)')
        if st.button('Predict one record',type='primary'):
            try:
                p,w=predict(single,r);st.dataframe(p.filter(regex='^modelmatch_'));st.caption(f'Model: {r["name"]}. '+unit)
                for warning in w:st.warning(warning)
            except Exception as e:st.error(str(e))
    with tab2:
        template=schema.head(0);download('Download required-column template',template.to_csv(index=False),'prediction_template.csv')
        u=st.file_uploader('New records',type=['csv','xlsx'],key='batch_upload')
        if u:
            try:
                sheet=st.selectbox('Prediction worksheet',pd.ExcelFile(io.BytesIO(u.getvalue())).sheet_names) if u.name.endswith('.xlsx') else 0
                batch=load_data(u.getvalue(),u.name,sheet);st.write('Extra columns preserved:',[c for c in batch if c not in features]);st.dataframe(batch.head())
                if st.button('Predict batch'):
                    p,w=predict(batch,r);S.predictions=p
                    for warning in w:st.warning(warning)
            except Exception as e:st.error(str(e))
        if S.get('predictions') is not None:st.dataframe(S.predictions);download('Download prediction results',S.predictions.to_csv(index=False),'predictions.csv')
    with tab3:
        idx=st.number_input('Reference record from development data',0,len(schema)-1,0);base=schema.iloc[[idx]].copy();st.write('Original inputs',base)
        adjusted=input_fields('whatif_',base.iloc[0].to_dict())
        if st.button('Compare outcomes'):
            try:
                a,w1=predict(base,r);b,w2=predict(adjusted,r);st.write('Original',a.filter(regex='^modelmatch_'));st.write('Adjusted',b.filter(regex='^modelmatch_'))
                for warning in w1+w2:st.warning(warning)
            except Exception as e:st.error(str(e))
        st.warning('Changing model inputs demonstrates a predictive association. It does not establish what would happen under a real intervention.')
elif page=='Exports':
    st.subheader('Take your work with you')
    st.caption('Exports contain the data or trained parameters you select. Keep sensitive exports private. A report is a file; it is not a hosted live dashboard.')
    download('Cleaned dataset',df.to_csv(index=False),'cleaned.csv')
    view=S.get('filtered',df);download('Filtered records',view.to_csv(index=False),'filtered.csv')
    dashboard=S.get('dashboard',{'title':'ModelMatch','charts':[{'kind':'Histogram','x':numeric[0],'title':'Distribution'}] if numeric else []})
    config=S.get('model_config',{});code=script(config,S.cleaning,dashboard);nb=notebook(code)
    download('Python script',code,'workflow.py','text/x-python');download('Jupyter notebook',nb,'workflow.ipynb','application/json')
    download('Runnable workflow package (modules + examples)',bundle(code,nb,dashboard),'ModelMatch_workflow.zip','application/zip')
    download('Dashboard configuration',json.dumps(dashboard,indent=2),'dashboard.json')
    download('Standalone Streamlit dashboard project',bundle(dashboard=dashboard,standalone=True),'ModelMatch_dashboard.zip','application/zip')
    r=S.get('result');figures=S.get('figures',[])
    if not figures:
        for c in dashboard.get('charts',[]):
            try:figures.append(chart(view,c))
            except Exception:pass
    download('HTML analysis report',report(view,figures,r,insight(view,numeric),S.log),'ModelMatch_report.html','text/html')
    download('Dependency versions',json.dumps(versions(),indent=2),'versions.json')
    if r:
        if 'comparison' in r:download('Model comparison metrics (validation)',r['comparison'].to_csv(index=False),'validation_metrics.csv')
        if 'test_metrics' in r:download('Final test metrics',json.dumps(r['test_metrics'],indent=2),'test_metrics.json')
        if r['kind']=='Forecasting':download('Future forecast',r['future'].rename('forecast').to_csv(),'forecast.csv')
        else:
            b=io.BytesIO();joblib.dump({'result':r,'versions':versions(),'transformations':S.log},b);download('Model + fitted preprocessing + versions',b.getvalue(),'modelmatch_model.joblib')
            st.warning('Serialized models can execute code when loaded. Only load joblib/pickle files from trusted sources. This app does not load uploaded models.')
    st.info('Optional external AI assistant is not enabled in this version. All analysis, recommendations, explanations and exports work without a paid API. No dataset is sent to an external AI service.')
    st.markdown('**Deploy a live dashboard:** push this project to GitHub, open Streamlit Community Cloud, choose the repository and `app.py`, then deploy. A public sharing URL is available only after successful deployment.')
st.divider();st.caption('Created by Blessing Sekeramavi · ModelMatch · Reproducible analysis, honest evaluation.')
