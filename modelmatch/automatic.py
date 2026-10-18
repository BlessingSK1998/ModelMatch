"""Deterministic data-driven chart planning and guided automatic modelling."""
import hashlib
import json
from itertools import combinations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from .data import quality
from .dashboard import chart,insight
from .ui import polish
from .modeling import train,unsupervised,recommendations
from .forecasting import forecast
from .prediction import predict
from .exports import report

MAX_CHARTS=80


def chart_plan(df,roles):
    """Cover each suitable field, plus bounded relationships. No fabricated coordinates."""
    numeric=[c for c in df.select_dtypes('number') if roles.get(c)=='numeric' and df[c].nunique()>1]
    categories=[c for c in df if roles.get(c)=='category' and 1<df[c].nunique()<=100]
    dates=[c for c in df if roles.get(c)=='date' and pd.to_datetime(df[c],errors='coerce',format='mixed').notna().sum()>1]
    specs=[]
    def add(section,kind,x=None,y=None,reason='',**kwargs):
        specs.append({'section':section,'kind':kind,'x':x,'y':y,'reason':reason,**kwargs})
    for c in numeric[:20]:
        add('Distributions','Histogram',c,c,'Shows the shape, spread and concentration of observed numeric values.',title=f'{c} · distribution')
        add('Distributions','Box',c,c,'Shows the median, interquartile range and potential unusual values without deleting them.',title=f'{c} · spread & potential outliers')
    for c in categories[:10]:add('Categories','Category frequencies',c,None,'Compares observed category counts; at most 30 categories are displayed.',title=f'{c} · category counts',aggregation='count')
    if len(numeric)>1:
        add('Relationships','Correlation',None,None,'Pearson correlations among confirmed numeric fields; association is not causation.',title='Numeric relationships · correlation matrix',columns=numeric[:20])
        pairs=[]
        corr=df[numeric[:20]].corr()
        for a,b in combinations(numeric[:20],2):
            value=corr.loc[a,b]
            if pd.notna(value):pairs.append((abs(value),a,b,value))
        for _,a,b,r in sorted(pairs,reverse=True)[:6]:add('Relationships','Scatter',a,b,f'Prioritised by absolute Pearson correlation (r={r:.3f}); up to 5,000 plotted records. Association does not establish causation.',title=f'{a} vs {b}')
    for c in categories[:3]:
        for n in numeric[-2:]:
            if df[c].nunique()<=20:add('Groups','Box',c,n,'Compares the distribution of a numeric measurement across observed categories.',title=f'{n} by {c}')
    for d in dates[:3]:
        parsed=pd.to_datetime(df[d],errors='coerce',format='mixed');span=(parsed.max()-parsed.min()).days
        granularity='D' if span<=400 else 'M'
        for n in numeric[-3:]:add('Time trends','Time trend',d,n,'Aggregates observed values by calendar period; missing periods remain gaps, not zeroes.',title=f'{n} over time',aggregation='mean',granularity=granularity)
    coordinate_fields=[c for c in df.select_dtypes('number') if roles.get(c)=='numeric']
    lat=next((c for c in coordinate_fields if c.lower() in ['latitude','lat']),None)
    lon=next((c for c in coordinate_fields if c.lower() in ['longitude','lon','lng']),None)
    if lat and lon and (df[lat].between(-90,90)&df[lon].between(-180,180)).any():add('Geography','Map',lat,lon,'Uses only supplied latitude/longitude pairs within valid ranges.',title='Observed geographic coverage')
    return specs[:MAX_CHARTS]


def prepare_chart(df,spec,roles):
    if spec['kind']=='Correlation':return chart(df[spec['columns']],spec)
    return chart(df,spec)


def suggest_targets(df,roles):
    priorities=['target','label','outcome','class','species','disease_progression','sales','revenue','price','churn','score','grade','demand']
    candidates=[]
    for index,c in enumerate(df):
        if roles.get(c) in ['identifier','group','date'] or df[c].nunique()<2:continue
        if roles.get(c)=='category' and (df[c].nunique()>30 or df[c].value_counts().min()<5):continue
        score=100-priorities.index(c.lower()) if c.lower() in priorities else 0
        candidates.append((score,index,c))
    return [c for _,_,c in sorted(candidates,reverse=True)]


def default_features(df,roles,target,group=None):
    return [c for c in df if c not in [target,group] and roles.get(c) not in ['identifier','group'] and df[c].nunique()>1 and df[c].isna().mean()<.8 and not(roles.get(c)=='category' and df[c].nunique()>100)][:40]


def analysis_signature(df,roles,source):
    return hashlib.sha256((source+json.dumps(roles,sort_keys=True)+str(df.shape)).encode()).hexdigest()


def render(df,roles,source,ml_view,navigate):
    S=st.session_state;signature=analysis_signature(df,roles,source)
    if S.get('auto_signature')!=signature:
        for key in list(S):
            if key.startswith('auto_'):S.pop(key,None)
        S.auto_signature=signature;S.auto_plan=chart_plan(df,roles);S.auto_quality=quality(df)
        S.auto_chart_settings={'title':'Automatic dataset analysis','charts':S.auto_plan,'kpis':[],'aggregation':'mean','filters':{}}
    plan=S.auto_plan
    st.subheader('Your automatic analysis is ready')
    st.success(f'{len(plan)} interactive charts prepared from this dataset. Distributions, categories, relationships, group comparisons, time trends and maps are included when the corresponding fields exist.')
    st.caption('No chart setup is required. Plotly visuals support hover, zoom, pan and image downloads. The chart inventory explains why each visual is included. Wide datasets are bounded to 20 numeric fields, 10 categories, 3 dates and 80 charts.')
    analysis_tab,predict_tab,inventory_tab=st.tabs(['Automatic graphs & insights','Automatic models & predictions','Chart inventory & exports'])
    with analysis_tab:
        sections=['Data quality']+list(dict.fromkeys(s['section'] for s in plan))
        section=st.segmented_control('Analysis section',sections,default=sections[1] if len(sections)>1 else 'Data quality',key='auto_section')
        view=df.copy()
        with st.expander('Interactive filters · update every graph and insight below'):
            candidates=[c for c in df if roles.get(c)=='category' and df[c].nunique()<=40]
            for c in st.multiselect('Filter categorical fields',candidates,key='auto_filter_fields'):
                values=sorted(df[c].astype(str).unique());chosen=st.multiselect(c,values,default=values,key='auto_filter_'+c);view=view[view[c].astype(str).isin(chosen)]
            nums=[c for c in df.select_dtypes('number') if roles.get(c)=='numeric' and df[c].nunique()>1]
            range_col=st.selectbox('Numeric range filter',['None']+nums,key='auto_range_col')
            if range_col!='None':
                lo,hi=float(df[range_col].min()),float(df[range_col].max())
                if np.isfinite(lo) and np.isfinite(hi):
                    limits=st.slider(range_col,lo,hi,(lo,hi),key='auto_range_values_'+range_col);view=view[view[range_col].between(*limits)]
        st.caption(f'{len(view):,} / {len(df):,} records shown. These filters never change the training data.')
        if view.empty:st.info('No records match these filters. Reset or widen the selections.')
        elif section=='Data quality':
            q=quality(view);left,right=st.columns([1.5,1])
            with left:
                missing=q.sort_values('missing_%',ascending=False).head(30)
                fig=polish(px.bar(missing,x='missing_%',y='column',orientation='h',title='Missing values by field',labels={'missing_%':'Missing values (%)','column':'Field'}));st.plotly_chart(fig,use_container_width=True,key='auto_missing')
            with right:
                with st.container(border=True):
                    st.markdown('**Calculated quality summary**')
                    st.write(f'{int(view.duplicated().sum()):,} exact duplicate records. {int(view.isna().sum().sum()):,} missing cells. {int((view.nunique()<=1).sum())} constant or empty fields.')
                    st.write('Review potential outliers; no unusual observations are removed automatically.')
                    st.button('Review and clean data',on_click=navigate,args=('Data Quality',),key='auto_review_quality')
            st.dataframe(q,hide_index=True,use_container_width=True)
        else:
            selected=[s for s in plan if s['section']==section]
            st.write(insight(view,[c for c in view.select_dtypes('number') if roles.get(c)=='numeric']))
            cols=st.columns(2)
            for i,spec in enumerate(selected):
                with cols[i%2]:
                    try:
                        fig=prepare_chart(view,spec,roles);fig.update_layout(height=370)
                        st.plotly_chart(fig,use_container_width=True,key='auto_graph_'+section+'_'+str(i));st.caption(spec['reason'])
                    except Exception as e:st.warning(f"{spec['title']}: {e}")
        st.download_button('Download displayed records',view.to_csv(index=False),'automatic_filtered.csv',key='auto_filtered_export')
        with st.expander('Display every generated interactive graph in one report'):
            if st.checkbox('Render all generated charts',key='auto_render_all'):
                cols=st.columns(2)
                for i,spec in enumerate(plan):
                    with cols[i%2]:
                        try:st.plotly_chart(prepare_chart(view,spec,roles),use_container_width=True,key='auto_all_'+str(i))
                        except Exception as e:st.caption(f"{spec['title']}: {e}")
    with predict_tab:
        st.subheader('From this dataset to a prediction')
        st.caption('Charts are automatic. Predictions need a meaningful target: confirm the suggested question and prediction-time fields once, then ModelMatch runs the full comparison, evaluation and prediction workflow for you.')
        targets=suggest_targets(df,roles);dates=[c for c in df if roles.get(c)=='date']
        if not targets:st.info('No suitable supervised target was found. You can still use clustering or anomaly detection in Model Lab.')
        else:
            target=st.selectbox('Suggested target · confirm what you want to predict',targets,key='auto_target')
            st.caption(f'Suggested question: can the other prediction-time fields estimate {target}? A suggestion is not proof that this target is appropriate.')
            labels=['Predict a number','Predict a category']+(['Forecast future values'] if dates and roles.get(target)=='numeric' else [])+['Discover groups','Detect unusual records']
            default=1 if roles.get(target)=='category' else 2 if dates and target.lower() in ['sales','revenue','demand'] else 0
            goal_label=st.selectbox('Confirm your prediction goal',labels,index=min(default,len(labels)-1),key='auto_goal')
            goal=dict(zip(['Predict a number','Predict a category','Forecast future values','Discover groups','Detect unusual records'],['Regression','Classification','Forecasting','Clustering','Anomaly detection']))[goal_label]
            cfg={'goal':goal,'target':target,'roles':roles.copy(),'question':f'Estimate {target} from prediction-time features'}
            if goal=='Forecasting':
                date=st.selectbox('Time field',dates,key='auto_forecast_date');series=pd.to_datetime(df[date],errors='coerce',format='mixed').dropna().sort_values().drop_duplicates();gap=series.diff().dropna().dt.total_seconds().median()/86400 if len(series)>1 else 1
                default_frequency='D' if gap<6 else 'W' if gap<20 else 'MS' if gap<80 else 'QS'
                frequency=st.selectbox('Forecast period',['D','W','MS','QS','YS'],index=['D','W','MS','QS','YS'].index(default_frequency),key='auto_frequency')
                horizon=int(st.number_input('Future periods to predict',1,90,7,key='auto_horizon'))
                cfg.update(date=date,frequency=frequency,aggregation='mean',horizon=horizon,season=7 if frequency=='D' else 12 if frequency=='MS' else 4)
                st.caption('Forecasting uses period means across all records. Separate entity forecasts and external future predictors are not implemented. Missing periods are rejected rather than silently filled.')
            else:
                group=st.selectbox('Repeated entity / grouping field',['None']+list(df),key='auto_group')
                group=None if group=='None' else group
                defaults=default_features(df,roles,target,group)
                features=st.multiselect('Prediction-time features · automatic selection, editable',[c for c in df if c not in [target,group] and roles.get(c) not in ['identifier','group']],default=defaults,key='auto_features_'+target+'_'+str(group))
                time_split=st.selectbox('Chronological evaluation field',['None']+dates,key='auto_time_split') if group is None and goal in ['Regression','Classification'] else 'None'
                cfg.update(features=features,group=group,split_date=None if time_split=='None' else time_split,k=3,contamination=.05,method='K-means')
                suspicious=[c for c in features if any(word in c.lower() for word in ['outcome','result','future','after','final','target','label'])]
                if suspicious:st.warning('Potential leakage to review: '+', '.join(suspicious))
                st.caption('Exclude information recorded after the outcome and target proxies. Automated checks cannot detect every leakage source. Numeric category codes can use Predict a category.')
            st.dataframe(recommendations(goal,df,cfg.get('features',[])),hide_index=True,use_container_width=True)
            confirmed=st.checkbox('I confirm this goal and target, the entity/time split, and that the selected features are available before the outcome.',key='auto_confirm_'+target+'_'+goal)
            if st.button('Run complete analysis & predictions',type='primary',disabled=not confirmed,key='auto_train'):
                progress=st.progress(0.,text='Preparing automatic workflow')
                try:
                    if goal in ['Regression','Classification']:result=train(df,cfg,lambda p,t:progress.progress(p,text=t))
                    elif goal=='Forecasting':result=forecast(df,cfg)
                    else:result=unsupervised(df,cfg)
                    S.result=result;S.model_config=cfg;S.pop('auto_html',None)
                    S.auto_prediction_key=json.dumps(cfg,sort_keys=True)
                    S.pop('predictions',None)
                    if goal!='Forecasting':
                        if goal in ['Regression','Classification']:
                            # Only final test records are used for the initial prediction preview.
                            labeled=df.dropna(subset=[target]).copy()
                            if cfg.get('split_date'):labeled=labeled.sort_values(cfg['split_date'],key=lambda s:pd.to_datetime(s),kind='stable')
                            prediction_data=labeled.iloc[result['splits']['test']].copy()
                        else:prediction_data=df.copy()
                        S.auto_predictions,S.auto_prediction_warnings=predict(prediction_data,result)
                    else:S.auto_predictions=result['future'].rename('forecast').reset_index().rename(columns={'index':'date'})
                    progress.progress(1.,text='Analysis, model comparison and predictions complete');st.success('Complete. Interactive evaluation charts and prediction results are below.')
                except Exception as e:st.error(f'The automatic workflow could not complete: {e}')
            if S.get('result') and S.get('auto_prediction_key')==json.dumps(cfg,sort_keys=True) and S.result.get('config')==cfg:
                ml_view(S.result)
                st.subheader('Generated prediction results')
                st.caption('Initial supervised predictions use the untouched final test records; forecasts cover future periods. Use Prediction Studio for new single or batch records.')
                st.dataframe(S.auto_predictions,use_container_width=True)
                st.download_button('Download generated predictions',S.auto_predictions.to_csv(index=False),'automatic_predictions.csv',key='auto_predictions_download')
                st.button('Predict on new records',on_click=navigate,args=('Prediction Studio',),key='auto_open_predictions')
            elif S.get('auto_prediction_key'):st.info('Settings changed. Run the automatic workflow again to generate results for this configuration.')
    with inventory_tab:
        st.dataframe(pd.DataFrame(plan)[[c for c in ['section','title','kind','x','y','reason'] if c in pd.DataFrame(plan)]],hide_index=True,use_container_width=True)
        st.download_button('Download automatic chart plan',json.dumps(plan,indent=2),'automatic_chart_plan.json')
        def create_dashboard():
            S.dashboard={'title':'Automatic dataset dashboard','charts':[s for s in plan if s['kind']!='Correlation'][:12],'kpis':[c for c in df.select_dtypes('number') if roles.get(c)=='numeric'][-2:],'aggregation':'mean','filters':{}}
            for k in ['dash_title','dash_kpis','dash_agg']:S.pop(k,None)
            S.workspace_page='Dashboard'
        st.button('Use generated charts as my dashboard',type='primary',key='auto_make_dashboard',on_click=create_dashboard)
        st.caption('The editable Dashboard starts with up to 12 generated visuals. The automatic report contains the entire generated chart plan.')
        if st.button('Build complete interactive HTML report',key='auto_build_report'):
            figures=[]
            with st.spinner('Building all suitable charts and the analysis report…'):
                figures.append(polish(px.bar(S.auto_quality,x='column',y='missing_%',title='Missing values by field')))
                for spec in plan:
                    try:figures.append(prepare_chart(df,spec,roles))
                    except Exception:pass
                S.auto_html=report(df,figures,S.get('result'),insight(df,[c for c in df.select_dtypes('number') if roles.get(c)=='numeric']),S.get('log',[]))
            st.success(f'Built a report with {len(figures)} interactive graphs.')
        if S.get('auto_html'):st.download_button('Download complete interactive report',S.auto_html,'ModelMatch_automatic_analysis.html','text/html',key='auto_html_download')
    nav=st.columns(3)
    nav[0].button('Open dashboard',on_click=navigate,args=('Dashboard',),key='auto_nav_dashboard',use_container_width=True)
    nav[1].button('Open BI Studio',on_click=navigate,args=('BI Studio',),key='auto_nav_studio',use_container_width=True)
    nav[2].button('Open Prediction Studio',on_click=navigate,args=('Prediction Studio',),key='auto_nav_predictions',use_container_width=True)
