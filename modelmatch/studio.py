"""BI-style report canvas, field configuration and code workbench UI."""
from html import escape
import json
import pandas as pd
import streamlit as st
from .dashboard import chart
from .ui import polish
from .workbench import run_sql,run_pandas,scratchpad_notebook


def render(df,roles,data_key):
    S=st.session_state
    if S.get('bi_source')!=data_key:
        for k in list(S):
            if k.startswith('bi_'):S.pop(k,None)
        S.bi_source=data_key;S.bi_tiles=[];S.bi_history=[]
    if 'bi_pending_title' in S:
        S.bi_report_title=S.pop('bi_pending_title');S.bi_layout=S.pop('bi_pending_layout','Two columns')
    nums=[c for c in df.select_dtypes('number') if roles.get(c) not in ['identifier','group']]
    categories=[c for c in df if roles.get(c)=='category']
    c1,c2=st.columns([3,1]);title=c1.text_input('Report name','Untitled report',key='bi_report_title');layout=c2.selectbox('Canvas layout',['Two columns','One column'],key='bi_layout')
    report_tab,code_tab,scratch_tab=st.tabs(['Report canvas','Code workbench','Python scratchpad'])
    with report_tab:
        canvas,fields=st.columns([3.1,1],gap='large')
        with fields:
            with st.container(border=True):
                st.markdown('<div class="insight-head">VISUALISATIONS</div>',unsafe_allow_html=True)
                visual=st.selectbox('Visual type',['Histogram','Bar','Line','Scatter','Box','Correlation','KPI'],key='bi_visual')
                x=st.selectbox('Axis / category',list(df),key='bi_x')
                y=st.selectbox('Value',nums or list(df),index=len(nums)-1 if nums else 0,key='bi_y')
                agg=st.selectbox('Calculation',['count','mean','sum','median'],key='bi_agg')
                chart_title=st.text_input('Visual title',value='My visual',key='bi_visual_title')
                if st.button('Add to report',type='primary',use_container_width=True):
                    kind={'Bar':'Group comparison','Line':'Time trend'}.get(visual,visual)
                    if len(S.bi_tiles)>=12:st.warning('Maximum 12 visuals per report.')
                    elif visual=='Line' and roles.get(x)!='date':st.warning('Choose a date field for a time trend; confirm its role in Data Quality.')
                    elif visual=='Histogram' and x not in nums:st.warning('Histograms require a numeric axis.')
                    elif visual in ['Scatter','Box','KPI'] and y not in nums:st.warning('Choose a numeric value field.')
                    else:S.bi_tiles.append({'source':'dataset','kind':kind,'x':x,'y':y,'aggregation':agg,'title':chart_title,'granularity':'M'});st.rerun()
                st.markdown('<div class="insight-head" style="margin-top:20px">DATA FIELDS</div>',unsafe_allow_html=True)
                search=st.text_input('Find a field',key='bi_find_field')
                for c in [c for c in df if search.lower() in c.lower()][:40]:
                    icon='Σ' if c in nums else '◷' if roles.get(c)=='date' else 'Aa'
                    st.markdown(f'<div class="field-row"><span>{icon}</span> {escape(c)}</div>',unsafe_allow_html=True)
        with canvas:
            st.subheader(title)
            st.caption(f'{len(df):,} records · {len(S.bi_tiles)} saved visuals · Changes are session-scoped')
            with st.expander('Report filters',expanded=False):
                selected=st.multiselect('Category filters',categories,default=[],key='bi_filter_columns')
                view=df.copy()
                for c in selected:
                    options=sorted(df[c].astype(str).unique())
                    if len(options)>100:st.caption(f'{c}: use the code workbench for high-cardinality filtering.');continue
                    values=st.multiselect(c,options,default=options,key='bi_filter_'+c)
                    view=view[view[c].astype(str).isin(values)]
            if not S.bi_tiles:
                with st.container(border=True):
                    st.markdown('<div class="canvas-empty"><div class="canvas-symbol">▦</div><h3>Your next report starts here.</h3><p>Choose a visual and fields in the right panel, or run a query<br>in Code workbench and pin its result to this canvas.</p></div>',unsafe_allow_html=True)
                    if nums and st.button('Create a starter report',type='primary'):
                        focus=nums[-1];S.bi_tiles=[{'source':'dataset','kind':'KPI','x':focus,'y':focus,'aggregation':'mean','title':'Average '+focus},{'source':'dataset','kind':'Histogram','x':focus,'y':focus,'title':focus+' distribution'}];st.rerun()
            columns=st.columns(2 if layout=='Two columns' else 1)
            for i,tile in enumerate(S.bi_tiles):
                with columns[i%len(columns)]:
                    with st.container(border=True):
                        try:
                            if tile['source']=='query':
                                data=pd.DataFrame(tile['data']);st.caption('Pinned query snapshot · independent of report filters')
                            else:data=view
                            if tile['kind']=='KPI':
                                val=getattr(pd.to_numeric(data[tile['y']],errors='coerce'),tile.get('aggregation','mean'))()
                                st.metric(tile['title'],f'{val:,.4g}' if pd.notna(val) else '—')
                                st.caption(f"{tile.get('aggregation','mean').title()} of nonmissing {tile['y']} values · {len(data):,} records")
                            elif tile['kind']=='Table':st.subheader(tile['title']);st.dataframe(data,use_container_width=True,height=270)
                            else:
                                fig=polish(chart(data,tile));fig.update_layout(height=350)
                                st.plotly_chart(fig,use_container_width=True,key='bi_plot_'+str(i))
                                st.download_button('Export visual',fig.to_html(include_plotlyjs=True),f'visual_{i+1}.html',key='bi_export_'+str(i))
                            with st.expander('Visual settings'):
                                changed=st.text_input('Title',tile['title'],key='bi_rename_'+str(i))
                                if st.button('Update title',key='bi_update_'+str(i)):tile['title']=changed;st.rerun()
                                moves=st.columns(2)
                                if moves[0].button('Move earlier',key='bi_up_'+str(i),disabled=i==0):S.bi_tiles[i-1],S.bi_tiles[i]=S.bi_tiles[i],S.bi_tiles[i-1];st.rerun()
                                if moves[1].button('Remove',key='bi_remove_'+str(i)):S.bi_tiles.pop(i);st.rerun()
                        except Exception as e:st.warning(f'This visual needs attention: {e}')
            st.divider()
            cfg={'version':1,'title':title,'layout':layout,'visuals':S.bi_tiles}
            st.download_button('Save report project',json.dumps(cfg,indent=2,default=str),'modelmatch_report.json')
            st.caption('Report projects include query snapshots if you pinned query results. Keep exports containing private records private. Dataset-backed visuals need the same dataset when reopened.')
            saved=st.file_uploader('Open report project',type=['json'],key='bi_project_upload')
            if saved and st.button('Open saved report'):
                try:
                    if len(saved.getvalue())>2_000_000:raise ValueError('Report configuration limit: 2 MB.')
                    config=json.loads(saved.getvalue());tiles=config['visuals']
                    if not isinstance(tiles,list) or len(tiles)>12:raise ValueError('Report supports at most 12 visuals.')
                    for t in tiles:
                        if t['kind'] not in ['KPI','Table','Histogram','Group comparison','Time trend','Scatter','Box','Correlation']:raise ValueError('Unsupported visual type.')
                        source=t.get('source')
                        if source not in ['dataset','query']:raise ValueError('Unknown visual source.')
                        d=pd.DataFrame(t['data']) if source=='query' else df
                        if len(d)>5000 and source=='query':raise ValueError('Query snapshot limit: 5,000 rows.')
                        if t['kind'] not in ['Correlation','Table'] and (t.get('x') not in d or t.get('y') not in d):raise ValueError('Report fields do not match this dataset.')
                        if t.get('aggregation','mean') not in ['count','mean','sum','median']:raise ValueError('Unsupported aggregation.')
                    S.bi_tiles=tiles;S.bi_pending_title=str(config.get('title','Untitled report'))[:150];S.bi_pending_layout=config.get('layout','Two columns');st.rerun()
                except Exception as e:st.error(f'Could not open report: {e}')
    with code_tab:
        st.markdown('<div class="workbench-bar"><span>⌘ &nbsp; DATA WORKBENCH</span><span>dataset / df &nbsp;·&nbsp; session only</span></div>',unsafe_allow_html=True)
        editor,reference=st.columns([3,1],gap='large')
        with reference:
            with st.container(border=True):
                st.markdown('**Available data**')
                st.code('SQL: dataset\nPython: df',language='text')
                st.caption('SQL: read-only SELECT queries. Python: an interpreted pandas-expression subset. Full scripts can be written in the scratchpad and exported.')
                st.caption('Up to 10,000 input rows, 5,000 result rows. No server-side imports, arbitrary Python, network calls or file access.')
                st.dataframe(pd.DataFrame({'Field':df.columns,'Role':[roles.get(c,'category') for c in df]}),hide_index=True,height=300,use_container_width=True)
        with editor:
            language=st.segmented_control('Query language',['SQL','Pandas expression'],default='SQL',key='bi_language')
            examples={'Preview records':('SELECT * FROM dataset LIMIT 25;','df.head(25)'), 'Summary statistics':('SELECT COUNT(*) AS records FROM dataset;','df.describe()')}
            if nums:
                col=nums[-1];quoted='"'+col.replace('"','""')+'"'
                examples['Numeric summary']=(f'SELECT AVG({quoted}) AS average_value, MIN({quoted}) AS minimum_value, MAX({quoted}) AS maximum_value FROM dataset;',f'df[{col!r}].describe()')
            if nums and categories:
                x=categories[0];y=nums[-1];qx='"'+x.replace('"','""')+'"';qy='"'+y.replace('"','""')+'"'
                examples['Compare groups']=(f'SELECT {qx}, AVG({qy}) AS average_value, COUNT(*) AS records FROM dataset GROUP BY {qx} ORDER BY average_value DESC;',f'df.groupby({x!r})[{y!r}].mean().reset_index()')
            starter=st.selectbox('Starter query',list(examples),key='bi_starter')
            code_key='bi_sql_code' if language=='SQL' else 'bi_pandas_code'
            if code_key not in S:S[code_key]=examples[starter][0 if language=='SQL' else 1]
            if st.button('Load starter query'):S[code_key]=examples[starter][0 if language=='SQL' else 1];st.rerun()
            with st.form('bi_code_form',border=False):
                code=st.text_area('Code editor',height=220,key=code_key,label_visibility='collapsed')
                run=st.form_submit_button('Run query' if language=='SQL' else 'Run expression',type='primary')
            st.caption('SQL identifiers containing spaces should be wrapped in double quotes. Pandas example: df.groupby("region")["sales"].mean().reset_index().')
            if run:
                try:
                    output=(run_sql if language=='SQL' else run_pandas)(df,code)
                    S.bi_result=output.frame;S.bi_run_info={'language':language,'code':code,'duration_ms':output.duration_ms,'truncated':output.truncated}
                    S.bi_history=([S.bi_run_info]+S.bi_history)[:10]
                except Exception as e:st.error(str(e))
            if S.get('bi_result') is not None:
                result=S.bi_result;info=S.bi_run_info
                st.markdown(f'**Results** · {len(result):,} rows × {len(result.columns)} columns · {info["duration_ms"]:,.0f} ms')
                st.caption('Last successful run: '+info['language']+'. Results remain a snapshot until you run the query again.')
                if info['truncated']:st.warning('Result truncated to 5,000 rows. Aggregate or narrow the query for a complete result.')
                st.dataframe(result,use_container_width=True,height=300)
                st.download_button('Download query result',result.to_csv(index=False),'query_result.csv')
                st.download_button('Download query',info['code'],'query.sql' if info['language']=='SQL' else 'expression.py')
                with st.expander('Visualise this result and pin to report',expanded=True):
                    types=['Table']+(['Histogram','Bar','Scatter','Line'] if len(result.columns) else [])
                    v=st.selectbox('Result visual',types,key='bi_result_visual')
                    x=st.selectbox('Result X field',list(result) or ['value'],key='bi_result_x')
                    n=list(result.select_dtypes('number'));y=st.selectbox('Result Y field',n or list(result) or ['value'],key='bi_result_y')
                    label=st.text_input('Pinned visual title','Query result',key='bi_pin_title')
                    kind={'Bar':'Group comparison','Line':'Time trend'}.get(v,v)
                    spec={'source':'query','kind':kind,'title':label,'x':x,'y':y,'aggregation':'mean','granularity':'D','data':json.loads(result.to_json(orient='records',date_format='iso'))}
                    if v!='Table':
                        try:st.plotly_chart(polish(chart(result,spec)),use_container_width=True,key='bi_query_preview')
                        except Exception as e:st.info(f'Choose compatible fields: {e}')
                    if st.button('Pin result to report',type='primary'):
                        if len(S.bi_tiles)>=12:st.warning('Report is full. Remove a visual before pinning another.')
                        elif len(result.to_json())>1_000_000:st.warning('Snapshot is too large. Aggregate or filter to a result under 1 MB.')
                        else:S.bi_tiles.append(spec);st.success('Pinned. Open Report canvas to see your visual.')
            with st.expander('Expression reference & session history'):
                st.code('df.head(25)\ndf[["region", "sales"]]\ndf[df["sales"] > 100]\ndf.groupby("region")["sales"].mean().reset_index()\ndf.groupby("region").agg({"sales": ["sum", "mean"]})\ndf["sales"].fillna(0).round(2)',language='python')
                st.caption('Supported: selecting/filtering columns, numeric arithmetic, comparisons, groupby, agg, count/sum/mean/median/min/max/std/var/nunique/size/first/last, describe, head/tail, value_counts, sort_values, sort_index, reset_index, rename, fillna, dropna, round, isna/notna, isin, between, abs and clip. Expressions do not modify the original data or model training data.')
                for item in S.bi_history:st.code(item['code'],language='sql' if item['language']=='SQL' else 'python')
    with scratch_tab:
        st.subheader('A place for your own Python')
        st.caption('Write a full script and download it as Python or a notebook. Full scripts run in your own Python/Jupyter environment; this hosted workspace runs only the supported expression subset and read-only SQL.')
        script=st.text_area('Python scratchpad',value='import pandas as pd\nimport plotly.express as px\n\ndf = pd.read_csv("dataset.csv")\nprint(df.describe(include="all"))\n\n# Add your analysis here.\n',height=380,key='bi_scratchpad')
        st.download_button('Download Python script',script,'analysis.py',mime='text/x-python')
        st.download_button('Download Jupyter notebook',scratchpad_notebook(script),'analysis.ipynb',mime='application/json')
        st.download_button('Download active dataset for this notebook',df.to_csv(index=False),'dataset.csv')
