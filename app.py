from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(page_title='Dynamic Player DNA', page_icon='⚽', layout='wide', initial_sidebar_state='expanded')
BASE=Path(__file__).resolve().parent
DATA=BASE/'data'/'demo_player_data.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

df=load_data()
MOTOR=['td_per90','hsr_per90','sprint_per90','acc_per90','dec_per90','pii']
DISPLAY={'td_per90':'TD / 90','hsr_per90':'HSR / 90','sprint_per90':'Sprint / 90','acc_per90':'ACC / 90','dec_per90':'DEC / 90','pii':'PII'}
STATE_ORDER=['Lower activity','Baseline','Higher activity']
THESIS_TRANSITION=np.array([[.740,.113,.147],[.066,.887,.047],[.189,.089,.722]])
THESIS_GMM=pd.DataFrame({'Profiles':[2,3,4,5],'AIC':[7547.30,7547.06,7548.14,7549.75],'BIC':[7636.28,7682.88,7730.79,7779.24],'ARI':[1.000,.967,.861,.679]})

st.markdown('''
<style>
.stApp{background:radial-gradient(circle at 85% 5%,rgba(50,190,120,.10),transparent 28%),linear-gradient(180deg,#07130f 0%,#091712 58%,#07110d 100%);color:#F3F6F4}
.block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:1550px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#081510 0%,#0a1712 100%);border-right:1px solid rgba(255,255,255,.08)}
.hero{border:1px solid rgba(126,231,135,.18);border-radius:26px;padding:30px 32px;background:radial-gradient(circle at 90% 0%,rgba(126,231,135,.12),transparent 30%),linear-gradient(135deg,rgba(255,255,255,.045),rgba(255,255,255,.012));margin-bottom:16px}
.kicker{color:#7EE787;font-weight:800;letter-spacing:.14em;text-transform:uppercase;font-size:.74rem}
.hero-title{font-size:3.5rem;font-weight:900;line-height:.98;margin:.4rem 0 .55rem}
.hero-sub{color:#C6D2CB;max-width:930px;font-size:1.06rem;line-height:1.6}
.pill{display:inline-block;border:1px solid rgba(126,231,135,.22);color:#CEF6D3;background:rgba(126,231,135,.055);padding:6px 10px;border-radius:999px;margin:11px 6px 0 0;font-size:.76rem;font-weight:750}
.banner{background:rgba(126,231,135,.055);border:1px solid rgba(126,231,135,.16);border-left:3px solid #7EE787;border-radius:12px;padding:11px 14px;color:#D9E6DE;margin-bottom:18px}
.card{border:1px solid rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.034),rgba(255,255,255,.015));border-radius:18px;padding:17px 18px;min-height:145px}
.card .num{color:#7EE787;font-size:1.55rem;font-weight:900}.card .label{font-weight:800;margin:.2rem 0}.card .desc{color:#A9B8B0;font-size:.88rem;line-height:1.45}
.insight{border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:15px 16px;background:rgba(255,255,255,.025);color:#D8E2DC}.insight b{color:#7EE787}
[data-testid="stMetric"]{border:1px solid rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.015));border-radius:16px;padding:14px 15px}
[data-testid="stMetricLabel"]{color:#A9B8B0} footer{visibility:hidden}
</style>
''',unsafe_allow_html=True)

@st.cache_data
def pca_bundle(data):
    scaler=StandardScaler(); X=scaler.fit_transform(data[MOTOR])
    pca=PCA(n_components=6,random_state=42); pcs=pca.fit_transform(X)
    pcdf=pd.DataFrame(pcs,columns=[f'PC{i}' for i in range(1,7)],index=data.index)
    return scaler,pca,pcdf

def gmm_bundle(data,k=2):
    scaler,pca,pcdf=pca_bundle(data)
    X=pcdf[['PC1','PC2','PC3']].to_numpy()
    gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=42,n_init=5)
    labels=gmm.fit_predict(X); probs=gmm.predict_proba(X)
    return gmm,labels,probs,pcdf

@st.cache_data
def enrich(data):
    gmm,labels,probs,pcdf=gmm_bundle(data,2)
    tmp=data.copy()
    comp={c:tmp.loc[labels==c,'pii'].mean() for c in [0,1]}
    high=max(comp,key=comp.get); low=1-high
    tmp['gmm_lower_prob']=probs[:,low]; tmp['gmm_higher_prob']=probs[:,high]
    tmp['hybridity']=1-np.abs(tmp['gmm_higher_prob']-tmp['gmm_lower_prob'])
    stab={}; hyb={}; dom={}
    for p,g in tmp.sort_values(['player_id','match_no']).groupby('player_id'):
        arr=g[['gmm_lower_prob','gmm_higher_prob']].to_numpy(); d=np.linalg.norm(np.diff(arr,axis=0),axis=1)
        stab[p]=float(np.clip(1-d.mean()/1.25,0,1)); hyb[p]=float(g.hybridity.mean()); dom[p]=float(np.maximum(g.gmm_lower_prob,g.gmm_higher_prob).mean())
    tmp['stability']=tmp.player_id.map(stab); tmp['mean_hybridity']=tmp.player_id.map(hyb); tmp['dominant_prob']=tmp.player_id.map(dom)
    return tmp

model_df=enrich(df)

def theme(fig,h=None):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(255,255,255,.018)',font=dict(color='#EAF0EC'),margin=dict(l=35,r=25,t=55,b=35),legend=dict(bgcolor='rgba(0,0,0,0)'))
    if h: fig.update_layout(height=h)
    fig.update_xaxes(gridcolor='rgba(255,255,255,.06)'); fig.update_yaxes(gridcolor='rgba(255,255,255,.06)')
    return fig

def empirical_transition(data):
    counts=np.zeros((3,3)); code={s:i for i,s in enumerate(STATE_ORDER)}
    for _,g in data.sort_values(['player_id','match_no']).groupby('player_id'):
        seq=[code[s] for s in g.hmm_state]
        for a,b in zip(seq[:-1],seq[1:]): counts[a,b]+=1
    return counts/counts.sum(axis=1,keepdims=True)

def survival(data):
    times=[]; observed=[]
    for _,g in data.sort_values(['player_id','match_no']).groupby('player_id'):
        s=g.hmm_state.tolist(); hit=None
        for i in range(len(s)-1):
            if s[i]=='Lower activity' and s[i+1]=='Higher activity': hit=i+2; break
        times.append(hit if hit is not None else len(s)); observed.append(hit is not None)
    surv=1.0; xs=[0]; ys=[1.0]
    for t in sorted(set(times)):
        at=sum(x>=t for x in times); d=sum((x==t) and o for x,o in zip(times,observed))
        if at and d:
            xs += [t,t]; ys += [surv,surv*(1-d/at)]; surv*=1-d/at
    xs.append(max(times)); ys.append(surv)
    return times,observed,xs,ys

st.sidebar.markdown('## ⚽ Dynamic Player DNA')
st.sidebar.caption('Advanced public portfolio demo')
page=st.sidebar.radio('Navigate',['Executive overview','Player DNA lab','Cohort explorer','PCA explorer','Clustering lab','Dynamic trajectory','HMM & transitions','Survival analysis','Methodology'],label_visibility='collapsed')
st.sidebar.divider()
teams=st.sidebar.multiselect('Team group',sorted(df.team.unique()),default=sorted(df.team.unique()))
roles=st.sidebar.multiselect('Role',sorted(df.role.unique()),default=sorted(df.role.unique()))
phases=st.sidebar.multiselect('Season phase',sorted(df.season_phase.unique()),default=sorted(df.season_phase.unique()))
filtered=model_df[model_df.team.isin(teams)&model_df.role.isin(roles)&model_df.season_phase.isin(phases)].copy()
if filtered.empty: st.warning('No data match the selected filters.'); st.stop()
st.sidebar.divider(); st.sidebar.caption('All player-level data are synthetic. Thesis metrics shown separately are aggregate results from the original research.')

st.markdown('''
<div class="hero"><div class="kicker">Football analytics · Machine learning · Dynamic profiling</div><div class="hero-title">Dynamic Player DNA</div><div class="hero-sub">An interactive portfolio application exploring not only <b>what type of profile a player resembles</b>, but also <b>how that profile changes from match to match</b>.</div><div><span class="pill">PCA</span><span class="pill">K-means</span><span class="pill">GMM</span><span class="pill">Stability</span><span class="pill">Hybridity</span><span class="pill">HMM</span><span class="pill">Kaplan–Meier</span></div></div>
<div class="banner"><b>Data privacy:</b> the interactive player-level dataset in this application is fully synthetic. The thesis used real academy data; no raw club or player records are published here.</div>
''',unsafe_allow_html=True)

if page=='Executive overview':
    st.subheader('Executive overview'); st.caption('Thesis aggregate results are shown as benchmarks; interactive charts use synthetic demo data.')
    c1,c2,c3,c4,c5=st.columns(5); c1.metric('Motor observations','799'); c2.metric('Players','37'); c3.metric('PC1–PC3 variance','83.80%'); c4.metric('GMM profiles','2'); c5.metric('HMM states','3')
    st.markdown('### Research pipeline'); cols=st.columns(7)
    blocks=[('01','Data','Cleaning, standardisation, per-90'),('02','PCA','Reduce correlated dimensions'),('03','K-means','Hard clustering baseline'),('04','GMM','Soft profile probabilities'),('05','DNA','Stability & hybridity'),('06','HMM','Hidden states & transitions'),('07','KM','Time-to-transition')]
    for col,(n,l,d) in zip(cols,blocks):
        with col: st.markdown(f'<div class="card"><div class="num">{n}</div><div class="label">{l}</div><div class="desc">{d}</div></div>',unsafe_allow_html=True)
    st.markdown('### Selected thesis findings')
    a,b=st.columns([1.15,1])
    with a:
        results=pd.DataFrame({'Metric':['K-means silhouette','GMM stability ARI','Mean DNA stability','Mean hybridity','HMM baseline persistence','KM median first transition'],'Result':['0.335','1.000','0.703','0.500','88.7%','8 appearances'],'Interpretation':['Moderate cluster separation','Very high repeatability','Average match-to-match profile stability','Average mixed membership','Probability of remaining baseline','Median time to first lower→higher transition']})
        st.dataframe(results,use_container_width=True,hide_index=True)
    with b:
        st.markdown('<div class="insight"><b>Core idea</b><br><br>Traditional clustering asks: <i>Which profile does this observation resemble?</i><br><br>Dynamic Player DNA adds: <i>How stable is that profile, how hybrid is it, and how does it evolve across appearances?</i></div>',unsafe_allow_html=True)

elif page=='Player DNA lab':
    st.subheader('Player DNA lab'); player=st.selectbox('Select fictional player',sorted(filtered.player_id.unique())); p=filtered[filtered.player_id==player].sort_values('match_no')
    c1,c2,c3,c4=st.columns(4); c1.metric('Appearances',len(p)); c2.metric('Stability',f'{p.stability.iloc[0]:.3f}'); c3.metric('Hybridity',f'{p.mean_hybridity.iloc[0]:.3f}'); c4.metric('Dominant-profile probability',f'{p.dominant_prob.iloc[0]:.3f}')
    left,right=st.columns([1.05,1])
    with left:
        pm=p[MOTOR].mean(); cohort=filtered.groupby('player_id')[MOTOR].mean(); scores=[float((cohort[c]<=pm[c]).mean()*100) for c in MOTOR]; labels=[DISPLAY[c] for c in MOTOR]
        fig=go.Figure(go.Scatterpolar(r=scores+[scores[0]],theta=labels+[labels[0]],fill='toself',name=player,line=dict(width=3))); fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),showlegend=False,title='Relative player profile · percentile vs cohort'); theme(fig,475); st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=go.Figure(); fig.add_trace(go.Scatter(x=p.match_no,y=p.gmm_lower_prob,mode='lines+markers',name='Lower profile')); fig.add_trace(go.Scatter(x=p.match_no,y=p.gmm_higher_prob,mode='lines+markers',name='Higher profile')); fig.update_layout(title='Soft GMM membership across appearances',xaxis_title='Appearance',yaxis_title='Probability',yaxis_range=[0,1]); theme(fig,475); st.plotly_chart(fig,use_container_width=True)
    st.markdown('### Similarity engine')
    pl=filtered.groupby('player_id')[MOTOR].mean(); X=StandardScaler().fit_transform(pl); D=euclidean_distances(X); idx=list(pl.index).index(player); order=np.argsort(D[idx])[1:6]
    meta=filtered.groupby('player_id')[['team','role']].first().reindex(pl.index)
    sim=pd.DataFrame({'Player':[pl.index[i] for i in order],'Distance':[round(float(D[idx,i]),3) for i in order],'Team':[meta.iloc[i].team for i in order],'Role':[meta.iloc[i].role for i in order]}); st.dataframe(sim,use_container_width=True,hide_index=True)
    st.download_button('Download synthetic player CSV',p.to_csv(index=False).encode('utf-8'),file_name=f'{player}_synthetic_demo.csv',mime='text/csv')

elif page=='Cohort explorer':
    st.subheader('Cohort explorer'); group=st.selectbox('Group comparison',['team','role','season_phase']); metric=st.selectbox('Metric',MOTOR,format_func=lambda x:DISPLAY[x])
    summary=filtered.groupby(group)[metric].agg(['mean','median','std','count']).reset_index(); st.dataframe(summary.round(2),use_container_width=True,hide_index=True)
    fig=px.box(filtered,x=group,y=metric,points='outliers',title=f'{DISPLAY[metric]} distribution by {group}'); theme(fig,430); st.plotly_chart(fig,use_container_width=True)

elif page=='PCA explorer':
    st.subheader('PCA explorer'); scaler,pca,pcdf=pca_bundle(filtered); evr=pca.explained_variance_ratio_; cum=np.cumsum(evr)
    c1,c2,c3,c4=st.columns(4); c1.metric('PC1',f'{evr[0]*100:.1f}%'); c2.metric('PC2',f'{evr[1]*100:.1f}%'); c3.metric('PC3',f'{evr[2]*100:.1f}%'); c4.metric('PC1–PC3',f'{cum[2]*100:.1f}%')
    left,right=st.columns([1.1,1])
    with left:
        plot=filtered[['player_id','team','role','season_phase']].copy(); plot['PC1']=pcdf.PC1.values; plot['PC2']=pcdf.PC2.values; color=st.selectbox('Colour by',['team','role','season_phase']); fig=px.scatter(plot,x='PC1',y='PC2',color=color,hover_data=['player_id'],title='PCA score space'); theme(fig,470); st.plotly_chart(fig,use_container_width=True)
    with right:
        load=pd.DataFrame(pca.components_[:3].T,index=[DISPLAY[c] for c in MOTOR],columns=['PC1','PC2','PC3']); fig=px.imshow(load,text_auto='.2f',aspect='auto',color_continuous_scale='RdBu',origin='lower',title='PCA loadings'); theme(fig,470); st.plotly_chart(fig,use_container_width=True)
    fig=go.Figure(); fig.add_trace(go.Bar(x=[f'PC{i}' for i in range(1,7)],y=evr*100,name='Individual')); fig.add_trace(go.Scatter(x=[f'PC{i}' for i in range(1,7)],y=cum*100,name='Cumulative',mode='lines+markers',yaxis='y2')); fig.update_layout(title='Explained variance',yaxis=dict(title='Individual %'),yaxis2=dict(title='Cumulative %',overlaying='y',side='right',range=[0,105])); theme(fig,390); st.plotly_chart(fig,use_container_width=True)
    st.info('Thesis benchmark: PC1–PC3 explained 83.80% of variance. The chart above is recomputed from the current synthetic/filtered demo data.')

elif page=='Clustering lab':
    st.subheader('Clustering lab'); mode=st.radio('Model',['K-means','Gaussian Mixture Model'],horizontal=True)
    scaler,pca,pcdf=pca_bundle(filtered); X=pcdf[['PC1','PC2','PC3']].to_numpy(); k=st.slider('Number of groups',2,5,2)
    if mode=='K-means':
        km=KMeans(n_clusters=k,n_init=20,random_state=42); labels=km.fit_predict(X); sil=silhouette_score(X,labels)
        plot=filtered[['player_id','team','role']].copy(); plot['PC1']=pcdf.PC1.values; plot['PC2']=pcdf.PC2.values; plot['Cluster']=[f'C{x+1}' for x in labels]; st.metric('Silhouette score',f'{sil:.3f}'); fig=px.scatter(plot,x='PC1',y='PC2',color='Cluster',hover_data=['player_id','team','role'],title=f'K-means · k={k}'); theme(fig,470); st.plotly_chart(fig,use_container_width=True); st.caption('Thesis benchmark: silhouette = 0.335, interpreted as moderate separation.')
    else:
        gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=42,n_init=5); labels=gmm.fit_predict(X); probs=gmm.predict_proba(X)
        plot=filtered[['player_id','team','role']].copy(); plot['PC1']=pcdf.PC1.values; plot['PC2']=pcdf.PC2.values; plot['Profile']=[f'P{x+1}' for x in labels]; plot['Confidence']=probs.max(axis=1)
        a,b,c=st.columns(3); a.metric('AIC',f'{gmm.aic(X):,.0f}'); b.metric('BIC',f'{gmm.bic(X):,.0f}'); c.metric('Mean membership confidence',f'{plot.Confidence.mean():.3f}'); fig=px.scatter(plot,x='PC1',y='PC2',color='Profile',size='Confidence',hover_data=['player_id','team','role'],title=f'GMM · {k} profiles'); theme(fig,470); st.plotly_chart(fig,use_container_width=True); st.markdown('### Thesis model-selection table'); st.dataframe(THESIS_GMM,use_container_width=True,hide_index=True); st.caption('The table contains aggregate thesis results, not synthetic-demo outputs.')

elif page=='Dynamic trajectory':
    st.subheader('Dynamic trajectory'); player=st.selectbox('Select fictional player',sorted(filtered.player_id.unique())); p=filtered[filtered.player_id==player].sort_values('match_no')
    c1,c2,c3=st.columns(3); c1.metric('Stability',f'{p.stability.iloc[0]:.3f}'); c2.metric('Mean hybridity',f'{p.mean_hybridity.iloc[0]:.3f}'); c3.metric('Dominant probability',f'{p.dominant_prob.iloc[0]:.3f}')
    fig=go.Figure(); fig.add_trace(go.Scatter(x=p.match_no,y=p.gmm_lower_prob,mode='lines+markers',stackgroup='one',name='Lower profile')); fig.add_trace(go.Scatter(x=p.match_no,y=p.gmm_higher_prob,mode='lines+markers',stackgroup='one',name='Higher profile')); fig.update_layout(title='Profile composition over time',xaxis_title='Appearance',yaxis_title='Membership probability',yaxis_range=[0,1]); theme(fig,430); st.plotly_chart(fig,use_container_width=True)
    metric=st.selectbox('Trajectory metric',MOTOR,format_func=lambda x:DISPLAY[x]); fig2=px.line(p,x='match_no',y=metric,markers=True,color='season_phase',title=f'{DISPLAY[metric]} across appearances'); theme(fig2,390); st.plotly_chart(fig2,use_container_width=True)
    st.markdown('<div class="insight"><b>Interpretation rule</b><br>High stability does not mean a “better” player. It means the profile is more repeatable between analysed appearances. High hybridity means the observation mixes the two GMM profiles more evenly.</div>',unsafe_allow_html=True)

elif page=='HMM & transitions':
    st.subheader('HMM & transitions'); st.caption('Original thesis transition matrix vs empirical synthetic-demo sequence.')
    left,right=st.columns(2)
    with left:
        mat=pd.DataFrame(THESIS_TRANSITION,index=STATE_ORDER,columns=STATE_ORDER); fig=px.imshow(mat,text_auto='.3f',aspect='auto',color_continuous_scale='Greens',title='Thesis transition matrix'); theme(fig,430); st.plotly_chart(fig,use_container_width=True)
    with right:
        emp=pd.DataFrame(empirical_transition(filtered),index=STATE_ORDER,columns=STATE_ORDER); fig=px.imshow(emp,text_auto='.3f',aspect='auto',color_continuous_scale='Blues',title='Synthetic demo · empirical transitions'); theme(fig,430); st.plotly_chart(fig,use_container_width=True)
    src=[];tgt=[];val=[]
    for i in range(3):
        for j in range(3): src.append(i);tgt.append(j);val.append(float(THESIS_TRANSITION[i,j]))
    fig=go.Figure(go.Sankey(node=dict(label=STATE_ORDER,pad=20,thickness=22),link=dict(source=src,target=tgt,value=val))); fig.update_layout(title='Thesis hidden-state transition flow'); theme(fig,430); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="insight"><b>88.7% baseline persistence</b><br>In the thesis model, if an observation was in the baseline state, the next analysed appearance remained baseline with an estimated probability of 88.7%. This is a conditional transition probability, not the percentage of all matches classified as baseline.</div>',unsafe_allow_html=True)

elif page=='Survival analysis':
    st.subheader('Kaplan–Meier · time to first transition'); st.caption('Event: first direct transition from Lower activity → Higher activity.')
    times,obs,xs,ys=survival(filtered); c1,c2,c3=st.columns(3); c1.metric('Synthetic-demo players',len(obs)); c2.metric('Observed events',sum(obs)); c3.metric('Right-censored',len(obs)-sum(obs))
    fig=go.Figure(go.Scatter(x=xs,y=ys,mode='lines',line_shape='hv',name='Synthetic demo KM')); fig.add_vline(x=8,line_dash='dash',annotation_text='Thesis median = 8'); fig.update_layout(title='Kaplan–Meier survival curve',xaxis_title='Appearance number',yaxis_title='Probability of not yet observing the event',yaxis_range=[0,1.03]); theme(fig,470); st.plotly_chart(fig,use_container_width=True)
    st.info('Original thesis: n=31; 24 events; 7 right-censored; Kaplan–Meier median = 8 appearances; bootstrap 95% CI = 4–16.')

elif page=='Methodology':
    st.subheader('Methodology & interpretation')
    items=[('PCA','Reduces correlated motor variables to a smaller number of components. Thesis PC1–PC3 explained 83.80% of variance.'),('K-means','Hard clustering: each player-match observation is assigned to one cluster. Silhouette evaluates separation; it does not create the assignment.'),('ARI','Adjusted Rand Index compares partitions and is used as a repeatability/stability measure. ARI = 1 means identical partitions, not “perfect truth”.'),('GMM','Soft clustering: each observation receives membership probabilities that sum to 1.'),('Stability','Match-to-match repeatability of the soft profile. It is not a measure of player quality.'),('Hybridity','How evenly a single appearance mixes the GMM profiles. 50/50 is more hybrid than 95/5.'),('HMM','Models sequences of consecutive appearances and estimates hidden activity states plus transition probabilities.'),('Kaplan–Meier','Estimates appearances until the first lower→higher transition while retaining right-censored players.')]
    for title,body in items:
        with st.expander(title): st.write(body)
    st.markdown('### Thesis scope'); st.write('Dynamic Player DNA is an analytical pipeline combining existing methods. It is not presented as a new machine-learning algorithm and is not intended to automatically rank talent.')
    st.markdown('### Data privacy'); st.write('The interactive dataset bundled with this public repository is fully synthetic. It does not contain anonymised copies of real player-match observations.')

st.divider(); st.caption('Dynamic Player DNA · advanced public portfolio demo · synthetic player-level data')
