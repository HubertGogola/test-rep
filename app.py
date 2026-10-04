
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Dynamic Player DNA", page_icon="⚽", layout="wide")
BASE=Path(__file__).resolve().parent
df=pd.read_csv(BASE/"data"/"demo_player_data.csv")

st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#07130f 0%,#0b1c16 55%,#08130f 100%);color:#F4F7F5}
.block-container{padding-top:1.4rem;max-width:1450px}
.hero{border:1px solid rgba(126,231,135,.2);border-radius:24px;padding:28px 30px;background:linear-gradient(135deg,rgba(255,255,255,.05),rgba(255,255,255,.015));margin-bottom:18px}
.eyebrow{color:#7EE787;font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;font-weight:800}
.hero-title{font-size:3.2rem;font-weight:900;line-height:1;margin:.25rem 0 .5rem}
.hero-sub{color:#C7D2CC;font-size:1.05rem;max-width:900px;line-height:1.55}
.chip{display:inline-block;padding:6px 10px;margin:8px 6px 0 0;border-radius:999px;border:1px solid rgba(126,231,135,.25);color:#CFF8D3;background:rgba(126,231,135,.07);font-size:.78rem;font-weight:700}
.note{border-left:3px solid #7EE787;padding:10px 14px;background:rgba(126,231,135,.06);border-radius:8px;color:#D8E7DE;margin:8px 0 18px}
[data-testid="stMetric"]{background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.08);padding:14px 16px;border-radius:16px}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="eyebrow">Football analytics · Machine learning · Bachelor thesis</div>
<div class="hero-title">Dynamic Player DNA</div>
<div class="hero-sub">Public portfolio demo inspired by a bachelor thesis on dynamic football player profiling.
The thesis analysis used academy data from <b>Cracovia</b>; this application contains <b>synthetic demo data only</b>.</div>
<div><span class="chip">PCA</span><span class="chip">K-means</span><span class="chip">GMM</span><span class="chip">HMM</span><span class="chip">Kaplan–Meier</span></div>
</div>
<div class="note"><b>Important:</b> no raw Cracovia or club-confidential data is published here.</div>
""",unsafe_allow_html=True)

st.sidebar.title("Player explorer")
player=st.sidebar.selectbox("Anonymous player",sorted(df.player_id.unique()))
st.sidebar.caption("Synthetic player IDs are used in the public demo.")
st.sidebar.divider()
st.sidebar.markdown("**Thesis context**")
st.sidebar.write("Academy football · season 2025/2026")
st.sidebar.write("799 motor observations / 37 players")
st.sidebar.write("260 detailed technical-tactical observations / 33 players")

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Motor observations","799")
m2.metric("Players","37")
m3.metric("PCA variance","83.80%")
m4.metric("GMM profiles","2")
m5.metric("HMM states","3")

tabs=st.tabs(["Overview","Player profile","Match trajectory","HMM states","Methodology"])

with tabs[0]:
    st.subheader("From static profiling to dynamic understanding")
    st.write("The core question is not only **what profile a player resembles**, but also **how that profile changes from match to match**.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("PCA components","3")
    c2.metric("K-means silhouette","0.335")
    c3.metric("GMM stability (ARI)","1.000")
    c4.metric("Median first transition","8 matches")

with tabs[1]:
    p=df[df.player_id==player].sort_values("match_no")
    st.subheader(f"Player profile · {player}")
    c1,c2,c3=st.columns(3)
    c1.metric("Stability",f"{p.player_stability.iloc[0]:.3f}")
    c2.metric("Mean hybridity",f"{p.hybridity.mean():.3f}")
    c3.metric("Mean PII",f"{p.pii.mean():.1f}")
    metrics={"Total distance":"total_distance","HSR":"hsr","Sprint":"sprint","Accelerations":"accelerations","Decelerations":"decelerations","PII":"pii"}
    scores=[]
    for label,col in metrics.items():
        value=p[col].mean()
        scores.append((df[col]<=value).mean()*100)
    fig=go.Figure(go.Scatterpolar(r=scores+[scores[0]],theta=list(metrics.keys())+[list(metrics.keys())[0]],fill="toself"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),showlegend=False,paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#E8F0EB"),height=480,title="Relative motor profile (demo percentiles)")
    st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    p=df[df.player_id==player].sort_values("match_no")
    st.subheader(f"Soft profile membership across matches · {player}")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=p.match_no,y=p.gmm_lower_prob,mode="lines+markers",name="Lower activity profile"))
    fig.add_trace(go.Scatter(x=p.match_no,y=p.gmm_higher_prob,mode="lines+markers",name="Higher activity profile"))
    fig.update_layout(xaxis_title="Match",yaxis_title="GMM membership probability",yaxis_range=[0,1],paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.02)",font=dict(color="#E8F0EB"),hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    fig2=px.line(p,x="match_no",y="pii",markers=True,title="PII trajectory (synthetic demo)")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.02)",font=dict(color="#E8F0EB"))
    st.plotly_chart(fig2,use_container_width=True)

with tabs[3]:
    p=df[df.player_id==player].sort_values("match_no").copy()
    order=["Lower activity","Baseline","Higher activity"]
    code={s:i for i,s in enumerate(order)}
    p["state_code"]=p.hmm_state.map(code)
    st.subheader(f"Hidden activity states · {player}")
    fig=go.Figure(go.Scatter(x=p.match_no,y=p.state_code,mode="lines+markers",text=p.hmm_state,hovertemplate="Match %{x}<br>%{text}<extra></extra>"))
    fig.update_layout(yaxis=dict(tickmode="array",tickvals=[0,1,2],ticktext=order,range=[-.3,2.3],title="HMM state"),xaxis_title="Match",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.02)",font=dict(color="#E8F0EB"))
    st.plotly_chart(fig,use_container_width=True)
    mat=pd.DataFrame([[.740,.113,.147],[.066,.887,.047],[.189,.089,.722]],index=order,columns=order)
    heat=px.imshow(mat,text_auto=".3f",aspect="auto",labels=dict(x="Next match state",y="Current match state",color="Probability"))
    heat.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#E8F0EB"))
    st.plotly_chart(heat,use_container_width=True)
    st.caption("0.887 means that a baseline state was followed by another baseline state with an estimated probability of 88.7%.")

with tabs[4]:
    st.subheader("Methodology in plain language")
    items=[
        ("PCA","Reduce dimensionality when motor variables are correlated; PC1–PC3 explained 83.80% of variance."),
        ("K-means","Baseline hard clustering: each player-match observation belongs to one cluster."),
        ("GMM","Soft profile membership, allowing intermediate appearances."),
        ("Stability & hybridity","Stability = match-to-match repeatability; hybridity = how mixed the two profile probabilities are."),
        ("HMM","Hidden motor-activity states and transition probabilities across consecutive appearances."),
        ("Kaplan–Meier","Appearances until the first lower→higher transition, while retaining right-censored cases.")
    ]
    for title,desc in items:
        st.markdown(f"**{title}** — {desc}")
    st.info("The original thesis used real academy data from Cracovia. This public repository deliberately uses synthetic data only.")

st.divider()
st.caption("Dynamic Player DNA · public portfolio demo · synthetic data only")
