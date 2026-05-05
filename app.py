# ================================================================
# CLV & CHURN PROBABILITY — STREAMLIT DASHBOARD
# Save as: clv_app.py
# Run with: streamlit run clv_app.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pickle
from scipy.stats import beta as beta_dist
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title = "CLV & Churn Intelligence",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
  .big-metric{font-size:2rem;font-weight:600;color:#2C3E50}
  .sub-metric{font-size:0.9rem;color:#7F8C8D}
  .seg-champion{background:#9B59B620;border-left:4px solid #9B59B6;padding:0.7rem;border-radius:5px}
  .seg-loyal   {background:#3498DB20;border-left:4px solid #3498DB;padding:0.7rem;border-radius:5px}
  .seg-recent  {background:#2ECC7120;border-left:4px solid #2ECC71;padding:0.7rem;border-radius:5px}
  .seg-lost    {background:#E74C3C20;border-left:4px solid #E74C3C;padding:0.7rem;border-radius:5px}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
@st.cache_data
def load_all():
    try:
        rfm      = pd.read_csv('rfm_with_churn.csv')
        survival = pd.read_csv('survival_data.csv')
        clv      = pd.read_csv('clv_predictions.csv', index_col=0)
        return rfm, survival, clv
    except Exception as e:
        st.error(f"Data not found: {e}. Run the main notebook first.")
        return None, None, None

@st.cache_resource
def load_models():
    models = {}
    for name, path in [('bgf','bgf_model.pkl'),('ggf','ggf_model.pkl'),
                        ('cox','cox_model.pkl')]:
        try:
            with open(path,'rb') as f:
                models[name] = pickle.load(f)
        except:
            models[name] = None
    return models

rfm_df, survival_df, clv_df = load_all()
models = load_models()

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/analytics.png", width=80)
st.sidebar.title("CLV Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("Navigate", [
    "🏠 Overview",
    "🔍 Customer Lookup",
    "📈 Survival Curves",
    "💰 CLV Leaderboard",
    "🧠 Bayesian Updater",
])

# ----------------------------------------------------------------
# PAGE 1: OVERVIEW
# ----------------------------------------------------------------
if page == "🏠 Overview":
    st.title("📊 Customer Lifetime Value & Churn Intelligence")
    st.markdown("*Probabilistic modelling using BG/NBD, Gamma-Gamma, and Survival Analysis*")
    st.markdown("---")

    if rfm_df is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Customers", f"{len(rfm_df):,}")
        with col2:
            churn_rate = (rfm_df['bayesian_churn_prob'] > 0.5).mean()
            st.metric("Est. Churn Rate", f"{churn_rate:.1%}")
        with col3:
            if clv_df is not None:
                avg_clv = clv_df['clv_12m'].mean()
                st.metric("Avg 12M CLV", f"£{avg_clv:.0f}")
        with col4:
            champions = (rfm_df['segment'] == 'Champions').sum()
            st.metric("Champion Customers", f"{champions:,}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            seg_counts = rfm_df['segment'].value_counts()
            fig = px.pie(
                values=seg_counts.values,
                names=seg_counts.index,
                title='Customer Segment Distribution',
                color_discrete_map={
                    'Champions':'#9B59B6','Loyal':'#3498DB',
                    'Recent':'#2ECC71','Lost':'#E74C3C'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            seg_spend = rfm_df.groupby('segment')['monetary'].mean().reset_index()
            fig = px.bar(
                seg_spend, x='segment', y='monetary',
                title='Average Spend by Segment',
                color='segment',
                color_discrete_map={
                    'Champions':'#9B59B6','Loyal':'#3498DB',
                    'Recent':'#2ECC71','Lost':'#E74C3C'
                },
                labels={'monetary':'Avg Spend (£)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.scatter(
                rfm_df, x='recency', y='frequency',
                color='segment', size='monetary',
                title='RFM Scatter — Bubble size = Total Spend',
                color_discrete_map={
                    'Champions':'#9B59B6','Loyal':'#3498DB',
                    'Recent':'#2ECC71','Lost':'#E74C3C'
                },
                labels={
                    'recency':'Recency (days)',
                    'frequency':'Frequency (purchases)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.histogram(
                rfm_df, x='bayesian_churn_prob',
                color='segment', nbins=30,
                title='Bayesian Churn Probability Distribution',
                color_discrete_map={
                    'Champions':'#9B59B6','Loyal':'#3498DB',
                    'Recent':'#2ECC71','Lost':'#E74C3C'
                },
                labels={'bayesian_churn_prob':'Churn Probability'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Pareto insight
        rfm_sorted  = rfm_df.sort_values('monetary', ascending=False)
        top20_rev   = rfm_sorted.head(int(len(rfm_df)*0.2))['monetary'].sum()
        pareto_pct  = top20_rev / rfm_df['monetary'].sum() * 100
        st.info(f"💡 **Pareto Insight:** Top 20% of customers contribute **{pareto_pct:.0f}%** of total revenue. Prioritise retention for Champions and Loyal segments.")


# ----------------------------------------------------------------
# PAGE 2: CUSTOMER LOOKUP
# ----------------------------------------------------------------
elif page == "🔍 Customer Lookup":
    st.header("🔍 Customer Deep Dive")
    st.markdown("Look up any customer — get their full DS profile")
    st.markdown("---")

    if rfm_df is not None:
        customer_id = st.selectbox(
            "Select Customer ID",
            rfm_df['CustomerID'].astype(str).tolist()
        )

        cust = rfm_df[rfm_df['CustomerID'].astype(str) == str(customer_id)].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("RFM Profile")
            st.metric("Segment",   cust['segment'])
            st.metric("Recency",   f"{cust['recency']:.0f} days")
            st.metric("Frequency", f"{cust['frequency']:.0f} purchases")
            st.metric("Monetary",  f"£{cust['monetary']:.2f}")
            st.metric("RFM Score", f"{cust['rfm_score']}/15")

        with col2:
            st.subheader("CLV Prediction")
            if clv_df is not None and str(customer_id) in clv_df.index:
                cust_clv = clv_df.loc[str(customer_id)]
                st.metric("12-Month CLV",
                          f"£{cust_clv['clv_12m']:.2f}")
                st.metric("Prob Still Active",
                          f"{cust_clv['prob_alive']:.1%}")
                st.metric("Predicted Purchases (90d)",
                          f"{cust_clv['predicted_purchases_90d']:.1f}")
            else:
                st.info("CLV data not available for this customer")

        with col3:
            st.subheader("Churn Risk")
            churn_prob = cust.get('bayesian_churn_prob', 0.5)

            if churn_prob >= 0.6:
                risk = "🔴 HIGH RISK"
                color = "#E74C3C"
                action = "Immediate re-engagement campaign"
            elif churn_prob >= 0.4:
                risk = "🟡 MEDIUM RISK"
                color = "#F39C12"
                action = "Send personalised offer"
            else:
                risk = "🟢 LOW RISK"
                color = "#2ECC71"
                action = "Maintain engagement"

            st.markdown(f"""
            <div style='background:{color}20;border:2px solid {color};
                        border-radius:10px;padding:1rem;text-align:center'>
                <h3 style='color:{color};margin:0'>{risk}</h3>
                <h2 style='color:{color};margin:0.3rem 0'>{churn_prob:.0%}</h2>
                <p style='color:#555;margin:0'>Churn Probability</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Action:** {action}")

        # Bayesian probability gauge
        st.markdown("---")
        st.subheader("Bayesian Churn Probability — How It Updates")

        # Simulate history for this customer
        np.random.seed(hash(str(customer_id)) % 2**32)
        churn_base = churn_prob
        history = [1 if np.random.random() > churn_base else 0 for _ in range(12)]

        from scipy.stats import beta as beta_dist
        alpha, beta_p = 2, 2
        probs = []
        for p in history:
            alpha  += p
            beta_p += (1-p)
            probs.append(beta_p / (alpha + beta_p))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1,13)), y=probs,
            mode='lines+markers', name='Churn Probability',
            line=dict(color=color, width=2)
        ))
        fig.add_hline(y=0.5, line_dash='dash', line_color='black',
                      annotation_text='50% threshold')
        fig.update_layout(
            title='Bayesian Churn Probability Over 12 Months',
            xaxis_title='Month', yaxis_title='P(Churn)',
            yaxis=dict(range=[0,1])
        )
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------
# PAGE 3: SURVIVAL CURVES
# ----------------------------------------------------------------
elif page == "📈 Survival Curves":
    st.header("📈 Survival Analysis — Customer Lifetime")
    st.markdown("*Kaplan-Meier curves show when each segment drops off*")
    st.markdown("---")

    if survival_df is not None:
        from lifelines import KaplanMeierFitter

        col1, col2 = st.columns(2)

        with col1:
            # KM curves
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = {'Champions':'#9B59B6','Loyal':'#3498DB',
                      'Recent':'#2ECC71','Lost':'#E74C3C'}

            for seg in ['Champions','Loyal','Recent','Lost']:
                seg_data = survival_df[survival_df['segment'] == seg]
                if len(seg_data) < 5: continue
                kmf = KaplanMeierFitter()
                kmf.fit(seg_data['lifetime_days'], seg_data['churned'],
                        label=f"{seg} (n={len(seg_data)})")
                kmf.plot_survival_function(ax=ax, ci_show=False,
                                           color=colors[seg])

            ax.set_title('Kaplan-Meier Survival Curves by Segment')
            ax.set_xlabel('Days since First Purchase')
            ax.set_ylabel('P(Still Active)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            # Churn rate by segment
            churn_by_seg = survival_df.groupby('segment')['churned'].agg(
                ['mean','count']
            ).reset_index()
            churn_by_seg.columns = ['Segment','Churn Rate','Count']
            churn_by_seg['Churn Rate'] = (churn_by_seg['Churn Rate']*100).round(1)

            fig = px.bar(
                churn_by_seg, x='Segment', y='Churn Rate',
                title='Churn Rate by Segment (%)',
                color='Segment',
                color_discrete_map=colors,
                text='Churn Rate'
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        # Median survival times
        st.subheader("Median Survival Times by Segment")
        median_data = []
        for seg in ['Champions','Loyal','Recent','Lost']:
            seg_data = survival_df[survival_df['segment'] == seg]
            if len(seg_data) < 5: continue
            kmf = KaplanMeierFitter()
            kmf.fit(seg_data['lifetime_days'], seg_data['churned'])
            median_data.append({
                'Segment': seg,
                'Median Survival (days)': kmf.median_survival_time_,
                'Churn Rate': f"{seg_data['churned'].mean():.1%}",
                'Count': len(seg_data)
            })

        st.dataframe(pd.DataFrame(median_data),
                     use_container_width=True, hide_index=True)


# ----------------------------------------------------------------
# PAGE 4: CLV LEADERBOARD
# ----------------------------------------------------------------
elif page == "💰 CLV Leaderboard":
    st.header("💰 Customer Lifetime Value Leaderboard")
    st.markdown("*Top customers ranked by predicted 12-month CLV*")

    if clv_df is not None and rfm_df is not None:

        # -----------------------------
        # 🔧 SAFE DATA ALIGNMENT FIX
        # -----------------------------
        clv_temp = clv_df.copy()
        rfm_temp = rfm_df.copy()

        # Ensure CustomerID exists in both
        if 'CustomerID' not in clv_temp.columns:
            clv_temp = clv_temp.reset_index()
            clv_temp.rename(columns={'index': 'CustomerID'}, inplace=True)

        if 'CustomerID' not in rfm_temp.columns:
            rfm_temp = rfm_temp.reset_index()

        # Clean column names (VERY IMPORTANT)
        clv_temp.columns = clv_temp.columns.str.strip()
        rfm_temp.columns = rfm_temp.columns.str.strip()

        # -----------------------------
        # 🔗 SAFE MERGE
        # -----------------------------
        clv_seg = clv_temp.merge(
            rfm_temp[['CustomerID', 'segment']],
            on='CustomerID',
            how='left'
        )

        # -----------------------------
        # 📊 METRICS
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            total_clv = clv_seg['clv_12m'].sum()
            st.metric("Total Predicted CLV (12M)", f"£{total_clv:,.0f}")

        with col2:
            top10 = clv_seg.nlargest(int(len(clv_seg) * 0.1), 'clv_12m')
            top10_pct = (top10['clv_12m'].sum() / total_clv) * 100
            st.metric("Top 10% Contribution", f"{top10_pct:.0f}%")

        with col3:
            st.metric("Total Customers", len(clv_seg))

        st.markdown("---")

        # -----------------------------
        # 🏆 TOP CUSTOMERS TABLE
        # -----------------------------
        top20 = clv_seg.nlargest(20, 'clv_12m')[[
            'CustomerID',
            'clv_12m',
            'prob_alive',
            'predicted_purchases_90d',
            'segment'
        ]].copy()

        top20.columns = [
            'CustomerID',
            '12M CLV (£)',
            'P(Alive)',
            'Pred Purchases (90d)',
            'Segment'
        ]

        st.subheader("🏆 Top 20 Customers by CLV")
        st.dataframe(top20, use_container_width=True, hide_index=True)

        # -----------------------------
        # 📊 CLV BY SEGMENT
        # -----------------------------
        seg_summary = clv_seg.groupby('segment')['clv_12m'].agg(
            ['mean', 'sum', 'count']
        ).reset_index()

        seg_summary.columns = [
            'Segment',
            'Avg CLV',
            'Total CLV',
            'Count'
        ]

        st.subheader("📊 CLV by Segment")

        st.dataframe(seg_summary, use_container_width=True)

        # -----------------------------
        # 📈 VISUALIZATION
        # -----------------------------
        fig = px.bar(
            seg_summary,
            x='Segment',
            y='Total CLV',
            color='Segment',
            text='Total CLV',
            title='Total CLV by Customer Segment'
        )

        fig.update_traces(texttemplate='£%{text:,.0f}', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # 📉 CLV vs CHURN SCATTER
        # -----------------------------
        fig2 = px.scatter(
            clv_seg,
            x='prob_alive',
            y='clv_12m',
            color='segment',
            size='clv_12m',
            title='CLV vs Probability of Being Active',
            labels={
                'prob_alive': 'P(Alive)',
                'clv_12m': '12-Month CLV (£)'
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("CLV or RFM data not loaded properly.")


# ----------------------------------------------------------------
# PAGE 5: BAYESIAN UPDATER
# ----------------------------------------------------------------
elif page == "🧠 Bayesian Updater":
    st.header("🧠 Live Bayesian Churn Probability Updater")
    st.markdown("*Simulate how churn probability updates month by month as new data arrives*")
    st.markdown("---")

    st.info("""
    **How it works:** We start with a prior belief (50/50 uncertain) and update it every month
    based on whether the customer purchased or not. The more data we have, the more confident
    the model becomes. This is Beta-Binomial Bayesian inference.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Set Purchase History")
        st.markdown("Toggle each month: 1 = purchased, 0 = did not purchase")

        months = []
        for i in range(1, 13):
            val = st.checkbox(f"Month {i}", value=True if i <= 6 else False)
            months.append(1 if val else 0)

        alpha_prior = st.slider("Prior strength (higher = stronger prior)", 1, 10, 2)
        beta_prior  = st.slider("Prior churn belief (higher = more likely to churn prior)", 1, 10, 2)

    with col2:
        # Compute Bayesian updates
        alpha = alpha_prior
        beta_p = beta_prior
        probs  = []
        ci_lows = []
        ci_highs = []

        for purchased in months:
            alpha  += purchased
            beta_p += (1 - purchased)
            prob    = beta_p / (alpha + beta_p)
            ci_low  = beta_dist.ppf(0.025, alpha, beta_p)
            ci_high = beta_dist.ppf(0.975, alpha, beta_p)
            probs.append(prob)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)

        month_labels = [f"M{i}" for i in range(1,13)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=month_labels, y=ci_highs,
            fill=None, mode='lines',
            line=dict(color='rgba(52,152,219,0)'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=month_labels, y=ci_lows,
            fill='tonexty', mode='lines',
            line=dict(color='rgba(52,152,219,0)'),
            fillcolor='rgba(52,152,219,0.2)',
            name='95% Credible Interval'
        ))
        fig.add_trace(go.Scatter(
            x=month_labels, y=probs,
            mode='lines+markers',
            name='Churn Probability',
            line=dict(color='#3498DB', width=2),
            marker=dict(size=8, color=[
                '#E74C3C' if p > 0.5 else '#2ECC71' for p in probs
            ])
        ))
        fig.add_hline(y=0.5, line_dash='dash', line_color='black',
                      annotation_text='50% threshold')

        # Mark purchases
        for i, (m, p) in enumerate(zip(month_labels, months)):
            fig.add_vline(
                x=m,
                line_color='green' if p == 1 else 'red',
                line_width=1, opacity=0.3
            )

        fig.update_layout(
            title=f'Bayesian Churn Probability Update — Final: {probs[-1]:.1%}',
            xaxis_title='Month',
            yaxis_title='P(Churn)',
            yaxis=dict(range=[0,1]),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Final assessment
        final_prob = probs[-1]
        if final_prob >= 0.6:
            st.error(f"🔴 **HIGH CHURN RISK: {final_prob:.0%}** — Immediate re-engagement needed")
        elif final_prob >= 0.4:
            st.warning(f"🟡 **MEDIUM CHURN RISK: {final_prob:.0%}** — Send personalised offer")
        else:
            st.success(f"🟢 **LOW CHURN RISK: {final_prob:.0%}** — Customer is healthy")

        # Posterior distribution visualisation
        st.subheader("Posterior Beta Distribution — Current Belief")
        x = np.linspace(0, 1, 200)
        y = beta_dist.pdf(x, alpha, beta_p)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x, y=y, fill='tozeroy',
            line=dict(color='#9B59B6', width=2),
            fillcolor='rgba(155,89,182,0.3)',
            name='Posterior Beta distribution'
        ))
        fig2.add_vline(x=final_prob, line_dash='dash',
                       line_color='red',
                       annotation_text=f'Churn prob: {final_prob:.1%}')
        fig2.update_layout(
            title='Current Posterior Belief about Churn Probability',
            xaxis_title='Churn Probability',
            yaxis_title='Density',
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Beta({alpha:.0f}, {beta_p:.0f}) — narrow distribution = high confidence | wide = uncertain")
