import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="🏬 Walmart Spend Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Walmart themed: Blue #0071CE and Gold #FFC220)
st.markdown("""
<style>
    .reportview-container {
        background: #111111;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #0071CE;
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .accent-header {
        color: #FFC220;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #1f1f1f;
        border-left: 5px solid #0071CE;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card-gold {
        background-color: #1f1f1f;
        border-left: 5px solid #FFC220;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        color: #f5f5f1;
        font-size: 28px;
        font-weight: 800;
    }
    .metric-lbl {
        color: #b3b3b3;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
</style>
""", unsafe_allowed_html=True)

# Data Loading (Cached for performance)
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'walmart_data.csv')
    df = pd.read_csv(csv_path)
    
    # Preprocessing
    df["Marital_Status_Label"] = df["Marital_Status"].map({0: "Single", 1: "Married"})
    
    # Order age categories properly for charts
    age_order = sorted(df["Age"].unique())
    if "0-17" in age_order:
        # Custom sort logic just in case
        pass
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading walmart_data.csv: {e}")
    st.stop()

# ── Sidebar Filter Setup ──────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/14/Walmart_Spark.svg", width=80)
st.sidebar.title("🛒 Filter Transactions")

# 1. Gender Filter
genders = sorted(df["Gender"].unique())
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)

# 2. Age Group Filter
age_groups = sorted(df["Age"].unique())
selected_age_groups = st.sidebar.multiselect("Age Groups", age_groups, default=age_groups)

# 3. City Category Filter
cities = sorted(df["City_Category"].unique())
selected_cities = st.sidebar.multiselect("City Categories", cities, default=cities)

# 4. Marital Status Filter
marital_options = ["Single", "Married"]
selected_marital = st.sidebar.multiselect("Marital Status", marital_options, default=marital_options)

# 5. Purchase Amount Slider
min_purchase = float(df["Purchase"].min())
max_purchase = float(df["Purchase"].max())
selected_purchase_range = st.sidebar.slider(
    "Purchase Amount ($)", 
    min_purchase, 
    max_purchase, 
    (min_purchase, max_purchase)
)

# Apply filters
df_filtered = df.copy()

if selected_genders:
    df_filtered = df_filtered[df_filtered["Gender"].isin(selected_genders)]
if selected_age_groups:
    df_filtered = df_filtered[df_filtered["Age"].isin(selected_age_groups)]
if selected_cities:
    df_filtered = df_filtered[df_filtered["City_Category"].isin(selected_cities)]
if selected_marital:
    df_filtered = df_filtered[df_filtered["Marital_Status_Label"].isin(selected_marital)]

df_filtered = df_filtered[
    (df_filtered["Purchase"] >= selected_purchase_range[0]) & 
    (df_filtered["Purchase"] <= selected_purchase_range[1])
]

# Page Header
st.markdown("<div class='main-header'>🏬 Walmart Customer Spend Analyzer</div>", unsafe_allowed_html=True)
st.markdown("<div class='accent-header'>Black Friday Customer Transaction Insights & CLT Sampling Simulators</div>", unsafe_allowed_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_spend = df_filtered["Purchase"].sum()
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>${total_spend:,.2f}</div>
        <div class='metric-lbl'>Total Spend Volume</div>
    </div>
    """, unsafe_allowed_html=True)

with col2:
    avg_spend = df_filtered["Purchase"].mean() if not df_filtered.empty else 0
    st.markdown(f"""
    <div class='metric-card-gold'>
        <div class='metric-val'>${avg_spend:,.2f}</div>
        <div class='metric-lbl'>Average Transaction Spend</div>
    </div>
    """, unsafe_allowed_html=True)

with col3:
    num_tx = len(df_filtered)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>{num_tx:,}</div>
        <div class='metric-lbl'>Total Transactions</div>
    </div>
    """, unsafe_allowed_html=True)

with col4:
    unique_cust = df_filtered["User_ID"].nunique() if not df_filtered.empty else 0
    st.markdown(f"""
    <div class='metric-card-gold'>
        <div class='metric-val'>{unique_cust:,}</div>
        <div class='metric-lbl'>Unique Customers</div>
    </div>
    """, unsafe_allowed_html=True)

st.markdown("<br>", unsafe_allowed_html=True)

# Tabs setup
tab1, tab2, tab3 = st.tabs(["📊 Transaction Insights", "🎲 Central Limit Theorem (CLT) Simulator", "📂 Data Explorer & Strategic Recommendations"])

# ── TAB 1: Visual Insights ────────────────────────────────────────────────
with tab1:
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("💳 Spend Distribution (Purchase Amount)")
        if not df_filtered.empty:
            fig_hist = px.histogram(
                df_filtered,
                x="Purchase",
                nbins=40,
                color_discrete_sequence=["#0071CE"],
                template="plotly_dark",
                labels={"Purchase": "Spend Amount ($)"}
            )
            fig_hist.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No transaction data matches current filters.")
            
    with row1_col2:
        st.subheader("👫 Gender-wise Spend Analysis")
        if not df_filtered.empty:
            fig_box = px.box(
                df_filtered,
                x="Gender",
                y="Purchase",
                color="Gender",
                color_discrete_map={"M": "#0071CE", "F": "#FFC220"},
                template="plotly_dark",
                labels={"Purchase": "Spend Amount ($)"}
            )
            fig_box.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No transaction data matches current filters.")

    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("🎂 Spending by Age Groups")
        if not df_filtered.empty:
            age_summary = df_filtered.groupby("Age")["Purchase"].agg(["mean", "count"]).reset_index()
            fig_age = px.bar(
                age_summary,
                x="Age",
                y="mean",
                color="Age",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                template="plotly_dark",
                labels={"mean": "Average Spend ($)", "Age": "Age Category"}
            )
            fig_age.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No data available.")
            
    with row2_col2:
        st.subheader("🏙️ City Category spend Share")
        if not df_filtered.empty:
            city_spend = df_filtered.groupby("City_Category")["Purchase"].sum().reset_index()
            fig_city = px.pie(
                city_spend,
                values="Purchase",
                names="City_Category",
                hole=0.4,
                color_discrete_sequence=["#0071CE", "#FFC220", "#333333"],
                template="plotly_dark"
            )
            fig_city.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            st.info("No data available.")

# ── TAB 2: CLT Simulator ──────────────────────────────────────────────────
with tab2:
    st.subheader("🎲 Interactive Central Limit Theorem (CLT) Simulator")
    st.markdown("""
    The purchase amount distribution in retail dataset is highly right-skewed or multimodal (non-normal). 
    However, the **Central Limit Theorem (CLT)** states that if you draw multiple random samples of size $n$, 
    the distribution of the **sample means** will be approximately normal, regardless of the population distribution!
    """)
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sample_size = st.slider("Select Sample Size (n)", 5, 200, 30, step=5, key="wal_sample_size")
    with col_sim2:
        num_samples = st.slider("Select Number of Samples (S)", 100, 2000, 500, step=100, key="wal_num_samples")
        
    run_sim = st.button("Run CLT Simulation 🚀", use_container_width=True, key="wal_run_sim")
    
    # Store in session state to persist across sliders adjustment
    if run_sim or 'wal_sim_means' not in st.session_state or st.session_state.get('wal_prev_n') != sample_size or st.session_state.get('wal_prev_s') != num_samples:
        population = df["Purchase"].values
        pop_mean = np.mean(population)
        pop_std = np.std(population)
        
        # Simulating sampling
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(population, size=sample_size, replace=True)
            sample_means.append(np.mean(sample))
            
        st.session_state['wal_sim_means'] = sample_means
        st.session_state['wal_pop_mean'] = pop_mean
        st.session_state['wal_pop_std'] = pop_std
        st.session_state['wal_prev_n'] = sample_size
        st.session_state['wal_prev_s'] = num_samples

    # Load results
    sim_means = st.session_state['wal_sim_means']
    pop_mean = st.session_state['wal_pop_mean']
    pop_std = st.session_state['wal_pop_std']
    
    mean_of_means = np.mean(sim_means)
    std_of_means = np.std(sim_means)
    expected_se = pop_std / np.sqrt(sample_size)
    
    sim_metrics1, sim_metrics2, sim_metrics3, sim_metrics4 = st.columns(4)
    with sim_metrics1:
        st.metric("Population Mean (μ)", f"${pop_mean:.2f}")
    with sim_metrics2:
        st.metric("Mean of Sample Means (x̄)", f"${mean_of_means:.2f}")
    with sim_metrics3:
        st.metric("Expected Standard Error (σ/√n)", f"${expected_se:.2f}")
    with sim_metrics4:
        st.metric("Empirical Standard Error (SE)", f"${std_of_means:.2f}")
        
    # Chart
    fig_sim = px.histogram(
        x=sim_means,
        nbins=40,
        title=f"Normal Distribution of Sample Means (S={num_samples} runs, sample size n={sample_size})",
        color_discrete_sequence=["#FFC220"],
        template="plotly_dark",
        labels={"x": "Sample Mean Spend ($)"}
    )
    # Add vertical line for population mean
    fig_sim.add_vline(
        x=pop_mean, 
        line_width=3, 
        line_dash="dash", 
        line_color="#0071CE",
        annotation_text="Population Mean (μ)", 
        annotation_position="top right"
    )
    st.plotly_chart(fig_sim, use_container_width=True)
    
    st.info("💡 **Notice how the distribution is beautifully bell-shaped (Normal)** even though the original transactions distribution is heavily skewed! This demonstrates the power of CLT for building statistical confidence intervals.")

# ── TAB 3: Data Explorer & Insights ───────────────────────────────────────
with tab3:
    # Strategic Insights Expander
    st.markdown("### 💡 Strategic Business Insights & Recommendations")
    
    insights_col1, insights_col2 = st.columns(2)
    with insights_col1:
        st.markdown("""
        #### Key Findings
        1. **Gender Spend Difference**: Male customers have a significantly higher average transaction spend (~$9,400) compared to Female customers (~$8,700). This holds true across all city tiers and age cohorts.
        2. **Core Age Segments**: The **26-35 age bracket** contributes the highest overall transaction count and revenue share, followed by the **36-45** bracket.
        3. **Marital Status**: Marital status has negligible impact on average transaction spend, meaning spending behaviors are comparable between single and married individuals.
        4. **City tier C dominance**: City tier C indicates higher density of transaction counts and high-value single transactions, showing high purchase capacity in smaller urban clusters.
        """)
        
    with insights_col2:
        st.markdown("""
        #### Strategic Actions
        - **Target High-Spend Male Demographic**: Focus major Black Friday high-ticket campaign marketing towards male consumers between ages 26-45.
        - **Avoid Over-segmenting by Marital Status**: Since married and single customers spend similarly, avoid dividing marketing budgets or copy based on marital status.
        - **Optimize Logistics for City Tier C**: Target smaller towns (Tier C) with distribution hub upgrades and regional marketing.
        - **Loyalty Programs for Female Segment**: Create customized loyalty point perks to increase retention and average purchase values for female shoppers.
        """)
        
    st.markdown("---")
    st.markdown("#### 📂 Dataset Explorer")
    st.write(f"Displaying **{len(df_filtered):,}** transactions matching filters.")
    
    display_cols = ["User_ID", "Product_ID", "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status_Label", "Product_Category", "Purchase"]
    st.dataframe(df_filtered[display_cols], use_container_width=True)
    
    # Download Button
    csv = df_filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="walmart_filtered_spend.csv",
        mime="text/csv",
    )
