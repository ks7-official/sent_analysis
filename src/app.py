import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Trader Sentiment Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and modern styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #262640 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #a0a0b0;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    
    .metric-change {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    .positive { color: #00d4aa; }
    .negative { color: #ff6b6b; }
    .neutral { color: #ffd93d; }
    .fear { color: #ff4757; }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .tab-container {
        background: rgba(30, 30, 46, 0.5);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #262640 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/merged_data.csv")
        return df
    except FileNotFoundError:
        # Generate sample data for demonstration
        np.random.seed(42)
        n_records = 1000
        accounts = [f"ACC_{i:04d}" for i in range(1, 101)]
        
        data = {
            'Account': np.random.choice(accounts, n_records),
            'Closed PnL': np.random.normal(50, 200, n_records),
            'classification': np.random.choice(['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'], 
                                            n_records, p=[0.1, 0.3, 0.2, 0.3, 0.1]),
            'fear_greed_index': np.random.randint(0, 100, n_records),
            'timestamp': pd.date_range(start='2024-01-01', periods=n_records, freq='H'),
            'leverage': np.random.uniform(1, 10, n_records),
            'trade_side': np.random.choice(['Long', 'Short'], n_records),
            'trade_count': np.random.randint(1, 10, n_records)
        }
        return pd.DataFrame(data)

df = load_data()

# Calculate metrics
def calculate_metrics(df):
    active_traders = df['Account'].nunique()
    avg_pnl = df['Closed PnL'].mean()
    win_rate = (df['Closed PnL'] > 0).mean() * 100
    total_pnl = df['Closed PnL'].sum()
    
    market_sentiment = df['classification'].mode()[0] if not df.empty else "Neutral"
    
    trader_change = 12.3  # Simulated
    pnl_change = 5.8     # Simulated
    
    return {
        'active_traders': active_traders,
        'trader_change': trader_change,
        'avg_pnl': avg_pnl,
        'pnl_change': pnl_change,
        'win_rate': win_rate,
        'market_sentiment': market_sentiment,
        'total_pnl': total_pnl
    }

metrics = calculate_metrics(df)

# Header
st.markdown("""
<div class="main-header">
    <h1>Trader Sentiment Dashboard</h1>
    <p>Real-time analysis of trader behavior and market sentiment</p>
</div>
""", unsafe_allow_html=True)

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📈 Active Traders</div>
        <div class="metric-value positive">{metrics['active_traders']:,}</div>
        <div class="metric-change positive">+{metrics['trader_change']:.1f}% from last period</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    pnl_color = "positive" if metrics['avg_pnl'] >= 0 else "negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💰 Avg PnL</div>
        <div class="metric-value {pnl_color}">${metrics['avg_pnl']:,.0f}</div>
        <div class="metric-change">Fear vs Greed analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    sentiment_color = {
        'Extreme Fear': 'fear',
        'Fear': 'negative', 
        'Neutral': 'neutral',
        'Greed': 'positive',
        'Extreme Greed': 'fear'
    }.get(metrics['market_sentiment'], 'neutral')
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📊 Market Sentiment</div>
        <div class="metric-value {sentiment_color}">{metrics['market_sentiment']}</div>
        <div class="metric-change">Dominant sentiment: {metrics['market_sentiment']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🎯 Win Rate</div>
        <div class="metric-value positive">{metrics['win_rate']:.1f}%</div>
        <div class="metric-change">Profitable trades</div>
    </div>
    """, unsafe_allow_html=True)

# Navigation tabs
st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Sentiment Analysis", "📊 Trader Performance", "🔍 Market Insights"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # PnL Distribution by Sentiment (Boxplot style)
        fig_box = go.Figure()
        
        sentiments = df['classification'].unique()
        colors = ['#ff4757', '#ff6b6b', '#ffd93d', '#00d4aa', '#ff4757']
        
        for i, sentiment in enumerate(sentiments):
            sentiment_data = df[df['classification'] == sentiment]['Closed PnL']
            fig_box.add_trace(go.Box(
                y=sentiment_data,
                name=sentiment,
                marker_color=colors[i % len(colors)],
                boxpoints='outliers'
            ))
        
        fig_box.update_layout(
            title="PnL Distribution by Sentiment",
            xaxis_title="Market Sentiment",
            yaxis_title="PnL ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # PnL Density Analysis (Violin plot)
        fig_violin = go.Figure()
        
        for i, sentiment in enumerate(sentiments):
            sentiment_data = df[df['classification'] == sentiment]['Closed PnL']
            fig_violin.add_trace(go.Violin(
                y=sentiment_data,
                name=sentiment,
                fillcolor=colors[i % len(colors)],
                opacity=0.7,
                line_color='white'
            ))
        
        fig_violin.update_layout(
            title="PnL Density Analysis",
            xaxis_title="Market Sentiment",
            yaxis_title="PnL ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig_violin, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Performers
        top_traders = df.groupby('Account')['Closed PnL'].sum().sort_values(ascending=False).head(10)
        
        fig_top = go.Figure(data=[
            go.Bar(
                x=top_traders.values,
                y=top_traders.index,
                orientation='h',
                marker_color='#00d4aa',
                text=[f'${x:,.0f}' for x in top_traders.values],
                textposition='auto'
            )
        ])
        
        fig_top.update_layout(
            title="Top 10 Performing Traders",
            xaxis_title="Total PnL ($)",
            yaxis_title="Account",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # PnL Distribution Histogram
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=df['Closed PnL'],
                nbinsx=30,
                marker_color='#667eea',
                opacity=0.8
            )
        ])
        
        fig_hist.update_layout(
            title="PnL Distribution",
            xaxis_title="PnL ($)",
            yaxis_title="Frequency",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    # Fear & Greed Index Over Time
    if 'timestamp' in df.columns and 'fear_greed_index' in df.columns:
        df_time = df.sort_values('timestamp')
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=df_time['timestamp'],
            y=df_time['fear_greed_index'],
            mode='lines+markers',
            name='Fear & Greed Index',
            line=dict(color='#ffd93d', width=2),
            marker=dict(size=4)
        ))
        
        fig_time.add_hline(y=20, line_dash="dash", line_color="red", 
                          annotation_text="Extreme Fear")
        fig_time.add_hline(y=80, line_dash="dash", line_color="green", 
                          annotation_text="Extreme Greed")
        
        fig_time.update_layout(
            title="Fear & Greed Index Timeline",
            xaxis_title="Time",
            yaxis_title="Index Value",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Sentiment Distribution Pie Chart
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_counts = df['classification'].value_counts()
        
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker_colors=['#ff4757', '#ff6b6b', '#ffd93d', '#00d4aa', '#ff4757']
            )
        ])
        
        fig_pie.update_layout(
            title="Market Sentiment Distribution",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Correlation Analysis
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            corr_data = df.select_dtypes(include=[np.number]).corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig_corr.update_layout(
                title="Correlation Matrix",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)

# Detailed Analysis Plots Section
st.markdown('<div class="section-header">📈 Detailed Analysis Plots</div>', unsafe_allow_html=True)

output_dir = "output"
if os.path.exists(output_dir):
    plot_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if plot_files:
        # Display plots in a grid
        cols_per_row = 3
        for i in range(0, len(plot_files), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(plot_files):
                    plot_path = os.path.join(output_dir, plot_files[i + j])
                    try:
                        image = Image.open(plot_path)
                        with col:
                            st.image(image, caption=plot_files[i + j], use_container_width=True)
                    except Exception as e:
                        col.warning(f"Failed to load {plot_files[i + j]}: {str(e)}")
    else:
        st.info("No plots found in the 'output' directory. Please ensure plots are generated and saved in the 'output' directory.")
else:
    st.info("The 'output' directory does not exist. Please create it and add the required plot files.")

# Data Table Section
st.markdown('<div class="section-header">📋 Raw Data</div>', unsafe_allow_html=True)

# Add filters
col1, col2, col3 = st.columns(3)
with col1:
    sentiment_filter = st.selectbox("Filter by Sentiment", 
                                   ["All"] + list(df['classification'].unique()))
with col2:
    min_pnl = st.number_input("Minimum PnL", value=float(df['Closed PnL'].min()))
with col3:
    max_pnl = st.number_input("Maximum PnL", value=float(df['Closed PnL'].max()))

# Apply filters
filtered_df = df.copy()
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df['classification'] == sentiment_filter]
filtered_df = filtered_df[
    (filtered_df['Closed PnL'] >= min_pnl) & 
    (filtered_df['Closed PnL'] <= max_pnl)
]

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Bitcoin Trader Sentiment Dashboard | Last updated: {}</p>
    <p>Powered by Streamlit & Plotly</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
