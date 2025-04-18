# NAS100_Trading_Assistant.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================
# CONFIGURATION
# ======================
st.set_page_config(
    page_title="NAS100 AI Trading Assistant", 
    layout="wide",
    page_icon="📈"
)

# ======================
# CORE FUNCTIONS
# ======================
def detect_zones(df, window=50, min_touches=3, tolerance=0.002):
    """Improved support/resistance detection"""
    highs = df['High']
    lows = df['Low']
    
    max_idx = argrelextrema(highs.values, np.greater_equal, order=window)[0]
    min_idx = argrelextrema(lows.values, np.less_equal, order=window)[0]
    
    def cluster_levels(points):
        clusters = []
        for p in points:
            matched = False
            for c in clusters:
                if abs(p - c['mean'])/c['mean'] < tolerance:
                    c['points'].append(p)
                    c['mean'] = np.mean(c['points'])
                    matched = True
                    break
            if not matched:
                clusters.append({'points': [p], 'mean': p})
        return [c['mean'] for c in clusters if len(c['points']) >= min_touches]
    
    resistance = cluster_levels(highs.iloc[max_idx].tolist())
    support = cluster_levels(lows.iloc[min_idx].tolist())
    
    return sorted(support), sorted(resistance)

def generate_signals(df, zones, use_volume=True):
    """Signal generation with volume confirmation"""
    support, resistance = zones
    signals = []
    min_diff = df['Close'].mean() * 0.005
    
    support = [s for s in support if not any(abs(s-x) < min_diff for x in support if x != s)]
    resistance = [r for r in resistance if not any(abs(r-x) < min_diff for x in resistance if x != r)]
    
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Support bounce (BUY)
        for s in support:
            if (prev['Low'] <= s * 1.002) and (current['Close'] > s * 1.002):
                if not use_volume or current['Volume'] > df['Volume'].rolling(20).mean().iloc[i]:
                    signals.append({
                        "Datetime": df.index[i],
                        "Signal": "Buy",
                        "Price": current['Close'],
                        "Type": "Support Bounce"
                    })
                    break
        
        # Resistance rejection (SELL)
        for r in resistance:
            if (prev['High'] >= r * 0.998) and (current['Close'] < r * 0.998):
                if not use_volume or current['Volume'] > df['Volume'].rolling(20).mean().iloc[i]:
                    signals.append({
                        "Datetime": df.index[i],
                        "Signal": "Sell",
                        "Price": current['Close'],
                        "Type": "Resistance Reject"
                    })
                    break
    
    return pd.DataFrame(signals).drop_duplicates(subset=['Datetime', 'Signal'])

def backtest(df, signals, sl_pct=1.5, tp_pct=3.0):
    """Enhanced backtesting engine"""
    if signals.empty:
        return {
            "equity": [10000],
            "stats": {
                "final_equity": 10000,
                "total_trades": 0,
                "win_rate": "0.0%",
                "max_drawdown": "0.0%"
            }
        }
    
    equity = 10000
    results = [equity]
    wins = 0
    peak = equity
    max_dd = 0
    
    for dt, row in signals.iterrows():
        try:
            idx = df.index.get_loc(dt)
            entry = row['Price']
            is_buy = row['Signal'] == 'Buy'
            
            sl = entry * (1 - sl_pct/100) if is_buy else entry * (1 + sl_pct/100)
            tp = entry * (1 + tp_pct/100) if is_buy else entry * (1 - tp_pct/100)
            
            for i in range(idx, min(idx+100, len(df))):
                current_low = df['Low'].iloc[i]
                current_high = df['High'].iloc[i]
                
                if is_buy:
                    if current_low <= sl:
                        pnl = -sl_pct
                        break
                    elif current_high >= tp:
                        pnl = tp_pct
                        wins += 1
                        break
                else:
                    if current_high >= sl:
                        pnl = -sl_pct
                        break
                    elif current_low <= tp:
                        pnl = tp_pct
                        wins += 1
                        break
            
            equity += (equity * (pnl/100))
            results.append(equity)
            
            if equity > peak:
                peak = equity
            current_dd = (peak - equity)/peak
            if current_dd > max_dd:
                max_dd = current_dd
                
        except:
            continue
    
    return {
        "equity": results,
        "stats": {
            "final_equity": round(equity, 2),
            "total_trades": len(signals),
            "win_rate": f"{wins/max(1,len(signals)):.1%}",
            "max_drawdown": f"{max_dd*100:.1f}%"
        }
    }

# ======================
# STREAMLIT UI
# ======================
st.title("📊 NAS100 AI Trading Assistant")

# Custom CSS
st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 10px;
    background: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.profit { color: #00aa00; }
.loss { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📤 Upload NAS100 Data (CSV)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["Datetime"], index_col="Datetime")
        
        # Data Validation
        if not {'Open','High','Low','Close'}.issubset(df.columns):
            st.error("❌ Missing required price columns (Open, High, Low, Close)")
            st.stop()

        # Generate Levels and Signals
        with st.spinner("🔍 Detecting support/resistance levels..."):
            support, resistance = detect_zones(df)
        
        with st.spinner("📈 Generating trade signals..."):
            use_volume = st.checkbox("Use Volume Confirmation", value=True)
            signals = generate_signals(df, (support, resistance), use_volume=use_volume)

        if not signals.empty:
            # Visualization
            st.subheader("📊 Price Chart with Signals")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name="Price"
            ), row=1, col=1)
            
            # Signals
            buys = signals[signals['Signal'] == 'Buy']
            sells = signals[signals['Signal'] == 'Sell']
            fig.add_trace(go.Scatter(
                x=buys['Datetime'], y=buys['Price'],
                mode='markers', name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sells['Datetime'], y=sells['Price'],
                mode='markers', name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                name="Volume", marker_color='rgba(100, 150, 200, 0.6)'
            ), row=2, col=1)
            
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Backtest Configuration
            st.subheader("⚙️ Backtest Parameters")
            col1, col2 = st.columns(2)
            with col1:
                sl_pct = st.slider("Stop Loss %", 0.5, 5.0, 1.5, step=0.1)
            with col2:
                tp_pct = st.slider("Take Profit %", 0.5, 10.0, 3.0, step=0.1)
            
            # Run Backtest
            with st.spinner("🧮 Running backtest..."):
                result = backtest(df, signals.set_index('Datetime'), sl_pct, tp_pct)
            
            # Performance Dashboard
            st.subheader("📊 Performance Metrics")
            cols = st.columns(4)
            metrics = [
                ("💰 Final Equity", f"${result['stats']['final_equity']:,.2f}", 
                 "profit" if result['stats']['final_equity'] >= 10000 else "loss"),
                ("🎯 Win Rate", result['stats']['win_rate'], None),
                ("📈 Total Trades", result['stats']['total_trades'], None),
                ("⚠️ Max Drawdown", result['stats']['max_drawdown'], "loss")
            ]
            
            for col, (title, value, style) in zip(cols, metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{title}</h3>
                        <h2 class="{style if style else ''}">{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Equity Curve
            st.subheader("📈 Equity Curve")
            eq_df = pd.DataFrame({
                'Equity': result['equity'],
                'Drawdown': [100*(1 - x/max(result['equity'][:i+1])) for i, x in enumerate(result['equity'])]
            })
            
            eq_fig = make_subplots(specs=[[{"secondary_y": True}]])
            eq_fig.add_trace(go.Scatter(
                x=eq_df.index, y=eq_df['Equity'],
                name="Equity", line=dict(color='#4e79a7')
            ), secondary_y=False)
            eq_fig.add_trace(go.Bar(
                x=eq_df.index, y=eq_df['Drawdown'],
                name="Drawdown", marker=dict(color='#e15759', opacity=0.3)
            ), secondary_y=True)
            eq_fig.update_layout(
                height=400,
                yaxis_title="Equity ($)",
                yaxis2=dict(title="Drawdown %", range=[0, 100])
            )
            st.plotly_chart(eq_fig, use_container_width=True)
            
            # Signals Table
            st.subheader("🔍 Trade Signals")
            st.dataframe(
                signals.style.format({'Price': '{:.2f}'})
                .applymap(lambda x: 'color: green' if x == 'Buy' else 'color: red', 
                         subset=['Signal']),
                height=400
            )
            
            # Data Export
            st.download_button(
                "📥 Export Signals as CSV",
                signals.to_csv(index=False),
                "nas100_signals.csv"
            )
            
        else:
            st.warning("⚠️ No trade signals generated with current parameters")
            
    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")

# Sample Data Generator
with st.expander("💡 Need sample data?"):
    sample_data = pd.DataFrame({
        'Datetime': pd.date_range(start='2024-01-01', periods=100, freq='5T'),
        'Open': np.linspace(18000, 18200, 100),
        'High': np.linspace(18005, 18205, 100),
        'Low': np.linspace(17995, 18195, 100),
        'Close': np.linspace(18000, 18200, 100),
        'Volume': np.random.randint(1000, 5000, 100)
    })
    st.download_button(
        "⬇️ Download Sample Data",
        sample_data.to_csv(index=False),
        "nas100_sample.csv"
    )
