import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from utils.data_processor import DataProcessor
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.predictor import StockPredictor
from utils.visualizations import ChartGenerator

# Page configuration
st.set_page_config(page_title="Stock Market & Sentiment Analysis Dashboard",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()
if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPredictor()
if 'chart_generator' not in st.session_state:
    st.session_state.chart_generator = ChartGenerator()


def main():
    st.title("📈 Stock Market & Sentiment Analysis Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # Theme toggle
        theme = st.radio("Theme", ["Light", "Dark"],
                         help="Select your preferred theme")

        # Apply theme changes
        if theme == "Dark":
            st.markdown("""
            <style>
            .stApp {
                background-color: #0E1117 !important;
                color: #FAFAFA !important;
            }
            .stSidebar {
                background-color: #262730 !important;
            }
            .stSidebar .stMarkdown, .stSidebar .stText, .stSidebar label {
                color: #FAFAFA !important;
            }
            .stSelectbox > div > div, .stFileUploader > div, .stRadio > div {
                background-color: #262730 !important;
                color: #FAFAFA !important;
            }
            .stButton > button {
                background-color: #FF6B6B !important;
                color: white !important;
                border: none !important;
            }
            .stMetric {
                background-color: #262730 !important;
                padding: 10px !important;
                border-radius: 5px !important;
            }
            .stMetric label, .stMetric .metric-value {
                color: #FAFAFA !important;
            }
            .stMarkdown, .stText, .stSubheader, .stHeader, h1, h2, h3, p {
                color: #FAFAFA !important;
            }
            .stDataFrame, .stTable {
                background-color: #262730 !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #262730 !important;
            }
            .stTabs [data-baseweb="tab"] {
                color: #FAFAFA !important;
            }
            .stExpander {
                background-color: #262730 !important;
                border: 1px solid #444 !important;
            }
            .stExpander .streamlit-expanderHeader {
                color: #FAFAFA !important;
            }
            div[data-testid="stFileUploadDropzone"] {
                background-color: #262730 !important;
                color: #FAFAFA !important;
                border: 2px dashed #FF6B6B !important;
            }
            div[data-testid="stFileUploadDropzone"] p {
                color: #FAFAFA !important;
            }
            </style>
            """,
                        unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
            .stApp {
                background-color: #FFFFFF !important;
                color: #262730 !important;
            }
            .stSidebar {
                background-color: #F0F2F6 !important;
            }
            .stButton > button {
                background-color: #FF6B6B !important;
                color: white !important;
                border: none !important;
            }
            </style>
            """,
                        unsafe_allow_html=True)

        st.markdown("---")

        # File upload section
        st.header("📁 Data Upload")

        # Social Media Data Upload
        st.subheader("Social Media Dataset")
        social_media_file = st.file_uploader(
            "Upload social media data (CSV)",
            type=['csv'],
            key="social_media",
            help="Required columns: date, text")

        # Stock Market Data Upload
        st.subheader("Stock Market Dataset")
        stock_market_file = st.file_uploader(
            "Upload stock market data (Excel)",
            type=['xlsx', 'xls'],
            key="stock_market",
            help="Required columns: date, close")

        # Process data button
        process_data = st.button("🔄 Process Data", type="primary")

    # Main content area
    if social_media_file is not None and stock_market_file is not None:
        if process_data or 'processed_data' in st.session_state:
            # Load and validate data
            try:
                with st.spinner("Loading and validating data..."):
                    # Load datasets
                    social_df = pd.read_csv(social_media_file)
                    stock_df = pd.read_excel(stock_market_file)

                    # Validate datasets
                    validation_result = st.session_state.data_processor.validate_datasets(
                        social_df, stock_df)

                    if not validation_result['valid']:
                        st.error(
                            f"❌ Data validation failed: {validation_result['message']}"
                        )
                        return

                    # Process data
                    processed_data = st.session_state.data_processor.process_data(
                        social_df, stock_df)
                    st.session_state.processed_data = processed_data

                # Perform sentiment analysis
                with st.spinner("Analyzing sentiment..."):
                    sentiment_data = st.session_state.sentiment_analyzer.analyze_sentiment(
                        st.session_state.processed_data['social_media'])
                    st.session_state.sentiment_data = sentiment_data

                # Merge data for analysis
                merged_data = st.session_state.data_processor.merge_datasets(
                    sentiment_data,
                    st.session_state.processed_data['stock_market'])
                st.session_state.merged_data = merged_data

                st.success("✅ Data processed successfully!")

                # Display analysis tabs
                display_analysis_tabs()

            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                st.info("Please check your data format and try again.")

    else:
        # Welcome screen
        display_welcome_screen()


def display_welcome_screen():
    st.markdown("""
    ## Welcome to the Stock Market & Sentiment Analysis Dashboard! 🎯
    
    This comprehensive dashboard provides:
    
    ### 📊 **Diagnostic Analysis**
    - Sentiment distribution visualization
    - Correlation analysis between sentiment and stock prices
    - Market trend identification
    
    ### 🔮 **Predictive Modeling**
    - Linear regression for next-day price prediction
    - Model performance metrics (MSE)
    - Prediction accuracy visualization
    
    ### 💡 **Prescriptive Analysis**
    - Buy signal identification
    - Marketing insights based on sentiment
    - Trading strategy recommendations
    
    ### 📈 **Performance Analysis**
    - Strategy returns vs market returns
    - Cumulative performance tracking
    - Risk-return analysis
    
    ---
    
    ### 🚀 **Getting Started**
    1. Upload your **Social Media Dataset** (CSV with columns: date, text)
    2. Upload your **Stock Market Dataset** (CSV with columns: date, close)
    3. Click **Process Data** to begin analysis
    
    ### 📋 **Data Requirements**
    
    **Social Media Dataset:**
    - `date`: Date column (formats: DD-MM-YYYY, MM/DD/YYYY, YYYY-MM-DD - time ignored if present)
    - `text`: Text content for sentiment analysis
    
    **Stock Market Dataset:**
    - `date`: Date column (formats: DD-MM-YYYY, MM/DD/YYYY, YYYY-MM-DD - time ignored if present)
    - `close`: Closing price of the stock
    - Format: Excel file (.xlsx or .xls)
    """)


def display_analysis_tabs():
    if 'merged_data' not in st.session_state:
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Diagnostic Analysis", "🔮 Predictive Modeling",
        "💡 Prescriptive Analysis", "📊 Performance Analysis"
    ])

    with tab1:
        display_diagnostic_analysis()

    with tab2:
        display_predictive_modeling()

    with tab3:
        display_prescriptive_analysis()

    with tab4:
        display_performance_analysis()


def display_diagnostic_analysis():
    st.header("🔍 Diagnostic Analysis")

    merged_data = st.session_state.merged_data

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Sentiment Distribution")
        # Create sentiment distribution pie chart
        sentiment_counts = merged_data['sentiment_label'].value_counts()

        fig_pie = px.pie(values=sentiment_counts.values,
                         names=sentiment_counts.index,
                         title="Sentiment Distribution",
                         color_discrete_map={
                             'Positive': '#2E8B57',
                             'Negative': '#DC143C',
                             'Neutral': '#4682B4'
                         })
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Download button for pie chart
        buffer = io.StringIO()
        fig_pie.write_html(buffer, include_plotlyjs='cdn')
        st.download_button(label="📥 Download Sentiment Chart",
                           data=buffer.getvalue(),
                           file_name="sentiment_distribution.html",
                           mime="text/html")

    with col2:
        st.subheader("📈 Sentiment vs Stock Price Correlation")

        # Calculate correlation
        correlation = merged_data['sentiment_score'].corr(merged_data['close'])

        st.metric(
            label="Correlation Coefficient",
            value=f"{correlation:.4f}",
            help="Correlation between sentiment score and stock price (-1 to 1)"
        )

        # Scatter plot
        fig_scatter = px.scatter(merged_data,
                                 x='sentiment_score',
                                 y='close',
                                 title="Sentiment Score vs Stock Price",
                                 labels={
                                     'sentiment_score': 'Sentiment Score',
                                     'close': 'Stock Price ($)'
                                 },
                                 trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Download button for scatter plot
        buffer = io.StringIO()
        fig_scatter.write_html(buffer, include_plotlyjs='cdn')
        st.download_button(label="📥 Download Correlation Chart",
                           data=buffer.getvalue(),
                           file_name="sentiment_correlation.html",
                           mime="text/html")

    # Time series analysis
    st.subheader("📅 Time Series Analysis")

    fig_time = make_subplots(rows=2,
                             cols=1,
                             subplot_titles=('Stock Price Over Time',
                                             'Sentiment Score Over Time'),
                             vertical_spacing=0.1)

    # Stock price line
    fig_time.add_trace(go.Scatter(x=merged_data['date'],
                                  y=merged_data['close'],
                                  mode='lines',
                                  name='Stock Price',
                                  line=dict(color='#1f77b4')),
                       row=1,
                       col=1)

    # Sentiment score line
    fig_time.add_trace(go.Scatter(x=merged_data['date'],
                                  y=merged_data['sentiment_score'],
                                  mode='lines',
                                  name='Sentiment Score',
                                  line=dict(color='#ff7f0e')),
                       row=2,
                       col=1)

    fig_time.update_layout(height=600,
                           title_text="Stock Price and Sentiment Over Time")
    fig_time.update_xaxes(title_text="Date", row=2, col=1)
    fig_time.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_time.update_yaxes(title_text="Sentiment Score", row=2, col=1)

    st.plotly_chart(fig_time, use_container_width=True)

    # Download button for time series
    buffer = io.StringIO()
    fig_time.write_html(buffer, include_plotlyjs='cdn')
    st.download_button(label="📥 Download Time Series Chart",
                       data=buffer.getvalue(),
                       file_name="time_series_analysis.html",
                       mime="text/html")


def display_predictive_modeling():
    st.header("🔮 Predictive Modeling")

    merged_data = st.session_state.merged_data

    # Train model and make predictions
    with st.spinner("Training predictive model..."):
        predictions, mse, model_data = st.session_state.predictor.predict_next_day_price(
            merged_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Model Performance")
        st.metric(label="Mean Squared Error (MSE)",
                  value=f"{mse:.4f}",
                  help="Lower MSE indicates better model performance")

        # Model accuracy visualization
        fig_accuracy = px.scatter(model_data,
                                  x='actual',
                                  y='predicted',
                                  title="Actual vs Predicted Prices",
                                  labels={
                                      'actual': 'Actual Price ($)',
                                      'predicted': 'Predicted Price ($)'
                                  })

        # Add perfect prediction line
        min_val = min(model_data['actual'].min(),
                      model_data['predicted'].min())
        max_val = max(model_data['actual'].max(),
                      model_data['predicted'].max())
        fig_accuracy.add_trace(
            go.Scatter(x=[min_val, max_val],
                       y=[min_val, max_val],
                       mode='lines',
                       name='Perfect Prediction',
                       line=dict(dash='dash', color='red')))

        st.plotly_chart(fig_accuracy, use_container_width=True)

    with col2:
        st.subheader("🔮 Next Day Predictions")

        # Display recent predictions
        recent_predictions = predictions.tail(10)

        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Scatter(x=recent_predictions['date'],
                       y=recent_predictions['actual_price'],
                       mode='lines+markers',
                       name='Actual Price',
                       line=dict(color='blue')))

        fig_pred.add_trace(
            go.Scatter(x=recent_predictions['date'],
                       y=recent_predictions['predicted_price'],
                       mode='lines+markers',
                       name='Predicted Price',
                       line=dict(color='red', dash='dash')))

        fig_pred.update_layout(title="Recent Price Predictions",
                               xaxis_title="Date",
                               yaxis_title="Price ($)")

        st.plotly_chart(fig_pred, use_container_width=True)

    # Prediction table
    st.subheader("📋 Prediction Details")

    # Display predictions table
    display_predictions = predictions[[
        'date', 'actual_price', 'predicted_price', 'prediction_error'
    ]].copy()
    display_predictions['prediction_error'] = display_predictions[
        'prediction_error'].round(4)
    display_predictions['actual_price'] = display_predictions[
        'actual_price'].round(2)
    display_predictions['predicted_price'] = display_predictions[
        'predicted_price'].round(2)

    st.dataframe(display_predictions.tail(20),
                 use_container_width=True,
                 column_config={
                     "date":
                     "Date",
                     "actual_price":
                     st.column_config.NumberColumn("Actual Price",
                                                   format="$%.2f"),
                     "predicted_price":
                     st.column_config.NumberColumn("Predicted Price",
                                                   format="$%.2f"),
                     "prediction_error":
                     st.column_config.NumberColumn("Error", format="%.4f")
                 })

    # Download options for predictions
    col1, col2 = st.columns(2)
    with col1:
        csv_buffer = io.StringIO()
        predictions.to_csv(csv_buffer, index=False)
        st.download_button(label="📥 Download Predictions CSV",
                           data=csv_buffer.getvalue(),
                           file_name="stock_predictions.csv",
                           mime="text/csv")

    with col2:
        # Download prediction chart
        buffer = io.StringIO()
        fig_pred.write_html(buffer, include_plotlyjs='cdn')
        st.download_button(label="📥 Download Prediction Chart",
                           data=buffer.getvalue(),
                           file_name="prediction_chart.html",
                           mime="text/html")


def display_prescriptive_analysis():
    st.header("💡 Prescriptive Analysis")

    merged_data = st.session_state.merged_data

    # Generate buy signals and insights
    buy_signals = st.session_state.data_processor.identify_buy_signals(
        merged_data)
    marketing_insights = st.session_state.sentiment_analyzer.generate_marketing_insights(
        merged_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Buy Signals")

        # Buy signals metrics
        total_signals = len(buy_signals)
        positive_sentiment_signals = len(
            buy_signals[buy_signals['sentiment_score'] > 0.5])

        st.metric("Total Buy Signals", total_signals)
        st.metric("High Sentiment Signals", positive_sentiment_signals)

        if total_signals > 0:
            # Visualize buy signals
            fig_signals = px.scatter(merged_data,
                                     x='date',
                                     y='close',
                                     title="Stock Price with Buy Signals",
                                     labels={
                                         'close': 'Stock Price ($)',
                                         'date': 'Date'
                                     })

            # Add buy signals
            fig_signals.add_trace(
                go.Scatter(x=buy_signals['date'],
                           y=buy_signals['close'],
                           mode='markers',
                           name='Buy Signals',
                           marker=dict(color='green',
                                       size=10,
                                       symbol='triangle-up')))

            st.plotly_chart(fig_signals, use_container_width=True)

            # Buy signals table
            st.subheader("📋 Buy Signals Details")
            display_signals = buy_signals[[
                'date', 'close', 'sentiment_score', 'price_change'
            ]].copy()
            display_signals['close'] = display_signals['close'].round(2)
            display_signals['sentiment_score'] = display_signals[
                'sentiment_score'].round(4)
            display_signals['price_change'] = display_signals[
                'price_change'].round(4)

            st.dataframe(display_signals,
                         use_container_width=True,
                         column_config={
                             "date":
                             "Date",
                             "close":
                             st.column_config.NumberColumn("Price",
                                                           format="$%.2f"),
                             "sentiment_score":
                             st.column_config.NumberColumn("Sentiment",
                                                           format="%.4f"),
                             "price_change":
                             st.column_config.NumberColumn("Price Change",
                                                           format="%.4f")
                         })

            # Download buy signals data
            col1, col2 = st.columns(2)
            with col1:
                csv_buffer = io.StringIO()
                buy_signals.to_csv(csv_buffer, index=False)
                st.download_button(label="📥 Download Buy Signals CSV",
                                   data=csv_buffer.getvalue(),
                                   file_name="buy_signals.csv",
                                   mime="text/csv")

            with col2:
                # Download buy signals chart
                buffer = io.StringIO()
                fig_signals.write_html(buffer, include_plotlyjs='cdn')
                st.download_button(label="📥 Download Signals Chart",
                                   data=buffer.getvalue(),
                                   file_name="buy_signals_chart.html",
                                   mime="text/html")
        else:
            st.info("No buy signals identified in the current dataset.")

    with col2:
        st.subheader("📊 Marketing Insights")

        # Display marketing insights
        for insight in marketing_insights:
            st.info(f"💡 {insight}")

        # Download marketing insights
        insights_text = "\n".join(
            [f"• {insight}" for insight in marketing_insights])
        st.download_button(label="📥 Download Marketing Insights",
                           data=insights_text,
                           file_name="marketing_insights.txt",
                           mime="text/plain")

        # Sentiment trend analysis
        st.subheader("📈 Sentiment Trends")

        # Calculate rolling sentiment average
        merged_data['sentiment_ma'] = merged_data['sentiment_score'].rolling(
            window=7).mean()

        fig_sentiment_trend = px.line(
            merged_data,
            x='date',
            y=['sentiment_score', 'sentiment_ma'],
            title="Sentiment Score Trend (7-day Moving Average)",
            labels={
                'value': 'Sentiment Score',
                'date': 'Date',
                'variable': 'Metric'
            })

        st.plotly_chart(fig_sentiment_trend, use_container_width=True)

        # Download sentiment trend chart
        buffer = io.StringIO()
        fig_sentiment_trend.write_html(buffer, include_plotlyjs='cdn')
        st.download_button(label="📥 Download Sentiment Trend Chart",
                           data=buffer.getvalue(),
                           file_name="sentiment_trend_chart.html",
                           mime="text/html")

    # Strategy recommendations
    st.subheader("🎯 Trading Strategy Recommendations")

    strategy_recommendations = st.session_state.data_processor.generate_strategy_recommendations(
        merged_data, buy_signals)

    for i, recommendation in enumerate(strategy_recommendations, 1):
        st.markdown(f"**{i}. {recommendation}**")

    # Download strategy recommendations
    recommendations_text = "\n".join(
        [f"{i+1}. {rec}" for i, rec in enumerate(strategy_recommendations)])
    st.download_button(label="📥 Download Strategy Recommendations",
                       data=recommendations_text,
                       file_name="strategy_recommendations.txt",
                       mime="text/plain")


def display_performance_analysis():
    st.header("📊 Performance Analysis")

    merged_data = st.session_state.merged_data
    buy_signals = st.session_state.data_processor.identify_buy_signals(
        merged_data)

    # Calculate strategy performance
    performance_data = st.session_state.data_processor.calculate_strategy_performance(
        merged_data, buy_signals)
    
    if len(buy_signals) == 0:
        st.warning("No buy signals found in the data. Performance analysis requires at least one buy signal.")
        st.info("Consider adjusting the sentiment threshold or price drop criteria in the strategy.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Strategy Performance")

        # Performance metrics
        total_return = performance_data['total_strategy_return']
        market_return = performance_data['total_market_return']
        excess_return = total_return - market_return

        st.metric("Strategy Return", f"{total_return:.2%}")
        st.metric("Market Return", f"{market_return:.2%}")
        st.metric("Excess Return",
                  f"{excess_return:.2%}",
                  delta=f"{excess_return:.2%}")

        # Sharpe ratio (simplified)
        if 'volatility' in performance_data:
            sharpe_ratio = total_return / performance_data[
                'volatility'] if performance_data['volatility'] > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.4f}")

    with col2:
        st.subheader("📊 Risk Metrics")

        # Risk metrics
        volatility = performance_data.get('volatility', 0)
        max_drawdown = performance_data.get('max_drawdown', 0)

        st.metric("Volatility", f"{volatility:.2%}")
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")

    # Cumulative returns chart
    st.subheader("📈 Cumulative Returns Comparison")

    returns_data = performance_data['returns_data']
    
    if not returns_data.empty and len(returns_data) > 0:
        fig_returns = px.line(
            returns_data,
            x='date',
            y=['strategy_cumulative_return', 'market_cumulative_return'],
            title="Strategy vs Market Cumulative Returns",
            labels={
                'value': 'Cumulative Return (%)',
                'date': 'Date',
                'variable': 'Strategy'
            })
    else:
        st.warning("No performance data available for chart generation")
        fig_returns = px.line(title="No Data Available")

    # Update legend
    fig_returns.for_each_trace(lambda trace: trace.update(
        name="Strategy Returns"
        if "strategy" in trace.name else "Market Returns"))

    st.plotly_chart(fig_returns, use_container_width=True)

    # Performance summary table
    st.subheader("📋 Performance Summary")

    summary_data = {
        'Metric': [
            'Total Strategy Return', 'Total Market Return', 'Excess Return',
            'Volatility', 'Max Drawdown', 'Number of Trades', 'Win Rate'
        ],
        'Value': [
            f"{total_return:.2%}", f"{market_return:.2%}",
            f"{excess_return:.2%}", f"{volatility:.2%}", f"{max_drawdown:.2%}",
            f"{len(buy_signals)}", f"{performance_data.get('win_rate', 0):.2%}"
        ]
    }

    st.dataframe(pd.DataFrame(summary_data),
                 use_container_width=True,
                 hide_index=True)

    # Download performance data
    st.subheader("📥 Download Performance Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download performance chart
        buffer = io.StringIO()
        fig_returns.write_html(buffer, include_plotlyjs='cdn')
        st.download_button(label="📥 Download Performance Chart",
                           data=buffer.getvalue(),
                           file_name="performance_chart.html",
                           mime="text/html")

    with col2:
        # Download performance data CSV
        csv_buffer = io.StringIO()
        returns_data.to_csv(csv_buffer, index=False)
        st.download_button(label="📥 Download Performance CSV",
                           data=csv_buffer.getvalue(),
                           file_name="performance_data.csv",
                           mime="text/csv")

    with col3:
        # Download summary report
        summary_report = f"""Performance Analysis Summary

Strategy Return: {total_return:.2%}
Market Return: {market_return:.2%}
Outperformance: {excess_return:.2%}
Volatility: {volatility:.2%}
Maximum Drawdown: {max_drawdown:.2%}

Total Trades: {len(buy_signals)}
Win Rate: {performance_data.get('win_rate', 0):.2%}
        """
        st.download_button(label="📥 Download Summary Report",
                           data=summary_report,
                           file_name="performance_summary.txt",
                           mime="text/plain")


if __name__ == "__main__":
    main()
