# Stock Market & Sentiment Analysis Dashboard - LLM Creation Prompt

Use this prompt with any LLM to recreate the complete Stock Market & Sentiment Analysis Dashboard:

---

## PROMPT FOR LLM:

Create a comprehensive Stock Market & Sentiment Analysis Dashboard using Streamlit with the following exact specifications:

### Core Requirements:
1. **Framework**: Use Streamlit for the web interface
2. **File Structure**: Create modular Python files with proper separation of concerns
3. **File Upload**: Support CSV (social media) and Excel (stock market) file formats
4. **Analysis Types**: Implement 4 main analysis categories in tabs

### Technical Architecture:

#### Main Application (app.py):
- Streamlit interface with sidebar configuration
- Theme switching (Light/Dark) with custom CSS
- File upload functionality for 2 datasets:
  - Social media data (CSV): columns 'date', 'text'
  - Stock market data (Excel): columns 'date', 'close'
- Session state management for utility classes
- Four analysis tabs: Diagnostic, Predictive, Prescriptive, Performance

#### Utility Modules (utils/ folder):

**1. data_processor.py:**
- DataProcessor class with methods:
  - validate_datasets(): Check required columns and formats
  - process_data(): Clean and prepare datasets
  - _parse_dates_flexible(): Handle multiple date formats (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY)
  - merge_datasets(): Combine sentiment and stock data on date
  - identify_buy_signals(): Find opportunities based on sentiment + price drops
  - generate_strategy_recommendations(): Create trading advice
  - calculate_strategy_performance(): Compute returns and metrics

**2. sentiment_analyzer.py:**
- SentimentAnalyzer class with methods:
  - clean_text(): Remove URLs, mentions, special characters
  - analyze_sentiment(): Use TextBlob for sentiment scoring
  - generate_marketing_insights(): Create actionable marketing advice
  - get_sentiment_summary(): Provide sentiment statistics

**3. predictor.py:**
- StockPredictor class with methods:
  - prepare_features(): Engineer features (moving averages, RSI, sentiment momentum)
  - calculate_rsi(): Technical indicator calculation
  - train_model(): Linear regression with StandardScaler
  - predict_next_day_price(): Generate price predictions
  - get_feature_importance(): Model interpretability
  - calculate_prediction_confidence(): Confidence intervals

**4. visualizations.py:**
- ChartGenerator class with methods:
  - create_sentiment_pie_chart(): Sentiment distribution
  - create_correlation_scatter(): Sentiment vs price correlation
  - create_time_series_chart(): Price and sentiment over time
  - create_prediction_chart(): Actual vs predicted prices
  - create_buy_signals_chart(): Price chart with buy signal markers
  - create_performance_chart(): Strategy vs market returns
  - create_sentiment_trend_chart(): Sentiment with moving average
  - Use consistent color palette and Plotly for all charts

### Analysis Tabs Implementation:

**Tab 1 - Diagnostic Analysis:**
- Sentiment distribution pie chart
- Sentiment vs stock price correlation scatter plot
- Time series visualization of both metrics
- Summary statistics and insights

**Tab 2 - Predictive Modeling:**
- Linear regression model for next-day price prediction
- Model performance metrics (MSE, R-squared)
- Actual vs predicted price scatter plot
- Prediction accuracy visualization
- Feature importance display

**Tab 3 - Prescriptive Analysis:**
- Buy signal identification (positive sentiment + price drop > 2%)
- Buy signals plotted on stock price chart
- Marketing insights based on sentiment analysis
- Sentiment trend analysis with 7-day moving average
- Trading strategy recommendations

**Tab 4 - Performance Analysis:**
- Strategy returns vs market returns comparison
- Cumulative performance tracking
- Risk metrics (volatility, maximum drawdown, Sharpe ratio)
- Performance summary table
- Win rate and trade statistics

### Dependencies Required:
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
textblob>=0.17.1
openpyxl>=3.1.0
statsmodels>=0.14.0
```

### Streamlit Configuration (.streamlit/config.toml):
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Key Features to Implement:
1. **Data Validation**: Comprehensive error handling and user feedback
2. **Flexible Date Parsing**: Support multiple date formats automatically
3. **Interactive Visualizations**: All charts should be interactive using Plotly
4. **Theme Support**: Light/dark theme switching with custom CSS
5. **Session State**: Maintain utility class instances across interactions
6. **Error Handling**: Graceful error messages and fallback displays
7. **Professional UI**: Clean, intuitive interface with clear navigation

### Business Logic:
- **Buy Signals**: Identify when sentiment score > 0.5 AND price change < -2%
- **Sentiment Classification**: Positive (>0.5), Neutral (0.4-0.6), Negative (<0.4)
- **Performance Calculation**: Compare strategy returns to buy-and-hold market returns
- **Risk Metrics**: Calculate volatility, maximum drawdown, and Sharpe ratio
- **Marketing Insights**: Generate actionable recommendations based on sentiment trends

### Data Processing Flow:
1. Upload and validate both datasets
2. Parse dates flexibly and merge on date column
3. Perform sentiment analysis on text data
4. Engineer features for prediction model
5. Generate visualizations and analysis
6. Display results in organized tabs

### CSS Styling:
Implement custom CSS for professional appearance with support for both light and dark themes, ensuring consistent styling across all components.

This dashboard should provide comprehensive analysis combining social media sentiment with stock market data to generate actionable insights for both trading and marketing decisions.

---

## Additional Instructions for LLM:
- Use proper Python coding practices with clear docstrings
- Implement comprehensive error handling
- Ensure all visualizations are interactive and downloadable
- Make the interface intuitive and professional
- Test with sample data to ensure functionality
- Create modular, maintainable code structure
- Follow Streamlit best practices for performance and UX