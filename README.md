# Stock Market & Sentiment Analysis Dashboard

A comprehensive Streamlit-based dashboard that combines stock market data analysis with social media sentiment analysis to provide insights into market trends and investor sentiment.

## Features

### 📊 Diagnostic Analysis
- Sentiment distribution visualization (pie charts)
- Correlation analysis between sentiment and stock prices
- Time series analysis of stock prices and sentiment trends

### 🔮 Predictive Modeling
- Linear regression model for next-day stock price prediction
- Model performance metrics (MSE, accuracy visualization)
- Feature importance analysis

### 💡 Prescriptive Analysis
- Buy signal identification based on sentiment and price drops
- Marketing insights based on sentiment analysis
- Trading strategy recommendations

### 📈 Performance Analysis
- Strategy returns vs market returns comparison
- Risk metrics (volatility, maximum drawdown)
- Cumulative performance tracking

### 🎨 User Interface
- Light/Dark theme switching
- Interactive Plotly visualizations
- Comprehensive download functionality for all charts and data
- Responsive design

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-sentiment-dashboard.git
cd stock-sentiment-dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements_github.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Data Requirements

### Social Media Dataset (CSV format)
- `date`: Date column (supports YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY formats)
- `text`: Text content for sentiment analysis

### Stock Market Dataset (Excel format)
- `date`: Date column (supports YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY formats)
- `close`: Closing price of the stock

## Usage

1. Upload your social media dataset (CSV file)
2. Upload your stock market dataset (Excel file)
3. Click "Process Data" to begin analysis
4. Navigate through the four analysis tabs:
   - **Diagnostic Analysis**: View sentiment distribution and correlations
   - **Predictive Modeling**: See stock price predictions and model performance
   - **Prescriptive Analysis**: Explore buy signals and marketing insights
   - **Performance Analysis**: Compare strategy performance against market

## Download Capabilities

The dashboard provides comprehensive download functionality:
- All charts as interactive HTML files
- All data tables as CSV files
- Marketing insights and strategy recommendations as text files
- Performance summary reports
- Complete processed datasets

## Technical Architecture

- **Frontend**: Streamlit web application
- **Data Processing**: Pandas for data manipulation
- **Machine Learning**: Scikit-learn for linear regression
- **Sentiment Analysis**: TextBlob for natural language processing
- **Visualization**: Plotly for interactive charts
- **File Support**: CSV and Excel file formats

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- textblob
- openpyxl
- statsmodels

## Configuration

The application includes a `.streamlit/config.toml` file for optimal deployment settings. The dashboard is configured to run on port 5000 and supports both light and dark themes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please open an issue in the GitHub repository.