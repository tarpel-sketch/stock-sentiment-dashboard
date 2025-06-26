# Sample Data

This folder contains sample data files to help you get started with the Stock Market & Sentiment Analysis Dashboard.

## File Formats

### Social Media Data (CSV)
Your CSV file should have the following columns:
- `date`: Date in YYYY-MM-DD format (time portion will be ignored if present)
- `text`: Social media text content for sentiment analysis

Example:
```csv
date,text
2019-01-01,"Great product launch today! Very excited about the future prospects."
2019-01-02,"Market conditions seem challenging but optimistic about recovery."
2019-01-03,"Disappointed with quarterly results. Hope management addresses concerns."
```

### Stock Market Data (Excel)
Your Excel file should have the following columns:
- `date`: Date in YYYY-MM-DD format
- `close`: Closing stock price (numeric value)

Example:
```
date        | close
2019-01-01  | 150.25
2019-01-02  | 148.75
2019-01-03  | 152.10
```

## Data Requirements

- Both datasets should cover the same time period for meaningful analysis
- Minimum 20 records recommended for reliable predictions
- Date formats are flexible (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY supported)
- Text data should be clean and representative of public sentiment
- Stock prices should be consistent (same stock, same currency)

## Tips for Best Results

1. Ensure your social media data represents relevant sentiment about the stock/company
2. Use daily data for best correlation analysis
3. Remove any obvious spam or irrelevant content from social media data
4. Verify that dates align between both datasets
5. Include enough historical data for trend analysis (recommended: 3+ months)