# Stock Market & Sentiment Analysis Dashboard

## Overview

This project is a comprehensive Streamlit-based dashboard that combines stock market data analysis with social media sentiment analysis to provide insights into market trends and investor sentiment. The application leverages machine learning for stock price prediction and natural language processing for sentiment analysis, presenting the results through interactive visualizations.

## System Architecture

The application follows a modular Python architecture with clear separation of concerns:

**Frontend**: Streamlit web application providing an interactive dashboard interface
**Backend Logic**: Utility modules handling data processing, sentiment analysis, prediction, and visualization
**Data Processing**: Pandas-based data manipulation and sklearn-based machine learning models
**Visualization**: Plotly-based interactive charts and graphs
**Deployment**: Replit-hosted with autoscale deployment configuration

## Key Components

### Core Application (`app.py`)
- Main Streamlit dashboard interface
- Session state management for utility class instances
- Configuration sidebar with theme selection
- Modular design allowing for easy extension

### Data Processing Module (`utils/data_processor.py`)
- **Purpose**: Validates and processes both social media and stock market datasets
- **Key Features**:
  - Dataset validation ensuring required columns (`date`, `text` for social media; `date`, `close` for stock data)
  - Data format validation (date parsing, numeric validation)
  - Empty dataset detection
- **Architecture Decision**: Centralized validation logic to ensure data quality before analysis

### Sentiment Analysis Module (`utils/sentiment_analyzer.py`)
- **Purpose**: Processes social media text data to extract sentiment scores
- **Key Features**:
  - Text preprocessing (URL removal, mention/hashtag cleaning, special character handling)
  - TextBlob-based sentiment analysis
  - Sentiment classification (Positive, Negative, Neutral)
- **Architecture Decision**: Used TextBlob for simplicity and quick implementation over more complex models

### Stock Prediction Module (`utils/predictor.py`)
- **Purpose**: Creates machine learning models for stock price prediction
- **Key Features**:
  - Feature engineering with lagged variables and moving averages
  - Technical indicators (RSI, price position)
  - Sklearn LinearRegression model with StandardScaler
  - Sentiment momentum and volatility features
- **Architecture Decision**: Linear regression chosen for interpretability and baseline performance

### Visualization Module (`utils/visualizations.py`)
- **Purpose**: Generates interactive charts and visualizations
- **Key Features**:
  - Plotly-based interactive charts
  - Consistent color palette across visualizations
  - Error handling with fallback error charts
  - Pie charts for sentiment distribution and scatter plots for correlations
- **Architecture Decision**: Plotly chosen for rich interactivity and integration with Streamlit

## Data Flow

1. **Data Input**: Social media and stock market datasets are uploaded/loaded
2. **Validation**: DataProcessor validates dataset structure and format
3. **Sentiment Analysis**: SentimentAnalyzer processes social media text to extract sentiment scores
4. **Feature Engineering**: StockPredictor creates predictive features from historical data
5. **Model Training**: Machine learning model is trained on engineered features
6. **Visualization**: ChartGenerator creates interactive charts for analysis
7. **Dashboard Display**: Streamlit renders results in user-friendly interface

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for dashboard interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualization library
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **TextBlob**: Natural language processing for sentiment analysis

### Deployment Dependencies
- **Python 3.11**: Runtime environment
- **Nix**: Package management system
- **Replit**: Hosting platform with autoscale deployment

## Deployment Strategy

**Platform**: Replit with autoscale deployment target
**Runtime**: Python 3.11 environment with Nix package management
**Port Configuration**: Application runs on port 5000
**Launch Command**: `streamlit run app.py --server.port 5000`
**Workflow**: Automated deployment with parallel workflow execution

The deployment is configured for automatic scaling based on demand, with Streamlit optimized for headless operation on the server.

## Recent Changes

- June 22, 2025: Initial setup and core application development
- Enhanced theme switching functionality with comprehensive CSS styling
- Updated file upload to accept Excel format for stock market data
- Implemented flexible date parsing to handle various date formats
- Simplified date validation to be more permissive (allows YYYY-MM-DD format)
- Added statsmodels dependency for statistical analysis
- Fixed data validation issues and improved error messaging

## Changelog

- June 22, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.

## Current Status

The Stock Market & Sentiment Analysis Dashboard is fully functional with the following capabilities:

**Core Features Implemented:**
- File upload supporting CSV (social media) and Excel (stock market) formats
- Flexible date parsing that handles YYYY-MM-DD format and ignores time components
- Light/Dark theme switching with comprehensive CSS styling
- Four main analysis tabs: Diagnostic, Predictive, Prescriptive, and Performance Analysis

**Data Processing:**
- Validates required columns (date, text for social media; date, close for stock market)
- Handles various date formats automatically
- Sentiment analysis using TextBlob
- Stock price prediction using linear regression
- Buy signal identification based on sentiment and price drops

**Visualization:**
- Interactive Plotly charts with download functionality
- Pie charts for sentiment distribution
- Scatter plots for correlation analysis
- Time series charts for trends
- Performance comparison charts

**Technical Implementation:**
- Arrow serialization compatibility fixes for data display
- Comprehensive error handling and validation
- Session state management for utility classes
- Responsive design with proper theme support

**Dependencies Installed:**
- streamlit, pandas, numpy, plotly, scikit-learn, textblob, openpyxl, statsmodels

The application is ready for production use and can handle real financial and social media datasets for comprehensive market sentiment analysis.

## GitHub Upload Preparation

**Files Created for GitHub Upload:**
- README.md - Comprehensive project documentation
- LICENSE - MIT license for open source distribution
- CONTRIBUTING.md - Guidelines for contributors
- setup.py - Python package configuration
- requirements_github.txt - Dependencies for GitHub (rename to requirements.txt after upload)
- .gitignore - Excludes Replit-specific files and sensitive data
- docs/DEPLOYMENT.md - Deployment instructions for various platforms
- sample_data/README.md - Data format guidelines and examples
- GITHUB_UPLOAD_GUIDE.md - Step-by-step upload instructions

**Ready for Upload:** The project structure is now optimized for GitHub with proper documentation, configuration files, and deployment guides. All Replit-specific files are excluded via .gitignore.

## LLM Recreation Template

**Created:** LLM_PROMPT_TEMPLATE.md - Comprehensive prompt for recreating this dashboard with any LLM
**Purpose:** Allows other developers to recreate the exact same dashboard functionality using different AI assistants
**Content:** Complete technical specifications, architecture details, and implementation requirements