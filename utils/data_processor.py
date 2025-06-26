import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        pass
    
    def validate_datasets(self, social_df, stock_df):
        """Validate that datasets have required columns and proper format"""
        try:
            # Check social media dataset columns
            required_social_cols = ['date', 'text']
            social_cols_missing = [col for col in required_social_cols if col not in social_df.columns]
            if social_cols_missing:
                return {
                    'valid': False,
                    'message': f"Social media dataset missing columns: {social_cols_missing}. Found columns: {list(social_df.columns)}"
                }
            
            # Check stock market dataset columns
            required_stock_cols = ['date', 'close']
            stock_cols_missing = [col for col in required_stock_cols if col not in stock_df.columns]
            if stock_cols_missing:
                return {
                    'valid': False,
                    'message': f"Stock market dataset missing columns: {stock_cols_missing}. Found columns: {list(stock_df.columns)}"
                }
            
            # Check if datasets have data
            if len(social_df) == 0:
                return {'valid': False, 'message': "Social media dataset is empty"}
            
            if len(stock_df) == 0:
                return {'valid': False, 'message': "Stock market dataset is empty"}
            
            # Simplified date validation - just check if pandas can parse them
            try:
                # Try to parse social media dates
                social_dates = pd.to_datetime(social_df['date'], errors='coerce')
                
                # Allow up to 20% invalid dates
                invalid_count = social_dates.isna().sum()
                if invalid_count > len(social_df) * 0.2:
                    sample_dates = social_df['date'].head(5).tolist()
                    return {'valid': False, 'message': f"Too many invalid dates in social media dataset. Sample dates: {sample_dates}"}
                    
            except Exception as e:
                return {'valid': False, 'message': f"Error with social media dates: {str(e)}"}
            
            try:
                # Try to parse stock market dates
                stock_dates = pd.to_datetime(stock_df['date'], errors='coerce')
                
                # Allow up to 20% invalid dates
                invalid_count = stock_dates.isna().sum()
                if invalid_count > len(stock_df) * 0.2:
                    sample_dates = stock_df['date'].head(5).tolist()
                    return {'valid': False, 'message': f"Too many invalid dates in stock market dataset. Sample dates: {sample_dates}"}
                    
            except Exception as e:
                return {'valid': False, 'message': f"Error with stock market dates: {str(e)}"}
            
            # Validate close price column with more specific error messages
            try:
                close_prices = pd.to_numeric(stock_df['close'], errors='coerce')
                if close_prices.isna().any():
                    invalid_prices = stock_df.loc[close_prices.isna(), 'close'].head(3).tolist()
                    return {'valid': False, 'message': f"Stock market dataset has non-numeric close prices: {invalid_prices}"}
            except Exception as e:
                return {'valid': False, 'message': f"Error parsing close prices: {str(e)}"}
            
            # Check for empty text entries
            empty_text_count = pd.isna(social_df['text']).sum() + (social_df['text'] == '').sum()
            if empty_text_count > len(social_df) * 0.5:
                return {'valid': False, 'message': f"Too many empty text entries in social media dataset: {empty_text_count}/{len(social_df)}"}
            
            return {'valid': True, 'message': "Validation successful"}
            
        except Exception as e:
            return {'valid': False, 'message': f"Validation error: {str(e)}"}
    
    def process_data(self, social_df, stock_df):
        """Process and clean the datasets"""
        try:
            # Process social media data
            social_processed = social_df.copy()
            
            # Simple date parsing
            social_processed['date'] = pd.to_datetime(social_processed['date'], errors='coerce')
            social_processed = social_processed.dropna(subset=['date', 'text'])
            social_processed['text'] = social_processed['text'].astype(str)
            
            # Process stock market data
            stock_processed = stock_df.copy()
            
            # Simple date parsing
            stock_processed['date'] = pd.to_datetime(stock_processed['date'], errors='coerce')
            
            stock_processed['close'] = pd.to_numeric(stock_processed['close'], errors='coerce')
            stock_processed = stock_processed.dropna(subset=['date', 'close'])
            
            # Ensure data types are compatible with Arrow serialization
            stock_processed = stock_processed.reset_index(drop=True)
            social_processed = social_processed.reset_index(drop=True)
            
            # Convert object columns to string to prevent Arrow conversion issues
            for col in stock_processed.columns:
                if stock_processed[col].dtype == 'object' and col != 'date':
                    stock_processed[col] = stock_processed[col].astype(str)
            
            for col in social_processed.columns:
                if social_processed[col].dtype == 'object' and col != 'date':
                    social_processed[col] = social_processed[col].astype(str)
            
            # Sort by date
            social_processed = social_processed.sort_values('date')
            stock_processed = stock_processed.sort_values('date')
            
            return {
                'social_media': social_processed,
                'stock_market': stock_processed
            }
            
        except Exception as e:
            raise Exception(f"Data processing error: {str(e)}")
    
    def _parse_dates_flexible(self, date_series):
        """Helper function to parse dates with multiple format support - date only"""
        # First, try pandas automatic parsing which should handle YYYY-MM-DD format
        try:
            parsed_dates = pd.to_datetime(date_series, errors='coerce')
            valid_count = len(parsed_dates) - parsed_dates.isna().sum()
            if valid_count > len(parsed_dates) * 0.8:  # If 80% success
                return parsed_dates
        except:
            pass
        
        # Convert to string and clean up
        date_series_str = date_series.astype(str).str.strip()
        # Extract only the date part (before space if datetime format)
        date_series_str = date_series_str.str.split(' ').str[0]
        
        # Common date formats without time
        date_formats = [
            '%Y-%m-%d',    # YYYY-MM-DD (your format)
            '%d-%m-%Y',    # DD-MM-YYYY
            '%d/%m/%Y',    # DD/MM/YYYY  
            '%m/%d/%Y',    # MM/DD/YYYY
            '%Y/%m/%d',    # YYYY/MM/DD
            '%m-%d-%Y',    # MM-DD-YYYY
        ]
        
        # Try each format
        for fmt in date_formats:
            try:
                parsed_dates = pd.to_datetime(date_series_str, format=fmt, errors='coerce')
                valid_count = len(parsed_dates) - parsed_dates.isna().sum()
                if valid_count > len(parsed_dates) * 0.8:  # If 80% success
                    return parsed_dates
            except Exception:
                continue
        
        # Final fallback
        try:
            return pd.to_datetime(date_series_str, errors='coerce', infer_datetime_format=True)
        except Exception:
            return pd.to_datetime(date_series, errors='coerce')
    
    def merge_datasets(self, sentiment_data, stock_data):
        """Merge sentiment and stock data on date"""
        try:
            # Aggregate sentiment data by date (if multiple entries per day)
            sentiment_agg = sentiment_data.groupby('date').agg({
                'sentiment_score': 'mean',
                'sentiment_label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            }).reset_index()
            
            # Merge datasets
            merged = pd.merge(
                stock_data,
                sentiment_agg,
                on='date',
                how='inner'
            )
            
            # Calculate additional features
            merged['price_change'] = merged['close'].pct_change()
            merged['price_direction'] = merged['price_change'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else 'flat')
            
            # Remove rows with NaN values
            merged = merged.dropna()
            
            return merged
            
        except Exception as e:
            raise Exception(f"Data merging error: {str(e)}")
    
    def identify_buy_signals(self, merged_data):
        """Identify buy signals based on sentiment and price drops"""
        try:
            # Buy signal criteria:
            # 1. Positive sentiment (sentiment_score > 0.5)
            # 2. Price drop (price_change < -0.02, i.e., more than 2% drop)
            buy_signals = merged_data[
                (merged_data['sentiment_score'] > 0.5) & 
                (merged_data['price_change'] < -0.02)
            ].copy()
            
            return buy_signals
            
        except Exception as e:
            raise Exception(f"Buy signal identification error: {str(e)}")
    
    def generate_strategy_recommendations(self, merged_data, buy_signals):
        """Generate trading strategy recommendations"""
        try:
            recommendations = []
            
            # Analysis of data
            avg_sentiment = merged_data['sentiment_score'].mean()
            sentiment_volatility = merged_data['sentiment_score'].std()
            price_volatility = merged_data['close'].std() / merged_data['close'].mean()
            
            # Recommendation 1: Sentiment threshold
            if avg_sentiment > 0.6:
                recommendations.append(
                    "High average sentiment detected. Consider increasing position sizes during positive sentiment periods."
                )
            elif avg_sentiment < 0.4:
                recommendations.append(
                    "Low average sentiment detected. Exercise caution and consider defensive strategies."
                )
            else:
                recommendations.append(
                    "Neutral sentiment environment. Maintain balanced approach with selective entries."
                )
            
            # Recommendation 2: Volatility-based
            if sentiment_volatility > 0.3:
                recommendations.append(
                    "High sentiment volatility observed. Use shorter holding periods and tighter stop-losses."
                )
            
            # Recommendation 3: Buy signals analysis
            if len(buy_signals) > 0:
                avg_return_after_signal = self._calculate_avg_return_after_signals(merged_data, buy_signals)
                if avg_return_after_signal > 0.02:
                    recommendations.append(
                        f"Buy signals show positive average return of {avg_return_after_signal:.2%}. Continue using sentiment-based entries."
                    )
                else:
                    recommendations.append(
                        "Buy signals show mixed results. Consider refining entry criteria or adding technical indicators."
                    )
            else:
                recommendations.append(
                    "No buy signals detected. Consider relaxing sentiment thresholds or adjusting price drop criteria."
                )
            
            # Recommendation 4: Market conditions
            recent_trend = merged_data['close'].tail(10).pct_change().mean()
            if recent_trend > 0.01:
                recommendations.append(
                    "Strong upward trend detected. Consider momentum-based strategies alongside sentiment analysis."
                )
            elif recent_trend < -0.01:
                recommendations.append(
                    "Downward trend detected. Focus on risk management and consider contrarian approaches."
                )
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
    
    def _calculate_avg_return_after_signals(self, merged_data, buy_signals):
        """Calculate average return after buy signals"""
        try:
            returns = []
            merged_indexed = merged_data.set_index('date')
            
            for _, signal in buy_signals.iterrows():
                signal_date = signal['date']
                signal_price = signal['close']
                
                # Find next trading day price
                future_dates = merged_indexed.index[merged_indexed.index > signal_date]
                if len(future_dates) > 0:
                    next_date = future_dates[0]
                    next_price = merged_indexed.loc[next_date, 'close']
                    return_pct = (next_price - signal_price) / signal_price
                    returns.append(return_pct)
            
            return np.mean(returns) if returns else 0
            
        except:
            return 0
    
    def calculate_strategy_performance(self, merged_data, buy_signals):
        """Calculate comprehensive strategy performance metrics"""
        try:
            if len(buy_signals) == 0:
                return {
                    'total_strategy_return': 0,
                    'total_market_return': 0,
                    'volatility': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'returns_data': pd.DataFrame(),
                    'trades': []
                }
            # Initialize performance tracking
            portfolio_value = 10000  # Starting with $10,000
            cash = portfolio_value
            shares = 0
            trades = []
            
            # Track daily portfolio values
            daily_values = []
            
            merged_indexed = merged_data.set_index('date')
            # Handle buy signal dates properly - convert to date only if datetime
            if len(buy_signals) > 0:
                if pd.api.types.is_datetime64_any_dtype(buy_signals['date']):
                    buy_signal_dates = set(buy_signals['date'].dt.date)
                else:
                    buy_signal_dates = set(pd.to_datetime(buy_signals['date']).dt.date)
            else:
                buy_signal_dates = set()
            
            for date, row in merged_indexed.iterrows():
                current_price = row['close']
                # Handle date conversion properly
                if hasattr(date, 'date'):
                    date_only = date.date()
                else:
                    date_only = pd.to_datetime(date).date()
                
                # Check if it's a buy signal day
                if date_only in buy_signal_dates and cash > current_price:
                    # Buy as many shares as possible
                    shares_to_buy = int(cash // current_price)
                    if shares_to_buy > 0:
                        shares += shares_to_buy
                        cash -= shares_to_buy * current_price
                        trades.append({
                            'date': date,
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': current_price
                        })
                
                # Calculate current portfolio value
                current_portfolio_value = cash + (shares * current_price)
                daily_values.append({
                    'date': date,
                    'portfolio_value': current_portfolio_value,
                    'market_value': (current_price / merged_indexed.iloc[0]['close']) * portfolio_value
                })
            
            # Convert to DataFrame
            performance_df = pd.DataFrame(daily_values)
            
            # Calculate returns
            performance_df['strategy_return'] = performance_df['portfolio_value'].pct_change()
            performance_df['market_return'] = performance_df['market_value'].pct_change()
            
            # Calculate cumulative returns
            performance_df['strategy_cumulative_return'] = (
                (performance_df['portfolio_value'] / portfolio_value - 1) * 100
            )
            performance_df['market_cumulative_return'] = (
                (performance_df['market_value'] / portfolio_value - 1) * 100
            )
            
            # Calculate performance metrics
            total_strategy_return = (performance_df['portfolio_value'].iloc[-1] / portfolio_value) - 1
            total_market_return = (performance_df['market_value'].iloc[-1] / portfolio_value) - 1
            
            # Calculate volatility
            strategy_volatility = performance_df['strategy_return'].std() * np.sqrt(252)  # Annualized
            market_volatility = performance_df['market_return'].std() * np.sqrt(252)
            
            # Calculate maximum drawdown
            running_max = performance_df['portfolio_value'].expanding().max()
            drawdown = (performance_df['portfolio_value'] - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            winning_trades = sum(1 for trade in trades if trade['action'] == 'buy')  # Simplified
            win_rate = winning_trades / len(trades) if trades else 0
            
            return {
                'total_strategy_return': total_strategy_return,
                'total_market_return': total_market_return,
                'volatility': strategy_volatility,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'returns_data': performance_df,
                'trades': trades
            }
            
        except Exception as e:
            # Return default values if calculation fails
            return {
                'total_strategy_return': 0,
                'total_market_return': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'returns_data': pd.DataFrame(),
                'trades': []
            }
