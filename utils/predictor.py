import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def prepare_features(self, merged_data):
        """Prepare features for prediction model"""
        try:
            # Create a copy of the data
            data = merged_data.copy()
            
            # Sort by date to ensure proper order
            data = data.sort_values('date')
            
            # Create lagged features
            data['close_lag1'] = data['close'].shift(1)
            data['close_lag2'] = data['close'].shift(2)
            data['sentiment_lag1'] = data['sentiment_score'].shift(1)
            
            # Create moving averages
            data['close_ma3'] = data['close'].rolling(window=3).mean()
            data['close_ma7'] = data['close'].rolling(window=7).mean()
            data['sentiment_ma3'] = data['sentiment_score'].rolling(window=3).mean()
            
            # Create price change features
            data['price_change_lag1'] = data['price_change'].shift(1)
            data['price_volatility'] = data['close'].rolling(window=5).std()
            
            # Create sentiment features
            data['sentiment_momentum'] = data['sentiment_score'] - data['sentiment_score'].shift(1)
            data['sentiment_volatility'] = data['sentiment_score'].rolling(window=5).std()
            
            # Create technical indicators
            data['rsi'] = self.calculate_rsi(data['close'])
            data['price_position'] = (data['close'] - data['close'].rolling(window=20).min()) / (
                data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min()
            )
            
            # Define feature columns
            feature_columns = [
                'close_lag1', 'close_lag2', 'sentiment_score', 'sentiment_lag1',
                'close_ma3', 'close_ma7', 'sentiment_ma3', 'price_change_lag1',
                'price_volatility', 'sentiment_momentum', 'sentiment_volatility',
                'rsi', 'price_position'
            ]
            
            # Remove rows with NaN values
            data = data.dropna()
            
            return data, feature_columns
            
        except Exception as e:
            raise Exception(f"Feature preparation error: {str(e)}")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def train_model(self, X, y):
        """Train the linear regression model"""
        try:
            # Initialize and fit the scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train the model
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate MSE
            mse = mean_squared_error(y_test, y_pred)
            
            return {
                'model': self.model,
                'scaler': self.scaler,
                'mse': mse,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
        except Exception as e:
            raise Exception(f"Model training error: {str(e)}")
    
    def predict_next_day_price(self, merged_data):
        """Predict next day stock prices using linear regression"""
        try:
            # Prepare features
            data, feature_columns = self.prepare_features(merged_data)
            
            if len(data) < 20:  # Minimum data requirement
                raise Exception("Insufficient data for prediction (minimum 20 records required)")
            
            # Prepare features and target
            X = data[feature_columns]
            y = data['close']
            
            # Train the model
            model_results = self.train_model(X, y)
            mse = model_results['mse']
            
            # Create predictions for the entire dataset
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Create prediction results
            prediction_results = data[['date', 'close']].copy()
            prediction_results['predicted_price'] = predictions
            prediction_results['prediction_error'] = abs(prediction_results['close'] - prediction_results['predicted_price'])
            prediction_results.rename(columns={'close': 'actual_price'}, inplace=True)
            
            # Create model performance data for visualization
            model_data = pd.DataFrame({
                'actual': model_results['y_test'],
                'predicted': model_results['y_pred']
            })
            
            return prediction_results, mse, model_data
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        try:
            if self.model is None or self.feature_columns is None:
                return None
            
            # Get feature coefficients
            coefficients = self.model.coef_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            })
            
            # Sort by absolute coefficient value
            importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            
            return importance_df
            
        except Exception as e:
            return None
    
    def predict_single_day(self, features):
        """Predict price for a single day given features"""
        try:
            if self.model is None or self.scaler is None:
                raise Exception("Model not trained yet")
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            raise Exception(f"Single day prediction error: {str(e)}")
    
    def calculate_prediction_confidence(self, merged_data):
        """Calculate confidence intervals for predictions"""
        try:
            if self.model is None:
                return None
            
            # Prepare features
            data, feature_columns = self.prepare_features(merged_data)
            X = data[feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Calculate residuals (simplified confidence estimation)
            actual_prices = data['close'].values
            residuals = actual_prices - predictions
            
            # Calculate standard error
            std_error = np.std(residuals)
            
            # Create confidence intervals (±2 standard errors)
            confidence_intervals = pd.DataFrame({
                'date': data['date'],
                'prediction': predictions,
                'lower_bound': predictions - 2 * std_error,
                'upper_bound': predictions + 2 * std_error,
                'confidence_width': 4 * std_error
            })
            
            return confidence_intervals
            
        except Exception as e:
            return None
