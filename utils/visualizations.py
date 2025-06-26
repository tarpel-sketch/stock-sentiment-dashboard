import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ChartGenerator:
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9f40',
            'info': '#17a2b8'
        }
    
    def create_sentiment_pie_chart(self, sentiment_data):
        """Create pie chart for sentiment distribution"""
        try:
            sentiment_counts = sentiment_data['sentiment_label'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'Positive': self.color_palette['success'],
                    'Negative': self.color_palette['danger'],
                    'Neutral': self.color_palette['info']
                }
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12
            )
            
            fig.update_layout(
                title_font_size=16,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating pie chart: {str(e)}")
    
    def create_correlation_scatter(self, merged_data):
        """Create scatter plot for sentiment vs stock price correlation"""
        try:
            fig = px.scatter(
                merged_data,
                x='sentiment_score',
                y='close',
                title="Sentiment Score vs Stock Price",
                labels={
                    'sentiment_score': 'Sentiment Score',
                    'close': 'Stock Price ($)'
                },
                trendline="ols",
                color='sentiment_label',
                color_discrete_map={
                    'Positive': self.color_palette['success'],
                    'Negative': self.color_palette['danger'],
                    'Neutral': self.color_palette['info']
                }
            )
            
            fig.update_layout(
                title_font_size=16,
                xaxis_title_font_size=12,
                yaxis_title_font_size=12
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating scatter plot: {str(e)}")
    
    def create_time_series_chart(self, merged_data):
        """Create time series chart for stock price and sentiment"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Stock Price Over Time', 'Sentiment Score Over Time'),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            # Stock price line
            fig.add_trace(
                go.Scatter(
                    x=merged_data['date'],
                    y=merged_data['close'],
                    mode='lines',
                    name='Stock Price',
                    line=dict(color=self.color_palette['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Sentiment score line
            fig.add_trace(
                go.Scatter(
                    x=merged_data['date'],
                    y=merged_data['sentiment_score'],
                    mode='lines',
                    name='Sentiment Score',
                    line=dict(color=self.color_palette['secondary'], width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                title_text="Stock Price and Sentiment Over Time",
                title_font_size=16,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating time series chart: {str(e)}")
    
    def create_prediction_chart(self, prediction_data):
        """Create chart for actual vs predicted prices"""
        try:
            fig = go.Figure()
            
            # Actual prices
            fig.add_trace(
                go.Scatter(
                    x=prediction_data['date'],
                    y=prediction_data['actual_price'],
                    mode='lines+markers',
                    name='Actual Price',
                    line=dict(color=self.color_palette['primary'])
                )
            )
            
            # Predicted prices
            fig.add_trace(
                go.Scatter(
                    x=prediction_data['date'],
                    y=prediction_data['predicted_price'],
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color=self.color_palette['danger'], dash='dash')
                )
            )
            
            fig.update_layout(
                title="Actual vs Predicted Stock Prices",
                title_font_size=16,
                xaxis_title="Date",
                yaxis_title="Price ($)",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating prediction chart: {str(e)}")
    
    def create_buy_signals_chart(self, merged_data, buy_signals):
        """Create chart showing stock price with buy signals"""
        try:
            fig = px.scatter(
                merged_data,
                x='date',
                y='close',
                title="Stock Price with Buy Signals"
            )
            
            # Update main trace to be a line
            fig.data[0].mode = 'lines'
            fig.data[0].name = 'Stock Price'
            fig.data[0].line = dict(color=self.color_palette['primary'])
            
            # Add buy signals
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals['date'],
                        y=buy_signals['close'],
                        mode='markers',
                        name='Buy Signals',
                        marker=dict(
                            color=self.color_palette['success'],
                            size=12,
                            symbol='triangle-up'
                        )
                    )
                )
            
            fig.update_layout(
                title_font_size=16,
                xaxis_title="Date",
                yaxis_title="Price ($)",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating buy signals chart: {str(e)}")
    
    def create_performance_chart(self, returns_data):
        """Create performance comparison chart"""
        try:
            fig = px.line(
                returns_data,
                x='date',
                y=['strategy_cumulative_return', 'market_cumulative_return'],
                title="Strategy vs Market Cumulative Returns"
            )
            
            # Update trace names and colors
            fig.data[0].name = 'Strategy Returns'
            fig.data[0].line = dict(color=self.color_palette['success'])
            fig.data[1].name = 'Market Returns'
            fig.data[1].line = dict(color=self.color_palette['primary'])
            
            fig.update_layout(
                title_font_size=16,
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating performance chart: {str(e)}")
    
    def create_sentiment_trend_chart(self, merged_data):
        """Create sentiment trend chart with moving average"""
        try:
            # Calculate moving average
            merged_data_copy = merged_data.copy()
            merged_data_copy['sentiment_ma'] = merged_data_copy['sentiment_score'].rolling(window=7).mean()
            
            fig = go.Figure()
            
            # Sentiment score
            fig.add_trace(
                go.Scatter(
                    x=merged_data_copy['date'],
                    y=merged_data_copy['sentiment_score'],
                    mode='lines',
                    name='Sentiment Score',
                    line=dict(color=self.color_palette['secondary'], width=1),
                    opacity=0.7
                )
            )
            
            # Moving average
            fig.add_trace(
                go.Scatter(
                    x=merged_data_copy['date'],
                    y=merged_data_copy['sentiment_ma'],
                    mode='lines',
                    name='7-Day Moving Average',
                    line=dict(color=self.color_palette['primary'], width=2)
                )
            )
            
            fig.update_layout(
                title="Sentiment Score Trend with Moving Average",
                title_font_size=16,
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating sentiment trend chart: {str(e)}")
    
    def create_model_accuracy_chart(self, model_data):
        """Create model accuracy visualization"""
        try:
            fig = px.scatter(
                model_data,
                x='actual',
                y='predicted',
                title="Model Accuracy: Actual vs Predicted Prices"
            )
            
            # Add perfect prediction line
            min_val = min(model_data['actual'].min(), model_data['predicted'].min())
            max_val = max(model_data['actual'].max(), model_data['predicted'].max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color=self.color_palette['danger'])
                )
            )
            
            fig.update_layout(
                title_font_size=16,
                xaxis_title="Actual Price ($)",
                yaxis_title="Predicted Price ($)",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_chart(f"Error creating model accuracy chart: {str(e)}")
    
    def create_error_chart(self, error_message):
        """Create a chart showing an error message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Chart Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        
        return fig
