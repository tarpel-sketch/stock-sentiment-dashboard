import pandas as pd
import numpy as np
from textblob import TextBlob
import re

class SentimentAnalyzer:
    def __init__(self):
        pass
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        try:
            # Convert to string
            text = str(text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Convert to lowercase and strip whitespace
            text = text.lower().strip()
            
            return text
            
        except Exception as e:
            return ""
    
    def analyze_sentiment(self, social_media_data):
        """Analyze sentiment of social media text data"""
        try:
            sentiment_data = social_media_data.copy()
            
            # Clean text data
            sentiment_data['cleaned_text'] = sentiment_data['text'].apply(self.clean_text)
            
            # Remove empty texts
            sentiment_data = sentiment_data[sentiment_data['cleaned_text'].str.len() > 0]
            
            # Analyze sentiment using TextBlob
            sentiment_scores = []
            sentiment_labels = []
            
            for text in sentiment_data['cleaned_text']:
                try:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    
                    # Normalize polarity to 0-1 range
                    normalized_score = (polarity + 1) / 2
                    sentiment_scores.append(normalized_score)
                    
                    # Categorize sentiment
                    if polarity > 0.1:
                        label = 'Positive'
                    elif polarity < -0.1:
                        label = 'Negative'
                    else:
                        label = 'Neutral'
                    
                    sentiment_labels.append(label)
                    
                except Exception as e:
                    # Default to neutral if analysis fails
                    sentiment_scores.append(0.5)
                    sentiment_labels.append('Neutral')
            
            # Add sentiment analysis results
            sentiment_data['sentiment_score'] = sentiment_scores
            sentiment_data['sentiment_label'] = sentiment_labels
            
            return sentiment_data
            
        except Exception as e:
            raise Exception(f"Sentiment analysis error: {str(e)}")
    
    def generate_marketing_insights(self, merged_data):
        """Generate marketing insights based on sentiment analysis"""
        try:
            insights = []
            
            # Overall sentiment analysis
            avg_sentiment = merged_data['sentiment_score'].mean()
            sentiment_counts = merged_data['sentiment_label'].value_counts()
            
            # Insight 1: Overall sentiment
            if avg_sentiment > 0.6:
                insights.append(
                    f"Strong positive sentiment detected (avg: {avg_sentiment:.2f}). "
                    "This is an excellent time for promotional campaigns and product launches."
                )
            elif avg_sentiment < 0.4:
                insights.append(
                    f"Negative sentiment trend observed (avg: {avg_sentiment:.2f}). "
                    "Focus on reputation management and addressing customer concerns."
                )
            else:
                insights.append(
                    f"Neutral sentiment environment (avg: {avg_sentiment:.2f}). "
                    "Opportunity to influence opinion through targeted marketing efforts."
                )
            
            # Insight 2: Sentiment distribution
            positive_pct = (sentiment_counts.get('Positive', 0) / len(merged_data)) * 100
            negative_pct = (sentiment_counts.get('Negative', 0) / len(merged_data)) * 100
            
            if positive_pct > 50:
                insights.append(
                    f"{positive_pct:.1f}% positive sentiment - leverage user-generated content "
                    "and testimonials in marketing campaigns."
                )
            
            if negative_pct > 30:
                insights.append(
                    f"{negative_pct:.1f}% negative sentiment detected - implement crisis "
                    "communication strategy and address key pain points."
                )
            
            # Insight 3: Sentiment vs Price correlation
            correlation = merged_data['sentiment_score'].corr(merged_data['close'])
            
            if correlation > 0.3:
                insights.append(
                    f"Strong positive correlation ({correlation:.2f}) between sentiment and stock price. "
                    "Sentiment-driven marketing campaigns may directly impact valuation."
                )
            elif correlation < -0.3:
                insights.append(
                    f"Negative correlation ({correlation:.2f}) between sentiment and stock price. "
                    "Market may be contrarian - investigate underlying factors."
                )
            
            # Insight 4: Sentiment volatility
            sentiment_volatility = merged_data['sentiment_score'].std()
            
            if sentiment_volatility > 0.25:
                insights.append(
                    f"High sentiment volatility ({sentiment_volatility:.2f}) indicates unstable "
                    "public opinion. Implement consistent messaging strategy."
                )
            
            # Insight 5: Recent sentiment trends
            if len(merged_data) >= 7:
                recent_sentiment = merged_data.tail(7)['sentiment_score'].mean()
                earlier_sentiment = merged_data.head(7)['sentiment_score'].mean()
                
                sentiment_change = recent_sentiment - earlier_sentiment
                
                if sentiment_change > 0.1:
                    insights.append(
                        f"Improving sentiment trend (+{sentiment_change:.2f}). "
                        "Capitalize on positive momentum with increased marketing spend."
                    )
                elif sentiment_change < -0.1:
                    insights.append(
                        f"Declining sentiment trend ({sentiment_change:.2f}). "
                        "Implement damage control and reputation recovery strategies."
                    )
            
            # Insight 6: Best performing sentiment days
            top_sentiment_days = merged_data.nlargest(5, 'sentiment_score')
            if len(top_sentiment_days) > 0:
                avg_price_on_high_sentiment = top_sentiment_days['close'].mean()
                overall_avg_price = merged_data['close'].mean()
                
                if avg_price_on_high_sentiment > overall_avg_price * 1.05:
                    insights.append(
                        "High sentiment days correlate with higher stock prices. "
                        "Focus marketing efforts on creating positive sentiment spikes."
                    )
            
            return insights
            
        except Exception as e:
            return [f"Error generating marketing insights: {str(e)}"]
    
    def get_sentiment_summary(self, sentiment_data):
        """Get a summary of sentiment analysis results"""
        try:
            summary = {
                'total_entries': len(sentiment_data),
                'avg_sentiment_score': sentiment_data['sentiment_score'].mean(),
                'sentiment_distribution': sentiment_data['sentiment_label'].value_counts().to_dict(),
                'sentiment_volatility': sentiment_data['sentiment_score'].std(),
                'most_positive_day': sentiment_data.loc[sentiment_data['sentiment_score'].idxmax(), 'date'],
                'most_negative_day': sentiment_data.loc[sentiment_data['sentiment_score'].idxmin(), 'date']
            }
            
            return summary
            
        except Exception as e:
            return {'error': f"Summary generation error: {str(e)}"}
