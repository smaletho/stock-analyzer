"""
Sentiment analyzer module for processing financial news and social media data.
"""
import logging
import json
import re
from datetime import datetime, timedelta
import httpx
import pandas as pd

from config import OLLAMA_HOST

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes sentiment from financial news and social media using Llama 3 via Ollama.
    """
    
    def __init__(self, model="llama3", ollama_host=OLLAMA_HOST, timeout=30):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model (str): Model name to use for sentiment analysis
            ollama_host (str): Host URL for Ollama server, defaults to environment variable
            timeout (int): Timeout in seconds for API calls
        """
        self.model = model
        self.ollama_host = ollama_host
        self.timeout = timeout
        logger.info(f"Sentiment Analyzer initialized with model: {model}")
        logger.info(f"Using Ollama server at: {ollama_host}")
        
        # Check if Ollama is likely running
        if 'localhost' in ollama_host or '127.0.0.1' in ollama_host:
            logger.info("Remember to start Ollama locally before using sentiment analysis")
            logger.info("Install Ollama from: https://ollama.com/download")
            logger.info("Pull the model with: ollama pull llama3")
        
    
    def analyze_sentiment(self, ticker, fetch_news=True, news_data=None):
        """
        Analyze sentiment for the given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            fetch_news (bool): Whether to fetch news data
            news_data (list, optional): Pre-fetched news data
            
        Returns:
            dict: Sentiment analysis results
        """
        if fetch_news and news_data is None:
            news_data = self._fetch_financial_news(ticker)
        
        if not news_data:
            logger.warning(f"No news data available for {ticker}")
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.5,
                'analysis': [],
                'summary': f"Insufficient data to analyze sentiment for {ticker}"
            }
        
        # Create a consolidated text for overall analysis
        combined_text = self._prepare_text_for_analysis(ticker, news_data)
        
        # Get sentiment from Llama 3
        try:
            sentiment_result = self._get_sentiment_from_llm(combined_text)
            logger.info(f"Sentiment analysis completed for {ticker}")
            return sentiment_result
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.5,
                'analysis': [],
                'summary': f"Error analyzing sentiment: {str(e)}"
            }
    
    def _fetch_financial_news(self, ticker):
        """
        Fetch financial news for the given ticker using DataFetcher.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            list: List of news items
        """
        logger.info(f"Fetching news for {ticker}")
        
        try:
            # Import DataFetcher to get real news data
            from data_fetcher import DataFetcher
            
            # Create DataFetcher instance
            # Note: Alpha Vantage API key needs to be set for real news data
            data_fetcher = DataFetcher()
            
            # Fetch news data for the last 14 days
            news_data = data_fetcher.fetch_news(ticker, days=14)
            
            if not news_data:
                logger.warning(f"No news data available for {ticker}. Consider adding an Alpha Vantage API key.")
            else:
                logger.info(f"Successfully fetched {len(news_data)} news articles for {ticker}")
                
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching financial news: {str(e)}")
            return []
    
    def _prepare_text_for_analysis(self, ticker, news_data):
        """
        Prepare text for sentiment analysis.
        
        Args:
            ticker (str): Stock ticker symbol
            news_data (list): News data
            
        Returns:
            str: Prepared text
        """
        # Combine all news into a format suitable for analysis
        combined_text = f"Recent news about {ticker}:\n\n"
        
        for item in news_data:
            combined_text += f"Date: {item.get('date', 'Unknown')}\n"
            combined_text += f"Source: {item.get('source', 'Unknown')}\n"
            combined_text += f"Title: {item.get('title', 'No title')}\n"
            combined_text += f"Content: {item.get('text', 'No content')}\n\n"
        
        # Add instructions for the LLM
        prompt = f"""
Based on the news articles above about {ticker} stock, analyze the overall sentiment. 
Consider the following aspects:
1. Overall sentiment toward the stock (very bullish, bullish, neutral, bearish, very bearish)
2. Key positive factors mentioned
3. Key negative factors or risks mentioned
4. Any significant events or announcements that might affect the stock price

Provide your analysis in JSON format with the following structure:
{{
  "overall_sentiment": "bullish/bearish/neutral/etc",
  "confidence": 0.8, // from 0 to 1
  "key_factors": [
    {{ "type": "positive", "description": "factor 1" }},
    {{ "type": "negative", "description": "factor 2" }}
  ],
  "summary": "A brief 1-2 sentence summary of your analysis"
}}

Ensure your response contains ONLY valid JSON, no other text before or after.
"""
        
        return combined_text + prompt
    
    def _get_sentiment_from_llm(self, text):
        """
        Get sentiment analysis from Llama 3 via Ollama.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Call Ollama API
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": text,
                        "stream": False
                    }
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract response text
                llm_response = result.get("response", "")
                
                # Parse the JSON response
                # First, find the JSON block (in case there's extra text)
                json_match = re.search(r'({[\s\S]*?})', llm_response)
                
                if json_match:
                    json_str = json_match.group(1)
                    sentiment_data = json.loads(json_str)
                else:
                    # Fallback if no proper JSON is found
                    logger.warning("Could not extract valid JSON from LLM response")
                    sentiment_data = {
                        "overall_sentiment": "neutral",
                        "confidence": 0.5,
                        "key_factors": [],
                        "summary": "Could not properly analyze sentiment from available data."
                    }
                
                # Ensure we have the expected structure
                if "overall_sentiment" not in sentiment_data:
                    sentiment_data["overall_sentiment"] = "neutral"
                if "confidence" not in sentiment_data:
                    sentiment_data["confidence"] = 0.5
                
                return sentiment_data
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            
            # If Ollama is not available, provide fallback functionality
            logger.warning("Using fallback sentiment analysis")
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text):
        """
        Fallback sentiment analysis when Ollama is not available.
        Uses a simple keyword-based approach.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        # Simple keyword-based sentiment analysis
        positive_keywords = [
            'upgrade', 'buy', 'strong buy', 'outperform', 'positive', 'growth', 
            'profit', 'exceed', 'better than expected', 'beat', 'success', 'up'
        ]
        
        negative_keywords = [
            'downgrade', 'sell', 'strong sell', 'underperform', 'negative', 'decline',
            'loss', 'miss', 'worse than expected', 'below', 'failure', 'down'
        ]
        
        # Count occurrences
        positive_count = sum(1 for word in positive_keywords if word in text.lower())
        negative_count = sum(1 for word in negative_keywords if word in text.lower())
        
        # Calculate sentiment
        total_count = positive_count + negative_count
        if total_count == 0:
            overall_sentiment = "neutral"
            confidence = 0.5
        else:
            sentiment_score = (positive_count - negative_count) / total_count
            
            if sentiment_score > 0.5:
                overall_sentiment = "very bullish"
                confidence = 0.8
            elif sentiment_score > 0.1:
                overall_sentiment = "bullish"
                confidence = 0.7
            elif sentiment_score > -0.1:
                overall_sentiment = "neutral"
                confidence = 0.6
            elif sentiment_score > -0.5:
                overall_sentiment = "bearish"
                confidence = 0.7
            else:
                overall_sentiment = "very bearish"
                confidence = 0.8
        
        # Extract some key factors (simplified)
        key_factors = []
        
        for word in positive_keywords:
            if word in text.lower():
                key_factors.append({
                    "type": "positive",
                    "description": f"Mentioned '{word}' in the news"
                })
                if len(key_factors) >= 2:
                    break
                
        for word in negative_keywords:
            if word in text.lower():
                key_factors.append({
                    "type": "negative",
                    "description": f"Mentioned '{word}' in the news"
                })
                if len(key_factors) >= 4:
                    break
        
        return {
            "overall_sentiment": overall_sentiment,
            "confidence": confidence,
            "key_factors": key_factors,
            "summary": f"Based on keyword analysis, the sentiment appears to be {overall_sentiment} with {confidence:.1f} confidence."
        }
