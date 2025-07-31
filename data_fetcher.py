"""
Data fetcher module for retrieving and validating stock data.
"""
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests

from config import ALPHA_VANTAGE_API_KEY

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Fetches historical and optionally real-time stock data from various sources,
    with data validation and normalization.
    """
    
    def __init__(self, primary_source="yahoo", backup_source="alpha_vantage", alpha_vantage_key=ALPHA_VANTAGE_API_KEY):
        """
        Initialize the data fetcher.
        
        Args:
            primary_source (str): Primary data source ('yahoo' or 'alpha_vantage')
            backup_source (str): Backup data source if primary fails
            alpha_vantage_key (str, optional): API key for Alpha Vantage, defaults to environment variable
        """
        self.primary_source = primary_source
        self.backup_source = backup_source
        self.alpha_vantage_key = alpha_vantage_key
        
        if self.primary_source == "alpha_vantage" and not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not found in environment variables. Defaulting to Yahoo Finance.")
            logger.info("To obtain an Alpha Vantage API key, visit: https://www.alphavantage.co/support/#api-key")
            self.primary_source = "yahoo"
    
    def fetch_stock_data(self, ticker, start_date=None, end_date=None, interval='1d'):
        """
        Fetch historical stock data for the given ticker with support for different timeframes.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (datetime or str, optional): Start date for historical data
            end_date (datetime or str, optional): End date for historical data
            interval (str, optional): Data timeframe ('1d'=daily, '1wk'=weekly, '1mo'=monthly)
            
        Returns:
            pandas.DataFrame: DataFrame with stock data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            # Adjust the default timespan based on the interval
            if interval == '1d':  # Daily data - 1 year
                days_back = 365
            elif interval == '1wk':  # Weekly data - 3 years
                days_back = 365 * 3
            elif interval == '1mo':  # Monthly data - 5 years
                days_back = 365 * 5
            else:  # Default to 1 year for unknown intervals
                days_back = 365
                
            start_date = datetime.now() - timedelta(days=days_back)
        
        # Format dates as strings if they're datetime objects
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
        
        logger.info(f"Fetching {interval} data for {ticker} from {start_str} to {end_str}")
        
        try:
            if self.primary_source == "yahoo":
                df = self._fetch_yahoo_data(ticker, start_date, end_date, interval)
            else:
                df = self._fetch_alpha_vantage_data(ticker, start_date, end_date)
        except Exception as e:
            logger.warning(f"Failed to fetch data from primary source: {str(e)}")
            try:
                if self.backup_source == "yahoo":
                    df = self._fetch_yahoo_data(ticker, start_date, end_date, interval)
                else:
                    df = self._fetch_alpha_vantage_data(ticker, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to fetch data from backup source: {str(e)}")
                raise RuntimeError(f"Could not fetch data for {ticker}") from e
        
        # Validate and clean the data
        df = self._validate_and_clean_data(df)
        
        return df
    
    def _fetch_yahoo_data(self, ticker, start_date, end_date, interval='1d'):
        """
        Fetch data from Yahoo Finance with support for different timeframes.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (datetime or str): Start date
            end_date (datetime or str): End date
            interval (str): Data interval/timeframe ('1d'=daily, '1wk'=weekly, '1mo'=monthly)
            
        Returns:
            pandas.DataFrame: DataFrame with stock data
        """
        # Add a buffer to the start date to ensure we get enough data for calculations
        # Calculate buffer based on interval
        if hasattr(start_date, 'strftime'):
            if interval == '1d':
                buffer_days = 60  # 60 days for daily data
            elif interval == '1wk':
                buffer_days = 120  # ~6 months for weekly data
            elif interval == '1mo':
                buffer_days = 240  # ~8 months for monthly data
            else:
                buffer_days = 60  # Default
                
            buffer_start = start_date - timedelta(days=buffer_days)
            start_str = buffer_start.strftime('%Y-%m-%d')
        else:
            # If start_date is already a string, use it directly
            start_str = start_date
        
        # Format end date if it's a datetime
        if hasattr(end_date, 'strftime'):
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = end_date
        
        df = yf.download(
            ticker,
            start=start_str,
            end=end_str,
            interval=interval,
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance")
        return df
    
    def _fetch_alpha_vantage_data(self, ticker, start_date, end_date):
        """
        Fetch data from Alpha Vantage.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pandas.DataFrame: DataFrame with stock data
        """
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        
        # Rename columns to match Yahoo Finance format
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        
        # Filter to the requested date range
        data = data[(data.index >= start_date.strftime('%Y-%m-%d')) & 
                     (data.index <= end_date.strftime('%Y-%m-%d'))]
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        logger.info(f"Successfully fetched {len(data)} records from Alpha Vantage")
        return data
    
    def _validate_and_clean_data(self, df):
        """
        Validate and clean the data.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            
        Returns:
            pandas.DataFrame: Clean and validated DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for and handle missing values
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            logger.warning(f"Found {null_counts} missing values in data")
            
            # Forward fill missing values (except Volume)
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].fillna(method='ffill')
            
            # For volume, fill with zeros
            df['Volume'] = df['Volume'].fillna(0)
            
            # Check if any missing values remain
            remaining_nulls = df.isnull().sum().sum()
            if remaining_nulls > 0:
                logger.warning(f"Could not fill all missing values, {remaining_nulls} remain")
                df = df.dropna()
        
        # Check for and handle duplicate dates
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate dates in data")
            df = df[~df.index.duplicated(keep='first')]
        
        # Check for and handle outliers (very simple approach)
        for col in ['Open', 'High', 'Low', 'Close']:
            mean = df[col].mean()
            std = df[col].std()
            outliers = (df[col] > mean + 5*std) | (df[col] < mean - 5*std)
            # Convert Series.sum() to scalar value to avoid Series truth value ambiguity
            # Use recommended approach to avoid FutureWarning
            outlier_count = outliers.sum()
            if isinstance(outlier_count, pd.Series):
                outlier_count = int(outlier_count.iloc[0])
            else:
                outlier_count = int(outlier_count)
                
            if outlier_count > 0:
                logger.warning(f"Found {outlier_count} potential outliers in {col}")
        
        # Sort by date
        df = df.sort_index()
        
        logger.info(f"Data validation and cleaning complete: {len(df)} clean records")
        return df
    
    def get_latest_price(self, ticker):
        """
        Get the latest available price for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Latest price information
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="1d")
            
            if hist.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            latest = hist.iloc[-1]
            return {
                'date': hist.index[-1].strftime('%Y-%m-%d'),
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'close': latest['Close'],
                'volume': latest['Volume']
            }
        except Exception as e:
            logger.error(f"Failed to get latest price: {str(e)}")
            raise
    
    def fetch_news(self, ticker, days=7, include_industry=True):
        """
        Fetch news articles related to the ticker and its industry using Alpha Vantage News API.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days to look back
            include_industry (bool): Whether to include industry news
            
        Returns:
            list: List of news articles or empty list if API key not provided
        """
        logger.info(f"Fetching news for {ticker} from the past {days} days")
        
        # Check if Alpha Vantage API key is available
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not provided. Cannot fetch news data.")
            logger.info("To fetch news, set an Alpha Vantage API key (free at https://www.alphavantage.co/support/#api-key)")
            return []
        
        # Industry keywords mapping for major tickers to enhance news retrieval
        industry_keywords = {
            # Technology
            'AAPL': ['Apple', 'iPhone', 'Mac', 'tech stocks', 'technology sector'],
            'MSFT': ['Microsoft', 'Azure', 'cloud computing', 'tech stocks'],
            'GOOGL': ['Google', 'Alphabet', 'search engine', 'tech stocks'],
            'META': ['Facebook', 'Meta', 'social media', 'tech stocks'],
            'AMZN': ['Amazon', 'e-commerce', 'cloud computing', 'retail'],
            
            # Financial
            'JPM': ['JPMorgan', 'banking', 'financial sector', 'interest rates'],
            'BAC': ['Bank of America', 'banking', 'financial sector'],
            'WFC': ['Wells Fargo', 'banking', 'financial sector'],
            'GS': ['Goldman Sachs', 'investment banking', 'financial sector'],
            'RKT': ['Rocket Companies', 'mortgage', 'housing market', 'real estate', 'interest rates', 'home loans'],
            
            # Automotive
            'TSLA': ['Tesla', 'electric vehicles', 'EV market', 'automotive sector'],
            'GM': ['General Motors', 'automotive sector', 'car manufacturing'],
            'F': ['Ford', 'automotive sector', 'car manufacturing'],
            
            # Retail
            'WMT': ['Walmart', 'retail sector', 'consumer spending'],
            'TGT': ['Target', 'retail sector', 'consumer spending'],
            
            # Energy
            'XOM': ['Exxon', 'oil prices', 'energy sector', 'petroleum'],
            'CVX': ['Chevron', 'oil prices', 'energy sector', 'petroleum'],
            
            # Default keywords for any ticker
            'DEFAULT': ['stock market', 'market trends', 'economic outlook', 'federal reserve', 'interest rates']
        }
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get specific industry keywords for this ticker
            topics = [ticker]  # Always include the ticker itself
            
            if include_industry:
                # Add industry keywords if available, otherwise use defaults
                if ticker in industry_keywords:
                    topics.extend(industry_keywords[ticker])
                else:
                    # For less common tickers, try to fetch company info to determine industry
                    try:
                        import yfinance as yf
                        company_info = yf.Ticker(ticker).info
                        if 'industry' in company_info and company_info['industry']:
                            topics.append(company_info['industry'])
                        if 'sector' in company_info and company_info['sector']:
                            topics.append(company_info['sector'])
                    except Exception as e:
                        logger.debug(f"Could not fetch additional company info: {str(e)}")
                    
                    # Add default keywords
                    topics.extend(industry_keywords['DEFAULT'])
            
            # Make multiple API requests if needed to get more comprehensive news
            all_articles = []
            
            # First, try with just the ticker for direct relevance
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_from": start_date.strftime('%Y%m%dT%H%M'),
                "limit": 50,
                "apikey": self.alpha_vantage_key
            }
            
            # Make API request
            import requests
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Process direct ticker news
            ticker_articles = []
            if 'feed' in data:
                for article in data['feed']:
                    ticker_articles.append({
                        "title": article.get('title', 'No title'),
                        "date": article.get('time_published', 'Unknown date'),
                        "source": article.get('source', 'Unknown source'),
                        "text": article.get('summary', 'No content available'),
                        "url": article.get('url', ''),
                        "sentiment": article.get('overall_sentiment_score', 0),
                        "relevance": "direct"  # Mark as directly related to ticker
                    })
            
            # If we got very few articles and industry inclusion is requested, add more
            if include_industry and len(ticker_articles) < 10:
                # Try using topics instead of tickers
                topics_str = " OR ".join(topics[:5])  # Limit to first 5 topics
                
                # Use topics API endpoint instead
                params = {
                    "function": "NEWS_SENTIMENT",
                    "topics": topics_str,
                    "time_from": start_date.strftime('%Y%m%dT%H%M'),
                    "limit": 50,
                    "apikey": self.alpha_vantage_key
                }
                
                response = requests.get(base_url, params=params)
                topic_data = response.json()
                
                # Process industry/topic news
                if 'feed' in topic_data:
                    for article in topic_data['feed']:
                        # Avoid duplicates by checking URLs
                        if not any(a['url'] == article.get('url', '') for a in ticker_articles):
                            ticker_articles.append({
                                "title": article.get('title', 'No title'),
                                "date": article.get('time_published', 'Unknown date'),
                                "source": article.get('source', 'Unknown source'),
                                "text": article.get('summary', 'No content available'),
                                "url": article.get('url', ''),
                                "sentiment": article.get('overall_sentiment_score', 0),
                                "relevance": "industry"  # Mark as industry related
                            })
            
            # Add all collected articles
            all_articles.extend(ticker_articles)
            
            logger.info(f"Successfully fetched {len(all_articles)} news articles for {ticker} and related topics")
            
            # Sort by date (newest first)
            all_articles.sort(key=lambda x: x.get('date', ''), reverse=True)
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return []
