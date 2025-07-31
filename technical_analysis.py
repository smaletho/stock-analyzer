"""
Technical analysis module for calculating moving averages and generating trading signals.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Performs technical analysis on stock data with customizable indicators.
    """
    
    def __init__(self, short_window=10, long_window=50):
        """
        Initialize the technical analyzer with custom parameters.
        
        Args:
            short_window (int): Short-term moving average window
            long_window (int): Long-term moving average window
        """
        self.short_window = short_window
        self.long_window = long_window
        logger.info(f"Technical Analyzer initialized with {short_window}/{long_window} day windows")
        
        # Default parameters - enhanced for short-term trading
        self.params = {
            # Moving Averages
            'short_window': short_window,
            'long_window': long_window,
            
            # RSI Parameters - shorter periods for faster response
            'rsi_period': 9,        # Shorter period for faster response (was 14)
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            
            # MACD Parameters
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2,
            
            # Volume Analysis
            'volume_window': 10,     # Shorter window for volume analysis (was 20)
            'volume_factor': 1.2,
            
            # Volatility Measures
            'atr_period': 14,
            
            # Stochastic Oscillator - new for short-term trading
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            
            # Price Level Analysis - new for short/term support/resistance
            'support_resistance_periods': 10,  # Number of days for S/R levels
            'price_target_buffer': 0.01,  # 1% buffer for price targets
            
            # Stop Loss / Take Profit - new for risk management
            'stop_loss_atr_multiple': 2.0,  # Stop loss at 2x ATR from entry
            'take_profit_atr_multiple': 3.0,  # Take profit at 3x ATR from entry
            
            # Signal Generation
            'signal_threshold': 0.8
        }
        
        # Update with custom parameters if provided
        if custom_params:
            self.params.update(custom_params)
            
        logger.info(f"Technical Analyzer initialized with {self.params['short_window']}/{self.params['long_window']} day windows")
    
    def analyze(self, df):
        """
        Enhanced technical analysis with multiple indicators and filters.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            
        Returns:
            pandas.DataFrame: DataFrame with signals and indicators
        """
        if len(df) == 0:
            logger.error("Cannot perform analysis on empty DataFrame")
            return pd.DataFrame()
        
        # Make a copy of the input DataFrame
        signals = df.copy()
        
        # Define helper function for safe column access
        def get_column(df, col_name):
            if isinstance(df.columns, pd.MultiIndex):
                for col in df.columns:
                    if isinstance(col, tuple) and col_name in col:
                        return df[col]
            return df[col_name]
        
        # Safely get Close and Volume columns
        close_vals = get_column(signals, 'Close')
        volume_vals = get_column(signals, 'Volume')
        
        # Calculate moving averages
        short_mavg = close_vals.rolling(window=self.params['short_window'], min_periods=1).mean()
        long_mavg = close_vals.rolling(window=self.params['long_window'], min_periods=1).mean()
        
        signals = self._safe_assign_column(signals, 'short_mavg', short_mavg)
        signals = self._safe_assign_column(signals, 'long_mavg', long_mavg)
        
        # Calculate additional indicators (RSI, MACD, Bollinger Bands, etc.)
        signals = self._calculate_additional_indicators(signals)
        
        # Get updated columns after calculating indicators
        short_mavg = get_column(signals, 'short_mavg')
        long_mavg = get_column(signals, 'long_mavg')
        rsi_vals = get_column(signals, 'rsi')
        macd_vals = get_column(signals, 'macd')
        macd_signal_vals = get_column(signals, 'macd_signal')
        bollinger_upper = get_column(signals, 'bollinger_upper')
        bollinger_lower = get_column(signals, 'bollinger_lower')
        bollinger_mid = get_column(signals, 'bollinger_mid')
        
        # Generate base signal from moving average crossover
        ma_signal = np.where(short_mavg > long_mavg, 1.0, 0.0)
        signals = self._safe_assign_column(signals, 'ma_signal', ma_signal)
        
        # Add RSI filter - avoid buying when overbought, avoid selling when oversold
        rsi_filter = np.ones(len(signals))
        rsi_filter[rsi_vals > self.params['rsi_overbought']] = 0.0  # Overbought - block buy signals
        rsi_filter[rsi_vals < self.params['rsi_oversold']] = 2.0  # Oversold - allow buy, block sell
        signals = self._safe_assign_column(signals, 'rsi_filter', rsi_filter)
        
        # Add volume filter - require higher than average volume for signals
        vol_avg = volume_vals.rolling(window=self.params['volume_window']).mean()
        vol_filter = np.where(volume_vals > vol_avg * self.params['volume_factor'], 1.0, 0.5)
        signals = self._safe_assign_column(signals, 'vol_filter', vol_filter)
        
        # Add MACD confirmation
        macd_filter = np.where(macd_vals > macd_signal_vals, 1.0, 0.0)
        signals = self._safe_assign_column(signals, 'macd_filter', macd_filter)
        
        # Add Bollinger Band filter
        bb_filter = np.ones(len(signals))
        band_width = (bollinger_upper - bollinger_lower) / bollinger_mid
        
        # Calculate proximity to bands
        lower_proximity = (close_vals - bollinger_lower) / (band_width * bollinger_mid + 0.0001)
        upper_proximity = (bollinger_upper - close_vals) / (band_width * bollinger_mid + 0.0001)
        
        # Apply filter rules
        bb_filter[lower_proximity < 0.1] = 1.5  # Close to lower band - amplify buy
        bb_filter[upper_proximity < 0.1] = 0.5  # Close to upper band - amplify sell
        signals = self._safe_assign_column(signals, 'bb_filter', bb_filter)
        
        # Calculate trend strength using ADX-like approach
        trend_strength = np.abs(short_mavg - long_mavg) / (long_mavg + 0.0001)  # Avoid division by zero
        signals = self._safe_assign_column(signals, 'trend_strength', trend_strength)
        
        # Get updated filter columns
        ma_signal = get_column(signals, 'ma_signal')
        rsi_filter = get_column(signals, 'rsi_filter')
        vol_filter = get_column(signals, 'vol_filter')
        bb_filter = get_column(signals, 'bb_filter')
        trend_strength = get_column(signals, 'trend_strength')
        
        # Calculate trend direction and strength - shorter term (10 days instead of 20) for faster response
        close_vals = get_column(signals, 'Close')
        
        # Calculate directional movement for trend detection (10-day direction for short-term trading)
        price_direction = np.sign(close_vals.pct_change(10).fillna(0))
        signals = self._safe_assign_column(signals, 'price_direction', price_direction)
        
        # Adaptive trend strength - stronger requirement in bear markets
        adx_threshold = np.where(price_direction > 0, 0.02, 0.03)  # Higher in bear markets
        trend_confirmed = np.where(trend_strength > adx_threshold, 1.0, 0.5)
        signals = self._safe_assign_column(signals, 'trend_confirmed', trend_confirmed)
        
        # Volatility-based filtering - require stronger signals in high volatility
        volatility = get_column(signals, 'atr_percent')
        vol_threshold = volatility.rolling(window=10).mean() * 1.2
        vol_filter_adaptive = np.where(volatility < vol_threshold, 1.0, 0.7)
        signals = self._safe_assign_column(signals, 'vol_filter_adaptive', vol_filter_adaptive)
        
        # Market regime detection (bull/bear/sideways)
        bull_market = price_direction > 0
        signals = self._safe_assign_column(signals, 'bull_market', bull_market.astype(float))
        
        # Get stochastic oscillator values for enhanced short-term signals
        stoch_k = get_column(signals, 'stoch_k')
        stoch_d = get_column(signals, 'stoch_d')
        
        # Stochastic crossover signal (K crossing above D = bullish, K crossing below D = bearish)
        stoch_above = stoch_k > stoch_d
        stoch_above_prev = stoch_above.shift(1).fillna(False)
        stoch_crossover_up = (stoch_above) & (~stoch_above_prev)  # K crossing above D
        stoch_crossover_down = (~stoch_above) & (stoch_above_prev)  # K crossing below D
        
        # Stochastic overbought/oversold filter
        stoch_filter = np.ones(len(signals))
        # Amplify buy signals when oversold (stoch_k < 20) and stoch_k crossing up through stoch_d
        stoch_filter[(stoch_k < self.params['stoch_oversold']) & stoch_crossover_up] = 1.5
        # Amplify sell signals when overbought (stoch_k > 80) and stoch_k crossing down through stoch_d
        stoch_filter[(stoch_k > self.params['stoch_overbought']) & stoch_crossover_down] = 0.5
        signals = self._safe_assign_column(signals, 'stoch_filter', stoch_filter)
        
        # Enhanced signal strength calculation - including stochastics for short-term responsiveness
        signal_strength = (
            ma_signal * 
            rsi_filter * 
            vol_filter * 
            bb_filter * 
            trend_confirmed * 
            vol_filter_adaptive * 
            stoch_filter *  # Added stochastic filter
            (1.0 + trend_strength * 1.5)  # Reduced from 2.0 to balance with stochastics
        )
        
        # Adjust threshold based on market regime and price position within Bollinger Bands
        percent_b = get_column(signals, 'percent_b')
        
        # More nuanced adaptive threshold for short-term trading:
        # - Lower threshold in bull markets (easier to enter)
        # - Lower threshold when price is near lower band (percent_b < 0.2) 
        # - Higher threshold when price is near upper band (percent_b > 0.8)
        base_threshold = self.params['signal_threshold']
        market_adj = np.where(bull_market, 0.9, 1.1)  # Market regime adjustment
        bb_adj = np.ones(len(signals))  # Bollinger Band position adjustment
        bb_adj[percent_b < 0.2] = 0.8  # Easier to buy when near lower band
        bb_adj[percent_b > 0.8] = 1.2  # Harder to buy when near upper band
        
        adaptive_threshold = base_threshold * market_adj * bb_adj
        
        signals = self._safe_assign_column(signals, 'signal_strength', signal_strength)
        signals = self._safe_assign_column(signals, 'adaptive_threshold', adaptive_threshold)
        
        # Generate final signal based on adaptive threshold
        signal = np.where(signal_strength > adaptive_threshold, 1.0, 0.0)
        signals = self._safe_assign_column(signals, 'signal', signal)
        
        # Calculate position changes (1 for buy, -1 for sell, 0 for hold)
        # Convert numpy array to pandas Series before calling diff()
        signal_series = pd.Series(signal, index=signals.index)
        position = signal_series.diff().fillna(0)
        signals = self._safe_assign_column(signals, 'position', position)
        
        # Calculate support and resistance levels
        min_price = close_vals.rolling(window=20).min()
        support_level = np.minimum(bollinger_lower, min_price)
        signals = self._safe_assign_column(signals, 'support_level', support_level)
        
        max_price = close_vals.rolling(window=20).max()
        resistance_level = np.maximum(bollinger_upper, max_price)
        signals = self._safe_assign_column(signals, 'resistance_level', resistance_level)
        
        # Calculate price targets
        current_close = close_vals.iloc[-1] if len(signals) > 0 else 0
        buy_target = support_level * 1.01  # 1% above support
        sell_target = resistance_level * 0.99  # 1% below resistance
        signals = self._safe_assign_column(signals, 'buy_target', buy_target)
        signals = self._safe_assign_column(signals, 'sell_target', sell_target)
        
        # Count actual signals generated using safe column access
        try:
            position_vals = get_column(signals, 'position')
            signal_count = (position_vals != 0).sum() if isinstance(position_vals, pd.Series) else 0
        except Exception as e:
            logger.warning(f"Error counting signals: {str(e)}")
            signal_count = 0
            
        logger.info(f"Technical analysis completed, {signal_count} signals generated")
        return signals
    
    def _calculate_additional_indicators(self, df):
        """
        Calculate enhanced technical indicators using customizable parameters.
        Enhanced for short-term trading with stochastics, stop-loss, and take-profit.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            
        Returns:
            pandas.DataFrame: DataFrame with added indicators
        """
        try:
            # Extract parameters for cleaner code - using proper parameter names
            rsi_period = self.params['rsi_period']
            bb_period = self.params['bb_period']
            bb_std = self.params['bb_std']
            macd_fast = self.params['macd_fast']
            macd_slow = self.params['macd_slow']
            macd_signal = self.params['macd_signal']
            volume_window = self.params['volume_window']
            atr_period = self.params['atr_period']
            stoch_k_period = self.params['stoch_k_period']
            stoch_d_period = self.params['stoch_d_period']
            stop_loss_atr = self.params['stop_loss_atr_multiple']
            take_profit_atr = self.params['take_profit_atr_multiple']
            
            # Helper function to safely get column values from potentially MultiIndex DataFrame
            def get_column(df, col_name):
                if isinstance(df.columns, pd.MultiIndex):
                    for col in df.columns:
                        if isinstance(col, tuple) and col_name in col:
                            return df[col]
                return df[col_name]
            
            # Safely get Close price data for calculations
            close_vals = get_column(df, 'Close')
            
            # Calculate RSI (Relative Strength Index) with shorter period for short-term trading
            delta = close_vals.diff()
            
            # Using boolean indexing with explicit comparisons to avoid Series truth value ambiguity
            gain_mask = delta > 0
            gain = delta.copy()
            gain[~gain_mask] = 0  # Set all non-positive values to 0
            gain = gain.rolling(window=rsi_period).mean()  # Using shorter period parameter
            
            loss_mask = delta < 0
            loss = (-delta).copy()
            loss[~loss_mask] = 0  # Set all non-negative values to 0
            loss = loss.rolling(window=rsi_period).mean()  # Using shorter period parameter
            
            # Avoid division by zero
            loss_zero_mask = loss == 0
            loss[loss_zero_mask] = np.nan  # Replace zeros with NaN to avoid division by zero
            rs = gain / loss
            
            # Calculate RSI and safely assign to DataFrame
            rsi_vals = 100 - (100 / (1 + rs))
            rsi_vals = rsi_vals.fillna(50)  # Fill initial NaN values
            df = self._safe_assign_column(df, 'rsi', rsi_vals)
            
            # Calculate Stochastic Oscillator
            low14 = close_vals.rolling(window=14).min()
            high14 = close_vals.rolling(window=14).max()
            k = ((close_vals - low14) / (high14 - low14)) * 100
            d = k.rolling(window=3).mean()
            df = self._safe_assign_column(df, 'stoch_k', k)
            df = self._safe_assign_column(df, 'stoch_d', d)
            
            # Calculate MACD (Moving Average Convergence Divergence)
            exp1 = close_vals.ewm(span=macd_fast, adjust=False).mean()
            exp2 = close_vals.ewm(span=macd_slow, adjust=False).mean()
            macd_vals = exp1 - exp2
            macd_signal_vals = macd_vals.ewm(span=macd_signal, adjust=False).mean()
            macd_hist_vals = macd_vals - macd_signal_vals
            
            # Safely assign MACD values to DataFrame
            df = self._safe_assign_column(df, 'macd', macd_vals)
            df = self._safe_assign_column(df, 'macd_signal', macd_signal_vals)
            df = self._safe_assign_column(df, 'macd_hist', macd_hist_vals)
            
            # Calculate MACD histogram momentum
            macd_momentum_vals = macd_hist_vals.diff().rolling(window=3).mean()
            df = self._safe_assign_column(df, 'macd_momentum', macd_momentum_vals)
            
            # Calculate Bollinger Bands with customizable parameters
            bb_mid = close_vals.rolling(window=bb_period).mean()  # Updated parameter name
            bb_std_vals = close_vals.rolling(window=bb_period).std()
            bb_upper = bb_mid + (bb_std_vals * bb_std)
            bb_lower = bb_mid - (bb_std_vals * bb_std)
            
            # Safely assign Bollinger Band values to DataFrame
            df = self._safe_assign_column(df, 'bollinger_mid', bb_mid)
            df = self._safe_assign_column(df, 'bollinger_std', bb_std_vals)
            df = self._safe_assign_column(df, 'bollinger_upper', bb_upper)
            df = self._safe_assign_column(df, 'bollinger_lower', bb_lower)
            
            # Calculate Bollinger Band width - indicator of volatility
            bb_width_vals = (bb_upper - bb_lower) / bb_mid
            df = self._safe_assign_column(df, 'bb_width', bb_width_vals)
            
            # Calculate %B indicator (position within Bollinger Bands from 0-1)
            # 1.0 = at upper band, 0.5 = at middle band, 0.0 = at lower band
            percent_b = (close_vals - bb_lower) / (bb_upper - bb_lower + 1e-6)  # Avoid division by zero
            df = self._safe_assign_column(df, 'percent_b', percent_b)
            
            # Calculate Average True Range (ATR) - measure of volatility
            # Safely extract price data for ATR calculation
            high_vals = get_column(df, 'High')
            low_vals = get_column(df, 'Low')
            
            high_low = high_vals - low_vals
            high_close = (high_vals - close_vals.shift()).abs()
            low_close = (low_vals - close_vals.shift()).abs()
            
            # Combine and calculate true range
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # Add ATR to DataFrame (safely handling MultiIndex) with proper parameter
            atr_vals = true_range.rolling(window=atr_period).mean()
            df = self._safe_assign_column(df, 'atr', atr_vals)
            
            # Calculate ATR percent and add to DataFrame
            atr_percent = atr_vals / close_vals * 100
            df = self._safe_assign_column(df, 'atr_percent', atr_percent)
            
            # Calculate stop-loss and take-profit levels for short-term trading
            # For long positions
            stop_loss_long = close_vals - (atr_vals * stop_loss_atr)
            take_profit_long = close_vals + (atr_vals * take_profit_atr)
            df = self._safe_assign_column(df, 'stop_loss_long', stop_loss_long)
            df = self._safe_assign_column(df, 'take_profit_long', take_profit_long)
            
            # For short positions
            stop_loss_short = close_vals + (atr_vals * stop_loss_atr)
            take_profit_short = close_vals - (atr_vals * take_profit_atr)
            df = self._safe_assign_column(df, 'stop_loss_short', stop_loss_short)
            df = self._safe_assign_column(df, 'take_profit_short', take_profit_short)
            
            # Enhanced volume analysis - safely handle MultiIndex columns
            volume_vals = get_column(df, 'Volume')
            volume_sma = volume_vals.rolling(window=volume_window).mean()
            df = self._safe_assign_column(df, 'volume_sma', volume_sma)
            df = self._safe_assign_column(df, 'volume_ratio', volume_vals / volume_sma)
            
            # Price momentum indicators - safely handle MultiIndex columns
            df = self._safe_assign_column(df, 'momentum_1d', close_vals.pct_change(periods=1))
            df = self._safe_assign_column(df, 'momentum_5d', close_vals.pct_change(periods=5))
            df = self._safe_assign_column(df, 'momentum_10d', close_vals.pct_change(periods=10))
            
            # Volatility ratio - safely handle MultiIndex columns
            short_vol = close_vals.rolling(window=5).std()
            long_vol = close_vals.rolling(window=20).std()
            df = self._safe_assign_column(df, 'volatility_ratio', short_vol / long_vol)
            
            # Fill any remaining NaN values
            # Use bfill() instead of fillna(method='bfill') to avoid FutureWarning
            df.bfill(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _safe_assign_column(self, df, column_name, values):
        """
        Safely assign values to a column in a DataFrame, handling MultiIndex columns.
        
        Args:
            df (pandas.DataFrame): The DataFrame to modify
            column_name (str): The name of the column to assign to
            values (array-like): The values to assign
            
        Returns:
            pandas.DataFrame: The modified DataFrame
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if we're dealing with MultiIndex columns
        if isinstance(result_df.columns, pd.MultiIndex):
            # Check if column already exists in MultiIndex
            col_exists = False
            for col in result_df.columns:
                if isinstance(col, tuple) and column_name in col:
                    result_df[col] = values
                    col_exists = True
                    break
            
            # If column doesn't exist, create it with an empty tuple for the second level
            if not col_exists:
                result_df[column_name, ''] = values
        else:
            # Standard columns - direct assignment
            result_df[column_name] = values
            
        return result_df
        
    def combine_signals_with_sentiment(self, technical_signal, sentiment_result):
        """
        Combine technical signals with sentiment analysis to produce a final recommendation.
        
        Args:
            technical_signal (pandas.Series): Latest technical signal
            sentiment_result (dict): Sentiment analysis results
            
        Returns:
            dict: Combined signal analysis
        """
        # Extract the technical signal (-1, 0, 1)
        tech_signal = 0
        
        # Handle potential MultiIndex for the 'position' column
        try:
            # Extract position value safely, accounting for potential MultiIndex
            position_key = None
            if isinstance(technical_signal.index, pd.MultiIndex) or isinstance(technical_signal, pd.DataFrame):
                # For MultiIndex Series or DataFrame
                for key in technical_signal.index if isinstance(technical_signal.index, pd.MultiIndex) else technical_signal.keys():
                    if isinstance(key, tuple) and 'position' in key:
                        position_key = key
                        break
                    elif key == 'position':
                        position_key = key
                        break
                
                if position_key is not None:
                    position_value = technical_signal[position_key]
                else:
                    # Try direct access for DataFrame with simple columns
                    position_value = technical_signal['position']
            else:
                # Simple Series with standard index
                position_value = technical_signal['position']
                
            # Convert to float to handle any remaining Series objects
            if hasattr(position_value, 'iloc'):
                position_value = float(position_value.iloc[0])
            else:
                position_value = float(position_value)
                
            # Determine signal based on position value
            if position_value > 0:
                tech_signal = 1  # Buy signal
            elif position_value < 0:
                tech_signal = -1  # Sell signal
        except Exception as e:
            logger.warning(f"Error extracting position value: {str(e)}. Using neutral technical signal.")
            tech_signal = 0  # Default to neutral if there's an error
            
        # Convert sentiment to a numeric value
        sentiment_map = {
            'very bullish': 2,
            'bullish': 1,
            'neutral': 0,
            'bearish': -1,
            'very bearish': -2
        }
        
        sentiment_score = sentiment_map.get(sentiment_result['overall_sentiment'].lower(), 0)
        
        # Apply weighting (default: 70% technical, 30% sentiment)
        tech_weight = 0.7
        sentiment_weight = 0.3
        
        # Calculate weighted score
        weighted_score = (tech_signal * tech_weight) + (sentiment_score * sentiment_weight)
        
        # Determine final signal
        if weighted_score >= 0.5:
            final_signal = 'STRONG BUY'
        elif weighted_score > 0:
            final_signal = 'BUY'
        elif weighted_score > -0.5:
            final_signal = 'HOLD'
        elif weighted_score > -1:
            final_signal = 'SELL'
        else:
            final_signal = 'STRONG SELL'
        
        # Prepare and return the combined analysis
        return {
            'final_signal': final_signal,
            'technical_score': tech_signal,
            'sentiment_score': sentiment_score,
            'weighted_score': weighted_score,
            'confidence': abs(weighted_score) * 2,  # Scale to 0-2 range
            'explanation': self._generate_signal_explanation(technical_signal, sentiment_result, final_signal)
        }
    
    def _generate_signal_explanation(self, technical_signal, sentiment_result, final_signal):
        """
        Generate a human-readable explanation for the signal.
        
        Args:
            technical_signal (pandas.Series): Technical signal data
            sentiment_result (dict): Sentiment analysis results
            final_signal (str): Final signal determination
            
        Returns:
            str: Explanation of the trading signal
        """
        explanation = []
        
        # Helper function to safely extract values from technical_signal
        def get_value(column_name):
            try:
                # Handle MultiIndex columns
                if isinstance(technical_signal.index, pd.MultiIndex) or isinstance(technical_signal, pd.DataFrame):
                    # Look for the column in MultiIndex or DataFrame columns
                    col_key = None
                    for key in technical_signal.index if isinstance(technical_signal.index, pd.MultiIndex) else technical_signal.keys():
                        if isinstance(key, tuple) and column_name in key:
                            col_key = key
                            break
                        elif key == column_name:
                            col_key = key
                            break
                    
                    if col_key is not None:
                        value = technical_signal[col_key]
                    else:
                        # Try direct access for DataFrame with simple columns
                        value = technical_signal[column_name]
                else:
                    # Simple Series with standard index
                    value = technical_signal[column_name]
                    
                # Convert to float to handle any remaining Series objects
                if hasattr(value, 'iloc'):
                    return float(value.iloc[0])
                else:
                    return float(value)
            except Exception as e:
                logger.warning(f"Error extracting {column_name} value: {str(e)}")
                return None
        
        # Get short and long moving averages
        short_mavg = get_value('short_mavg')
        long_mavg = get_value('long_mavg')
        
        # Explain the technical component
        if short_mavg is not None and long_mavg is not None:
            if short_mavg > long_mavg:
                explanation.append(
                    f"The short-term moving average (${short_mavg:.2f}) is above the "
                    f"long-term moving average (${long_mavg:.2f}), indicating an upward trend."
                )
            else:
                explanation.append(
                    f"The short-term moving average (${short_mavg:.2f}) is below the "
                    f"long-term moving average (${long_mavg:.2f}), indicating a downward trend."
                )
        else:
            explanation.append("Unable to determine trend from moving averages.")
        
        # Add RSI context if available
        rsi = get_value('rsi')
        if rsi is not None:
            if rsi > 70:
                explanation.append(f"RSI is high at {rsi:.1f}, suggesting the stock may be overbought.")
            elif rsi < 30:
                explanation.append(f"RSI is low at {rsi:.1f}, suggesting the stock may be oversold.")
            else:
                explanation.append(f"RSI is neutral at {rsi:.1f}.")
        else:
            explanation.append("RSI data not available.")
        
        # Explain the sentiment component
        explanation.append(
            f"Market sentiment is {sentiment_result['overall_sentiment']} based on recent news and social media analysis."
        )
        
        # Explain the final decision
        explanation.append(f"Combined analysis suggests a '{final_signal}' signal.")
        
        return " ".join(explanation)
