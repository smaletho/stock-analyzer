"""
Backtester module for evaluating trading strategy performance.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtests a trading strategy using historical data.
    """
    
    def __init__(self, initial_capital=10000, position_size=1.0):
        """
        Initialize the backtester.
        
        Args:
            initial_capital (float): Initial capital for backtesting
            position_size (float): Position size as a fraction of portfolio (0-1)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        logger.info(f"Backtester initialized with ${initial_capital} capital, {position_size*100}% position size")
    
    def run_backtest(self, signals_df, commission=0.001):
        """
        Run a backtest on the given signals dataframe.
        
        Args:
            signals_df (pandas.DataFrame): DataFrame with price data and signals
            commission (float): Commission rate as a fraction of trade value
            
        Returns:
            dict: Backtest results
        """
        try:
            # Check if DataFrame is empty
            if len(signals_df) == 0:
                logger.error("Cannot run backtest on empty DataFrame")
                return {
                    'success': False,
                    'error': 'Empty signals DataFrame'
                }
            
            # Make a copy to avoid modifying the original
            backtest_df = signals_df.copy()
            
            # Initialize portfolio columns with explicit float dtypes
            backtest_df['holdings'] = 0.0
            backtest_df['cash'] = float(self.initial_capital)
            backtest_df['portfolio'] = float(self.initial_capital)
            backtest_df['returns'] = 0.0
            
            # Add price returns for benchmark
            try:
                backtest_df['price_returns'] = backtest_df['Close'].pct_change()
            except Exception as e:
                logger.warning(f"Error calculating price returns: {str(e)}")
            
            # Helper function to safely get values from DataFrame
            def safe_get(df, idx, col):
                try:
                    # Handle MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        for column in df.columns:
                            if isinstance(column, tuple) and col in column:
                                value = df[column].iloc[idx]
                                if hasattr(value, 'iloc'):
                                    return float(value.iloc[0])
                                return float(value)
                    
                    # Handle standard columns
                    value = df[col].iloc[idx]
                    if hasattr(value, 'iloc'):
                        return float(value.iloc[0])
                    return float(value)
                except Exception as e:
                    logger.debug(f"Error getting value for {col} at index {idx}: {str(e)}")
                    return 0.0
            
            # Execute trading simulation
            for i in range(1, len(backtest_df)):
                try:
                    # Get previous position and cash
                    prev_holdings = safe_get(backtest_df, i-1, 'holdings')
                    prev_cash = safe_get(backtest_df, i-1, 'cash')
                    
                    # Get current price and signal
                    curr_price = safe_get(backtest_df, i, 'Close')
                    position_signal = safe_get(backtest_df, i, 'position')
                    
                    # Get additional signal metrics for adaptive position sizing
                    signal_strength = safe_get(backtest_df, i, 'signal_strength')
                    atr_pct = safe_get(backtest_df, i, 'atr_percent')
                    bull_market = safe_get(backtest_df, i, 'bull_market')
                    
                    # Calculate adaptive position size based on volatility and market regime
                    # Lower position size in high volatility environments
                    base_position_size = self.position_size
                    
                    # Volatility adjustment - reduce size in high volatility
                    vol_factor = min(1.0, 5.0 / max(atr_pct, 0.5))  # Normalize ATR
                    
                    # Market regime adjustment - more aggressive in bull markets
                    regime_factor = 1.0 + (0.2 * bull_market)
                    
                    # Signal strength adjustment - scale by signal confidence
                    signal_factor = min(1.0, signal_strength / 1.5)  # Normalize to max 1.0
                    
                    # Calculate final position size with limits
                    adaptive_position = base_position_size * vol_factor * regime_factor * signal_factor
                    adaptive_position = min(1.0, max(0.1, adaptive_position))  # Limit between 10-100%
                    
                    # Log position sizing factors for debugging
                    logger.debug(f"Position sizing - vol:{vol_factor:.2f}, regime:{regime_factor:.2f}, "
                                f"signal:{signal_factor:.2f}, final:{adaptive_position:.2f}")
                    
                    # Get stop-loss and take-profit levels for risk management
                    stop_loss_long = safe_get(backtest_df, i-1, 'stop_loss_long') if prev_holdings > 0 else None
                    take_profit_long = safe_get(backtest_df, i-1, 'take_profit_long') if prev_holdings > 0 else None
                    entry_price = safe_get(backtest_df, i-1, 'entry_price') if prev_holdings > 0 else None
                    
                    # Check if stop-loss or take-profit was hit during this period
                    stop_loss_hit = False
                    take_profit_hit = False
                    
                    # For existing long positions, check if stop-loss or take-profit was hit
                    if prev_holdings > 0 and stop_loss_long is not None and entry_price is not None:
                        # Check if price went below stop-loss at any point
                        low_price = safe_get(backtest_df, i, 'Low')
                        if low_price <= stop_loss_long:
                            stop_loss_hit = True
                            logger.debug(f"Stop loss triggered at {stop_loss_long:.2f} (entry: {entry_price:.2f})")
                        
                        # Check if price went above take-profit at any point
                        high_price = safe_get(backtest_df, i, 'High')
                        if high_price >= take_profit_long:
                            take_profit_hit = True
                            logger.debug(f"Take profit triggered at {take_profit_long:.2f} (entry: {entry_price:.2f})")
                    
                    # Handle stop-loss or take-profit hits before regular signal processing
                    if stop_loss_hit or take_profit_hit:
                        # Calculate exit price (we'll use stop-loss or take-profit level as the exit price)
                        exit_price = stop_loss_long if stop_loss_hit else take_profit_long
                        
                        # Exit the position at the calculated price
                        exit_value = prev_holdings * exit_price * (1 - commission)
                        backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash + exit_value)
                        backtest_df.loc[backtest_df.index[i], 'holdings'] = 0.0
                        
                        # Record the type of exit for analysis
                        exit_type = 'stop_loss' if stop_loss_hit else 'take_profit'
                        backtest_df.loc[backtest_df.index[i], 'exit_type'] = exit_type
                        
                        # Calculate and record P&L for this trade
                        if entry_price is not None:
                            pnl_pct = ((exit_price / entry_price) - 1) * 100
                            backtest_df.loc[backtest_df.index[i], 'trade_pnl'] = pnl_pct
                            logger.debug(f"Trade closed via {exit_type} with {pnl_pct:.2f}% P&L")
                    
                    # If no stop/take-profit hit, process regular signals
                    elif position_signal > 0:  # Buy signal
                        max_shares = (prev_cash * adaptive_position) / (curr_price * (1 + commission))
                        backtest_df.loc[backtest_df.index[i], 'holdings'] = float(max_shares)
                        backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash - (max_shares * curr_price * (1 + commission)))
                        
                        # Record entry price for future stop-loss/take-profit calculations
                        backtest_df.loc[backtest_df.index[i], 'entry_price'] = float(curr_price)
                        
                    elif position_signal < 0:  # Sell signal
                        exit_value = prev_holdings * curr_price * (1 - commission)
                        backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash + exit_value)
                        backtest_df.loc[backtest_df.index[i], 'holdings'] = 0.0
                        
                        # Record the exit type
                        backtest_df.loc[backtest_df.index[i], 'exit_type'] = 'signal'
                        
                        # Calculate and record P&L if we have entry price
                        if entry_price is not None:
                            pnl_pct = ((curr_price / entry_price) - 1) * 100
                            backtest_df.loc[backtest_df.index[i], 'trade_pnl'] = pnl_pct
                            logger.debug(f"Trade closed via signal with {pnl_pct:.2f}% P&L")
                    
                    else:  # Hold position
                        backtest_df.loc[backtest_df.index[i], 'holdings'] = float(prev_holdings)
                        backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash)
                        
                        # Carry forward entry price if we're holding
                        if prev_holdings > 0 and entry_price is not None:
                            backtest_df.loc[backtest_df.index[i], 'entry_price'] = entry_price
                    
                    # Calculate portfolio value
                    holdings_value = safe_get(backtest_df, i, 'holdings') * curr_price
                    cash_value = safe_get(backtest_df, i, 'cash')
                    portfolio_value = holdings_value + cash_value
                    backtest_df.loc[backtest_df.index[i], 'portfolio'] = float(portfolio_value)
                    
                    # Calculate return
                    prev_portfolio = safe_get(backtest_df, i-1, 'portfolio')
                    if prev_portfolio > 0:
                        return_value = (portfolio_value / prev_portfolio) - 1
                        backtest_df.loc[backtest_df.index[i], 'returns'] = float(return_value)
                except Exception as e:
                    logger.warning(f"Error in iteration {i}: {str(e)}")
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(backtest_df)
            
            # Count signals
            signal_count = 0
            for i in range(len(backtest_df)):
                try:
                    pos = safe_get(backtest_df, i, 'position')
                    if pos != 0:
                        signal_count += 1
                except Exception:
                    pass
            
            # Prepare final portfolio value and returns
            final_portfolio = safe_get(backtest_df, -1, 'portfolio')
            initial_close = safe_get(backtest_df, 0, 'Close')
            final_close = safe_get(backtest_df, -1, 'Close')
            
            total_return = (final_portfolio / self.initial_capital) - 1
            buy_hold_return = (final_close / initial_close) - 1 if initial_close > 0 else 0
            
            # Prepare results
            results = {
                'success': True,
                'signals_count': signal_count,
                'final_portfolio': final_portfolio,
                'total_return': total_return,
                'buy_and_hold_return': buy_hold_return,
                'metrics': metrics,
                'backtest_df': backtest_df,
                'summary': f"Strategy returned {metrics['total_return_pct']:.2f}% vs buy-and-hold {metrics['buy_hold_return_pct']:.2f}%"
            }
            
            logger.info(f"Backtest completed: {results['summary']}")
            return results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            return {
                'success': False,
        
        # Handle standard columns
        value = df[col].iloc[idx]
        if hasattr(value, 'iloc'):
            return float(value.iloc[0])
        return float(value)
    except Exception as e:
        logger.debug(f"Error getting value for {col} at index {idx}: {str(e)}")
        return 0.0
            
# Execute trading simulation
for i in range(1, len(backtest_df)):
    try:
        # Get previous position and cash
        prev_holdings = safe_get(backtest_df, i-1, 'holdings')
        prev_cash = safe_get(backtest_df, i-1, 'cash')
        
        # Get current price and signal
        curr_price = safe_get(backtest_df, i, 'Close')
        position_signal = safe_get(backtest_df, i, 'position')
        
        # Get additional signal metrics for adaptive position sizing
        signal_strength = safe_get(backtest_df, i, 'signal_strength')
        atr_pct = safe_get(backtest_df, i, 'atr_percent')
        bull_market = safe_get(backtest_df, i, 'bull_market')
        
        # Calculate adaptive position size based on volatility and market regime
        # Lower position size in high volatility environments
        base_position_size = self.position_size
        
        # Volatility adjustment - reduce size in high volatility
        vol_factor = min(1.0, 5.0 / max(atr_pct, 0.5))  # Normalize ATR
        
        # Market regime adjustment - more aggressive in bull markets
        regime_factor = 1.0 + (0.2 * bull_market)
        
        # Signal strength adjustment - scale by signal confidence
        signal_factor = min(1.0, signal_strength / 1.5)  # Normalize to max 1.0
        
        # Calculate final position size with limits
        adaptive_position = base_position_size * vol_factor * regime_factor * signal_factor
        adaptive_position = min(1.0, max(0.1, adaptive_position))  # Limit between 10-100%
        
        # Log position sizing factors for debugging
        logger.debug(f"Position sizing - vol:{vol_factor:.2f}, regime:{regime_factor:.2f}, "
                    f"signal:{signal_factor:.2f}, final:{adaptive_position:.2f}")
        
        # Get stop-loss and take-profit levels for risk management
        stop_loss_long = safe_get(backtest_df, i-1, 'stop_loss_long') if prev_holdings > 0 else None
        take_profit_long = safe_get(backtest_df, i-1, 'take_profit_long') if prev_holdings > 0 else None
        entry_price = safe_get(backtest_df, i-1, 'entry_price') if prev_holdings > 0 else None
        
        # Check if stop-loss or take-profit was hit during this period
        stop_loss_hit = False
        take_profit_hit = False
        
        # For existing long positions, check if stop-loss or take-profit was hit
        if prev_holdings > 0 and stop_loss_long is not None and entry_price is not None:
            # Check if price went below stop-loss at any point
            low_price = safe_get(backtest_df, i, 'Low')
            if low_price <= stop_loss_long:
                stop_loss_hit = True
                logger.debug(f"Stop loss triggered at {stop_loss_long:.2f} (entry: {entry_price:.2f})")
            
            # Check if price went above take-profit at any point
            high_price = safe_get(backtest_df, i, 'High')
            if high_price >= take_profit_long:
                take_profit_hit = True
                logger.debug(f"Take profit triggered at {take_profit_long:.2f} (entry: {entry_price:.2f})")
        
        # Handle stop-loss or take-profit hits before regular signal processing
        if stop_loss_hit or take_profit_hit:
            # Calculate exit price (we'll use stop-loss or take-profit level as the exit price)
            exit_price = stop_loss_long if stop_loss_hit else take_profit_long
            
            # Exit the position at the calculated price
            exit_value = prev_holdings * exit_price * (1 - commission)
            backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash + exit_value)
            backtest_df.loc[backtest_df.index[i], 'holdings'] = 0.0
            
            # Record the type of exit for analysis
            exit_type = 'stop_loss' if stop_loss_hit else 'take_profit'
            backtest_df.loc[backtest_df.index[i], 'exit_type'] = exit_type
            
            # Calculate and record P&L for this trade
            if entry_price is not None:
                pnl_pct = ((exit_price / entry_price) - 1) * 100
                backtest_df.loc[backtest_df.index[i], 'trade_pnl'] = pnl_pct
                logger.debug(f"Trade closed via {exit_type} with {pnl_pct:.2f}% P&L")
        
        # If no stop/take-profit hit, process regular signals
        elif position_signal > 0:  # Buy signal
            max_shares = (prev_cash * adaptive_position) / (curr_price * (1 + commission))
            backtest_df.loc[backtest_df.index[i], 'holdings'] = float(max_shares)
            backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash - (max_shares * curr_price * (1 + commission)))
            
            # Record entry price for future stop-loss/take-profit calculations
            backtest_df.loc[backtest_df.index[i], 'entry_price'] = float(curr_price)
            
        elif position_signal < 0:  # Sell signal
            exit_value = prev_holdings * curr_price * (1 - commission)
            backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash + exit_value)
            backtest_df.loc[backtest_df.index[i], 'holdings'] = 0.0
            
            # Record the exit type
            backtest_df.loc[backtest_df.index[i], 'exit_type'] = 'signal'
            
            # Calculate and record P&L if we have entry price
            if entry_price is not None:
                pnl_pct = ((curr_price / entry_price) - 1) * 100
                backtest_df.loc[backtest_df.index[i], 'trade_pnl'] = pnl_pct
                logger.debug(f"Trade closed via signal with {pnl_pct:.2f}% P&L")
        
        else:  # Hold position
            backtest_df.loc[backtest_df.index[i], 'holdings'] = float(prev_holdings)
            backtest_df.loc[backtest_df.index[i], 'cash'] = float(prev_cash)
            
            # Carry forward entry price if we're holding
            if prev_holdings > 0 and entry_price is not None:
                backtest_df.loc[backtest_df.index[i], 'entry_price'] = entry_price
        
        # Calculate portfolio value
        holdings_value = safe_get(backtest_df, i, 'holdings') * curr_price
        cash_value = safe_get(backtest_df, i, 'cash')
        portfolio_value = holdings_value + cash_value
        backtest_df.loc[backtest_df.index[i], 'portfolio'] = float(portfolio_value)
        
        # Calculate return
        prev_portfolio = safe_get(backtest_df, i-1, 'portfolio')
        if prev_portfolio > 0:
            return_value = (portfolio_value / prev_portfolio) - 1
            backtest_df.loc[backtest_df.index[i], 'returns'] = float(return_value)
    except Exception as e:
        logger.warning(f"Error in iteration {i}: {str(e)}")
            
# Calculate performance metrics
def _calculate_performance_metrics(self, backtest_df):
    """
    Calculate comprehensive performance metrics from backtest results for short-term trading.
    
    Args:
        backtest_df (pandas.DataFrame): DataFrame with backtest results
        
    Returns:
        dict: Dictionary with performance metrics
    """
    try:
        # Calculate returns
        final_portfolio = safe_get(backtest_df, -1, 'portfolio')
        initial_close = safe_get(backtest_df, 0, 'Close')
        final_close = safe_get(backtest_df, -1, 'Close')
        
        total_return = (final_portfolio / self.initial_capital) - 1
        buy_hold_return = (final_close / initial_close) - 1 if initial_close > 0 else 0
        
        # Convert to percentages
        total_return_pct = total_return * 100
        buy_hold_return_pct = buy_hold_return * 100
        
        # Calculate alpha (outperformance vs buy and hold)
        alpha = total_return - buy_hold_return
        alpha_pct = alpha * 100
        
        # Calculate Sharpe Ratio: (Mean Return - Risk Free Rate) / Standard Deviation
        # Assuming risk-free rate of 0.02 (2%) annualized, adjusting for daily
        daily_returns = backtest_df['returns']
        risk_free_daily = 0.02 / 252  # 252 trading days per year
        excess_returns = daily_returns - risk_free_daily
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0
        
        # Calculate Sortino Ratio: (Mean Return - Risk Free Rate) / Downside Standard Deviation
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = excess_returns.mean() / downside_returns.std() if downside_returns.std() > 0 else 0.0
        
        # Calculate Calmar Ratio: (Mean Return - Risk Free Rate) / Maximum Drawdown
        max_drawdown = 0.0
        peak_portfolio = self.initial_capital
        for i in range(len(backtest_df)):
            portfolio_value = safe_get(backtest_df, i, 'portfolio')
            if portfolio_value > peak_portfolio:
                peak_portfolio = portfolio_value
            drawdown = (peak_portfolio - portfolio_value) / peak_portfolio
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        calmar_ratio = excess_returns.mean() / max_drawdown if max_drawdown > 0 else 0.0
        
        # Prepare metrics dictionary
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'buy_hold_return': buy_hold_return,
            'buy_hold_return_pct': buy_hold_return_pct,
            'alpha': alpha,
            'alpha_pct': alpha_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
        
        return metrics
    except Exception as e:
        logger.warning(f"Error calculating performance metrics: {str(e)}")
        return {}
            
        Returns:
            list: List of trade dictionaries
        """
        try:
            # Simplified implementation
            return []
        except Exception as e:
            logger.warning(f"Error extracting trade history: {str(e)}")
            return []
