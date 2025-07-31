#!/usr/bin/env python
"""
Main entry point for the AI Stock Analysis Tool.
"""
import argparse
import logging
import sys
import pandas as pd
from datetime import datetime, timedelta

from data_fetcher import DataFetcher
from technical_analysis import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from backtester import Backtester
from optimizer import StrategyOptimizer
from visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Stock analysis tool with technical indicators and backtesting")
    
    # Required arguments
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    
    # Optional arguments
    parser.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--days", type=int, default=365, help="Number of days of historical data to analyze")
    parser.add_argument("--short_window", type=int, default=10, help="Short moving average window length")
    parser.add_argument("--long_window", type=int, default=50, help="Long moving average window length")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting")
    parser.add_argument("--optimize", action="store_true", help="Optimize strategy parameters")
    parser.add_argument("--sentiment", action="store_true", help="Include sentiment analysis")
    parser.add_argument("--timeframe", type=str, default="1d", choices=["1d", "1wk", "1mo"], 
                        help="Data timeframe: 1d=daily, 1wk=weekly, 1mo=monthly")
    parser.add_argument("--timeframe_analysis", action="store_true", 
                        help="Analyze strategy performance across different timeframes")
    parser.add_argument("--compare_strategies", action="store_true",
                        help="Compare performance of original vs enhanced strategy")
    
    return parser.parse_args()

def run_timeframe_analysis(args):
    """Run analysis across multiple timeframes and compare performance."""
    logger.info(f"Starting multi-timeframe analysis for {args.ticker}")
    
    timeframes = ['1d', '1wk', '1mo']
    backtest_results = {}
    
    # Set up visualizer for charts
    visualizer = Visualizer()
    
    for timeframe in timeframes:
        logger.info(f"Analyzing {args.ticker} with {timeframe} timeframe")
        
        # Adjust time periods based on timeframe
        end_date = datetime.now()
        if timeframe == '1d':
            start_date = end_date - timedelta(days=365)  # 1 year for daily
            short_window = args.short_window
            long_window = args.long_window
        elif timeframe == '1wk':
            start_date = end_date - timedelta(days=365 * 3)  # 3 years for weekly
            short_window = max(3, args.short_window // 5)  # Adjust for weekly data
            long_window = max(10, args.long_window // 5)
        elif timeframe == '1mo':
            start_date = end_date - timedelta(days=365 * 5)  # 5 years for monthly
            short_window = max(2, args.short_window // 20)  # Adjust for monthly data
            long_window = max(6, args.long_window // 20)
        
        # Fetch data for this timeframe
        try:
            fetcher = DataFetcher()
            df = fetcher.fetch_stock_data(args.ticker, start_date, end_date, interval=timeframe)
            
            if len(df) == 0:
                logger.warning(f"No data available for {timeframe} timeframe")
                continue
                
            logger.info(f"Fetched {len(df)} data points for {timeframe} timeframe")
            
            # Run technical analysis with adjusted windows
            analyzer = TechnicalAnalyzer(short_window=short_window, long_window=long_window)
            signals_df = analyzer.analyze(df)
            
            # Run backtesting
            backtester = Backtester()
            results = backtester.run_backtest(signals_df)
            
            if results['success']:
                backtest_results[timeframe] = {
                    'total_return': results['total_return'],
                    'buy_hold_return': results['buy_and_hold_return'],
                    'alpha': results['total_return'] - results['buy_and_hold_return'],
                    'signals_count': results['signals_count'],
                    'data_points': len(df),
                    'sharpe_ratio': results['metrics']['sharpe_ratio'] if 'sharpe_ratio' in results['metrics'] else 0,
                    'win_rate': results['metrics']['win_rate'] if 'win_rate' in results['metrics'] else 0
                }
                
                # Generate plot for this timeframe
                plot_filename = visualizer.plot_backtest_results(
                    results, f"{args.ticker}_{timeframe}", save=True
                )
                logger.info(f"Saved backtest plot for {timeframe} timeframe to {plot_filename}")
            else:
                logger.warning(f"Backtesting failed for {timeframe} timeframe: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error in {timeframe} timeframe analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Print comparative analysis results
    print("\n" + "="*50)
    print(f"TIMEFRAME ANALYSIS FOR {args.ticker}")
    print("="*50)
    
    if not backtest_results:
        print("No successful results across any timeframe.")
        return
    
    # Create a comparison table
    print(f"{'Timeframe':<10} {'Return %':<10} {'Buy&Hold %':<10} {'Alpha %':<10} {'Signals':<10} {'Sharpe':<10}")
    print("-"*60)
    
    best_timeframe = None
    best_alpha = -float('inf')
    
    for timeframe, results in backtest_results.items():
        ret_pct = results['total_return'] * 100
        bh_pct = results['buy_hold_return'] * 100
        alpha_pct = results['alpha'] * 100
        signals = results['signals_count']
        sharpe = results['sharpe_ratio']
        
        print(f"{timeframe:<10} {ret_pct:>9.2f}% {bh_pct:>9.2f}% {alpha_pct:>9.2f}% {signals:>9} {sharpe:>9.2f}")
        
        # Track best performing timeframe by alpha
        if alpha_pct > best_alpha:
            best_alpha = alpha_pct
            best_timeframe = timeframe
    
    print("\nBest performing timeframe: " + best_timeframe)
    print(f"Recommended command: py main.py --ticker {args.ticker} --timeframe {best_timeframe} --backtest")
    print("="*50)
    
    return

def compare_strategies(args):
    """Compare performance between original and enhanced strategy."""
    logger.info(f"Starting strategy comparison for {args.ticker}")
    
    # Set up test parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Initialize visualization
    visualizer = Visualizer()
    
    # Fetch data
    try:
        fetcher = DataFetcher()
        df = fetcher.fetch_stock_data(args.ticker, start_date, end_date, interval=args.timeframe)
        
        if len(df) == 0:
            logger.error("Fetched dataframe is empty")
            return
        logger.info(f"Successfully fetched {len(df)} data points for comparison")
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Run original strategy with default parameters
    original_analyzer = TechnicalAnalyzer(short_window=10, long_window=50)
    # Use reasonable threshold for original strategy
    original_params = original_analyzer.params.copy()
    original_params['signal_threshold'] = 0.7  # Reasonable threshold for base strategy
    original_analyzer.params = original_params
    original_signals = original_analyzer.analyze(df)
    
    # Run enhanced strategy with optimized parameters
    enhanced_analyzer = TechnicalAnalyzer(short_window=21, long_window=68)
    # Enhanced features are enabled by default
    enhanced_signals = enhanced_analyzer.analyze(df)
    
    # Run backtests
    backtester = Backtester()
    original_results = backtester.run_backtest(original_signals)
    enhanced_results = backtester.run_backtest(enhanced_signals)
    
    # Generate visualizations
    original_plot = visualizer.plot_backtest_results(
        original_results, f"{args.ticker}_original", save=True
    )
    enhanced_plot = visualizer.plot_backtest_results(
        enhanced_results, f"{args.ticker}_enhanced", save=True
    )
    
    # Print comparison results
    print("\n" + "="*60)
    print(f"STRATEGY COMPARISON FOR {args.ticker}")
    print("="*60)
    
    # Extract metrics
    if original_results['success'] and enhanced_results['success']:
        orig_return = original_results['total_return'] * 100
        enh_return = enhanced_results['total_return'] * 100
        bh_return = original_results['buy_and_hold_return'] * 100
        
        orig_signals = original_results['signals_count']
        enh_signals = enhanced_results['signals_count']
        
        orig_sharpe = original_results['metrics'].get('sharpe_ratio', 0)
        enh_sharpe = enhanced_results['metrics'].get('sharpe_ratio', 0)
        
        orig_drawdown = original_results['metrics'].get('max_drawdown_pct', 0)
        enh_drawdown = enhanced_results['metrics'].get('max_drawdown_pct', 0)
        
        # Create comparison table
        print(f"{'Metric':<20} {'Original':<12} {'Enhanced':<12} {'Difference':<12}")
        print("-"*60)
        print(f"{'Return %':<20} {orig_return:>11.2f}% {enh_return:>11.2f}% {enh_return-orig_return:>11.2f}%")
        print(f"{'Buy & Hold %':<20} {bh_return:>11.2f}% {bh_return:>11.2f}% {0:>11.2f}%")
        print(f"{'Alpha %':<20} {orig_return-bh_return:>11.2f}% {enh_return-bh_return:>11.2f}% {(enh_return-bh_return)-(orig_return-bh_return):>11.2f}%")
        print(f"{'Signals Count':<20} {orig_signals:>11} {enh_signals:>11} {enh_signals-orig_signals:>11}")
        print(f"{'Sharpe Ratio':<20} {orig_sharpe:>11.2f} {enh_sharpe:>11.2f} {enh_sharpe-orig_sharpe:>11.2f}")
        print(f"{'Max Drawdown %':<20} {orig_drawdown:>11.2f}% {enh_drawdown:>11.2f}% {enh_drawdown-orig_drawdown:>11.2f}%")
        
        # Summary
        if enh_return > orig_return:
            print("\nEnhanced strategy outperformed the original strategy.")
            improvement = ((enh_return / orig_return) - 1) * 100 if orig_return > 0 else 100
            print(f"Return improvement: {improvement:.1f}%")
        else:
            print("\nEnhanced strategy did not outperform the original strategy.")
            underperformance = ((orig_return / enh_return) - 1) * 100 if enh_return > 0 else 100
            print(f"Return underperformance: {underperformance:.1f}%")
        
        print("\nStrategy profiles:")
        print(f"Original: MA({10}/{50}), standard parameters")
        print(f"Enhanced: MA({21}/{68}), adaptive position sizing, trend following, volatility filters")
        
        print("\nPlots saved to:")
        print(f"- {original_plot}")
        print(f"- {enhanced_plot}")
    else:
        print("Unable to complete comparison due to backtest failures.")
    
    print("="*60)
    return

def run_analysis(args):
    """Run the stock analysis pipeline with support for different timeframes."""
    logger.info(f"Starting analysis for {args.ticker}")
    
    # Check for special analysis modes
    if args.timeframe_analysis:
        return run_timeframe_analysis(args)
    
    if args.compare_strategies:
        return compare_strategies(args)
    
    # Calculate the start date based on the requested days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Fetch and prepare data using the selected timeframe
    try:
        fetcher = DataFetcher()
        df = fetcher.fetch_stock_data(args.ticker, start_date, end_date, interval=args.timeframe)
        
        # Check if dataframe is empty after fetching
        if len(df) == 0:  # Using len instead of df.empty to avoid Series truth value ambiguity
            logger.error("Fetched dataframe is empty")
            return
        logger.info(f"Successfully fetched {len(df)} data points for {args.ticker} with {args.timeframe} timeframe")
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    
    # Perform technical analysis
    analyzer = TechnicalAnalyzer(short_window=args.short_window, long_window=args.long_window)
    signals_df = analyzer.analyze(df)
    
    # Generate visualization
    visualizer = Visualizer()
    plot_path = visualizer.plot_signals(signals_df, args.ticker)
    logger.info(f"Technical analysis chart saved to {plot_path}")
    
    # Perform sentiment analysis if requested
    if args.sentiment:
        try:
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_results = sentiment_analyzer.analyze_sentiment(args.ticker)
            logger.info(f"Sentiment analysis completed: {sentiment_results['overall_sentiment']}")
            
            # Combine technical and sentiment analyses
            final_signal = analyzer.combine_signals_with_sentiment(
                signals_df.iloc[-1], sentiment_results
            )
            logger.info(f"Final trading signal after sentiment integration: {final_signal}")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
    
    # Run backtesting if requested
    if args.backtest:
        try:
            backtester = Backtester()
            backtest_results = backtester.run_backtest(signals_df)
            logger.info(f"Backtesting completed: {backtest_results['summary']}")
            
            # Visualize backtest results
            backtest_plot = visualizer.plot_backtest_results(backtest_results, args.ticker)
            logger.info(f"Backtest visualization saved to {backtest_plot}")
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
    
    # Run parameter optimization if requested
    if args.optimize:
        try:
            optimizer = StrategyOptimizer()
            optimization_results = optimizer.optimize_parameters(df)
            logger.info("Optimization completed")
            logger.info(f"Optimal short window: {optimization_results['short_window']}")
            logger.info(f"Optimal long window: {optimization_results['long_window']}")
            logger.info(f"Expected performance: {optimization_results['expected_performance']}")
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
    
    # Print final summary
    print("\n" + "="*50)
    print(f"ANALYSIS SUMMARY FOR {args.ticker}")
    print("="*50)
    
    # Handle potential MultiIndex DataFrame (like from Yahoo Finance)
    try:
        # If df is a MultiIndex DataFrame, get the Close column properly
        if isinstance(df.columns, pd.MultiIndex):
            # Check if Close is part of a tuple in columns
            close_col = None
            for col in df.columns:
                if isinstance(col, tuple) and 'Close' in col:
                    close_col = col
                    break
            
            if close_col:
                latest_price = df[close_col].iloc[-1]
            else:
                latest_price = df['Close'].iloc[-1]
        else:
            latest_price = df['Close'].iloc[-1]
            
        print(f"Latest Price: ${float(latest_price):.2f} (as of {df.index[-1].strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"Error displaying latest price: {str(e)}")
    
    # Handle MultiIndex in signals_df
    if len(signals_df) > 0:  # Use len() instead of .empty to avoid Series truth value issues
        latest_signal = signals_df.iloc[-1]
        
        # Handle potential MultiIndex for the 'signal' column
        try:
            # Try to extract the signal value, accounting for MultiIndex
            if isinstance(signals_df.columns, pd.MultiIndex):
                signal_val = None
                for col in signals_df.columns:
                    if isinstance(col, tuple) and 'signal' in col:
                        signal_val = latest_signal[col]
                        break
                if signal_val is None:
                    signal_val = latest_signal['signal']
            else:
                signal_val = latest_signal['signal']
        except Exception as e:
            logger.warning(f"Error extracting signal value: {str(e)}")
            signal_val = None
        
        # Extract short and long moving averages
        try:
            # Get the short moving average
            short_mavg_val = None
            if isinstance(signals_df.columns, pd.MultiIndex):
                for col in signals_df.columns:
                    if isinstance(col, tuple) and 'short_mavg' in col:
                        short_mavg_val = signals_df[col].iloc[-1]
                        break
            else:
                if 'short_mavg' in signals_df.columns:
                    short_mavg_val = signals_df['short_mavg'].iloc[-1]
            
            # Get the long moving average
            long_mavg_val = None
            if isinstance(signals_df.columns, pd.MultiIndex):
                for col in signals_df.columns:
                    if isinstance(col, tuple) and 'long_mavg' in col:
                        long_mavg_val = signals_df[col].iloc[-1]
                        break
            else:
                if 'long_mavg' in signals_df.columns:
                    long_mavg_val = signals_df['long_mavg'].iloc[-1]
            
            # Print the moving averages if available
            if short_mavg_val is not None:
                print(f"Short-term MA ({args.short_window}-day): ${float(short_mavg_val):.2f}")
            if long_mavg_val is not None:
                print(f"Long-term MA ({args.long_window}-day): ${float(long_mavg_val):.2f}")
        except Exception as e:
            logger.warning(f"Error displaying moving averages: {str(e)}")
    
    # Extract key price levels
    if len(signals_df) > 0:
        # Safely extract support and resistance levels
        support_level = None
        resistance_level = None
        buy_target = None
        sell_target = None
        
        try:
            if 'support_level' in signals_df.columns:
                support_level = signals_df['support_level'].iloc[-1]
            if 'resistance_level' in signals_df.columns:
                resistance_level = signals_df['resistance_level'].iloc[-1]
            if 'buy_target' in signals_df.columns:
                buy_target = signals_df['buy_target'].iloc[-1]
            if 'sell_target' in signals_df.columns:
                sell_target = signals_df['sell_target'].iloc[-1]
        except Exception as e:
            logger.debug(f"Error extracting price levels: {str(e)}")
        
        print("\nKey Price Levels:")
        if support_level is not None:
            print(f"Support: ${support_level:.2f}")
        if resistance_level is not None:
            print(f"Resistance: ${resistance_level:.2f}")
        if buy_target is not None:
            print(f"Buy Target: ${buy_target:.2f}")
        if sell_target is not None:
            print(f"Sell Target: ${sell_target:.2f}")
            
        # Add RSI and other indicator values
        try:
            if 'rsi' in signals_df.columns:
                current_rsi = signals_df['rsi'].iloc[-1]
                print(f"\nCurrent RSI: {current_rsi:.1f}")
                if current_rsi > 70:
                    print("RSI indicates OVERBOUGHT conditions")
                elif current_rsi < 30:
                    print("RSI indicates OVERSOLD conditions")
        except Exception:
            pass
    
    # Print sentiment analysis if available
    if args.sentiment and 'sentiment_results' in locals():
        print("\nSentiment Analysis:")
        print(f"Overall: {sentiment_results['overall_sentiment']}")
        print(f"Confidence: {sentiment_results.get('confidence', 0.0):.2f}")
    
    # Print backtest results summary if available
    if args.backtest and 'backtest_results' in locals() and backtest_results.get('success', False):
        print("\nBacktest Results:")
        print(f"Strategy Return: {backtest_results['total_return']:.2%}")
        print(f"Buy & Hold Return: {backtest_results['buy_and_hold_return']:.2%}")
        if 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            if 'sharpe_ratio' in metrics:
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print("=" * 50)

if __name__ == "__main__":
    try:
        args = parse_args()
        run_analysis(args)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
