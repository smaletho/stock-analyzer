"""
Visualizer module for creating charts and visual representations of analysis results.
"""
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

from config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Creates visual representations and reports for stock analysis results.
    """
    
    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR, theme="darkgrid"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save output plots, defaults to environment variable
            theme (str): Seaborn theme to use
        """
        self.output_dir = output_dir
        sns.set_theme(style=theme)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Visualizer initialized with output directory: {output_dir}")
    
    def plot_signals(self, signals_df, ticker, save=True):
        """
        Plot the price chart with moving averages and trading signals.
        
        Args:
            signals_df (pandas.DataFrame): DataFrame with price data and signals
            ticker (str): Stock ticker symbol
            save (bool): Whether to save the plot to a file
            
        Returns:
            str: Path to the saved plot file (if save=True), or None
        """
        # Check if DataFrame is empty using len instead of .empty to avoid Series truth value ambiguity
        if len(signals_df) == 0:
            logger.error("Cannot create plot from empty DataFrame")
            return None
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot price and moving averages
        try:
            plt.plot(signals_df['Close'], label='Close Price', alpha=0.6, linewidth=2)
            plt.plot(signals_df['short_mavg'], label=f"{signals_df['short_mavg'].name}-Day MA", linewidth=1.5)
            plt.plot(signals_df['long_mavg'], label=f"{signals_df['long_mavg'].name}-Day MA", linewidth=1.5)
            
            # Plot buy and sell signals - use boolean masks to avoid Series truth value ambiguity
            buy_mask = signals_df['position'] > 0
            sell_mask = signals_df['position'] < 0
            
            buy_signals = signals_df[buy_mask]
            sell_signals = signals_df[sell_mask]
        except Exception as e:
            logger.error(f"Error during plotting: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', s=100, label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', s=100, label='Sell Signal')
        
        # Format the plot
        plt.title(f"{ticker} Moving Average Crossover Signals", fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/{ticker}_signals_{timestamp}.png"
            plt.savefig(filename)
            logger.info(f"Saved signals plot to {filename}")
            plt.close()
            return filename
        else:
            return None
    
    def plot_backtest_results(self, backtest_results, ticker, save=True):
        """
        Plot backtest results showing portfolio value over time.
        
        Args:
            backtest_results (dict): Results from the backtester
            ticker (str): Stock ticker symbol
            save (bool): Whether to save the plot to a file
            
        Returns:
            str: Path to the saved plot file (if save=True), or None
        """
        if not backtest_results['success']:
            logger.error("Cannot plot unsuccessful backtest results")
            return None
        
        backtest_df = backtest_results['backtest_df']
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot portfolio value and benchmark
        initial_capital = backtest_df['portfolio'].iloc[0]
        benchmark = backtest_df['Close'] / backtest_df['Close'].iloc[0] * initial_capital
        
        ax1.plot(backtest_df.index, backtest_df['portfolio'], label='Strategy', linewidth=2)
        ax1.plot(backtest_df.index, benchmark, label='Buy and Hold', linewidth=1.5, alpha=0.7)
        
        # Mark buy and sell points on the portfolio curve
        buy_signals = backtest_df[backtest_df['position'] > 0]
        sell_signals = backtest_df[backtest_df['position'] < 0]
        
        ax1.scatter(buy_signals.index, buy_signals['portfolio'], marker='^', color='g', s=100, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['portfolio'], marker='v', color='r', s=100, label='Sell')
        
        # Format the upper plot
        ax1.set_title(f"{ticker} Backtest Results", fontsize=16)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot portfolio returns
        ax2.fill_between(
            backtest_df.index, 
            backtest_df['returns'].cumsum() * 100, 
            0, 
            where=(backtest_df['returns'].cumsum() > 0), 
            color='green', 
            alpha=0.3, 
            label='Cumulative Return'
        )
        ax2.fill_between(
            backtest_df.index, 
            backtest_df['returns'].cumsum() * 100, 
            0, 
            where=(backtest_df['returns'].cumsum() < 0), 
            color='red', 
            alpha=0.3
        )
        
        ax2.plot(backtest_df.index, backtest_df['returns'].cumsum() * 100, color='blue', label='Cumulative %')
        
        # Format the lower plot
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format date axis for both plots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add performance metrics as text
        metrics = backtest_results['metrics']
        metrics_text = f"Total Return: {metrics['total_return_pct']:.2f}% | Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        metrics_text += f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}% | Win Rate: {metrics['win_rate_pct']:.2f}%"
        
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12, bbox={'facecolor': 'lightgrey', 'alpha': 0.5, 'pad': 5})
        
        # Save the plot if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/{ticker}_backtest_{timestamp}.png"
            plt.savefig(filename)
            logger.info(f"Saved backtest plot to {filename}")
            plt.close()
            return filename
        else:
            return None
    
    def plot_optimization_results(self, optimization_results, save=True):
        """
        Plot optimization results.
        
        Args:
            optimization_results (dict): Results from the optimizer
            save (bool): Whether to save the plot to a file
            
        Returns:
            str: Path to the saved plot file (if save=True), or None
        """
        if not optimization_results['success']:
            logger.error("Cannot plot unsuccessful optimization results")
            return None
        
        all_results = optimization_results['all_results']
        
        # Convert to DataFrame for easier plotting
        results_df = pd.DataFrame(all_results)
        
        # Create the heatmap data
        if len(results_df) > 5:  # Only create heatmap if we have enough data points
            # Get unique short and long window values
            short_windows = sorted(results_df['short_window'].unique())
            long_windows = sorted(results_df['long_window'].unique())
            
            # Create meshgrid
            heatmap_data = np.zeros((len(short_windows), len(long_windows)))
            heatmap_data[:] = np.nan  # Fill with NaNs initially
            
            # Fill the heatmap with Sharpe ratios
            for _, row in results_df.iterrows():
                short_idx = short_windows.index(row['short_window']) if row['short_window'] in short_windows else None
                long_idx = long_windows.index(row['long_window']) if row['long_window'] in long_windows else None
                
                if short_idx is not None and long_idx is not None:
                    heatmap_data[short_idx, long_idx] = row['sharpe_ratio']
            
            # Create the plot
            plt.figure(figsize=(12, 10))
            
            # Plot heatmap
            ax = plt.subplot(111)
            im = ax.imshow(heatmap_data, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
            
            # Set ticks
            ax.set_xticks(np.arange(len(long_windows)))
            ax.set_yticks(np.arange(len(short_windows)))
            ax.set_xticklabels(long_windows)
            ax.set_yticklabels(short_windows)
            
            # Label the axes
            plt.xlabel('Long Window')
            plt.ylabel('Short Window')
            plt.title('Parameter Optimization Results (Sharpe Ratio)')
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Mark the best parameters
            best_short = optimization_results['short_window']
            best_long = optimization_results['long_window']
            
            best_short_idx = short_windows.index(best_short) if best_short in short_windows else None
            best_long_idx = long_windows.index(best_long) if best_long in long_windows else None
            
            if best_short_idx is not None and best_long_idx is not None:
                ax.plot(best_long_idx, best_short_idx, 'r*', markersize=15)
                
            plt.tight_layout()
        else:
            # If not enough data for heatmap, create scatter plot
            plt.figure(figsize=(10, 6))
            
            plt.scatter(results_df['short_window'], results_df['long_window'], 
                        c=results_df['sharpe_ratio'], cmap='viridis', s=100)
            
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('Short Window')
            plt.ylabel('Long Window')
            plt.title('Parameter Optimization Results')
            
            # Mark the best parameters
            best_short = optimization_results['short_window']
            best_long = optimization_results['long_window']
            plt.scatter([best_short], [best_long], color='red', s=200, marker='*')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        # Add optimization details as text
        opt_text = f"Optimization Method: {optimization_results['method'].capitalize()}\n"
        opt_text += f"Best Parameters: Short Window={best_short}, Long Window={best_long}\n"
        opt_text += f"Sharpe Ratio: {optimization_results['sharpe_ratio']:.2f} | Total Return: {optimization_results['total_return_pct']:.2f}%"
        
        plt.figtext(0.5, 0.01, opt_text, ha='center', fontsize=12, bbox={'facecolor': 'lightgrey', 'alpha': 0.5, 'pad': 5})
        
        # Save the plot if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/optimization_results_{timestamp}.png"
            plt.savefig(filename)
            logger.info(f"Saved optimization plot to {filename}")
            plt.close()
            return filename
        else:
            return None
    
    def create_performance_report(self, backtest_results, optimization_results=None, sentiment_results=None):
        """
        Create a comprehensive performance report.
        
        Args:
            backtest_results (dict): Results from the backtester
            optimization_results (dict, optional): Results from the optimizer
            sentiment_results (dict, optional): Results from sentiment analysis
            
        Returns:
            str: HTML report content
        """
        if not backtest_results['success']:
            logger.error("Cannot create report from unsuccessful backtest results")
            return "<h1>Error: Backtest was not successful</h1>"
        
        metrics = backtest_results['metrics']
        trades = backtest_results['trades']
        
        # Start building the HTML report
        report = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Analysis Performance Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .summary {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .metrics {
                    display: flex;
                    flex-wrap: wrap;
                }
                .metric-box {
                    background-color: #ffffff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px;
                    min-width: 200px;
                    flex-grow: 1;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .positive {
                    color: green;
                }
                .negative {
                    color: red;
                }
            </style>
        </head>
        <body>
            <h1>Stock Analysis Performance Report</h1>
            <div class="summary">
                <h2>Performance Summary</h2>
                <p><strong>Total Return:</strong> <span class="{0}">{1:.2f}%</span> (Buy & Hold: {2:.2f}%)</p>
                <p><strong>Time Period:</strong> {3} days ({4:.1f} years)</p>
                <p><strong>Annualized Return:</strong> <span class="{5}">{6:.2f}%</span></p>
            </div>
        """.format(
            "positive" if metrics['total_return'] > 0 else "negative",
            metrics['total_return_pct'],
            metrics['buy_hold_return_pct'],
            metrics['days'],
            metrics['years'],
            "positive" if metrics['annualized_return'] > 0 else "negative",
            metrics['annualized_return_pct']
        )
        
        # Add key metrics
        report += """
            <h2>Key Metrics</h2>
            <div class="metrics">
                <div class="metric-box">
                    <h3>Risk-Adjusted Returns</h3>
                    <p><strong>Sharpe Ratio:</strong> {0:.2f}</p>
                    <p><strong>Sortino Ratio:</strong> {1:.2f}</p>
                    <p><strong>Maximum Drawdown:</strong> {2:.2f}%</p>
                </div>
                <div class="metric-box">
                    <h3>Trading Statistics</h3>
                    <p><strong>Total Trades:</strong> {3}</p>
                    <p><strong>Win Rate:</strong> {4:.2f}%</p>
                </div>
        """.format(
            metrics['sharpe_ratio'],
            metrics['sortino_ratio'],
            metrics['max_drawdown_pct'],
            metrics['total_trades'],
            metrics['win_rate_pct']
        )
        
        # Add optimization results if available
        if optimization_results and optimization_results['success']:
            report += """
                <div class="metric-box">
                    <h3>Optimized Parameters</h3>
                    <p><strong>Optimization Method:</strong> {0}</p>
                    <p><strong>Short Window:</strong> {1}</p>
                    <p><strong>Long Window:</strong> {2}</p>
                    <p><strong>Expected Sharpe Ratio:</strong> {3:.2f}</p>
                </div>
            """.format(
                optimization_results['method'].capitalize(),
                optimization_results['short_window'],
                optimization_results['long_window'],
                optimization_results['sharpe_ratio']
            )
        
        # Add sentiment results if available
        if sentiment_results:
            report += """
                <div class="metric-box">
                    <h3>Sentiment Analysis</h3>
                    <p><strong>Overall Sentiment:</strong> {0}</p>
                    <p><strong>Confidence:</strong> {1:.2f}</p>
                    <p><strong>Key Points:</strong> {2}</p>
                </div>
            """.format(
                sentiment_results.get('overall_sentiment', 'N/A'),
                sentiment_results.get('confidence', 0),
                sentiment_results.get('summary', 'No summary available')
            )
        
        report += "</div>"  # Close the metrics div
        
        # Add trade history
        if trades:
            report += """
                <h2>Trade History</h2>
                <table>
                    <tr>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Duration (Days)</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Profit/Loss</th>
                        <th>Return (%)</th>
                    </tr>
            """
            
            for trade in trades:
                profit_class = "positive" if trade['profit'] > 0 else "negative"
                report += """
                    <tr>
                        <td>{0}</td>
                        <td>{1}</td>
                        <td>{2}</td>
                        <td>${3:.2f}</td>
                        <td>${4:.2f}</td>
                        <td class="{5}">${6:.2f}</td>
                        <td class="{5}">{7:.2f}%</td>
                    </tr>
                """.format(
                    trade['entry_date'].strftime('%Y-%m-%d'),
                    trade['exit_date'].strftime('%Y-%m-%d'),
                    trade['duration_days'],
                    trade['entry_price'],
                    trade['exit_price'],
                    profit_class,
                    trade['profit'],
                    trade['profit_pct'] * 100
                )
            
            report += "</table>"
        
        # Close the HTML document
        report += """
            <div style="margin-top: 30px; text-align: center; font-size: 0.8em; color: #777;">
                <p>Generated on {0}</p>
            </div>
        </body>
        </html>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return report
