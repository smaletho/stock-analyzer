"""
Strategy optimizer module for refining trading strategy parameters.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import scikit-optimize components
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm

from technical_analysis import TechnicalAnalyzer
from backtester import Backtester

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Optimizes trading strategy parameters using various optimization techniques.
    """
    
    def __init__(self, method="bayesian", iterations=50, initial_capital=10000):
        """
        Initialize the strategy optimizer.
        
        Args:
            method (str): Optimization method ('bayesian' or 'genetic')
            iterations (int): Number of iterations for optimization
            initial_capital (float): Initial capital for backtesting
        """
        self.method = method
        self.iterations = iterations
        self.initial_capital = initial_capital
        logger.info(f"Strategy Optimizer initialized with {method} method, {iterations} iterations")
    
    def optimize_parameters(self, data_df, short_window_range=(5, 30), long_window_range=(20, 200)):
        """
        Optimize strategy parameters.
        
        Args:
            data_df (pandas.DataFrame): DataFrame with stock price data
            short_window_range (tuple): Range for short window optimization
            long_window_range (tuple): Range for long window optimization
            
        Returns:
            dict: Optimization results
        """
        if data_df.empty:
            logger.error("Cannot optimize with empty DataFrame")
            return {
                'success': False,
                'error': 'Empty data DataFrame'
            }
        
        logger.info(f"Starting {self.method} optimization with {self.iterations} iterations")
        
        if self.method == "bayesian":
            return self._bayesian_optimization(data_df, short_window_range, long_window_range)
        elif self.method == "genetic":
            return self._genetic_optimization(data_df, short_window_range, long_window_range)
        else:
            logger.error(f"Unknown optimization method: {self.method}")
            return {
                'success': False,
                'error': f"Unknown optimization method: {self.method}"
            }
    
    def _bayesian_optimization(self, data_df, short_window_range, long_window_range):
        """
        Perform Bayesian optimization for strategy parameters.
        
        Args:
            data_df (pandas.DataFrame): DataFrame with stock price data
            short_window_range (tuple): Range for short window optimization
            long_window_range (tuple): Range for long window optimization
            
        Returns:
            dict: Optimization results
        """
        logger.info("Setting up enhanced Bayesian optimization with multiple parameters")
        
        # Define a more comprehensive search space
        space = [
            # Moving average windows
            Integer(short_window_range[0], short_window_range[1], name="short_window"),
            Integer(long_window_range[0], long_window_range[1], name="long_window"),
            
            # RSI thresholds
            Integer(20, 40, name="rsi_oversold"),   # Oversold threshold (buy signal)
            Integer(60, 80, name="rsi_overbought"), # Overbought threshold (sell signal)
            
            # Bollinger Band settings
            Real(1.5, 3.0, name="bb_std"),          # Standard deviation multiplier
            Integer(10, 30, name="bb_window"),      # Bollinger Band window
            
            # Volume filter settings
            Real(0.8, 2.0, name="volume_factor"),    # Volume comparison factor
            Integer(5, 30, name="volume_window"),   # Volume lookback window
            
            # MACD settings
            Integer(5, 15, name="macd_fast"),       # Fast EMA window
            Integer(20, 40, name="macd_slow"),      # Slow EMA window
            Integer(5, 15, name="macd_signal")      # Signal line window
        ]
        
        # Store results for each iteration
        self.all_results = []
        
        # Define the objective function with more parameters
        @use_named_args(space)
        def objective(short_window, long_window, rsi_oversold, rsi_overbought, 
                      bb_std, bb_window, volume_factor, volume_window, 
                      macd_fast, macd_slow, macd_signal):
            # Validate parameter combinations
            if short_window >= long_window or macd_fast >= macd_slow:
                return 0.0
            
            # Create an instance of our enhanced analyzer
            custom_params = {
                'short_window': short_window,
                'long_window': long_window,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'bb_std': bb_std,
                'bb_window': bb_window,
                'volume_factor': volume_factor,
                'volume_window': volume_window,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'macd_signal': macd_signal
            }
            
            # Create analyzer with these parameters
            analyzer = TechnicalAnalyzer(
                short_window=short_window, 
                long_window=long_window,
                custom_params=custom_params
            )
            
            # Analyze data and run backtest
            signals_df = analyzer.analyze(data_df)
            backtester = Backtester(initial_capital=self.initial_capital)
            results = backtester.run_backtest(signals_df)
            
            # If backtest failed or results are invalid, return a poor score
            if not results.get('success', False):
                return 0.0
                
            # Calculate a composite score based on multiple metrics
            metrics = results['metrics']
            
            # Extract metrics with proper error handling
            try:
                total_return = metrics.get('total_return', 0)
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                max_drawdown = metrics.get('max_drawdown', 1)  # Higher is worse
                win_rate = metrics.get('win_rate', 0)
                
                # Protect against NaN values
                if np.isnan(total_return) or np.isnan(sharpe_ratio) or np.isnan(max_drawdown) or np.isnan(win_rate):
                    return 0.0
                    
                # Composite score: Prioritize return and Sharpe ratio, penalize drawdowns
                # Formula gives higher scores to better strategies
                score = (2 * total_return) + sharpe_ratio + win_rate - (3 * max_drawdown)
                
                # Store detailed results for analysis
                self.all_results.append({
                    **custom_params,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'composite_score': score
                })
                
                # Return negative score because optimizer minimizes
                return -score
                
            except Exception as e:
                logger.warning(f"Error calculating optimization score: {str(e)}")
                return 0.0
        
        # Run Bayesian optimization
        logger.info("Running Bayesian optimization...")
        try:
            result = gp_minimize(
                objective,
                space,
                n_calls=self.iterations,
                random_state=42
            )
            
            # Extract best parameters
            short_window_opt = result.x[0]
            long_window_opt = result.x[1]
            
            # Re-run with optimal parameters to get final results
            analyzer = TechnicalAnalyzer(short_window=short_window_opt, long_window=long_window_opt)
            signals_df = analyzer.analyze(data_df)
            
            backtester = Backtester(initial_capital=self.initial_capital)
            final_results = backtester.run_backtest(signals_df)
            
            # Prepare optimization results
            optimization_results = {
                'success': True,
                'method': 'bayesian',
                'short_window': short_window_opt,
                'long_window': long_window_opt,
                'sharpe_ratio': final_results['metrics']['sharpe_ratio'],
                'total_return': final_results['metrics']['total_return'],
                'total_return_pct': final_results['metrics']['total_return_pct'],
                'max_drawdown': final_results['metrics']['max_drawdown'],
                'win_rate': final_results['metrics']['win_rate'],
                'iterations': self.iterations,
                'all_results': self.all_results,
                'expected_performance': final_results['metrics']
            }
            
            logger.info(f"Bayesian optimization completed: short_window={short_window_opt}, long_window={long_window_opt}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {str(e)}")
            return {
                'success': False,
                'error': f"Optimization failed: {str(e)}"
            }
    
    def _genetic_optimization(self, data_df, short_window_range, long_window_range):
        """
        Perform genetic algorithm optimization for strategy parameters.
        
        Args:
            data_df (pandas.DataFrame): DataFrame with stock price data
            short_window_range (tuple): Range for short window optimization
            long_window_range (tuple): Range for long window optimization
            
        Returns:
            dict: Optimization results
        """
        # Store all results
        self.all_results = []
        
        # Define the evaluation function
        def evaluate(individual):
            short_window = individual[0]
            long_window = individual[1]
            
            # Ensure short_window is less than long_window
            if short_window >= long_window:
                return (-1000.0,)  # Heavily penalize invalid combinations
            
            # Create analyzer with the given parameters
            analyzer = TechnicalAnalyzer(short_window=short_window, long_window=long_window)
            signals_df = analyzer.analyze(data_df)
            
            # Run backtest
            backtester = Backtester(initial_capital=self.initial_capital)
            results = backtester.run_backtest(signals_df)
            
            # Store the results
            self.all_results.append({
                'short_window': short_window,
                'long_window': long_window,
                'sharpe_ratio': results['metrics']['sharpe_ratio'],
                'total_return': results['metrics']['total_return'],
                'max_drawdown': results['metrics']['max_drawdown'],
                'win_rate': results['metrics']['win_rate']
            })
            
            # Use Sharpe ratio as fitness
            if not results['success'] or np.isnan(results['metrics']['sharpe_ratio']):
                return (0.0,)
                
            return (results['metrics']['sharpe_ratio'],)
        
        try:
            # Set up the genetic algorithm
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Define genes
            toolbox.register("attr_short", random.randint, short_window_range[0], short_window_range[1])
            toolbox.register("attr_long", random.randint, long_window_range[0], long_window_range[1])
            
            # Define individual and population
            toolbox.register("individual", tools.initCycle, creator.Individual, 
                            (toolbox.attr_short, toolbox.attr_long), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Genetic operators
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt, 
                            low=[short_window_range[0], long_window_range[0]], 
                            up=[short_window_range[1], long_window_range[1]], 
                            indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create initial population
            population = toolbox.population(n=20)
            
            # Run the genetic algorithm
            logger.info("Running genetic algorithm optimization...")
            algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, 
                            ngen=self.iterations, verbose=False)
            
            # Get best individual
            best_individual = tools.selBest(population, k=1)[0]
            short_window_opt = best_individual[0]
            long_window_opt = best_individual[1]
            
            # Re-run with optimal parameters to get final results
            analyzer = TechnicalAnalyzer(short_window=short_window_opt, long_window=long_window_opt)
            signals_df = analyzer.analyze(data_df)
            
            backtester = Backtester(initial_capital=self.initial_capital)
            final_results = backtester.run_backtest(signals_df)
            
            # Prepare optimization results
            optimization_results = {
                'success': True,
                'method': 'genetic',
                'short_window': short_window_opt,
                'long_window': long_window_opt,
                'sharpe_ratio': final_results['metrics']['sharpe_ratio'],
                'total_return': final_results['metrics']['total_return'],
                'total_return_pct': final_results['metrics']['total_return_pct'],
                'max_drawdown': final_results['metrics']['max_drawdown'],
                'win_rate': final_results['metrics']['win_rate'],
                'iterations': self.iterations,
                'all_results': self.all_results,
                'expected_performance': final_results['metrics']
            }
            
            logger.info(f"Genetic optimization completed: short_window={short_window_opt}, long_window={long_window_opt}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Genetic optimization failed: {str(e)}")
            return {
                'success': False,
                'error': f"Optimization failed: {str(e)}"
            }
        finally:
            # Clean up DEAP global variables to prevent conflicts in future runs
            if hasattr(creator, 'FitnessMax'):
                del creator.FitnessMax
            if hasattr(creator, 'Individual'):
                del creator.Individual
    
    def grid_search(self, data_df, short_window_range, long_window_range, step_short=1, step_long=5):
        """
        Perform grid search for strategy parameters.
        
        Args:
            data_df (pandas.DataFrame): DataFrame with stock price data
            short_window_range (tuple): Range for short window optimization
            long_window_range (tuple): Range for long window optimization
            step_short (int): Step size for short window
            step_long (int): Step size for long window
            
        Returns:
            dict: Grid search results
        """
        short_windows = range(short_window_range[0], short_window_range[1] + 1, step_short)
        long_windows = range(long_window_range[0], long_window_range[1] + 1, step_long)
        
        results = []
        
        # Set up progress bar
        total_combinations = len(short_windows) * len(long_windows)
        counter = 0
        
        logger.info(f"Running grid search with {total_combinations} parameter combinations...")
        
        # Iterate through all combinations
        for short_window in short_windows:
            for long_window in long_windows:
                counter += 1
                if counter % 10 == 0:
                    logger.info(f"Progress: {counter}/{total_combinations} combinations tested")
                
                # Skip invalid combinations (short window must be less than long window)
                if short_window >= long_window:
                    continue
                
                # Create analyzer with the given parameters
                analyzer = TechnicalAnalyzer(short_window=short_window, long_window=long_window)
                signals_df = analyzer.analyze(data_df)
                
                # Run backtest
                backtester = Backtester(initial_capital=self.initial_capital)
                backtest_results = backtester.run_backtest(signals_df)
                
                # Store results
                if backtest_results['success']:
                    results.append({
                        'short_window': short_window,
                        'long_window': long_window,
                        'sharpe_ratio': backtest_results['metrics']['sharpe_ratio'],
                        'total_return': backtest_results['metrics']['total_return'],
                        'total_return_pct': backtest_results['metrics']['total_return_pct'],
                        'max_drawdown': backtest_results['metrics']['max_drawdown'],
                        'win_rate': backtest_results['metrics']['win_rate'],
                    })
        
        # Find best parameters based on Sharpe ratio
        if results:
            best_result = max(results, key=lambda x: x['sharpe_ratio'])
            
            logger.info(f"Grid search completed: best parameters found - short_window={best_result['short_window']}, long_window={best_result['long_window']}")
            
            return {
                'success': True,
                'method': 'grid_search',
                'short_window': best_result['short_window'],
                'long_window': best_result['long_window'],
                'sharpe_ratio': best_result['sharpe_ratio'],
                'total_return': best_result['total_return'],
                'total_return_pct': best_result['total_return_pct'],
                'max_drawdown': best_result['max_drawdown'],
                'win_rate': best_result['win_rate'],
                'iterations': counter,
                'all_results': results
            }
        else:
            logger.error("Grid search failed: no valid parameter combinations found")
            return {
                'success': False,
                'error': 'No valid parameter combinations found'
            }
