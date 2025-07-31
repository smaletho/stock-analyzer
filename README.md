# AI Stock Analysis Tool

A Python-based AI tool that analyzes stocks using moving averages and sentiment analysis to generate trading signals.

## Features

- **Technical Analysis**: Calculates short-term (10-day) and long-term (50-day) moving averages
- **Signal Generation**: Detects buy/sell signals based on moving average crossovers
- **Market Data**: Fetches historical and real-time data using Yahoo Finance API
- **Sentiment Analysis**: Integrates Llama 3 via Ollama for news and social media sentiment analysis
- **Backtesting**: Evaluates strategy performance on historical data
- **Self-Optimization**: Automatically refines parameters using Bayesian Optimization and Genetic Algorithms
- **Reinforcement Learning**: Improves decision-making based on past market conditions

## Setup and Installation

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Ensure Ollama is installed with Llama 3 model (see [Ollama's documentation](https://github.com/ollama/ollama))

## Usage

```bash
python main.py --ticker SYMBOL
```

Replace `SYMBOL` with the stock ticker you want to analyze (e.g., AAPL, MSFT, GOOGL).

## Components

- `main.py`: Entry point for the application
- `data_fetcher.py`: Handles data retrieval and preprocessing
- `technical_analysis.py`: Implements moving average calculations and signal generation
- `sentiment_analyzer.py`: Connects to Llama 3 for sentiment analysis
- `backtester.py`: Contains backtesting functionality
- `optimizer.py`: Implements parameter optimization algorithms
- `visualizer.py`: Generates visualizations and reports

## Requirements

See `requirements.txt` for the complete list of dependencies.
