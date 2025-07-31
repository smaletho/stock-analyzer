# Setting Up API Keys

This application uses real financial data from external APIs. To function properly, you'll need to set up the necessary API keys.

## Required API Keys

1. **Alpha Vantage API Key**
   - Used for retrieving financial news and sentiment data
   - Free tier available with limitations (500 requests per day/5 per minute)
   - [Get your free API key here](https://www.alphavantage.co/support/#api-key)

2. **Ollama (Optional, for local LLM sentiment analysis)**
   - Used for advanced sentiment analysis through Llama 3
   - Free, self-hosted solution
   - [Download Ollama here](https://ollama.com/download)
   - After installation, run: `ollama pull llama3`

## Setting Up Environment Variables

1. Create a `.env` file in the project root:
   ```bash
   cp .env.template .env
   ```

2. Edit the `.env` file and add your API keys:
   ```
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
   OLLAMA_HOST=http://localhost:11434
   ```

3. Save the file and restart the application

## Usage Notes

- If Alpha Vantage API key is not provided, the application will fall back to Yahoo Finance for price data, but news and sentiment features will be limited
- If Ollama is not running, the application will use a simple keyword-based sentiment analysis as a fallback
- You can modify other settings like output directory in the `.env` file as well

## API Rate Limits

- **Alpha Vantage**: Free tier is limited to 5 requests per minute and 500 requests per day
- Consider upgrading to a paid plan for more intensive usage
