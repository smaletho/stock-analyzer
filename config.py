"""
Configuration module for managing environment variables and API keys.
"""
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# Validate required API keys
if not ALPHA_VANTAGE_API_KEY:
    logger.warning("ALPHA_VANTAGE_API_KEY not found in environment variables.")
    logger.info("To obtain an Alpha Vantage API key, visit: https://www.alphavantage.co/support/#api-key")

# Application settings
DEFAULT_OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
