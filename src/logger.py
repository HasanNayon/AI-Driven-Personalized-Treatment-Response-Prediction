"""
Logger Configuration
Centralized logging for the entire application
"""

import sys
from pathlib import Path
from loguru import logger
from src.config import config

# Remove default logger
logger.remove()

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Console logger with color
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=config.get('logging.level', 'INFO'),
    colorize=True
)

# File logger
logger.add(
    config.get('logging.file', 'logs/app.log'),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=config.get('logging.level', 'INFO'),
    rotation="10 MB",
    retention="1 week",
    compression="zip"
)

def get_logger(name: str):
    """
    Get a logger instance with a specific name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
