"""
Module that provides a centralized logging configuration for the FastAPI application.
It ensures consistent logging across the modules. 

Cristian Piacente
"""

import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a specified name.

    The logger is configured to write logs to fastapi_app.log, located in the challenge_3 directory, 
    
    Parameters:
    - name (str): The name of the logger.

    Returns:
    - logging.Logger: A configured logger instance for the specified name.
    """

    log_file_path = Path(__file__).parent.parent / 'fastapi_app.log'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(name)

    return logger