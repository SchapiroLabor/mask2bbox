import logging
import sys

def set_logger(name=__name__,
               log_level='info',
               log_format=None) -> logging.Logger:
    """
    Sets up a logger with a single StreamHandler to stdout.

    Args:
        name (str): The name of the logger (usually __name__).
        log_level (str): Logging level as a string (e.g., 'info', 'debug', 'error').
        log_format (str, optional): Optional custom logging format. Defaults to a standard format.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Get logger and prevent duplicate handlers
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    # Converts string 'info' into logging.INFO
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Set logger format
    log_format = log_format or '%(asctime)s - %(levelname)s - %(message)s'

    # Apply formating and plug it into the logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    return logger