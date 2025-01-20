import logging

def setup_logger(name, log_file, level=logging.INFO):
    """
    Setup a logger to write logs to a file.

    Args:
        name (str): Logger name.
        log_file (str): Path to the log file.
        level: Logging level.

    Returns:
        Logger object.
    """
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger