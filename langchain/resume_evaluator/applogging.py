import logging

# 1. Define a Custom Level for Success (between INFO and WARNING)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kws)

# Add the success method to the Logger class dynamically
logging.Logger.success = success

class CustomFormatter(logging.Formatter):
    # Standard Colors
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Custom/Special Colors (Using 256-color or RGB)
    # Orange isn't a standard 16-color ANSI, but you can use RGB:
    ORANGE = "\033[38;2;255;165;0m" 

    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    # Symbols using Unicode
    CHECK = "\u2714"
    CROSS = "\u2716"
    INFO_CIRC = "\u2139" # Small 'i' symbol

    # Define formats for different levels
    # %(message)s is the standard placeholder for your text
    FORMATS = {
        logging.INFO: f"{YELLOW}{BOLD}{INFO_CIRC}{RESET}  %(message)s",
        SUCCESS_LEVEL: f"{GREEN}{BOLD}{CHECK}{RESET}  %(message)s",
        logging.ERROR: f"{RED}{BOLD}{CROSS}{RESET}  %(message)s",
        logging.WARNING: f"{GRAY}{BOLD}!(WARNING){RESET} %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, "%(levelname)s: %(message)s")
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    
    return logger

log = get_logger("test_app")

if __name__ == "__main__":
    # Your logic here
    log.info(f"Loading environment settings")
    log.success("System initialized.")
    log.error("Failed to load .env file.")
    log.warning("You are out of memory")
    
