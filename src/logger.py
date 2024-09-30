import logging
import os
from datetime import datetime

# Specifying logs directory at the root level
logs_dir = os.path.join("C:\\ML Projects", "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Basic logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s %(lineno)d %(name)s - %(message)s",
    level=logging.INFO,
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(lineno)d %(name)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

if __name__ == "__main__":
    logging.info("Logging has started")
