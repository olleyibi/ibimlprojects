import logging
import os
from datetime import datetime

# Generate a log file name using the current date and time
# Example: "10_19_2025_17_15_45.log"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory where all log files will be stored
logs_path = os.path.join(os.getcwd(), "logs")

# Create the "logs" folder if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Create the full path to the specific log file
# e.g., C:\path\to\project\logs\10_19_2025_17_15_45.log
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging settings
# - filename: path to the log file
# - format: defines how each log entry will look
# - level: minimum logging level (INFO = includes INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] Line:%(lineno)d | %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Optional: log a startup message to confirm logging works
logging.info("Logging has been configured successfully.")
