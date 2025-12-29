import time
import logging
import os
from pathlib import Path
from src.performance_monitor import TensorBoardMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Monitor")

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "/workspace/datasets")
LOG_DIR = os.getenv("LOG_DIR", "/workspace/tensorboard_logs")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60")) # Check every minute

def main():
    logger.info(f" ML TensorBoard Monitor Started")
    logger.info(f" Input: {DATA_DIR}")
    logger.info(f" Output: {LOG_DIR}")

    monitor = TensorBoardMonitor(LOG_DIR)

    while True:
        try:
            # 1. Load Data
            df = monitor.load_data(DATA_DIR)

            # 2. Update TensorBoard
            if not df.empty:
                monitor.update_dashboard(df)
            else:
                logger.warning("Datasets empty. Waiting for data...")

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f" Monitor Loop Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
