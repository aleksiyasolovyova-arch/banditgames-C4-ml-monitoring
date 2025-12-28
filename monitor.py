"""
Connect4 ML Monitor - Performance Analytics Microservice

Analyzes player performance and generates comparison reports.
"""

import os
import logging
import time
from performance_monitor import PlayerPerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main monitoring loop"""
    # Configuration from environment
    postgres_host = os.getenv('POSTGRES_HOST', 'platform_postgres')
    postgres_port = os.getenv('POSTGRES_PORT', '5432')
    postgres_db = os.getenv('POSTGRES_DB', 'postgres')
    postgres_user = os.getenv('POSTGRES_USER', 'user')
    postgres_password = os.getenv('POSTGRES_PASSWORD', 'password')
    
    output_dir = os.getenv('OUTPUT_DIR', '/workspace/reports')
    interval_hours = int(os.getenv('REPORT_INTERVAL_HOURS', '1'))
    
    # Build connection string
    postgres_conn_str = (
        f"postgresql://{postgres_user}:{postgres_password}@"
        f"{postgres_host}:{postgres_port}/{postgres_db}"
    )
    
    logger.info("=" * 80)
    logger.info("CONNECT4 ML MONITOR STARTED")
    logger.info("=" * 80)
    logger.info(f"PostgreSQL: {postgres_host}:{postgres_port}/{postgres_db}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Report interval: {interval_hours} hour(s)")
    logger.info("=" * 80)
    
    # Create monitor
    monitor = PlayerPerformanceMonitor(
        output_dir=output_dir,
        postgres_conn_str=postgres_conn_str
    )
    
    # Main loop
    while True:
        try:
            logger.info("Generating performance report...")
            
            # Generate full report
            report_dir = monitor.generate_full_report()
            
            logger.info(f"Report generated: {report_dir}")
            logger.info(f"Waiting {interval_hours} hour(s) until next report...")
            
            # Sleep
            time.sleep(interval_hours * 3600)
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            logger.info(f"Retrying in {interval_hours} hour(s)...")
            time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    main()
