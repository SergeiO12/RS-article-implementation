import sys
import os
import logging
from datetime import datetime
from new import main

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log_file = os.path.join(log_dir, log_filename)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

sys.stdout = open(log_file, 'a')
sys.stderr = sys.stdout

# Execute the main function from new.py
if __name__ == "__main__":
        main()