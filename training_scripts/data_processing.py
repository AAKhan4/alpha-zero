from data.data_parser.processor import DataProcessor
from games.go.go import Go
import os
from time import time

# Define directories for raw and processed data
raw_data_dir = "./data/raw_data/go/9x9"
processed_data_dir = "./data/processed_data/go/9x9"

# Create processed data directory if it doesn't exist
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

game = Go()  # Initialize game to process data for

# Process the raw data and save it in the processed data directory
timer_start = time()
processor = DataProcessor(game, raw_data_dir, processed_data_dir)
processor.process_data()
timer_end = time()
time_taken = timer_end - timer_start
hours, rem = divmod(time_taken, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Data processing completed in  {int(hours)}h:{int(minutes)}m:{int(seconds)}s.")