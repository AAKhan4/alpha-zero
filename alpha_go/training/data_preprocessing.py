from data.data_parser.processor import DataProcessor
from games.base_game import BaseGame
from games.go.go import Go
import os

# Define directories for raw and processed data
raw_data_dir = "../data/raw_data/go/9x9"
processed_data_dir = "../data/processed_data/go/9x9"

# Create processed data directory if it doesn't exist
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

game = Go()  # Initialize game to process data for

# Process the raw data and save it in the processed data directory
processor = DataProcessor(game, raw_data_dir, processed_data_dir)
processor.process_data()