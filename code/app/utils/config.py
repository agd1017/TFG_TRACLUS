import os

# Use a relative path with respect to the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the folder to store the uploaded files
UPLOAD_FOLDER = os.path.join(BASE_DIR, "saved_results")

# Optionally, define a path for training data (currently commented out)
# TRAIN_DATA = os.path.join(BASE_DIR, "train_data", "taxis_trajectory", "train_1.xlsx")