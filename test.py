import unittest
import pandas as pd
from libs.utils import *

CONFIG_FILE_PATH = ".config.yaml"

class TestInputFile(unittest.TestCase):

    def setUp(self):

        config = load_config_file(CONFIG_FILE_PATH)
        self.parquet_file_path = F"{config["input"][0]}.parquet"

        # Define the expected number of rows
        self.expected_rows = 1446552  # Replace XXX with the number of rows you expect
        self.expected_cols = 4

    def test_data_shape(self):
        # Read the Parquet file
        df = pd.read_parquet(self.parquet_file_path)
        
        # Get the actual number of rows
        
        actual_rows = df.shape[0]
        actual_cols = df.shape[1]
       
        
        print(df.head())

        # Assert the actual number of rows equals the expected number
        self.assertEqual(actual_rows, self.expected_rows, 
                         f"Expected {self.expected_rows} rows, but got {actual_rows}.")
        
        self.assertEqual(actual_cols, self.expected_cols,
                         f"Expected {self.expected_cols} rows, but got {actual_cols}.")

if __name__ == '__main__':
    unittest.main()
