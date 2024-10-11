import unittest
import pandas as pd
from libs.utils import *
from libs.constants import CONFIG_FILE_PATH

class TestInputFile(unittest.TestCase):

    def setUp(self):

        config = load_config_file(CONFIG_FILE_PATH)
        self.parquet_file_path = F"{config["input"][0]}.parquet"

        # Define the expected number of rows
        self.expected_cols = 4

    def test_data_shape(self):
        # Read the Parquet file
        df = pd.read_parquet(self.parquet_file_path)
        
        # Get the actual number of rows
        
        actual_rows = df.shape[0]
        actual_cols = df.shape[1]
    
        # Assert the actual number of rows equals the expected number
        self.assertEqual(actual_rows > 0, True
                         f"Expected more than 1 rows, Got 0")
        
        self.assertEqual(actual_cols, self.expected_cols,
                         f"Expected {self.expected_cols} rows, but got {actual_cols}.")

if __name__ == '__main__':
    unittest.main()
