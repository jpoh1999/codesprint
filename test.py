import unittest
import pandas as pd
from libs.utils import *
from libs.constants import CONFIG_FILE_PATH
from model.model import Model

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
        self.assertEqual(actual_rows > 0, True, f"Expected more than 1 rows, Got 0")
        
        self.assertEqual(actual_cols, self.expected_cols,
                         f"Expected {self.expected_cols} rows, but got {actual_cols}.")

    def test_solution() :
        """
            Test the solution of one file
        """
        model = Model()
    
        config = load_config_file(CONFIG_FILE_PATH)

        output_dir = config["output"][0]
        scores_path= config["output"][1]
        scores_file = f"{scores_path}.csv"

        scores_list = []

        df = pd.read_csv("input/slot_profiles/49.csv", header=0, index_col=0, na_filter=False)

        # Convert the index to int
        df.index = df.index.astype(int)

        # Convert the columns to int
        df.columns = df.columns.astype(int)

        slot_name = "49"
            
        initial_score, final_score, reduction = model.solve(df, slot_name, output_dir)

        scores_list.append([slot_name, initial_score, final_score, reduction])
        scores_df = pd.DataFrame(scores_list, columns=["slot", "initial_score", "final_score", "reduction"])
        scores_df.to_csv(scores_file)

if __name__ == '__main__':
    unittest.main()
