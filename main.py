from libs.containers_generator import generate_and_convert_to_long_table
from libs.utils import *
import os

CONFIG_PATH = ".config.yaml"

# Example usage
if __name__ == "__main__":

    
    config = load_config_file(CONFIG_PATH)

    container_file_name = f"{config["input"][0]}.parquet"

    # Make directories for input and output
    os.makedirs("input/", exist_ok = True)
    os.makedirs("output/", exist_ok = True)
    
    # Generate 50,000 variations and convert them to a long table
    df_long_table = generate_and_convert_to_long_table(num_variations=50000, num_rows=10, num_cols=6)
    
    # Show the first few rows of the DataFrame
    print(df_long_table.head())
    
    # Optionally, save the DataFrame to a Parquet file
    df_long_table.to_parquet(container_file_name, index=False)