from libs.containers_generator import ContainerGenerator
from libs.slot_profile_reader import SlotProfileReader
from libs.utils import *
from libs.constants import CONFIG_FILE_PATH
import os

def generate_dirs() :
    """
        Generate relevant dirs
    """
    os.makedirs("input/", exist_ok = True)
    os.makedirs("output/", exist_ok = True)

def generate_containers() :
    """ Generate containers data """
    config = load_config_file(CONFIG_FILE_PATH)
    container_file_name = f"{config["input"][0]}.parquet"

    # Generate 50,000 variations and convert them to a long table
    cg = ContainerGenerator()
    df_long_table = cg.generate(num_slots=50000, num_rows=10, num_levels=6)
    
    # Show the first few rows of the DataFrame
    # print(df_long_table.head())
    
    # Optionally, save the DataFrame to a Parquet file
    df_long_table.to_parquet(container_file_name, index=False)


def preprocess_containers_data() :
    """ 
        Preprocess the raw containers data from long format
        into a pivot table for each slot
    """
    cr = SlotProfileReader()
    config = load_config_file(CONFIG_FILE_PATH)
    
    slot_profiles_dir = f"{config["input"][1]}"

    cr.read_and_process(slot_profiles_dir)

if __name__ == "__main__":

    # generate_containers()    
    preprocess_containers_data()
    

    
    