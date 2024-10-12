from libs.containers_generator import ContainerGenerator
from libs.slot_profile_reader import SlotProfileReader
from model.model import Model
from libs.utils import *
from libs.constants import CONFIG_FILE_PATH
import os
import pandas as pd

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
    df_long_table = cg.generate(num_slots=50, num_rows=10, num_levels=6)
    
    # Show the first few rows of the DataFrame
    # print(df_long_table.head())
    
    # Save the DataFrame to a Parquet file
    df_long_table.to_parquet(container_file_name, index=False)

    print("Finished generating containers data!")


def preprocess_containers_data() :
    """ 
        Preprocess the raw containers data from long format
        into a pivot table for each slot
    """
    cr = SlotProfileReader()
    config = load_config_file(CONFIG_FILE_PATH)
    
    slot_profiles_dir = f"{config["input"][1]}"

    cr.read_and_process(slot_profiles_dir)

def run_model() :
    """
        Run model for solving a slot
    """
    model = Model()
    
    config = load_config_file(CONFIG_FILE_PATH)
    slot_path = config["input"][1]
    output_dir = config["output"][0]
    scores_path= config["output"][1]

    scores_file = f"{scores_path}.csv"

    # Read the CSV file
    file_list = [f for f in os.listdir(slot_path) if f.endswith('.csv')]
    scores_list = []
    for file in file_list:
        file_path = os.path.join(slot_path, file)
        df = pd.read_csv(file_path, header=0, index_col=0, na_filter=False)

        # Convert the index to int
        df.index = df.index.astype(int)

        # Convert the columns to int
        df.columns = df.columns.astype(int)

        slot_name = file.split(".")[0]
        
        initial_score, final_score, reduction = model.solve(df, slot_name, output_dir)
        
        scores_list.append([slot_name, initial_score, final_score, reduction])
    
    scores_df = pd.DataFrame(scores_list, columns=["slot", "initial_score", "final_score", "reduction"])
    scores_df.to_csv(scores_file)

if __name__ == "__main__":

    # generate_containers()    
    # preprocess_containers_data()
    run_model()


    
    