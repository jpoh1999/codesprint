from libs.containers_generator import ContainerGenerator
from libs.slot_profile_reader import SlotProfileReader
from model.model import Model
from libs.utils import *
from libs.constants import CONFIG_FILE_PATH
import os
import pandas as pd
import multiprocessing
import time  # For simulating time-consuming functions


def generate_containers() :
    """ Generate containers data """

    config = load_config_file(CONFIG_FILE_PATH)
    container_file_name = f"{config["input"][0]}.parquet"

    # Generate 50,000 variations and convert them to a long table
    cg = ContainerGenerator()
    df_long_table = cg.generate(num_slots=5, num_rows=10, num_levels=6)
    
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

def run_greedy_model() :
    """
        Run model for solving a slot
    """
    greedy_logger = configure_logger('greedy_model_logger', 'greedy_model.log')
    model = Model(greedy_logger)
    print("Starting greedy model....")
    config = load_config_file(CONFIG_FILE_PATH)
    slot_path = config["input"][1]
    greedy_out_dir = config["output"][0]
    greedy_scores_path= config["output"][1]

    scores_file = f"{greedy_scores_path}.csv"

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
        
        initial_score, final_score, reduction = model.solve(df, slot_name, greedy_out_dir)
        
        scores_list.append([slot_name, initial_score, final_score, reduction])
    
    print("Greedy model finished.")

def run_random_model() :
    """
        Run random model for solving a slot
    """
    random_logger = configure_logger('random_model_logger', 'random_model.log')
    print("Starting random model....")
    model = Model(random_logger, random=True)
    
    config = load_config_file(CONFIG_FILE_PATH)
    slot_path = config["input"][1]
    random_out_dir = config["output"][1]
    random_scores_path= config["output"][3]

    scores_file = f"{random_scores_path}.csv"

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
        
        initial_score, final_score, reduction = model.solve(df, slot_name, random_out_dir)
        
        scores_list.append([slot_name, initial_score, final_score, reduction])
    
    print("Random model finished.")

def test_solution() :
    """
        Test the solution of one solution file
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

    slot_name = "43"
        
    initial_score, final_score, reduction = model.solve(df, slot_name, output_dir)

    scores_list.append([slot_name, initial_score, final_score, reduction])
    scores_df = pd.DataFrame(scores_list, columns=["slot", "initial_score", "final_score", "reduction"])
    scores_df.to_csv(scores_file)

def read_moves_file(file_path):
    moves_list = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            # Strip the line of any extra spaces or newline characters, then split by commas
            parts = line.strip().split(',')
            # Convert each part to an integer and form a tuple
            move_tuple = tuple(map(int, parts))
            # Append the tuple to the list
            moves_list.append(move_tuple)
    return moves_list

if __name__ == "__main__":

    # These two processes need to be done in order.
    generate_containers()    
    preprocess_containers_data()
    
    # These other two processes can be done in parallel execution.
    greedy_process = multiprocessing.Process(target=run_greedy_model)
    random_process = multiprocessing.Process(target=run_random_model)
    
    # Start both processes
    greedy_process.start()
    random_process.start()

    # Wait for both processes to complete
    greedy_process.join()
    random_process.join()

    print("Finished running models")
    
    # test_solution()


    
    