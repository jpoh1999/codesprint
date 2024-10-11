import random
import string
import pandas as pd

# Function to randomly generate a container name
def random_container_name():
    letters = ''.join(random.choices(string.ascii_uppercase, k=1))  # Random letter
    digits = ''.join(random.choices(string.digits, k=3))            # Random digits
    return letters + digits

# Function to generate a variation of containers (between 0 and 60 containers)
def generate_variation(num_rows=10, num_cols=6):
    # Randomly choose the number of containers between 0 and 60
    num_containers = random.randint(0, num_rows * num_cols)
    
    # Initialize empty grid
    grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    
    # Place containers starting from the ground level in random columns
    for _ in range(num_containers):
        col = random.randint(0, num_cols - 1)  # Random column
        row = next((r for r in range(num_rows) if grid[r][col] == 0), None)  # First empty row
        if row is not None:
            grid[row][col] = 1  # Place a container (represented as 1)
    
    return grid

# Function to convert the grid into long table format with random container names
def grid_to_long_table(grid, variation_id):
    num_rows = len(grid)
    num_cols = len(grid[0])
    
    long_table = []
    
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] == 1:
                # For each container, generate a random container name and append to the long table
                long_table.append({
                    'VariationID': variation_id,
                    'Row': row + 1,  # Row index starts from 1
                    'Col': col + 1,  # Col index starts from 1
                    'ContainerID': random_container_name()
                })
    
    return long_table

# Function to generate all variations and convert them into a long table
def generate_and_convert_to_long_table(num_variations=50000, num_rows=10, num_cols=6):
    all_variations_data = []
    
    for variation_id in range(num_variations):
        # Generate grid for each variation
        grid = generate_variation(num_rows, num_cols)
        
        # Convert the grid to long table format and append to the overall data
        variation_data = grid_to_long_table(grid, variation_id)
        all_variations_data.extend(variation_data)
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(all_variations_data)
    
    return df

