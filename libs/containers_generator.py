from libs.utils import *

class ContainerGenerator() :
    """
        Class to generate data for containers as input data for model    
    """

    def __init__(self) -> None:
        """
            Initialize some fixed strings to be used for Procedural Generation
        """
        self.weight = ["H", "M", "L"]
        self.container_type = ["G", "HC"]
        self.portmarks = ["ABCDE", "FGHIJ", "KLMNO", "PQRST"]
        self.logger = configure_logger("Container_Generator", "container_generator.log")

    def generate_random_time(self):
        """Generate a random start and end time, ensuring start < end with a distribution of 80% future and 20% past."""
        current_time = datetime.now()
        future_time = random.choices([True, False], weights=[0.8, 0.2])[0]  # 80% chance for future, 20% for past

        if future_time:
            # Generate a random start time within the next 24 hours
            start_time = current_time + timedelta(days=random.randint(0, 1), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            # Generate an end time that is at least 1 hour after the start time
            end_time = start_time + timedelta(hours=random.randint(1, 5))  # End time between 1 to 5 hours after start time
        else:
            # Generate a random start time within the past 24 hours
            start_time = current_time - timedelta(days=random.randint(0, 1), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            # Generate an end time that is at least 1 hour before the start time
            end_time = start_time - timedelta(hours=random.randint(1, 5))  # End time between 1 to 5 hours before start time

        return start_time.strftime("%Y%m%d_%H%M%S"), end_time.strftime("%Y%m%d_%H%M%S")
    
    def random_container_name(self):
        """ Helper function to generate a random container name """
        letters = ''.join(random.choices(string.ascii_uppercase, k=1))  # Random letter
        digits = ''.join(random.choices(string.digits, k=3))            # Random digits
        
        return letters + digits

    # Function to generate a variation of containers (between 0 and 60 containers)
    def generate_slot(self, num_levels=6, num_rows=10):
        """
            Generate a random valid container layout 

            Args :
                - num_rows [default : 6] (int) : num of rows for the container yard
                - num_cols [default : 10] (int) : num of levels for the container yard

            
            Return :
                grid (list) : the list of the container positions
        """
        # Randomly choose the number of containers between 0 and 60
        num_containers = random.randint(0, num_levels * num_rows)
    
        # Initialize empty grid with num_levels rows and num_rows columns
        grid = [[0 for _ in range(num_rows)] for _ in range(num_levels)]
        
        # Place containers starting from the ground level in random columns
        while (num_containers > 0) :
            row = random.randint(0, num_rows - 1)  # Random column, we fill this entire
            level = random.randint(0, num_levels - 1)  # Random row

            for i in range(row) :
                grid[level][i] = 1
            
            num_containers -= row
            

        return grid
      
    def grid_to_long_table(self, grid, slot_id):
        """
            Transforms generated data into long format to stimulate the real world

            Args :
                - grid (list) : the list of the position of the containers
                - slot_id (int) : the slot number
            
            Returns :
                long_table (list[dict]) : list of container objects  
        """
        num_rows = len(grid)
        num_levels = len(grid[0])
        
        long_table = []
        
        for row in range(num_rows):
            for level in range(num_levels):
                if grid[row][level] == 1:
                    # Generate random start and end times
                    start_time, end_time = self.generate_random_time()
                        
                    # Append the container info to the long table
                    long_table.append({
                        'Start_Time': start_time,
                        'End_Time': end_time,
                        'Weight': random.choice(self.weight),
                        'Type': random.choice(self.container_type),
                        'Mark': random.choice(self.portmarks),
                        'Slot': slot_id,
                        'Level': level + 1,  # Row index starts from 1
                        'Row': row + 1,     # Col index starts from 1
                        'ContainerID': self.random_container_name()
                    })
        
        return long_table
    

    def generate(self, slots=50000, rows=10, levels=6):
        """
            Main function to generate data for stimulation

            Args :
                - num_variations [default = 50000] (int) : the number of slots 
                - num_rows [default = 10] (int) : the number of rows in each slot
                - num_cols [default = 6] (int) : the number of levels for each slot

            Returns :
                df (pd.Dataframe) : the long table for input
        """
        all_slots_data = []
        
        for slot_id in range(slots):
            # Generate grid for each variation
            grid = self.generate_slot(rows, levels)
            
            # Convert the grid to long table format and append to the overall data
            slot_data = self.grid_to_long_table(grid, slot_id)
            all_slots_data.extend(slot_data)
        
        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(all_slots_data)
        
        return df

