from libs.utils import *
from libs.constants import CONFIG_FILE_PATH
from datetime import datetime

class SlotProfileReader() :
    """
        Pivot the dataframe of long table into a valid slot yard slot layout

    """
    def calculate_time_differences(self, row):
        current_time = datetime.now()
        
        # Parse the start_time and end_time from the row
        start_time = datetime.strptime(row['Start_Time'], "%Y%m%d_%H%M%S")
        end_time = datetime.strptime(row['End_Time'], "%Y%m%d_%H%M%S")
        
        # Calculate the time differences in hours
        st = int((current_time - start_time).total_seconds() // 3600)  # Difference in hours from start_time
        end = int((current_time - end_time).total_seconds() // 3600)    # Difference in hours from end_time
        
        return st, end

    def read_and_process(self, output_dir: str):
        """
        Each slot profile is saved as a CSV file named '{slotNumber}.csv'.
        """
        # Make directory for slot profiles if not already created
        
        shutil.rmtree(output_dir) # i used shutil cos i lazy to check if its empty
        os.makedirs(output_dir)

        # Load the configuration file
        config = load_config_file(CONFIG_FILE_PATH)
        containers_file_path = f"{config['input'][0]}.parquet"
        
        # Read the Parquet file
        df = pd.read_parquet(containers_file_path)

        # Validate and group by 'Slot'
        grouped_df = df.groupby('Slot')

        # Process each group and save as CSV
        for slot_number, pivot_data in grouped_df:  # Group by the slot number
            
            # Pivot the DataFrame to arrange containers in the correct layout
            pivot_data = self.validate_no_floating_containers(pivot_data)

            # print(pivot_data)
           
            # Save to CSV
            csv_file_name = f"{output_dir}/{slot_number}.csv"  # CSV file name
            pivot_data.to_csv(csv_file_name)

            print(f"Saved slot profile to {csv_file_name}")

        return
        
    
    def validate_no_floating_containers(self, group):
        # Sort the group by Level and Row to maintain order
        group_sorted = group.sort_values(by=['Level', 'Row'])
        
        # Create a 6x10 DataFrame filled with empty strings
        max_levels = 6
        max_rows = 10
        # Set the index to be inverted (from 6 to 1)
        layout = pd.DataFrame(index=range(max_levels, 0, -1), columns=range(1, max_rows + 1), data="")
        
        # Fill the layout with container IDs where applicable
        for _, row in group_sorted.iterrows():
            level = row['Level']
            container_row = row['Row']
            etb, etu = self.calculate_time_differences(row)
            container_id = f"{etb}:{etu}_{row['Mark']}_{row['Type']}_{row['Weight']}"
            
            # Assign the container ID to the corresponding position
            if 1 <= level <= max_levels and 1 <= container_row <= max_rows:
                layout.at[level, container_row] = container_id

        return layout
    
    