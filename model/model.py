import pandas as pd
import os
from libs.utils import load_config_file
from libs.constants import CONFIG_FILE_PATH
from sortedcontainers import SortedList

pd.set_option('display.max_columns', None)

class Model() :
    """
        Smart Shuffling Model based on Deep Reinforcement Learning Approach 
    """

    def __init__(self) :
        """
            Define all configuration variables
        """
        self.model_configs = load_config_file(CONFIG_FILE_PATH)['model']
        self.weight_class = {"H": 3, "M": 2, "L": 1}
        self.opr_factor = 0.75
        self.height_factor = [1, 100, 1000, 10000, 1000000, 10000000]


    def get_stowage_score_containers(self, top_container, bot_container) :
        """
            Containers are in the form of "starttime:endtime_mark_type_weight"

            The score derived from the two containers are based on the following rules :
                if top_container's endtime - bot_container's endtime >= 2 : add self.model_configs["time_overstow_factor"]
                if top_container's weight is not in the correct order : add self.model_configs["weight_overstow_factor"] 
                if top_container's type is not equal to bot_container's type : add self.model_configs["type_overstow_factor"]
                if top_container's mark is not equal to bot_container's mark : add self.model_configs["mark_overstow_factor"]
            
            Args :
                - top_container (str) : the str of the top container
                - bot_container (str) : the str of the bot container
            
            Returns :
                total_score (int) : the stowage score between the two containers
        """

        # Extract the components from the containers
        top_time, top_mark, top_type, top_weight = top_container.split("_")
        bot_time, bot_mark, bot_type, bot_weight = bot_container.split("_")
        
        # Convert the time and weight to numeric types for comparison
        _, top_endtime = top_time.split(":")
        bot_starttime, _ = bot_time.split(":")
        top_endtime = int(top_endtime)
        bot_starttime = int(bot_starttime)
        top_weight = int(self.weight_class[top_weight])
        bot_weight = int(self.weight_class[bot_weight])
        
        total_score = 0
        
        # Rule 1: Check time difference
        if top_endtime - bot_starttime >= 2:
            total_score += self.model_configs["time_overstow_factor"]
        
        # Rule 2: Check if weight is in the correct order
        if top_weight > bot_weight:
            total_score += self.model_configs["weight_overstow_factor"]
        
        # Rule 3: Check if types are different
        if top_type != bot_type:
            total_score += self.model_configs["type_overstow_factor"]
        
        # Rule 4: Check if marks are different
        if top_mark != bot_mark:
            total_score += self.model_configs["mark_overstow_factor"]
        
        return total_score

    
    def get_operating_factor(self, df: pd.DataFrame):
        """
            This function returns the number of non-empty cells in the given DataFrame.
            
            Args:
                - df: The DataFrame to analyze.
            
            Returns:
                - int: The total number of non-empty cells in the DataFrame.
        """
        # Count the non-empty cells in each column and sum them up
        non_empty_cells = (df != "").sum().sum()
        
        return non_empty_cells

    def calculate_stowage_score_row(self, df : pd.DataFrame, row : int):
        """
            Calculate the stowage scores for the containers in this row

            Args :
                - df (pd.DataFrame) : the dataframe of the slot layout
                - row (int) : the row of the current slot 
            
            Returns :
                row_stowage_score (int) : stowage score for current row
        """

        row_stowage_score = 0
        slot_max_level = -1
    
        for i,k in enumerate(df.index): 
            top_container = df.loc[k,row]
            if top_container == "" :
                continue
            
            if slot_max_level == -1: 
                slot_max_level = i
            
            for j in df.index[i+1:]:
                bot_container = df.loc[j, row]
                row_stowage_score += self.get_stowage_score_containers(top_container, bot_container)
        
        row_stowage_score *= self.height_factor[slot_max_level] 
        
        if slot_max_level == -1:
            row_stowage_score = self.model_configs["empty_row_factor"] # High penalty for empty rows

        return row_stowage_score
    
    def make_move(self, df, rows_scores: list, rows: list, i: int, j: int, level_i: int, level_j: int):
        """
        Simulate a move by moving the top element from rows[i] to rows[j].
        Modify both the rows and rows_scores at indices i and j.
        
        Args:
            - df: DataFrame representing the container layout.
            - rows_scores: List of scores associated with each row.
            - rows: List of actual rows (with corresponding data).
            - i, j: Indices of the rows to "move" or swap.
            - level_i: The level of the top element in column i.
            - level_j: The level of the top element in column j.
        
        Returns:
            - new_rows_scores: A new list of rows_scores after the move.
        """
        # Copy the rows_scores to create new ones for swapping
        new_rows_scores = rows_scores[:]
        # Move the top element from row[i] to row[j + 1]    
        df.loc[level_j, j], df.loc[level_i, i] = df.loc[level_i, i], ""  # Empty the original top position

        # Recalculate the stowage scores for rows i and j after the move
        new_rows_scores[i] = self.calculate_stowage_score_row(df, i)
        new_rows_scores[j] = self.calculate_stowage_score_row(df, j)
        
        return new_rows_scores
    
    # Helper function to get the top index and increment for moves
    def get_top_and_increment(self, df, stack):
        top = df[df[stack] != ""].index[0] if not df[df[stack] != ""].empty else None
        return top, (top + 1) if top is not None else 1

    def backtrack(self, df, row_scores, left_stack, aux_stack, right_stack, threshold, best_moves, depth=0, best_score=float('inf'), moves=None):
        """
        Recursive function that attempts to reduce stowage scores for a given set of rows.

        Args:
            - df: DataFrame representing the container layout (immutable).
            - row_scores: List of stowage scores for the rows.
            - left_stack: List representing the left stack (source).
            - aux_stack: List representing the auxiliary stack.
            - right_stack: List representing the right stack (destination).
            - threshold: Target score threshold.
            - depth: Current recursion depth (for controlling the recursion limit).
            - best_score: The lowest score seen so far.
            - moves: List of moves made to reach the current state (i.e., row swaps).

        Returns:
            - The full row_scores of each row after optimization.
            - A list of moves that led to the best state.
        """
        
        if moves is None:
            moves = []

        # Calculate the current stowage scores
        curr_row_scores = [self.calculate_stowage_score_row(df, row) for row in [left_stack, aux_stack, right_stack]]
        current_score = sum(curr_row_scores)

        # Return if the current configuration meets the threshold
        if current_score <= threshold:
            return current_score, moves, df

        # Return if depth limit is reached
        if depth > 5:
            return best_score, best_moves, df

        # Update the best state seen so far
        if current_score < best_score:
            best_score = current_score
            best_moves = moves[:]

        # Attempt moves, prioritizing empty stacks
        for src_stack, dest_stack in [(left_stack, aux_stack), (aux_stack, right_stack), (right_stack, left_stack)]:
            top_src, top_dest = self.get_top_and_increment(df, src_stack), self.get_top_and_increment(df, dest_stack)

            # Check if source has a disk to move and prioritize empty stacks for destination
            if top_src[0] is not None:

                # If destination stack is empty, use it as a target
                dest_index = top_dest[1]

                if dest_index == len(df) :
                    continue
                
                # Create a new DataFrame copy to perform the move
                new_df = df.copy()

                # Make the move in the copied DataFrame
                new_row_scores = self.make_move(new_df, row_scores, [left_stack, aux_stack, right_stack], src_stack, dest_stack, top_src[0], dest_index)
                new_moves = moves + [(src_stack, top_src[0], dest_stack, dest_index)]  # Track the move
                
                # Recursively attempt to improve the state
                result_score, result_moves, temp = self.backtrack(new_df, new_row_scores, left_stack, aux_stack, right_stack, threshold, best_moves, depth + 1, best_score, new_moves)

                # Update best_score and best_moves if needed
                if result_score < best_score:
                    best_score = result_score
                    best_moves = result_moves
                    df = temp

        # Return the best scores encountered and the moves
        return best_score, best_moves, df
    
    def solve(self, df : pd.DataFrame, slot_name : int, output_dir : str) :
        """
            The main function for the model run

            Args :
                - df (pd.DataFrame) : the dataframe of slot profile
                - slot_name (int) : the 
        """
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{slot_name}.txt"
        n = df.shape[1]
        row_scores = [0] * (n + 1) # dictionary for 3 to n containers shuffling
        sorted_rows = SortedList(key=lambda row: row_scores[row])

        initial_score = 0
        
        if (self.get_operating_factor(df) > self.opr_factor * df.shape[0] * df.shape[1]) :
            print("No point shuffling slot... Operating capacity > 75% of the slot")
            return

        for row in df.columns: # from 1 to 10
            
            score = self.calculate_stowage_score_row(df, row)
            row_scores[row] = score
            if not df[df[row] != ""].empty :
                initial_score += score
            
            sorted_rows.add(row)
        
        # print(f"Before shuffling scores {initial_score}")
        # print(df)

        moves, final_df = self.greedy_sol(df, row_scores, sorted_rows)
        
        self.write_sol_to_txt(slot_name, output_file, moves)

        final_scores = sum(self.calculate_stowage_score_row(final_df, row) for row in sorted_rows)
        reduction = initial_score - final_scores

        # print("After shuffling")
        # print(final_scores)
        print(f"Reduction in score = for slot_profile {slot_name} is {reduction}")

        return initial_score, final_scores, reduction

    def write_sol_to_txt(self, slot_name : int, output_file : str, moves : list):
        """
            Helper function to write file to txt

            Args :
                slot_name (str) : the name of the slot
                output_file (str) : the name of the output file
                moves (list[tuple]) : the moves made
            
            Returns :
                None
        """
        columns = ["slot","fm_row","fm_level","to_row","to_level"]
        
        # Write to the file
        with open(output_file, 'w') as f:
            # Write the header
            f.write(', '.join(columns) + '\n')
            
            # Write each tuple
            for row in moves:
                f.write(slot_name + ', ' + ', '.join(map(str, row)) + '\n')

    def greedy_sol(self, df : pd.DataFrame, row_scores : list, sorted_rows : SortedList):
        """
            The greedy's iteration to solve the slot profile
            It try to resolve a slot profile by
            using empty rows as auxillary/dest rows if possible
            if not , we find the next rows with the lowest scores

            Args :
                df (pd.DataFrame) : the dataframe of shuffle slots
                row_scores (list) : the list of row scores
                sorted_rows (sortedlist) : the list of rows that are not empty
            
            Returns :
                all_moves (list[tuple]), final_df (pd.DataFrame) : the solution of moves and the final layout 
        """
        final_df = df.copy()
        # Initialize a list to hold empty rows based on the DataFrame
        empty_rows = [row for row in sorted_rows if df[df[row] != ""].empty]

        all_moves = []

        # Remove all empty rows from sorted_rows
        sorted_rows = [row for row in sorted_rows if not df[df[row] != ""].empty]

        all_moves = []
        n = len(row_scores)   # Number of valid rows available

        for i in range(n - 3):

            # Check to get the largest row not in empty_rows for the left stack
            if (len(sorted_rows) == 0) :
                print("No rows to shuffle!")
                break

            left = sorted_rows.pop(-1)
        
            # Check for empty rows for aux, or grab from sorted_rows
            if empty_rows:
                aux = empty_rows.pop(0)  # Get an empty row for the aux stack
            else:
                aux = sorted_rows.pop(0)  # Get the next highest usable row

            # Check for empty rows for right, or grab from sorted_rows
            if empty_rows:
                right = empty_rows.pop(0)  # Get an empty row for the right stack
            else:
                right = sorted_rows.pop(0)  # Get the next highest usable row

            # Prepare rows for scoring and backtracking
            rows = [left, aux, right]
            # print(rows)
            current_score = [self.calculate_stowage_score_row(final_df, row) for row in rows]
            # print(f"Current score for selected rows: {sum(current_score)}")

            # Perform backtracking to optimize the selected rows
            best_score, best_moves, temp = self.backtrack(final_df, row_scores, left, aux, right, 0, best_score=sum(current_score), best_moves=[])
                        
            if (len(best_moves) > 0) :
                final_df = temp
            # Collect all moves from the best combination
            all_moves.extend(best_moves)

            # Re-add the rows back to sorted_rows
            for row in rows:
                if final_df[final_df[row] != ""].empty :
                    sorted_rows.append(row)  # Use append instead of add, since sorted_rows is a list
                else :
                    empty_rows.append(row)

        return all_moves, final_df
        