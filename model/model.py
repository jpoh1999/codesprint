from libs.utils import *
from libs.constants import CONFIG_FILE_PATH
from sortedcontainers import SortedList

pd.set_option('display.max_columns', None)

class Model() :
    """
        Smart Shuffling Model based on Deep Reinforcement Learning Approach 
    """

    def __init__(self, logger, model_config, height_factor, stop_factor : int = 10000, max_moves = 10, depth : int = 5, epochs : int = 1000, random : bool = False, penalty_per_move : int = 10000) :
        """
            Define all configuration variables
        """
        self.model_configs = model_config
        self.logger = logger
        self.weight_class = {"H": 3, "M": 2, "L": 1}
        self.opr_factor = 0.75
        self.height_factor = height_factor
        self.stowage_factor = sum(self.model_configs.values()) - self.model_configs['empty_row_factor'] 
        self.early_stop_factor = stop_factor
        self.max_moves = max_moves
        self.depth = depth
        self.epochs = epochs
        self.random = random
        self.penalty_per_move = penalty_per_move


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
                row_stowage_score += self.get_stowage_score_containers(top_container, bot_container) * self.height_factor[j] 
        
        if slot_max_level == -1:
            row_stowage_score = self.model_configs["empty_row_factor"] # High penalty for empty rows

        return row_stowage_score
    
    def make_move(self, df, i: int, j: int, level_i: int, level_j: int):
        """
        Simulate a move by moving the top element from rows[i] to rows[j].
        Modify both the rows and rows_scores at indices i and j.
        
        Args:
            - df: DataFrame representing the container layout
            - i, j: Indices of the rows to "move" or swap.
            - level_i: The level of the top element in column i.
            - level_j: The level of the top element in column j.
        
        Returns:
            - new_rows_scores: A new list of rows_scores after the move.
        """
        # Move the top element from row[i] to row[j + 1]    
        df.loc[level_j, j], df.loc[level_i, i] = df.loc[level_i, i], ""  # Empty the original top position

        return
    
    # Helper function to get the top index and increment for moves
    def get_top_and_increment(self, df, stack):
        top = df[df[stack] != ""].index[0] if not df[df[stack] != ""].empty else None
        return top, (top + 1) if top is not None else 1

    def backtrack(self, df, left_stack, aux_stack, right_stack, threshold, depth=0, moves=None):
        """
        Recursive function that attempts to reduce stowage scores for a given set of rows.

        Args:
            - df: DataFrame representing the container layout (immutable).
            - row_scores: List of stowage scores for the rows.
            - left_stack: List representing the left stack (source).
            - aux_stack: List representing the auxiliary stack.
            - right_stack: List representing the right stack (destination).
            - local_threshold: Target score threshold for current row.
            - global_threshold : Threshold for all rows in the slot
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
        # best_row_scores = [self.calculate_stowage_score_row(df, row) for row in [left_stack, aux_stack, right_stack]]
        best_score = self.get_score_for_used_rows(df)
        best_moves = moves[:]
        best_df = df

        # Return if the current configuration meets the threshold
        # if best_score <= threshold :
        #     return best_score, moves, df

        # Return if depth limit is reached
        if depth > self.depth:
            return best_score, best_moves, df        

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
                self.make_move(new_df, src_stack, dest_stack, top_src[0], dest_index)

                curr_score = self.get_score_for_used_rows(new_df)

                new_moves = moves + [(src_stack, top_src[0], dest_stack, dest_index, curr_score)]  # Track the move
                
                # Recursively attempt to improve the state
                result_score, result_moves, temp = self.backtrack(new_df, left_stack, aux_stack, right_stack, threshold, depth + 1, new_moves)

                # Update best_score and best_moves if needed
                if result_score + (self.stowage_factor * threshold) <= best_score :
                    best_score = result_score
                    best_moves = result_moves
                    best_df = temp

        # Return the best scores encountered and the moves
        return best_score, best_moves, best_df

    def get_score_for_used_rows(self, df):
        non_empty_rows = [row for row in df.columns if not df[df[row] != ""].empty]
        best_row_scores = [self.calculate_stowage_score_row(df, row) for row in non_empty_rows]

        best_score = sum(best_row_scores)
        return best_score
    
    def update_affected_moves(self, start, end , move_list, to_add, curr_row, curr_level) :
        """
        Update the affected rows after pruning

        Args :
            - pruned_move_list list[tuple] : the original list of moves 
            - end (int) : the ending index where rows/moves will be affected
            - start (int) : the starting index where rows/moves will be affected. 
            - new_pruned_move_list list[tuple]: the copy of the original list we will be updating
            - curr_move_fm_row (int) : the row we will be pruning from
            - curr_move_to_level (int) : the new updated level

        Returns :
            None, Update changes to new_pruned_move_list 
        
        Example :
            After pruning x = (3,2,1,3) at index start - 1
                          ....  ....
                          (5,4,1,4) --> affected row 1
                          (1,4,2,3) --> affected row 2
                          from y = (1,3,1,2) --> (3,2,1,2) : we need to change previous (1,4) to (1,3) because we pruned x,
                          all to_moves, that were on top of x have to be brought down by 1 level
                          and all from_moves needs to be updated too

        """
        temp = curr_level # Create a dummy variable for the curr_move_to_level
        pruned_move_list = move_list.copy()
        # self.logger.debug(f"Affected row = {curr_row}")
        # self.logger.debug(f"Starting temp = {temp} : {start} to {end}")
        for k in range(start + 1, end) : # Loop from the earlier affected row to current row

            if to_add[k] == False :
                continue

            move = pruned_move_list[k]

            if move[2] == curr_row :
                
                pruned_move_list[k] = (move[0], move[1],move[2], temp, move[4]) 
                temp += 1
            
            if move[0] == curr_row:
                temp -= 1
                pruned_move_list[k] = (move[0], temp, move[2], move[3], move[4]) 

                

        return pruned_move_list
        

    def prune_moves(self, moves):
        """
            Prune the moves list to reduce to only useful moves

            Args :
                - moves (list[tuple])
            
            Returns :
                - pruned_move_list
        """
        pruned_moves = []  # Start with an empty list
        to_add = [True] * len(moves)  # Boolean array to track moves to be added

        for i in range(len(moves)):
            if not to_add[i]:
                continue
            
            prev_move = moves[i]  # Keep the current move as is
            seen = set()
            

            for j in range(i + 1, len(moves)):
                if not to_add[j]:
                    continue
                
                current_move = moves[j]  # Get the current move
                if prev_move[0] == current_move[0] :
                    break
                # Check if the last pruned move's destination matches the current move's start
                if prev_move[2:4] == current_move[0:2] and not current_move[2] in seen:
                    
                    seen.add(prev_move[0])
                    
                    if prev_move[0:2] == current_move[2:4]:
                        to_add[i] = False
                        to_add[j] = False
                        break

                    # Merge the moves by creating a new tuple
                    # Keep the first 4 elements from prev_move and the last elements from current_move
                    prev_move = prev_move[0:2] + current_move[2:]

                    to_add[j] = False  # Mark this move as merged and not to be added

                    # the affected cell would be [current_move[:2]]
                    # [1,4,2,5]
                    # [2,5,3,2]
                    # We eliminate the diags [2,5]
                    # Now for all i to j , we have to update all moves to/from row 2 and 5
                 
                    curr_row = current_move[0]
                    curr_level = current_move[1]

                    moves = self.update_affected_moves(i, len(moves), moves, to_add, curr_row, curr_level)
        
                else :
                    seen.add(current_move[2])
                    seen.add(current_move[0])


            if to_add[i] :
                # Append the merged move (as a tuple) to pruned_moves
                pruned_moves.append(prev_move)

        return pruned_moves


    def solve(self, slot_profile : pd.DataFrame, slot_name : int, output_dir : str) :
        """
            The main function for the model run

            Args :
                - df (pd.DataFrame) : the dataframe of slot profile
                - slot_name (int) : the 
        """

        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{slot_name}.txt"
        n = slot_profile.shape[1]
        row_scores = [0] * (n + 1) # dictionary for 3 to n containers shuffling
        sorted_rows = SortedList(key=lambda row: row_scores[row])

        initial_score = 0
        
        if (self.get_operating_factor(slot_profile) > self.opr_factor * slot_profile.shape[0] * slot_profile.shape[1]) :
            self.logger.info("No point shuffling slot... Operating capacity > 75% of the slot")
            return

        for row in slot_profile.columns: # from 1 to 10
            
            score = self.calculate_stowage_score_row(slot_profile, row)
            row_scores[row] = score
            initial_score += score
            sorted_rows.add(row)
        
        self.logger.info(f"Before shuffling scores, the score of slot {slot_name} is {initial_score}")
        # self.logger.debug(df)

        moves, final_df = self.run_model(slot_profile, sorted_rows)
        pruned_moves = self.prune_moves(moves)
        
        self.write_sol_to_txt(slot_name, output_file, pruned_moves)

        final_scores = sum(self.calculate_stowage_score_row(final_df, row) for row in final_df.columns)
        reduction = initial_score - final_scores


        self.logger.info(f"After shuffling, the final score of the slot {slot_name} is {final_scores}")
        self.logger.info(f"Reduction in score = for slot_profile {slot_name} is {reduction}")
        self.logger.info("The moves made are :")
        for move in pruned_moves :
            self.logger.info(move)
        self.logger.info("------------------- END OF SHUFFLE ------------------")
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
        columns = ["slot","fm_row","fm_level","to_row","to_level","score"]
        
        # Write to the file
        with open(output_file, 'w') as f:
            # Write the header
            f.write(', '.join(columns) + '\n')
            
            # Write each tuple
            for row in moves:
                f.write(slot_name + ', ' + ', '.join(map(str, row)) + '\n')

    def run_model(self, df : pd.DataFrame, sorted_rows : SortedList):
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

        # Initialize separate lists for empty and sorted_rows for greedy_sol :
        empty_rows = [row for row in sorted_rows if df[df[row] != ""].empty]
        sorted_rows = [row for row in sorted_rows if not df[df[row] != ""].empty]

        # To store the moves made
        all_moves = []
        n = df.columns.shape[0]
        
        best_score = sum([self.calculate_stowage_score_row(df, row) for row in sorted_rows])
        epoch = 0
        while (epoch < self.epochs) :

            epoch += 1

            # Check the stopping conditions :
            # 1 : nothing left to shuffle, we managed to fix all overstows
            # 2 : early_stopping factor
            # 3 : max_moves
            if (len(sorted_rows) == 0) or len(all_moves) > self.max_moves:
                self.logger.info("Done")
                break
            if self.random :
                left, aux, right = self.get_random_trios(sorted_rows, 1, n)
            else :
                left, aux, right = self.get_greedy_trios(sorted_rows, empty_rows)

            rows = [left, aux, right] # use for greedy solution
            # self.logger.debug(rows)
            # current_score = [self.calculate_stowage_score_row(final_df, row) for row in rows]
            # self.logger.debug(f"Current score for selected rows: {sum(current_score)}")

            # Perform backtracking to optimize the selected rows
            curr_score, curr_moves, curr_state = self.backtrack(final_df, left, aux, right, 10)
            self.logger.info(f"The current score at step {epoch} = {curr_score}")
            curr_score += self.penalty_per_move * len(curr_moves)

            # Stopping conditions for greedy and random :
            if (not self.random and curr_score >= best_score) :
                break
            
            elif self.random and curr_score + self.early_stop_factor > best_score: # equivalent to improvement score < self.stop_factor
                break
            
            else :
                final_df = curr_state
                best_score = curr_score
                all_moves.extend(curr_moves) 
                
            # Re-add the rows back to sorted_rows
            for row in rows:
                if final_df[final_df[row] != ""].empty :
                    sorted_rows.append(row)  # Use append instead of add, since sorted_rows is a list
                else :
                    empty_rows.append(row)

        return all_moves, final_df

    def evaluate(self, file_list : list, input_dir : str, out_dir : str):
        scores_list = []
        total_reduction = 0
        
        os.makedirs(out_dir, exist_ok = True)
        
        for file in file_list:
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path, header=0, index_col=0, na_filter=False)

            # Convert the index to int
            df.index = df.index.astype(int)

            # Convert the columns to int
            df.columns = df.columns.astype(int)

            slot_name = file.split(".")[0]
            
            initial_score, final_score, reduction = self.solve(df, slot_name, out_dir)

            total_reduction += reduction

        scores_list.append([slot_name, initial_score, final_score, reduction])
        
        scores_df = pd.DataFrame(scores_list, columns=["slot", "initial_score", "final_score", "reduction"])
        
        return scores_df, total_reduction

    def tune_model(self, file_list : list, in_dir : str, out_dir : str) :
        """
            Do a grid search to find the best hyper parameters for BT model

            We will use stowage score reduction as an evaluation score 

            Hyper-parameters :
                stowage_scores : [][] 2d
                height_factors : [][] 2d
                weight_factors : [][] 2d
                penalty_per_move : [] 1d
                epochs : [] 1d
                max_moves : [] 1d
                depth : [] 1d
        """
        self.logger.info("STARTING TUNING...")
        best_score = float("inf")
        time_grid = [100,1000,10000,100000]
        weight_grid = [1, 50, 200, 500, 1000]
        ctype_grid = [1, 50, 200, 500, 1000]
        mark_grid = [1, 50, 200, 500, 1000]
        empty_row_grid = [1000, 5000, 10000, 200000, 10000000]
        depth_grid = [1, 3, 5, 7, 10]
        epochs_grid = [1, 30, 100, 500, 1000]
        max_moves_grid = [1, 3, 5, 10, 15]
        penalty_per_moves_grid = [1, 50, 300, 500, 1000]
        stop_grid = [100,10000,100000,1000000,10000000]

        height_grid = [[1, 100, 1000, 10000, 1000000, 10000000],
                  [1, 10, 100, 200, 500, 10000],
                  [1, 200, 300, 5000, 10000, 1000000]]
    

        param_combinations = list(itertools.product(depth_grid, 
                                                    epochs_grid, 
                                                    time_grid, 
                                                    weight_grid, 
                                                    ctype_grid, 
                                                    mark_grid, 
                                                    empty_row_grid, 
                                                    max_moves_grid, 
                                                    penalty_per_moves_grid, 
                                                    height_grid, 
                                                    stop_grid))

        for i,comb in enumerate(param_combinations) :
            depth, epochs, time, weight, ctype, mark, empty_row, max_moves, penalty_per_moves, height, stop = comb
            model_config = {
                "time_overstow_factor" : time,
                "weight_overstow_factor" : weight,
                "type_overstow_factor" : ctype,
                "mark_overstow_factor" : mark,
                "empty_row_factor" : empty_row
            }
            model_name = f"Model_{i}"
            log_file_dir = "model/tune"

            out_path = f"{out_dir}/{model_name}"

            logger = configure_logger(model_name, f"{log_file_dir}/{model_name}.log")

            print(model_config)
            model = Model(logger, 
                          model_config=model_config, 
                          height_factor=height,
                          stop_factor=stop,
                          max_moves=max_moves,
                          depth=depth,
                          epochs=epochs,
                          penalty_per_move=penalty_per_moves
                          )
            
            _, curr_score = model.evaluate(file_list, in_dir, out_path)

            self.logger.info(f"Current score for {model_name} : {curr_score}...")
            if curr_score < best_score :
                best_score = curr_score
                self.model_configs = model.model_configs
                self.update_config() # Update the parameters of the config file with the best score
            
            self.logger.info(f"END OF FINE TUNING ... BEST SCORE : {best_score}")

    def update_config(self) :
        # Write the updated config back to the YAML file
        with open(CONFIG_FILE_PATH, 'w') as file:
            yaml.dump(self.model_configs, file, default_flow_style=False)

    def get_random_trios(self, sorted_rows, start, end, count=2):
        """
            Helper function to randomly select two rows

            Args :
                sorted_rows : 
        """
        left = sorted_rows.pop(-1)

        possible_values = [i for i in range(start, end + 1) if i != left]
        
        if len(possible_values) < count:
            msg = "Not enough unique integers available."
            self.logger.error(msg)
            raise ValueError(msg)

        # Randomly sample for aux and right without replacement
        aux, right = random.sample(possible_values, count)

        return left, aux, right


    def get_greedy_trios(self, sorted_rows, empty_rows):
        """
            Greedy helper function to greedily get 3 rows

            Args :
                sorted_rows : the queue of sorted rows
                empty_rows : the list of empty rows, we prioritize empty rows
            
            Returns :
                left, aux, right : trio of stacks to shuffle
        """
        left = sorted_rows.pop(-1)
        # Check for empty rows for right, or grab from sorted_rows
        if empty_rows:
            aux = empty_rows.pop(0)  # Get an empty row for the aux stack
        else :
            aux = sorted_rows.pop(0)  # Get the next highest usable row
        
        # Check for empty rows for right, or grab from sorted_rows
        if empty_rows:
            right = empty_rows.pop(0)  # Get an empty row for the right stack
        else:
            right = sorted_rows.pop(0)  # Get the next highest usable row
        
        return left,aux,right
    
