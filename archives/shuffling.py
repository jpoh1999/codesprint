
# KEEP ALL PREVIOUS ITERATION OF CODES #

def get_three_sum_closest(self, row_scores: list, sorted_index: list, target: int) -> list:
        """
            Finds the three indices from sorted_index whose corresponding row_scores
            sum is closest to the target.
            
            Parameters:
            - row_scores: List of scores corresponding to some items.
            - sorted_index: List of indices corresponding to row_scores.
            - target: The target sum we want to get close to.
            
            Returns:
            - List of three indices whose row_scores sum is closest to the target.
        """
        
        # Sort row_scores (by sorted_index)
        sorted_index.sort(key=lambda x: row_scores[x])
        
        mindist = float('inf')  # To track the minimum distance from the target
        result_indices = [0, 0, 0]  # To store the indices of the closest triplet
        
        # Traverse the sorted row_scores
        for i in range(len(row_scores) - 2):
            left = i + 1
            right = len(row_scores) - 1
            
            while left < right:
                # Calculate the sum for the current triplet of scores
                sum_score = row_scores[sorted_index[i]] + row_scores[sorted_index[left]] + row_scores[sorted_index[right]]
                
                # Update the result if this is closer to the target
                if abs(target - sum_score) < mindist:
                    mindist = abs(target - sum_score)
                    result_indices = [sorted_index[i], sorted_index[left], sorted_index[right]]
                
                # Adjust pointers based on how the sum compares to the target
                if sum_score > target:
                    right -= 1
                else:
                    left += 1
        
        return result_indices

   # def update(self, best_state : tuple, row_scores : list) :
    #     """
    #         Update the row_scores by removing 
    #         the row with the lowest score from the list

    #         Args : 
    #             - best_state (tuple) : the tuple of (row_scores, row_indices) of the 3sums
    #             - row_scores (list) : the list of row_scores
    #             - row (list) : the list of row indices

    #         Returns :

    #             None, update the row_scores with the best_state
    #     """
    #     three_sum_scores, three_sum_indices = best_state
    #     for row in three_sum_indices :
    #         row_scores[row] = three_sum_scores[row]

    def backtrack(self, df, rows_scores, rows, threshold, depth=0, best_score=float('inf'), moves=None, best_moves=None):
        """
        Recursive function that attempts to reduce stowage scores for a given set of rows.
        
        Args:
            - df: DataFrame representing the container layout.
            - rows_scores: List of stowage scores for the rows.
            - rows: List of rows (with scores).
            - threshold: Target score threshold.
            - depth: Current recursion depth (for controlling the recursion limit).
            - best_score: The lowest score seen so far.
            - moves: List of moves made to reach the current state (i.e., row swaps).
            - best_moves: The best moves made to achieve the lowest score.
        
        Returns:
            - The full row_scores of each row after optimization.
            - A list of best moves that led to the best state.
        """
        
        if moves is None:
            moves = []

        if best_moves is None:
            best_moves = []

        # Calculate the current score as the sum of all row scores
        current_score = sum(rows_scores)
        
        # If the current configuration meets the threshold, return the state and the moves
        if current_score <= threshold:
            return rows_scores, moves
        
        # If depth limit or max recursion reached, return the best seen so far
        if depth > 10 or current_score >= best_score * 2:  # Max of 5 moves for optimization
            return rows_scores, best_moves
        
        # Track the best state seen so far
        if current_score < best_score:
            best_score = current_score
            best_moves = moves[:]  # Update the best moves

        # Try recursive steps (Hanoi-like moves)
        for i in rows:
            # Find top_i: The top-most non-empty element in column i (row[i])
            if df[df[i] != ""].empty:
                continue  # Skip empty columns
            
            level_i = df[df[i] != ""].index[0]  # Get the index of the top-most container
            
            for j in rows:
                if i != j:  # Don't swap with itself
                    # Find top_j: The top-most non-empty element in column j (row[j])
                    level_j = df[df[j] != ""].index[0] if not df[df[j] != ""].empty else 0  # Get the index of the top-most container
                    
                    if level_j + 1 == len(df) :
                        break
                    # Try a move (move top element from row[i] to row[j])
                    new_rows_scores = self.make_move(df, rows_scores, rows, i, j, level_i, level_j)
                    
                    # Append the move to the list of moves
                    new_moves = moves + [(i, level_i, j, level_j + 1)]  # Track the move

                    # Recursively attempt to improve the state
                    result_scores, result_moves = self.backtrack(df, new_rows_scores, rows, threshold, depth + 1, best_score, new_moves, best_moves)
                    
                    # If a valid result is found, return it
                    if sum(result_scores) <= threshold:
                        return result_scores, result_moves
                    else:
                        # Backtrack the move in reverse direction
                        new_rows_scores = self.make_move(df, rows_scores, rows, j, i, level_j + 1, level_i - 1)

        # If no solution is found, return the best scores encountered and the moves
        return rows_scores, best_moves

# current_score = [self.calculate_stowage_score_row(df, row) for row in sorted_rows]
        # print(f"Current score : {current_score}")
        # best_scores, best_moves = self.backtrack(df, row_scores, sorted_rows, 100000)
        # best_score = [self.calculate_stowage_score_row(df, row) for row in best_scores]
        # print(f"Best score : {current_score}")
        # return best_moves

def three_sum_sol(self, n, row_scores, slot_score):
        sorted_index = sorted(sorted_index, key=lambda i: row_scores[i])
        
        target = slot_score // n * 3 # this will be our target for threeSums

        for i in range(n - 3) :
            rows = self.get_three_sum_closest(row_scores, sorted_index, target) # this will be the 3 rows selected for solving
            self.backtrack(rows)

def backtrack(self, df, rows_scores, rows, threshold, depth=0, best_score=float('inf'), moves=None, best_moves=None):
        """
        Recursive function that attempts to reduce stowage scores for a given set of rows.
        
        Args:
            - df: DataFrame representing the container layout.
            - rows_scores: List of stowage scores for the rows.
            - rows: List of rows (with scores).
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

        if best_moves is None:
            best_moves = []

        # Calculate the current stowage scores
        curr_row_scores = [self.calculate_stowage_score_row(df, row) for row in rows]
        current_score = sum(curr_row_scores)
        
        # If the current configuration meets the threshold, return the state and the moves
        if current_score <= threshold:
            return rows_scores, best_moves
        
        # If depth limit reached, check for improvement and return best_moves or empty
        if depth > 5:
            if current_score >= best_score:  # No improvement
                return rows_scores, []  # Return an empty move list
            return rows_scores, best_moves  # Return the best moves seen so far
        
        # Track the best state seen so far
        if current_score < best_score:
            best_score = current_score
            best_moves = moves[:]
        
        # Try recursive steps (Hanoi-like moves)
        for i in rows:
            # Find top_i: The top-most non-empty element in column i (row[i])
            if df[df[i] != ""].empty:
                continue  # Skip empty columns
            
            level_i = df[df[i] != ""].index[0]  # Get the index of the top-most container
            
            for j in rows:
                if i != j:  # Don't swap with itself

                    # Find top_j: The top-most non-empty element in column j (row[j])
                    if df[df[j] != ""].empty:
                        level_j = 0  # If column j is empty, we place at level 0 (first level)
                    else:
                        level_j = df[df[j] != ""].index[0]  # Get the index of the top-most container

                    if level_j + 1 == len(df):
                        break

                    # Try a move (move top element from row[i] to row[j])
                    new_rows_scores = self.make_move(df, rows_scores, rows, i, j, level_i, level_j)
                    
                    # Append the move to the list of moves
                    new_moves = moves + [(i, level_i, j, level_j + 1)]  # Track the move

                    # Recursively attempt to improve the state
                    result_score, result_moves = self.backtrack(df, new_rows_scores, rows, threshold, depth + 1, best_score, new_moves)
                    
                    # If a valid result is found, return it
                    if sum(self.calculate_stowage_score_row(df, row) for row in rows) <= threshold:
                        return result_score, result_moves
                    else:
                        new_rows_scores = self.make_move(df, rows_scores, rows, j, i, level_j + 1, level_i - 1)

        # If no solution is found, return the best scores encountered and the moves
        return rows_scores, best_moves

    def backtrack(self, df, row_scores, left_stack, aux_stack, right_stack, threshold, best_moves, depth=0, best_score=float('inf'), moves=None):
        """
        Recursive function that attempts to reduce stowage scores for a given set of rows.

        Args:
            - df: DataFrame representing the container layout.
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
            return current_score, moves

        # Return if depth limit is reached
        if depth > 10:
            return best_score, best_moves

        # Update the best state seen so far
        if current_score < best_score:
            best_score = current_score
            best_moves = moves[:]

        # Helper function to get the top index and increment for moves
        def get_top_and_increment(stack):
            top = df[df[stack] != ""].index[0] if not df[df[stack] != ""].empty else None
            return top, (top + 1) if top is not None else 1

        # Attempt moves, prioritizing empty stacks
        for src_stack, dest_stack in [(left_stack, aux_stack), (aux_stack, right_stack), (right_stack, left_stack)]:
            top_src, top_dest = get_top_and_increment(src_stack), get_top_and_increment(dest_stack)

            # Check if source has a disk to move and prioritize empty stacks for destination
            if top_src[0] is not None:
                # If destination stack is empty, use it as a target
                dest_index = top_dest[1] if top_dest[0] is None else top_dest[0]
                if dest_index is None:  # If destination is empty, set to 1
                    dest_index = 1

                new_row_scores = self.make_move(df, row_scores, [left_stack, aux_stack, right_stack], src_stack, dest_stack, top_src[0], dest_index)
                new_moves = moves + [(src_stack, top_src[0], dest_stack, dest_index)]  # Track the move
                
                # Recursively attempt to improve the state
                result_score, result_moves = self.backtrack(df, new_row_scores, left_stack, aux_stack, right_stack, threshold, best_moves, depth + 1, best_score, new_moves)

                # Update best_score and best_moves if needed
                if result_score < best_score:
                    best_score = result_score
                    best_moves = result_moves

        # Return the best scores encountered and the moves
        return best_score, best_moves