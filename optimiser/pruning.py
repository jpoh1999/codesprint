def prune(moves):
    """
    Prune the input set of moves to remove redundancies

    Args:
        - moves:2D array representing the moves for a slot shuffle
    
    Returns:
        - a 2D array representing the optimised set of moves
    """
    
    #Use hashmap to track history of moved containers. Key will always represent the latest move
    move_tracker = {}
    max_row = 10
    #Use list to track the no of containers that was moved into each row
    row_tracker = [0] * max_row
    
    slot_num = moves[0][0]
    ret = []
    
    for move in moves:
        fm_row, fm_level, to_row, to_level = move[1:]
        #Create a new move tracker for a newly container
        if (fm_row, fm_level) not in move_tracker:
            move_tracker[(to_row, to_level)] = [0, (fm_row, fm_level)]
        else:
            #Update key of container history to newest move
            move_tracker[(to_row, to_level)] = move_tracker.pop((fm_row, fm_level))
            #If there has not been containers moved on top of fm_row, fm_level, move is redundant
            if not(move_tracker[(to_row, to_level)][0] == row_tracker[fm_row - 1]):
                move_tracker[(to_row, to_level)].append((fm_row, fm_level))
            row_tracker[fm_row - 1] -= 1
        row_tracker[to_row - 1] += 1
        #Record the current moved no. of containers in the row to the first index
        move_tracker[(to_row, to_level)][0] = row_tracker[to_row - 1]

    #Interate backwards through the solution set to maintain order of movement
    for move in reversed(moves):
        to_row, to_level = move[-2:]
        if (to_row, to_level) in move_tracker:
            #If only no more locations left, continue (first element stores the row moved container count)
            if len(move_tracker[(to_row, to_level)]) == 1:
                continue
            fm_row, fm_level = move_tracker[(to_row, to_level)][-1]
            ret.insert(0, [slot_num, fm_row, fm_level, to_row, to_level])
            move_tracker[(fm_row, fm_level)] = move_tracker.pop((to_row, to_level))[:-1]

    return ret