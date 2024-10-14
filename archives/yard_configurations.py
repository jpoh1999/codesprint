import pandas as pd

def generate_yard_layout(n_items, prefix, x_min_start=1, dist=1, y_min=1, y_max=50) -> pd.DataFrame:
    """
    General function to generate layout data for either blocks or berths.

    Args :
        - n_items: Number of items to generate (blocks or berths).
        - prefix: Prefix for the item identifier (e.g., 'N' for block, 'T' for berth).
        - x_min_start: Starting value for x_min.
        - dist: Distance between x_min and x_max for each row.
        - y_min: Minimum y value (fixed across all rows).
        - y_max: Maximum y value (fixed across all rows).

    Returns:
        - df (pd.DataFrame) :  DataFrame with the layout configuration.
    """
    
    # Create list of rows
    data = []

    for i in range(n_items):
        row_id = prefix + str(i + 1).zfill(3)  # Generate the identifier with prefix and zero-padded sequence
        row = [row_id, x_min_start + i * dist, x_min_start + (i + 1) * dist, y_min, y_max]
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=["id", "x_min", "x_max", "y_min", "y_max"])
    
    return df


generate_yard_layout(50, "N")
generate_yard_layout(10, "T")


