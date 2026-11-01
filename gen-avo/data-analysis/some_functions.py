import pandas as pd
import numpy as np
import os

def fetch_and_save_data(
    firebase_ref: str,
    output_path: str,
    output_path2: str,
) -> pd.DataFrame:
    """
    Fetches data from Firestone collection, makes it pretty and saves it to CSV
   
    Args:
        firebase_ref (str): Firestone database subjects collection reference
        output_path (str): Path to save final CSV file to
       
    Returns: 
        Tuple[pd.DataFrame]:
            - pd.DataFrame: The concatenated dataframe of all processed subject data
    """

    all_subject_dfs = []
    
    #Loop over each document in the 'subjects' collection 
    docs = list(firebase_ref.stream())  # Materialize the stream as a list to reuse it.
    for sub in docs:
        sub_data = sub.to_dict()

    #for sub in firebase_ref.stream():
        #sub_data = sub.to_dict()  # Convert document's data to a dictionary
    
        sub_df = pd.DataFrame.from_dict(sub_data["trial_data"], orient='index')
        #print(sub_df)
        sub_df = sub_df.reset_index().rename(columns={'index': 'idx'})
        sub_df['ppt'] = sub.id
        print(sub.id)

        sub_df = sub_df.sort_values(by="trial")
        sub_df = sub_df.sort_values(by=["run", "state", "trial"])
                          
        sub_df["date"] = sub_data["date"]
        sub_df["ID"] = sub_data["subjectID"]

        #pth = "/Users/verra/Documents/AVGEN/data/pilots/"
        unique_id = str(sub_df['ID'].iloc[0]) 
        output_file = os.path.join(output_path2, f"{unique_id}.csv")
        if 'feedback' not in sub_df.columns:
            sub_df['feedback'] = 0

        column_order = ['idx', 'ppt', 'date', 'ID', 'run', 'state', 'trial', 'csPosition', 'csValue', 'RT','rating', 'ratingCoordinate','path', 'moveTime', 'nMoves', 'starving','slider','time', 'dogCollection','scheduleID','userInput', 'feedback']
        sub_df = sub_df[column_order]
        sub_df.to_csv(output_file, index=False)

        all_subject_dfs.append(sub_df)
    
    all_subject_df = pd.concat(all_subject_dfs)

    #column_order = ['idx', 'ppt', 'date', 'ID', 'run', 'state', 'trial', 'csPosition', 'csValue', 'RT','rating', 'ratingCoordinate','path', 'moveTime', 'nMoves', 'starving','slider','time', 'dogCollection','scheduleID','userInput']
    all_subject_df = all_subject_df[column_order]

    #save to csv
    all_subject_df.to_csv(output_path, index=False)
    return all_subject_df



def plot_ratings(ax, data, grid_size, cmap, alpha, vmin, vmax, label):
    """Plots the ratings for a given state on a single subplot."""
    grid = np.zeros((grid_size, grid_size))
    if data.empty:
        return False

    # CS+ marker
    #first_row = data.iloc[0]
    #cs_position = first_row['csPosition']
    #cs_value = first_row['csValue']
    #if isinstance(cs_position, (list, tuple)) and len(cs_position) == 2:
    #    cs_x, cs_y = cs_position
    #    grid[cs_x, cs_y] = cs_value if not pd.isna(cs_value) else np.nan

    # Ratings
    for _, row in data.iterrows():
        coordinates = row['ratingCoordinate']
        try:
            if isinstance(coordinates, str):
                x, y = ast.literal_eval(coordinates)
            elif isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
                x, y = coordinates
            else:
                continue
            grid[x, y] = row['rating']
        except (ValueError, SyntaxError):
            continue

    ax.imshow(grid, interpolation='none', cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    return True