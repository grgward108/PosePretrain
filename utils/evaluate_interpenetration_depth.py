import re
import os
import pandas as pd

def extract_and_save_interpenetration_depth(log_file_path, save_csv_path):
    # List to hold extracted data
    data = []

    # Regular expression to extract average_interpenetration_depth_cm and verts_info
    regex_depth = r"interpenetration_depth_cm:\s*([0-9.]+)"
    regex_verts = r"verts_info:.*contact:\s*([0-9]+)"

    # Open the log file and parse lines
    try:
        with open(log_file_path, 'r') as file:
            current_depth = None
            current_verts_info = None
            for line in file:
                # Search for the line that contains the average_interpenetration_depth_cm value
                match_depth = re.search(regex_depth, line)
                if match_depth:
                    current_depth = float(match_depth.group(1))
                
                # Search for the line that contains the verts_info contact value
                match_verts = re.search(regex_verts, line)
                if match_verts:
                    current_verts_info = int(match_verts.group(1))
                
                # If both values are found, append them to the data list
                if current_depth is not None and current_verts_info is not None:
                    data.append([current_depth, current_verts_info])
                    # Reset the variables after appending to avoid incorrect pairings
                    current_depth = None
                    current_verts_info = None

        # Convert the data to a DataFrame with two columns
        df = pd.DataFrame(data, columns=['interpenetration_depth_cm', 'verts_info_contact'])

        # Calculate the average of each column
        averages = pd.DataFrame([df.mean()], columns=df.columns)

        # Concatenate the original DataFrame with the averages DataFrame
        df = pd.concat([df, averages], ignore_index=True)

        # Save to a CSV file
        df.to_csv(save_csv_path, index=False)
        print(f"CSV file saved at {save_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
