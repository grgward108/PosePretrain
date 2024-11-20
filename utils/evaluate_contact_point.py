import re
import pandas as pd
import os

# Dictionary to hold extracted data for each object
data = {}

# Updated regular expression to extract 'contact' value from the 'verts_info' section
regex = r"verts_info:.*contact:\s*([0-9]+)"

# List of objects to process (sorted alphabetically)
objects = sorted(["toothpaste", "camera", "wineglass", "fryingpan", "binoculars", "mug"])

# Ask the user for the experiment name
exp_name = input("Please enter the experiment name: ")

# Loop over each object and process its log file
for obj in objects:
    log_file_path = f"../results/{exp_name}/GraspPose/{obj}/{obj}.log"
    
    # List to hold extracted data for the current object
    object_data = []
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Search for the line that contains the contact value in 'verts_info'
                match = re.search(regex, line)
                if match:
                    # Append the value to the object_data list
                    object_data.append(int(match.group(1)))  # Using int as 'contact' seems to be an integer
        
        # Store the data for the current object
        data[obj] = object_data

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing '{obj}': {e}")

# Convert the collected data to a DataFrame, filling missing values with NaN if some objects have more values than others
df = pd.DataFrame.from_dict(data, orient='index').transpose()

# Sort columns alphabetically
df = df.reindex(sorted(df.columns), axis=1)

# Display the DataFrame
print(df)

# Create the path if it doesn't exist
output_dir = f"../results/{exp_name}/GraspPose"
os.makedirs(output_dir, exist_ok=True)

# Save the DataFrame to CSV in the specified directory
csv_file_path = os.path.join(output_dir, f"{exp_name}_contact_values.csv")
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved at: {csv_file_path}")
