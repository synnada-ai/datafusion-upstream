import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print('Usage: python analyze.py [new_dir] [old_dir]')
    sys.exit(1)

new_dir = sys.argv[1]
old_dir = sys.argv[2]

# Define the path to the target directory
target_dir = 'target/criterion'

# Define the regex pattern to match the directory names
pattern = re.compile(r'column_count_(\d+)_selectivity_(\d+)/indices/(\d+)')

# Lists to store the input variables and percent differences
input_variables = []
percent_differences = []

# Function to calculate mean and percent difference
def calculate_percent_difference(base_file, new_file):
    base_df = pd.read_csv(base_file)
    new_df = pd.read_csv(new_file)

    base_mean = (base_df['sample_measured_value'] / base_df['iteration_count']).mean()
    new_mean = (new_df['sample_measured_value'] / new_df['iteration_count']).mean()

    percent_difference = ((new_mean - base_mean) / base_mean) * 100
    return percent_difference

# Iterate over the directories in the target directory
for root, dirs, files in os.walk(target_dir):
    for dir_name in dirs:
        rel_path = os.path.relpath(os.path.join(root, dir_name), target_dir)
        match = pattern.match(rel_path)
        if match:
            column_count = int(match.group(1))
            selectivity_modulo = int(match.group(2))
            batch_size = int(match.group(3))

            base_file = os.path.join(root, dir_name, old_dir, 'raw.csv')
            new_file = os.path.join(root, dir_name, new_dir, 'raw.csv')

            if os.path.exists(base_file) and os.path.exists(new_file):
                percent_difference = calculate_percent_difference(base_file, new_file)
                input_variables.append(np.log2([batch_size, np.exp2(selectivity_modulo)]))
                percent_differences.append(percent_difference)

# Convert lists to numpy arrays
input_variables = np.array(input_variables)
percent_differences = np.array(percent_differences)

# Create a DataFrame and name the columns
df = pd.DataFrame(input_variables, columns=['batch_size', 'selectivity_modulo'])
df['percent_difference'] = percent_differences

# Create a 2D bubble chart
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['batch_size'], df['selectivity_modulo'], s=np.abs(df['percent_difference']) * 5, c=df['percent_difference'], cmap='coolwarm', alpha=0.6, vmin=-100, vmax=100)

# Label each data point
for i, row in df.iterrows():
    plt.text(row['batch_size'], row['selectivity_modulo'], f'{row["percent_difference"]:.1f}', fontsize=8, ha='center', va='center')

# Set labels
plt.xlabel('Batch Size')
plt.ylabel('Selectivity Modulo')
plt.title('Batch Size vs Selectivity Modulo')

# Add color bar
plt.colorbar(label='Percent Difference')

# Show plot
plt.show()
