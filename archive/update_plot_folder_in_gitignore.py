import os
from datetime import datetime

# Define the root directory for your plots
root_dir = 'plots'

# Find the latest date folder
latest_date = None
latest_date_path = None

for plot_type in os.listdir(root_dir):
    plot_type_path = os.path.join(root_dir, plot_type)
    if os.path.isdir(plot_type_path):
        for date_folder in os.listdir(plot_type_path):
            date_folder_path = os.path.join(plot_type_path, date_folder)
            if os.path.isdir(date_folder_path):
                try:
                    date = datetime.strptime(date_folder, '%Y%m%d')
                    if latest_date is None or date > latest_date:
                        latest_date = date
                        latest_date_path = date_folder_path
                except ValueError:
                    continue

# Update .gitignore
with open('.gitignore', 'w') as gitignore:
    gitignore.write('/plots/*\n')
    gitignore.write('!/plots/**/\n')
    if latest_date_path:
        # Convert to relative path
        relative_path = os.path.relpath(latest_date_path, os.getcwd())
        gitignore.write(f'!{relative_path}/\n')

