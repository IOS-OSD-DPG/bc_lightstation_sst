import pandas as pd
import glob
import os

# Convert ".txt" data files received from the research scientist to .csv files (easier to work with)
# Keep -999.99, -99.99 NA values for now...

old_dir = os.getcwd()
new_dir = os.path.dirname(old_dir)
os.chdir(new_dir)

txt_files = glob.glob(new_dir + "/data/*.txt")
txt_files.sort()

for f in txt_files:
    # Only convert the data from the selected 8 stations
    if not any([name in f for name in ['Departure', 'Egg', 'McInnes', 'Nootka']]):
        # Read fixed-width lines
        txt_df = pd.read_fwf(f, index_col=0, skiprows=3, skipfooter=1, engine="python")
        csv_filename = f.replace(".txt", ".csv")
        txt_df.to_csv(csv_filename)

# Change current directory back to original
os.chdir(old_dir)