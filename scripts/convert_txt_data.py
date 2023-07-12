import pandas as pd
import glob

# Convert ".txt" data files received from the research scientist to .csv files (easier to work with)
# Keep -999.99, -99.99 NA values for now...

wdir = "C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\" \
       "lighthouse_data\\update_20230706\\raw_data\\"

txt_files = glob.glob(wdir + "*.txt")
txt_files.sort()

for f in txt_files:
    # Read fixed-width lines
    txt_df = pd.read_fwf(f, index_col=0, skiprows=3, skipfooter=1, engine="python")
    csv_filename = f.replace(".txt", ".csv")
    txt_df.to_csv(csv_filename)
