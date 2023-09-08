import pandas as pd
import glob
import os

# Use the monthly mean data already provided to compute the 30-year means

old_dir = os.getcwd()
new_dir = os.path.dirname(old_dir)
os.chdir(new_dir)

file_list = glob.glob(
    new_dir +
    '\\data\\monthly\\*Average_Monthly_Sea_Surface_Temperatures*.csv'
)

# # Remove all french repeat files
# file_list = [f for f in file_list_dupe if 'french' not in f]
# file_list.sort()

# climatology for 1991-2020
clim_df = pd.DataFrame(
    index=[os.path.basename(f) for f in file_list],
    columns=pd.read_csv(file_list[0], skiprows=1, index_col='YEAR').columns)

for f in file_list:
    dfin = pd.read_csv(f, skiprows=1, index_col='YEAR',
                       na_values=[99.99, 999.9, 999.99])
    clim_df.loc[os.path.basename(f), :] = dfin.loc[1991:2020].mean(axis=0)

clim_file_name = os.path.join(
    new_dir,
    'analysis',
    'lighthouse_sst_climatology_1991-2020.csv'
)
clim_df.to_csv(clim_file_name)
