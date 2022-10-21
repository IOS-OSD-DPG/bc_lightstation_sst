import pandas as pd
import glob
import os

SEASONS = {'winter': ['JAN', 'FEB', 'MAR'],
           'spring': ['APR', 'MAY', 'JUN'],
           'summer': ['JUL', 'AUG', 'SEP'],
           'fall': ['OCT', 'NOV', 'DEC']}

# Compute monthly anomalies using the 1991-2020 lighthouse climatology

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'lighthouse_data\\'

file_list_with_dupe = glob.glob(
    parent_dir +
    'DATA_-_Active_Sites\\*\\*Average_Monthly_Sea_Surface_Temperatures*.csv')

# Remove all french repeat files
file_list = [f for f in file_list_with_dupe if 'french' not in f]
file_list.sort()

# coord_file = parent_dir + 'lighthouse_coordinates.csv'
# coord_df = pd.read_csv(coord_file)

clim_dir = parent_dir + 'climatological_monthly_means\\'
clim_file = clim_dir + 'lighthouse_sst_climatology_1991-2020.csv'
clim_df = pd.read_csv(clim_file, index_col=[0])

output_dir = parent_dir + 'monthly_anomalies\\'

for i in range(len(file_list)):
    # Monthly data has 999.99 fill value, daily data has 999.9 fill value
    dfin = pd.read_csv(file_list[i], skiprows=1, index_col='YEAR',
                       na_values=[999.9, 999.99])

    dfout = pd.DataFrame(index=dfin.index, columns=dfin.columns)
    for mth in dfin.columns:
        # Subtract the climatological mean
        clim_mth_mean = clim_df.loc[os.path.basename(file_list[i]), mth]
        dfout.loc[:, mth] = dfin.loc[:, mth] - clim_mth_mean

    output_name = output_dir + os.path.basename(file_list[i])
    dfout.to_csv(output_name)
