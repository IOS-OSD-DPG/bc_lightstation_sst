import pandas as pd
import glob
import os

MONTH_NAMES = ['JAN', 'FEB', 'MAR',
               'APR', 'MAY', 'JUN',
               'JUL', 'AUG', 'SEP',
               'OCT', 'NOV', 'DEC']

# Compute DAILY anomalies using the 1991-2020 lighthouse climatology

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'lighthouse_data\\'

clim_dir = parent_dir + 'climatological_monthly_means\\'
clim_file = clim_dir + 'lighthouse_sst_climatology_1991-2020.csv'
clim_df = pd.read_csv(clim_file, index_col=[0])
clim_df['Station_name'] = [x.split('_')[0] + '_' + x.split('_')[1]
                           for x in clim_df.index]
clim_df.set_index('Station_name', inplace=True)

file_list_with_dupe = glob.glob(
    parent_dir +
    'DATA_-_Active_Sites\\*\\*Daily*.csv')

# Remove all french repeat files
file_list = [f for f in file_list_with_dupe if 'french' not in f]
file_list.sort()

output_dir = parent_dir + 'daily_anomalies\\'

for i in range(len(file_list)):
    # Monthly data has 999.99 fill value, daily data has 999.9 fill value
    dfin = pd.read_csv(file_list[i], skiprows=1, index_col='DATE (YYYY-MM-DD)',
                       usecols=[0, 2], na_values=[999.9, 999.99])
    station_name = os.path.basename(file_list[i]).split('_')[0] + '_' + \
        os.path.basename(file_list[i]).split('_')[1]
    # Initialize output dataframe
    dfout = pd.DataFrame(index=dfin.index, columns=dfin.columns)
    # Create a column for the month number and year
    dfout['Year'] = pd.to_datetime(dfin.index).year
    dfout['Month_number'] = pd.to_datetime(dfin.index).month
    # Iterate through the months
    for j in range(len(MONTH_NAMES)):
        # Subtract the climatological mean
        mth_name = MONTH_NAMES[j]
        mth_num = j + 1
        clim_mth_mean = clim_df.loc[station_name, mth_name]
        mth_mask = dfout.loc[:, 'Month_number'] == mth_num
        dfout.loc[mth_mask, 'TEMPERATURE ( C )'
                  ] = dfin.loc[mth_mask, 'TEMPERATURE ( C )'] - clim_mth_mean

    output_name = output_dir + os.path.basename(file_list[i])
    dfout.to_csv(output_name)
