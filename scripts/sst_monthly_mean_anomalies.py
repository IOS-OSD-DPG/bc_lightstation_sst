import pandas as pd
import os
import glob
# import numpy as np

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'lighthouse_data\\'

input_dir = parent_dir + 'DATA_-_Active_Sites\\'

output_dir = parent_dir + 'monthly_anom_from_monthly_mean\\'

input_files_dupe = glob.glob(
    input_dir + '*\\*Average_Monthly_Sea_Surface_Temperatures*.csv')

input_files = [f for f in input_files_dupe if 'french' not in f]
input_files.sort()

clim_dir = parent_dir + 'climatological_monthly_means\\'
clim_file = clim_dir + 'lighthouse_sst_climatology_1991-2020.csv'
clim_df = pd.read_csv(clim_file, index_col=[0])
clim_df['Station_name'] = [x.split('_')[0] + '_' + x.split('_')[1]
                           for x in clim_df.index]
clim_df.set_index('Station_name', inplace=True)

for i in range(len(input_files)):
    station_name = os.path.basename(input_files[i]).split('_')[0] + '_' + \
                   os.path.basename(input_files[i]).split('_')[1]

    dfin = pd.read_csv(input_files[i], skiprows=1, index_col=[0],
                       na_values=[999.9, 999.99])
    dfout = pd.DataFrame(index=dfin.index, columns=dfin.columns)

    for month in dfin.columns:
        dfout.loc[:, month
                  ] = dfin.loc[:, month] - clim_df.loc[station_name, month]

    dfout.to_csv(output_dir + station_name + '_monthly_anom_from_monthly_mean.csv')

"""
input_dir = parent_dir + 'daily_anomalies\\'

output_dir = parent_dir + 'monthly_mean_anomalies\\'

input_files = glob.glob(input_dir + '*.csv')
input_files.sort()

for f in input_files:
    dfin = pd.read_csv(f)
    years_available = np.unique(dfin.loc[:, 'Year'])
    all_months = np.arange(1, 12+1)
    dfout = pd.DataFrame(index=years_available, columns=all_months)
    for y in years_available:
        for m in all_months:
            mth_subsetter = np.logical_and(dfin.loc[:, 'Year'] == y,
                                           dfin.loc[:, 'Month_number'] == m)
            dfout.loc[y, m] = np.nanmean(dfin.loc[mth_subsetter, 'TEMPERATURE ( C )'])

    dfout.to_csv(output_dir + os.path.basename(f))
"""
