import pandas as pd
import os
import glob
# import numpy as np

# Compute monthly mean sea surface temperature anomalies
# Compute the anomalies from the 1991-2020 climatology


# parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
#              'lighthouse_data\\'

"""
input_dir = parent_dir + 'DATA_-_Active_Sites\\'

output_dir = parent_dir + 'monthly_anom_from_monthly_mean\\'

input_files_dupe = glob.glob(
    input_dir + '*\\*Average_Monthly_Sea_Surface_Temperatures*.csv')

input_files = [f for f in input_files_dupe if 'french' not in f]
input_files.sort()
"""

# July 7, 2023: Newest files in a different place than before

# input_dir = parent_dir + 'update_20230706\\raw_data\\'
# output_dir = parent_dir + 'update_20230706\\monthly_anom_from_monthly_mean\\'

# ^Change to working in the repo directory from my Documents folder
old_dir = os.getcwd()
new_dir = os.path.dirname(old_dir)  # Migrate up a directory level
os.chdir(new_dir)

# !! May need to change this extension depending on the file names of subsequent updates
input_files_all = glob.glob(new_dir + './data/*MonthlyTemp.csv')
input_files_all.sort()
input_files = []

# Remove stations that aren't being used
for f in input_files_all:
    # print(os.path.basename(f), any([name in f for name in ['Departure', 'Egg', 'McInnes', 'Nootka']]))
    if not any([name in f for name in ['Departure', 'Egg', 'McInnes', 'Nootka']]):
        input_files.append(f)

# Import climatological data
clim_file = os.path.join(new_dir, 'analysis', 'lighthouse_sst_climatology_1991-2020.csv')
clim_df = pd.read_csv(clim_file, index_col=[0])
# clim_df['Station_name'] = [x.split('_')[0] + '_' + x.split('_')[1] for x in clim_df.index]
clim_df['Station_name'] = [
    x.split('_')[0] + x.split('_')[1] if x.split('_')[1] == "Rocks"
    else x.split('_')[0] for x in clim_df.index]
clim_df.set_index('Station_name', inplace=True)

for i in range(len(input_files)):
    # station_name = os.path.basename(input_files[i]).split('_')[0] + '_' + \
    #                os.path.basename(input_files[i]).split('_')[1]
    station_name = os.path.basename(input_files[i]).split('MonthlyTemp')[0]

    # dfin = pd.read_csv(input_files[i], skiprows=1, index_col=[0],
    #                    na_values=[999.9, 999.99])
    dfin = pd.read_csv(input_files[i], index_col=[0],
                       na_values=[99.99, 999.9, 999.99])
    dfout = pd.DataFrame(index=dfin.index, columns=dfin.columns)

    for month in dfin.columns:
        dfout.loc[:, month
                  ] = dfin.loc[:, month] - clim_df.loc[station_name, month.upper()]

    dfout.to_csv(os.path.join(new_dir, 'analysis',
                              station_name + '_monthly_anom_from_monthly_mean.csv'))

os.chdir(old_dir)

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
