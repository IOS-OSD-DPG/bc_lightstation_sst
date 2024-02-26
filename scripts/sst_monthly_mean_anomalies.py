import pandas as pd
import os
import glob
import numpy as np

# Compute monthly mean sea surface temperature anomalies
# Compute the anomalies from the 1991-2020 climatology

MONTH_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

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


def compute_monthly_anomalies(obs_data_suffix: str):
    """
    Compute monthly mean SST anomalies from monthly mean observations
    :return: nothing, but save anomaly data to csv files in the analysis folder of the project
    """
    # ^Change to working in the repo directory from my Documents folder
    old_dir = os.getcwd()
    new_dir = os.path.dirname(old_dir)  # Migrate up a directory level
    os.chdir(new_dir)

    # !! May need to change this extension depending on the file names of subsequent updates
    input_files_all = glob.glob(new_dir + f'./data/monthly/*{obs_data_suffix}')
    input_files_all.sort()

    # Initialize flag to use daily data if monthly not available
    use_daily = False

    if len(input_files_all) == 0:
        print('No monthly mean files found with suffix', obs_data_suffix, 'in directory',
              new_dir, '; trying daily file folder')

        input_files_all = glob.glob(new_dir + f'./data/daily/*{obs_data_suffix}')
        input_files_all.sort()
        if len(input_files_all) > 0:
            use_daily = True
        else:
            print('No daily data found with suffix', obs_data_suffix, 'in directory',
                  new_dir, '; returning None')
            return None

    input_files = input_files_all

    # # Remove stations that aren't being used
    # for f in input_files_all:
    #     # print(os.path.basename(f), any([name in f for name in ['Departure', 'Egg', 'McInnes', 'Nootka']]))
    #     if not any([name in f for name in ['Departure', 'Egg', 'McInnes', 'Nootka']]):
    #         input_files.append(f)

    # Import climatological data
    clim_file = os.path.join(new_dir, 'analysis',
                             'lighthouse_sst_climatology_1991-2020.csv')
    clim_df = pd.read_csv(clim_file, index_col=[0])
    # clim_df['Station_name'] = [x.split('_')[0] + '_' + x.split('_')[1] for x in clim_df.index]
    # clim_df['Station_name'] = [
    #     x.split('_')[0] + x.split('_')[1] if x.split('_')[1] == "Rocks"
    #     else x.split('_')[0] for x in clim_df.index
    # ]
    clim_df['Station_name'] = [
        x.split('_')[0] + ' ' + x.split('_')[1] for x in clim_df.index
    ]
    clim_df.set_index('Station_name', inplace=True)

    for i in range(len(input_files)):
        if obs_data_suffix == '.csv':
            if not use_daily:
                station_name = os.path.basename(input_files[i]).split('_')[0] + ' ' + \
                               os.path.basename(input_files[i]).split('_')[1]

                df_monthly = pd.read_csv(input_files[i], skiprows=1, index_col=[0],
                                         na_values=[99.99, 999.9, 999.99])

        elif obs_data_suffix == '.txt':
            if use_daily:
                station_name = os.path.basename(input_files[i]).split('DailySalTemp')[0]

                df_daily = pd.read_fwf(input_files[i], skiprows=3, na_values=[99.99, 999.9, 999.99])

                # Convert daily data to monthly means
                unique_years = np.unique(df_daily.loc[:, 'Year'])

                # Initialize DataFrame filled with NaNs
                df_monthly = pd.DataFrame(index=unique_years, columns=MONTH_ABBREV)

                # Populate the monthly DataFrame
                for year in unique_years:
                    for mth_idx in range(len(MONTH_ABBREV)):
                        subsetter = np.logical_and(
                            df_daily.loc[:, 'Year'] == year,
                            df_daily.loc[:, 'Month'] == mth_idx + 1
                        )
                        mth = MONTH_ABBREV[mth_idx]
                        df_monthly.loc[year, mth] = df_daily.loc[subsetter, 'Temperature(C)'].mean(
                            skipna=True
                        )

                # Export the results in a csv file matching the usual style
                df_monthly.index.name = 'Year'
                monthly_file_name = os.path.join(
                    new_dir, 'data', 'monthly',
                    os.path.basename(input_files[i]).replace('DailySalTemp.txt', 'MonthlyTemp.csv')
                )
                df_monthly.to_csv(monthly_file_name)
            else:
                station_name = os.path.basename(input_files[i]).split('MonthlyTemp')[0]

                df_monthly = pd.read_fwf(input_files[i], skiprows=3, index_col=[0],
                                         na_values=[99.99, 999.9, 999.99])

        try:
            dfout = pd.DataFrame(index=df_monthly.index, columns=df_monthly.columns)
        except NameError:
            print('Monthly dataframe does not exist')
            return

        # Compute the anomalies for each month
        # Station names in index of clim_df may be different than the input station names
        # csv files and climatology file have full station name e.g. "Amphitrite Point";
        # txt files have partial station name e.g. "Amphitrite"
        # station_locator = np.where([station_name.lower() in x.lower() for x in clim_df.index])[0][0]
        clim_station_name = clim_df.index[i]
        if not all([lett in clim_station_name for lett in station_name]):
            print('station names do not match:', station_name, clim_station_name)
        # print(station_locator, clim_station_name)

        for month in df_monthly.columns:
            # print(dfin, clim_df, sep='\n\n')
            dfout.loc[:, month] = df_monthly.loc[:, month] - clim_df.loc[clim_station_name, month.upper()]

        # Export monthly anomalies to file
        dfout.to_csv(
            os.path.join(
                new_dir,
                'analysis',
                station_name.replace(' ', '_') + '_monthly_anom_from_monthly_mean.csv'
            )
        )

    # Change working directory back
    os.chdir(old_dir)
    return


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
