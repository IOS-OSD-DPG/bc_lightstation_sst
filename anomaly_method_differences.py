import pandas as pd
import glob
import os

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'lighthouse_data\\'

dir1 = parent_dir + 'monthly_anom_from_daily_mean\\'
dir2 = parent_dir + 'monthly_anom_from_monthly_mean\\'

files1 = glob.glob(dir1 + '*.csv')
files2 = glob.glob(dir2 + '*.csv')

files1.sort()
files2.sort()

station_names = [
    os.path.basename(folder)
    for folder in glob.glob(parent_dir + 'DATA_-_Active_Sites\\*')
]

output_dir = parent_dir + 'monthly_anom_differences\\'

for i in range(len(files1)):
    print(station_names[i])
    if station_names[i] in files1[i] and station_names[i] in files2[i]:
        df1 = pd.read_csv(files1[i], index_col=[0])
        # df1.dropna(axis='index', how='all', inplace=True)
        arr1 = df1.to_numpy()
        df2 = pd.read_csv(files2[i], index_col=[0])
        # df2.dropna(axis='index', how='all', inplace=True)
        arr2 = df2.to_numpy()
        dfout = pd.DataFrame(
            data=arr2-arr1, index=df2.index, columns=df2.columns)
        dfout_filename = output_dir + f'{station_names[i]}_monthly_anom_diffs.csv'
        dfout.to_csv(dfout_filename, index=True)
    else:
        print('File mismatch !')
        print(os.path.basename(files1[i]))
        print(os.path.basename(files2[i]))

