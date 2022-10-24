import pandas as pd
import os
import glob
import numpy as np

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'lighthouse_data\\'

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
