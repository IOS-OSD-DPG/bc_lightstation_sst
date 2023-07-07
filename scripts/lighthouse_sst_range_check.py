import numpy as np
import pandas as pd
import glob
from os.path import basename

COAST_N_PAC_RANGE_0m = (-2.10, 35.00)  # degrees C

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\lighthouse_data\\'

input_files_dupe = glob.glob(parent_dir + 'DATA_-_Active_Sites\\*\\*Daily*.csv')

input_files = [f for f in input_files_dupe if 'french' not in f]
input_files.sort()

output_dir = parent_dir + 'range_check\\'

# Name summary statistics file to output
summary_stats_file = output_dir + 'range_check_summary_stats.csv'
summary_stats_df = pd.DataFrame(
    index=[basename(f) for f in input_files],
    columns=['Num_input_obs', 'Num_nan_obs', 'Num_flagged_obs',
             'fraction_passing_obs'])

# Iterate through the files doing a range check on each observation
for f in input_files:
    dfin = pd.read_csv(f, skiprows=1, na_values=[999.9, 999.99])
    dfin['T_range_flag'] = np.logical_or(
        dfin.loc[:, 'TEMPERATURE ( C )'] < COAST_N_PAC_RANGE_0m[0],
        dfin.loc[:, 'TEMPERATURE ( C )'] > COAST_N_PAC_RANGE_0m[1])  # np.zeros(len(dfin))

    dfout = dfin.loc[~dfin.loc[:, 'T_range_flag'],
                     ['DATE (YYYY-MM-DD)', 'TEMPERATURE ( C )']]

    summary_stats_df.loc[basename(f), :] = [
        len(dfin), sum(dfin.loc[:, 'TEMPERATURE ( C )'].isna()),
        sum(dfin.loc[:, 'T_range_flag']),
        sum(~dfin.loc[:, 'T_range_flag']) / len(dfin)]

    output_name = output_dir + basename(f)
    dfout.to_csv(output_name, index=False)

summary_stats_df.to_csv(summary_stats_file)
