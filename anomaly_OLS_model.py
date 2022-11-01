import statsmodels.api as sm
import pandas as pd
import os
import numpy as np
from patsy import dmatrices
import glob

parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'our_warming_ocean\\lighthouse_data\\' \
             'monthly_anom_from_monthly_mean\\'

data_files = glob.glob(parent_dir + '*.csv')
data_files.sort()

for infile in data_files:
    # infile = parent_dir + 'Amphitrite_Point_monthly_anom_from_monthly_mean.csv'

    dfin = pd.read_csv(infile, index_col=[0])

    # ------------------Flatten the data values-----------------
    n_years = len(dfin)
    n_months = 12

    # Flatten the data into 1 dim for plot
    # Convert the names of the months to numeric
    month_numeric = np.linspace(0, 1, 12 + 1)[:-1]

    date_numeric = np.zeros(shape=(n_years, len(month_numeric)))
    for i in range(n_years):
        for j in range(n_months):
            date_numeric[i, j] = dfin.index[i] + month_numeric[j]

    date_flat = date_numeric.flatten()
    anom_flat = dfin.to_numpy().flatten()

    # ------------------OLS model fitting--------------------

    # https://www.statsmodels.org/stable/gettingstarted.html

    # Do not include nan values in the dataframe for the model
    dfmod = pd.DataFrame(
        {'Date': date_flat[~pd.isna(anom_flat)],
         'Anomaly': anom_flat[~pd.isna(anom_flat)]}
    )

    # create design matrices
    y, X = dmatrices('Anomaly ~ Date', data=dfmod, return_type='dataframe')

    print(y.shape, X.shape)

    mod = sm.OLS(y, X)    # Describe model

    res = mod.fit()       # Fit model

    print(res.summary())   # Summarize model

    model_summary = infile.replace('.csv', '_OLS.txt')
    with open(model_summary, 'w') as txtfile:
        txtfile.write(res.summary().as_text())
