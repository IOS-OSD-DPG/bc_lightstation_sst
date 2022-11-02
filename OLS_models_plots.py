import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
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

    mod = sm.OLS(y, X)    # Describe model
    res = mod.fit()       # Fit model

    # print(res.summary())   # Summarize model

    # Plotting

    # Code amended from statsmodels graphics function
    # https://github.com/statsmodels/statsmodels/blob/27335040eb196eb06f68cf2e4425009e32d430e3/statsmodels/graphics/regressionplots.py#L77

    stn_name = os.path.basename(infile).split('_')[0] + ' ' + os.path.basename(infile).split('_')[1]

    fig, ax = plt.subplots()
    # sm.graphics.plot_fit(res, 'Date', ax=ax, set_linewidth=3)
    y = res.model.endog
    x1 = res.model.exog[:, 1]  # Date
    x1_argsort = np.argsort(x1)
    y = y[x1_argsort]
    x1 = x1[x1_argsort]
    # Get 95% confidence intervals
    # calculate standard deviation and confidence interval for prediction
    # iv_l, iv_u: lower and upper confidence bounds; (default: alpha = 0.05)
    _, iv_l, iv_u = wls_prediction_std(res)

    ax.scatter(x1, y, c='b', marker='o', s=3, label=res.model.endog_names)

    # Check significance
    if all(res.pvalues < 0.05):
        ax.plot(x1, res.fittedvalues[x1_argsort], color='r',
                label='OLS fit')
        # ax.vlines()
        ax.fill_between(x1,  iv_l[x1_argsort], iv_u[x1_argsort], color='grey', alpha=0.2)

    ax.legend(loc='upper left', numpoints=1)
    ybot, ytop = plt.ylim()
    ax.set_ylim((-max(abs(np.array([ybot, ytop]))),
                 max(abs(np.array([ybot, ytop])))))
    ax.minorticks_on()
    plt.tick_params(which='major', direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(which='minor', direction='in',
                    bottom=True, top=True, left=True, right=True)
    ax.set_xlabel('Year')
    ax.set_ylabel('SST anomaly ($^\circ$C)')
    ax.set_title(f'{stn_name} SST anomaly OLS model fit and CIs')
    fig.tight_layout(pad=1.0)
    plt.savefig(os.path.join(parent_dir, 'least_squares',
                             stn_name.replace(' ', '_') + '_ols_fit_CI.png'))
    plt.close(fig)

    # # Plot model residuals vs fitted values to check for unwanted patterns
    # fig, ax = plt.subplots()
    # ax.scatter(res.fittedvalues, res.resid)
    # left, right = ax.get_xlim()
    # ax.axhline(y=0, c='k')
    # ybot, ytop = plt.ylim()
    # ax.set_ylim((-max(abs(np.array([ybot, ytop]))),
    #              max(abs(np.array([ybot, ytop])))))
    # ax.minorticks_on()
    # plt.tick_params(which='major', direction='in',
    #                 bottom=True, top=True, left=True, right=True)
    # plt.tick_params(which='minor', direction='in',
    #                 bottom=True, top=True, left=True, right=True)
    # ax.set_xlabel('Fitted value')
    # ax.set_ylabel('Residual')
    # ax.set_title(f'{stn_name} SST anomaly OLS model')
    # fig.tight_layout()
    # plt.savefig(os.path.join(parent_dir, 'least_squares',
    #                          stn_name.replace(' ', '_') + '_ols_resid_vs_fitted.png'))
    # plt.close(fig)

    # # Save the model summary to a txt file
    # model_summary = os.path.join(
    #     parent_dir, 'least_squares',
    #     os.path.basename(infile).replace('.csv', '_OLS.txt'))
    # with open(model_summary, 'w') as txtfile:
    #     txtfile.write(res.summary().as_text())
