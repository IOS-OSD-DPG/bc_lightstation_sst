import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np
from patsy import dmatrices
from statsmodels.api import OLS
import dataframe_image as dfi


def lstq_model(anom_1d, date_numeric_1d):
    # https://www.statsmodels.org/stable/gettingstarted.html

    # Do not include nan values in the dataframe for the model
    dfmod = pd.DataFrame(
        {'Date': date_numeric_1d[~pd.isna(anom_1d)],
         'Anomaly': anom_1d[~pd.isna(anom_1d)]}
    )

    # create design matrices
    y, X = dmatrices('Anomaly ~ Date', data=dfmod, return_type='dataframe')

    mod = OLS(y, X)  # Describe model

    res = mod.fit()  # Fit model, return regression results

    # print(res.summary())  # Summarize model
    return res


def date_to_flat_numeric(all_years):
    # Convert the names of the months to numeric

    n_years = len(all_years)
    n_months = 12

    month_numeric = np.linspace(0, 1, 12 + 1)[:-1]

    date_numeric = np.zeros(shape=(n_years, len(month_numeric)))
    for i in range(n_years):
        for j in range(n_months):
            date_numeric[i, j] = all_years[i] + month_numeric[j]

    date_flat = date_numeric.flatten()
    return date_flat


def plot_lighthouse_t(anom_file: str, station_name: str, subplot_letter: str, plot_name: str,
                      best_fit=None):
    """
    Plot lightstation sea surface temperature anomalies in grey with a trend line
    :param anom_file: path of file containing SST anomaly data
    :param station_name: name of the lightstation
    :param subplot_letter: a letter to add to the corner of the plot if it will be a subplot
    :param plot_name: full path name to save the plot to
    :param best_fit: None or "lstq" or the full path to a csv file containing the best fit results
    :return:
    """
    anom_df = pd.read_csv(anom_file, index_col=[0])

    # station_name = anom_file.split('_')[0] + ' ' + anom_file.split('_')[1]

    # Flatten the data into 1 dim for plot
    date_flat = date_to_flat_numeric(anom_df.index)
    anom_flat = anom_df.to_numpy().flatten()

    fig, ax = plt.subplots(figsize=[6, 3])  # width, height [6,3] [3.12, 1.56]

    ax.plot(date_flat, anom_flat, c='grey')  # Plot the anomalies

    # Add a best-fit line
    if best_fit is None:
        pass
    elif best_fit == "lstq":
        # First remove nans
        date_nonan = date_flat[~np.isnan(anom_flat)]
        anom_nonan = anom_flat[~np.isnan(anom_flat)]
        # Compute polynomial using least squares
        poly = np.polynomial.Polynomial.fit(
            date_nonan, anom_nonan, deg=1)
        x_linspace, y_hat_linspace = poly.linspace(n=100)
        ax.plot(x_linspace, y_hat_linspace, c='k')
        # # Plot confidence limits around the linear fit
        # # ci is some confidence interval
        # ci = None
        # ax.fill_between(x_linspace, (y_hat_linspace - ci),
        #                 (y_hat_linspace - ci), c='b', alpha=0.1)

        # STATSMODELS
        # regres = lstq_model(anom_flat, date_flat)
        # # Check if the linear trend is significant before plotting
        # if all(regres.pvalues < 0.05):
        #     ax.plot(date_nonan, regres.fittedvalues, c='r')
        #     ax.fill_between(
        #         date_nonan,
        #         regres.fittedvalues - regres.conf_int().loc['Date', 0],
        #         regres.fittedvalues + regres.conf_int().loc['Date', 1],
        #         c='r', alpha=0.1)
    elif type(best_fit) == str:
        # Assume it's the path to a file
        regr_results = pd.read_csv(best_fit)
        mask = [station_name.split(" ")[0] in nm for nm in regr_results.iloc[:, 0]]
        slope, y_int = float(regr_results.iloc[mask, 1]), float(regr_results.iloc[mask, 3])
        # Plot the line
        x_linspace = np.linspace(date_flat[0], date_flat[-1], 100)
        y_hat_linspace = slope/100 * x_linspace + y_int  # Slope in per 100 years?
        ax.plot(x_linspace, y_hat_linspace, c='k')

    # Plot formatting
    ax.set_xlim((min(date_flat), max(date_flat)))
    ybot, ytop = plt.ylim()
    ax.set_ylim((-max(abs(np.array([ybot, ytop]))),
                 max(abs(np.array([ybot, ytop])))))
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature anomaly ($^\circ$C)')
    ax.minorticks_on()
    plt.tick_params(which='major', direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(which='minor', direction='in',
                    bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', bottom=False, direction='in')
    # ax.set_title(
    #     f'Time series of temperature anomalies at {station_name}')
    ax.set_title(f'({subplot_letter}) {station_name}')
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close(fig)
    return


def get_record_st_en(df: pd.DataFrame):
    """
    Get the start and end month and year of the full data record.
    Assume that any 999.99, etc. values have been read in as NaNs
    """
    st_mth = df.columns[~df.isna().iloc[0, :]][0]
    st_yr = df.index[0]
    en_mth = df.columns[~df.isna().iloc[len(df) - 1, :]][-1]
    en_yr = df.index[-1]
    return st_mth, st_yr, en_mth, en_yr


def trend_table_image():
    """
    Save the table of analysis periods, trends, and confidence intervals as an image.
    Include the Cummins & Masson (2014) data for comparison.
    Code requires and assumes that stations are only listed in alphabetical order everywhere
    :return:
    """
    old_dir = os.getcwd()
    new_dir = os.path.dirname(old_dir)
    os.chdir(new_dir)

    analysis_dir = os.path.join(new_dir, 'analysis')
    figures_dir = os.path.join(new_dir, 'figures', 'current')
    data_dir = os.path.join(new_dir, 'data')

    # Get file containing new trends
    new_trend_file = './analysis/monte_carlo_max_siml50000_st_cummins.csv'
    df_new_trend = pd.read_csv(new_trend_file)

    cm2014_file = './analysis/cummins_masson_2014_trends.csv'
    cm2014_df = pd.read_csv(cm2014_file)

    # Make a dataframe to print to an image containing the analysis period
    # and the trends and confidence intervals for each station
    df_img = cm2014_df

    # Initialize columns for containing the updated results
    df_img['New analysis time period'] = np.repeat('', len(df_img))
    df_img['New least-squares trend in C/century'] = np.repeat('', len(df_img))

    # Get the raw data in csv format
    raw_files = glob.glob(data_dir + '/*MonthlyTemp.csv')
    if len(raw_files) == 0:
        print('No monthly temperature data files found; try a different search key')
        return
    raw_files.sort()

    # Iterate through all the files
    for i in range(len(raw_files)):
        df_in = pd.read_csv(raw_files[i], index_col='Year', na_values=[99.99, 999.9, 999.99])

        # Get the start and end month and year of analysis period
        # with the end including the very last record, different than nans_to_strip() function
        st_mth, st_yr, en_mth, en_yr = get_record_st_en(df_in)

        # Both for deg C/century
        trend_i = np.round(df_new_trend.loc[i, 'Theil-Sen slope [deg C/century]'], decimals=2)
        conf_int_i = np.round(
            df_new_trend.loc[i, 'Monte Carlo confidence limit [deg C/century]'],
            decimals=2
        )

        # Update the dataframe to be exported as an image
        df_img.loc[i, 'New analysis time period'] = f'{st_mth} {st_yr} - {en_mth} {en_yr}'
        df_img.loc[i, 'New least-squares trend in C/century'] = '{:.2f} +/- {:.2f}'.format(
            trend_i, conf_int_i)

    # Export the table in PNG format
    df_img.set_index('Station', inplace=True)
    df_img_name = os.path.join(
        figures_dir,
        os.path.basename(new_trend_file).replace('.csv', '_cm2014.png')
    )
    dfi.export(df_img, os.path.join(figures_dir, df_img_name))

    # Export the table in csv format too
    df_img.to_csv(
        os.path.join(analysis_dir, os.path.basename(df_img_name).replace('png', 'csv')),
        index=False
    )

    # Change the current directory back
    os.chdir(old_dir)
    return


def plot_climatology(clim_file, output_dir):
    clim_df = pd.read_csv(clim_file, index_col=[0])
    month_numbers = np.arange(1, 12 + 1)

    fig, ax = plt.subplots(figsize=[6, 4.5])  # width, height
    for station in clim_df.index:
        station_name = station.split('_')[0] + ' ' + station.split('_')[1]
        ax.plot(month_numbers, clim_df.loc[station, :],
                label=station_name, marker='.')

    plt.legend()
    ax.set_xlim((min(month_numbers), max(month_numbers)))
    ax.set_xlabel('Month')
    ax.set_ylabel('Temperature ($^\circ$C)')
    ax.set_title('Climatological monthly mean temperatures for 1991-2020')
    plt.tight_layout()
    png_name = 'bc_lightstation_monthly_mean_climatologies_1991-2020.png'
    plt.savefig(os.path.join(output_dir, png_name))
    plt.close()
    return


def plot_filled_sst_resid(anom_file: str, all_time_mean: float, plot_name: str):
    # Replicate a plot that Peter C shared in his 2022 SOPO presentation
    # for all 8 lighthouse stations

    # Read in the anomaly data
    anom_df = pd.read_csv(anom_file, index_col=[0])

    station_name = os.path.basename(anom_file).split('_')[0] + ' ' + os.path.basename(anom_file).split('_')[1]
    print(station_name)

    # Flatten the data into 1 dim for plot
    date_flat = date_to_flat_numeric(anom_df.index)
    anom_flat = anom_df.to_numpy().flatten()

    # Make the least squares linear model
    res = lstq_model(anom_flat, date_flat)

    # Make the plot
    fig, ax = plt.subplots(figsize=[6, 3])
    # Move trend line down
    ax.plot(date_flat[~pd.isna(anom_flat)], res.fittedvalues - 3, c='k', linewidth=3)
    ax.fill_between(date_flat[~pd.isna(anom_flat)],
                    res.resid,
                    np.zeros(len(anom_flat[~pd.isna(anom_flat)])),
                    where=res.resid > 0,
                    color='r')
    ax.fill_between(date_flat[~pd.isna(anom_flat)],
                    res.resid,
                    np.zeros(len(anom_flat[~pd.isna(anom_flat)])),
                    where=res.resid < 0,
                    color='b')
    # Adjust plot bounds
    ybot, ytop = plt.ylim()
    ax.set_ylim((-max(abs(np.array([ybot, ytop]))) - 1,
                 max(abs(np.array([ybot, ytop]))) + 1))
    ax.set_xlim((np.nanmin(date_flat), np.nanmax(date_flat)))
    # Adjust tick direction
    plt.tick_params(which='major', direction='in',
                    bottom=True, top=True, left=True, right=True)
    # Add grid
    ax.grid(color='grey', alpha=0.5)
    # Label axes
    ax.set_ylabel('Temperature Anomaly')
    # Add text to plot
    plt.text(
        0.03, 0.89,
        station_name + ' mean temperature ' + str(np.round(all_time_mean, 2)) +
        '$^{\circ}$C, trend ' + str(np.round(res.params.Date * 100, 2)) + '$^{\circ}$C/100y',
        transform=ax.transAxes, backgroundcolor='white'
    )
    # Save fig
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close(fig)
    return


def plot_filled_daily_sst(daily_file, plot_name):
    """
    Translated from Peter Chandler's matlab script. Use daily SST data to plot
    365-day rolling averaged SST anomalies and compute and plot a OLS line below the curve.
        - Replace any missing observations with the mean value of the series.
        - Pad the start and end of the series with the mean value in prep for taking rolling mean
        - Take 365-day rolling mean
        - Remove start and end pads
        - Compute least squares trend and 95% confidence intervals
        - When plotting the averaged data, subtract the mean value of the series to get anomalies
    :param daily_file: absolute path to file containing daily observations in csv format
    :param plot_name: absolute path and file name to save the output plot to
    :return:
    """

    station_name = os.path.basename(daily_file).split('_')[0] + ' ' + os.path.basename(daily_file).split('_')[1]
    print(station_name)

    # Only read in date and temperature columns
    df = pd.read_csv(daily_file, skiprows=1, na_values=[999.9, 999.99], usecols=[0, 2])
    # Remove any rows that are all nans (Langara problem)
    df.dropna(axis='index', how='all', inplace=True)
    # Compute mean temperature for all time
    mean_temp = df.loc[:, 'TEMPERATURE ( C )'].mean(skipna=True)

    # Check for missing data
    percent_missing_data = sum(df.loc[:, 'TEMPERATURE ( C )'].isna()) / len(df) * 100
    print('{:.2f}'.format(percent_missing_data), 'percent missing data')

    # Replace nan in temperature time series with "normal" value
    df.loc[df.loc[:, 'TEMPERATURE ( C )'].isna(), 'TEMPERATURE ( C )'] = mean_temp

    # Pad the time series before and after with normal values to smooth edges
    df['DATE (YYYY-MM-DD)'] = pd.to_datetime(df['DATE (YYYY-MM-DD)'])
    df['DATE (float year)'] = df.loc[:, 'DATE (YYYY-MM-DD)'].dt.year + \
                              df.loc[:, 'DATE (YYYY-MM-DD)'].dt.dayofyear / 365
    # x = np.concatenate((
    #     -pd.timedelta_range(start='1 day', periods=365).sort_values(ascending=False)
    #     + df.loc[0, 'DATE (YYYY-MM-DD)'],
    #     df.loc[:, 'DATE (YYYY-MM-DD)'].to_numpy(),
    #     pd.timedelta_range(start='1 day', periods=365)
    #     + df.loc[len(df) - 1, 'DATE (YYYY-MM-DD)']
    # )).
    x = np.concatenate((
        -np.arange(365, 0, -1) / 365 + df.loc[0, 'DATE (float year)'],
        df.loc[:, 'DATE (float year)'].to_numpy(),
        np.arange(1, 366, 1) / 365 + df.loc[len(df) - 1, 'DATE (float year)']))
    y = np.concatenate((
        np.repeat(mean_temp, 365),
        df.loc[:, 'TEMPERATURE ( C )'].to_numpy(),
        np.repeat(mean_temp, 365)
    ))

    # Lowpass filter on daily temp data over 365 days
    yy = pd.Series(
        data=y
    ).rolling(window=365, win_type='boxcar').mean().to_numpy()

    # Remove beginning and end pads
    x = x[365:len(x) - 366]
    yy = yy[365:len(yy) - 366]

    # Calculate the long-term trend

    # poly = np.polynomial.Polynomial.fit(x, yy, deg=1)
    # # Reduce data points to yearly representers
    # x2, y2 = poly.linspace(n=len(x[::365]))
    # # Calculate trend in deg C per 100 y
    # # trend = (y2[-1] - y2[0])/(x2[-1] - x2[0]) * 100  # by hand
    # trend = poly.convert().coef[1] * 100

    # Use statsmodels
    res = lstq_model(yy, x)
    # Reduce data points to yearly representers
    y2 = res.fittedvalues[::365]
    x2 = x[::365]
    # Calculate the trend in deg C per 100 years
    trend = res.params.iloc[1] * 100

    # Calculate the 95% confidence interval on the trend

    # # CI for predicted mean
    # # The array has the lower and the upper limit of the confidence
    # # interval in the columns.
    # ci95_predint = res.get_prediction().conf_int(alpha=0.05)
    # CI for the estimated parameters in deg C per 100 years
    ci95_estparam = res.conf_int(alpha=0.05)
    diff1, diff2 = ci95_estparam.iloc[1, :] * 100 - trend
    # Arbitrary threshold
    if diff1 + diff2 > 1e-8:
        print('upper and lower confidence intervals not equal', abs(diff1), diff2)
        return
    else:
        ci95_trend = abs(diff1)

    # Make the plot

    fig, ax = plt.subplots(figsize=[7, 3])
    ax.fill_between(x, yy - mean_temp, np.zeros(len(yy)),
                    where=yy - mean_temp > 0, color='r')
    ax.fill_between(x, yy - mean_temp, np.zeros(len(yy)),
                    where=yy - mean_temp < 0, color='b')
    ax.plot(x2, y2 - 1 - mean_temp, c='k', linewidth=3)

    # # Check poly.coef value
    # ax.plot(x2, poly.coef[0] + poly.coef[1]*x2 - 1 - mean_temp, c='green')

    ax.set_xlim((np.nanmin(x), np.nanmax(x)))
    # Adjust tick direction
    plt.tick_params(which='major', direction='in',
                    bottom=True, top=True, left=True, right=True)
    # Add grid
    ax.grid(color='grey', alpha=0.5)
    # Label axes
    ax.set_ylabel('Temperature Anomaly')
    rounded_mean = np.round(mean_temp, 2) if mean_temp < 10 else np.round(mean_temp, 1)
    plt.text(
        0.03, 0.89,
        station_name + ' mean temperature ' + str(rounded_mean) +
        '$^{\circ}$C, trend ' + r'{:.2f} $\pm$ {:.2f}'.format(trend, ci95_trend) +
        '$^{\circ}$C/100y',
        transform=ax.transAxes, backgroundcolor='white')

    # Save fig
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close(fig)
    return


def sst_all_time_mean(monthly_mean_file):
    df = pd.read_csv(monthly_mean_file, skiprows=1, index_col=[0],
                     na_values=[999.9, 999.99])
    # Take mean of mean
    all_time_mean = df.mean(axis='index').mean()

    return np.round(all_time_mean, 2) if all_time_mean < 10 else np.round(all_time_mean, 1)


"""
parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'our_warming_ocean\\lighthouse_data\\'

# Plot climatological monthly means at each station

# Plot lighthouse temperature anomalies with linear trendline
data_dir = parent_dir + 'monthly_anom_from_monthly_mean\\'
anom_files = glob.glob(data_dir + '*.csv', recursive=False)

station_names = [
    os.path.basename(folder).replace('_', ' ')
    for folder in glob.glob(parent_dir + 'DATA_-_Active_Sites\\*')
]
subplot_letters = 'abcdefgh'

anom_files.sort()
station_names.sort()

for k in range(len(anom_files)):
    png_filename = data_dir + station_names[k].replace(' ', '_') + '_monthly_mean_sst_anomalies_ols.png'

    plot_lighthouse_t(anom_files[k], station_names[k],
                      subplot_letter=subplot_letters[k],
                      plot_name=png_filename, best_fit=True)

# Differences
diff_dir = parent_dir + 'monthly_anom_differences\\'
diff_files = glob.glob(diff_dir + '*.csv')
diff_files.sort()

for k in range(len(diff_files)):
    png_filename = diff_dir + station_names[k].replace(' ', '_') + '_sst_anomaly_diffs.png'
    plot_lighthouse_t(diff_files[k], station_names[k],
                      subplot_letter=subplot_letters[k],
                      plot_name=png_filename, best_fit=False)


lighthouse_clim = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
                  'our_warming_ocean\\lighthouse_data\\' \
                  'climatological_monthly_means\\' \
                  'lighthouse_sst_climatology_1991-2020.csv'

plot_dir = os.path.dirname(lighthouse_clim)

plot_climatology(lighthouse_clim, plot_dir)
"""

station_names = [
    "Amphitrite Point", "Bonilla Island", "Chrome Island", "Entrance Island", "Kains Island",
    "Langara Island", "Pine Island", "Race Rocks"
]
subplot_letters = 'abcdefgh'

# Plot climatological monthly means at each station
parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'our_warming_ocean\\lighthouse_data\\update_20230706\\'

output_folder = os.path.join(parent_dir, 'monthly_anom_from_monthly_mean', 'monte_carlo')

anom_files = glob.glob(parent_dir + 'monthly_anom_from_monthly_mean\\*.csv',
                       recursive=False)
anom_files.sort()

monte_carlo_results = os.path.join(
    parent_dir, "monthly_anom_from_monthly_mean",
    "monte_carlo", "monte_carlo_max_siml50000_st_cummins.csv"
)

for stn, f, letter in zip(station_names, anom_files, subplot_letters):
    png_filename = os.path.join(
        output_folder, stn.replace(' ', '_') + '_monthly_mean_sst_anomalies_theilsen.png'
    )
    plot_lighthouse_t(f, stn, letter, png_filename, best_fit=monte_carlo_results)

"""
# Plot lighthouse temperature anomalies with linear trendline
output_dir = parent_dir + 'monthly_anom_from_monthly_mean\\least_squares\\'
anom_files = glob.glob(parent_dir + 'monthly_anom_from_monthly_mean\\*.csv',
                       recursive=False)
anom_files.sort()

monthly_mean_files_dupl = glob.glob(
    parent_dir + 'DATA_-_Active_Sites\\*\\*Average_Monthly_Sea_Surface_Temperatures*.csv')
monthly_mean_files = [f for f in monthly_mean_files_dupl if 'french' not in f]
monthly_mean_files.sort()

for i in range(len(anom_files)):
    plot_filled_sst_resid(
        anom_files[i], sst_all_time_mean(monthly_mean_files[i]),
        output_dir + f'{station_names[i]}_sst_resid_with_trend_and_mean.png')
"""
# ----------------------------------daily data-------------------------------
"""
daily_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
            'lighthouse_data\\DATA_-_Active_Sites\\'

daily_files_dupe = glob.glob(
    daily_dir + '*\\*Daily_Sea_surface_Temperature*.csv')

daily_files = [f for f in daily_files_dupe if 'french' not in f]

output_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'lighthouse_data\\daily_trend_plots\\'

for f in daily_files:
    stn = os.path.basename(f).split('_')[0] + '_' + os.path.basename(f).split('_')[1]
    plot_filename = output_dir + stn + '_trend_ci_from_daily_sst.png'
    plot_filled_daily_sst(f, plot_filename)
"""
