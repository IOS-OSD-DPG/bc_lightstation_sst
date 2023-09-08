import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import glob
import numpy as np
from patsy import dmatrices
from statsmodels.api import OLS
import dataframe_image as dfi
from scripts.trend_estimation import flatten_dframe
import seaborn as sns
from blume.table import table

STATION_NAMES = [
    "Amphitrite Point", "Bonilla Island", "Chrome Island",
    "Departure Bay", "Egg Island", "Entrance Island", "Kains Island",
    "Langara Island", "McInnes Island", "Nootka Point", "Pine Island",
    "Race Rocks"
]
MONTH_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
                'Oct', 'Nov', 'Dec']


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


def plot_monthly_anomalies(anom_file: str, station_name: str, plot_name: str, best_fit=None):
    """
    Plot lightstation sea surface temperature anomalies in grey with a trend line
    :param anom_file: path of file containing SST anomaly data
    :param station_name: name of the lightstation
    :param plot_name: full path name to save the plot to
    :param best_fit: None or "lstq" or the full path to a csv file containing the
    best fit results
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
        if sum(mask) == 0:
            print('station_name,', station_name, ', does not match any in',
                  best_fit)
            return
        # Get least-squares trend from entire record
        slope = float(regr_results.iloc[mask, 1])
        # Get least-squares y-intercept from entire record
        y_int = float(regr_results.iloc[mask, 5])
        # Plot the line
        x_linspace = np.linspace(date_flat[0], date_flat[-1], 100)
        y_hat_linspace = slope / 100 * x_linspace + y_int  # Slope in per 100 years?
        ax.plot(x_linspace, y_hat_linspace, c='k')
        # Add CI dotted or different colour lines
        y_ci = float(regr_results.iloc[mask, 4])
        y_lower_ci = y_hat_linspace - y_ci
        y_upper_ci = y_hat_linspace + y_ci
        ax.plot(x_linspace, y_lower_ci, '-b')
        ax.plot(x_linspace, y_upper_ci, '-b')

    # Plot formatting
    ax.set_xlim((min(date_flat), max(date_flat)))
    # ybot, ytop = plt.ylim()
    # ax.set_ylim((-max(abs(np.array([ybot, ytop]))),
    #              max(abs(np.array([ybot, ytop])))))
    ax.set_ylim((-4, 4))
    ax.set_ylabel('Temperature anomaly ($^\circ$C)')
    ax.minorticks_on()
    plt.tick_params(which='major', direction='in',
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(which='minor', direction='in',
                    bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', bottom=False, direction='in')
    # ax.set_title(
    #     f'Time series of temperature anomalies at {station_name}')
    ax.set_title(station_name, loc='left')
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close(fig)
    return


def plot_data_gaps():
    """
    Make a bar-like plot showing where there are gaps in each lightstations' record
    Follow
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py
    Make mask of 0s and 1s based on whether there is a measurement
    Plot both 0s and 1s in the horizontal bar chart and plot the zeros as white so they don't show up
    :return:
    """
    # old_dir = os.getcwd()
    # new_dir = os.path.dirname(old_dir)
    # os.chdir(new_dir)

    # Read in the data
    raw_file_list = glob.glob('.\\data\\monthly\\*MonthlyTemp.csv')
    if len(raw_file_list) == 0:
        print('Check suffix of raw files; empty file list returned')
        return
    raw_file_list.sort(reverse=True)

    min_year, max_year = [1921, 2023]

    # Make the plot
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()

    for i in range(len(raw_file_list)):
        df = pd.read_csv(raw_file_list[i], index_col='Year',
                         na_values=[99.99, 999.9, 999.99])
        # Flatten the dataframe
        date, sst = flatten_dframe(df)
        # Make a mask from the dataframe using ~pd.isna(df)
        mask = ~pd.isna(sst)
        sst_here = np.ones(sum(mask)) * (2 * i)
        # Plot the data with vline marker (square 's' marker too big?)
        # Change marker opacity with alpha
        ax.scatter(date[mask], sst_here, marker='|', s=10, color='b')
        # Label the series with the station name
        ax.text(x=max_year + 2, y=sst_here[0] - 0.2,
                s=STATION_NAMES[len(STATION_NAMES) - 1 - i])

    # Style the plot
    ax.tick_params(axis="x", direction="in", bottom=True, top=True)
    ax.tick_params(axis="y", direction="in", left=True, right=True)
    ax.set(yticklabels=[])  # Remove y axis tick labels
    old_xticks = ax.get_xticks()  # Remove the x ticks below the station names
    new_xticks = np.arange(old_xticks[0], old_xticks[-2] + 1, 10, dtype=int)
    ax.set_xticks(new_xticks)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_xlim((min_year - 2, max_year + 25))
    plt.tight_layout()

    plt.savefig('.\\figures\\lightstation_data_gaps.png')
    plt.close(fig)

    return


def plot_daily_filled_anomalies(year: int, suffix: str, do_smooth: bool, window=None):
    """
    Plot the most recent year's raw data on top of the daily average of all time
    for each station
    inputs:
        - year: The year for which to plot the daily data
        - do_smooth: Whether to smooth the daily average data to get a smooth curve
        - window: size of window (in units of days) to smooth with a rolling mean.
                  If do_smooth is *True*, then window defaults to 21 days if a size is not given
    :return:
    """
    # old_dir = os.getcwd()
    # new_dir = os.path.dirname(old_dir)
    # os.chdir(new_dir)
    new_dir = os.getcwd()

    # Use daily data instead of monthly mean observations
    daily_file_list = glob.glob(new_dir + f'\\data\\daily\\*{suffix}')
    if len(daily_file_list) == 0:
        print('Check suffix of raw files; empty file list returned')
        return
    daily_file_list.sort()

    final_daily_list = daily_file_list
    # for elem in daily_file_list:
    #     if all([nm not in elem for nm in ['Departure', 'Egg', 'McInnes', 'Nootka']]):
    #         final_daily_list.append(elem)

    if len(final_daily_list) != len(STATION_NAMES):
        print('List of daily data files does not match length of STATION_NAMES global variable! Exiting')
        return

    # # Need a smoother curve than the monthly climatology --> compute daily climatology
    # climatology_file = '.\\analysis\\lighthouse_sst_climatology_1991-2020.csv'
    # # Add numeric index and push the station name index to the first column
    # df_clim = pd.read_csv(climatology_file)

    days_per_year = 365
    month_numbers = np.arange(1, 12 + 1, 1)

    # Set y axis limits to visually compare plots more easily
    ylim = (4, 20)

    xtick_labels = MONTH_ABBREV
    # Get the xtick locations in the correct units
    month_datetime = pd.to_datetime([f'{m}/1/{year}' for m in month_numbers])
    xtick_locations = month_datetime.dayofyear.to_numpy()

    # subplot_letters = 'abcdefgh'

    for i in range(len(final_daily_list)):
        # Read in fixed width file
        if suffix.endswith('txt'):
            df_obs = pd.read_fwf(final_daily_list[i], skiprows=3,
                                 na_values=[99.99, 999.9, 999.99])
            temperature_colname = 'Temperature(C)'
        elif suffix.endswith('csv'):
            df_obs = pd.read_csv(final_daily_list[i], skiprows=1,
                                 na_values=[99.99, 999.9, 999.99])
            df_obs['Year'] = [x[:4] for x in df_obs['DATE (YYYY-MM-DD)']]
            df_obs['Month'] = [x[5:7] for x in df_obs['DATE (YYYY-MM-DD)']]
            df_obs['Day'] = [x[8:10] for x in df_obs['DATE (YYYY-MM-DD)']]
            temperature_colname = 'TEMPERATURE ( C )'
        else:
            print('Suffix', suffix, 'not supported; exiting !')
            return

        # Convert the year-month-day columns into floats
        df_obs['Datetime'] = pd.to_datetime(df_obs.loc[:, ['Year', 'Month', 'Day']])
        # df_obs['Float_year'] = df_obs['Datetime'].dt.dayofyear / days_per_year + df_obs['Year']

        # Mask of selected year
        mask_year = [int(y) == year for y in df_obs.loc[:, 'Year']]

        # Check that there are data available from the selected year
        try:
            last_dayofyear = df_obs.loc[mask_year, 'Datetime'].dt.dayofyear.to_numpy()[-1]
            # print('last day of year', year, ':', last_dayofyear)
        except IndexError:
            print(f"No data available from {STATION_NAMES[i]} in year {year}; skipping")
            continue

        # Compute the average of all observations for each day of the year (not just avg over 1991-2020)
        # Initialize an array to hold the daily climatological values
        daily_clim = pd.Series(data=days_per_year, dtype=float)

        # Populate the array with mean values
        for k in range(1, days_per_year + 1):
            daily_clim.loc[k - 1] = df_obs.loc[
                df_obs['Datetime'].dt.dayofyear == k, temperature_colname
            ].mean(skipna=True)
        if do_smooth:
            # Use 21-day rolling average as default
            if window is None:
                window = 21
            # Pad the start of the climatology with *window*-number of days
            # of the end of the climatology, so that there are no nans in January
            daily_clim_pad = pd.Series(data=np.zeros(len(daily_clim) + window))

            # End index is inclusive in pandas .loc, so need the -1
            # Must include the tolist() part otherwise daily_clim_pad gets set to NaNs
            # maybe because the indices aren't aligned?
            daily_clim_pad.loc[:window - 1] = daily_clim.loc[len(daily_clim) - window:].tolist()
            daily_clim_pad.loc[window:] = daily_clim.loc[:].tolist()
            # Set the labels at the center of the window, defaults to right of window
            daily_clim_smooth = daily_clim_pad.rolling(window=window, center=True).mean()
            # Remove the pads
            clim_to_plot = daily_clim_smooth.loc[window:]

            # Reset the index of the series to start at zero instead of leaving it
            # starting at 14; don't keep the original index as a new column
            clim_to_plot.reset_index(drop=True, inplace=True)
        else:
            clim_to_plot = daily_clim

        # Plot the climatology
        fig, ax = plt.subplots(figsize=[6, 3])
        ax.plot(np.arange(1, 365 + 1), clim_to_plot, c='k', linewidth=1.5)

        # Plot the anomalies: x, y1, y2
        # Get the last day of year of the chosen year to index the climatology with
        ax.fill_between(
            df_obs.loc[mask_year, 'Datetime'].dt.dayofyear,
            clim_to_plot.loc[:last_dayofyear - 1],  # -1 because indexing starts at zero
            df_obs.loc[mask_year, temperature_colname],
            where=(
                    df_obs.loc[mask_year, temperature_colname].to_numpy() >
                    clim_to_plot.loc[:last_dayofyear - 1].to_numpy()
            ),
            color='r'
        )
        ax.fill_between(
            df_obs.loc[mask_year, 'Datetime'].dt.dayofyear,
            clim_to_plot.loc[:last_dayofyear - 1],
            df_obs.loc[mask_year, temperature_colname],
            where=(
                    df_obs.loc[mask_year, temperature_colname].to_numpy() <
                    clim_to_plot.loc[:last_dayofyear - 1].to_numpy()
            ),
            color='b'
        )
        # Format the plot

        # Set the y limits
        ax.set_ylim(ylim)
        # Label the x and y axes, units are integer day of year
        ax.set_xticks(ticks=xtick_locations, labels=xtick_labels, rotation=45)
        ax.tick_params(axis="x", direction="in", bottom=True, top=True)
        ax.tick_params(axis="y", direction="in", left=True, right=True)
        # Add light grey grid lines
        ax.set_axisbelow(True)
        plt.grid(which='major', axis='y', color='lightgrey', linestyle='-')
        # Add labels
        ax.set_ylabel('Temperature ($^\circ$C)')
        ax.set_title(STATION_NAMES[i], loc='left')
        plt.tight_layout()

        # Save the plot
        plot_name = STATION_NAMES[i].replace(' ', '_') + f'_daily_anomalies_{year}.png'

        if do_smooth:
            plot_name = plot_name.replace('.png', f'_{window}_day_smooth.png')

        plt.savefig(
            os.path.join(new_dir, 'figures', plot_name),
            dpi=300)
        plt.close(fig)

    # os.chdir(old_dir)
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


def multi_trend_table_image():
    """
    Save the table of analysis periods, trends, and confidence intervals as an image.
    ***Include both OLS and Theil-Sen trends.
    Include the Cummins & Masson (2014) data for comparison.
    Code requires and assumes that stations are only listed in alphabetical order
    everywhere
    :return:
    """
    old_dir = os.getcwd()
    new_dir = os.path.dirname(old_dir)
    os.chdir(new_dir)

    analysis_dir = os.path.join(new_dir, 'analysis')
    figures_dir = os.path.join(new_dir, 'figures')
    data_dir = os.path.join(new_dir, 'data', 'monthly')

    # Get file containing new trends
    new_trend_file = './analysis/monte_carlo_max_siml50000_ols_st_cummins.csv'
    df_new_trend = pd.read_csv(new_trend_file)

    cm2014_file = './analysis/cummins_masson_2014_trends.csv'
    cm2014_df = pd.read_csv(cm2014_file)

    # Make a dataframe to print to an image containing the analysis period
    # and the trends and confidence intervals for each station
    df_img = cm2014_df

    # Add a second level header in the dataframe
    old_header = df_img.columns.tolist()
    # print(old_header)
    new_header = [[' ', 'C&M (2014)', 'C&M (2014)'],
                  old_header]
    df_img.columns = new_header

    # Initialize columns for containing the updated results
    df_img[('New', 'Analysis period')] = np.repeat('', len(df_img))
    df_img[('New', 'Least-squares (Theil-Sen) trend in C/century')] = np.repeat(
        '', len(df_img))

    # Get the raw data in csv format
    raw_files = glob.glob(data_dir + '/*MonthlyTemp.csv')
    if len(raw_files) == 0:
        print('No monthly temperature data files found; try a different search key')
        return
    raw_files.sort()

    # Iterate through all the files
    for i in range(len(raw_files)):
        df_in = pd.read_csv(raw_files[i], index_col='Year',
                            na_values=[99.99, 999.9, 999.99])

        # Get the start and end month and year of analysis period
        # with the end including the very last record, different than nans_to_strip() function
        st_mth, st_yr, en_mth, en_yr = get_record_st_en(df_in)

        # Both for deg C/century
        ols_trend_i = np.round(
            df_new_trend.loc[i, 'Least-squares entire trend [deg C/century]'],
            decimals=2
        )
        sen_trend_i = np.round(
            df_new_trend.loc[i, 'Theil-Sen entire trend [deg C/century]'],
            decimals=2
        )
        conf_int_i = np.round(
            df_new_trend.loc[i, 'Monte Carlo confidence limit [deg C/century]'],
            decimals=2
        )

        # Update the dataframe to be exported as an image
        # Pandas multi-index: http://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
        # Use table
        df_img.loc[i, ('New', 'Analysis period')] = f'{st_mth} {st_yr} - {en_mth} {en_yr}'
        df_img.loc[
            i, ('New', 'Least-squares (Theil-Sen) trend in C/century')
        ] = '{:.2f} ({:.2f}) +/- {:.2f}'.format(ols_trend_i, sen_trend_i, conf_int_i)

    # Update the index
    df_img.rename_axis(index='Station')
    df_img.set_index(keys=[df_img.loc[:, (' ', 'Station')].values],
                     inplace=True)
    df_img.drop(columns=(' ', 'Station'), inplace=True)

    # Name for files
    df_img_name = os.path.join(
        figures_dir,
        os.path.basename(new_trend_file).replace('.csv', '_cm2014.png')
    )

    # Export the table in PNG format
    dfi.export(df_img, os.path.join(figures_dir, df_img_name))

    # Export the table in csv format first
    df_img.to_csv(
        os.path.join(analysis_dir, os.path.basename(df_img_name).replace('png', 'csv')),
        index=True
    )

    # Change the current directory back
    os.chdir(old_dir)
    return


def plot_climatology(clim_file, output_dir):
    clim_df = pd.read_csv(clim_file, index_col=[0])
    month_numbers = np.arange(1, 12 + 1)

    fig, ax = plt.subplots(figsize=[6, 4.5])  # width, height
    line_styles = ['-', ':', '--'] * 4
    # line_styles.sort()
    for i in range(len(clim_df)):
        station = clim_df.index[i]
        station_name = station.split('_')[0] + ' ' + station.split('_')[1]
        ax.plot(month_numbers, clim_df.loc[station, :],
                label=station_name, marker='.', linestyle=line_styles[i])

    plt.legend(fontsize='small', frameon=True)
    ax.set_xlim((min(month_numbers), max(month_numbers)))
    ax.set_xticks(month_numbers, labels=MONTH_ABBREV, rotation=45)
    ax.set_ylabel('Temperature ($^\circ$C)')
    # ax.set_title('Climatological monthly mean temperatures for 1991-2020')
    plt.tight_layout()
    png_name = 'bc_lightstation_monthly_mean_climatologies_1991-2020.png'
    plt.savefig(os.path.join(output_dir, png_name), dpi=300)
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


def make_density_subplot(ax: plt.Axes, df: pd.DataFrame, var: str,
                         year_ranges: list):
    """
    Make density subplot using seaborn with an accompanying statistics table
    :param ax:
    :param df: data
    :param var: variable
    :param year_ranges: list of 30-year chunks for separating the data into for statistics
    :return:
    """
    # Column names for statistic tables
    stat_table_cols = ['Time Period', 'Mean', 'Median', 'Skewness', 'Min',
                       'Max', 'SD', '25%ile', '75%ile']

    # Do a second check that there are enough data for density plot (10+ years)
    # Don't apply this to the stats table though !
    mask_density_plot = np.repeat(True, len(df))
    year_ranges_density_plot = []
    if int(year_ranges[0][-4:]) - int(year_ranges[0][:4]) < 10:
        mask_density_plot[df.loc[:, 'Year Range'] == year_ranges[0]] = False
    else:
        year_ranges_density_plot.append(year_ranges[0])

    for i in range(1, len(year_ranges)):
        year_ranges_density_plot.append(year_ranges[i])

    print(np.nanmin(df.loc[:, 'Year']))
    print('year ranges for density plot:', year_ranges_density_plot)

    df_density_plot = df.loc[mask_density_plot, :]

    # Suppress legend since it's in the time series plot
    # Have to reverse dataframe order to plot newest data in front
    color_end = 3 - len(year_ranges_density_plot) if 3 >= len(year_ranges_density_plot) else None
    sns.kdeplot(df_density_plot.loc[::-1, :], x=var, hue='Year Range', fill=True,
                legend=False, ax=ax, palette=sns.color_palette()[3:color_end:-1])
    # Add a light grey grid behind the curves
    ax.set_axisbelow(True)
    plt.grid(which='major', axis='both', color='lightgrey')
    # Save grid limits for later since seaborn is cutting off plot edges
    ylim = ax.get_ylim()
    # Save yticks for plotting boxplots later
    yticks = ax.get_yticks()
    # Add dashed vertical lines at +-3 temperature anomalies and at 0 deg C
    if var == 'Temperature Anomaly(C)':
        ax.vlines(x=[-3, 3], ymin=ylim[0], ymax=ylim[1], linestyles='dashed',
                  linewidth=0.75, colors='grey', zorder=1.5)
        ax.vlines(x=0, ymin=ylim[0], ymax=ylim[1], linestyles='solid', linewidth=0.75,
                  colors='grey', zorder=1.5)

    # Label the x-axis
    if 'Temperature' in var:
        ax.set_xlabel(var.replace('(C)', ' ($^\circ$C)'))
    else:
        ax.set_xlabel(var)
    # # Move x ticks to top
    # ax.tick_params(axis='x', labeltop=True, labelbottom=False)

    # ---------Statistics table----------

    # Add a table containing statistics below the subplot
    cellText = []
    for yr in year_ranges[::-1]:
        # pandas mean defaults to skipna=True
        mask = df['Year Range'] == yr
        cellText.append(
            [
                yr,
                "%.2f" % np.round(df.loc[mask, var].mean(), 2),
                "%.2f" % np.round(df.loc[mask, var].median(), 2),
                "%.2f" % np.round(df.loc[mask, var].skew(), 2),
                "%.2f" % np.round(df.loc[mask, var].min(), 2),
                "%.2f" % np.round(df.loc[mask, var].max(), 2),
                "%.2f" % np.round(df.loc[mask, var].std(), 2),
                "%.2f" % np.round(df.loc[mask, var].quantile(q=0.25), 2),
                "%.2f" % np.round(df.loc[mask, var].quantile(q=0.75), 2)
            ]
        )
    # tab = ax.table(cellText=cellText,  # 2d list of str
    #                  rowLabels=None,
    #                  rowColours=None,
    #                  colLabels=stat_table_cols,
    #                  loc='bottom',
    #                  fontsize=24)

    # Add colors to table rows corresponding to density curves by year range
    ncols = len(cellText[0])
    tab = table(ax, cellText=cellText[::-1], rowLabels=None,
                # cellColours=[
                #     [sns.color_palette('pastel')[3:color_end:-1][k]] * ncols
                #     for k in range(len(year_ranges))
                # ],
                cellColours=[
                    [sns.color_palette('pastel')[4 - len(year_ranges):4][k]] * ncols
                    for k in range(len(year_ranges))
                ],
                colColours=[sns.color_palette('pastel')[-3]] * ncols,  # grey for column headers
                colLabels=stat_table_cols, loc='top', cellLoc='center',
                colWidths=[
                    1.5 / ncols, 1 / ncols, 1 / ncols, 1.25 / ncols,
                    .75 / ncols, .75 / ncols, .75 / ncols, 1 / ncols,
                    1 / ncols
                ])
    # Disable auto font size
    tab.auto_set_font_size(False)
    tab.set_fontsize(4)
    # # Resize columns: make min, max, std columns narrower and Year Range wider
    # # col: The indices of the columns to auto-scale, int or sequence of ints.
    # tab.auto_set_column_width(col=np.arange(len(cellText[0])))

    # ------------------------------

    # Add a modified box plot:
    df_stats = pd.DataFrame(
        cellText,
        columns=['Year Range', 'Mean', 'Median', 'Skew', 'Min', 'Max',
                 'STD', '25%ile', '75%ile']
    )

    # Remove first year range in box plots if it's too short
    for i, yr in enumerate(year_ranges):
        if yr not in year_ranges_density_plot:
            df_stats.drop(index=i, inplace=True)
            df_stats.reset_index(drop=True, inplace=True)

    # Change datatype from string to float, except for Year Range column
    for col in df_stats.columns[1:]:
        df_stats.loc[:, col] = [float(x) for x in df_stats[col]]

    # Add column containing the y coordinates for plotting the boxplots
    # Set the y values at which the box plots will be plotted, descending order
    # Leave some buffer room at the top and bottom edges of the plot
    df_stats['y values'] = yticks[1:len(df_stats) + 1]

    # median as black dot, drawn on top of larger coloured mean dot
    # Artists with higher zorder are drawn on top.
    # https://matplotlib.org/stable/gallery/misc/zorder_demo.html#zorder-demo
    sns.scatterplot(df_stats, x='Median', y='y values', legend=False,
                    ax=ax, c='k', marker='o', s=10,
                    edgecolor='white', zorder=2.5)

    # mean as filled dot, match opacity of density curves with alpha
    sns.scatterplot(df_stats, x='Mean', y='y values', hue='Year Range',
                    palette=sns.color_palette()[3:color_end:-1],
                    legend=False, ax=ax, marker='o', s=18,
                    edgecolor='k', zorder=2.4)

    # Plot min max and quantiles as whiskers of the modified box plot
    # Need to make a new dataframe with different format to plot lines
    # with seaborn
    df_whiskers = pd.DataFrame(
        columns=['Year Range', 'minmax', 'quantiles', 'y values'])

    for i in range(len(year_ranges_density_plot)):
        yr = year_ranges_density_plot[i]
        # Add a row for min and 25%ile
        df_whiskers.loc[len(df_whiskers)] = [
            yr,
            float(df_stats.loc[i, 'Min']),
            float(df_stats.loc[i, '25%ile']),
            df_stats.loc[i, 'y values']
        ]
        # Add a row for max and 75%ile
        df_whiskers.loc[len(df_whiskers)] = [
            yr,
            float(df_stats.loc[i, 'Max']),
            float(df_stats.loc[i, '75%ile']),
            df_stats.loc[i, 'y values']
        ]

        # min/max
        sns.lineplot(
            df_whiskers.loc[len(df_whiskers) - 2:len(df_whiskers) - 1],
            x='minmax',
            y='y values',
            legend=False,
            ax=ax, c='k', linewidth=0.75, zorder=2.2
        )
        # quantiles as a thicker line
        sns.lineplot(
            df_whiskers.loc[len(df_whiskers) - 2:len(df_whiskers) - 1],
            x='quantiles',
            y='y values',
            legend=False,
            ax=ax, c='k', linewidth=2, zorder=2.3
        )

    # Reset x and y axis limits
    if var == 'Temperature(C)':
        ax.set_xlim(0, 27)
    elif var == 'Temperature Anomaly(C)':
        ax.set_xlim(-8, 8)

    ax.set_ylim(*ylim)  # Use * to unpack a tuple

    # # Add buffer to y axis
    # ax.margins(y=0.1)

    # Add subplot title with padding so it doesn't overlap with table above plot
    title_pad = len(cellText) * 10 + 20
    if var == 'Temperature(C)':
        ax.set_title('Density of Temperature Observations', loc='left',
                     pad=title_pad)
    elif var == 'Temperature Anomaly(C)':
        ax.set_title('Density of Temperature Anomalies', loc='left',
                     pad=title_pad)
    return


def plot_daily_T_statistics():
    """
    Plot daily SST and anomaly density curves with statistics tables
    :return:
    """
    # Divide the data into 30-year chunks working backwards
    daily_file_list = glob.glob('.\\data\\daily\\*.txt')
    if len(daily_file_list) == 0:
        print('Check suffix of raw files; empty file list returned')
        return
    daily_file_list.sort()

    # Remove stations we don't want
    final_daily_list = []
    for elem in daily_file_list:
        if all([nm not in elem for nm in ['Departure', 'Egg', 'McInnes', 'Nootka']]):
            final_daily_list.append(elem)

    # Check on number of files remaining
    if len(final_daily_list) != len(STATION_NAMES):
        print('List of daily data files does not match length of STATION_NAMES global variable! Exiting')
        return

    days_per_year = 365

    # # Lower the font size
    # sns.set_theme(font_scale=0.5)
    mpl.rcParams['font.size'] = 5

    # Iterate through the files and make a plot for each station
    # for i in range(7, len(final_daily_list)):
    for i in range(len(final_daily_list)):
        print(os.path.basename(final_daily_list[i]))
        # Read in fixed width file
        df_obs = pd.read_fwf(final_daily_list[i], skiprows=3,
                             na_values=[99.99, 999.9, 999.99])

        # Compute all-time daily means and use to get anomalies

        # Convert the year-month-day columns into floats
        df_obs['Datetime'] = pd.to_datetime(df_obs.loc[:, ['Year', 'Month', 'Day']])
        # df_obs['Float_year'] = df_obs['Datetime'].dt.dayofyear / days_per_year + df_obs['Year']

        # Compute the average of all observations for each day of the year
        # Initialize an array to hold the daily climatological values
        daily_clim = pd.Series(data=days_per_year, dtype=float)

        # Initialize column for anomalies
        df_obs['Temperature Anomaly(C)'] = np.zeros(len(df_obs))

        # Populate the array with mean values
        for k in range(1, days_per_year + 1):
            daily_clim.loc[k - 1] = df_obs.loc[
                df_obs['Datetime'].dt.dayofyear == k, 'Temperature(C)'
            ].mean(skipna=True)
            # Populate the anomalies column
            df_obs.loc[
                df_obs['Datetime'].dt.dayofyear == k,
                'Temperature Anomaly(C)'
            ] = df_obs.loc[
                    df_obs['Datetime'].dt.dayofyear == k,
                    'Temperature(C)'
                ] - daily_clim.loc[k - 1]

        # Create masks for data subsetting by year
        year_col = df_obs['Year'].to_numpy()
        df_obs['Year Range'] = np.repeat('', len(df_obs))
        year_ranges_all = ['1904-1933', '1934-1963', '1964-1993', '1994-2023']
        year_ranges = []  # Initialize list for all available year ranges to not iterate over a changing list length

        for k in range(len(year_ranges_all)):
            yr = year_ranges_all[k]
            st = int(yr.split('-')[0])
            en = int(yr.split('-')[1])
            # Proceed if there are data from the select year range
            obs_exist_from_year_range = len(df_obs.loc[(st <= year_col) & (year_col <= en), 'Year Range']) > 0
            # min_years_in_range = 10  # short enough to capture 1920's data at Race Rocks
            # enough_years_in_year_range = np.nanmin(df_obs.loc[:, 'Year']) - st > min_years_in_range
            if obs_exist_from_year_range:  # & enough_years_in_year_range:
                if st < int(df_obs['Year'].min()):
                    st = int(df_obs['Year'].min())
                    year_ranges_all[k] = str(st) + '-' + yr.split('-')[1]
                    yr = year_ranges_all[k]
                # Populate the Year Range column
                df_obs.loc[(st <= year_col) & (year_col <= en), 'Year Range'] = yr
                year_ranges.append(yr)
            # else:
            #     # Drop year range from list since no obs from that time
            #     year_ranges.remove(yr)
            #     # # Modify the earliest range containing data to what year
            #     # # each record starts
            #     # st = int(df_obs['Year'].min())
            #     # year_ranges[k - 1] = str(st) + '-' + year_ranges[k - 1].split('-')[1]
            #     # yr = year_ranges[k - 1]
            #     # df_obs.loc[(st <= year_col) & (year_col <= en), 'Year Range'] = yr

        # Temporarily change the matplotlib.rcParams object
        # with mpl.rc_context({'font.size': 14, 'font.weight': 'light'}):
        # Plot temperature (nrows, ncols, index)
        ax1 = plt.subplot(211)  # 212)
        # Enforce gaps in lineplot where there are nan data values
        # The source code looks like lineplot drops nans from the DataFrame before plotting unfortunately
        # seaborn.pointplot could fix the problem but it was too slow
        # sns.lineplot(df_obs, x='Datetime', y='Temperature(C)', hue='Year Range',
        #              ax=ax1, linewidth=0.5)
        # sns.pointplot(df_obs, x='Datetime', y='Temperature(C)', hue='Year Range',
        #               ax=ax1)

        #          '1904-1933', '1934-1963', '1964-1993', '1994-2023'
        colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # sns.color_palette(n_colors=len(year_ranges))
        for k in range(len(year_ranges)):
            yr = year_ranges[k]
            mask = df_obs.loc[:, 'Year Range'] == yr
            ax1.plot(df_obs.loc[mask, 'Datetime'], df_obs.loc[mask, 'Temperature(C)'],
                     linewidth=0.2, marker='o', markeredgecolor=None, markersize=0.3,
                     c=colours[4 - len(year_ranges) + k], label=yr,
                     markerfacecolor=colours[4 - len(year_ranges) + k])
        plt.grid(which='major', axis='both', color='lightgrey')
        ax1.legend(loc='upper left', ncol=2)
        ax1.set_ylim((0, 27))
        ax1.margins(y=0.1)
        ax1.set_ylabel('Temperature ($^\circ$C)')
        # # Increase x tick frequency to every 10 years
        # xticks = ax1.get_xticks()
        # ax1.set_xticks(np.arange(xticks[0], xticks[-1] + 1, 10))
        # Remove x axis label
        ax1.set_xlabel(None)
        ax1.set_title(STATION_NAMES[i], loc='left')

        # Compute density curve for temperature
        ax2 = plt.subplot(223)  # 221)

        make_density_subplot(ax2, df_obs, 'Temperature(C)', year_ranges)

        # Plot the temperature anomalies
        ax3 = plt.subplot(224)  # 222

        make_density_subplot(ax3, df_obs, 'Temperature Anomaly(C)', year_ranges)

        # plt.suptitle(STATION_NAMES[i], horizontalalignment='left')
        # Add padding between first and second row of plots
        plt.subplots_adjust(hspace=-0.25)
        plt.tight_layout()
        plt_name = STATION_NAMES[i].replace(' ', '_') + '_sst_statistics.png'
        plt.savefig(f'.\\figures\\{plt_name}', dpi=400)
        plt.close()

        # # Format the plot
        # # Label the x and y axes, units are integer day of year
        # ax.tick_params(axis="x", direction="in", bottom=True, top=True)
        # ax.tick_params(axis="y", direction="in", left=True, right=True)

    return


def run_plot(
        monthly_anom: bool = False,
        clim: bool = False,
        daily_anom: bool = False,
        daily_anom_window=None,
        daily_stats: bool = False,
        availability: bool = False
):
    """
    Run plotting functions
    :param monthly_anom: Plot monthly anomalies with least-squares trend
    and Monte Carlo 95% confidence interval
    :param clim: Plot monthly climatologies
    :param daily_anom: plot daily anomalies
    :param daily_anom_window: the size of window to use for plotting daily
    anomalies, if smoothing them
    :param daily_stats: plot statistics of daily data
    :param availability: plot data availability (monthly and annual observation
    counts)
    :return:
    """
    old_dir = os.getcwd()
    new_dir = os.path.dirname(old_dir)
    os.chdir(new_dir)

    # Plot climatological monthly means at each station
    output_folder = os.path.join(new_dir, 'figures')

    input_folder = os.path.join(new_dir, 'analysis')

    anom_files = glob.glob(
        input_folder + '\\*monthly_anom_from_monthly_mean.csv',
        recursive=False
    )
    if len(anom_files) == 0:
        print('No anomaly files returned with suffix monthly_anom_from_monthly_mean.csv',
              'in directory', input_folder, '; exiting !')
    else:
        anom_files.sort()

    # Get trends and confidence intervals file
    monte_carlo_results = os.path.join(
        input_folder,
        "monte_carlo_max_siml50000_ols_st_cummins.csv"
    )

    if availability:
        plot_data_gaps()

    if monthly_anom:
        for stn, f in zip(STATION_NAMES, anom_files):
            png_filename = os.path.join(
                output_folder, stn.replace(' ', '_') + '_monthly_mean_sst_anomalies_ols.png'
            )
            plot_monthly_anomalies(f, stn, png_filename, best_fit=monte_carlo_results)

    if clim:
        clim_file = os.path.join(
            input_folder, 'lighthouse_sst_climatology_1991-2020.csv')
        plot_climatology(clim_file=clim_file, output_dir=output_folder)

    if daily_anom:
        plot_daily_filled_anomalies(2023, do_smooth=True, window=daily_anom_window,
                                    suffix='.csv')

    if daily_stats:
        plot_daily_T_statistics()

    os.chdir(old_dir)
    return


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
