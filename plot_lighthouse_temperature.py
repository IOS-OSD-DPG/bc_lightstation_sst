import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np


def plot_lighthouse_t(anom_file, station_name, output_dir):
    anom_df = pd.read_csv(anom_file, index_col='YEAR')
    n_years = len(anom_df)
    n_months = 12

    # station_name = anom_file.split('_')[0] + ' ' + anom_file.split('_')[1]

    # Flatten the data into 1 dim for plot
    # Convert the names of the months to numeric
    month_numeric = np.linspace(0, 1, 12 + 1)[:-1]

    date_numeric = np.zeros(shape=(n_years, len(month_numeric)))
    for i in range(n_years):
        for j in range(n_months):
            date_numeric[i, j] = anom_df.index[i] + month_numeric[j]

    date_flat = date_numeric.flatten()
    anom_flat = anom_df.to_numpy().flatten()

    fig, ax = plt.subplots(figsize=[6, 3])  # width, height

    ax.plot(date_flat, anom_flat, c='grey')

    # Add a best-fit line
    # First remove nans
    date_nonan = date_flat[~np.isnan(anom_flat)]
    anom_nonan = anom_flat[~np.isnan(anom_flat)]
    # Compute polynomial
    poly = np.polynomial.Polynomial.fit(
        date_nonan, anom_nonan, deg=1)
    x_linspace, y_hat_linspace = poly.linspace(n=100)
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
    ax.set_title(
        f'Time series of temperature anomalies at {station_name}')
    plt.tight_layout()
    png_filename = station_name.replace(' ', '_') + '_anomalies.png'
    plt.savefig(output_dir + png_filename)
    plt.close()
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
    png_filename = 'bc_lighthouse_monthly_mean_climatologies_1991-2020.png'
    plt.savefig(os.path.join(output_dir, png_filename))
    plt.close()
    return


parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
             'our_warming_ocean\\lighthouse_data\\'

# Plot climatological monthly means at each station

# Plot lighthouse temperature anomalies with linear trendline
data_dir = parent_dir + 'monthly_anomalies\\'
anom_files = glob.glob(data_dir + '*.csv')

station_names = [
    os.path.basename(folder).replace('_', ' ')
    for folder in glob.glob(parent_dir + 'DATA_-_Active_Sites\\*')
]

anom_files.sort()
station_names.sort()

for i in range(len(anom_files)):
    plot_lighthouse_t(anom_files[i], station_names[i], output_dir=data_dir)

# ---------------------------------------------------------------

lighthouse_clim = 'C:\\Users\\HourstonH\\Documents\\charles\\' \
                  'our_warming_ocean\\lighthouse_data\\' \
                  'climatological_monthly_means\\' \
                  'lighthouse_sst_climatology_1991-2020.csv'

plot_dir = os.path.dirname(lighthouse_clim)

plot_climatology(lighthouse_clim, plot_dir)
