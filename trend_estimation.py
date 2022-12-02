import glob
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from sklearn.linear_model import TheilSenRegressor
from statsmodels.tsa.stattools import acovf
from scipy.interpolate import interp1d
import os

from patsy import dmatrices
import statsmodels.api as sm
from scipy.stats import t


def acvf_te(y, N_max):
    """DEPREC
    Autocovariance function for serially correlated data record.
    Thomson & Emery (2014) eqn. 3.139a.
    :param N_max: Maximum number of reasonable lag values (starting at zero lag and
    going to (<<N/2) that can be calculated before the summation becomes erratic.
    :param y: The dependent variable
    :return: CC(tau_kappa), where tau_kappa = k * dtau and dtau is the lag time step
    """
    y_bar = np.mean(y)
    N = len(y)
    CC = np.zeros(N)
    # +1 because Python ranges not inclusive
    for k in range(0, N_max + 1):
        for i in range(1, N - k):
            CC[k] += (y[i] - y_bar) * (y[i + k] - y_bar)
        CC[k] *= 1 / (N - 1 - k)
        # C[0] should equal np.var(y), the variance of the series
    return CC


def integral_timescale_discrete(dtau, CC, m):
    """DEPREC
    Compute the integral timescale T for the data record
    :param m: number of lag values incorporated in the summation
    :param dtau: lag time step (could be the same as the sampling time step, dt
    :param CC: Autocovariance function values
    :return: T, the integral timescale
    """
    # How to choose how many lag values to incorporate into the summation?
    TT = 0
    for k in range(0, m - 1):
        TT += dtau / 2 * (CC[k * dtau + dtau] + CC[k * dtau])
    TT *= 2 / CC[0]
    return TT


def C(k, y):
    """
    Autocovariance function for some k in [0,...,N_max]
    :param k:
    :param y: data series
    :return: C, the autocovariance of y
    """
    # todo confirm indexing is correct
    N = len(y)
    y_bar = np.mean(y)
    C_k = 0
    # In the textbook, i starts at 1 not zero
    for i in range(0, N - k):
        C_k += (y[i] - y_bar) * (y[i + k] - y_bar)
    C_k *= 1 / (N - 1 - k)

    return C_k


def integral_timescale_corrected(dtau, y, m):
    T = 0
    # m = N_max ?
    # todo confirm indexing is correct
    for k in range(0, m - 1):  # Add +1??
        # C(tau_k + dtau) = C(k*dtau + dtau) = C((k + 1)*dtau)
        T += dtau / 2 * (C(k + 1, y) + C(k, y))
    T *= 2 / C(0, y)

    return T


def effective_sample_size(N, dt, T):
    """
    Calculate the effective sample size of a time series dataset.
    N*dt is the total length (duration) of the record.
    Reference: Thomson & Emery, 2014 (3rd ed.), eqn. 3.138
    :param N: Number of samples in the series
    :param dt: Sampling increment
    :param T: Integral timescale (discrete) for the data record
    :return: N_star, the effective degrees of freedom
    """
    return N * dt / T


def flatten_dframe(df: pd.DataFrame):
    """
    Flatten dataframe into 1d. Convert year-month to float date.
    :param df:
    :return: flattened date (x) and flattened temperature record (y)
    """
    years = df.index.to_numpy(float)
    months = np.linspace(0, 1, 12 + 1)[:-1]
    years2d, months2d = np.meshgrid(years, months)
    x = (years2d.T + months2d.T).flatten()
    y = df.to_numpy(float).flatten()
    return x, y


def standard_error_of_estimate(y, y_est):
    """
    compute the standard error of the estimate, s_eta
    :param y:
    :param y_est: y-hat
    :return:
    """
    N = len(y)
    s_eta = (1 / (N - 2) * sum((y - y_est) ** 2)) ** (1 / 2)

    return s_eta


def std_dev_x(x):
    """
    Standard deviation for the x variable
    :param x:
    :return:
    """
    N = len(x)
    x_bar = np.mean(x)
    s_x = (1 / (N - 1) * sum((x - x_bar) ** 2)) ** (1/2)

    return s_x


def conf_limit(s_eta, N_star, s_x):
    """
    Compute the confidence limit on least squares slope
    :param s_eta: standard error of the estimate
    :param N_star: effective sample size
    :param s_x: standard deviation for the x variable
    :return:
    """
    deg_freedom = N_star - 2  # nu
    alpha = 0.05
    # t.cdf(x:quantiles, arg1:shape parameters)
    # return (s_eta * t.cdf(alpha / 2, df=deg_freedom)) / ((N_star - 1) ** .5 * s_x)
    # Use two-tailed t-test t.ppf()
    # https://stackoverflow.com/questions/19339305/python-function-to-get-the-t-statistic
    return (s_eta * t.ppf(1 - alpha / 2, df=deg_freedom)) / ((N_star - 1) ** .5 * s_x)


def main_te(compute_cis=False, delta_tau_factor=1, nlag_vals_to_use=50):
    """
    Thomson & Emery (2014) confidence limit calculation method
    :param delta_tau_factor: factor to multiply delta-t with to get delta-tau.
    delta-tau=factor*delta-t. delta-tau represents the lag time step
    :param compute_cis: do or do not compute confidence intervals
    :param nlag_vals_to_use: Number of lag values to include in the summation.
    Default number is the same as Patrick uses with Monte Carlo: 50 lags
    :return:
    """
    parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
                 'lighthouse_data\\monthly_anom_from_monthly_mean\\'
    file_list = glob.glob(parent_dir + '*monthly_anom_from_monthly_mean.csv')
    file_list.sort()
    # Number of leading and trailing nans to discard
    nstrip = np.zeros((len(file_list), 2), dtype=int)
    nstrip[0, :] = [7, 8]  # Amphitrite
    nstrip[1, :] = [3, 8]  # Bonilla
    nstrip[2, :] = [3, 8]  # Chrome
    nstrip[3, :] = [4, 36]  # Entrance
    nstrip[4, :] = [0, 8]  # Kains
    nstrip[5, :] = [50, 9]  # Langara
    nstrip[6, :] = [0, 8]  # Pine
    nstrip[7, :] = [1, 8]  # Race Rocks
    data_file = parent_dir + 'Amphitrite_Point_monthly_anom_from_monthly_mean.csv'
    # Initialize dataframe to hold results
    if compute_cis:
        df_res = pd.DataFrame(columns=['Record length',
                                       'Original degrees of freedom',
                                       'Effective degrees of freedom',
                                       'OLS slope [deg C/century]',
                                       'Original confidence limit [deg C/century]',
                                       'Effective confidence limit [deg C/century]'])
    else:
        df_res = pd.DataFrame(columns=['Record length',
                                       'Original degrees of freedom',
                                       'Effective degrees of freedom'])
    # Iterate through each lighthouse station
    for j in range(len(file_list)):
        data_file = file_list[j]
        data_file_idx = j
        basename = os.path.basename(data_file)
        station_name = basename.split('_')[0] + ' ' + basename.split('_')[1]
        dframe = pd.read_csv(data_file, index_col=[0])
        # Reformat dataframe into 1d with float type date
        xx, yy = flatten_dframe(dframe)

        # Remove leading and trailing nans
        xx = xx[nstrip[data_file_idx, 0]:-nstrip[data_file_idx, 1]]
        yy = yy[nstrip[data_file_idx, 0]:-nstrip[data_file_idx, 1]]
        # fill gaps with spline interpolation here as well as in Patrick's method??
        xx_with_gaps = xx[pd.notna(yy)]
        yy_with_gaps = yy[pd.notna(yy)]
        # First-order spline interpolation function
        spline_fn = interp1d(xx_with_gaps, yy_with_gaps, kind='slinear')
        yy_filled = spline_fn(xx)

        # Set up parameters for analysis
        NN = len(yy_filled)
        time_step = xx[1] - xx[0]  # Unit of 365 days (a year ish)
        # lag time step, delta-tau
        delta_tau = delta_tau_factor * time_step
        # acv = acvf_te(yy_filled, nlag_vals_to_use)
        # Check that acv approaches zero as tau approaches N (see pg. 274)
        # plt.plot(acv)
        it = integral_timescale_corrected(delta_tau, yy_filled, nlag_vals_to_use)
        ESS = effective_sample_size(NN, time_step, it)
        N_star = ESS - 2
        print('The effective degrees of freedom are', N_star)
        print('The original degrees of freedom are N-2 =', NN - 2)

        if compute_cis:
            # Compute least-squares slope
            # Do not include nan values in the dataframe for the model
            dfmod = pd.DataFrame(
                {'Date': xx, 'Anomaly': yy_filled}
            )
            # create design matrices
            y, X = dmatrices('Anomaly ~ Date', data=dfmod, return_type='dataframe')

            mod = sm.OLS(y, X)  # Describe model
            res = mod.fit()  # Fit model
            y_estimate = res.fittedvalues
            # Compute confidence intervals for least squares trend
            SEE = standard_error_of_estimate(yy_filled, y_estimate)
            SDX = std_dev_x(xx)
            CI_effective = conf_limit(SEE, N_star, SDX)
            print('The confidence interval on the least-squares slope is:',
                  CI_effective)
            CI_original = (res.conf_int(alpha=0.05).iloc[1, 1] -
                           res.conf_int(alpha=0.05).iloc[1, 0])/2
            df_res.loc[station_name] = [NN, NN - 2, N_star, res.params[1] * 100,
                                        CI_original * 100, CI_effective * 100]
        else:
            df_res.loc[station_name] = [NN, NN - 2, N_star]

    # Export results
    results_filename = os.path.join(
        parent_dir, 'effective_df',
        'lighthouse_effective_deg_freedom_nlag{}_dtau{}_Dec1.csv'.format(
            nlag_vals_to_use, int(delta_tau * 100)))
    # Since delta-t is less than 1 and delta-tau depends on delta-t, x by 100
    df_res.to_csv(results_filename, index=True)
    return


def main_ols():
    parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
                 'lighthouse_data\\monthly_anom_from_monthly_mean\\'
    file_list = glob.glob(parent_dir + '*monthly_anom_from_monthly_mean.csv')
    file_list.sort()
    # Number of leading and trailing nans to discard
    nstrip = np.zeros((len(file_list), 2), dtype=int)
    nstrip[0, :] = [7, 8]  # Amphitrite
    nstrip[1, :] = [3, 8]  # Bonilla
    nstrip[2, :] = [3, 8]  # Chrome
    nstrip[3, :] = [4, 36]  # Entrance
    nstrip[4, :] = [0, 8]  # Kains
    nstrip[5, :] = [50, 9]  # Langara
    nstrip[6, :] = [0, 8]  # Pine
    nstrip[7, :] = [1, 8]  # Race Rocks
    # data_file = parent_dir + 'Amphitrite_Point_monthly_anom_from_monthly_mean.csv',
    # Initialize dataframe to hold results
    df_res = pd.DataFrame(columns=['OLS degrees of freedom'])

    for j in range(len(file_list)):
        data_file = file_list[j]
        data_file_idx = j
        basename = os.path.basename(data_file)
        station_name = basename.split('_')[0] + ' ' + basename.split('_')[1]
        dframe = pd.read_csv(data_file, index_col=[0])
        # Reformat dataframe into 1d with float type date
        xx, yy = flatten_dframe(dframe)

        # Remove leading and trailing nans
        xx = xx[nstrip[data_file_idx, 0]:-nstrip[data_file_idx, 1]]
        yy = yy[nstrip[data_file_idx, 0]:-nstrip[data_file_idx, 1]]
        # fill gaps with spline interpolation here as well as in Patrick's method??
        xx_with_gaps = xx[pd.notna(yy)]
        yy_with_gaps = yy[pd.notna(yy)]
        # First-order spline interpolation function
        spline_fn = interp1d(xx_with_gaps, yy_with_gaps, kind='slinear')
        yy_filled = spline_fn(xx)

        # Do not include nan values in the dataframe for the model
        dfmod = pd.DataFrame(
            {'Date': xx, 'Anomaly': yy_filled}
        )
        # create design matrices
        y, X = dmatrices('Anomaly ~ Date', data=dfmod, return_type='dataframe')

        mod = sm.OLS(y, X)  # Describe model
        res = mod.fit()  # Fit model

        df_res.loc[station_name] = res.df_resid

    # Export results
    results_filename = os.path.join(
        parent_dir, 'effective_df', 'lighthouse_ols_deg_freedom.csv')
    df_res.to_csv(results_filename, index=True)
    return


# --------------------------------------------------------------------------------
# Patrick's method


def monte_carlo_trend(max_siml, maxlag, time, data_record, sen_flag):
    """
    Trend analysis via Monte Carlo simulations. Code translated from Patrick Cummins'
    Matlab code. See Cummins & Masson (2014) and Cummins & Ross (2020)
    :param max_siml:
    :param maxlag:
    :param time:
    :param data_record:
    :param sen_flag:
    :return: siml_trends,mean_acvf,std_acvf,lags, mean_spec
    """
    npts = len(data_record)
    print('Number of points:', npts)
    # data_std = np.std(data_record)

    # # Set index of Nyquist frequency
    # nqst = (npts + 1)/2  # odd number of pts
    # if npts % 2 == 0:
    #     nqst = npts/2 + 1  # even number of pts

    # Initialize output arrays for simulation results
    siml_trends = np.zeros(max_siml)
    # Autocovariance function
    siml_acvf = np.zeros(max_siml, 2 * maxlag + 1)

    data_mag = abs(fft(data_record))

    mean_spec = np.zeros(npts)

    for nsim in range(max_siml):
        # Set the seed
        np.random.seed(1234 + nsim)

        # Generate a 1-by-npts row vector of uniformly distributed
        # numbers in the interval [-2pi, 2pi)?
        ph_ang = 2 * np.pi * np.random.random_sample(npts)
        dummy_fft = data_mag * np.exp(li * ph_ang)  # todo

        # Take inverse fft to obtain simulated time series
        dummy_ts = np.sqrt(2) * np.real(ifft(dummy_fft))

        # Calculate the mean power spectrum of each simulated time series
        # to form average
        mean_spec += (abs(fft(dummy_ts)) ** 2) / npts

        # Theil-Sen regression to find trend
        res = TheilSenRegressor(random_state=0).fit(time, dummy_ts)
        siml_trends[nsim] = res.get_params()

        # As a check, compute the autocovariance for each simulated time series
        # Cannot specify normalization option as "biased" unlike in Matlab...
        xc = acovf(dummy_ts, nlag=maxlag)
        siml_acvf[nsim, :] = xc

    # Get the mean and std dev of the autocorrelation functions
    mean_acvf = np.mean(siml_acvf)
    std_acvf = np.std(siml_acvf)

    # Complete averaging to get the mean power spectrum
    mean_spec = mean_spec / max_siml

    return siml_trends, mean_acvf, std_acvf, mean_spec  # ,lags


def calc_trend():
    max_siml = 50000
    maxlag = 50
    return

# ------------------------------------------------------------------------
