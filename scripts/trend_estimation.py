import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from sklearn.linear_model import TheilSenRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
from statsmodels.tsa.stattools import acovf
from scipy.interpolate import interp1d
import os
from tqdm import trange
from patsy import dmatrices
import statsmodels.api as sm
from scipy.stats import t


# def acvf_te(y, N_max):
#     """DEPREC
#     Autocovariance function for serially correlated data record.
#     Thomson & Emery (2014) eqn. 3.139a.
#     :param N_max: Maximum number of reasonable lag values (starting at zero lag and
#     going to (<<N/2) that can be calculated before the summation becomes erratic.
#     :param y: The dependent variable
#     :return: CC(tau_kappa), where tau_kappa = k * dtau and dtau is the lag time step
#     """
#     y_bar = np.mean(y)
#     N = len(y)
#     CC = np.zeros(N)
#     # +1 because Python ranges not inclusive
#     for k in range(0, N_max + 1):
#         for i in range(1, N - k):
#             CC[k] += (y[i] - y_bar) * (y[i + k] - y_bar)
#         CC[k] *= 1 / (N - 1 - k)
#         # C[0] should equal np.var(y), the variance of the series
#     return CC


def C(k, y):
    """
    Thomson & Emery (2014), eqn. 3.139a
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


def integral_timescale_discrete(dtau, y, m):
    """
    Thomson & Emery (2014), eqn. 3.137a
    Integral timescale T for a data record, discrete case
    :param dtau: lag time step
    :param y: data record
    :param m: number of lag values included in the summation
    :return: T, the discrete integral timescale
    """
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
    :param df: dataframe with years as row indices and months of each year as column indices
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
    Thomson & Emery (2014), eqn. 3.134
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
    Thomson & Emery (2014), eqn. 3.135a
    Standard deviation for the x variable
    :param x:
    :return:
    """
    N = len(x)
    x_bar = np.mean(x)
    s_x = (1 / (N - 1) * sum((x - x_bar) ** 2)) ** (1 / 2)

    return s_x


def conf_limit(s_eta, N_star, s_x):
    """
    Thomson & Emery (2014), eqn. 3.136a
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


def nans_to_strip(abridged: bool):
    """
    From Patrick Cummins Matlab script.

    Store the numbers of observations to strip from the beginning and end
    of each lightstation record.

    The entire record is needed for trend calculations. The abridged
    record is needed for CI calculations.

    These numbers will need to be updated each time the plots are updated
    and if the list of stations being analyzed changes!
    """
    # Number of stations being analyzed
    N = 8
    nstrip = np.zeros((N, 2), dtype=int)
    nstrip[0, :] = [7, 9]  # Amphitrite  ,8
    nstrip[1, :] = [3, 9]  # Bonilla ,8
    nstrip[2, :] = [3, 2]  # Chrome ,8, doesn't have 2023 data yet
    nstrip[3, :] = [4, 48]  # Entrance to Dec 2019
    nstrip[4, :] = [0, 9]  # Kains
    nstrip[5, :] = [50, 9]  # Langara starting Mar 1940
    nstrip[6, :] = [0, 9]  # Pine ,8
    nstrip[7, :] = [1, 9]  # Race Rocks  ,8

    if not abridged:
        nstrip[3, :] = [4, 9]  # Entrance to Mar 2023
        nstrip[5, :] = [9, 9]  # Langara starting Oct 1936
    return nstrip


def treat_nans(x: np.ndarray, y: np.ndarray, nstart: int, nend: int):
    """
    From Patrick Cummins Matlab script
    Use first-order spline interpolation to fill gaps in lightstation SST data
    """
    # Remove leading and trailing nans that were identified in the nans_to_strip() step
    xx = x[nstart:-nend]
    yy = y[nstart:-nend]
    # fill gaps with spline interpolation here as well as in Patrick's method??
    xx_with_gaps = xx[pd.notna(yy)]
    yy_with_gaps = yy[pd.notna(yy)]
    # First-order spline interpolation function
    xx_filled = xx
    spline_fn = interp1d(xx_with_gaps, yy_with_gaps, kind='slinear')
    yy_filled = spline_fn(xx_filled)

    return xx_filled, yy_filled


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
    nstrip = nans_to_strip(abridged=True)

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
    for i in range(len(file_list)):
        data_file = file_list[i]
        data_file_idx = i
        basename = os.path.basename(data_file)
        station_name = basename.split('_')[0] + ' ' + basename.split('_')[1]
        dframe = pd.read_csv(data_file, index_col=[0])
        # Reformat dataframe into 1d with float type date
        x, y = flatten_dframe(dframe)

        xx_gaps, yy_gaps, xx, yy_filled = treat_nans(x, y, nstrip[data_file_idx, 0],
                                                     nstrip[data_file_idx, 1])

        # Set up parameters for analysis
        NN = len(yy_filled)
        time_step = xx[1] - xx[0]  # Unit of 365 days (a year ish)
        # lag time step, delta-tau
        delta_tau = delta_tau_factor * time_step
        # acv = acvf_te(yy_filled, nlag_vals_to_use)
        # Check that acv approaches zero as tau approaches N (see pg. 274)
        # plt.plot(acv)
        it = integral_timescale_discrete(delta_tau, yy_filled, nlag_vals_to_use)
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
                           res.conf_int(alpha=0.05).iloc[1, 0]) / 2
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
    # Comment out any testing before running
    parent_dir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
                 'lighthouse_data\\monthly_anom_from_monthly_mean\\'
    file_list = glob.glob(parent_dir + '*monthly_anom_from_monthly_mean.csv')
    file_list.sort()
    # Number of leading and trailing nans to discard
    nstrip = nans_to_strip(abridged=True)
    # data_file = parent_dir + 'Amphitrite_Point_monthly_anom_from_monthly_mean.csv',
    # Initialize dataframe to hold results
    df_res = pd.DataFrame(columns=['Start date', 'End date', 'Number of observations',
                                   'OLS degrees of freedom',
                                   'OLS trend (deg C/century)', '95% CI (deg C/century)'])

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

        ####### TEST #######
        # Only take data from 1981-2022
        mask_satellite_duration = xx >= 1981 + np.linspace(0, 1, 13)[8]
        xx = xx[mask_satellite_duration]
        yy = yy[mask_satellite_duration]

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

        df_res.loc[station_name] = [
            min(xx), max(xx), len(xx), res.df_resid, res.params.Date * 100,
                                                     (res.conf_int().loc['Date', 1] - res.conf_int().loc[
                                                         'Date', 0]) / 2 * 100]

    # Export results
    results_filename = os.path.join(
        parent_dir, 'least_squares', 'lighthouse_ols_trends_Sept1981-present.csv')
    df_res.to_csv(results_filename, index=True)
    return


# --------------------------------------------------------------------------------
# Patrick's method


def TheilSen_Cummins(data: np.ndarray):
    """
    Patrick Cummins' Theil-Sen regression code, translated from Matlab.
    Source:
    Gilbert, Richard O. (1987), "6.5 Sen's Nonparametric Estimator of
    Slope", Statistical Methods for Environmental Pollution Monitoring,
    John Wiley and Sons, pp. 217-219, ISBN 978-0-471-28878-7

    :param data: A MxD matrix with M observations. The first D-1 columns
    are the explanatory variables and the Dth column is the response such that
    data = [x1, x2, ..., x(D-1), y]
    :return:
        m: Estimated slope of each explanatory variable with respect to the
            response variable. Therefore, m will be a vector of D-1 slopes.
        b: Estimated offsets.
    """
    sz = data.shape
    if len(sz) != 2 or sz[0] < 2:
        print('Expecting MxD data matrix with at least 2 observations.')
        return

    if sz[1] == 2:  # Normal 2-D case
        # X = NaN(n) returns an n-by-n matrix of NaN values in matlab
        CC = np.repeat(np.nan, (sz[0] * sz[0])).reshape((sz[0], sz[0]))
        # Accumulate slopes
        for i in range(sz[0]):
            CC[i, i:] = (data[i, 1] - data[i:, 1]) / (data[i, 0] - data[i:, 0])

        # Make mask for finite values (nan not finite)
        k = np.isfinite(CC)
        m = np.median(CC[k])  # Slope estimate: take median of finite slopes

        kd = np.isfinite(data[:, 0])
        # calculate intercept if requested
        bb = np.median(data[kd, 1] - m * data[kd, 0])

        return m, bb, CC


def monte_carlo_trend(max_siml: int, maxlag, time, data_record: np.ndarray, ncores_to_use=None,
                      sen_flag=0):
    """
    Trend analysis via Monte Carlo simulations. Code translated from Patrick Cummins'
    Matlab code. See Cummins & Masson (2014) and Cummins & Ross (2020).

    Starting with a given sample time series, a set of random time series
    is generated such that the mean auto-correlation function of these time series
    is the same as that of the original record. The program returns the linear trend of each of these
    random time series (siml_trends) along with the mean and standard deviation of the
    acvf of the random time series, as well as the mean power spectrum.
    Trends are in the same units as the input data.

    Program assumes that the input data time series (time, data_record) has zero mean
    and is free of gaps


    Theil-Sen regression references:
    Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830,
        2011.
    Theil-Sen Estimators in a Multiple Linear Regression Model, 2009 Xin Dang, Hanxiang
        Peng, Xueqin Wang and Heping Zhang http://home.olemiss.edu/~xdang/papers/MTSE.pdf

    :param ncores_to_use: number of cores to use for Theil-sen regression. *None*
    means 1 core, -1 means all cores
    :param max_siml: maximum number of simulations to do
    :param maxlag: maximum number of lags to use
    :param time: independent variable
    :param data_record: dependent variable, must not include nans
    :param sen_flag: use Theil-Sen or OLS linear regression, default Theil-Sen
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
    # Matlab xcorr returns vector of size (2 × maxlag + 1)
    # siml_acvf = np.zeros((max_siml, 2 * maxlag + 1))
    siml_acvf = np.zeros((max_siml, maxlag + 1))

    # Has shape (npts,)
    # data_mag = abs(fft(yy_filled))
    data_mag = abs(fft(data_record))

    # Initialize array for mean power spectrum
    mean_spec = np.zeros(npts)

    # Iterate through the simulations and display a progress bar
    for nsim in trange(max_siml):
        # Set the seed so that results can be reproduced, unlike the matlab code which shuffles the seed
        np.random.seed(42 + nsim)

        # Generate a 1-by-npts (1, npts) row vector of uniformly distributed
        # numbers in the interval [-2pi, 2pi)
        ph_ang = 2 * np.pi * np.random.random_sample(npts)

        # Should this be 1d or 2d from matrix multiplication?
        dummy_fft = data_mag * np.exp(1j * ph_ang)  # 1i imaginary number in matlab

        # Take inverse fft to obtain simulated time series
        dummy_ts = np.sqrt(2) * np.real(ifft(dummy_fft))

        # Calculate the mean power spectrum of each simulated time series
        # to form average
        mean_spec += (abs(fft(dummy_ts)) ** 2) / npts

        if sen_flag == 1:
            # Theil-Sen regression to find trend
            # .fit(X: training data, y: target values)
            # X must be 2D so use np.newaxis: see
            # https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py
            res = TheilSenRegressor(fit_intercept=True, random_state=42, n_jobs=ncores_to_use).fit(
                time[:, np.newaxis], dummy_ts)
            if len(res.coef_ == 1):
                siml_trends[nsim] = res.coef_[0]
            else:
                print('Warning: fit has more than 1 coefficient:', res.coef_)
                return
        elif sen_flag == 2:
            # Use Patrick's function
            # m, bb, CC = TheilSen_Cummins()
            siml_trends[nsim] = TheilSen_Cummins(np.array([time, dummy_ts]).T)[0]
        else:
            # Ordinary least-squares linear regression
            # Do not include nan values in the dataframe for the model
            dfmod = pd.DataFrame(
                {'Date': time, 'Anomaly': dummy_ts}
            )
            # create design matrices
            y, X = dmatrices('Anomaly ~ Date', data=dfmod, return_type='dataframe')

            mod = sm.OLS(y, X)  # Describe model
            res = mod.fit()  # Fit model
            siml_trends[nsim] = res.params.Date

        # As a check, compute the autocovariance for each simulated time series
        # Cannot specify normalization option as "biased" unlike in Matlab...
        # Select the biased option to check that eqn. A.4 in Appendix is satisfied????
        # Here, Size of returned array xc is nlag + 1
        xc = acovf(dummy_ts, nlag=maxlag)
        # siml_acvf[i, :] needs to be the same length as the acovf() output
        siml_acvf[nsim, :] = xc

    # Get the mean and std dev of the autocorrelation functions
    mean_acvf = np.mean(siml_acvf)
    std_acvf = np.std(siml_acvf)

    # Complete averaging to get the mean power spectrum
    mean_spec = mean_spec / max_siml

    return siml_trends, mean_acvf, std_acvf, mean_spec  # ,lags


def ols_model(anom_1d: np.ndarray, date_numeric_1d: np.ndarray):
    # https://www.statsmodels.org/stable/gettingstarted.html

    # Do not include nan values in the dataframe for the model
    dfmod = pd.DataFrame(
        {'Date': date_numeric_1d[~pd.isna(anom_1d)],
         'Anomaly': anom_1d[~pd.isna(anom_1d)]}
    )

    # create design matrices
    y, X = dmatrices('Anomaly ~ Date', data=dfmod, return_type='dataframe')

    mod = sm.OLS(y, X)  # Describe model

    res = mod.fit()  # Fit model, return regression results

    # print(res.summary())  # Summarize model
    return res


def calc_trend(search_string: str, max_siml=None, ncores_to_use=None,
               sen_flag: int = 0):
    """
    Calculate the trend using least squares and Theil-Sen regression
    and calculate the 95% confidence interval using the *max_siml* number of simulations
    :param search_string: string to use with glob to find the anomaly data files, i.e.,
        "monthly_anom_from_monthly_mean.csv", so all lightstation files would end with that
        string
    :param max_siml: maximum number of simulations to do
    :param ncores_to_use: number of cores for sen_flag=1 option processing with scikit-learn
    :param sen_flag: 0 (least-squares and Patrick Theil-Sen), 1 (least-squares and scikit-learn Theil-Sen)
    :return:
    """
    # Search for the data files in the project directory
    old_dir = os.getcwd()
    new_dir = os.path.dirname(old_dir)
    os.chdir(new_dir)
    analysis_dir = os.path.join(new_dir, 'analysis')

    file_list = glob.glob(analysis_dir + f'\\*{search_string}')
    if len(file_list) == 0:
        print('Did not locate any files in', analysis_dir, 'with suffix', search_string)
        return
    file_list.sort()

    # Number of leading and trailing nans to discard
    # abridged: Langara and Entrance Island
    nstrip_abridged = nans_to_strip(abridged=True)  # len(file_list)
    # nstrip_entire = nans_to_strip(abridged=False)

    # # Testing
    # data_file = parent_dir + 'Amphitrite_Point_monthly_anom_from_monthly_mean.csv'
    # data_file_idx = 0

    # Initialize outputs dataframe
    df_res = pd.DataFrame(columns=['Least-squares entire trend [deg C/century]',
                                   'Least-squares abridged trend [deg C/century]',
                                   'Theil-Sen entire trend [deg C/century]',
                                   'Monte Carlo confidence limit [deg C/century]',
                                   'Least-squares y-intercept'])

    # Monte Carlo parameters
    max_siml = 500 if max_siml is None else max_siml  # 50000  # test with a smaller faster number
    maxlag = 50

    # Iterate through each lighthouse station
    for i in range(len(file_list)):
        data_file = file_list[i]
        data_file_idx = i
        basename = os.path.basename(data_file)
        station_name = basename.split('_')[0] + ' ' + basename.split('_')[1]
        print(station_name)
        dframe = pd.read_csv(data_file, index_col=[0])
        # Reformat dataframe into 1d with float type date
        x, y = flatten_dframe(dframe)

        # Get series without nans but keeping data gaps
        x_gaps = x[pd.notna(y)]
        y_gaps = y[pd.notna(y)]

        # Compute trends using the abridged and entire records (Langara and Entrance)
        # WITHOUT filling gaps with interpolation or otherwise

        # Ordinary least-squares linear regression on entire record with gaps
        # Do not include nan values in the dataframe for the model
        res_ols_entire = ols_model(y_gaps, x_gaps)
        trend_century_ols_entire = res_ols_entire.params.Date * 100

        # y-intercept for OLS on entire record with gaps
        y_intercept_ols = res_ols_entire.params.Intercept
        # # Could also take the upper confidence limit assuming they're the same size
        # ci_century_ols = np.mean(abs(res.fittedvalues - res.conf_int().loc['Date', 0]))

        # Compute the trends on the gappy abridged records
        nstrip_0, nstrip_1 = [nstrip_abridged[data_file_idx, 0], nstrip_abridged[data_file_idx, 1]]

        x_gaps_abridged = x[nstrip_0:-nstrip_1][pd.notna(y[nstrip_0:-nstrip_1])]
        y_gaps_abridged = y[nstrip_0:-nstrip_1][pd.notna(y[nstrip_0:-nstrip_1])]

        res_ols_abridged = ols_model(y_gaps_abridged, x_gaps_abridged)
        trend_century_ols_abridged = res_ols_abridged.params.Date * 100

        # Compute the Theil-Sen trends on the gappy entire records
        if sen_flag == 1:
            # Use the scikit-learn Theil-Sen regression algorithm
            res_sen = TheilSenRegressor(fit_intercept=True, random_state=42).fit(
                x_gaps[:, np.newaxis], y_gaps)
            # Convert values from deg C/year to deg C/century
            trend_century_sen = res_sen.coef_[0] * 100
            # # Get y-intercept
            # y_intercept_sen = res_sen.intercept_
        else:
            # Use Patrick's Theil-Sen code (preferred)
            # Returns m, b, C, where b is the y-intercept
            trend_yearly = TheilSen_Cummins(np.array([x_gaps, y_gaps]).T)[0]
            trend_century_sen = trend_yearly * 100

        # ----------------------confidence interval------------------------

        # # Form time series with nan entries removed and retaining data gaps
        # notna_ind = pd.notna(y)
        # temp_anomaly_gaps = y[notna_ind]
        # time_gaps = x[notna_ind]
        #
        # # Calculate trend in time series with gaps using OLS
        # ls_trend_line_gaps = lstq_model(
        #     temp_anomaly_gaps, time_gaps).fittedvalues.to_numpy()

        # Strip away NaN entries at beginning and end of records to avoid extrapolation
        # Use spline interpolation to fill data gaps
        spline_fn = interp1d(x_gaps_abridged, y_gaps_abridged, kind='slinear')
        x_filled_abridged = x[nstrip_0:-nstrip_1]
        y_filled_abridged = spline_fn(x_filled_abridged)

        # Check the least-squares trend of the filled time series
        # Get results
        lstq_res = ols_model(y_filled_abridged, x_filled_abridged)
        ls_trend_line = lstq_res.fittedvalues.to_numpy()
        # Detrend the anomalies
        temp_anomaly_detrend = y_filled_abridged - ls_trend_line

        # Monte Carlo simulations to get items to compute 95% confidence interval
        siml_trends, mean_acvf, std_acvf, mean_spec = monte_carlo_trend(
            max_siml, maxlag, x_filled_abridged, temp_anomaly_detrend,
            ncores_to_use=ncores_to_use, sen_flag=sen_flag)

        # Calculate statistics of siml_trends
        # nbins = 100 if max_siml == 50000 else 5
        nbins = 100 if max_siml >= 500 else 10
        # density=True -> the result is the value of the
        # probability *density* function at the bin, normalized such that
        # the *integral* over the range is 1
        # N is an array containing the count in each bin
        # Want to bin the data using the cumulative density function (cdf) estimate
        count, bin_edges = np.histogram(siml_trends, bins=nbins)  # , density=True)
        N_input = sum(count)
        # Initialize container for CDF values
        N = np.zeros(nbins)
        for k in range(nbins):
            N[k] = sum(count[:k] / N_input)
            # for j in range(k):
            #     N[k] += count[j]/N_input

        # Establish confidence intervals
        alim_low = 0.025  # for 95 % confidence interval
        alim_high = 0.975
        for n in range(nbins - 1):
            if N[n] < alim_low <= N[n + 1]:
                temp_conf_int_low = 0.5 * (bin_edges[n] + bin_edges[n + 1])
            if N[n] <= alim_high < N[n + 1]:
                temp_conf_int_high = 0.5 * (bin_edges[n] + bin_edges[n + 1])

        temp_conf_int_95 = 0.5 * (-temp_conf_int_low + temp_conf_int_high)
        conf_int_century = temp_conf_int_95 * 100  # Convert from deg C/year to deg C/century

        df_res.loc[station_name] = [trend_century_ols_entire, trend_century_ols_abridged,
                                    trend_century_sen, conf_int_century, y_intercept_ols]

        print('OLS trend (entire record):', trend_century_ols_entire)
        print('OLS trend (abridged record):', trend_century_ols_abridged)
        print('Theil-Sen trend (entire record):', trend_century_sen)
        print('Monte Carlo 95% Conf int (abridged record):', conf_int_century)
        print('OLS y-intercept (entire record):', y_intercept_ols)

    # Save the regression output to a csv file, named accordingly
    if sen_flag == 1:
        regression_type = 'st'  # Theil-Sen
    else:
        regression_type = 'st_cummins'

    # Save to the repo directory eventually
    # Export the results to a CSV file
    df_res.to_csv('./analysis/monte_carlo_max_siml{}_ols_{}.csv'.format(max_siml, regression_type))

    # Change current dir back
    os.chdir(old_dir)

    return
