## bc_lightstation_sst

Note: This site is underway. All figures are provisional.

Make plots of British Columbia lightstation sea surface temperature (SST) anomalies and compute the linear trends and confidence limits. 
Compute the monthly mean anomalies by subtracting monthly mean climatologies for 1991-2020 from the monthly mean 
observations.

The data source for BC Lightstation Sea-surface Temperature and Salinity Data (Pacific), 1914-present, can be found [here](https://open.canada.ca/data/en/dataset/719955f2-bf8e-44f7-bc26-6bd623e82884).

The BC Lightstation dataset is also available on [CIOOS Pacific](https://catalogue.cioospacific.ca/dataset/ca-cioos_654a4ece-7271-4f5a-ba60-b50a62dbd051), but it is not updated as frequently and does not have any data after Nov. 2019 (as of Aug. 2023).

Processing order:
1. convert_txt_to_csv.py
   - May not be needed if csv-format data files are available
   - Convert txt data files to csv file format for easier access with pandas
1. lightstation_sst_climatology.py
   - Monthly mean climatologies for each station for the period 1991-2020
   - Does not need to be rerun if using the 1991-2020 climatologies
2. sst_monthly_mean_anomalies.py 
   - Compute monthly mean anomalies by subtracting monthly mean data from the 1991-2020 climatology
3. trend_estimation.py (run via run_trend_estimation.py)
   - Estimate the trends in SST anomalies using least-squares and Theil-Sen regression 
   - Compute confidence intervals on the least-squares trends using the nonparametric Monte Carlo approach described by Cummins & Masson (2014)
4. plot_lightstation_temperature.py
   - `plot_data_gaps()` Data availability for each station for all time
   - `plot_daily_filled_anomalies()` Data from most recent year plotted on top of average for all time for each station
   - `plot_monthly_anomalies()` Least-squares and confidence intervals of time series anomalies
   - `plot_daily_T_statistics()` Density of temperature and temperature anomalies by 30-year period (1994-2023, 1964-1993, 1934-1963, 1904-1933) with boxplots.
   
Mapping scripts:
* get_lightstation_coords_from_txt.py: Get coordinates of lightstations from txt data files
* map_lightstation_points.py: Make a Basemap plot showing the locations of each lightstation

Other scripts:
* lighthouse_sst_range_check.py: Check if all data are within the "accepted" range for coastal North Pacific sea-surface temperature, as defined by the NOAA World Ocean Database (WOD18; Garcia et al., 2018)
* anomaly_method_differences.py: Compare the monthly mean anomaly values calculated from daily observations and from monthly mean observations

<details>

<summary>Details & Background</summary>

*anomaly_method_differences.py* compares two ways of calculating monthly mean anomalies. One method is to subtract the 
climatology from daily data to get daily anomalies, then take monthly means of the daily anomalies (see 
*sst_daily_anomalies_deprec.py*). The other method is to subtract the climatology from monthly mean data to get the monthly 
mean anomalies (see *sst_monthly_mean_anomalies.py*). The second method agrees with other data collection projects by 
IOS so is used here.

It is necessary to account for **serial correlation** within the data records when estimating confidence limits around 
trends. To account for this feature, two methods are offered for calculating confidence limits. The first is described 
by Thomson & Emery (2014, pp. 272-275) and assumes that the number of degrees of freedom for the t-distribution are 
given by the effective number of degrees of freedom, ν=N*-2, where N* (<N) is the effective sample size. N* is 
calculated from the integral timescale T for the data record, where T in turn depends on the autocovariance function. 
ν is used to calculate the confidence limits on the trend (e.g., using the least-squares formula for confidence limits). 
This method will be referenced as the "effective sample size" method.

The second method is a Monte Carlo approach used by Cummins & Masson (2014). This is better to use if the 
autocorrelation structure is not approximated well by a first-order autoregressive process (AR-1) process. The anomaly 
data is detrended by subtracting the ordinary least squares trend from it. Then 50,000 random time series are generated
having the same autocorrelation structure as the data record using a discrete inverse Fourier transform followed by 
a discrete Fourier transform. The trend of each is estimated with Theil-Sen regression. The 95% confidence interval on 
the trend of the true time series is then taken as the 95% confidence interval on the set of trends of the random time 
series. The functions in *trend_estimation.py* used for this method were translated from MatLab scripts written by 
Patrick Cummins.

</details>

### References
Cummins, P. F. & Ross, T. (2020). Secular trends in water properties at Station P in the northeast Pacific: An updated analysis. Progress in Oceanography, *186*(2020). https://doi.org/10.1016/j.pocean.2020.102329  

Cummins, P. F. & Masson, D. (2014). Climatic variability and trends in the surface waters of coastal British Columbia. Progress in Oceanography, *120*(2014), pp. 279–290. http://dx.doi.org/10.1016/j.pocean.2013.10.002  

Garcia, H. E., T. P. Boyer, R. A. Locarnini, O. K. Baranova, M. M. Zweng (2018). World Ocean Database 2018: User’s Manual (prerelease). A.V. Mishonov, Technical Ed., NOAA, Silver Spring, MD (Available at https://www.NCEI.noaa.gov/OC5/WOD/pr_wod.html).  

Thomson, R. E., & Emery, W. J. (2014). *Data Analysis Methods in Physical Oceanography: Second and Revised Edition (3rd ed.)*. Elsevier Science.

