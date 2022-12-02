## bc_lighthouse_sst

Note: The project is still underway as the autocorrelation of the dataset needs to be dealt with.  
\
Make plots of BC lighthouse sea surface temperature anomalies relative to climatologies for 1991-2020. Mean climatologies for 1991-2020 are computed and subtracted from monthly mean observations from all time. The resulting anomalies are plotted.

Processing order:
1. lighthouse_sst_climatology.py
2. sst_daily_anomalies.py
3. sst_monthly_mean_anomalies.py
4. plot_lighthouse_temperature.py

*anomaly_method_differences.py* compares two ways of calculating monthly mean anomalies. One method is to subtract the climatology from daily data to get daily anomalies, then take monthly means of the daily anomalies. The other method is to subtract the climatology from monthly mean data to get the monthly mean anomalies. The second method agrees with other data collection projects by IOS so is used here.

It is necessary to account for serial correlation within the data records when estimating confidence limits around trends.

### References
Cummins, P. F. & Ross, T. (2020). Secular trends in water properties at Station P in the northeast Pacific: An updated analysis. Progress in Oceanography, *186*(2020). https://doi.org/10.1016/j.pocean.2020.102329  

Cummins, P. F. & Masson, D. (2014). Climatic variability and trends in the surface waters of coastal British Columbia. Progress in Oceanography, *120*(2014), pp. 279–290. http://dx.doi.org/10.1016/j.pocean.2013.10.002  

Garcia, H. E., T. P. Boyer, R. A. Locarnini, O. K. Baranova, M. M. Zweng (2018). World Ocean Database 2018: User’s Manual (prerelease). A.V. Mishonov, Technical Ed., NOAA, Silver Spring, MD (Available at https://www.NCEI.noaa.gov/OC5/WOD/pr_wod.html).  

