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
