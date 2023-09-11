from scripts.trend_estimation import calc_trend
# import os

# # Patrick

# Working directory
# wdir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
#        'lighthouse_data\\update_20230706\\monthly_anom_from_monthly_mean\\'

# calc_trend(wdir, max_siml=50000, ncores_to_use=None, sen_flag=0)
calc_trend(search_string="monthly_anom_from_monthly_mean.csv",
           max_siml=50000, ncores_to_use=None, sen_flag=0)

"""
# Testing
# x = np.random.random_sample(1000)
xx = np.random.normal(size=1000)
num_bins = 25
counts, edges = np.histogram(xx, num_bins)
cdf = np.zeros(num_bins)
for b in range(num_bins):
    cdf[b] = sum(counts[:b]/len(xx))

plt.bar(x=np.arange(num_bins), height=cdf)

plt.hist(xx, num_bins)


# test
test_data = np.array([xx, yy_filled]).T
[m, b, C] = TheilSen_Cummins(test_data)
"""
