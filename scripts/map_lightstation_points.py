from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

old_wd = os.getcwd()
new_wd = os.path.dirname(old_wd)
os.chdir(new_wd)

point_file = '.\\data\\coordinates.csv'

df = pd.read_csv(point_file, index_col=[0])

# Add column containing abbreviation of each station name
df['Name_abbrev'] = [n[0] + n.split(' ')[1][0] for n in df.index]

# Initialize plot
fig, ax = plt.subplots()

# Set up basemap
left_lon, bot_lat, right_lon, top_lat = [-133, 48, -120, 55]

m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
            urcrnrlon=right_lon, urcrnrlat=top_lat,
            projection='lcc',
            resolution='h', lat_0=0.5 * (bot_lat + top_lat),
            lon_0=0.5 * (left_lon + right_lon))

m.drawcoastlines(linewidth=0.2)
m.drawparallels(np.arange(bot_lat, top_lat, 1), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(left_lon, right_lon, 2), labels=[0, 0, 0, 1])
# m.drawparallels(np.arange(bot_lat, top_lat, 0.3), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(left_lon, right_lon, 1), labels=[0, 0, 0, 1])
# m.drawmapboundary(fill_color='white')
m.fillcontinents(color='0.8')

# Plot the lightstation locations
x, y = m(df.loc[:, 'Longitude'].to_numpy(), df.loc[:, 'Latitude'].to_numpy())
# Use zorder to plot the points on top of the continents
m.scatter(x, y, marker='o', color='r', s=10, zorder=5)

# Label the points with the station name
# https://stackoverflow.com/questions/59740782/labelling-multiple-points-from-csv-in-basemap
pad = ' '
for i in range(len(df.index)):
    if df.index[i] == 'Departure Bay':
        # Flip the text so the data point is on its right instead of its left
        ax.annotate(df.index[i], (x[i], y[i]), xytext=(-64, -37), textcoords='offset points',
                    fontsize=8, rotation=30)
        pass
    elif df.index[i] == 'Langara Island':
        ax.annotate('Langara\nIsland', (x[i], y[i]), fontsize=8, rotation=30)
    else:
        ax.annotate(pad + df.index[i], (x[i], y[i]), fontsize=8, rotation=30)

# Plot formatting
plot_name = '.\\figures\\lightstation_map.png'
plt.tight_layout()
plt.savefig(plot_name, dpi=300)  # Save at lower quality than dpi=400
plt.close(fig)

# Reset the working directory
os.chdir(old_wd)
