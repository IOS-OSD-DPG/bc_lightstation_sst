import pandas as pd
import glob

wdir = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
       'lighthouse_data\\update_20230706\\raw_data\\'

files = glob.glob(wdir + '*.txt')
files.sort()

outdf = pd.DataFrame()

for f in files:
    indf = pd.read_fwf(f, skiprows=2, nrows=1, header=None)
    print(indf)

coords_txt = 'C:\\Users\\HourstonH\\Documents\\charles\\our_warming_ocean\\' \
             'lighthouse_data\\update_20230706\\coordinates.txt'

txt_df = pd.read_fwf(coords_txt, header=None)