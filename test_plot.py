import pandas as pd
from mup.coord_check import get_coord_data, plot_coord_data
import numpy as np

df = pd.read_pickle('/home/berlin/mup/coord_checks/df_pickle_runatest1.pkl')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
filename = '/home/berlin/mup/coord_checks/ssm_mu1_runatest1.png'
plot_coord_data(df.dropna(), save_to=filename)