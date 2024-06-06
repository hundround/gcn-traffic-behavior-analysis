import numpy as np
import pandas as pd

# literature data 1
hk_speed = pd.read_csv(r'C:\FILES\thesis-vsc-files\data\hk\hk_speed.csv')
hk_speed = pd.DataFrame(hk_speed)

hk_adj = pd.read_csv(r'C:\FILES\thesis-vsc-files\data\hk\hk_adj.csv', header=None)
hk_adj = pd.DataFrame(hk_adj)

hk_adj.to_numpy()
hk_speed = np.transpose(hk_speed.to_numpy())

np.save(r'C:\FILES\thesis-vsc-files\data\hk\hk_speed.npy', hk_speed)
np.save(r'C:\FILES\thesis-vsc-files\data\hk\hk_adj.npy', hk_adj)

