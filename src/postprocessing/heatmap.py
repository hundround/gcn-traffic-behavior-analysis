import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

# dataset
ground_data = pd.read_csv(r"C:\FILES\thesis-vsc-files\data\hk\roadnet_groundnet.csv", header=None)
pred_data = pd.read_csv(r"C:\FILES\thesis-vsc-files\data\hk\roadnet_prediction.csv", header=None)

# size = ground_data # .iloc[:int(ground_data.shape[0]), :ground_data.shape[1]]

# heatmap
fig = plt.figure(figsize=(10,8))
heatmap_ground = sns.heatmap(70*ground_data, vmin=0, vmax=80, cmap='YlOrRd', annot=False, fmt=".2f")
# heatmap_pred = sns.heatmap(70*pred_data, vmin=0, vmax=80, cmap='YlOrRd', annot=False, fmt=".2f")
heatmap_ground.invert_yaxis()

cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=20)  # Change 14 to your desired font size

plt.title('')
plt.xlabel('Road section', fontsize=22)
plt.ylabel('Time', fontsize=22)
plt.ylim(0,int(ground_data.shape[0]))
plt.xlim(0,ground_data.shape[1])
plt.xticks(np.arange(0, ground_data.shape[1], int(0.20*ground_data.shape[1]))+0.5, labels=np.arange(0, ground_data.shape[1], int(0.2*ground_data.shape[1])), rotation='horizontal', fontsize=20)
plt.yticks(np.arange(0,ground_data.shape[0],int(0.1*ground_data.shape[0]))+0.5, labels=np.arange(0,ground_data.shape[0],int(0.1*ground_data.shape[0])), fontsize=20)
plt.show()
