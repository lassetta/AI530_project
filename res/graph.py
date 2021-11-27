import matplotlib.pyplot as plt 
import numpy as np


arr1 = np.loadtxt("../experiment2_cleaned_v2/CNN_aug/augacc.csv",delimiter = ',')
arr2 = np.loadtxt("../experiment2_cleaned_v2/Res_aug/augacc.csv",delimiter = ',')
arr3 = np.loadtxt("../experiment2_cleaned_v2/MLP_aug/augacc.csv",delimiter = ',')
arr4 = np.loadtxt("../experiment2_cleaned_v2/ATTN_aug/augacc.csv",delimiter = ',')


arr11 = np.loadtxt("../experiment3_cleaned/CNN_aug/augacc.csv",delimiter = ',')
arr22 = np.loadtxt("../experiment3_cleaned/Res_aug/augacc.csv",delimiter = ',')
arr33 = np.loadtxt("../experiment3_cleaned/MLP_aug/augacc.csv",delimiter = ',')
arr44 = np.loadtxt("../experiment3_cleaned/ATTN_aug/augacc.csv",delimiter = ',')
'''
arr1 = np.loadtxt("../experiment2_cleaned_v2/CNN_aug/no_augacc.csv",delimiter = ',')
arr2 = np.loadtxt("../experiment2_cleaned_v2/Res_aug/no_augacc.csv",delimiter = ',')
arr3 = np.loadtxt("../experiment2_cleaned_v2/MLP_aug/no_augacc.csv",delimiter = ',')
arr4 = np.loadtxt("../experiment2_cleaned_v2/ATTN_aug/no_augacc.csv",delimiter = ',')


arr11 = np.loadtxt("../experiment3_cleaned/CNN_aug/no_augacc.csv",delimiter = ',')
arr22 = np.loadtxt("../experiment3_cleaned/Res_aug/no_augacc.csv",delimiter = ',')
arr33 = np.loadtxt("../experiment3_cleaned/MLP_aug/no_augacc.csv",delimiter = ',')
arr44 = np.loadtxt("../experiment3_cleaned/ATTN_aug/no_augacc.csv",delimiter = ',')
'''


plt.figure(1)

plt.title("Network Comparisons - Training Curves")
plt.plot(arr1[:,0], label = 'CNN - fisheye')
plt.plot(arr11[:,0], label = 'CNN - baseline')
plt.plot(arr2[:,0], label = 'Resnet - fisheye')
plt.plot(arr22[:,0], label = 'Resnet - baseline')
plt.plot(arr3[:,0], label = 'MLP - fisheye')
plt.plot(arr33[:,0], label = 'MLP - baseline')
plt.plot(arr4[:,0], label = 'ViT - fisheye')
plt.plot(arr44[:,0], label = 'ViT - baseline')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.legend()
plt.show()
