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


fig, axs = plt.subplots(1,2,figsize = (15,5))

axs[0].plot(arr1[:,0], label = 'CNN - fisheye')
axs[0].plot(arr11[:,0], label = 'CNN - baseline')
axs[0].plot(arr2[:,0], label = 'Resnet - fisheye')
axs[0].plot(arr22[:,0], label = 'Resnet - baseline')
axs[0].plot(arr3[:,0], label = 'MLP - fisheye')
axs[0].plot(arr33[:,0], label = 'MLP - baseline')
axs[0].plot(arr4[:,0], label = 'ViT - fisheye')
axs[0].plot(arr44[:,0], label = 'ViT - baseline')
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Accuracy (%)")
axs[0].set_ylim(40,98)
axs[0].grid()
axs[0].legend(loc = "lower right")

axs[1].plot(arr1[:,1], label = 'CNN - fisheye')
axs[1].plot(arr11[:,1], label = 'CNN - baseline')
axs[1].plot(arr2[:,1], label = 'Resnet - fisheye')
axs[1].plot(arr22[:,1], label = 'Resnet - baseline')
axs[1].plot(arr3[:,1], label = 'MLP - fisheye')
axs[1].plot(arr33[:,1], label = 'MLP - baseline')
axs[1].plot(arr4[:,1], label = 'ViT - fisheye')
axs[1].plot(arr44[:,1], label = 'ViT - baseline')
axs[1].set_ylim(40,98)
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy (%)")
axs[1].grid()
#axs[1].legend()
axs[1].legend(loc = "lower right")
plt.savefig("experiments12_res.png", bbox_inches = "tight")
