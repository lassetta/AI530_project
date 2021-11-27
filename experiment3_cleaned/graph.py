import matplotlib.pyplot as plt 
import numpy as np


arr1 = np.loadtxt("CNN_aug/augacc.csv",delimiter = ',')
arr2 = np.loadtxt("Res_aug/augacc.csv",delimiter = ',')
arr3 = np.loadtxt("MLP_aug/augacc.csv",delimiter = ',')
arr4 = np.loadtxt("ATTN_aug/augacc.csv",delimiter = ',')


arr11 = np.loadtxt("CNN_aug/no_augacc.csv",delimiter = ',')
arr22 = np.loadtxt("Res_aug/no_augacc.csv",delimiter = ',')
arr33 = np.loadtxt("MLP_aug/no_augacc.csv",delimiter = ',')
arr44 = np.loadtxt("ATTN_aug/no_augacc.csv",delimiter = ',')


fig, axs = plt.subplots(2)

axs[0].plot(arr1[:,0], color = 'b', label = 'CNN - augmented')
axs[0].plot(arr2[:,0], color = 'c', label = 'Resnet - augmented')
axs[0].plot(arr3[:,0], color = 'r', label = 'MLP - augmented')
axs[0].plot(arr4[:,0], color = 'orange', label = 'ViT - augmented')
axs[0].plot(arr11[:,0], color = 'b', linestyle = '--', label = 'CNN')
axs[0].plot(arr22[:,0], color = 'c', linestyle = '--', label = 'Resnet')
axs[0].plot(arr33[:,0], color = 'r', linestyle = '--',label = 'MLP')
axs[0].plot(arr44[:,0], color = 'orange', linestyle = '--', label = 'ViT')
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Training Accuracy (%)")
axs[0].grid()
axs[0].legend()

axs[1].plot(arr1[:,1], color = 'b', label = 'CNN - augmented')
axs[1].plot(arr2[:,1], color = 'c', label = 'Resnet - augmented')
axs[1].plot(arr3[:,1], color = 'r', label = 'MLP - augmented')
axs[1].plot(arr4[:,1], color = 'orange', label = 'ViT - augmented')
axs[1].plot(arr11[:,1], color = 'b', linestyle = '--', label = 'CNN')
axs[1].plot(arr22[:,1], color = 'c', linestyle = '--', label = 'Resnet')
axs[1].plot(arr33[:,1], color = 'r', linestyle = '--',label = 'MLP')
axs[1].plot(arr44[:,1], color = 'orange', linestyle = '--', label = 'ViT')
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Validation Accuracy (%)")
axs[1].grid()
axs[1].legend()
plt.show()
