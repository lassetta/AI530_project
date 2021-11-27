import matplotlib.pyplot as plt 
import numpy as np



arr1 = np.loadtxt("../experiment3_cleaned/Pretrained_aug/augVGGpreacc.csv",delimiter = ',')
arr2 = np.loadtxt("../experiment3_cleaned/Pretrained_aug/augRESNETpreacc.csv",delimiter = ',')
arr3 = np.loadtxt("../experiment3_cleaned/Pretrained_aug/augATNNpreacc.csv",delimiter = ',')
arr11 = np.loadtxt("../experiment2_cleaned_v2/Pretrained_aug/augVGGpreacc.csv",delimiter = ',')
arr22 = np.loadtxt("../experiment2_cleaned_v2/Pretrained_aug/augRESNETpreacc.csv",delimiter = ',')
arr33 = np.loadtxt("../experiment2_cleaned_v2/Pretrained_aug/augATNNpreacc.csv",delimiter = ',')


fig, axs = plt.subplots(1,2)

axs[0].plot(arr1[:,0], label = 'VGG16 - original')
axs[0].plot(arr2[:,0], label = 'Resnet18 - original')
axs[0].plot(arr3[:,0], label = 'ViT - original')
axs[0].plot(arr11[:,0], label = 'VGG16 - fisheye')
axs[0].plot(arr22[:,0], label = 'Resnet18 - fisheye')
axs[0].plot(arr33[:,0], label = 'ViT - fisheye')
#axs[0].plot(arr44[:,0], color = 'orange', linestyle = '--', label = 'ViT')
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Accuracy (%)")
axs[0].set_title("Training Data")
axs[0].grid()
axs[0].legend()

axs[1].plot(arr1[:,1], label = 'VGG16 - original')
axs[1].plot(arr2[:,1], label = 'Resnet18 - original')
axs[1].plot(arr3[:,1], label = 'ViT - original')
axs[1].plot(arr11[:,1], label = 'VGG16 - fisheye')
axs[1].plot(arr22[:,1], label = 'Resnet18 - fisheye')
axs[1].plot(arr33[:,1], label = 'ViT - fisheye')
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy (%)")
axs[1].set_title("Validation Data")
axs[1].grid()
axs[1].legend()
plt.show()
