# Global
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

fig,axes = plt.subplots(100,100,figsize=(100,100))

for i,ax in enumerate(axes.flat):

    img = cv.imread(f"dataset/captchas/{i+1:05}.gif")

    try:
        ax.imshow(img, vmin=0, vmax=255)
        # ax.set_title(f"{i+1}")
    except:
        print(f"rainsed and exeption in {i+1}")

    ax.axis('off')
    del img

fig.savefig('testimg')
