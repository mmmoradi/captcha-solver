import tensorflow as tf
import keras

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split

# Global
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
from preprocessing import *

label_dictionary = {}

for index, label in np.loadtxt("dataset/kaggleemnist/emnist-balanced-mapping.txt", delimiter=" ", dtype=np.int64):
    label_dictionary[index] = chr(label)



mymodel = keras.models.load_model('checkpoint2.model.keras')

def predict(number):
    
    text = []
    
    for i in range(number):
    
        img = cv.imread(f"dataset/captchas/{i+1:05}.gif")
    
        c,_ = cluster(filter1((img)))
    
        captcha = []
        
        for j in range(len(c)):
            
            inputimg = cv.copyMakeBorder(cv.resize(c[j], (24,24)) ,2,2,2,2,cv.BORDER_CONSTANT).T / 255.0
            
            prob = mymodel.predict(inputimg.reshape((1,28,28,1)), verbose=0)
            
            captcha.append(label_dictionary[np.argmax(prob)])
        
        text.append((i+1,f"{''.join(captcha)}"))
        
        del img, c, captcha

    return text

predict(1000)