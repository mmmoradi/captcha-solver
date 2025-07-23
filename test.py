import os
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


dataset_dir = "dataset/captchas"

existing_files = [(f,int(f[:5]),f[6:11]) for f in os.listdir(dataset_dir) if re.match(r"(\d{5})-([A-Za-z0-9]+)\.gif", f)]

for file, number, code in existing_files:

    print(f'#{number} filename {file} code {code}')

