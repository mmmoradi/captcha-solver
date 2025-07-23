import cv2 as cv
import numpy as np

# first way

maskpng = cv.imread("mask.png")

def filter1(img, debug=False):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_image = cv.blur(gray_img, (4, 4))
    gblur_img = cv.GaussianBlur(blur_image, (0, 0), 6)
    sharp_img = cv.addWeighted(gray_img, 1.80, gblur_img, -0.60, 0)
    sharp_not_img = cv.bitwise_not(sharp_img)
    _, img_zeroone = cv.threshold(sharp_not_img, 20, 255, cv.THRESH_BINARY)
    
    mask = cv.cvtColor(maskpng, cv.COLOR_BGR2GRAY)
    _, t_mask = cv.threshold(mask, 70, 255, cv.THRESH_BINARY)
    masker = cv.bitwise_not(t_mask)
    try:
        img_zeroone = cv.bitwise_and(img_zeroone, masker)
    except:
        pass
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    opening_img = cv.morphologyEx(img_zeroone, cv.MORPH_OPEN, kernel)
    
    out = cv.GaussianBlur(opening_img,(3,3),0)
    # out = opening_img

    out = cv.copyMakeBorder(out ,10,10,10,10,cv.BORDER_CONSTANT)

    return out
    

# second way

mymask = cv.imread("mymask.png")

def filter2(img):
        
    a = cv.bitwise_xor(img, mymask)
    b = cv.bitwise_not(a)
    c = cv.cvtColor(b, cv.COLOR_BGR2GRAY)
    d = cv.threshold(c,180,255,cv.THRESH_BINARY)[1]
    e = cv.bitwise_not(d)
    
    c = cv.adaptiveThreshold(c, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3,50)
    
    n = cv.bitwise_and(c, e)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    out = cv.morphologyEx(cv.GaussianBlur(n,(3,3),0), cv.MORPH_OPEN, kernel, )

    out = cv.copyMakeBorder(out ,10,10,10,10,cv.BORDER_CONSTANT)

    return out


def cluster(img):
    
    pix = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]:
                pix.append([i,j])
    
    npix = np.float32(pix)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    retval, labels, centers = cv.kmeans(npix, 5, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    c = centers.astype(int)
    
    # sort c basd on second cordinate
    c = np.sort(c.view('i8,i8'), order=['f1'], axis=0).view(int)
    
    
    charimg = []
        
    for i in range(5):
        charimg.append(img[c[i,0]-14:c[i,0]+14,c[i,1]-14:c[i,1]+14])

    return charimg, c


# TODO papare img

def prepare(img):
    pass