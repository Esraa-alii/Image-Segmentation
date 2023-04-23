import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt




def mean_shift_method(Input_Img,Num_of_iteration):
    
    img = cv.imread(Input_Img)
    # filter to reduce noise
    img = cv.medianBlur(img, 3)

    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    ms = MeanShift(bandwidth=bandwidth, max_iter=Num_of_iteration,  bin_seeding=True)
    ms.fit(flat_image)
    labeled=ms.labels_


    # get number of segments
    segments = np.unique(labeled)


    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))
    plt.imshow(result)
    plt.axis("off")
    plt.savefig('./images/output/mean_shift_out.png')


def Optimized_Thresholding(Input_Img,option):
    img = cv.imread(Input_Img, cv.IMREAD_GRAYSCALE)
    img = cv.medianBlur(img,5)
    if option == 'Global' :
        ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
        plt.imshow(th1)
        plt.axis("off")
        plt.savefig('./images/output/optimal_out1.png')
        return './images/output/optimal_out1.png'
    if option == 'Local' :
        th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
        plt.imshow(th2)
        plt.axis("off")
        plt.savefig('./images/output/optimal_out2.png')
        return './images/output/optimal_out2.png'
        


    
    