import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import cv2

def otsu_threshold(gray):
    pixel_number = gray.shape[0] * gray.shape[1] # number of pixels
    mean_weight = 1.0/pixel_number # sum of all weights
    his, bins = np.histogram(gray, np.arange(0,257)) # calculating the histogram of the image
    final_thresh = -1 # defining the best threshold calculated
    final_variance = -1 # defining the highest between class variance
    intensity_arr = np.arange(256) # creating array of all the possible pixel values (0-255)
    # Iterating through all the possible pixel values from the histogram as thresholds
    for t in bins[0:-1]:
        pcb = np.sum(his[:t]) # summing the frequency of the values before the threshold (background)
        pcf = np.sum(his[t:]) # summing the frequency of the values after the threshold (foreground)
        Wb = pcb * mean_weight # calculating the weight of the background (divide the frequencies by the sum of all weights)
        Wf = pcf * mean_weight # calculating the weight of the foreground

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb) # calculating the mean of the background (multiply the background 
        # pixel value with its weight, then divide it with the sum of frequencies of the background)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf) # calculating the mean of the foreground
        
        variance = Wb * Wf * (mub - muf) ** 2 # calculate the between class variance

        if variance > final_variance: # compare the variance in each step with the previous
            final_thresh = t
            final_variance = variance

    return final_thresh


def local_thresholding(image, t1, t2, t3, t4, mode):
    # If the image is colored, change it to grayscale, otherwise take the image as it is
    if (image.ndim == 3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif (image.ndim == 2):
        gray = image

    height, width = gray.shape # get the height and width of the image
    # In this case we will divide the image into a 2x2 grid image
    half_height = height//2 
    half_width = width//2

    # Getting the four section of the 2x2 image
    section_1 = gray[:half_height, :half_width]
    section_2 = gray[:half_height, half_width:]
    section_3 = gray[half_height:, :half_width]
    section_4 = gray[half_height:, half_width:]

    # Check if the threshold is calculated through Otsu's method or given by the user
    if (mode == 1): # calculating the threshold using Otsu's methond for each section
        t1 = otsu_threshold(section_1)
        t2 = otsu_threshold(section_2)
        t3 = otsu_threshold(section_3)
        t4 = otsu_threshold(section_4)

    # Applying the threshold of each section on its corresponding section
    section_1[section_1 > t1] = 255
    section_1[section_1 < t1] = 0

    section_2[section_2 > t2] = 255
    section_2[section_2 < t2] = 0

    section_3[section_3 > t3] = 255
    section_3[section_3 < t3] = 0

    section_4[section_4 > t4] = 255
    section_4[section_4 < t4] = 0

    # Regroup the sections to form the final image
    top_section = np.concatenate((section_1, section_2), axis = 1)
    bottom_section = np.concatenate((section_3, section_4), axis = 1)
    final_img = np.concatenate((top_section, bottom_section), axis=0)

        # final_img = gray.copy()
        # final_img[gray > t] = 255
        # final_img[gray < t] = 0

    
    plt.imshow(final_img, cmap = 'gray')
    path = "images/output/local.png"
    plt.axis("off")
    plt.savefig(path)
    return path


def global_thresholding(image, t, mode):
    # If the image is colored, change it to grayscale, otherwise take the image as it is
    if (image.ndim == 3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif (image.ndim == 2):
        gray = image

    # Check if the threshold is calculated through Otsu's method or the threshold is given by the user
    if (mode == 1): # calculating the threshold using Otsu's methond for the whole image
        t = otsu_threshold(gray)

    # Applying the threshold on the image whether it is calculated or given by the user according to the previous condition
    final_img = gray.copy()
    final_img[gray > t] = 255
    final_img[gray < t] = 0

    plt.imshow(final_img, cmap = 'gray')
    path = "images/output/global.png"
    plt.axis("off")
    plt.savefig(path)
    return path