import numpy as np

def spectral_threshold(image,threshold_num,thresholds):
    # sorting the threshold ascendingly to prevent any error in thresholding
    thresholds = np.sort(thresholds)
    # number of intervals between the thresholds
    intervals = threshold_num - 1
    # define the grayscale gradient corresponding to the number of thresholds
    step = float(1/threshold_num)
    grayscale_gradient = np.arange(0, (1+step), step)
    grayscale_gradient = grayscale_gradient[grayscale_gradient<=1]

    new_image = np.where( (image > 0 ) & (image < (thresholds[0]/255)), grayscale_gradient[0], 0)
    for i in np.arange(threshold_num-1):
        mask = np.where(((image > (thresholds[i]/255)) & (image < (thresholds[i+1]/255))), grayscale_gradient[i+1], 0)
        new_image += mask
    new_image += np.where((image > (thresholds[-1]/255)), grayscale_gradient[-1], 0)
    
    return new_image