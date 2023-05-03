import numpy as np
import cv2

def RegionGrowing(image_path, seed):
    image=cv2.imread(image_path)

    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    seed_x_point, seed_y_point = seed
    threshold = image[seed_x_point, seed_y_point]
    queue = []
    queue.append((seed_x_point, seed_y_point))

    
    while queue:
       
        seed_x_point, seed_y_point = queue.pop(0)
        if seed_x_point < 0 or seed_y_point < 0 or seed_x_point >= height or seed_y_point >= width:
            continue
        
        if mask[seed_x_point, seed_y_point]:
            continue

        if abs(image[seed_x_point, seed_y_point] - threshold).any() <= 10:
           
            mask[seed_x_point, seed_y_point] = 255
            queue.append((seed_x_point - 1, seed_y_point))
            queue.append((seed_x_point + 1, seed_y_point))
            queue.append((seed_x_point, seed_y_point - 1))
            queue.append((seed_x_point, seed_y_point + 1))
    
    return mask