#image_filters.py
import numpy as np
import cv2 as ocv
import math

class ImageFilters:
    @staticmethod
    def decrease_brightness(img):
        height, width = np.shape(img)
        filter_result = np.copy(img) / 255
        for row in range(height):
            for col in range(width):
                filter_result[row][col] = img[row][col] * 0.5
                
        return filter_result

    @staticmethod
    def increase_brightness(img):
        height, width = np.shape(img)
        # Normalize the image to [0, 1]
        filter_result = np.copy(img) / 255.0
        
        for row in range(height):
            for col in range(width):
                filter_result[row][col] = filter_result[row][col] * 2
                if filter_result[row][col] > 1:
                    filter_result[row][col] = 1
        
        return (filter_result * 255).astype(np.uint8)
    @staticmethod
    def negative_filter(img):
        return 255 - img

    @staticmethod
    def power_law_filter(img):
        height, width = np.shape(img)
        filter_result = np.copy(img) / 255
        for row in range(height):
            for col in range(width):
                filter_result[row][col] = 1 * (filter_result[row][col]) ** 2
        return (filter_result * 255).astype(np.uint8)

    @staticmethod
    def log_filter(img):
        filter_result = np.copy(img)
        h, w = np.shape(img)
        for row in range(h):
            for col in range(w):
                # Ensure the pixel value is greater than 0 to avoid log(0)
                if img[row][col] > 0: # To avoid log(0) 
                    filter_result[row][col] = 30 * (np.log2(img[row][col]))
                else:
                    filter_result[row][col] = 0
        return filter_result

    @staticmethod
    def inverse_log_filter(img):
        filter_result = np.copy(img)
        h, w = np.shape(img)
        for row in range(h):
            for col in range(w):
                # Apply the inverse log transformation (2^x)
                if img[row][col] > 0:  # Avoid taking the log of zero
                    filter_result[row][col] = np.power(2, img[row][col] / 30)
                else:
                    filter_result[row][col] = 0
        return filter_result

    @staticmethod
    def sobel_filter(img):
        filter_result = img.copy() / 255.0

        Gx = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]

        Gy = [[-1, -2, -1],
            [0,   0,  0],
            [1,   2,  1]]

        width , height = np.shape(img)

        filter_size = 3

        offset = filter_size // 2


        for row in range(offset, width - offset):
            for col in range(offset, height - offset):
                area = img[row - offset: row + offset + 1, col - offset : col + offset + 1]
                Gx2 = np.sum(np.multiply(Gx, area)) ** 2
                Gy2 = np.sum(np.multiply(Gy, area)) ** 2
                filter_result[row][col] = np.sqrt(Gx2 + Gy2)

        return filter_result

    @staticmethod
    def normalize_image(img):
        height, width = np.shape(img)
        filter_result = np.zeros((height, width))
        I_min = np.min(img)
        I_max = np.max(img)
        L = 256
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                filter_result[row][col] = ((img[row][col] - I_min) / (I_max - I_min)) * (L - 1)
        return filter_result.astype(np.uint8)

    @staticmethod
    def gaussian_filter(img):
        height, width = np.shape(img)

        filter_result = np.copy(img)

        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16.0

        for row in range(1, height - 1):
            for col in range(1, width - 1):
                filter_result[row][col] = np.sum([
                    img[row-1][col-1] * gaussian_kernel[0][0], img[row-1][col] * gaussian_kernel[0][1], img[row-1][col+1] * gaussian_kernel[0][2],
                    img[row][col-1] * gaussian_kernel[1][0], img[row][col] * gaussian_kernel[1][1], img[row][col+1] * gaussian_kernel[1][2],
                    img[row+1][col-1] * gaussian_kernel[2][0], img[row+1][col] * gaussian_kernel[2][1], img[row+1][col+1] * gaussian_kernel[2][2]
                ])
        return filter_result

    @staticmethod
    def histogram_filter(img):
        histogram = [0] * 256
        height, width = img.shape
        for row in range(height):
            for col in range(width):
                histogram[img[row, col]] += 1
        hist_img = np.zeros((300, 300), dtype=np.uint8)
        max_val = max(histogram)
        hist_normalized = [int((value / max_val) * 300) for value in histogram]
        for x in range(256):
            for y in range(hist_normalized[x]):
                hist_img[299 - y, x] = 255
        return hist_img

    @staticmethod
    def histogram_equalization(img):
        # Get img dimensions
        height, width = img.shape
        total_pixels = height * width
        
        # Calculate histogram (frequency of each intensity level)
        histogram = np.zeros(256)
        for i in range(height):
            for j in range(width):
                intensity = img[i, j]
                histogram[intensity] += 1
        
        pdf = histogram / total_pixels
        
        cdf = np.zeros(256)
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + pdf[i]
        
        cdf_normalized = np.round((cdf) * 255)
        
        equalized_image = np.zeros_like(img)
        for i in range(height):
            for j in range(width):
                original_intensity = img[i, j]
                equalized_image[i, j] = cdf_normalized[original_intensity]
                
        return equalized_image.astype(np.uint8)
    @staticmethod
    def histogram_matching(source_image, target_image):
        flat_src = source_image.flatten()
        flat_target = target_image.flatten()
        L = 256
        src_counts = np.bincount(flat_src, minlength=L)
        target_counts = np.bincount(flat_target, minlength=L)
        src_CDF = np.cumsum(src_counts) / flat_src.size
        target_CDF = np.cumsum(target_counts) / flat_target.size
        map_to_target = np.zeros(L, dtype=np.uint8)
        for i in range(L):
            closest_idx = np.argmin(np.abs(target_CDF - src_CDF[i]))
            map_to_target[i] = closest_idx
        return map_to_target[flat_src].reshape(source_image.shape)

    @staticmethod
    def prewitt_filter(img):
        filter_result = img.copy()

        Gx = [[-1, -1, -1],
            [0,0,0],
            [1,1,1]]

        Gy = [[-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]]

        width , height = np.shape(img)

        filter_size = 3

        offset = filter_size // 2


        for row in range(offset, width - offset):
            for col in range(offset, height - offset):
                area = img[row - offset: row + offset + 1, col - offset : col + offset + 1]
                Gx2 = np.sum(np.multiply(Gx, area)) ** 2
                Gy2 = np.sum(np.multiply(Gy, area)) ** 2
                filter_result[row][col] = np.sqrt(Gx2 + Gy2)
        return filter_result

    @staticmethod
    def average_filter(img):
        filter_result = np.copy(img) / 255.0
        width, height = np.shape(img)
        for row in range(1, width -1, 1):
            for col in range(1, height -1, 1):
                filter_result[row][col] = (img[row][col]) * (1/9) + (img[row-1][col] * (1/9)) +\
                                (img[row-1][col+1] * (1/9)) + (img[row+1][col] * (1/9)) +\
                                (img[row+1][col+1] * (1/9)) + (img[row+1][col-1] * (1/9)) +\
                                ((img[row-1][col-1] * (1/9))) + (img[row][col+1] * (1/9)) +\
                                (img[row][col-1] * (1/9))  # hard coding
        return filter_result

    @staticmethod
    def max_filter(img):
        filter_result = np.copy(img) / 255.0
        width, height = np.shape(img)
        for row in range(1, width -1, 1):
            for col in range(1, height -1, 1):
                list = [img[row][col] , img[row-1][col] , 
                        img[row-1][col+1] , img[row+1][col] ,
                        img[row+1][col+1] , img[row+1][col+-1] , 
                        img[row-1][col-1] , img[row][col+1] , img[row][col-1]]
                filter_result[row][col] = np.max(list)
        return filter_result

    @staticmethod
    def min_filter(img):
        filter_result = np.copy(img) / 255.0
        width, height = np.shape(img)
        for row in range(1, width -1, 1):
            for col in range(1, height -1, 1):
                list = [img[row][col] , img[row-1][col] , 
                        img[row-1][col+1] , img[row+1][col] ,
                        img[row+1][col+1] , img[row+1][col+-1] , 
                        img[row-1][col-1] , img[row][col+1] , img[row][col-1]]
                filter_result[row][col] = np.min(list)
        return filter_result
    
    @staticmethod
    def median_filter(img):
        filter_result = np.copy(img) / 255.0
        width, height = np.shape(img)
        for row in range(1, width -1, 1):
            for col in range(1, height -1, 1):
                list = [img[row][col] , img[row-1][col] , 
                        img[row-1][col+1] , img[row+1][col] ,
                        img[row+1][col+1] , img[row+1][col+-1] , 
                        img[row-1][col-1] , img[row][col+1] , img[row][col-1]]
                filter_result[row][col] = np.median(list)
        return filter_result