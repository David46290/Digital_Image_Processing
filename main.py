import cv2
import random
import image_processing as imgp
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('is_this_a_pigeon.jpg') 
    # img = cv2.imread('gundam_rg_2.jpg')
    # img = cv2.imread('zowort_heavy.jpg')
    # img = cv2.imread('destiny.jpg')
    # img = cv2.imread('.//AI_test//gundam_1.jpg')
    img = img / 255
    
    img_ds = imgp.pooling(img, shrinkage=6)
    # img_ds = imgp.resize(img, (450, 800))
    # img_ds = resize(img, (450, 600))
    # img_ds = resize(img, [img.shape[1], img.shape[0]])

    # histogram(img_ds, channel_order='BGR')
    # img_fft = sfft.fft2(img_ds)
    # img_fft_2, img_filtered = pass_filter(img_ds, span=0.9, pass_type='low')
    # img_filtered2 = fft_kernel_conv(img_ds, std=3)
    
    # img_ds = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
    #                 [7, 6, 5, 4, 3, 2, 1, 1],
    #                 [5, 9, 7, 2, 3, 4, 1, 0],
    #                 [1, 3, 5, 7, 9, 7, 5, 3],
    #                 [1, 1, 8, 1, 8, 2, 3, 4],
    #                 [5, 9, 7, 2, 3, 4, 1, 0]]).reshape(6, 8, 1)
    # img_ds = np.concatenate((img_ds, img_ds, img_ds), axis=2)
    
    img_processed = np.copy(img_ds)
    img_processed2 = np.copy(img_ds)
    # img_noised = np.copy(img_ds)
    
    # DCT(img_processed2, shape_result=None, norm='ortho', process_type='pass', pass_quadrant='1', pass_rate=0.5)
    imgp.pass_filter(img_processed, span=0.8, pass_type='low')
    imgp.pass_filter(img_processed2, span=0.8, pass_type='high')
    
    
    # salt_pepper(img_noised)
    # img_processed = np.copy(img_noised)
    # edge_detect_laplace(img_processed, direction='vert_hori')
    # edge_detect_laplace(img_processed, direction='45_degree')
    # log_transform(img_processed, c=2)
    # log_transform(img_processed, c=20, normalized=True)
    # contrast(img_processed2, mode='cubic', value_contrast=7, step_threshold=0.2)
    # gaussian_filter(img_processed, sigma=7, radius=10, order=0)
    # gaussian_filter(img_processed2, sigma=7, radius=10, order=0)
    
    # median_filter(img_processed, window_size=3, step_size=1)
    # negatives(img_processed2)
    # log_transform(img_processed2, c=2)
    # edge_detect_laplace(img_processed2, direction='45_degree')
    # histogram(img_processed, channel_order='BGR')    
    # img_ds = np.clip(img_ds + img_processed2, 0, 1)
    # img_processed = np.clip(img_ds - img_processed, 0, 1)
    # img_processed2 = np.clip(img_ds - img_processed2, 0, 1)
    # img_compressed = DCT_compress(img_ds, shrinkage=0.5)
    
    cv2.imshow('image',img)
    cv2.imshow('pooled', img_ds)
    # cv2.imshow('noised', img_noised)
    # cv2.imshow('1', img_processed)
    # cv2.imshow('2', img_processed2)
    # cv2.imshow('log&norm - og', (img_processed2 - img_ds))
    # cv2.imshow('compressed', img_compressed)
    # cv2.imshow('mixed', img_mixed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('processed1.jpg', img_processed*255)
    # cv2.imwrite('processed2.jpg', img_processed2*255)
    