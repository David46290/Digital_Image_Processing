import cv2
import random
import scipy.fftpack as sfft
import scipy.ndimage as simg
import numpy as np
from scipy import signal as scisig
from matplotlib import pyplot as plt


def pooling(image, shrinkage):
    diemnsion = image.shape[2]
    h = image.shape[0]
    w = image.shape[1]
    h_pooled = h // shrinkage
    w_pooled = w // shrinkage
    new_image = np.zeros((h_pooled, w_pooled, diemnsion))
    
    for idx_c in range(diemnsion):
        idx_w = 0
        idx_w_pooled = 0
        while (idx_w_pooled<w_pooled):
            idx_h = 0
            idx_h_pooled = 0
            while (idx_h_pooled<h_pooled):
                # print(f'idx_h: {idx_h}| idx_w: {idx_w}| idx_h_pooled: {idx_h_pooled}| idx_w_pooled: {idx_w_pooled}| ')
                box_h = [idx_h, idx_h+(shrinkage-1)]
                box_w = [idx_w, idx_w+(shrinkage-1)]
                new_image[idx_h_pooled, idx_w_pooled, idx_c] = np.amax(image[idx_h:idx_h+(shrinkage-1), idx_w:idx_w+(shrinkage-1), idx_c])
                idx_h = idx_h + shrinkage
                idx_h_pooled = idx_h_pooled + 1
            idx_w = idx_w + shrinkage
            idx_w_pooled = idx_w_pooled + 1
            
    return new_image            
        
def pass_filter(image, span=0.1, pass_type='low'):
    diemnsion = image.shape[2]
    h = image.shape[0]
    w = image.shape[1]
    
    h_center = h // 2
    w_center = w // 2
       
    
    image_fft = sfft.fft2(image) # 0 freqeucny component locates at to-left corner
    threshold_h = [int(h_center-(0.5*h)*(span)), int(h_center+(0.5*h)*(span))]
    threshold_w = [int(w_center-(0.5*w)*(span)), int(w_center+(0.5*w)*(span))]
    
    if pass_type == 'high':
        image_fft[0:threshold_h[0], :, :] = 0
        image_fft[threshold_h[1]:, :, :] = 0
        image_fft[:, 0:threshold_w[0], :] = 0
        image_fft[:, threshold_w[1]:, :] = 0
    elif pass_type == 'low':
        image_fft[threshold_h[0]:threshold_h[1], threshold_w[0]:threshold_w[1], :] = 0
    else:
        print('no assigned pass type for freq-pass')
    image[:, :, :] = np.abs(np.fft.ifft2(image_fft))
    

def fft_kernel_conv(image, std=1):
    kernel = np.outer(scisig.gaussian(image.shape[0], std), scisig.gaussian(image.shape[1], std))
    freq_kernel = sfft.fft2(sfft.ifftshift(kernel))
    image_fft = sfft.fft2(image)
    for idx_c in range(image.shape[2]):
        image_fft[:, :, idx_c] = image_fft[:, :, idx_c] * freq_kernel
    return sfft.ifft2(image_fft).real

def histogram(image, channel_order='RGB', bins=50):
    #  be sure the channel order is R->G->B
    dimension = image.shape[2]
    value_boundary = [0, 256]
    channel_map = {'R':'maroon', 'G':'seagreen', 'B':'royalblue'}
    channel = []
    color = []
    
    for channel_code in channel_order:
        channel.append(channel_code)
        color.append(channel_map[channel_code])
    
    x_tick = np.arange(value_boundary[0], value_boundary[1], 15)
    for c_idx in range(dimension):
        plt.figure(figsize=(8, 5), dpi=300) 
        counts, bins = np.histogram((image[:, :, c_idx].ravel()*255).astype(int), bins=bins)
        plt.hist(bins[:-1], bins, weights=counts, color=color[c_idx])
        plt.yscale('log')
        plt.margins(x=0.02)
        plt.title(f'{channel[c_idx]}', fontsize=16)
        plt.grid()
        plt.ylabel('Amount', fontsize=14)
        plt.xlabel('Pixel Value', fontsize=14)
        plt.xticks(x_tick, fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        
def negatives(image):
    image[:, :, :] = np.clip(-1 * (image - 1), 0, 1)

    
def log_transform(image, c):
    for c_idx in range(image.shape[2]):
        image[:, :, c_idx] = np.clip(c * np.log(1 + image[:, :, c_idx]), 0, 1)


def DCT(image, shape_result=None, norm='ortho', process_type='random', pass_rate=0.5, pass_quadrant='1234'):
    image_dct = sfft.dctn(image, type=2, shape=shape_result, norm=norm)
    
    if process_type == 'random':
        # randomly edit element and see what can be changed
        image_dct = image_dct * np.random.rand(image.shape[0], image.shape[1], image.shape[2])
    elif process_type == 'pass':
        #[1, 2]
        #[3, 4]
        threshold_h = int(image.shape[0] * pass_rate)
        threshold_w = int(image.shape[1] * pass_rate)
        if '1' not in pass_quadrant:
            image_dct[:threshold_h, :threshold_w, :] = 0
        if '2' not in pass_quadrant:
            image_dct[:threshold_h, threshold_w:image.shape[1], :] = 0
        if '3' not in pass_quadrant:
            image_dct[threshold_h:image.shape[0], :threshold_w, :] = 0
        if '4' not in pass_quadrant:
            image_dct[threshold_h:, threshold_w:, :] = 0
    image[:, :, :] = sfft.idctn(image_dct, type=2, shape=shape_result, norm=norm)

def edge_detect_laplace(image, direction='vert_hori'):
    if direction == 'vert_hori':
        filter_ = np.array([[0, 1, 0], 
                            [1, -4, 1], 
                            [0, 1, 0]])
    elif direction == '45_degree':
        filter_ = np.array([[1, 0, 1], 
                            [0,-4, 0], 
                            [1, 0, 1]])
    for c_idx in range(image.shape[2]):
        image[:, :, c_idx] = np.clip(scisig.convolve2d(image[:, :, c_idx], filter_, mode='same'), 0, 1)

def median_filter(image, window_size=3, step_size=1):
    image_new = np.copy(image)
    for idx_c in range(image.shape[2]):
        idx_h = 0
        while idx_h < image.shape[0]:
            idx_w = 0
            window_border_h = [max(0, idx_h-window_size), min(idx_h+window_size+1, image.shape[0]-1)]
            while idx_w < image.shape[1]:
                window_border_w = [max(0, idx_w-window_size), min(idx_w+window_size+1, image.shape[1]-1)]
                image_new[idx_h, idx_w, idx_c] = np.median(np.ravel(image[window_border_h[0]:window_border_h[1],
                                                                      window_border_w[0]:window_border_w[1],
                                                                      idx_c]))
                # print(np.median(np.ravel(image[window_border_h[0]:window_border_h[1],
                #                                                       window_border_w[0]:window_border_w[1],
                #                                                       idx_c])))
                # test = image[window_border_h[0]:window_border_h[1],window_border_w[0]:window_border_w[1],
                #              idx_c]
                idx_w += step_size
            idx_h += step_size
    image[:, :, :] = image_new[:, :, :]

def contrast(image, value_contrast=1, value_bright=0, mode='simple', step_threshold=0.5):
    if mode == 'simple':
        value_contrast = 1 + value_contrast/10
        for h_idx in range(image.shape[0]):
            for w_idx in range(image.shape[1]):
                for c_idx in range(image.shape[2]):
                    image[h_idx, w_idx, c_idx] = np.clip(value_contrast*image[h_idx, w_idx, c_idx] + value_bright, 0, 1) 

    elif mode == 'sigmoid':
        for h_idx in range(image.shape[0]):
            for w_idx in range(image.shape[1]):
                for c_idx in range(image.shape[2]):
                    image[h_idx, w_idx, c_idx] = np.clip(1 / (1 + np.exp(-1*value_contrast * (image[h_idx, w_idx, c_idx]-step_threshold))), 0, 1)
        

def gaussian_filter(image, sigma=1, radius=1, derivative=0):
    for c_idx in range(image.shape[2]):
        image[:, :, c_idx] = np.clip(simg.gaussian_filter(image[:, :, c_idx], sigma=sigma, radius=radius, order=derivative), 0, 1)


if __name__ == '__main__':
    # img = cv2.imread('is_this_a_pigeon.jpg') 
    # img = cv2.imread('gundam_rg_2.jpg')
    img = cv2.imread('commander_quant_hg.jpg')
    
    img_ds = pooling(img, shrinkage=6)
    img = img / 255
    img_ds = img_ds / 255
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
    
    img_fucked_up = np.copy(img_ds)
    # DCT(img_fucked_up, shape_result=None, norm='ortho', process_type='pass', pass_quadrant='234', pass_rate=0.1)
    # pass_filter(img_fucked_up, span=0.9, pass_type='high')
    edge_detect_laplace(img_fucked_up, direction='45_degree')
    # contrast(img_fucked_up, mode='sigmoid', value_contrast=20, step_threshold=0.3)
    # gaussian_filter(img_fucked_up, sigma=7, radius=1, derivative=0)
    log_transform(img_fucked_up, c=10)
    # median_filter(img_fucked_up, window_size=3, step_size=1)
    
    # negatives(img_fucked_up)
    # histogram(img_fucked_up, channel_order='BGR')    
    
    img_mixed = np.clip(img_ds - img_fucked_up, 0, 1)
    
    # cv2.imshow('image',img)
    # cv2.imshow('pooled', img_ds)
    cv2.imshow('random bullshit', img_fucked_up)
    # cv2.imshow('mixed', img_mixed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    
    
