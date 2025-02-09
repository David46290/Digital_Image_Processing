import cv2
import random
import scipy.fftpack as sfft
import scipy.ndimage as simg
import numpy as np
from scipy import signal as scisig
from matplotlib import pyplot as plt
from PIL import Image

def min_max_normal(array_2d):
    return (array_2d-np.amin(array_2d)) / (np.amax(array_2d) - np.amin(array_2d))
                             

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

def resize(image, size):
    new_img = Image.fromarray(np.uint8(image*255))
    new_img = new_img.resize(size, Image.BILINEAR)
    return np.asarray(new_img) 
        
def pass_filter(image, span=0.1, pass_type='low'):
    diemnsion = image.shape[2]
    h = image.shape[0]
    w = image.shape[1]
    
    h_center = h // 2
    w_center = w // 2
       
    
    image_fft = sfft.fft2(image) # 0 freqeucny component locates at to-left corner
    np.fft.fftshift(image_fft)
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

    
def log_transform(image, c, normalized=False):
    for c_idx in range(image.shape[2]):
        if not normalized:
            image[:, :, c_idx] = np.clip(c * np.log(1 + image[:, :, c_idx]), 0, 1)
        else:
            image[:, :, c_idx] = min_max_normal(c * np.log(1 + image[:, :, c_idx]))

def DCT(image, shape_result=None, norm='ortho', process_type='random', pass_rate=0.5, pass_quadrant='1234'):
    image_dct = sfft.dctn(image, type=2, shape=shape_result, norm=norm)
    changed_value = -10
    if process_type == 'random':
        # randomly edit element and see what can be changed
        image_dct = image_dct * np.random.rand(image.shape[0], image.shape[1], image.shape[2])
    elif process_type == 'pass':
        #[1, 2]
        #[3, 4]
        threshold_h = int(image.shape[0] * pass_rate)
        threshold_w = int(image.shape[1] * pass_rate)
        if '1' not in pass_quadrant:
            image_dct[:threshold_h, :threshold_w, :] = changed_value
        if '2' not in pass_quadrant:
            image_dct[:threshold_h, threshold_w:image.shape[1], :] = changed_value
        if '3' not in pass_quadrant:
            image_dct[threshold_h:image.shape[0], :threshold_w, :] = changed_value
        if '4' not in pass_quadrant:
            image_dct[threshold_h:, threshold_w:, :] = changed_value
    image[:, :, :] = np.clip(sfft.idctn(image_dct, type=2, shape=image.shape, norm=norm), 0, 1)

def DCT_compress(image, norm='ortho', shrinkage=0.5):
    pixel_skip_rate = int(1 // shrinkage)
    new_h = int(image.shape[0] * shrinkage)
    new_w = int(image.shape[1] * shrinkage)
    image_dct = sfft.dctn(image, type=2, norm=norm)
    new_dct = np.zeros((new_h, new_w, image.shape[2]))
    for idx_c in range(image_dct.shape[2]):
        idx_h = 0
        idx_new_h = 0
        while idx_new_h < new_h:
            idx_w = 0
            idx_new_w = 0
            while idx_new_w < new_w:
                if (idx_h % pixel_skip_rate == 0) and (idx_w % pixel_skip_rate == 0):
                    new_dct[idx_new_h, idx_new_w, idx_c] = image_dct[idx_h, idx_w, idx_c] 
                idx_w = idx_w + pixel_skip_rate
                idx_new_w = idx_new_w + 1
            idx_h = idx_h + pixel_skip_rate
            idx_new_h = idx_new_h + 1
                    
    return np.clip(sfft.idctn(new_dct, type=2, norm=norm), 0, 1)


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
                idx_w += step_size
            idx_h += step_size
    image[:, :, :] = image_new[:, :, :]

def contrast(image, value_contrast=1, value_bright=0, mode='linear', step_threshold=0.5):
    if mode == 'linear':
        for h_idx in range(image.shape[0]):
            for w_idx in range(image.shape[1]):
                for c_idx in range(image.shape[2]):
                    image[h_idx, w_idx, c_idx] = np.clip(value_contrast*image[h_idx, w_idx, c_idx] + value_bright, 0, 1) 

    elif mode == 'sigmoid':
        for h_idx in range(image.shape[0]):
            for w_idx in range(image.shape[1]):
                for c_idx in range(image.shape[2]):
                    image[h_idx, w_idx, c_idx] = np.clip(1 / (1 + np.exp(-1*value_contrast * (image[h_idx, w_idx, c_idx]-step_threshold))) + value_bright, 0, 1)
    
    elif mode == 'cubic':
        for h_idx in range(image.shape[0]):
            for w_idx in range(image.shape[1]):
                for c_idx in range(image.shape[2]):
                    image[h_idx, w_idx, c_idx] = np.clip(value_contrast * (image[h_idx, w_idx, c_idx]-step_threshold)**3 + 0.5 + value_bright, 0, 1)
        

def gaussian_filter(image, sigma=1, radius=1, order=0):
    for c_idx in range(image.shape[2]):
        image[:, :, c_idx] = np.clip(simg.gaussian_filter(image[:, :, c_idx], sigma=sigma, radius=radius, order=order), 0, 1)

def salt_pepper(image):
    for h_idx in range(image.shape[0]):
        for w_idx in range(image.shape[1]):
            for c_idx in range(image.shape[2]):
                rng = np.random.randint(low=0, high=11)
                if rng < 2:
                    image[h_idx, w_idx, c_idx] = 0
                elif rng > 7:
                    image[h_idx, w_idx, c_idx] = 1




    
    
    