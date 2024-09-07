import cv2
import random
import scipy.fftpack as sfft
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
        # image_fft[:, threshold_w[0]:threshold_w[1], :] = 0
    else:
        print('no assigned pass type for freq-pass')
    
    image_filtered = np.abs(np.fft.ifft2(image_fft))
    return image_fft , image_filtered

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
    image_negative = -1 * ((image*255).astype(int) - 255)
    return image_negative / 255
    
def log_transform(image, c, is_c_function=False):
    if not is_c_function:
        return c * np.log(1 + image)
    else:
        result_of_c = c(image)
        return result_of_c * np.log(1 + image)

def DCT(image, shape_result=None, norm='ortho', process_type='random', pass_rate=0.5, pass_quadrant='1234'):
    diemnsion = image.shape[2]
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
        
    
    image_idct = sfft.idctn(image_dct, type=2, shape=shape_result, norm=norm)
    return image_dct, image_idct

if __name__ == '__main__':
    img = cv2.imread('is_this_a_pigeon.jpg') 
    # img = cv2.imread('gundam_rg_2.jpg')
    # img = cv2.imread('commander_quant_hg.jpg')
    img_ds = pooling(img, shrinkage=6)
    img = img / 255
    img_ds = img_ds / 255
    # img_fft = sfft.fft2(img_ds)
    # img_fft_2, img_filtered = pass_filter(img_ds, span=0.9, pass_type='low')
    # img_filtered2 = fft_kernel_conv(img_ds, std=3)
    
    # histogram(img_ds, channel_order='BGR')
    img_dct, img_idct = DCT(img_ds, shape_result=None, norm='ortho'
                            , process_type='pass', pass_quadrant='134')
    
    # cv2.imshow('image',img)
    cv2.imshow('pooled',img_ds)
    # cv2.imshow('pooled_fft',np.real(img_fft))
    # cv2.imshow('passed',img_filtered)
    # cv2.imshow('negative', negatives(img_ds))
    # cv2.imshow('log', log_transform(img_ds, c=5))
    cv2.imshow('dct_idct',img_idct)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    
    
