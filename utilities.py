import torch

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt


def convert_show_rgb(grayscale_input, ab_input, original_image):
    grayscale_input = grayscale_input.squeeze(0)
    grayscale_input = grayscale_input.cpu()
    ab_input = ab_input.squeeze(0)
    ab_input = ab_input.cpu()
    
    color_image = torch.cat((grayscale_input, ab_input), 0).detach().numpy()        # combine channels
    color_image = color_image.transpose((1, 2, 0))                                  # rescale for matplotlib
    grayscale_image = grayscale_input.numpy()
    grayscale_image = grayscale_image.transpose((1,2,0))
    grayscale_image = grayscale_image.reshape(224,224)
    color_image[:, :, 0:1] = (color_image[:, :, 0:1] * 100)                         #Multiply by 100 : L varies form 0 to 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128

    color_image = lab2rgb(color_image.astype(np.float64))                           #For floating point pixel values vary form 0 - 1
    fig,ax = plt.subplots(1,3, figsize=(8, 8))
    ax[0].imshow(grayscale_image,cmap='gray')
    ax[1].imshow(original_image)
    ax[2].imshow(color_image)
    
    ax[0].title.set_text("Grayscale Image")
    ax[1].title.set_text("Original Image")
    ax[2].title.set_text("CNN Output Image")
    
    plt.show()
    
def convert_frame_rgb(grayscale_input, ab_input):
    grayscale_input = grayscale_input.squeeze(0)
    grayscale_input = grayscale_input.cpu()
    ab_input = ab_input.squeeze(0)
    ab_input = ab_input.cpu()
    
    color_image = torch.cat((grayscale_input, ab_input), 0).detach().numpy()        # combine channels
    color_image = color_image.transpose((1, 2, 0))                                  # rescale for matplotlib
    grayscale_image = grayscale_input.numpy()
    grayscale_image = grayscale_image.transpose((1,2,0))
    grayscale_image = grayscale_image.reshape(224,224)
    color_image[:, :, 0:1] = (color_image[:, :, 0:1] * 100)                         #Multiply by 100 : L varies form 0 to 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128

    color_image = lab2rgb(color_image.astype(np.float64))                           #For floating point pixel values vary form 0 - 1
    
    return color_image
    