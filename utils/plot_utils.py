# Assignment imports
from pyraws.utils.visualization_utils import equalize_tensor
import matplotlib.pyplot as plt
import torch

# ############################################################# Assignment functions
def plot_image(image, equalize=True, downsampling_factor=(2,2),figsize=(20,20)):

    # Applying reshaping and downsampling
    image=image.permute(2, 1, 0)
    image = image[::downsampling_factor[0], ::downsampling_factor[1]]
    if not equalize:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.imshow(image,figsize=figsize)
        return fig
    else:
        image=image.type(torch.float32)
        fig, ax = plt.subplots(1,2,figsize=figsize)
        ax[0].imshow(image/image.max())
        ax[0].set_title("Raw")
        ax[1].imshow(image)
        image_equalized=equalize_tensor(image)
        ax[1].imshow(image_equalized/image_equalized.max())
        ax[1].set_title("Filtered")
        return fig

