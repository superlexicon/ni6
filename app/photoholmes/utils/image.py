import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .jpeg_utils import get_dct_coefficients_and_qtables, JPEGInfo

logger = logging.getLogger(__name__)


def read_image(path: str | Path) -> Tensor:
    """
    Read an image from a file and return it as a tensor.

    Args:
        path (str): The path to the image file.

    Returns:
        Tensor: The image as a tensor.
    """
    return torch.from_numpy(
        cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2RGB).transpose(2, 0, 1)
    )


def save_image(path: str, img: Tensor | NDArray, *args):
    """
    Save an image to a file.

    Args:
        path (str): The path to the file.
        img (Tensor | NDArray): The image to save.
        *args: Additional arguments to pass to `cv.imwrite`.
    """

    if isinstance(img, Tensor):
        img_bgr = cv.cvtColor(tensor2numpy(img), cv.COLOR_RGB2BGR)
    else:
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(path, img_bgr, *args)


def tensor2numpy(image: Tensor) -> NDArray:
    """
    Convert a tensor to a numpy array and transpose the dimensions.

    Args:
        image (Tensor): The image to convert.

    Returns:
        NDArray: The image as a numpy array.
    """
    # Move tensor to CPU if on GPU/MPS device
    if image.device.type != 'cpu':
        image = image.cpu()
    img = image.numpy()
    return img.transpose(1, 2, 0) if image.ndim > 2 else img


def plot(
    image: Tensor | NDArray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Function for easily plotting one image.

    Args:
        image (Tensor | NDArray): The image to plot.
        title Optional[str]: The title of the plot.
        save_path Optional[str]: The path to save the plot to.
    """
    if isinstance(image, Tensor):
        image = tensor2numpy(image)
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis(False)
    if save_path is not None:
        plt.savefig(save_path)
        print("Figure saved at:", save_path)
    plt.show()


def plot_multiple(
    images: List[Tensor | NDArray],
    title: Optional[str] = None,
    ncols: int = 4,
    save_path: Optional[str] = None,
    titles: Optional[List[Optional[str]]] = None,
):
    """
    Function for easily plotting multiple images.

    Args:
        images (List[Tensor | NDArray]): The images to plot.
        title Optional[str]: The title of the plot.
        ncols int: The number of columns in the plot.
        save_path Optional[str]: The path to save the plot to.
        titles Optional[List[Optional[str]]]: The titles of the images.
    """
    N = len(images)
    nrows = np.ceil(N / ncols).astype(int)
    if titles is None:
        titles = [None] * len(images)  # type: ignore
    if nrows > 1:
        _, ax = plt.subplots(nrows, ncols)
        for n, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = tensor2numpy(img)
            i = n // ncols
            j = n % ncols
            ax[i, j].imshow(img)
            ax[i, j].set_title(titles[n])  # type: ignore
            ax[i, j].set_axis_off()
    else:
        fig, ax = plt.subplots(1, N)
        for n, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = tensor2numpy(img)
            ax[n].imshow(img)
            ax[n].set_title(titles[n])  # type: ignore
            ax[n].set_axis_off()
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print("Figure saved at:", save_path)
    plt.show()


def overlay_mask(img: NDArray, heatmap: NDArray) -> NDArray:
    """
    Overlay a heatmap on an image.

    Args:
        img (NDArray): The image.
        heatmap (NDArray): The heatmap.

    Returns:
        NDArray: The image with the heatmap overlayed.
    """
    # Normalize the heatmap to 0-255 and convert to 8-bit unsigned integer
    heatmap_normalized = cv.normalize(
        heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX
    )
    heatmap_uint8 = np.uint8(heatmap_normalized)

    # Apply the color map
    heatmap_img = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

    # Superimpose the heatmap on the image
    super_imposed_img = cv.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    # Convert superimposed image from BGR to RGB for plotting
    super_imposed_img_rgb = cv.cvtColor(super_imposed_img, cv.COLOR_BGR2RGB)
    return super_imposed_img_rgb


def read_jpeg_data(
    image_path: str,
    num_dct_channels: Optional[int] = None,
    all_quant_tables: bool = False,
    suppress_not_jpeg_warning: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Reads image from path and returns DCT coefficient matrix for each channel and the
    quantization matrixes. Uses modern JPEG processing libraries instead of jpegio.

    Args:
        image_path (str): Path to the image.
        num_dct_channels (int, optional): Number of channels to read from the DCT stream.
            Defaults to None.
        all_quant_tables (bool, optional): If True, returns all quantization tables.
            Defaults to False.
        suppress_not_jpeg_warning (bool, optional): If True, suppresses the warning
            when the image is not in JPEG format. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor]: DCT coefficients and quantization tables.
    """
    return get_dct_coefficients_and_qtables(
        image_path=image_path,
        num_dct_channels=num_dct_channels,
        all_quant_tables=all_quant_tables,
        suppress_not_jpeg_warning=suppress_not_jpeg_warning
    )


# jpegio functions replaced with modern implementation in jpeg_utils.py
# Legacy _qtables_from_jpeg and _DCT_from_jpeg functions are no longer needed
