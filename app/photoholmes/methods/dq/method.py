from typing import List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from ...base import BasePhotoHolmesMethod as BaseMethod, BenchmarkOutput
from .utils import ZIGZAG, fft_period, histogram_period
from ...postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)


class DQ(BaseMethod):
    """
    Implementation of DQ [Lin, et.al. 2009]. The method detects forgeries
    exploiting double quantization discrepancies. It uses the DCT coefficients
    of the image to calculate the BPPM (Block Posterior Probability Map) heatmap.
    """

    def __init__(self, number_frecs: int = 8, alpha: float = 1.0, **kwargs) -> None:
        """
        Initializes the DQ class with specified parameters.

        Args:
            number_frecs (int): Number of frequency components to consider.
                Defaults to 8.
            alpha (float): Alpha value used in period detection.
                Defaults to 1.0.
        """
        super().__init__(**kwargs)
        self.number_frecs = number_frecs
        self.alpha = alpha

    def predict(
        self, dct_coefficients: NDArray, image_size: Tuple[int, int]
    ) -> NDArray:
        """
        Predicts the BPPM (Block Posterior Probability Map) heatmap from DCT
        coefficients.

        Args:
            dct_coefficients (np.ndarray): Array containing DCT coefficients of the
                image.
            image_size (Tuple[int, int]): Tuple representing the dimensions of the
                image.

        Returns:
            Tensor: The predicted BPPM heatmap as a PyTorch tensor.
        """
        # Handle different input formats
        if len(dct_coefficients.shape) == 5:
            # Format: (channels, height_blocks, width_blocks, 8, 8)
            # Need to reshape to (channels, height_blocks*8, width_blocks*8)
            channels, hb, wb, block_h, block_w = dct_coefficients.shape
            if block_h == 8 and block_w == 8:
                # Reshape from DCT blocks to full DCT coefficient matrix
                M, N = hb * 8, wb * 8
                # Reshape each channel: (hb, wb, 8, 8) -> (hb*8, wb*8)
                reshaped_channels = []
                for c in range(channels):
                    # Reshape the blocks into full coefficient matrix
                    channel_coef = dct_coefficients[c].reshape(hb, wb, 8, 8)
                    # Use einsum to rearrange blocks into full matrix
                    channel_coef = channel_coef.transpose(0, 2, 1, 3).reshape(M, N)
                    reshaped_channels.append(channel_coef)
                dct_coefficients = np.stack(reshaped_channels)
            else:
                raise ValueError(f"Expected 8x8 blocks, got {block_h}x{block_w}")
        elif len(dct_coefficients.shape) == 3:
            # Expected format: (channels, height, width)
            channels, M, N = dct_coefficients.shape
        elif len(dct_coefficients.shape) == 4:
            # Format: (channels, 1, height, width) - squeeze the extra dimension
            channels, _, M, N = dct_coefficients.shape
            dct_coefficients = dct_coefficients.squeeze(1) if dct_coefficients.shape[1] == 1 else dct_coefficients
        elif len(dct_coefficients.shape) == 2:
            # Format: (height, width) - single channel, add channel dimension
            M, N = dct_coefficients.shape
            channels = 1
            dct_coefficients = dct_coefficients.reshape(1, M, N)
        else:
            raise ValueError(f"Unexpected dct_coefficients shape: {dct_coefficients.shape}")

        BPPM = np.zeros((M // 8, N // 8))
        for channel in range(channels):
            channel_coefs = dct_coefficients[channel]
            BPPM += self._calculate_BPPM_channel(
                channel_coefs, ZIGZAG[: self.number_frecs]
            )
        BPPM_norm = torch.from_numpy(BPPM / channels)

        # Handle any NaN or infinite values
        BPPM_norm = torch.nan_to_num(BPPM_norm, nan=0.0, posinf=0.0, neginf=0.0)

        BPPM_upsampled = simple_upscale_heatmap(BPPM_norm, 8)
        heatmap = resize_heatmap_with_trim_and_pad(BPPM_upsampled, image_size)

        # Handle any NaN values in final heatmap
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

        return heatmap

    def benchmark(
        self, dct_coefficients: NDArray, image_size: Tuple[int, int]
    ) -> BenchmarkOutput:
        """
        Benchmarks the DQ method using provided DCT coefficients and image size.

        Args:
            dct_coefficients (np.ndarray): DCT coefficients of the image.
            image_size (Tuple[int, int]): Dimensions of the image.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholders for mask and
                detection.
        """
        heatmap = self.predict(dct_coefficients, image_size)
        return {"heatmap": torch.from_numpy(heatmap), "mask": None, "detection": None}

    def _detect_period(self, histogram: NDArray) -> int:
        """
        Detects the repeating period in a histogram.

        Args:
            histogram (np.ndarray): The input histogram from image frequencies.

        Returns:
            int: The detected period within the histogram.
        """
        if len(histogram) < 2:
            return 1
        else:
            p_H = histogram_period(histogram, self.alpha)
            p_fft = fft_period(histogram)
            p = min(p_H, p_fft)
            return p

    def _calculate_Pu(
        self, coefficients_f: NDArray, histogram: NDArray, period: int
    ) -> NDArray:
        """
        Calculates Pu values for given frequency coefficients based on histogram.

        Args:
            coefficients_f (np.ndarray): Frequency-specific DCT coefficients.
            histogram (np.ndarray): Histogram of values for the specific frequency.
            period (int): Detected period for the histogram of the frequency.

        Returns:
            np.ndarray: Array of Pu values for the given frequency coefficients.
        """
        coefficients_f -= np.min(coefficients_f)
        M, N = coefficients_f.shape

        histogram_padded = np.pad(histogram, (0, period))

        # Convert DCT coefficients to integer indices
        # DCT coefficients are floating point values that need to be mapped to histogram bins
        coefficient_indices = coefficients_f.ravel()

        # Ensure coefficients are within valid histogram range
        coefficient_indices = np.clip(coefficient_indices, 0, len(histogram) - 1).astype(int)

        # Get histogram values for the coefficient indices and period
        histogram_range = histogram_padded[
            coefficient_indices[:, np.newaxis] + np.arange(period) - 1
        ]

        # Calculate Pu_f with division by zero protection
        histogram_sums = np.sum(histogram_range, axis=1)
        Pu_f = np.zeros_like(histogram_sums, dtype=float)

        # Only calculate division for non-zero sums to avoid division by zero
        non_zero_mask = histogram_sums > 0
        Pu_f[non_zero_mask] = histogram_range[non_zero_mask, 1] / histogram_sums[non_zero_mask]

        Pu_f = Pu_f.reshape((M, N))

        return Pu_f

    def _calculate_BPPM_f(self, DCT_coefficients_f: NDArray) -> NDArray:
        """
        Calculates BPPM values for given DCT coefficients at a specific frequency.

        Args:
            DCT_coefficients_f (np.ndarray): DCT coefficients for a specific frequency.

        Returns:
            np.ndarray: Calculated BPPM values for the given frequency.
        """
        hmax = np.max(DCT_coefficients_f)
        hmin = np.min(DCT_coefficients_f)

        if hmax - hmin:
            # Use a fixed number of bins for better histogram resolution
            bins_count = 256  # Fixed bin count for better resolution

            hist, _ = np.histogram(
                DCT_coefficients_f, bins=bins_count, range=(hmin, hmax)
            )
            p = self._detect_period(hist[1:-1])
            if p != 1:
                Pu = self._calculate_Pu(DCT_coefficients_f, hist, p)
                Pt = 1 / p
                BPPM_f = Pt / (Pu + Pt)
                saturated = (DCT_coefficients_f == DCT_coefficients_f.min()) | (
                    DCT_coefficients_f == DCT_coefficients_f.max()
                )
                BPPM_f[saturated] = 0

                return BPPM_f

        return np.zeros_like(DCT_coefficients_f)

    def _calculate_BPPM_channel(
        self, DCT_coefs: NDArray, fs: List[Tuple[int, int]]
    ) -> NDArray:
        """
        Calculates BPPM values for all frequencies within a single image channel.

        Args:
            DCT_coefs (np.ndarray): DCT coefficients for the image channel.
            fs (List[Tuple[int, int]]): List of frequency tuples to consider.

        Returns:
            np.ndarray: Aggregated BPPM values for the image channel.
        """
        M, N = DCT_coefs.shape
        BPPM = np.zeros((len(fs), M // 8, N // 8))
        for i in range(len(fs)):
            DCT_coefficients_f = DCT_coefs[fs[i][0] :: 8, fs[i][1] :: 8]
            BPPM[i] = self._calculate_BPPM_f(DCT_coefficients_f)

        return BPPM.sum(axis=0) / self.number_frecs
