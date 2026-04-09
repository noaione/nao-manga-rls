# (c) 2025- anon & ryou, Unknown license
# The only changes by me is replacing OpenCV with PIL
# and making some imports lazy to reduce the initial load time of the CLI

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageChops

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor
    from torch.nn import Sequential as TorchSequential

__all__ = (
    "OGSOV",
    "detect_image_color",
    "detect_image_color_ogsov",
    "is_grayscale_palette",
)


@dataclass(frozen=True)
class DetectedColor:
    is_color: bool
    """:bool: whether the image is detected as color or not"""
    confidence: int  # 0-100, only for ML-based detection
    """:int: confidence level of the prediction, only for ML-based detection, 0-100"""
    reason: str | None = None
    """:str | None: reason for the prediction, can be used for debugging or logging purposes"""
    should_convert: bool = False
    """:bool: whether the image should be converted to grayscale, can be used for further processing"""


class OGSOV:
    """Color detection model, credits to anon for the implementation"""

    def __init__(self, weights_file: PathLike) -> None:
        """
        Weights file is a .npz file containing the weights for the model.

        The model is currently private, so no public release yet.
        """

        import numpy as np
        import torch
        from scipy.stats import kurtosis, skew

        self.__np = np
        self.__torch = torch
        self.__skew = skew
        self.__kurtosis = kurtosis

        # load the weights
        loaded_data_dict = np.load(weights_file, allow_pickle=False)

        # lookup array
        self.lookup_mask: "ndarray" = loaded_data_dict["mask_lookup"].astype(bool)

        # create classifiers models
        self.classifiers = torch.nn.ModuleList([self._create_base_model() for _ in range(5)])
        self._load_base_model(loaded_data_dict)

    def _create_base_model(self) -> "TorchSequential":
        x = self.__torch.nn.Sequential(
            self.__torch.nn.Linear(32, 512),
            self.__torch.nn.Tanh(),
            self.__torch.nn.Identity(),
            self.__torch.nn.Linear(512, 512),
            self.__torch.nn.Tanh(),
            self.__torch.nn.Identity(),
            self.__torch.nn.Linear(512, 1),
        )
        return x

    def _load_base_model(self, data: dict[str, "ndarray"]) -> None:
        # convert the keys to torch tensor
        weight_tensors: dict[str, "Tensor"] = {
            key: self.__torch.from_numpy(value) for key, value in data.items() if key[0].isdigit()
        }

        # load the dict
        self.classifiers.load_state_dict(weight_tensors)

    def _forward_layers(self, x: "Tensor") -> "Tensor":
        # store results
        results_tensors = []

        # compute the forward pass
        with self.__torch.no_grad():
            for classifer_model in self.classifiers:
                y = classifer_model(x)
                y = self.__torch.sigmoid(y)
                results_tensors.append(y.detach())

            # merge the tensors
            combined_tenors = self.__torch.cat(results_tensors, dim=-1)

            # 1. Calculate the mean probability across all classifiers
            mean_prob = self.__torch.mean(combined_tenors, dim=1)

            # 2. Determine the predicted class (0 or 1)
            predicted_class = (mean_prob >= 0.5).float()

            # 3. Calculate confidence
            confidence = self.__torch.where(predicted_class == 1, mean_prob, 1.0 - mean_prob)

        # 4. Stack them into (N, 2)
        stacked_vector = self.__torch.stack([predicted_class, confidence], dim=1)
        stacked_vector = stacked_vector.detach()

        return stacked_vector

    def binary_mask_lookup(self, bgr_array: "ndarray") -> "ndarray":
        # 0. Store original dimensions
        h, w, _ = bgr_array.shape

        # 1. Flat the image out
        fi = bgr_array.reshape(-1, 3)

        # 3. Map to indices and reshape
        mask_images = self.lookup_mask[fi[:, 0], fi[:, 1], fi[:, 2]]
        mask_images = mask_images.reshape(h, w)

        # return the binary mask
        return mask_images

    def extract_colurfullness(self, bgr_array: "ndarray") -> "ndarray":
        # compute the distance formulate here
        b_array = bgr_array[:, :, 0].astype(self.__np.float32)
        g_array = bgr_array[:, :, 1].astype(self.__np.float32)
        r_array = bgr_array[:, :, 2].astype(self.__np.float32)

        # actually compute the distances here
        rg_distance = r_array - g_array
        yb_distance = (0.5 * (r_array + g_array)) - b_array

        # compute and ensure the value is 0-1
        colorfull_distance = self.__np.sqrt((rg_distance**2) + (yb_distance**2))
        colorfull_distance = self.__np.clip(colorfull_distance, 0.0, 285.0)
        colorfull_distance = (colorfull_distance / 285.0).astype(self.__np.float32)

        # return the float32 array
        return colorfull_distance

    def _metadata_extractor(self, bgr_image: "ndarray") -> "ndarray":
        # dont change constants
        top_n_count = 17
        median_scaler = 1000

        # compute the colorfullness
        color_array = self.extract_colurfullness(bgr_image)

        # extract for top-n + support
        flat_array = color_array.ravel()

        # 1. Get indices of the top N elements
        partitioned_indices = self.__np.argpartition(flat_array, -top_n_count)
        top_indices = partitioned_indices[-top_n_count:]
        top_vectors = flat_array[top_indices]

        # Get the "Pool" for random sampling (everything EXCEPT top_indices)
        remaining_indices_pool = partitioned_indices[:-top_n_count]
        remaning_vectors = flat_array[remaining_indices_pool].astype(self.__np.float32)

        # interger based values
        integer_vectors = self.__np.round(remaning_vectors * median_scaler).astype(self.__np.uint16)

        # basic Statistics
        std = float(self.__np.std(remaning_vectors))
        med = float(self.__np.median(integer_vectors) / median_scaler)

        # moments
        skw = float(self.__skew(remaning_vectors))
        krt = float(self.__kurtosis(remaning_vectors))

        # get the quantiles valuse
        quantiles = self.__np.percentile(remaning_vectors, [0, 25, 50, 75, 100])

        # Interquartile Range (IQR)
        iqr = quantiles[3] - quantiles[1]

        # l2 norm value
        l2_norm = float(self.__np.linalg.norm(remaning_vectors))

        # create the stacked vectors
        output_features_vectors = self.__np.concatenate([top_vectors, [std, med, skw, krt, iqr, l2_norm], quantiles])

        # return the vectors
        return output_features_vectors

    def compute_image_vectors(self, bgr_array: "ndarray") -> "ndarray":
        # compute the base vectors
        metadata_vectors = self._metadata_extractor(bgr_array)

        # compute the generic lookup
        mono_bool_array = self.binary_mask_lookup(bgr_array).astype(bool)

        # extract additional info
        total_color_pixels_value = int(self.__np.sum(mono_bool_array))
        tc_0 = float(min(total_color_pixels_value, 2) / 2.0)
        tc_1 = float(min(total_color_pixels_value, 16) / 16.0)
        tc_2 = float(min(total_color_pixels_value, 128) / 128.0)
        tc_3 = float(min(total_color_pixels_value, 1024) / 1024.0)

        # merge the extra features and metadata features
        output_features = self.__np.concatenate([
            metadata_vectors,
            [tc_0, tc_1, tc_2, tc_3],
        ])

        # Ensure non NaN values
        output_features = self.__np.nan_to_num(output_features, True, 0.0, 0.0, 0.0)

        # ensure data type
        output_features = output_features.astype(self.__np.float32)

        # Retun the clean vectors
        return output_features

    def predict(self, bgr_image: "ndarray") -> tuple[bool, int]:
        # extract the feature vectores
        feature_vectors = self.compute_image_vectors(bgr_image)

        # ensure batch
        feature_vectors = self.__np.expand_dims(feature_vectors, axis=0)
        feature_vectors = self.__torch.from_numpy(feature_vectors)

        # forward pass
        prediction_tensor = self._forward_layers(feature_vectors)

        # Extract the Features
        predicted_class = int(prediction_tensor[0, 0].item())
        predicted_confidence = int(prediction_tensor[0, 1].item() * 100)

        # return the prediction
        is_input_color = bool(predicted_class)
        return is_input_color, predicted_confidence


_preload_ogsov_instance: dict[str, OGSOV] = {}


def is_grayscale_palette(palette: list[int]) -> bool:
    for i in range(0, len(palette), 3):
        r, g, b = palette[i : i + 3]
        if (r != g) or (g != b):
            return False

    return True


def detect_image_color(img: Image.Image) -> DetectedColor:
    """
    Basic color detection, very fast but not very accurate, can be used as a pre-check before the ML-based detection.
    """

    if img.mode == "P":
        # is gray palette
        palette = img.getpalette(rawmode=None)
        if palette is not None and is_grayscale_palette(palette):
            return DetectedColor(False, 100, "is grayscale palette", True)

    if img.mode == "L":
        return DetectedColor(False, 100, "is grayscale mode", False)

    if img.mode == "RGB":
        rgb = img.split()
        if (
            ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] == 0
            and ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] == 0
        ):
            return DetectedColor(False, 100, "is grayscale RGB", True)

    return DetectedColor(True, 0, None, False)


def detect_image_color_ogsov(img_bytes: bytes, *, weights_file: PathLike) -> DetectedColor:
    """
    ML-version of color detection, more accurate but slower than the basic one.
    """

    import numpy as np

    global _preload_ogsov_instance

    # load the model if not loaded
    if isinstance(weights_file, Path):
        weight_str = str(weights_file.resolve())
    else:
        weight_str = str(weights_file)

    instance = _preload_ogsov_instance.get(weight_str)
    if instance is None:
        instance = OGSOV(weights_file)
        _preload_ogsov_instance[weight_str] = instance

    image = Image.open(BytesIO(img_bytes))
    if (detected := detect_image_color(image)) and not detected.is_color:
        image.close()
        return detected  # if the basic detection already says it's not color, we can skip the ML-based detection

    rgb_array = np.array(image.convert("RGB"))
    bgr_array = rgb_array[:, :, ::-1]

    # predict the colorfulness
    is_color, confidence = instance.predict(bgr_array)
    image.close()
    return DetectedColor(is_color, confidence, "ML-based detection", is_color)
