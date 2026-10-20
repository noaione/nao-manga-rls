# (c) 2025- anon & ryou, Unknown license
# The only changes by me is replacing OpenCV with PIL
# and making some imports lazy to reduce the initial load time of the CLI

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from PIL import Image, ImageChops

from .lazy import get_einops_rearrange, get_numpy, get_onnxruntime, get_torch

if TYPE_CHECKING:
    from numpy import ndarray
    from onnxruntime import InferenceSession
    from torch import Tensor
    from torch.nn import Sequential as TorchSequential

__all__ = (
    "OGSOV",
    "DetectedColor",
    "detect_image_color",
    "detect_image_color_ml",
    "is_grayscale_palette",
    "register_ogsov_instance",
    "register_ogsox_instance",
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


class MangaColorInference(Protocol):
    """The root protocol inference"""

    def predict(self, bgr_image: "ndarray", *, fast_method: bool = False) -> tuple[bool, int]: ...


class OGSOV(MangaColorInference):
    """Color detection model, credits to anon for the implementation"""

    def __init__(self, weights_file: PathLike) -> None:
        """
        Weights file is a .npz file containing the weights for the model.

        The model is currently private, so no public release yet.
        """

        np = get_numpy()
        torch = get_torch()
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

    def compute_image_vectors_hist(self, bgr_array: "ndarray") -> "ndarray":
        fi = bgr_array.reshape(-1, 3).astype(self.__np.int32)
        b, g, r = fi[:, 0], fi[:, 1], fi[:, 2]

        rg = r - g
        yb2 = r + g - 2 * b
        d2 = 4 * rg * rg + yb2 * yb2

        hist = self.__np.bincount(d2)  # length = max_d2 + 1
        n = d2.shape[0]

        # bin values, bit-identical to the original float32 path
        vals = self.__np.minimum(
            self.__np.sqrt(self.__np.arange(len(hist), dtype=self.__np.float32)) * self.__np.float32(0.5),
            self.__np.float32(285.0),
        ) / self.__np.float32(285.0)

        # top 17 (descending), removed from the histogram
        top_n = 17
        h2 = hist.copy()
        nz = self.__np.flatnonzero(h2)
        top = self.__np.empty(top_n, dtype=self.__np.float32)
        got = 0
        for bin_ in nz[::-1]:
            take = min(int(h2[bin_]), top_n - got)
            top[got : got + take] = vals[bin_]
            h2[bin_] -= take
            got += take
            if got == top_n:
                break

        m = n - top_n
        mf = float(m)

        # order statistics via cumulative counts
        cum = self.__np.cumsum(h2)

        def elem(k):  # value at 0-based sorted position k
            return float(vals[self.__np.searchsorted(cum, k, side="right")])

        def lerp(q):
            pos = q / 100.0 * (mf - 1.0)
            lo, hi = int(self.__np.floor(pos)), int(self.__np.ceil(pos))
            a, bb = elem(lo), elem(hi)
            return a + (pos - lo) * (bb - a)

        quantiles = self.__np.array([elem(0), lerp(25.0), lerp(50.0), lerp(75.0), elem(m - 1)])
        iqr = quantiles[3] - quantiles[1]

        # weighted moments in float64
        cnt = h2.astype(self.__np.float64)
        vd = vals.astype(self.__np.float64)
        s1 = float(cnt @ vd)
        mean = s1 / mf
        l2_norm = float(self.__np.sqrt(cnt @ (vd * vd)))
        dv = vd - mean
        dv2 = dv * dv
        m2 = float(cnt @ dv2) / mf
        m3 = float(cnt @ (dv2 * dv)) / mf
        m4 = float(cnt @ (dv2 * dv2)) / mf
        std = float(self.__np.sqrt(m2))
        if m2 > 0.0:
            skw = m3 / m2**1.5
            krt = m4 / (m2 * m2) - 3.0
        else:
            skw = 0.0
            krt = 0.0

        # median of round(v*1000) integers (banker's rounding, np.round)
        scaler = 1000.0
        if m % 2 == 1:
            med = float(self.__np.round(elem(m // 2) * scaler)) / scaler
        else:
            med = (
                (float(self.__np.round(elem(m // 2 - 1) * scaler)) + float(self.__np.round(elem(m // 2) * scaler)))
                / 2.0
                / scaler
            )

        # mask hits (unchanged: one fancy-index gather)
        hits = int(self.__np.sum(self.lookup_mask[fi[:, 0], fi[:, 1], fi[:, 2]]))
        tc = [min(hits, k) / k for k in (2.0, 16.0, 128.0, 1024.0)]

        out = self.__np.concatenate([top, [std, med, skw, krt, iqr, l2_norm], quantiles, tc])
        return self.__np.nan_to_num(out, True, 0.0, 0.0, 0.0).astype(self.__np.float32)

    def predict(self, bgr_image: "ndarray", *, fast_method: bool = False) -> tuple[bool, int]:
        # extract the feature vectores
        if fast_method:
            feature_vectors = self.compute_image_vectors_hist(bgr_image)
        else:
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


def _ogsox_session(fpath: PathLike) -> "InferenceSession":
    ort = get_onnxruntime()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(fpath, options, providers=["CPUExecutionProvider"])


class OGSOX(MangaColorInference):
    TILE: ClassVar[int] = 512
    MASK_CAP: ClassVar[int] = 1024  # the ONNX graph does min(sum(mask), 1024)/...

    def __init__(self, lookup_table_fpath: PathLike, classifier_fpath: PathLike):
        np = get_numpy()

        self.__np = np
        mask = np.load(lookup_table_fpath, allow_pickle=False)["mask_lookup"].astype(bool)
        self.min_spread = self._min_spread(mask)
        self.lookup_flat = np.ascontiguousarray(mask).reshape(-1)  # index = (b << 16) | (g << 8) | r
        self.model_session = _ogsox_session(classifier_fpath)
        self._buf: "ndarray | None" = None
        self._buffer_lock = Lock()

        # For older shit
        self.lookup_array = mask

    def _min_spread(self, mask3d: "ndarray") -> int:
        """Smallest max-min channel spread among True entries of mask[b, g, r] (256 if none)."""
        ar = self.__np.arange(256, dtype=self.__np.int16)
        gr_max, gr_min = self.__np.maximum(ar[:, None], ar[None, :]), self.__np.minimum(ar[:, None], ar[None, :])
        best = 256
        for b in range(256):
            sel = mask3d[b]
            if sel.any():
                best = min(best, int((self.__np.maximum(gr_max, b) - self.__np.minimum(gr_min, b))[sel].min()))
        return best

    def _starts(self, full: int) -> list[int]:
        n = math.ceil(full / self.TILE)
        return [0] if n == 1 else self.__np.round(self.__np.linspace(0, full - self.TILE, n)).astype(int).tolist()

    def _tile_origins(self, h: int, w: int) -> list[tuple[int, int]]:
        """Top-left (y, x) of every tile on the page padded up to at least 512x512 (padding is never materialised)."""
        return [(y, x) for y in self._starts(max(h, self.TILE)) for x in self._starts(max(w, self.TILE))]

    def _spread(self, bgr: "ndarray") -> "ndarray":
        """uint8 (H, W): max(b,g,r) - min(b,g,r). 0 == exactly gray."""
        b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
        return self.__np.maximum(self.__np.maximum(b, g), r) - self.__np.minimum(self.__np.minimum(b, g), r)

    def _capped_hits(self, px_bgr: "ndarray") -> int:
        """min(lookup hits among these (K,3) pixels, MASK_CAP), scanning 4096 at a time and stopping at the cap."""
        count = 0
        for i in range(0, len(px_bgr), 4096):
            blk = px_bgr[i : i + 4096].astype(self.__np.int32)
            count += int(self.__np.count_nonzero(self.lookup_flat[(blk[:, 0] << 16) | (blk[:, 1] << 8) | blk[:, 2]]))
            if count >= self.MASK_CAP:
                return self.MASK_CAP
        return count

    def _tile_mask_count(self, sub_bgr: "ndarray", sub_spread: "ndarray") -> int:
        cand = sub_spread >= self.min_spread
        if not cand.any():
            return 0
        return self._capped_hits(sub_bgr[cand])

    def _batch_buffer(self, n: int) -> "ndarray":
        if self._buf is None or self._buf.shape[0] < n:
            self._buf = self.__np.zeros((n, 4, self.TILE, self.TILE), dtype=self.__np.float32)
        return self._buf[:n]

    def _build_batch(self, bgr: "ndarray") -> "ndarray":
        h, w, _ = bgr.shape
        origins = self._tile_origins(h, w)
        chw = self.__np.ascontiguousarray(bgr.transpose(2, 0, 1), dtype=self.__np.float32)  # one pass over the page
        spread = self._spread(bgr)
        batch = self._batch_buffer(len(origins))
        for i, (y, x) in enumerate(origins):
            vh, vw = min(self.TILE, h - y), min(self.TILE, w - x)
            t = batch[i]
            if vh < self.TILE or vw < self.TILE:
                t[:3] = 0  # padded tile: stale data outside the valid region must be zero
            t[:3, :vh, :vw] = chw[:, y : y + vh, x : x + vw]
            t[3] = 0
            count = self._tile_mask_count(bgr[y : y + vh, x : x + vw], spread[y : y + vh, x : x + vw])
            if count:
                t[3].reshape(-1)[:count] = 1.0  # only the SUM of this channel matters to the model
        return batch

    def score_fast(self, bgr_array: "ndarray") -> float:
        with self._buffer_lock:
            batch = self._build_batch(bgr_array)
            output = self.model_session.run(
                ["final_output"],
                {"input_features": batch},
            )
            first_output = cast("ndarray", output[0])
            return float(first_output.item())

    def _preprocess_mask_lookup(self, bgr_array: "ndarray") -> "ndarray":
        # 0. Store original dimensions
        h, w, _ = bgr_array.shape

        # 1. Flat the image out
        fi = bgr_array.reshape(-1, 3)

        # 3. Map to indices and reshape
        mask_images: "ndarray" = self.lookup_array[fi[:, 0], fi[:, 1], fi[:, 2]]
        mask_images: "ndarray" = mask_images.reshape(h, w).astype(bool)

        # return the binary mask
        return mask_images

    def _tile_image_even_overlap_improved(self, image: "ndarray"):
        """
        Parameters:
        - image: np.ndarray (H, W, C) or (H, W)

        Returns:
        - tiles: list of np.ndarray
        - coordinates: list of tuples (x1, y1, x2, y2)
        """

        # original size
        img_h, img_w = image.shape[:2]
        tile_h = 512
        tile_w = 512

        # 1. Pad upfront if ANY dimension is smaller than the target tile size
        pad_h = max(0, tile_h - img_h)
        pad_w = max(0, tile_w - img_w)

        if pad_h > 0 or pad_w > 0:
            if len(image.shape) == 3:
                image = self.__np.pad(
                    image,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                image = self.__np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

            # Update dimensions after padding so tiling logic works correctly
            img_h, img_w = image.shape[:2]

        # 2. Calculate evenly spaced starting coordinates for Y (rows)
        num_tiles_h = math.ceil(img_h / tile_h)
        if num_tiles_h == 1:
            y_starts = [0]
        else:
            y_starts = self.__np.round(self.__np.linspace(0, img_h - tile_h, num_tiles_h)).astype(int)

        # 3. Calculate evenly spaced starting coordinates for X (columns)
        num_tiles_w = math.ceil(img_w / tile_w)
        if num_tiles_w == 1:
            x_starts = [0]
        else:
            x_starts = self.__np.round(self.__np.linspace(0, img_w - tile_w, num_tiles_w)).astype(int)

        tiles = []
        coordinates = []

        # 4. Slice out the patches using the evenly distributed coordinates
        for y in y_starts:
            for x in x_starts:
                y1, y2 = y, y + tile_h
                x1, x2 = x, x + tile_w

                tile = image[y1:y2, x1:x2]

                tiles.append(tile)
                coordinates.append((int(x1), int(y1), int(x2), int(y2)))

        return tiles, coordinates

    def predict(self, bgr_image: "ndarray", *, fast_method: bool = False) -> tuple[bool, int]:
        if fast_method:
            predicted_value = self.score_fast(bgr_image)
        else:
            rearrange = get_einops_rearrange()

            # 1.1 Binary Lookup
            lookup_mask = self._preprocess_mask_lookup(bgr_image)
            # 1.2 ensure its float32
            bgrc_array: "ndarray" = self.__np.dstack((bgr_image, lookup_mask))
            bgrc_array: "ndarray" = bgrc_array.astype(self.__np.float32)
            # 1.3 Tile the images to a fix size
            image_tiles_list, _ = self._tile_image_even_overlap_improved(bgrc_array)

            # 1.4 stack them to array and ensure formatting
            batched_image_array = self.__np.array(image_tiles_list)  # [N H W C]
            batched_image_array = rearrange(batched_image_array, "n h w c -> n c h w")

            # make the prediction
            combined_predictions: list["ndarray"] = self.model_session.run(
                output_names=["final_output"],
                input_feed={"input_features": batched_image_array},
            )  # type: ignore

            # get the predicted values
            predicted_value: float = float(combined_predictions[0].item())

        is_color = predicted_value >= 0.0
        confidance = abs(predicted_value) / 5.0
        return is_color, int(confidance * 100)


__preload_ogsov_instance__: dict[str, OGSOV] = {}
__preload_ogsox_instance__: dict[str, OGSOX] = {}


def register_ogsov_instance(weights_file: PathLike) -> OGSOV:
    global __preload_ogsov_instance__

    # load the model if not loaded
    if isinstance(weights_file, Path):
        weight_str = str(weights_file.resolve())
    else:
        weight_str = str(weights_file)

    instance = __preload_ogsov_instance__.get(weight_str)
    if instance is None:
        instance = OGSOV(weights_file)
        __preload_ogsov_instance__[weight_str] = instance
    return instance


def register_ogsox_instance(lookup_table_fpath: PathLike, classifier_fpath: PathLike) -> OGSOX:
    global __preload_ogsox_instance__

    # load the model if not loaded
    if isinstance(lookup_table_fpath, Path):
        lookup_str = str(lookup_table_fpath.resolve())
    else:
        lookup_str = str(lookup_table_fpath)

    if isinstance(classifier_fpath, Path):
        classifier_str = str(classifier_fpath.resolve())
    else:
        classifier_str = str(classifier_fpath)

    # hash the path
    combined_path = lookup_str.strip() + classifier_str.strip()
    instance = __preload_ogsox_instance__.get(combined_path)
    if instance is None:
        instance = OGSOX(lookup_table_fpath, classifier_fpath)
        __preload_ogsox_instance__[combined_path] = instance
    return instance


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


def detect_image_color_ml(img_bytes: bytes, *, kernel: MangaColorInference, fast_method: bool = False) -> DetectedColor:
    """
    ML-version of color detection, more accurate but slower than the basic one.
    """

    import numpy as np

    image = Image.open(BytesIO(img_bytes))
    if (detected := detect_image_color(image)) and not detected.is_color:
        image.close()
        return detected  # if the basic detection already says it's not color, we can skip the ML-based detection

    rgb_array = np.array(image.convert("RGB"))
    bgr_array = rgb_array[:, :, ::-1]

    # predict the colorfulness
    is_color, confidence = kernel.predict(bgr_array, fast_method=fast_method)
    image.close()
    return DetectedColor(is_color, confidence, "ML-based detection", is_color)
