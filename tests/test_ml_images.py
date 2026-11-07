"""
Tests for the PIL to tensor image pipeline in :mod:`nmanga.ml.images`.

The model always sees RGB; these tests pin down how each source mode is decoded for inference and
what mode comes back out, including the deliberate palette choice: a palette page is decoded to
RGB (or RGBA when it carries transparency) and never requantized onto the old palette.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from nmanga.ml.images import image_to_tensor, tensor_to_image


def _rgb_gradient(size: tuple[int, int] = (8, 6)) -> Image.Image:
    width, height = size
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[..., 0] = (np.arange(width, dtype=np.uint8) * 8)[None, :]
    data[..., 1] = (np.arange(height, dtype=np.uint8) * 8)[:, None]
    data[..., 2] = 64
    return Image.fromarray(data)


def _gray_gradient(size: tuple[int, int] = (8, 6)) -> Image.Image:
    width, height = size
    data = np.zeros((height, width), dtype=np.uint8)
    data[:] = (np.arange(width, dtype=np.uint8) * 4)[None, :]
    data += (np.arange(height, dtype=np.uint8) * 8)[:, None]
    return Image.fromarray(data)


def _palette_image() -> Image.Image:
    image = Image.new("P", (4, 4), 0)
    image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0, 0, 0] * 253)
    image.putdata([0, 1, 2, 0] * 4)
    return image


def _repeat_runner(scale: int):
    """A nearest neighbour upsampler: the colour of every pixel is predictable."""

    def run(tile: torch.Tensor) -> torch.Tensor:
        return tile.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)

    return run


def _upscaled(array: np.ndarray, scale: int) -> np.ndarray:
    return array.repeat(scale, axis=0).repeat(scale, axis=1)


def test_image_to_tensor_is_rgb_float32_nchw():
    image = _rgb_gradient()

    tensor = image_to_tensor(image)

    assert tensor.shape == (1, 3, 6, 8)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    expected = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


def test_grayscale_is_expanded_to_three_channels():
    tensor = image_to_tensor(_gray_gradient())

    assert tensor.shape == (1, 3, 6, 8)
    torch.testing.assert_close(tensor[:, 0], tensor[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(tensor[:, 1], tensor[:, 2], rtol=0, atol=0)


def test_alpha_is_not_fed_to_the_model():
    rgba = _rgb_gradient().convert("RGBA")

    tensor = image_to_tensor(rgba)

    assert tensor.shape == (1, 3, 6, 8)
    torch.testing.assert_close(tensor, image_to_tensor(_rgb_gradient()), rtol=0, atol=0)


def test_grayscale_page_round_trips_as_l():
    image = _gray_gradient()
    prediction = _repeat_runner(2)(image_to_tensor(image))

    output = tensor_to_image(prediction, image)

    assert output.mode == "L"
    assert output.size == (16, 12)
    np.testing.assert_array_equal(np.asarray(output), _upscaled(np.asarray(image), 2))


def test_bilevel_page_comes_back_as_l():
    image = _gray_gradient().convert("1")
    prediction = _repeat_runner(1)(image_to_tensor(image))

    output = tensor_to_image(prediction, image)

    assert output.mode == "L"
    assert output.size == image.size
    np.testing.assert_array_equal(np.asarray(output), np.asarray(image.convert("L")))


def test_rgba_keeps_and_resizes_the_alpha_channel():
    base = _rgb_gradient().convert("RGBA")
    alpha = Image.linear_gradient("L").resize(base.size, Image.Resampling.BILINEAR)
    base.putalpha(alpha)
    prediction = _repeat_runner(2)(image_to_tensor(base))

    output = tensor_to_image(prediction, base)

    assert output.mode == "RGBA"
    assert output.size == (16, 12)
    expected_alpha = alpha.resize(output.size, Image.Resampling.BICUBIC)
    np.testing.assert_array_equal(np.asarray(output.getchannel("A")), np.asarray(expected_alpha))
    np.testing.assert_array_equal(np.asarray(output.convert("RGB")), _upscaled(np.asarray(base.convert("RGB")), 2))


def test_la_page_round_trips_as_la():
    alpha = Image.linear_gradient("L").resize((8, 6), Image.Resampling.BILINEAR)
    image = Image.merge("LA", [_gray_gradient(), alpha])
    prediction = _repeat_runner(2)(image_to_tensor(image))

    output = tensor_to_image(prediction, image)

    assert output.mode == "LA"
    assert output.size == (16, 12)
    expected_alpha = alpha.resize(output.size, Image.Resampling.BICUBIC)
    np.testing.assert_array_equal(np.asarray(output.getchannel("A")), np.asarray(expected_alpha))


def test_palette_input_is_decoded_to_rgb_not_requantized():
    image = _palette_image()
    prediction = _repeat_runner(2)(image_to_tensor(image))

    output = tensor_to_image(prediction, image)

    assert output.mode == "RGB"
    assert output.getpalette() is None
    assert output.size == (8, 8)
    np.testing.assert_array_equal(np.asarray(output), _upscaled(np.asarray(image.convert("RGB")), 2))


def test_palette_transparency_is_decoded_to_rgba():
    image = _palette_image()
    image.info["transparency"] = 0
    prediction = _repeat_runner(2)(image_to_tensor(image))

    output = tensor_to_image(prediction, image)

    decoded = image.convert("RGBA")
    assert output.mode == "RGBA"
    assert output.size == (8, 8)
    expected_alpha = decoded.getchannel("A").resize(output.size, Image.Resampling.BICUBIC)
    np.testing.assert_array_equal(np.asarray(output.getchannel("A")), np.asarray(expected_alpha))
    np.testing.assert_array_equal(np.asarray(output.convert("RGB")), _upscaled(np.asarray(decoded.convert("RGB")), 2))


def test_output_size_follows_the_prediction():
    template = _rgb_gradient((4, 3))

    output = tensor_to_image(torch.zeros(1, 3, 10, 14), template)

    assert output.mode == "RGB"
    assert output.size == (14, 10)


def test_values_are_clipped_and_rounded():
    values = torch.tensor([0.0, 0.25, 0.5, 1.0, -0.5, 1.5])
    prediction = values.view(1, 1, 1, 6).expand(1, 3, 1, 6).contiguous()

    output = tensor_to_image(prediction, Image.new("RGB", (6, 1)))

    assert [output.getpixel((index, 0)) for index in range(6)] == [
        (0, 0, 0),
        (64, 64, 64),
        (128, 128, 128),
        (255, 255, 255),
        (0, 0, 0),
        (255, 255, 255),
    ]


def test_template_is_not_modified():
    template = _rgb_gradient((8, 6)).convert("RGBA")
    before = np.asarray(template).copy()

    tensor_to_image(torch.zeros(1, 3, 4, 4), template)

    assert template.mode == "RGBA"
    assert template.size == (8, 6)
    np.testing.assert_array_equal(np.asarray(template), before)


@pytest.mark.parametrize("shape", [(3, 4, 4), (1, 4, 4, 4), (2, 3, 4, 4)])
def test_tensor_to_image_rejects_non_rgb_nchw(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="NCHW"):
        tensor_to_image(torch.zeros(*shape), Image.new("RGB", (4, 4)))
