# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import cv2
import numpy as np

def normalize(meta, mean, std):
    img = meta["img"].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta["img"] = img
    return meta


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img


def color_aug_and_norm(meta, kwargs):
    img = meta["img"]
    
    # Build sequence
    from imgaug import augmenters as iaa
    sequence = []
    activate_fn = lambda: random.choice(range(3)) == 0 # pick 1/3 only, otherwise tooo slow
    if "average_blur" in kwargs and activate_fn():
        sequence.append(iaa.AverageBlur(k=tuple(kwargs["average_blur"])))  
    if "gaussian_blur" in kwargs and activate_fn():
        sequence.append(iaa.GaussianBlur(sigma=tuple(kwargs["gaussian_blur"])))
    if "motion_blur" in kwargs and activate_fn():
        sequence.append(iaa.MotionBlur(k=kwargs["motion_blur"][0], angle=kwargs["motion_blur"][1:]))
    if "multiply" in kwargs and activate_fn():
        sequence.append(iaa.Multiply(mul=tuple(kwargs["multiply"]), per_channel=random.choice([False, True])))
    if "multiply_hue" in kwargs and activate_fn():
        sequence.append(iaa.MultiplyHue(mul=tuple(kwargs["multiply_hue"])))
    if "multiply_saturation" in kwargs and activate_fn():
        sequence.append(iaa.MultiplySaturation(mul=tuple(kwargs["multiply_saturation"])))
    if "gamma_contrast" in kwargs and activate_fn():
        sequence.append(iaa.GammaContrast(gamma=tuple(kwargs["gamma_contrast"]), per_channel=random.choice([False, True])))
    if "sigmoid_contrast" in kwargs and activate_fn():
        sequence.append(iaa.SigmoidContrast(gain=tuple(kwargs["sigmoid_contrast"][:2]), cutoff=tuple(kwargs["sigmoid_contrast"][2:]), per_channel=random.choice([False, True])))
    if "log_contrast" in kwargs and activate_fn():
        sequence.append(iaa.LogContrast(gain=tuple(kwargs["log_contrast"]), per_channel=random.choice([False, True])))
    if "linear_contrast" in kwargs and activate_fn():
        sequence.append(iaa.LinearContrast(alpha=tuple(kwargs["linear_contrast"]), per_channel=random.choice([False, True])))
    if "emboss" in kwargs and activate_fn():
        sequence.append(iaa.Emboss(alpha=tuple(kwargs["emboss"][:2]), strength=tuple(kwargs["emboss"][2:])))
    if "additive_gaussian_noise" in kwargs and activate_fn():
        sequence.append(iaa.AdditiveGaussianNoise(scale=kwargs["additive_gaussian_noise"], per_channel=random.choice([False, True])))
    if "salt" in kwargs and activate_fn():
        sequence.append(iaa.Salt(p=kwargs["salt"], per_channel=random.choice([False, True])))
    if "pepper" in kwargs and activate_fn():
        sequence.append(iaa.Pepper(p=kwargs["pepper"], per_channel=random.choice([False, True])))

    # Apply transformation
    if len(sequence) > 0:
        transforms = iaa.Sequential(sequence, random_order=True)
        img = transforms(images=[img])[0]
    
    # Normalize image
    img = img.astype(np.float32) / 255
    img = _normalize(img, *kwargs["normalize"])
    meta["img"] = img
    return meta
