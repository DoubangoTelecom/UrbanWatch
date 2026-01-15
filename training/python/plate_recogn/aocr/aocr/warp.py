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

import math
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def get_flip_matrix(prob=0.5):
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


def get_perspective_matrix(perspective=0.0):
    """

    :param perspective:
    :return:
    """
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    return P


def get_rotation_matrix(degree=0.0):
    """

    :param degree:
    :return:
    """
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
    return R


def get_scale_matrix(ratio=(1, 1)):
    """

    :param ratio:
    """
    Scl = np.eye(3)
    scale = random.uniform(*ratio)
    Scl[0, 0] *= scale
    Scl[1, 1] *= scale
    return Scl


def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
    """

    :param width_ratio:
    :param height_ratio:
    """
    Str = np.eye(3)
    Str[0, 0] *= random.uniform(*width_ratio)
    Str[1, 1] *= random.uniform(*height_ratio)
    return Str


def get_shear_matrix(degree):
    """

    :param degree:
    :return:
    """
    Sh = np.eye(3)
    Sh[0, 1] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # x shear (deg)
    Sh[1, 0] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # y shear (deg)
    return Sh


def get_translate_matrix(translate, width, height):
    """

    :param translate:
    :return:
    """
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation
    return T

def get_minimum_dst_shape(
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    divisible: Optional[int] = None,
) -> Tuple[int, int]:
    """Calculate minimum dst shape"""
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape

    if src_w / src_h < dst_w / dst_h:
        ratio = dst_h / src_h
    else:
        ratio = dst_w / src_w

    dst_w = int(ratio * src_w)
    dst_h = int(ratio * src_h)

    if divisible and divisible > 0:
        dst_w = max(divisible, int((dst_w + divisible - 1) // divisible * divisible))
        dst_h = max(divisible, int((dst_h + divisible - 1) // divisible * divisible))
    return dst_w, dst_h

class ShapeTransform:
    """Shape transforms including resize, random perspective, random scale,
    random stretch, random rotation, random shear, random translate,
    and random flip.

    Args:
        perspective: Random perspective factor.
        scale: Random scale ratio.
        stretch: Width and height stretch ratio range.
        rotation: Random rotate degree.
        shear: Random shear degree.
        translate: Random translate ratio.
        flip: Random flip probability.
    """

    def __init__(
        self,
        perspective: float = 0.0,
        scale: Tuple[int, int] = (1, 1),
        stretch: Tuple = ((1, 1), (1, 1)),
        rotation: float = 0.0,
        shear: float = 0.0,
        translate: float = 0.0,
        flip: float = 0.0,
        **kwargs
    ):
        self.perspective = perspective
        self.scale_ratio = scale
        self.stretch_ratio = stretch
        self.rotation_degree = rotation
        self.shear_degree = shear
        self.flip_prob = flip
        self.translate_ratio = translate

    def __call__(self, raw_img):
        height = raw_img.shape[0]  # shape(h,w,c)
        width = raw_img.shape[1]

        # center
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2

        P = get_perspective_matrix(self.perspective)
        C = P @ C

        Scl = get_scale_matrix(self.scale_ratio)
        C = Scl @ C

        Str = get_stretch_matrix(*self.stretch_ratio)
        C = Str @ C

        R = get_rotation_matrix(self.rotation_degree)
        C = R @ C

        Sh = get_shear_matrix(self.shear_degree)
        C = Sh @ C

        F = get_flip_matrix(self.flip_prob)
        C = F @ C

        T = get_translate_matrix(self.translate_ratio, width, height)
        M = T @ C

        img = self._perspective_warp(raw_img, M)
        return img

    # https://stackoverflow.com/a/59741739
    def _perspective_warp(self, image: np.ndarray, transform: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners_aft = cv2.perspectiveTransform(corners_bef, transform)
        xmin = math.floor(corners_aft[:, 0, 0].min())
        ymin = math.floor(corners_aft[:, 0, 1].min())
        xmax = math.ceil(corners_aft[:, 0, 0].max())
        ymax = math.ceil(corners_aft[:, 0, 1].max())
        translate = np.eye(3)
        translate[0, 2] = -xmin
        translate[1, 2] = -ymin
        corrected_transform = np.matmul(translate, transform)
        return cv2.warpPerspective(image, corrected_transform, (math.ceil(xmax - xmin), math.ceil(ymax - ymin)), borderValue=random.randint(0, 255),  borderMode=random.choice([cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE]))
