# Morpholy — Geometric Morphometrics Plugin for YRTools
# Copyright (C) 2024  Zhi-Jie Xu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
广义 Procrustes 分析（Generalized Procrustes Analysis）模块。

仅依赖 numpy 和 scipy，不使用任何外部形态测量学库。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.linalg import svd


def translate_to_centroid(landmarks: np.ndarray) -> np.ndarray:
    """
    将地标集平移到质心（质心移至原点）。

    参数
    ----
    landmarks : (n, 2) 数组

    返回
    ----
    translated : (n, 2) 数组
    """
    centroid = landmarks.mean(axis=0)
    return landmarks - centroid


def centroid_size(landmarks: np.ndarray) -> float:
    """
    计算质心大小（到质心距离的平方和再开方）。

    参数
    ----
    landmarks : (n, 2) 已质心化的地标数组

    返回
    ----
    cs : float
    """
    centered = landmarks - landmarks.mean(axis=0)
    return float(np.sqrt(np.sum(centered ** 2)))


def scale_to_unit_cs(landmarks: np.ndarray) -> np.ndarray:
    """
    将地标缩放到单位质心大小。

    参数
    ----
    landmarks : (n, 2) 数组（通常已质心化）

    返回
    ----
    scaled : (n, 2) 数组
    """
    cs = centroid_size(landmarks)
    if cs == 0:
        return landmarks.copy()
    return landmarks / cs


def rotate_to_reference(
    specimen: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """
    通过 SVD 求最优旋转，将 specimen 配准到 reference。

    要求两者均已质心化且缩放到单位质心大小。

    参数
    ----
    specimen  : (n, 2) 待旋转形状
    reference : (n, 2) 参考形状

    返回
    ----
    rotated : (n, 2) 旋转后的 specimen
    """
    # M = reference^T · specimen
    M = reference.T @ specimen
    U, _s, Vt = svd(M)
    # 确保旋转矩阵行列式为 +1（避免反射）
    D = np.eye(2)
    D[1, 1] = np.linalg.det(U @ Vt)
    R = U @ D @ Vt
    return specimen @ R.T


def _procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个已对齐形状之间的 Procrustes 距离。"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def gpa(
    landmarks_list: List[np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[np.ndarray], np.ndarray, List[float], List[float]]:
    """
    全迭代广义 Procrustes 分析。

    参数
    ----
    landmarks_list : 长度为 p 的列表，每个元素为 (n, 2) 地标数组
    max_iter       : 最大迭代次数
    tol            : 收敛容差（均值形状变化量）

    返回
    ----
    aligned_shapes      : 对齐后的形状列表，每个为 (n, 2) 数组
    mean_shape          : (n, 2) 均值形状
    centroid_sizes      : 各标本的质心大小列表（原始尺度）
    procrustes_distances: 各标本到均值形状的 Procrustes 距离列表
    """
    if len(landmarks_list) < 2:
        raise ValueError("GPA 至少需要 2 个标本")

    # 第 0 步：计算并记录原始质心大小
    cs_list = [centroid_size(lm) for lm in landmarks_list]

    # 第 1 步：平移 + 缩放到单位质心大小
    shapes = [scale_to_unit_cs(translate_to_centroid(lm)) for lm in landmarks_list]

    # 第 2 步：以第一个标本为初始参考
    reference = shapes[0].copy()

    prev_mean: np.ndarray | None = None

    for _iteration in range(max_iter):
        # 将所有标本旋转至当前参考
        aligned = [rotate_to_reference(s, reference) for s in shapes]

        # 更新均值形状并重新缩放
        mean = np.mean(aligned, axis=0)
        mean = scale_to_unit_cs(translate_to_centroid(mean))

        # 检查收敛
        if prev_mean is not None:
            delta = float(np.sqrt(np.sum((mean - prev_mean) ** 2)))
            if delta < tol:
                break

        prev_mean = mean.copy()
        reference = mean

    # 最终对齐（使用收敛后的均值形状作参考）
    aligned_shapes = [rotate_to_reference(s, reference) for s in shapes]
    mean_shape = np.mean(aligned_shapes, axis=0)
    mean_shape = scale_to_unit_cs(translate_to_centroid(mean_shape))

    # 计算每个标本到均值形状的 Procrustes 距离
    pd_list = [_procrustes_distance(s, mean_shape) for s in aligned_shapes]

    return aligned_shapes, mean_shape, cs_list, pd_list
