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
形状 PCA 模块 — 对 Procrustes 对齐坐标进行主成分分析。

使用 numpy 实现，不依赖外部形态测量学库。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def shape_pca(
    aligned_shapes: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对向量化的对齐形状坐标进行 PCA。

    参数
    ----
    aligned_shapes : 长度为 p 的列表，每个元素为 (n, 2) 数组

    返回
    ----
    scores                 : (p, k) 得分矩阵，k = min(p-1, 2n)
    loadings               : (2n, k) 载荷矩阵（主成分向量）
    eigenvalues            : (k,) 特征值
    explained_variance_ratio: (k,) 各 PC 解释方差比例
    mean_shape             : (n, 2) 均值形状
    """
    p = len(aligned_shapes)
    if p < 2:
        raise ValueError("PCA 至少需要 2 个标本")

    n = aligned_shapes[0].shape[0]

    # 构造数据矩阵 X: (p, 2n)，按行排列每个标本的展平坐标
    X = np.vstack([s.flatten() for s in aligned_shapes])  # (p, 2n)

    # 均值形状
    mean_vec = X.mean(axis=0)
    mean_shape = mean_vec.reshape(n, 2)

    # 中心化
    X_centered = X - mean_vec  # (p, 2n)

    # SVD 分解（比直接对协方差矩阵特征分解数值更稳定）
    # X_centered = U · S · Vt
    U, s_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 特征值 = s^2 / (p-1)
    eigenvalues = (s_vals ** 2) / (p - 1)

    # 解释方差比例
    total_var = eigenvalues.sum()
    if total_var > 0:
        explained_variance_ratio = eigenvalues / total_var
    else:
        explained_variance_ratio = np.zeros_like(eigenvalues)

    # 载荷矩阵：Vt 的行即为主成分方向，形状 (k, 2n)，转置得 (2n, k)
    loadings = Vt.T  # (2n, k)

    # 得分矩阵: (p, k)
    scores = X_centered @ loadings  # (p, k)

    return scores, loadings, eigenvalues, explained_variance_ratio, mean_shape


def reconstruct_shape(
    mean_shape: np.ndarray,
    loadings: np.ndarray,
    scores: np.ndarray,
    pc_indices: List[int],
) -> np.ndarray:
    """
    从指定 PC 的得分重建形状。

    参数
    ----
    mean_shape : (n, 2) 均值形状
    loadings   : (2n, k) 载荷矩阵
    scores     : (k,) 单个标本的得分向量，或 (p, k) 多个标本
    pc_indices : 要使用的 PC 索引列表（0-based）

    返回
    ----
    shape : (n, 2) 重建形状，或 (p, n, 2) 多标本时
    """
    n = mean_shape.shape[0]
    mean_vec = mean_shape.flatten()

    if scores.ndim == 1:
        # 单标本
        delta = np.zeros(2 * n)
        for idx in pc_indices:
            delta += scores[idx] * loadings[:, idx]
        return (mean_vec + delta).reshape(n, 2)
    else:
        # 多标本 (p, k)
        p = scores.shape[0]
        delta = np.zeros((p, 2 * n))
        for idx in pc_indices:
            delta += np.outer(scores[:, idx], loadings[:, idx])
        reconstructed = mean_vec + delta  # (p, 2n)
        return reconstructed.reshape(p, n, 2)
