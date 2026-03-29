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
薄板样条（Thin Plate Spline）变换模块。

纯 numpy 实现，不依赖外部形态测量学库。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _rbf(r: np.ndarray) -> np.ndarray:
    """TPS 径向基函数 U(r) = r² log(r²)，r=0 时为 0。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = r ** 2
        result = np.where(r2 == 0, 0.0, r2 * np.log(r2 + 1e-300))
    return result


def _build_L_matrix(source: np.ndarray) -> np.ndarray:
    """
    构建 TPS L 矩阵（(n+3) × (n+3)）。

    参数
    ----
    source : (n, 2) 源控制点

    返回
    ----
    L : (n+3, n+3) 矩阵
    """
    n = source.shape[0]
    # K 矩阵：n×n，K[i,j] = U(||pi - pj||)
    diff = source[:, np.newaxis, :] - source[np.newaxis, :, :]  # (n, n, 2)
    r = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
    K = _rbf(r)

    # P 矩阵：n×3，[1, x, y]
    P = np.hstack([np.ones((n, 1)), source])

    # 构建 L
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    # 右下角全 0（自然边界条件）
    return L


def compute_tps_weights(
    source: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算从 source 控制点到 target 控制点的 TPS 权重与仿射参数。

    参数
    ----
    source : (n, 2) 源控制点（如参考形状地标）
    target : (n, 2) 目标控制点（如目标形状地标）

    返回
    ----
    weights : (n, 2) 每个控制点的非线性权重
    affine  : (3, 2) 仿射项 [a0, ax, ay] × [x分量, y分量]
    """
    n = source.shape[0]
    L = _build_L_matrix(source)

    # 右侧向量：目标坐标 + 零边界条件
    rhs = np.vstack([target, np.zeros((3, 2))])  # (n+3, 2)

    # 求解线性系统（使用最小二乘以防奇异）
    coef, _res, _rank, _sv = np.linalg.lstsq(L, rhs, rcond=None)

    weights = coef[:n]   # (n, 2)
    affine = coef[n:]    # (3, 2)

    return weights, affine


def tps_interpolate(
    source: np.ndarray,
    weights: np.ndarray,
    affine: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    将 TPS 变换应用到任意点集。

    参数
    ----
    source  : (n, 2) 源控制点
    weights : (n, 2) TPS 权重
    affine  : (3, 2) 仿射参数
    points  : (m, 2) 待变换点坐标

    返回
    ----
    warped : (m, 2) 变换后坐标
    """
    m = points.shape[0]
    n = source.shape[0]

    # 计算每个待变换点到每个控制点的距离
    diff = points[:, np.newaxis, :] - source[np.newaxis, :, :]  # (m, n, 2)
    r = np.sqrt(np.sum(diff ** 2, axis=2))  # (m, n)
    U = _rbf(r)  # (m, n)

    # 仿射部分 P_pts: (m, 3)
    P_pts = np.hstack([np.ones((m, 1)), points])

    # 合并：warped = U·weights + P_pts·affine
    warped = U @ weights + P_pts @ affine  # (m, 2)
    return warped


def tps_warp_grid(
    source: np.ndarray,
    target: np.ndarray,
    grid_density: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 TPS 对规则网格进行变形，用于可视化变形场。

    参数
    ----
    source        : (n, 2) 源控制点
    target        : (n, 2) 目标控制点
    grid_density  : 网格每方向采样点数

    返回
    ----
    grid_x_src  : (grid_density, grid_density) 原始网格 X 坐标
    grid_y_src  : (grid_density, grid_density) 原始网格 Y 坐标
    grid_x_warp : (grid_density, grid_density) 变形后网格 X 坐标
    grid_y_warp : (grid_density, grid_density) 变形后网格 Y 坐标
    """
    weights, affine = compute_tps_weights(source, target)

    # 在 source 范围内建立规则网格
    x_min, y_min = source.min(axis=0) - 0.1
    x_max, y_max = source.max(axis=0) + 0.1

    xs = np.linspace(x_min, x_max, grid_density)
    ys = np.linspace(y_min, y_max, grid_density)
    grid_x_src, grid_y_src = np.meshgrid(xs, ys)

    # 将网格展平后变换
    grid_pts = np.column_stack([grid_x_src.ravel(), grid_y_src.ravel()])
    warped_pts = tps_interpolate(source, weights, affine, grid_pts)

    grid_x_warp = warped_pts[:, 0].reshape(grid_density, grid_density)
    grid_y_warp = warped_pts[:, 1].reshape(grid_density, grid_density)

    return grid_x_src, grid_y_src, grid_x_warp, grid_y_warp
