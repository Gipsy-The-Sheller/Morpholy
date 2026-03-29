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
统计检验模块：回归分析与群组差异置换检验。

不依赖外部形态测量学库，仅使用 numpy/scipy。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _vectorize_shapes(shape_matrix: np.ndarray) -> np.ndarray:
    """
    将 (p, n, 2) 或 (p, 2n) 形状矩阵展平为 (p, 2n)。
    """
    if shape_matrix.ndim == 3:
        p = shape_matrix.shape[0]
        return shape_matrix.reshape(p, -1)
    return shape_matrix


def _ss_total(X: np.ndarray) -> float:
    """总平方和（中心化后）。"""
    X_c = X - X.mean(axis=0)
    return float(np.sum(X_c ** 2))


def _ss_residual(X: np.ndarray, fitted: np.ndarray) -> float:
    """残差平方和。"""
    return float(np.sum((X - fitted) ** 2))


# ---------------------------------------------------------------------------
# Procrustes 回归（形状 ~ 连续变量，如质心大小）
# ---------------------------------------------------------------------------

def procrustes_regression(
    shape_matrix: np.ndarray,
    covariate: np.ndarray,
    n_permutations: int = 999,
    random_state: int | None = None,
) -> Tuple[np.ndarray, float, float, float]:
    """
    形状对连续协变量（如质心大小）的回归分析。

    通过置换检验评估显著性（999次置换）。

    参数
    ----
    shape_matrix  : (p, n, 2) 或 (p, 2n) 对齐形状矩阵
    covariate     : (p,) 连续协变量向量（如质心大小）
    n_permutations: 置换次数
    random_state  : 随机种子（可选）

    返回
    ----
    coefficients : (2n,) 回归系数（形状变化向量）
    r_squared    : 决定系数 R²
    f_stat       : F 统计量
    p_value      : 置换检验 p 值
    """
    rng = np.random.default_rng(random_state)

    X = _vectorize_shapes(shape_matrix)  # (p, 2n)
    p, k = X.shape
    y = np.asarray(covariate, dtype=np.float64)  # (p,)

    if len(y) != p:
        raise ValueError(f"协变量长度 ({len(y)}) 与标本数 ({p}) 不匹配")

    # 构建设计矩阵 [1, covariate]
    design = np.column_stack([np.ones(p), y])  # (p, 2)

    # OLS: B = (D^T D)^{-1} D^T X
    DTD_inv = np.linalg.inv(design.T @ design)
    B = DTD_inv @ design.T @ X  # (2, 2n)
    fitted = design @ B  # (p, 2n)

    ss_tot = _ss_total(X)
    ss_res = _ss_residual(X, fitted)
    ss_reg = ss_tot - ss_res

    # R²
    r_squared = (ss_reg / ss_tot) if ss_tot > 1e-12 else 0.0
    r_squared = float(max(0.0, min(1.0, r_squared)))

    # F 统计量（1 个预测变量，共 k 个响应变量合并）
    df_reg = 1
    df_res = p - 2
    if df_res <= 0 or ss_res == 0:
        f_stat = 0.0
    else:
        f_stat = (ss_reg / df_reg) / (ss_res / df_res)

    # 置换检验
    count_extreme = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        design_perm = np.column_stack([np.ones(p), y_perm])
        B_perm = DTD_inv_for(design_perm) @ design_perm.T @ X
        fitted_perm = design_perm @ B_perm
        ss_res_perm = _ss_residual(X, fitted_perm)
        ss_reg_perm = ss_tot - ss_res_perm
        if df_res > 0 and ss_res_perm > 0:
            f_perm = (ss_reg_perm / df_reg) / (ss_res_perm / df_res)
        else:
            f_perm = 0.0
        if f_perm >= f_stat:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    # 回归系数（斜率，即协变量对应的形状变化向量）
    coefficients = B[1]  # (2n,)

    return coefficients, float(r_squared), float(f_stat), float(p_value)


def DTD_inv_for(design: np.ndarray) -> np.ndarray:
    """安全计算 (D^T D)^{-1}，失败时用 lstsq 伪逆。"""
    try:
        return np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(design.T @ design)


# ---------------------------------------------------------------------------
# 群组差异置换检验（类 MANOVA）
# ---------------------------------------------------------------------------

def group_difference(
    shape_matrix: np.ndarray,
    groups: List[str],
    n_permutations: int = 999,
    random_state: int | None = None,
) -> Tuple[float, float]:
    """
    基于置换检验的群组差异检验（Procrustes ANOVA）。

    使用组间 SS / 组内 SS 作为 F 统计量近似（Pillai's trace 类似量）。

    参数
    ----
    shape_matrix  : (p, n, 2) 或 (p, 2n) 对齐形状矩阵
    groups        : 长度为 p 的群组标签列表
    n_permutations: 置换次数
    random_state  : 随机种子（可选）

    返回
    ----
    f_stat  : F 统计量
    p_value : 置换检验 p 值
    """
    rng = np.random.default_rng(random_state)

    X = _vectorize_shapes(shape_matrix)  # (p, 2n)
    p = X.shape[0]
    groups = list(groups)

    if len(groups) != p:
        raise ValueError(f"群组标签数 ({len(groups)}) 与标本数 ({p}) 不匹配")

    unique_groups = list(dict.fromkeys(groups))
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("至少需要 2 个群组")

    def _f_stat(X: np.ndarray, grp: List[str]) -> float:
        grand_mean = X.mean(axis=0)
        ss_between = 0.0
        ss_within = 0.0
        for g in unique_groups:
            mask = np.array([gi == g for gi in grp])
            Xg = X[mask]
            ng = Xg.shape[0]
            if ng == 0:
                continue
            gm = Xg.mean(axis=0)
            ss_between += ng * float(np.sum((gm - grand_mean) ** 2))
            ss_within += float(np.sum((Xg - gm) ** 2))
        df_b = n_groups - 1
        df_w = p - n_groups
        if df_w <= 0 or ss_within == 0:
            return 0.0
        return (ss_between / df_b) / (ss_within / df_w)

    observed_f = _f_stat(X, groups)

    count_extreme = 0
    grp_arr = np.array(groups)
    for _ in range(n_permutations):
        perm_grp = rng.permutation(grp_arr).tolist()
        f_perm = _f_stat(X, perm_grp)
        if f_perm >= observed_f:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return float(observed_f), float(p_value)
