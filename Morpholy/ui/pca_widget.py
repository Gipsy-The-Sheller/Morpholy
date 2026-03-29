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
PCAWidget — 形状主成分分析面板。

散点图使用 matplotlib FigureCanvasQTAgg，支持按群组着色与形变网格预览。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QMessageBox, QGroupBox, QSizePolicy, QSplitter,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Morpholy.core.pca import shape_pca, reconstruct_shape
from Morpholy.core.tps_transform import tps_warp_grid


class PCAWidget(QWidget):
    """
    PCA 面板：
    - 运行 PCA、散点图（PC1 vs PC2）按群组着色
    - 选择 PC，显示 ±2SD 形变网格
    """

    def __init__(self, session: Dict, parent: QWidget | None = None):
        super().__init__(parent)
        self._session = session
        # PCA 结果缓存
        self._scores: Optional[np.ndarray] = None
        self._loadings: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._evr: Optional[np.ndarray] = None
        self._mean_shape: Optional[np.ndarray] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # 顶部控制
        ctrl = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run PCA")
        self._btn_run.clicked.connect(self._run_pca)
        ctrl.addWidget(self._btn_run)

        ctrl.addWidget(QLabel("显示 PC:"))
        self._cmb_pcx = QComboBox()
        self._cmb_pcx.setMinimumWidth(70)
        ctrl.addWidget(self._cmb_pcx)
        ctrl.addWidget(QLabel("vs"))
        self._cmb_pcy = QComboBox()
        self._cmb_pcy.setMinimumWidth(70)
        ctrl.addWidget(self._cmb_pcy)

        self._btn_plot = QPushButton("🔄 刷新散点图")
        self._btn_plot.clicked.connect(self._update_scatter)
        ctrl.addWidget(self._btn_plot)

        ctrl.addStretch()

        ctrl.addWidget(QLabel("形变预览 PC:"))
        self._cmb_pc_deform = QComboBox()
        self._cmb_pc_deform.setMinimumWidth(70)
        ctrl.addWidget(self._cmb_pc_deform)

        self._btn_deform = QPushButton("📊 显示形变网格")
        self._btn_deform.clicked.connect(self._show_deform)
        ctrl.addWidget(self._btn_deform)

        layout.addLayout(ctrl)

        # 状态标签
        self._lbl_status = QLabel("请先运行 GPA，再运行 PCA。")
        layout.addWidget(self._lbl_status)

        # 图形区域（分割：散点图 | 形变网格）
        splitter = QSplitter(Qt.Horizontal)

        # 散点图
        self._fig_scatter = Figure(figsize=(5, 4), facecolor="#1e1e2e")
        self._canvas_scatter = FigureCanvas(self._fig_scatter)
        self._canvas_scatter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self._canvas_scatter)

        # 形变网格
        self._fig_deform = Figure(figsize=(5, 4), facecolor="#1e1e2e")
        self._canvas_deform = FigureCanvas(self._fig_deform)
        self._canvas_deform.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self._canvas_deform)

        layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # PCA 计算
    # ------------------------------------------------------------------

    def _run_pca(self):
        aligned: Optional[List[np.ndarray]] = self._session.get("aligned_shapes")
        if not aligned or len(aligned) < 2:
            QMessageBox.warning(self, "数据不足", "请先运行 GPA 以获得对齐形状。")
            return
        try:
            scores, loadings, eigenvalues, evr, mean_shape = shape_pca(aligned)
        except Exception as e:
            QMessageBox.critical(self, "PCA 错误", str(e))
            return

        self._scores = scores
        self._loadings = loadings
        self._eigenvalues = eigenvalues
        self._evr = evr
        self._mean_shape = mean_shape

        # 写回 session
        self._session["pca_scores"] = scores
        self._session["pca_loadings"] = loadings
        self._session["pca_eigenvalues"] = eigenvalues
        self._session["pca_evr"] = evr
        self._session["pca_mean_shape"] = mean_shape

        n_pcs = scores.shape[1]
        pc_labels = [f"PC{i+1}" for i in range(n_pcs)]

        for cmb in (self._cmb_pcx, self._cmb_pcy, self._cmb_pc_deform):
            cmb.clear()
            cmb.addItems(pc_labels)
        if n_pcs >= 2:
            self._cmb_pcy.setCurrentIndex(1)

        evr_pct = evr * 100
        summary = "  ".join(
            f"PC{i+1}: {evr_pct[i]:.1f}%" for i in range(min(5, n_pcs))
        )
        self._lbl_status.setText(f"PCA 完成。{summary}")

        self._update_scatter()

    # ------------------------------------------------------------------
    # 散点图
    # ------------------------------------------------------------------

    def _update_scatter(self):
        if self._scores is None:
            return

        pcx = self._cmb_pcx.currentIndex()
        pcy = self._cmb_pcy.currentIndex()

        fig = self._fig_scatter
        fig.clf()
        ax = fig.add_subplot(111, facecolor="#252535")
        ax.tick_params(colors="#ccc")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

        x = self._scores[:, pcx]
        y = self._scores[:, pcy]

        groups = self._session.get("groups", None)
        if groups and len(groups) == len(x):
            unique_g = list(dict.fromkeys(groups))
            colors = cm.tab10(np.linspace(0, 0.9, len(unique_g)))
            for g, c in zip(unique_g, colors):
                mask = np.array([gi == g for gi in groups])
                ax.scatter(x[mask], y[mask], c=[c], label=g, s=50, alpha=0.85)
            ax.legend(
                facecolor="#2b2b3b", edgecolor="#555",
                labelcolor="#ccc", fontsize=8
            )
        else:
            ax.scatter(x, y, c="#5599ff", s=50, alpha=0.85)

        evr_pct = self._evr * 100 if self._evr is not None else None
        xlabel = f"PC{pcx+1}" + (f" ({evr_pct[pcx]:.1f}%)" if evr_pct is not None else "")
        ylabel = f"PC{pcy+1}" + (f" ({evr_pct[pcy]:.1f}%)" if evr_pct is not None else "")
        ax.set_xlabel(xlabel, color="#ccc")
        ax.set_ylabel(ylabel, color="#ccc")
        ax.set_title("PCA 散点图", color="#cdd6f4")

        self._canvas_scatter.draw()

    # ------------------------------------------------------------------
    # 形变网格
    # ------------------------------------------------------------------

    def _show_deform(self):
        if self._scores is None or self._mean_shape is None:
            QMessageBox.warning(self, "无数据", "请先运行 PCA。")
            return

        pc_idx = self._cmb_pc_deform.currentIndex()
        sd = float(np.std(self._scores[:, pc_idx]))

        scores_pos = self._scores[0].copy()
        scores_neg = self._scores[0].copy()
        scores_pos[pc_idx] = 2 * sd
        scores_neg[pc_idx] = -2 * sd

        shape_pos = reconstruct_shape(
            self._mean_shape, self._loadings, scores_pos, [pc_idx]
        )
        shape_neg = reconstruct_shape(
            self._mean_shape, self._loadings, scores_neg, [pc_idx]
        )

        fig = self._fig_deform
        fig.clf()
        fig.patch.set_facecolor("#1e1e2e")

        ax_pos = fig.add_subplot(121, facecolor="#252535")
        ax_neg = fig.add_subplot(122, facecolor="#252535")

        for ax, shape, title, color in [
            (ax_pos, shape_pos, f"PC{pc_idx+1} +2SD", "#ff9966"),
            (ax_neg, shape_neg, f"PC{pc_idx+1} −2SD", "#66aaff"),
        ]:
            try:
                gx, gy, gwx, gwy = tps_warp_grid(self._mean_shape, shape, grid_density=15)
                # 绘制变形网格
                for row_i in range(gx.shape[0]):
                    ax.plot(gwx[row_i], gwy[row_i], color="#555", lw=0.6)
                for col_j in range(gx.shape[1]):
                    ax.plot(gwx[:, col_j], gwy[:, col_j], color="#555", lw=0.6)
            except Exception:
                pass
            ax.scatter(shape[:, 0], shape[:, 1], c=color, s=30, zorder=5)
            ax.scatter(
                self._mean_shape[:, 0], self._mean_shape[:, 1],
                c="#aaa", s=15, zorder=4, alpha=0.5
            )
            ax.set_title(title, color="#cdd6f4", fontsize=9)
            ax.tick_params(colors="#ccc", labelsize=7)
            ax.set_aspect("equal")

        fig.tight_layout(pad=1.0)
        self._canvas_deform.draw()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def refresh(self, session: Dict):
        self._session = session
        n = len(session.get("aligned_shapes", []))
        self._lbl_status.setText(
            f"会话中有 {n} 个对齐形状。点击 Run PCA 开始分析。"
            if n >= 2 else "请先运行 GPA，再运行 PCA。"
        )
        # 如果 session 中已有 PCA 结果，恢复缓存并刷新图形
        if "pca_scores" in session:
            self._scores = session["pca_scores"]
            self._loadings = session["pca_loadings"]
            self._eigenvalues = session["pca_eigenvalues"]
            self._evr = session["pca_evr"]
            self._mean_shape = session["pca_mean_shape"]
            self._update_scatter()
