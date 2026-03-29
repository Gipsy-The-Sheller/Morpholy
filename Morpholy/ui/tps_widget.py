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
TPSWidget — 薄板样条变形网格可视化面板。

从会话中选择参考标本和目标标本，展示 TPS 变形场。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QMessageBox, QFileDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Morpholy.core.tps_transform import tps_warp_grid, compute_tps_weights
from Morpholy.core.tps_io import Specimen


class TPSWidget(QWidget):
    """
    TPS 变形网格可视化面板。

    选择参考标本与目标标本，渲染从参考到目标的 TPS 变形场，
    并显示地标点位置叠加。
    """

    def __init__(self, session: Dict, parent: QWidget | None = None):
        super().__init__(parent)
        self._session = session
        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 控制栏
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("参考标本:"))
        self._cmb_ref = QComboBox()
        self._cmb_ref.setMinimumWidth(140)
        ctrl.addWidget(self._cmb_ref)

        ctrl.addWidget(QLabel("目标标本:"))
        self._cmb_tgt = QComboBox()
        self._cmb_tgt.setMinimumWidth(140)
        ctrl.addWidget(self._cmb_tgt)

        self._btn_plot = QPushButton("📊 绘制变形网格")
        self._btn_plot.clicked.connect(self._plot_tps)
        ctrl.addWidget(self._btn_plot)

        ctrl.addStretch()

        self._btn_export = QPushButton("💾 导出 PNG")
        self._btn_export.clicked.connect(self._export_png)
        ctrl.addWidget(self._btn_export)

        layout.addLayout(ctrl)

        # 状态标签
        self._lbl_status = QLabel("请先完成 GPA 以获得对齐形状，再使用此面板。")
        self._lbl_status.setWordWrap(True)
        layout.addWidget(self._lbl_status)

        # matplotlib 画布
        self._fig = Figure(figsize=(8, 6), facecolor="#1e1e2e")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas, stretch=1)

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------

    def _get_shapes(self) -> Optional[List[np.ndarray]]:
        """优先使用对齐形状，否则使用原始形状。"""
        aligned = self._session.get("aligned_shapes")
        if aligned:
            return aligned
        specimens: List[Specimen] = self._session.get("specimens", [])
        if specimens:
            return [s.landmarks for s in specimens]
        return None

    def _get_labels(self) -> List[str]:
        """获取标本标签列表。"""
        specimens: List[Specimen] = self._session.get("specimens", [])
        aligned = self._session.get("aligned_shapes")
        n = len(aligned) if aligned else len(specimens)
        labels = []
        for i in range(n):
            if i < len(specimens) and specimens[i].specimen_id:
                labels.append(specimens[i].specimen_id)
            else:
                labels.append(f"Specimen {i + 1}")
        return labels

    def _plot_tps(self):
        shapes = self._get_shapes()
        if shapes is None or len(shapes) < 2:
            QMessageBox.warning(self, "数据不足", "请先完成 GPA 或加载至少 2 个标本。")
            return

        ref_idx = self._cmb_ref.currentIndex()
        tgt_idx = self._cmb_tgt.currentIndex()

        if ref_idx < 0 or tgt_idx < 0:
            QMessageBox.warning(self, "选择无效", "请选择有效的参考与目标标本。")
            return
        if ref_idx == tgt_idx:
            QMessageBox.warning(self, "选择无效", "参考与目标标本不能相同。")
            return

        source = shapes[ref_idx]
        target = shapes[tgt_idx]

        if source.shape != target.shape:
            QMessageBox.critical(self, "错误", "两个标本的地标数量不一致。")
            return

        try:
            gx, gy, gwx, gwy = tps_warp_grid(source, target, grid_density=20)
        except Exception as e:
            QMessageBox.critical(self, "TPS 错误", f"计算变形网格失败：\n{e}")
            return

        self._fig.clf()
        ax = self._fig.add_subplot(111, facecolor="#252535")

        # 参考网格（浅灰，虚线）
        for row_i in range(gx.shape[0]):
            ax.plot(gx[row_i], gy[row_i], color="#555", lw=0.5, ls="--")
        for col_j in range(gx.shape[1]):
            ax.plot(gx[:, col_j], gy[:, col_j], color="#555", lw=0.5, ls="--")

        # 变形网格（蓝色）
        for row_i in range(gwx.shape[0]):
            ax.plot(gwx[row_i], gwy[row_i], color="#5599ff", lw=0.8)
        for col_j in range(gwx.shape[1]):
            ax.plot(gwx[:, col_j], gwy[:, col_j], color="#5599ff", lw=0.8)

        # 地标点
        ax.scatter(source[:, 0], source[:, 1], c="#aaaaaa", s=40,
                   zorder=6, label="参考地标", marker="o")
        ax.scatter(target[:, 0], target[:, 1], c="#ff9966", s=40,
                   zorder=6, label="目标地标", marker="^")

        labels = self._get_labels()
        ax.set_title(
            f"TPS 变形: {labels[ref_idx]} → {labels[tgt_idx]}",
            color="#cdd6f4"
        )
        ax.set_aspect("equal")
        ax.tick_params(colors="#ccc")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        ax.legend(facecolor="#2b2b3b", edgecolor="#555", labelcolor="#ccc", fontsize=8)

        self._fig.tight_layout()
        self._canvas.draw()
        self._lbl_status.setText(
            f"已绘制 {labels[ref_idx]} → {labels[tgt_idx]} 的 TPS 变形网格。"
        )

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出图像", "tps_deformation.png",
            "PNG 图像 (*.png);;All Files (*)"
        )
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=150, facecolor=self._fig.get_facecolor())
            QMessageBox.information(self, "成功", f"已保存至：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", str(e))

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def refresh(self, session: Dict):
        self._session = session
        shapes = self._get_shapes()
        labels = self._get_labels()

        for cmb in (self._cmb_ref, self._cmb_tgt):
            cmb.clear()
            cmb.addItems(labels)

        if shapes and len(shapes) >= 2:
            self._cmb_tgt.setCurrentIndex(1)
            self._lbl_status.setText(f"已加载 {len(shapes)} 个形状，可选择参考和目标标本。")
        else:
            self._lbl_status.setText("请先完成 GPA 或加载至少 2 个标本，再使用此面板。")
