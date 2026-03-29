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
StatsWidget — 统计检验面板。

包含两个标签页：
  1. Allometry（同速生长回归）：形状对质心大小的回归分析
  2. Group Differences（群组差异）：置换 MANOVA 检验
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QTextEdit, QPlainTextEdit, QMessageBox, QSizePolicy,
    QSplitter,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Morpholy.core.stats import procrustes_regression, group_difference


class StatsWidget(QWidget):
    """
    统计检验面板。
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

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabBar::tab { background:#2b2b3b; color:#aaa; padding:6px 16px; }"
            "QTabBar::tab:selected { background:#3d3d55; color:#cdf; }"
        )
        tabs.addTab(self._build_allometry_tab(), "同速生长回归（Allometry）")
        tabs.addTab(self._build_groups_tab(), "群组差异（Group Differences）")
        layout.addWidget(tabs)

    # ---- Allometry 标签 ---------------------------------------------------

    def _build_allometry_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl = QHBoxLayout()
        self._btn_regress = QPushButton("▶  运行回归分析")
        self._btn_regress.clicked.connect(self._run_regression)
        ctrl.addWidget(self._btn_regress)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Horizontal)

        # 左：回归散点图
        self._fig_reg = Figure(figsize=(5, 4), facecolor="#1e1e2e")
        self._canvas_reg = FigureCanvas(self._fig_reg)
        self._canvas_reg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self._canvas_reg)

        # 右：文字结果
        self._txt_reg = QTextEdit()
        self._txt_reg.setReadOnly(True)
        self._txt_reg.setPlaceholderText("回归结果将显示在此处……")
        self._txt_reg.setMaximumWidth(320)
        splitter.addWidget(self._txt_reg)

        layout.addWidget(splitter, stretch=1)
        return w

    # ---- Groups 标签 -------------------------------------------------------

    def _build_groups_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(QLabel(
            "每行输入一个标本的群组标签（行数需与标本数一致）："
        ))

        self._txt_groups = QPlainTextEdit()
        self._txt_groups.setPlaceholderText(
            "例如：\nGroupA\nGroupA\nGroupB\nGroupB\n…"
        )
        self._txt_groups.setMaximumHeight(160)
        layout.addWidget(self._txt_groups)

        ctrl = QHBoxLayout()
        self._btn_groups = QPushButton("▶  运行群组差异检验")
        self._btn_groups.clicked.connect(self._run_group_diff)
        ctrl.addWidget(self._btn_groups)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._txt_grp_result = QTextEdit()
        self._txt_grp_result.setReadOnly(True)
        self._txt_grp_result.setPlaceholderText("检验结果将显示在此处……")
        layout.addWidget(self._txt_grp_result, stretch=1)

        return w

    # ------------------------------------------------------------------
    # 同速生长回归
    # ------------------------------------------------------------------

    def _run_regression(self):
        aligned = self._session.get("aligned_shapes")
        cs_list = self._session.get("centroid_sizes")

        if not aligned or len(aligned) < 3:
            QMessageBox.warning(
                self, "数据不足",
                "回归分析至少需要 3 个对齐形状（请先运行 GPA）。"
            )
            return
        if not cs_list or len(cs_list) != len(aligned):
            QMessageBox.warning(self, "数据不足", "缺少质心大小数据，请先运行 GPA。")
            return

        shape_matrix = np.array(aligned)  # (p, n, 2)
        covariate = np.log(np.array(cs_list))  # 对质心大小取对数（常见做法）

        try:
            coef, r2, f_stat, p_val = procrustes_regression(
                shape_matrix, covariate
            )
        except Exception as e:
            QMessageBox.critical(self, "回归错误", str(e))
            return

        # 写入 session
        self._session["regression_coef"] = coef
        self._session["regression_r2"] = r2

        # 散点图：Procrustes 距离 vs log(CS)
        pd_list = self._session.get("procrustes_distances", [])
        self._fig_reg.clf()
        ax = self._fig_reg.add_subplot(111, facecolor="#252535")

        if pd_list and len(pd_list) == len(covariate):
            ax.scatter(covariate, pd_list, c="#5599ff", s=40, alpha=0.85)
            # 拟合趋势线
            z = np.polyfit(covariate, pd_list, 1)
            x_line = np.linspace(covariate.min(), covariate.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), c="#ff9966", lw=1.5)
            ax.set_ylabel("Procrustes 距离", color="#ccc")
        else:
            # 若无 PD，绘制形状空间第一主成分得分 vs log(CS)
            p_arr = np.array(aligned).reshape(len(aligned), -1)
            pc1 = (p_arr - p_arr.mean(0)) @ coef / (np.linalg.norm(coef) + 1e-12)
            ax.scatter(covariate, pc1, c="#5599ff", s=40, alpha=0.85)
            z = np.polyfit(covariate, pc1, 1)
            x_line = np.linspace(covariate.min(), covariate.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), c="#ff9966", lw=1.5)
            ax.set_ylabel("形状投影（回归方向）", color="#ccc")

        ax.set_xlabel("log(质心大小)", color="#ccc")
        ax.set_title("同速生长回归", color="#cdd6f4")
        ax.tick_params(colors="#ccc")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        self._fig_reg.tight_layout()
        self._canvas_reg.draw()

        # 文字结果
        result = (
            f"<b>回归分析结果</b><br><br>"
            f"R² = {r2:.4f}<br>"
            f"F 统计量 = {f_stat:.4f}<br>"
            f"p 值（置换检验, 999次）= {p_val:.4f}<br><br>"
            f"{'<font color=\"#66ff99\"><b>显著 (p < 0.05)</b></font>' if p_val < 0.05 else '<font color=\"#ff9966\">不显著 (p ≥ 0.05)</font>'}"
        )
        self._txt_reg.setHtml(result)

    # ------------------------------------------------------------------
    # 群组差异检验
    # ------------------------------------------------------------------

    def _run_group_diff(self):
        aligned = self._session.get("aligned_shapes")
        if not aligned or len(aligned) < 4:
            QMessageBox.warning(
                self, "数据不足",
                "群组差异检验至少需要 4 个对齐形状（请先运行 GPA）。"
            )
            return

        raw_text = self._txt_groups.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "无输入", "请输入群组标签（每行一个）。")
            return

        group_labels = [line.strip() for line in raw_text.splitlines() if line.strip()]

        if len(group_labels) != len(aligned):
            QMessageBox.warning(
                self, "行数不匹配",
                f"输入了 {len(group_labels)} 个标签，"
                f"但会话中有 {len(aligned)} 个标本。"
            )
            return

        unique_grp = list(dict.fromkeys(group_labels))
        if len(unique_grp) < 2:
            QMessageBox.warning(self, "群组不足", "至少需要 2 个不同群组。")
            return

        shape_matrix = np.array(aligned)

        try:
            f_stat, p_val = group_difference(shape_matrix, group_labels)
        except Exception as e:
            QMessageBox.critical(self, "检验错误", str(e))
            return

        # 写入 session
        self._session["groups"] = group_labels

        # 显示结果
        group_summary = ", ".join(
            f"{g}: {group_labels.count(g)}" for g in unique_grp
        )
        result = (
            f"<b>群组差异置换检验结果</b><br><br>"
            f"群组分布: {group_summary}<br>"
            f"F 统计量 = {f_stat:.4f}<br>"
            f"p 值（置换检验, 999次）= {p_val:.4f}<br><br>"
            f"{'<font color=\"#66ff99\"><b>显著 (p < 0.05)</b></font>' if p_val < 0.05 else '<font color=\"#ff9966\">不显著 (p ≥ 0.05)</font>'}"
            f"<br><br><i>提示：群组信息已同步至 PCA 散点图（切换到 PCA 页面刷新）。</i>"
        )
        self._txt_grp_result.setHtml(result)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def refresh(self, session: Dict):
        self._session = session
        n = len(session.get("aligned_shapes", []))
        if n < 3:
            self._txt_reg.setPlaceholderText(
                f"请先运行 GPA（当前 {n} 个对齐形状，需 ≥ 3）……"
            )
