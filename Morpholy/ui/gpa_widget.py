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
GPAWidget — 广义 Procrustes 分析面板。

使用 QThread 异步计算，避免阻塞 UI。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QProgressBar, QLabel, QMessageBox, QHeaderView,
    QGroupBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from Morpholy.core.gpa import gpa
from Morpholy.core.tps_io import Specimen


class _GPAWorker(QThread):
    """后台线程：执行 GPA 计算。"""
    finished = pyqtSignal(list, object, list, list)   # aligned, mean, cs, pd
    error = pyqtSignal(str)

    def __init__(self, landmarks_list: List[np.ndarray]):
        super().__init__()
        self._lm_list = landmarks_list

    def run(self):
        try:
            aligned, mean, cs, pd = gpa(self._lm_list)
            self.finished.emit(aligned, mean, cs, pd)
        except Exception as e:
            self.error.emit(str(e))


class GPAWidget(QWidget):
    """
    GPA 面板：读取 session 中的标本，运行 GPA，展示结果表格。
    """

    def __init__(self, session: Dict, parent: QWidget | None = None):
        super().__init__(parent)
        self._session = session
        self._worker: _GPAWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # 顶部控制区
        ctrl = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run GPA")
        self._btn_run.setFixedHeight(32)
        self._btn_run.clicked.connect(self._run_gpa)
        ctrl.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # 不确定进度（旋转）
        self._progress.setFixedHeight(20)
        self._progress.setVisible(False)
        ctrl.addWidget(self._progress, stretch=1)
        layout.addLayout(ctrl)

        # 状态标签
        self._lbl_status = QLabel("请先在 Digitize 页面加载标本数据，再运行 GPA。")
        self._lbl_status.setWordWrap(True)
        layout.addWidget(self._lbl_status)

        # 结果表格
        grp = QGroupBox("结果：标本 Procrustes 统计")
        grp_layout = QVBoxLayout(grp)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["标本 ID", "质心大小", "Procrustes 距离"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("alternate-background-color: #252535;")
        grp_layout.addWidget(self._table)
        layout.addWidget(grp, stretch=1)

    # ------------------------------------------------------------------
    # GPA 计算
    # ------------------------------------------------------------------

    def _run_gpa(self):
        specimens: List[Specimen] = self._session.get("specimens", [])
        if len(specimens) < 2:
            QMessageBox.warning(
                self, "数据不足",
                "GPA 至少需要 2 个标本。请在 Digitize 页面添加更多标本。"
            )
            return

        lm_list = [spec.landmarks for spec in specimens]
        # 验证地标数量一致
        n_lm = lm_list[0].shape[0]
        for i, lm in enumerate(lm_list):
            if lm.shape[0] != n_lm:
                QMessageBox.critical(
                    self, "错误",
                    f"标本 {i+1} 的地标数 ({lm.shape[0]}) "
                    f"与第一个标本 ({n_lm}) 不一致，无法运行 GPA。"
                )
                return

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._lbl_status.setText("正在运行 GPA……")

        self._worker = _GPAWorker(lm_list)
        self._worker.finished.connect(self._on_gpa_done)
        self._worker.error.connect(self._on_gpa_error)
        self._worker.start()

    def _on_gpa_done(
        self,
        aligned: List[np.ndarray],
        mean: np.ndarray,
        cs: List[float],
        pd: List[float],
    ):
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)

        # 写入 session
        self._session["aligned_shapes"] = aligned
        self._session["mean_shape"] = mean
        self._session["centroid_sizes"] = cs
        self._session["procrustes_distances"] = pd

        specimens: List[Specimen] = self._session.get("specimens", [])
        self._table.setRowCount(len(aligned))
        for i, (al, c, p) in enumerate(zip(aligned, cs, pd)):
            spec_id = specimens[i].specimen_id if i < len(specimens) else str(i + 1)
            self._table.setItem(i, 0, QTableWidgetItem(spec_id))
            self._table.setItem(i, 1, QTableWidgetItem(f"{c:.6f}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{p:.6f}"))

        self._lbl_status.setText(
            f"GPA 完成。{len(aligned)} 个标本已对齐。"
            f"均值形状地标数: {mean.shape[0]}。"
        )

    def _on_gpa_error(self, msg: str):
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        self._lbl_status.setText("GPA 运行失败。")
        QMessageBox.critical(self, "GPA 错误", f"运行 GPA 时出错：\n{msg}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def refresh(self, session: Dict):
        self._session = session
        n = len(session.get("specimens", []))
        self._lbl_status.setText(
            f"会话中有 {n} 个标本。点击 Run GPA 开始计算。"
            if n >= 2 else
            "请先在 Digitize 页面加载标本数据（至少 2 个），再运行 GPA。"
        )
