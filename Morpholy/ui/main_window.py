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
MorpholyWindow — 主窗口，包含左侧导航栏与右侧堆叠页面。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QStackedWidget, QToolBar, QAction, QMessageBox, QFileDialog,
    QSizePolicy,
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt

from Morpholy.core.tps_io import parse_tps, write_tps, Specimen
from Morpholy.ui.digitizer import DigitizerWidget
from Morpholy.ui.gpa_widget import GPAWidget
from Morpholy.ui.pca_widget import PCAWidget
from Morpholy.ui.tps_widget import TPSWidget
from Morpholy.ui.stats_widget import StatsWidget

_HERE = Path(__file__).parent.parent  # Morpholy 包根目录


class MorpholyWindow(QWidget):
    """
    Morpholy 主窗口。

    左侧为 QListWidget 导航栏，右侧为 QStackedWidget 内容区。
    各页面通过共享的 session 字典交换数据，切换页面时调用 page.refresh(session)。
    """

    PAGE_NAMES = ["Digitize", "GPA", "PCA", "Deformation", "Statistics"]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Morpholy — Geometric Morphometrics")
        self.setMinimumSize(1000, 680)

        # 共享会话数据字典
        self.session: Dict = {}

        self._build_ui()
        self._connect_signals()
        self._apply_style()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 工具栏
        self._toolbar = self._make_toolbar()
        root_layout.addWidget(self._toolbar)

        # 主体区域（侧边栏 + 内容）
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # 左侧导航栏
        self._nav = QListWidget()
        self._nav.setFixedWidth(140)
        self._nav.setFont(QFont("Segoe UI", 10))
        for name in self.PAGE_NAMES:
            item = QListWidgetItem(name)
            item.setTextAlignment(Qt.AlignCenter)
            self._nav.addItem(item)
        self._nav.setCurrentRow(0)

        # 右侧堆叠页
        self._stack = QStackedWidget()
        self._pages = {
            "Digitize": DigitizerWidget(self.session),
            "GPA": GPAWidget(self.session),
            "PCA": PCAWidget(self.session),
            "Deformation": TPSWidget(self.session),
            "Statistics": StatsWidget(self.session),
        }
        for name in self.PAGE_NAMES:
            self._stack.addWidget(self._pages[name])

        body_layout.addWidget(self._nav)
        body_layout.addWidget(self._stack, stretch=1)
        root_layout.addWidget(body, stretch=1)

    def _make_toolbar(self) -> QToolBar:
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            "QToolBar { background: #2b2b3b; border-bottom: 1px solid #444; spacing: 4px; }"
            "QToolButton { color: #ccc; padding: 4px 10px; border-radius: 3px; }"
            "QToolButton:hover { background: #3d3d55; }"
        )

        icon_dir = _HERE / "icons"

        act_open = QAction("📂  Open TPS", self)
        act_open.setToolTip("打开 TPS 文件")
        toolbar.addAction(act_open)
        self._act_open = act_open

        act_save = QAction("💾  Save TPS", self)
        act_save.setToolTip("保存 TPS 文件")
        toolbar.addAction(act_save)
        self._act_save = act_save

        toolbar.addSeparator()

        act_about = QAction("ℹ️  About", self)
        act_about.setToolTip("关于 Morpholy")
        toolbar.addAction(act_about)
        self._act_about = act_about

        return toolbar

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._nav.currentRowChanged.connect(self._on_page_changed)
        self._act_open.triggered.connect(self._open_tps)
        self._act_save.triggered.connect(self._save_tps)
        self._act_about.triggered.connect(self._show_about)

        # DigitizerWidget 保存到 session 后自动刷新 GPA 页
        dig: DigitizerWidget = self._pages["Digitize"]
        dig.session_saved.connect(lambda: self._pages["GPA"].refresh(self.session))

    def _on_page_changed(self, row: int):
        self._stack.setCurrentIndex(row)
        name = self.PAGE_NAMES[row]
        page = self._pages[name]
        if hasattr(page, "refresh"):
            page.refresh(self.session)

    # ------------------------------------------------------------------
    # 工具栏动作
    # ------------------------------------------------------------------

    def _open_tps(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 TPS 文件", "", "TPS Files (*.tps *.TPS);;All Files (*)"
        )
        if not path:
            return
        try:
            specimens = parse_tps(path)
            self.session["specimens"] = specimens
            self.session["tps_path"] = path
            # 同步到 digitizer
            dig: DigitizerWidget = self._pages["Digitize"]
            dig.load_specimens(specimens)
            QMessageBox.information(
                self, "成功", f"已加载 {len(specimens)} 个标本。"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取 TPS 文件：\n{e}")

    def _save_tps(self):
        specimens: list[Specimen] = self.session.get("specimens", [])
        if not specimens:
            QMessageBox.warning(self, "警告", "会话中没有可保存的标本数据。")
            return
        default = self.session.get("tps_path", "output.tps")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 TPS 文件", default,
            "TPS Files (*.tps);;All Files (*)"
        )
        if not path:
            return
        try:
            write_tps(specimens, path)
            self.session["tps_path"] = path
            QMessageBox.information(self, "成功", f"已保存到：{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：\n{e}")

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Morpholy",
            "<b>Morpholy v0.1.0</b><br>"
            "Geometric Morphometrics Analysis Plugin for YRTools<br><br>"
            "Author: Zhi-Jie Xu<br>"
            "License: GPL-3.0<br><br>"
            "Features: TPS I/O · GPA · PCA · TPS Deformation · Statistics",
        )

    # ------------------------------------------------------------------
    # 样式
    # ------------------------------------------------------------------

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
            }
            QListWidget {
                background: #2b2b3b;
                border: none;
                border-right: 1px solid #444;
                outline: none;
            }
            QListWidget::item {
                padding: 12px 0;
                color: #aaa;
            }
            QListWidget::item:selected {
                background: #3d3d55;
                color: #cdf;
                border-left: 3px solid #5599ff;
            }
            QListWidget::item:hover {
                background: #333347;
            }
            QStackedWidget {
                background: #1e1e2e;
            }
            QPushButton {
                background: #3d3d55;
                color: #cdd6f4;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 12px;
            }
            QPushButton:hover { background: #4a4a6a; }
            QPushButton:pressed { background: #2b2b3b; }
            QTableWidget {
                background: #252535;
                gridline-color: #444;
                border: 1px solid #444;
            }
            QHeaderView::section {
                background: #2b2b3b;
                color: #aaa;
                border: 1px solid #444;
                padding: 4px;
            }
            QScrollBar:vertical {
                background: #2b2b3b;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 4px;
            }
            """
        )
