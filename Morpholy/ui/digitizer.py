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
DigitizerWidget — 图像地标数字化面板。

支持加载图像、鼠标点击放置编号地标点、多标本管理。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QSpinBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSizePolicy, QGroupBox,
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QSize

from Morpholy.core.tps_io import Specimen


class ClickableImageLabel(QLabel):
    """
    支持点击的图像标签，将点击坐标（缩放到原始图像坐标）发射信号。
    """
    clicked = pyqtSignal(float, float)  # 原始图像坐标 (x, y)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)
        self._pixmap_orig: Optional[QPixmap] = None  # 原始（未缩放）图像
        self._landmarks: List[QPoint] = []            # 显示坐标（用于绘制）
        self._landmark_orig: List[tuple] = []         # 原始坐标

    def set_pixmap(self, pixmap: QPixmap):
        """加载原始图像并缩放到 label 大小显示。"""
        self._pixmap_orig = pixmap
        self._update_display()

    def _update_display(self):
        if self._pixmap_orig is None:
            return
        scaled = self._pixmap_orig.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def set_landmarks(self, landmarks_orig: List[tuple]):
        """从原始坐标列表更新显示坐标。"""
        self._landmark_orig = list(landmarks_orig)
        self.update()

    def _orig_to_display(self, ox: float, oy: float) -> QPoint:
        """将原始图像坐标转换为当前显示坐标。"""
        if self._pixmap_orig is None or self.pixmap() is None:
            return QPoint(int(ox), int(oy))
        orig_w = self._pixmap_orig.width()
        orig_h = self._pixmap_orig.height()
        disp_w = self.pixmap().width()
        disp_h = self.pixmap().height()
        if orig_w == 0 or orig_h == 0:
            return QPoint(0, 0)
        sx = disp_w / orig_w
        sy = disp_h / orig_h
        return QPoint(int(ox * sx), int(oy * sy))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap_orig is not None:
            if self.pixmap() is None:
                return
            # 缩放因子
            orig_w = self._pixmap_orig.width()
            orig_h = self._pixmap_orig.height()
            disp_w = self.pixmap().width()
            disp_h = self.pixmap().height()
            if disp_w == 0 or disp_h == 0:
                return
            ox = event.x() * orig_w / disp_w
            oy = event.y() * orig_h / disp_h
            # 边界裁剪
            ox = max(0.0, min(float(orig_w), ox))
            oy = max(0.0, min(float(orig_h), oy))
            self.clicked.emit(ox, oy)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._landmark_orig or self._pixmap_orig is None or self.pixmap() is None:
            return
        painter = QPainter(self)
        for idx, (ox, oy) in enumerate(self._landmark_orig):
            pt = self._orig_to_display(ox, oy)
            # 红色实心圆
            painter.setPen(QPen(QColor(255, 50, 50), 2))
            painter.setBrush(QColor(255, 50, 50))
            painter.drawEllipse(pt, 5, 5)
            # 白色编号
            painter.setPen(QPen(Qt.white, 1))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(pt.x() + 7, pt.y() - 4, str(idx + 1))
        painter.end()


class DigitizerWidget(QWidget):
    """
    图像地标数字化面板。

    多标本管理：左侧列表选择标本，右侧图像区域点击放置地标。
    """
    session_saved = pyqtSignal()  # 数据已保存到 session

    def __init__(self, session: Dict, parent: QWidget | None = None):
        super().__init__(parent)
        self._session = session
        self._specimens: List[Specimen] = []
        self._current_idx: int = -1
        # 每个标本的地标坐标（原始像素，列表形式）
        self._lm_store: List[List[tuple]] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 左侧控制面板
        left = QWidget()
        left.setFixedWidth(200)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        # 标本列表
        grp_spec = QGroupBox("标本列表")
        grp_spec_layout = QVBoxLayout(grp_spec)
        self._spec_list = QListWidget()
        self._spec_list.setMaximumHeight(200)
        grp_spec_layout.addWidget(self._spec_list)

        btn_add_spec = QPushButton("➕ 新建标本")
        btn_add_spec.clicked.connect(self._add_specimen)
        grp_spec_layout.addWidget(btn_add_spec)

        btn_del_spec = QPushButton("🗑 删除标本")
        btn_del_spec.clicked.connect(self._delete_specimen)
        grp_spec_layout.addWidget(btn_del_spec)

        left_layout.addWidget(grp_spec)

        # 图像操作
        grp_img = QGroupBox("图像")
        grp_img_layout = QVBoxLayout(grp_img)

        btn_load_img = QPushButton("📂 加载图像")
        btn_load_img.clicked.connect(self._load_image)
        grp_img_layout.addWidget(btn_load_img)

        left_layout.addWidget(grp_img)

        # 地标控制
        grp_lm = QGroupBox("地标控制")
        grp_lm_layout = QVBoxLayout(grp_lm)

        n_lm_row = QHBoxLayout()
        n_lm_row.addWidget(QLabel("最大数量:"))
        self._spin_nlm = QSpinBox()
        self._spin_nlm.setRange(1, 500)
        self._spin_nlm.setValue(10)
        n_lm_row.addWidget(self._spin_nlm)
        grp_lm_layout.addLayout(n_lm_row)

        btn_remove_last = QPushButton("↩ 撤销最后一个")
        btn_remove_last.clicked.connect(self._remove_last)
        grp_lm_layout.addWidget(btn_remove_last)

        btn_clear = QPushButton("🗑 清除所有地标")
        btn_clear.clicked.connect(self._clear_landmarks)
        grp_lm_layout.addWidget(btn_clear)

        self._lbl_count = QLabel("地标数: 0")
        grp_lm_layout.addWidget(self._lbl_count)

        left_layout.addWidget(grp_lm)

        btn_save_session = QPushButton("💾 保存到会话")
        btn_save_session.clicked.connect(self._save_to_session)
        left_layout.addWidget(btn_save_session)

        left_layout.addStretch()
        main_layout.addWidget(left)

        # 标本列表切换信号
        self._spec_list.currentRowChanged.connect(self._spec_list_current_changed)

        # 右侧图像区域
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._img_label = ClickableImageLabel()
        self._img_label.setMinimumSize(400, 400)
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img_label.clicked.connect(self._on_image_click)
        self._scroll.setWidget(self._img_label)
        main_layout.addWidget(self._scroll, stretch=1)

    # ------------------------------------------------------------------
    # 标本管理
    # ------------------------------------------------------------------

    def load_specimens(self, specimens: List[Specimen]):
        """从 TPS 文件加载标本列表。"""
        self._specimens = list(specimens)
        self._lm_store = [
            [(float(lm[0]), float(lm[1])) for lm in spec.landmarks]
            for spec in specimens
        ]
        self._spec_list.clear()
        for i, spec in enumerate(specimens):
            label = spec.specimen_id or f"Specimen {i + 1}"
            self._spec_list.addItem(label)
        if specimens:
            self._spec_list.setCurrentRow(0)
            self._switch_to(0)

    def _add_specimen(self):
        idx = len(self._specimens)
        spec = Specimen(
            landmarks=np.zeros((0, 2)),
            specimen_id=f"Specimen {idx + 1}",
        )
        # 空标本需要给一个占位地标（解决 Specimen 校验问题）
        spec_real = Specimen(
            landmarks=np.zeros((1, 2)),
            specimen_id=f"Specimen {idx + 1}",
        )
        self._specimens.append(spec_real)
        self._lm_store.append([])
        self._spec_list.addItem(spec_real.specimen_id)
        self._spec_list.setCurrentRow(idx)
        self._switch_to(idx)

    def _delete_specimen(self):
        row = self._spec_list.currentRow()
        if row < 0 or row >= len(self._specimens):
            return
        self._specimens.pop(row)
        self._lm_store.pop(row)
        self._spec_list.takeItem(row)
        if self._specimens:
            new_row = min(row, len(self._specimens) - 1)
            self._spec_list.setCurrentRow(new_row)
            self._switch_to(new_row)
        else:
            self._current_idx = -1
            self._img_label.set_landmarks([])
            self._lbl_count.setText("地标数: 0")

    def _switch_to(self, idx: int):
        self._current_idx = idx
        lms = self._lm_store[idx] if idx < len(self._lm_store) else []
        self._img_label.set_landmarks(lms)
        self._lbl_count.setText(f"地标数: {len(lms)}")

        # 如果标本有图像路径，尝试自动加载
        spec = self._specimens[idx]
        if spec.image and not self._img_label._pixmap_orig:
            from pathlib import Path
            if Path(spec.image).exists():
                px = QPixmap(spec.image)
                if not px.isNull():
                    self._img_label.set_pixmap(px)

    # ------------------------------------------------------------------
    # 图像 & 地标操作
    # ------------------------------------------------------------------

    def _load_image(self):
        if self._current_idx < 0:
            QMessageBox.warning(self, "警告", "请先选择或创建一个标本。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "加载图像", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        px = QPixmap(path)
        if px.isNull():
            QMessageBox.critical(self, "错误", "无法加载图像文件。")
            return
        self._img_label.set_pixmap(px)
        self._specimens[self._current_idx].image = path

    def _on_image_click(self, ox: float, oy: float):
        if self._current_idx < 0:
            return
        lms = self._lm_store[self._current_idx]
        max_lm = self._spin_nlm.value()
        if len(lms) >= max_lm:
            return
        lms.append((ox, oy))
        self._img_label.set_landmarks(lms)
        self._lbl_count.setText(f"地标数: {len(lms)}")

    def _remove_last(self):
        if self._current_idx < 0:
            return
        lms = self._lm_store[self._current_idx]
        if lms:
            lms.pop()
            self._img_label.set_landmarks(lms)
            self._lbl_count.setText(f"地标数: {len(lms)}")

    def _clear_landmarks(self):
        if self._current_idx < 0:
            return
        self._lm_store[self._current_idx].clear()
        self._img_label.set_landmarks([])
        self._lbl_count.setText("地标数: 0")

    # ------------------------------------------------------------------
    # 保存到 session
    # ------------------------------------------------------------------

    def _save_to_session(self):
        """
        将所有标本的地标数据打包为 Specimen 列表，存入 session。
        """
        updated: List[Specimen] = []
        for i, spec in enumerate(self._specimens):
            lms = self._lm_store[i]
            if not lms:
                continue
            arr = np.array(lms, dtype=np.float64)
            updated.append(Specimen(
                landmarks=arr,
                image=spec.image,
                specimen_id=spec.specimen_id or f"Specimen {i + 1}",
                scale=spec.scale,
                curves=spec.curves,
            ))
        if not updated:
            QMessageBox.warning(self, "警告", "没有有效的地标数据可保存。")
            return
        self._session["specimens"] = updated
        QMessageBox.information(
            self, "成功", f"已保存 {len(updated)} 个标本到会话。"
        )
        self.session_saved.emit()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def refresh(self, session: Dict):
        self._session = session
        specimens = session.get("specimens", [])
        if specimens and specimens != self._specimens:
            self.load_specimens(specimens)

    def _spec_list_current_changed(self, row: int):
        if 0 <= row < len(self._specimens):
            self._switch_to(row)
