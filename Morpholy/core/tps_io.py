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
TPS 文件格式解析与写入模块。

支持字段：LM=, IMAGE=, ID=, SCALE=, CURVES=
数据模型：Specimen 列表，每个 Specimen 包含 n×2 numpy 地标数组、图像路径、ID、比例尺。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Specimen:
    """单个标本的地标数据。"""
    landmarks: np.ndarray          # shape (n, 2), float64
    image: str = ""                # 图像文件路径
    specimen_id: str = ""          # 标本 ID
    scale: float = 1.0             # 比例尺（单位/像素）
    curves: List[List[int]] = field(default_factory=list)  # 曲线地标点索引组

    def __post_init__(self):
        self.landmarks = np.asarray(self.landmarks, dtype=np.float64)
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != 2:
            raise ValueError(
                f"landmarks 必须为 (n, 2) 形状数组，得到 {self.landmarks.shape}"
            )

    @property
    def n_landmarks(self) -> int:
        return self.landmarks.shape[0]


def parse_tps(filepath: str | Path) -> List[Specimen]:
    """
    解析 TPS 文件，返回 Specimen 列表。

    TPS 格式示例::

        LM=4
        1.0 2.0
        3.0 4.0
        5.0 6.0
        7.0 8.0
        IMAGE=specimen1.jpg
        ID=1
        SCALE=0.05
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8", errors="replace")
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    specimens: List[Specimen] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # LM= 开头表示一个新标本块
        m = re.match(r"^LM=(\d+)$", line, re.IGNORECASE)
        if m:
            n_lm = int(m.group(1))
            i += 1
            coords: List[List[float]] = []
            for _ in range(n_lm):
                if i >= n:
                    break
                coord_line = lines[i].strip()
                i += 1
                if not coord_line:
                    continue
                parts = coord_line.split()
                if len(parts) >= 2:
                    coords.append([float(parts[0]), float(parts[1])])

            # 读取可选字段直到下一个 LM= 或文件末尾
            image = ""
            spec_id = ""
            scale = 1.0
            curves: List[List[int]] = []
            while i < n:
                peek = lines[i].strip()
                if re.match(r"^LM=", peek, re.IGNORECASE):
                    break
                if re.match(r"^IMAGE=", peek, re.IGNORECASE):
                    image = peek[6:].strip()
                    i += 1
                elif re.match(r"^ID=", peek, re.IGNORECASE):
                    spec_id = peek[3:].strip()
                    i += 1
                elif re.match(r"^SCALE=", peek, re.IGNORECASE):
                    try:
                        scale = float(peek[6:].strip())
                    except ValueError:
                        scale = 1.0
                    i += 1
                elif re.match(r"^CURVES=(\d+)$", peek, re.IGNORECASE):
                    n_curves = int(re.match(r"^CURVES=(\d+)$", peek, re.IGNORECASE).group(1))
                    i += 1
                    for _ in range(n_curves):
                        if i >= n:
                            break
                        curve_line = lines[i].strip()
                        i += 1
                        # POINTS= 行指定曲线点数
                        pm = re.match(r"^POINTS=(\d+)$", curve_line, re.IGNORECASE)
                        if pm:
                            n_pts = int(pm.group(1))
                            curve_pts: List[int] = []
                            for _ in range(n_pts):
                                if i >= n:
                                    break
                                pt_line = lines[i].strip()
                                i += 1
                                try:
                                    curve_pts.append(int(pt_line))
                                except ValueError:
                                    pass
                            curves.append(curve_pts)
                elif not peek:
                    i += 1
                else:
                    # 未知字段，跳过
                    i += 1

            if coords:
                lm_array = np.array(coords, dtype=np.float64)
                specimens.append(
                    Specimen(
                        landmarks=lm_array,
                        image=image,
                        specimen_id=spec_id,
                        scale=scale,
                        curves=curves,
                    )
                )
        else:
            i += 1

    return specimens


def write_tps(specimens: List[Specimen], filepath: str | Path) -> None:
    """
    将 Specimen 列表写入 TPS 文件。

    参数
    ----
    specimens : Specimen 对象列表
    filepath  : 输出文件路径
    """
    filepath = Path(filepath)
    lines: List[str] = []

    for spec in specimens:
        lines.append(f"LM={spec.n_landmarks}")
        for lm in spec.landmarks:
            lines.append(f"{lm[0]:.6f} {lm[1]:.6f}")
        if spec.image:
            lines.append(f"IMAGE={spec.image}")
        if spec.specimen_id:
            lines.append(f"ID={spec.specimen_id}")
        if spec.scale != 1.0:
            lines.append(f"SCALE={spec.scale:.6f}")
        if spec.curves:
            lines.append(f"CURVES={len(spec.curves)}")
            for curve in spec.curves:
                lines.append(f"POINTS={len(curve)}")
                for pt in curve:
                    lines.append(str(pt))
        lines.append("")  # 空行分隔标本

    filepath.write_text("\n".join(lines), encoding="utf-8")
