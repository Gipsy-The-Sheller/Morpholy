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

from Morpholy.ui.main_window import MorpholyWindow


class MorpholyEntry:
    """YRTools 插件入口点，实例化并返回主窗口。"""

    def run(self) -> MorpholyWindow:
        return MorpholyWindow()
