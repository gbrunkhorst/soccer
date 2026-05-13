"""
Soccer field visualization module with composable OOP design.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Repo-root diagrams/ (independent of cwd when running scripts)
_DIAGRAMS_DIR = Path(__file__).resolve().parent.parent / "diagrams"

# US Youth Soccer Small Sided Games Manual (2017), "12-U Modified Rules", Law 1 — SMALL-SIDED FIELD.
# https://washingtonyouthsoccer.org/wp-content/uploads/2019/09/US_Youth_Small_Sided_Games_Manual_2017_3-22-18.pdf
# Washington Youth Soccer field-layout PDF was not reachable from this environment; WYS aligns with USYS national markings.
_YD = 0.9144  # yards to metres
_FT = 0.3048  # feet to metres

# On-diagram labels: compact tuple style, modest size (horizontal only via _text_h)
_FONT_AXIS = 11
_FONT_TITLE = 13
_FONT_SUPTITLE = 9
_FONT_FIELD = 11
_FONT_MARK = 9
_FONT_NOTE = 8
_FONT_FORMATION_NUM = 11
_FONT_FORMATION_LABEL = 7
_FORMATION_MARKER_R = 1.65


class SoccerField:
    """
    Represents a soccer field with standard FIFA markings.
    Allows composition with formations, player positions, and tactical elements.
    """

    # FIFA standard dimensions (meters)
    STANDARD_LENGTH = 105
    STANDARD_WIDTH = 68

    # USYS 12-U / 9v9 — allowable field size (touchline × goal line), yards then metres
    U9V9_LENGTH_MIN_YD = 70
    U9V9_LENGTH_MAX_YD = 80
    U9V9_WIDTH_MIN_YD = 45
    U9V9_WIDTH_MAX_YD = 55

    # Exemplar field inside the USYS range (midpoint), for dimension plates
    U9V9_LENGTH_YD = 75
    U9V9_WIDTH_YD = 50
    U12_LENGTH = U9V9_LENGTH_YD * _YD
    U12_WIDTH = U9V9_WIDTH_YD * _YD

    # Field markings (meters) - Standard FIFA
    GOAL_AREA_LENGTH = 18.32
    GOAL_AREA_WIDTH = 40.32
    PENALTY_LENGTH = 16.5
    PENALTY_WIDTH = 40.32
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_ARC_RADIUS = 9.15  # Law 1: arc from penalty mark, same as adult here
    CORNER_ARC_RADIUS = 1
    GOAL_HEIGHT = 2.44
    GOAL_WIDTH = 7.32
    PENALTY_SPOT_DISTANCE = 11

    # USYS 12-U markings (yards from manual, stored as metres)
    U12_GOAL_AREA_LENGTH_YD = 5
    U12_GOAL_AREA_WIDTH_YD = 16
    U12_PENALTY_LENGTH_YD = 14
    U12_PENALTY_WIDTH_YD = 36
    U12_CENTER_CIRCLE_RADIUS_YD = 8
    U12_PENALTY_ARC_RADIUS_YD = 8
    U12_PENALTY_SPOT_DISTANCE_YD = 10
    U12_GOAL_AREA_LENGTH = U12_GOAL_AREA_LENGTH_YD * _YD
    U12_GOAL_AREA_WIDTH = U12_GOAL_AREA_WIDTH_YD * _YD
    U12_PENALTY_LENGTH = U12_PENALTY_LENGTH_YD * _YD
    U12_PENALTY_WIDTH = U12_PENALTY_WIDTH_YD * _YD
    U12_CENTER_CIRCLE_RADIUS = U12_CENTER_CIRCLE_RADIUS_YD * _YD
    U12_PENALTY_ARC_RADIUS = U12_PENALTY_ARC_RADIUS_YD * _YD
    U12_CORNER_ARC_RADIUS = 1.0  # Law 1 corner arc (metres)
    U12_GOAL_WIDTH_FT = 18
    U12_GOAL_HEIGHT_FT = 6
    U12_GOAL_WIDTH = U12_GOAL_WIDTH_FT * _FT
    U12_GOAL_HEIGHT = U12_GOAL_HEIGHT_FT * _FT
    U12_PENALTY_SPOT_DISTANCE = U12_PENALTY_SPOT_DISTANCE_YD * _YD

    def __init__(
        self,
        field_type: Literal["standard", "u12"] = "standard",
        orientation: Literal["landscape", "portrait"] = "landscape",
    ):
        """
        Initialize a soccer field.

        Args:
            field_type: "standard" (FIFA adult), "u12" (USYS 12-U / 9v9), or custom dimensions
            orientation: landscape — length along horizontal axis; portrait — length along vertical axis
        """
        self.orientation = orientation
        self._field_type = field_type
        if field_type == "standard":
            self.length = self.STANDARD_LENGTH
            self.width = self.STANDARD_WIDTH
            self._set_standard_markings()
        elif field_type == "u12":
            self.length = self.U12_LENGTH
            self.width = self.U12_WIDTH
            self._set_u12_markings()
        else:
            raise ValueError("field_type must be 'standard' or 'u12'")

        self.fig = None
        self.ax = None
        self._setup_figure()

    def _set_standard_markings(self):
        """Set standard FIFA field markings."""
        self.goal_area_length = self.GOAL_AREA_LENGTH
        self.goal_area_width = self.GOAL_AREA_WIDTH
        self.penalty_length = self.PENALTY_LENGTH
        self.penalty_width = self.PENALTY_WIDTH
        self.center_circle_radius = self.CENTER_CIRCLE_RADIUS
        self.penalty_arc_radius = self.PENALTY_ARC_RADIUS
        self.corner_arc_radius = self.CORNER_ARC_RADIUS
        self.goal_height = self.GOAL_HEIGHT
        self.goal_width = self.GOAL_WIDTH
        self.penalty_spot_distance = self.PENALTY_SPOT_DISTANCE

    def _set_u12_markings(self):
        """Set USYS 12-U (9v9) field markings."""
        self.goal_area_length = self.U12_GOAL_AREA_LENGTH
        self.goal_area_width = self.U12_GOAL_AREA_WIDTH
        self.penalty_length = self.U12_PENALTY_LENGTH
        self.penalty_width = self.U12_PENALTY_WIDTH
        self.center_circle_radius = self.U12_CENTER_CIRCLE_RADIUS
        self.penalty_arc_radius = self.U12_PENALTY_ARC_RADIUS
        self.corner_arc_radius = self.U12_CORNER_ARC_RADIUS
        self.goal_height = self.U12_GOAL_HEIGHT
        self.goal_width = self.U12_GOAL_WIDTH
        self.penalty_spot_distance = self.U12_PENALTY_SPOT_DISTANCE

    def _xp(self, along_length: float, along_width: float) -> float:
        return along_width if self.orientation == "portrait" else along_length

    def _yp(self, along_length: float, along_width: float) -> float:
        return along_length if self.orientation == "portrait" else along_width

    def _plot_line(self, L1: float, W1: float, L2: float, W2: float, **kwargs) -> None:
        self.ax.plot(
            [self._xp(L1, W1), self._xp(L2, W2)],
            [self._yp(L1, W1), self._yp(L2, W2)],
            **kwargs,
        )

    def _plot_polyline_field(self, Ls: np.ndarray, Ws: np.ndarray, **kwargs) -> None:
        xs = [self._xp(L, W) for L, W in zip(Ls, Ws)]
        ys = [self._yp(L, W) for L, W in zip(Ls, Ws)]
        self.ax.plot(xs, ys, **kwargs)

    def _add_rect_field(self, L0: float, W0: float, dL: float, dW: float, **kwargs) -> None:
        if self.orientation == "landscape":
            xy = (L0, W0)
            w, h = dL, dW
        else:
            xy = (W0, L0)
            w, h = dW, dL
        self.ax.add_patch(
            patches.Rectangle(xy, w, h, linewidth=2, edgecolor="white", facecolor="none", **kwargs)
        )

    def _add_field_background(self) -> None:
        if self.orientation == "landscape":
            xy, w, h = (0, 0), self.length, self.width
        else:
            xy, w, h = (0, 0), self.width, self.length
        self.ax.add_patch(
            patches.Rectangle(xy, w, h, linewidth=2, edgecolor="white", facecolor="#2d5016")
        )

    def _plot_margin(self) -> float:
        """Extra field-space margin for annotations (larger for 9v9 dimension plates)."""
        return 14.0 if self._field_type == "u12" else 8.0

    def _setup_figure(self) -> None:
        """Set up the matplotlib figure and axes."""
        m = self._plot_margin()
        if self.orientation == "landscape":
            self.fig, self.ax = plt.subplots(1, figsize=(14, 9), dpi=150)
            self.ax.set_xlim(-m, self.length + m)
            self.ax.set_ylim(-m, self.width + m)
        else:
            self.fig, self.ax = plt.subplots(1, figsize=(9, 14), dpi=150)
            self.ax.set_xlim(-m, self.width + m)
            self.ax.set_ylim(-m, self.length + m)

        self._add_field_background()

        self.ax.set_aspect("equal")
        self.ax.set_facecolor("#1a3a0a")
        self.fig.patch.set_facecolor("#1a3a0a")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")

    def _arc_points_deg(self, Lc: float, Wc: float, r: float, theta1: float, theta2: float, n: int = 160):
        """Arc in field (L,W) with angles in degrees CCW from +length axis (landscape x)."""
        angles = np.linspace(math.radians(theta1), math.radians(theta2), n)
        Ls = Lc + r * np.cos(angles)
        Ws = Wc + r * np.sin(angles)
        return Ls, Ws

    def draw_markings(self):
        """Draw standard FIFA field markings."""
        spot_ms = 8
        # Center line
        self._plot_line(self.length / 2, 0, self.length / 2, self.width, color="white", linewidth=2)

        # Center circle
        self.ax.add_patch(
            patches.Circle(
                (self._xp(self.length / 2, self.width / 2), self._yp(self.length / 2, self.width / 2)),
                self.center_circle_radius,
                fill=False,
                edgecolor="white",
                linewidth=2,
            )
        )

        # Center spot
        self.ax.plot(
            self._xp(self.length / 2, self.width / 2),
            self._yp(self.length / 2, self.width / 2),
            "w.",
            markersize=spot_ms,
        )

        # Goal areas
        self._add_rect_field(
            0,
            (self.width - self.goal_area_width) / 2,
            self.goal_area_length,
            self.goal_area_width,
        )
        self._add_rect_field(
            self.length - self.goal_area_length,
            (self.width - self.goal_area_width) / 2,
            self.goal_area_length,
            self.goal_area_width,
        )

        # Penalty areas
        self._add_rect_field(0, (self.width - self.penalty_width) / 2, self.penalty_length, self.penalty_width)
        self._add_rect_field(
            self.length - self.penalty_length,
            (self.width - self.penalty_width) / 2,
            self.penalty_length,
            self.penalty_width,
        )

        # Penalty spots
        self.ax.plot(
            self._xp(self.penalty_spot_distance, self.width / 2),
            self._yp(self.penalty_spot_distance, self.width / 2),
            "w.",
            markersize=spot_ms,
        )
        self.ax.plot(
            self._xp(self.length - self.penalty_spot_distance, self.width / 2),
            self._yp(self.length - self.penalty_spot_distance, self.width / 2),
            "w.",
            markersize=spot_ms,
        )

        # Penalty arcs — radius from penalty mark (Law 1 / USYS), outside the penalty area
        r = self.penalty_arc_radius
        dx = self.penalty_length - self.penalty_spot_distance
        if dx < r:
            theta = math.degrees(math.acos(dx / r))
            Ls1, Ws1 = self._arc_points_deg(
                self.penalty_spot_distance, self.width / 2, r, 360 - theta, 360 + theta
            )
            self._plot_polyline_field(Ls1, Ws1, color="white", linewidth=2)
            Ls2, Ws2 = self._arc_points_deg(
                self.length - self.penalty_spot_distance,
                self.width / 2,
                r,
                180 - theta,
                180 + theta,
            )
            self._plot_polyline_field(Ls2, Ws2, color="white", linewidth=2)
        elif abs(dx - r) < 1e-9:
            self.ax.plot(
                self._xp(self.penalty_length, self.width / 2),
                self._yp(self.penalty_length, self.width / 2),
                "w.",
                markersize=5,
            )

        # Corner arcs (FIFA Law 1: 1 m radius)
        corners = [(0, 0), (0, self.width), (self.length, 0), (self.length, self.width)]
        corner_angle_pairs = [(0, 90), (270, 360), (90, 180), (180, 270)]
        for (Lc, Wc), (t1, t2) in zip(corners, corner_angle_pairs):
            Ls, Ws = self._arc_points_deg(Lc, Wc, self.corner_arc_radius, t1, t2, n=48)
            self._plot_polyline_field(Ls, Ws, color="white", linewidth=2)

        # Goals (decorative frame outside goal line)
        gh = self.goal_height
        self._plot_line(-1, (self.width - gh) / 2, 0, (self.width - gh) / 2, color="white", linewidth=1)
        self._plot_line(-1, (self.width + gh) / 2, 0, (self.width + gh) / 2, color="white", linewidth=1)
        self._plot_line(-1, (self.width - gh) / 2, -1, (self.width + gh) / 2, color="white", linewidth=1)

        self._plot_line(
            self.length,
            (self.width - gh) / 2,
            self.length + 1,
            (self.width - gh) / 2,
            color="white",
            linewidth=1,
        )
        self._plot_line(
            self.length,
            (self.width + gh) / 2,
            self.length + 1,
            (self.width + gh) / 2,
            color="white",
            linewidth=1,
        )
        self._plot_line(
            self.length + 1,
            (self.width - gh) / 2,
            self.length + 1,
            (self.width + gh) / 2,
            color="white",
            linewidth=1,
        )

        return self

    @staticmethod
    def _t2(yd_a: float, yd_b: float) -> str:
        """Depth × width style pair, yards only."""
        return f"({yd_a:g},{yd_b:g})"

    @staticmethod
    def _t1(yd: float) -> str:
        return f"({yd:g})"

    def _field_range_tuple_yd(self) -> str:
        return (
            f"(({self.U9V9_LENGTH_MIN_YD},{self.U9V9_LENGTH_MAX_YD}),"
            f"({self.U9V9_WIDTH_MIN_YD},{self.U9V9_WIDTH_MAX_YD})) yd"
        )

    def _corner_yd(self) -> str:
        """FIFA 1 m corner arc, shown in yards for consistency."""
        return self._t1(round(self.corner_arc_radius / _YD, 1))

    def _text_h(
        self,
        L: float,
        W: float,
        s: str,
        *,
        ha: str = "center",
        va: str = "center",
        fontsize: float = _FONT_NOTE,
        fontweight: str | None = None,
    ) -> None:
        """Axis text with horizontal lines only (rotation=0)."""
        self.ax.text(
            self._xp(L, W),
            self._yp(L, W),
            s,
            ha=ha,
            va=va,
            color="white",
            fontsize=fontsize,
            fontweight=fontweight,
            rotation=0,
        )

    def draw_dimensions(self):
        """Add dimension labels (horizontal text only)."""
        if self._field_type == "u12":
            self._draw_dimensions_u12()
        else:
            self._draw_dimensions_standard()
        return self

    def _draw_dimensions_u12(self) -> None:
        top = 10.0
        side = 8.0

        # USYS touchline × goal-line ranges, yards only
        self._text_h(
            self.length / 2,
            self.width + top,
            self._field_range_tuple_yd(),
            ha="center",
            va="bottom",
            fontsize=_FONT_FIELD,
            fontweight="bold",
        )

        self._text_h(
            self.penalty_length / 2,
            (self.width - self.penalty_width) / 2 - 2.5,
            self._t2(self.U12_PENALTY_LENGTH_YD, self.U12_PENALTY_WIDTH_YD),
            ha="center",
            va="top",
            fontsize=_FONT_MARK,
        )
        self._text_h(
            self.goal_area_length / 2,
            (self.width - self.goal_area_width) / 2 - 1.5,
            self._t2(self.U12_GOAL_AREA_LENGTH_YD, self.U12_GOAL_AREA_WIDTH_YD),
            ha="center",
            va="top",
            fontsize=_FONT_MARK,
        )

        self._text_h(
            self.penalty_spot_distance,
            self.width / 2 + 3.0,
            self._t1(self.U12_PENALTY_SPOT_DISTANCE_YD),
            ha="center",
            va="bottom",
            fontsize=_FONT_MARK,
        )

        self._text_h(
            self.length / 2 + self.center_circle_radius + 2.0,
            self.width / 2 + 0.5,
            self._t1(self.U12_CENTER_CIRCLE_RADIUS_YD),
            ha="left",
            va="center",
            fontsize=_FONT_NOTE,
        )

        self._text_h(
            self.length * 0.72,
            self.width * 0.12,
            self._corner_yd(),
            ha="center",
            va="bottom",
            fontsize=_FONT_NOTE,
        )

        # Recommended goal opening: 18' × 6' → (6, 2) yd
        g_w_yd = self.U12_GOAL_WIDTH_FT / 3.0
        g_h_yd = self.U12_GOAL_HEIGHT_FT / 3.0
        self._text_h(
            self.length / 2,
            -side,
            self._t2(g_w_yd, g_h_yd),
            ha="center",
            va="top",
            fontsize=_FONT_NOTE,
        )

    def _draw_dimensions_standard(self) -> None:
        top, side = 8.0, 6.0
        self._text_h(
            self.length / 2,
            self.width + top,
            f"{self.length:g} m",
            ha="center",
            va="bottom",
            fontsize=_FONT_FIELD,
            fontweight="bold",
        )
        self._text_h(
            -side,
            self.width / 2,
            f"{self.width:g} m",
            ha="right",
            va="center",
            fontsize=_FONT_FIELD,
            fontweight="bold",
        )
        self._text_h(
            self.penalty_length / 2,
            (self.width - self.penalty_width) / 2 - 2,
            f"{self.penalty_length:g} m",
            ha="center",
            va="top",
            fontsize=_FONT_MARK,
        )
        self._text_h(
            self.penalty_length + 1,
            self.width / 2,
            f"{self.penalty_width:g} m",
            ha="left",
            va="center",
            fontsize=_FONT_MARK,
        )
        self._text_h(
            self.goal_area_length / 2,
            (self.width - self.goal_area_width) / 2 - 1,
            f"{self.goal_area_length:g} m",
            ha="center",
            va="top",
            fontsize=_FONT_NOTE,
        )
        self._text_h(
            self.penalty_spot_distance,
            self.width / 2 + 2,
            f"{self.penalty_spot_distance:g} m",
            ha="center",
            va="bottom",
            fontsize=_FONT_NOTE,
        )
        self._text_h(
            self.length / 2 + self.center_circle_radius + 1,
            self.width / 2,
            f"r = {self.center_circle_radius:g} m",
            ha="left",
            va="center",
            fontsize=_FONT_NOTE,
        )

    def draw_formation_positions(
        self,
        spots: list[tuple[float, float, int, str]],
        *,
        marker_radius: float = _FORMATION_MARKER_R,
    ) -> "SoccerField":
        """
        Draw numbered position markers (L, W in field metres along length × width).

        Spots: (along_length, along_width, jersey_number, role_label).
        """
        r = marker_radius
        for L, W, num, label in spots:
            x = self._xp(L, W)
            y = self._yp(L, W)
            self.ax.add_patch(
                patches.Circle(
                    (x, y),
                    r,
                    facecolor="white",
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=5,
                )
            )
            self.ax.text(
                x,
                y,
                str(num),
                ha="center",
                va="center",
                color="#1a3a0a",
                fontsize=_FONT_FORMATION_NUM,
                fontweight="bold",
                zorder=6,
            )
            self.ax.text(
                x,
                y - r - 1.15,
                label,
                ha="center",
                va="top",
                color="white",
                fontsize=_FONT_FORMATION_LABEL,
                zorder=6,
                linespacing=0.95,
                path_effects=[
                    pe.withStroke(linewidth=2.8, foreground="#1a3a0a", joinstyle="round"),
                ],
            )
        return self

    def formation_4_3_1_spots_own_goal_bottom(self) -> list[tuple[float, float, int, str]]:
        """
        4-3-1 compacted into the defensive half (striker on the centre line).
        Own goal at small L; 6 lines up with the wing backs (2/3); 8/10 ahead.
        """
        Lm, Wm = self.length, self.width
        L_wing = Lm * 0.29
        return [
            (Lm * 0.028, Wm * 0.50, 1, "Keeper"),
            (Lm * 0.17, Wm * 0.62, 4, "Right center back"),
            (Lm * 0.17, Wm * 0.38, 5, "Left center back"),
            (L_wing, Wm * 0.50, 6, "Defensive\ncenter mid"),
            (L_wing, Wm * 0.88, 2, "Right wing back"),
            (L_wing, Wm * 0.12, 3, "Left wing back"),
            (Lm * 0.41, Wm * 0.68, 8, "Right offensive\ncenter mid"),
            (Lm * 0.41, Wm * 0.32, 10, "Left offensive\ncenter mid"),
            (Lm * 0.50, Wm * 0.50, 9, "Striker"),
        ]

    def save_formation(self, filename: str, *, title: str, output_dir: Path | None = None) -> "SoccerField":
        """Save field + markings only: no dimension plates, axis labels, or USYS footnote."""
        out = Path(output_dir) if output_dir is not None else _DIAGRAMS_DIR
        out.mkdir(parents=True, exist_ok=True)
        full_path = out / filename

        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.ax.set_title(title, fontsize=_FONT_TITLE, fontweight="bold", color="white", pad=12)
        self.fig.suptitle("")

        plt.tight_layout()
        plt.savefig(full_path, dpi=150, facecolor="#1a3a0a")
        print(f"Formation saved: {full_path}")
        plt.close()
        return self

    def save(self, filename, output_dir=_DIAGRAMS_DIR):
        """Save the field diagram to a PNG file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        full_path = output_path / filename
        if self._field_type == "u12":
            self.ax.set_xlabel(
                "Width (m)" if self.orientation == "portrait" else "Length (m)",
                fontsize=_FONT_AXIS,
            )
            self.ax.set_ylabel(
                "Length (m)" if self.orientation == "portrait" else "Width (m)",
                fontsize=_FONT_AXIS,
            )
            note = f"{self._field_range_tuple_yd()} · USYS SSG (2017) 12-U Law 1"
            self.fig.suptitle(note, fontsize=_FONT_SUPTITLE, color="white", y=0.02)
            self.ax.set_title(
                "9v9 field (USYS 12-U)",
                fontsize=_FONT_TITLE,
                fontweight="bold",
                color="white",
                pad=10,
            )
        else:
            self.ax.set_xlabel("Length (m)", fontsize=_FONT_AXIS)
            self.ax.set_ylabel("Width (m)", fontsize=_FONT_AXIS)
            self.ax.set_title(
                f"FIFA Standard Field — {self.length} m × {self.width} m",
                fontsize=_FONT_TITLE,
                fontweight="bold",
                color="white",
            )

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig(full_path, dpi=150, facecolor="#1a3a0a")
        print(f"Field saved: {full_path}")
        plt.close()

        return self
