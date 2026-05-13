"""
Generate field dimension diagrams (9v9 / USYS 12-U).
"""

from field import SoccerField


def generate_field_dimension_pngs() -> None:
    """Two plates: landscape and portrait; markings per USYS SSG Manual (2017) Law 1."""
    for orientation, filename in (
        ("landscape", "field-dimensions-9v9-landscape.png"),
        ("portrait", "field-dimensions-9v9-portrait.png"),
    ):
        field = SoccerField("u12", orientation=orientation)
        field.draw_markings().draw_dimensions()
        field.save(filename)


def generate_formation_4_3_1_portrait() -> None:
    """9v9 USYS field, portrait: 4-3-1 positions only (no dimension plates)."""
    field = SoccerField("u12", orientation="portrait")
    field.draw_markings()
    field.draw_formation_positions(field.formation_4_3_1_spots_own_goal_bottom())
    field.save_formation("formation-4-3-1-9v9-portrait.png", title="4-3-1")


if __name__ == "__main__":
    generate_field_dimension_pngs()
    generate_formation_4_3_1_portrait()
