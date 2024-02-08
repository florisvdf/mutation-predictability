from typing import List, Union
from pathlib import Path
import pandas as pd

import nglview as nv
from nglview.color import ColormakerRegistry

from predictability.constants import BINARY_RESIDUE_FEATURES


def assign_pretty_feature_names(old_feature_name):
    if old_feature_name == "is_buried":
        return "Buried"
    elif old_feature_name == "is_connected":
        return "Many contacts"
    elif old_feature_name == "is_close_to_as":
        return "Close to active site"
    elif old_feature_name == "is_secondary":
        return "Part of helix or sheet"


color_palette = [
    "#EF476F",
    "#06D6A0",
    "#F78C6B",
    "#0CB0A9",
    "#FFD166",
    "#118AB2",
    "#83D483",
    "#073B4C",
]

feature_mapping = {
    ("Buried", "Positive"): "Buried",
    ("Buried", "Negative"): "Exposed",
    ("Many contacts", "Positive"): "Many contacts",
    ("Many contacts", "Negative"): "Few contacts",
    ("Part of helix or sheet", "Positive"): "Part of helix or sheet",
    ("Part of helix or sheet", "Negative"): "Part of loop",
    ("Close to active site", "Positive"): "Close to active site",
    ("Close to active site", "Negative"): "Distant to active site",
}

color_mapping = {
    "Buried": "#EF476F",
    "Exposed": "#06D6A0",
    "Many contacts": "#F78C6B",
    "Few contacts": "#0CB0A9",
    "Close to active site": "#FFD166",
    "Distant to active site": "#118AB2",
    "Part of helix or sheet": "#83D483",
    "Part of loop": "#073B4C",
}


def show_structure(pdb_file_path: Union[Path, str], coloring: List[List[str]]):
    view = nv.show_file(str(pdb_file_path))
    cm = ColormakerRegistry
    cm.add_selection_scheme("awesome", coloring)
    view.center()
    view.clear_representations()
    view.add_cartoon(color="awesome")
    return view


def get_ngl_colorings(structural_characeteristics: pd.DataFrame, color_map: dict):
    colorings = {}
    for characteristic in BINARY_RESIDUE_FEATURES:
        colorings[characteristic] = [
            [color_map[int(row[characteristic])], str(row["residue_number"])]
            for _, row in structural_characeteristics.iterrows()
        ]
    return colorings
