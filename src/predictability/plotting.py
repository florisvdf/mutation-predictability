from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import nglview as nv
import numpy as np
import pandas as pd
import seaborn as sns
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

features_to_df_descriptors = {
    "Buried": ("is_buried", 1),
    "Exposed": ("is_buried", 0),
    "Many contacts": ("is_connected", 1),
    "Few contacts": ("is_connected", 0),
    "Close to active site": ("is_close_to_as", 1),
    "Distant to active site": ("is_close_to_as", 0),
    "Part of helix or sheet": ("is_secondary", 1),
    "Part of loop": ("is_secondary", 0),
}

df_descriptors_to_features = {
    ("is_buried", 1): "Buried",
    ("is_buried", 0): "Exposed",
    ("is_connected", 1): "Many contacts",
    ("is_connected", 0): "Few contacts",
    ("is_close_to_as", 1): "Close to active site",
    ("is_close_to_as", 0): "Distant to active site",
    ("is_secondary", 1): "Part of helix or sheet",
    ("is_secondary", 0): "Part of loop",
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


def get_ngl_colorings(structural_characteristics: pd.DataFrame, color_map: dict):
    colorings = {}
    for characteristic in BINARY_RESIDUE_FEATURES:
        colorings[characteristic] = [
            [color_map[int(row[characteristic])], str(row["residue_number"])]
            for _, row in structural_characteristics.iterrows()
        ]
    return colorings


def plot_marginal_probabilities(marginal_probabilities, labels, ax=None):
    return_subplot = ax is None
    mask = np.tri(8, k=-1).transpose()

    if ax is not None:
        sns.heatmap(np.round(marginal_probabilities, 2), annot=True, mask=mask, ax=ax)
    else:
        fig, ax = plt.subplots()
        sns.heatmap(np.round(marginal_probabilities, 2), annot=True, mask=mask, ax=ax)

    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=45, va="center_baseline")
    ax.tick_params(axis="both", which="major", length=6, width=2, direction="out")

    if return_subplot:
        plt.show()
        fig.tight_layout()
        return fig, ax
