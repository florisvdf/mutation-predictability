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
