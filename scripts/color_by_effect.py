import os

from pymol import cmd

PYMOL_PATH = os.environ.get("PYMOL_PATH")


def read_source(source, property):
    file_contents = []
    with open(source, "r") as f:
        for line in f:
            file_contents.append(line.strip("\n").split(","))
    header = file_contents[0]
    position_idx = header.index("residue_number")
    property_idx = header.index(property)
    position_features = {
        line[position_idx]: float(line[property_idx]) for line in file_contents[1:]
    }
    return position_features


def minmax_scale_effects(position_effects):
    min_val = min(position_effects.values())
    max_val = max(position_effects.values())
    scaled_position_effects = {
        pos: (effect - min_val) / (max_val - min_val)
        for pos, effect in position_effects.items()
    }
    return scaled_position_effects


def visualize_property(mol, source, property, palette="rainbow", mode="putty"):
    """
    Replaces B-factors with property values from a csv file (e.g. std_effect, mean_effect).

    usage: color_by_effect mol, source, [palette, [mode]]

    mol = any object selection (within one single object though)
    source = path of the csv file storing features for the positions
    mode = whether to color or display putty

    example: color_by_effect 1LVM, <path_to_source>
    """

    cmd.alter(mol, "b=0")
    residue_properties = read_source(source, property)
    scaled_effects = minmax_scale_effects(residue_properties)
    {
        "putty": scale_putty,
        "color": color,
    }[mode](mol, scaled_effects, palette)


def color(mol, residue_properties, palette):
    obj = cmd.get_object_list(mol)[0]
    for position, property in residue_properties.items():
        cmd.alter(f"{mol} and resi {position} and n. CA", f"b={property}")
        cmd.show_as("cartoon", mol)
        cmd.spectrum("b", palette, "%s and n. CA " % mol)
        cmd.ramp_new(
            "count",
            obj,
            [min(residue_properties.values()), max(residue_properties.values())],
            palette,
        )
        cmd.recolor()


def scale_putty(mol, residue_properties, *args):
    obj = cmd.get_object_list(mol)[0]
    for position, property in residue_properties.items():
        cmd.alter(f"{mol} and resi {position} and n. CA", f"b={property}")
    cmd.iterate(f"{mol} and n. CA", 'print(f"Residue: {resi}, B-factor: {b}")')
    cmd.show_as("cartoon", mol)
    cmd.cartoon("putty", mol)
    cmd.set("cartoon_putty_scale_min", 0, obj)
    cmd.set("cartoon_putty_scale_max", 1, obj)
    cmd.set("cartoon_putty_transform", 5, mol)
    cmd.set("cartoon_putty_radius", 3, obj)


cmd.extend("visualize_property", visualize_property)
