import math
import os
import re
from pathlib import Path
from typing import Union
from biotite.sequence.io.fasta import FastaFile
from Bio.PDB import PDBParser
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

from predictability.constants import BINARY_RESIDUE_FEATURES


def read_fasta(path: Union[str, Path]):
    return FastaFile.read(str(path))


def distance(one, other):
    return math.sqrt(
        (one.x_coord - other.x_coord) ** 2
        + (one.y_coord - other.y_coord) ** 2
        + (one.z_coord - other.z_coord) ** 2
    )


def dist_to_active_site(ca, active_site):
    return min(distance(ca, a) for a in active_site)


def assign_classes(data, feature_table, mutation_col="mutation", features="all"):
    if features == "all":
        features = BINARY_RESIDUE_FEATURES
    data["residue_number"] = data[mutation_col].map(
        lambda x: int(x[1:-1]) if x != "" else None
    )
    feature_table = feature_table[features + ["residue_number"]]
    data = pd.merge(data, feature_table, on="residue_number", how="left")
    return data


def sequence_to_mutations(sequence, reference):
    return "-".join(
        [
            f"{aa_ref}{pos+1:03d}{aa_var}"
            for pos, (aa_var, aa_ref) in enumerate(zip(sequence, reference))
            if aa_var != aa_ref
        ]
    )


def assign_mutations(df, reference):
    df["mutations"] = df["sequence"].map(lambda x: sequence_to_mutations(x, reference))
    return df


def get_buriedness(protein):
    buriedness = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein, protein)[0]

    atoms = []
    for chain in structure.get_chains():
        for res in chain:
            if res.id[0] == "W" or res.id[0].startswith("H_"):
                continue
            else:
                atoms.extend([atom for atom in res.get_atoms()])

    if not atoms:
        raise ValueError("Could not parse atoms in the pdb file")

    conv = ConvexHull([atom.coord for atom in atoms])
    for i, atom in enumerate(atoms):
        coord = atom.coord
        res = {
            "chain_id": atom.parent.parent.id,
            "residue_name": atom.parent.resname,
            "residue_number": int(atom.parent.id[1]),
            "buriedness": np.nan,
        }

        if i in conv.vertices:
            dist = 0
            res["buriedness"] = dist
        else:
            dist = np.inf
            for face in conv.equations:
                _dist = abs(np.dot(coord, face[:-1]) + face[-1])
                _dist = _dist / np.linalg.norm(face[:-1])
                if _dist < dist:
                    dist = _dist
            res["buriedness"] = dist
        buriedness.append(res)

    return (
        pd.DataFrame.from_records(buriedness)
        .groupby(["chain_id", "residue_name", "residue_number"], as_index=False)
        .mean()
        .sort_values(["chain_id", "residue_number"])
    )


def update_environment_variables(shell: str):
    home_dir = os.environ.get("HOME")
    with open(f"{home_dir}/.{shell}rc", "r") as file:
        bashrc_contents = file.read()
    pattern = r"export\s+(\w+)\s*=\s*(.*)"
    matches = re.findall(pattern, bashrc_contents)
    env_vars = {}
    for match in matches:
        key = match[0]
        value = match[1].strip("\"'")
        if key == "PATH":
            value = value.strip("$PATH:")
        env_vars[key] = value
    os.environ.update(env_vars)


def split_sel(data, mutation_col="mutation", ratio=0.1, seed=42):
    """
    Ensures that every mutated position in validation is also observed in train
    """
    np.random.seed(seed)
    max_test_size = int(ratio * len(data))
    positions = data[mutation_col].map(lambda x: x[1:-1]).unique()
    np.random.shuffle(positions)
    data["split"] = "train"
    for position in positions:
        if len(data[data["split"] == "valid"]) >= max_test_size:
            break
        corresponding_rows = data[data[mutation_col].str.contains(position)]
        if len(corresponding_rows) <= 1:
            continue
        else:
            data.loc[
                data[mutation_col].str.contains(position), "split"
            ] = np.random.choice(
                ["train", "valid"], size=len(corresponding_rows), p=[1 - ratio, ratio]
            )
    return data


def sel_kfold(data, position_col="residue_number", k=10):
    ratio = 1 / k
    k_indices = []
    positions = data[position_col].unique()
    np.random.shuffle(positions)
    for i in range(k):
        train_indices = np.array([])
        val_indices = np.array([])
        for position in positions:
            matching_indices = np.argwhere(
                (data["residue_number"] == position).values
            ).flatten()
            if len(matching_indices) == 0:
                continue
            n_matching_samples = len(matching_indices)
            slice_size = int(ratio * n_matching_samples)
            fold_val_indices = matching_indices[
                i * slice_size : (i + 1) * slice_size
            ].astype(int)
            val_indices = np.concatenate((val_indices, fold_val_indices))
            fold_train_indices = np.setdiff1d(matching_indices, fold_val_indices)
            train_indices = np.concatenate((train_indices, fold_train_indices))
        k_indices.append((train_indices.astype(int), val_indices.astype(int)))
    return k_indices
