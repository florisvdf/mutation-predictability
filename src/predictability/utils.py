import math
from math import fmod
import itertools
import os
import re
import requests
from pathlib import Path
from typing import Union, List
from loguru import logger
from dataclasses import dataclass
import prody
from biopandas.pdb import PandasPdb
from biotite.sequence import ProteinSequence
from biotite.sequence.io.fasta import FastaFile
from Bio.PDB import PDBParser
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

from predictability.constants import BINARY_RESIDUE_FEATURES


class StructuralCharacterizer:
    def __init__(
        self,
        filename: str,
        active_site_residues: List[int],
        reference_sequence: str = None,
    ):
        self.filename = filename
        self.reference_sequence = reference_sequence
        self.active_site_residues = active_site_residues
        self.structure = (
            PandasPdb()
            .read_pdb(self.filename)
            .df["ATOM"]
            .query('alt_loc == "" or alt_loc == "A"')
        )
        self.residue_characteristics = (
            self.structure[["residue_number", "residue_name"]]
            .drop_duplicates()
            .assign(
                residue_name=lambda d: d.residue_name.apply(
                    ProteinSequence.convert_letter_3to1
                )
            )
        )
        self.assign_buriedness()
        self.assign_number_of_contacts()
        self.assign_distance_to_active_site()
        self.assign_secondary_structure()
        self.binarize_structural_characteristics()

    @staticmethod
    def coord_distance(atom1: pd.Series, atom2: pd.Series):
        return math.sqrt(
            (atom1.x_coord - atom2.x_coord) ** 2
            + (atom1.y_coord - atom2.y_coord) ** 2
            + (atom1.z_coord - atom2.z_coord) ** 2
        )

    @staticmethod
    def distance_to_active_site(ca, active_site):
        return min(coord_distance(ca, a) for a in active_site)

    def assign_buriedness(self):
        buriedness = get_buriedness(self.filename).loc[
            lambda d: (d.chain_id == "A") & (~d.residue_name.isin(["HOH", "ACI", "CA"]))
        ]
        self.residue_characteristics = self.residue_characteristics.merge(
            buriedness[["residue_number", "buriedness"]], on=["residue_number"]
        )

    def assign_number_of_contacts(self):
        ca_atoms = (
            self.structure.loc[lambda d: d.atom_name == "CA"]
            .copy()
            .assign(contacts=np.nan)[
                ["residue_number", "contacts", "x_coord", "y_coord", "z_coord"]
            ]
            .set_index("residue_number")
        )
        for ca in ca_atoms.itertuples():
            contacts = 0
            for other in ca_atoms.itertuples():
                if ca.Index != other.Index:
                    dist = coord_distance(ca, other)
                    if dist < 7.3:
                        contacts += 1
            ca_atoms.loc[ca.Index, "contacts"] = contacts
        self.residue_characteristics = self.residue_characteristics.merge(
            ca_atoms.reset_index()[["residue_number", "contacts"]]
        )

    def assign_distance_to_active_site(self):
        active_site = list(
            self.structure.loc[
                lambda d: (d.residue_number.isin(self.active_site_residues))
                & (d.atom_name == "CA")
            ].itertuples()
        )
        as_distances = (
            self.structure.loc[lambda d: d.atom_name == "CA"]
            .copy()[["residue_number", "x_coord", "y_coord", "z_coord"]]
            .set_index("residue_number")
        )
        as_distances["distance_to_active_site"] = [
            self.distance_to_active_site(ca, active_site)
            for ca in as_distances.itertuples()
        ]
        self.residue_characteristics = self.residue_characteristics.merge(
            as_distances.reset_index()[["residue_number", "distance_to_active_site"]]
        )

    def assign_secondary_structure(self):
        _, header = prody.parsePDB(self.filename, header=True)
        residues = self.structure.loc[lambda d: d.atom_name == "CA"].copy()[
            ["residue_number"]
        ]
        ranges = [
            Range(start, stop)
            for _, _, _, _, start, stop in header["helix_range"] + header["sheet_range"]
        ]
        residues["is_secondary"] = [
            any(v in r for r in ranges) for v in self.structure.residue_number.unique()
        ]
        self.residue_characteristics = self.residue_characteristics.merge(residues)

    def binarize_structural_characteristics(self):
        self.residue_characteristics["is_buried"] = (
            self.residue_characteristics.buriedness
            > self.residue_characteristics.buriedness.quantile(1 - 1 / 2)
        ).astype(bool)
        self.residue_characteristics["is_connected"] = (
            self.residue_characteristics.contacts
            > self.residue_characteristics.contacts.quantile(1 - 1 / 2)
        ).astype(bool)
        self.residue_characteristics["is_close_to_as"] = (
            self.residue_characteristics.distance_to_active_site
            < self.residue_characteristics.distance_to_active_site.quantile(1 - 1 / 2)
        ).astype(bool)


@dataclass
class Range:
    start: int
    stop: int

    def __contains__(self, value):
        return self.stop >= value >= self.start


def read_fasta(path: Union[str, Path]):
    return FastaFile.read(str(path))


def coord_distance(one: pd.Series, other: pd.Series):
    return math.sqrt(
        (one.x_coord - other.x_coord) ** 2
        + (one.y_coord - other.y_coord) ** 2
        + (one.z_coord - other.z_coord) ** 2
    )


def dist_to_active_site(ca, active_site):
    return min(coord_distance(ca, a) for a in active_site)


def download_pdb(pdb_id: str, path: Union[str, Path]):
    response = requests.get(f"https://files.rcsb.org/view/{pdb_id}.pdb")
    with open(Path(path) / f"{pdb_id}.pdb", "w") as f:
        f.write(response.text)


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


def ssm_kfold(data, position_col="residue_number", k=10, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    ratio = 1 / k
    k_indices = []
    positions = data[position_col].unique()
    np.random.shuffle(positions)
    for i in range(k):
        train_indices = np.array([])
        val_indices = np.array([])
        for position in positions:
            matching_indices = np.argwhere(
                (data[position_col] == position).values
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


def assign_ssm_folds(data, position_col="residue_number", n_folds=10, random_seed=None):
    df = data.copy().reset_index()
    if random_seed is not None:
        np.random.seed(random_seed)
    positions = data[position_col].unique()
    np.random.shuffle(positions)
    df["ssm_fold"] = pd.Series(dtype="Int64")
    for position in positions:
        matching_indices = np.argwhere(
            (data[position_col] == position).values
        ).flatten()
        for fold_number, index in zip(itertools.cycle(np.random.permutation(n_folds)), matching_indices):
            df.loc[index, "ssm_fold"] = fold_number
    return df


class ProteinGym:
    def __init__(self, proteingym_location, meta_data_path):
        self.proteingym_location = proteingym_location
        self.meta_data_path = meta_data_path
        self.reference_information = pd.read_csv(self.meta_data_path, index_col=None)
        self.available_pdbs = {}

    def update_reference_information(self):
        logger.info("Updating reference information with structure information.")
        pdb_entry_pattern = r".*PDB; [A-Z0-9]{4};.*"
        active_site_pattern = r".*ACT_SITE.*"
        region_pattern = r"A=(\d+-\d+)"
        for i, row in self.reference_information.iterrows():
            uniprot_id = row["UniProt_ID"]
            region_mutated = row["region_mutated"]
            response = requests.get(
                f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
            )
            if response.status_code == 400:
                continue
            body = response.text
            pdb_entries = list(
                filter(
                    lambda x: True if re.match(pdb_entry_pattern, x) else False,
                    body.split("\n"),
                )
            )
            active_site_entries = list(
                filter(
                    lambda x: True if re.match(active_site_pattern, x) else False,
                    body.split("\n")
                )
            )
            active_site_residues = "-".join(map(lambda x: x.split()[-1], active_site_entries))
            uniprot_sequence = fetch_uniprot_sequence(uniprot_id)
            self.available_pdbs[uniprot_id] = pdb_entries
            self.reference_information.loc[i, "has_pdb_structure"] = (
                len(pdb_entries) > 0
            )
            self.reference_information.loc[i, "structure_covers_mutated_region"] = False
            self.reference_information.loc[i, "uniprot_sequence"] = uniprot_sequence
            self.reference_information.loc[i, "active_site"] = active_site_residues
            for entry in pdb_entries:
                match = re.search(region_pattern, entry)
                if match:
                    structure_region = match.group(1)
                    structure_covers_mutated_region = region_is_subregion(
                        region_mutated, structure_region
                    )
                    if structure_covers_mutated_region:
                        self.reference_information.loc[
                            i, "structure_covers_mutated_region"
                        ] = True
                        break

    def describe_dataset(self, dataset_name):
        return self.reference_information[
            self.reference_information["DMS_id"] == dataset_name
        ]

    def fetch_msa(self, dataset_name):
        uniprot_id = self.describe_dataset(dataset_name)["UniProt_ID"].values[0]
        matching_msa_paths = list(
            (Path(self.proteingym_location) / "MSA_files/DMS").rglob(f"{uniprot_id}*")
        )
        logger.info(f"Found {len(matching_msa_paths)} matching MSA files")
        return read_fasta(matching_msa_paths[0])

    def prepare_dataset(self, dataset_name):
        data = pd.read_csv(
            Path(self.proteingym_location)
            / f"cv_folds_singles_substitutions/{dataset_name}.csv"
        ).rename(columns={"mutated_sequence": "sequence"})
        return data


def fetch_uniprot_sequence(uniprot_id: str):
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
    body = response.text
    sequence = "".join(body.split("\n")[1:])
    return sequence


def region_is_subregion(region1: str, region2: str):
    start1, end1 = tuple(int(value) for value in region1.split("-"))
    start2, end2 = tuple(int(value) for value in region2.split("-"))
    return (start1 >= start2) & (end1 <= end2)
