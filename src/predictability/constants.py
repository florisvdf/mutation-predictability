from pathlib import Path

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
AA_ALPHABET_GREMLIN = list("ARNDCQEGHILKMFPSTWYV-")
BINARY_RESIDUE_FEATURES = [
    "is_buried",
    "is_connected",
    "is_close_to_as",
    "is_secondary",
]
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
THREE_TO_SINGLE_LETTER_CODES = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
