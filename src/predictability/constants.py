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
