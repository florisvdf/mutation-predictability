import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from predictability.constants import AA_ALPHABET, AA_ALPHABET_GREMLIN


class PottsModel:
    """Direct coupling analysis model for generating evolutionary embeddings and
    calculating sequence energies"""

    def __init__(self, hi: np.array = None, jij: np.array = None):
        self.hi = hi
        self.jij = jij

    def run_gremlin(self, msa_path: str):
        if msa_path.endswith("a2m"):
            msa_path = self.format_a2m(msa_path)
        with TemporaryDirectory() as temp_dir:
            output_mrf_path = f"{temp_dir}/output.mrf"
            _ = subprocess.run(
                [
                    "gremlin_cpp",
                    "-i",
                    msa_path,
                    "-mrf_o",
                    output_mrf_path,
                    "-o",
                    f"{temp_dir}/ouputs",
                    "-gap_cutoff",
                    "1",
                    "-max_iter",
                    "100",
                ]
            )
            self.hi, self.jij = self.parse_mrf(output_mrf_path)

    @staticmethod
    def parse_mrf(mrf_path: str):
        with open(mrf_path, "r") as f:
            v_dict = {}
            w_dict = {}
            for line in f:
                split_line = line.split()
                mrf_id = split_line[0]
                if line.startswith("V"):
                    v_dict[mrf_id] = list(map(float, split_line[1:]))
                elif line.startswith("W"):
                    w_dict[mrf_id] = list(map(float, split_line[1:]))
        sequence_length = len(v_dict)
        v = np.zeros((sequence_length, 21))
        w = np.zeros((sequence_length, sequence_length, 21, 21))
        for key, value in v_dict.items():
            v_idx = int(*re.findall(r"\d+", key))
            v[v_idx] = value
        for key, value in w_dict.items():
            w_idx = list(map(lambda x: int(x), re.findall(r"\d+", key)))
            w[tuple(w_idx)] = np.reshape(value, (21, 21))
        return v, w

    @staticmethod
    def format_a2m(msa_path):
        file_name = Path(msa_path).name
        directory = Path(msa_path).parent
        formatted_msa_path = f"{directory}/modified_{file_name}"
        with open(msa_path, "r") as rf:
            with open(formatted_msa_path, "w") as wf:
                for line in rf:
                    if line.startswith(">"):
                        wf.write(line)
                    else:
                        wf.write(line.upper().replace(".", "-"))
        return formatted_msa_path

    @staticmethod
    def tokenize(letter):
        return AA_ALPHABET_GREMLIN.index(letter)

    def embed(self, sequences: List[str]):
        tokenized_sequences = []
        for seq in sequences:
            seq = list(map(self.tokenize, seq))
            tokenized_sequences.append(seq)
        seqs = np.array(tokenized_sequences)
        sequence_length = seqs.shape[1]
        embeddings = []
        for pos1 in range(sequence_length):
            hi_features = self.hi[pos1, seqs[:, pos1]].reshape(1, -1)
            jij_features = []
            for pos2 in range(sequence_length):
                jij_ = self.jij[pos1, pos2, seqs[:, pos1], seqs[:, pos2]]
                jij_features.append(jij_)
            jij_features = np.array(jij_features)
            emission_features = np.vstack((hi_features, jij_features))
            embeddings.append(emission_features)
        embeddings = np.transpose(np.array(embeddings), (2, 0, 1))
        return embeddings

    def predict(self, sequences: Union[pd.DataFrame, List[str]]):
        if isinstance(sequences, pd.DataFrame):
            sequences = sequences["sequence"]
        embeddings = self.embed(sequences)
        predictions = self._calculate_sequence_energy(embeddings)
        return predictions

    @staticmethod
    def _calculate_sequence_energy(embeddings):
        return np.sum(embeddings, axis=(1, 2))

    @classmethod
    def load(cls, model_directory: Union[Path, str]) -> "PottsModel":
        instance = cls(
            hi=np.load(Path(model_directory) / "hi.npy"),
            jij=np.load(Path(model_directory) / "Jij.npy"),
        )
        return instance


class PottsRegressor:
    def __init__(
        self,
        potts_path: Union[Path, str] = None,
        msa_path: Union[Path, str] = None,
        encoder: str = "energies",
        regressor_type: str = "ridge",
        **top_model_kwargs,

    ):
        self.potts_path = potts_path
        self.msa_path = msa_path
        self.encoder = encoder
        self.regressor_type = regressor_type
        if msa_path is not None:
            logger.info("Running Gremlin locally and saving emission parameters")
            self.potts_model = PottsModel()
            self.potts_model.run_gremlin(msa_path=self.msa_path)
        elif self.potts_path is not None:
            logger.info(f"Loading Potts model locally from: {self.potts_path}")
            self.potts_model = PottsModel.load(self.potts_path)
        self.top_model = {"ridge": Ridge, "extra_trees": ExtraTreesRegressor}[
            self.regressor_type
        ](**top_model_kwargs)

    def fit(self, data: pd.DataFrame, target: str):
        encodings = self.encode(data["sequence"])
        self.top_model.fit(encodings, data[target])

    def encode(self, sequences: Union[List[str], pd.Series]):
        return {
            "energies": self.compute_residue_energies,
            "augmented": self.compute_augmentation,
        }[self.encoder](sequences)

    def compute_residue_energies(self, sequences: Union[List[str], pd.Series]):
        embeddings = self.potts_model.embed(sequences)
        residue_energies = np.sum(embeddings, axis=2)
        return residue_energies

    def compute_augmentation(self, sequences: Union[List[str], pd.Series]):
        sequence_densities = self.potts_model.predict(sequences)
        one_hot_encodings = self.encode_sequences_one_hot(sequences).reshape(
            len(sequences), -1
        )
        return np.hstack((one_hot_encodings, sequence_densities.reshape(-1, 1)))

    @staticmethod
    def encode_sequences_one_hot(sequences: Union[List[str], pd.Series]):
        return np.array(
            [
                np.array(
                    [np.eye(len(AA_ALPHABET))[AA_ALPHABET.index(aa)] for aa in sequence]
                )
                for sequence in sequences
            ]
        )

    def predict(self, data: pd.DataFrame):
        encodings = self.encode(data["sequence"])
        return self.top_model.predict(encodings)


class RITARegressor:
    def __init__(
        self,
        top_model_type="ridge",
        pooling: str = "mean",
        device="cpu",
        **top_model_kwargs,
    ):
        self.top_model_type = top_model_type
        self.pooling = pooling
        self.device = device
        self.top_model_kwargs = top_model_kwargs
        self.model = None
        self.tokenizer = None
        self.embed_dim = 2048
        self.top_model = {
            "ridge": Ridge,
            "extra_trees": ExtraTreesRegressor,
            "mlp": MLPRegressor,
        }[self.top_model_type](**self.top_model_kwargs)

    def load_rita_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "lightonai/RITA_xl", trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")

    def embed(self, data):
        """
        The tokenizer creates a tokenized sequence of length len(sequence) + 1.
        No BOS token is created, but there is an EOS token. When using pooling option
        'last', embeddings are extracted by taking hidden_states[:, -2, :]
        (so only the hidden state of the last residue). When using option 'mean',
        embeddings are extracting by taking the mean of the second hidden states' axis.
        """
        embeddings_shape = (len(data), 2 * self.embed_dim)
        embeddings = np.empty(embeddings_shape)
        for i, sequence in tqdm(enumerate(data["sequence"])):
            with torch.no_grad():
                for j, p in enumerate([sequence, sequence[::-1]]):
                    tokenized_sequence = torch.tensor(
                        [self.tokenizer.encode(p)], device=self.device
                    )
                    outputs = self.embed_tokenized_sequence(tokenized_sequence)
                    embeddings[
                        i, self.embed_dim * j : self.embed_dim * (j + 1)
                    ] = outputs
        return embeddings

    def embed_tokenized_sequence(self, tokenized_sequence: torch.tensor):
        return {
            "mean": self.extract_mean_pooled_embeddings,
            "last": self.extract_last_hidden_state_embeddings,
        }[self.pooling](tokenized_sequence)

    def extract_mean_pooled_embeddings(self, tokenized_sequence: torch.tensor):
        return torch.mean(self.model(tokenized_sequence).hidden_states, dim=1).cpu()

    def extract_last_hidden_state_embeddings(self, tokenized_sequence: torch.tensor):
        return self.model(tokenized_sequence).hidden_states[:, -2, :].cpu()

    def fit(self, data: pd.DataFrame, target: str, embeddings=None):
        if embeddings is None:
            self.load_rita_model()
            embeddings = self.embed(data)
        logger.info(f"Fitting {self.top_model_type}")
        self.top_model.fit(embeddings, data[target])

    def predict(self, data, embeddings=None):
        if embeddings is None:
            embeddings = self.embed(data)
        return self.top_model.predict(embeddings)


class ResidueAgnosticRegressor:
    def __init__(self, wildtype_sequence, regressor_type="ridge"):
        self.wildtype_sequence = wildtype_sequence
        self.top_model = {"ridge": Ridge(), "extra_trees": ExtraTreesRegressor()}[
            regressor_type
        ]

    def assign_mutated(self, sequence):
        return [
            0 if aa_var == aa_wt else 1
            for aa_var, aa_wt in zip(sequence, self.wildtype_sequence)
        ]

    def prepare_inputs(self, sequences):
        inputs = np.array(list(map(self.assign_mutated, sequences)))
        return inputs

    def fit(self, data:pd.DataFrame, target: str):
        inputs = self.prepare_inputs(data["sequence"])
        self.top_model.fit(inputs, data[target])

    def predict(self, data):
        inputs = self.prepare_inputs(data["sequence"])
        return self.top_model.predict(inputs)


class PartialLeastSquares:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pls = PLSRegression(self.n_components)

    @staticmethod
    def encode_sequences(sequences: List[str]) -> np.array:
        n_samples = len(sequences)
        sequence_length = len(sequences[0])
        alphabet_length = len(AA_ALPHABET)
        encodings = np.empty((n_samples, sequence_length * alphabet_length))
        for i, sequence in enumerate(sequences):
            oh_sequence = np.concatenate(
                [np.eye(alphabet_length)[AA_ALPHABET.index(res)] for res in sequence]
            )
            encodings[i] = oh_sequence
        return encodings

    def fit(self, data: pd.DataFrame, target: str):
        sequences = data["sequence"].tolist()
        targets = data[target]
        encodings = self.encode_sequences(sequences)
        self.pls.fit(encodings, targets)

    def predict(self, data: pd.DataFrame) -> np.array:
        sequences = data["sequence"].tolist()
        encodings = self.encode_sequences(sequences)
        return self.pls.predict(encodings)
