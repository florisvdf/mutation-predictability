from tqdm import tqdm
from typing import List
import re
from loguru import logger
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from predictability.constants import AA_ALPHABET


class PottsModel:
    """Direct coupling analysis model for generating evolutionary embeddings and
    calculating sequence energies"""

    def __init__(self, hi: np.array = None, jij: np.array = None):
        self.hi = hi
        self.jij = jij
        self.alphabet = list("ARNDCQEGHILKMFPSTWYV-")

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

    def tokenize(self, letter):
        return self.alphabet.index(letter)

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

    @classmethod
    def load(cls, model_directory: str) -> "PottsModel":
        instance = cls(
            hi=np.load(model_directory / "hi.npy"),
            jij=np.load(model_directory / "Jij.npy"),
        )
        return instance


class PottsRegressor:
    def __init__(self, potts_path=None, msa_path=None, regressor_type="ridge"):
        if msa_path is not None:
            logger.info("Running Gremlin locally and saving emission parameters")
            self.potts_model = PottsModel()
            self.potts_model.run_gremlin(msa_path=msa_path)
        elif potts_path is not None:
            logger.info(f"Loading Potts model locally from: {potts_path}")
            self.potts_model = PottsModel.load(potts_path)
        self.top_model = {"ridge": Ridge(), "extra_trees": ExtraTreesRegressor()}[
            regressor_type
        ]

    def fit(self, data, property):
        embeddings = self.potts_model.embed(data["sequence"])
        residue_energies = np.sum(embeddings, axis=2)
        self.top_model.fit(residue_energies, data[property])

    def predict(self, data):
        embeddings = self.potts_model.embed(data["sequence"])
        residue_energies = np.sum(embeddings, axis=2)
        return self.top_model.predict(residue_energies)


class RITARegressor:
    def __init__(self, top_model_type="ridge", top_model_kwargs={}):
        self.top_model_type = top_model_type
        self.top_model_kwargs = top_model_kwargs
        self.model = AutoModelForCausalLM.from_pretrained(
            "lightonai/RITA_xl", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")
        self.embed_dim = 2048
        self.top_model = {
            "ridge": Ridge,
            "extra_trees": ExtraTreesRegressor,
            "mlp": MLPRegressor,
        }[self.top_model_type](**self.top_model_kwargs)

    def embed(self, data):
        """
        The tokenizer creates a tokenized sequence of length len(sequence) + 1. No BOS token is created, but there is an
        EOS token. Embeddings are extracted by taking hidden_states[:, -2, :] (so only the hidden state of the last residue).
        """
        embeddings_shape = (len(data), 2 * self.embed_dim)
        embeddings = np.empty(embeddings_shape)
        for i, sequence in tqdm(enumerate(data["sequence"])):
            with torch.no_grad():
                for j, p in enumerate([sequence, sequence[::-1]]):
                    tokenized_sequence = torch.tensor([self.tokenizer.encode(p)])
                    input_tokens = tokenized_sequence
                    outputs = self.model(input_tokens).hidden_states[:, -2, :]
                    embeddings[
                        i, self.embed_dim * j : self.embed_dim * (j + 1)
                    ] = outputs
        return embeddings

    def fit(self, data, property, embeddings=None, batch_size=8):
        if embeddings is None:
            embeddings = self.embed(data, batch_size)
        logger.info(f"Fitting {self.top_model_type}")
        self.top_model.fit(embeddings, data[property])

    def predict(self, data, embeddings=None, batch_size=8):
        if embeddings is None:
            embeddings = self.embed(data, batch_size)
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

    def fit(self, data, property):
        inputs = self.prepare_inputs(data["sequence"])
        self.top_model.fit(inputs, data[property])

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

    def fit(self, data: pd.DataFrame, target):
        sequences = data["sequence"].tolist()
        targets = data[target]
        encodings = self.encode_sequences(sequences)
        self.pls.fit(encodings, targets)

    def predict(self, data: pd.DataFrame) -> np.array:
        sequences = data["sequence"].tolist()
        encodings = self.encode_sequences(sequences)
        return self.pls.predict(encodings)
