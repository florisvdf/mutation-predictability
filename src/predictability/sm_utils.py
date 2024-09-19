import os
import re
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
import pandas as pd

from predictability.constants import DATA_ROOT
from predictability.models import PottsRegressor, RITARegressor


def download_artifacts_from_s3(
    uri: str, out_dir: Union[str, Path], directory: str = "output", artifacts: list = None
):
    """Download output.tar.gz and extract files given S3 uri.
    If any artifacts are passed, only those files will be extracts"""
    try:
        from sagemaker.s3 import S3Downloader
    except ImportError:
        raise ImportError
    with TemporaryDirectory() as fh:
        S3Downloader.download(uri, fh)
        with tarfile.open(Path(fh) / f"{directory}.tar.gz") as t:
            if artifacts is not None:
                for artifact in artifacts:
                    t.extract(artifact, str(out_dir))
            else:
                t.extractall(str(out_dir))

def get_rita_embeddings(
        path: Union[Path, str],
        dataset: str,
        device: str = "cpu",
        embeddings_uri: str = None,
):
    if embeddings_uri is not None:
        download_artifacts_from_s3(embeddings_uri, path, directory="output")
    else:
        data = pd.read_csv(DATA_ROOT / f"{dataset}/data.csv")
        model = RITARegressor(device=device)
        embeddings = model.embed(data)
        np.save(embeddings, Path(path) / "embeddings.npy")
    return Path(path) / "embeddings.npy"

def get_potts_emissions(
        path: Union[Path, str],
        msa_path: Union[Path, str] = None,
        emissions_uri: str = None,
):
    if emissions_uri is not None:
        download_artifacts_from_s3(emissions_uri, path, directory="model")
    else:
        model = PottsRegressor(
            msa_path=msa_path,
        )
        np.save(model.potts_model.hi, Path(path) / "hi.npy")
        np.save(model.potts_model.jij, Path(path) / "Jij.npy")
    return path


def read_sm_credentials(account_name: str):
    home_dir = os.environ.get("HOME")
    with open(f"{home_dir}/.sm_{account_name}", "r") as file:
        contents = file.read()
    pattern = r"export\s+(\w+)\s*=\s*(.*)"
    matches = re.findall(pattern, contents)
    env_vars = {}
    for match in matches:
        key = match[0]
        value = match[1].strip("\"'")
        env_vars[key] = value
    os.environ.update(env_vars)
