import tarfile
from typing import Union
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import numpy as np

from predictability.models import RITARegressor, PottsRegressor
from predictability.constants import DATA_ROOT


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
        training_job_names: dict = None,
        default_bucket: str = None
):
    if training_job_names is not None:
        job_name = training_job_names[dataset]
        embeddings_uri = f"s3://{default_bucket}/{job_name}/output/output.tar.gz"
        download_artifacts_from_s3(embeddings_uri, path, directory="output")
    else:
        data = pd.read_csv(DATA_ROOT / f"{dataset}/data.csv")
        model = RITARegressor(device=device)
        embeddings = model.embed(data)
        np.save(embeddings, Path(path) / "embeddings.npy")
    return Path(path) / "embeddings.npy"

def get_potts_emissions(
        path: Union[Path, str],
        dataset: str,
        msa_path: Union[Path, str] = None,
        training_job_names: dict = None,
        default_bucket: str = None
):
    if training_job_names is not None:
        job_name = training_job_names[dataset]
        emissions_uri = f"s3://{default_bucket}/{job_name}/output/model.tar.gz"
        download_artifacts_from_s3(emissions_uri, path, directory="model")
    else:
        model = PottsRegressor(
            msa_path=msa_path,
        )
        np.save(model.potts_model.hi, Path(path) / "hi.npy")
        np.save(model.potts_model.jij, Path(path) / "Jij.npy")
    return path
