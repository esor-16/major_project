import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def raw_ibm_df():
    from pipeline import data

    return data.load_dataset(str(PROJECT_ROOT / "IBM.csv"))


@pytest.fixture(scope="session")
def processed_df(raw_ibm_df):
    from pipeline import data

    return data.preprocess(raw_ibm_df)


@pytest.fixture(scope="session")
def small_split(processed_df):
    """A small, fast-to-fit subsample: stratified so both classes survive
    the sample and the downstream train/test split.
    """
    from pipeline import data

    sample = processed_df.groupby("Churn", group_keys=False)[processed_df.columns].apply(
        lambda g: g.sample(n=min(len(g), 150), random_state=0)
    ).reset_index(drop=True)
    return data.split_data(sample, random_state=0), sample
