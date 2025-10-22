from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class Dataset:
    meta: Dict
    df: pd.DataFrame
    index: int
