### Data loading

You can load datasets using openml with src/data/load_datasets.py
Run as follows for downloading 300 datasets into data directory:
  `python3 -m src.data.load_datasets --output-dir data --n-datasets 300`

### Metafeatures extraction
You can extract base, structural and statistical metafeatures for dataset something like that:
  `python3 -m src.mtft data data/metafeatures`
