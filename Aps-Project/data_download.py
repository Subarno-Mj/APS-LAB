"""Download both Kaggle datasets into data/raw/.

Option A (recommended):  pip install kagglehub   then   python data_download.py
    Public datasets download without an API key; if Kaggle asks for one, put
    kaggle.json in %USERPROFILE%\\.kaggle\\ (Kaggle -> Settings -> Create New Token).
Option B (manual): download from the two pages below and unzip into data/raw/.
    https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
    https://www.kaggle.com/datasets/vipulshinde/incident-response-log
"""
import shutil
from pathlib import Path

from src import config

NEEDED = {
    config.TICKETS_KAGGLE: config.TICKETS_FILE.name,
    config.INCIDENTS_KAGGLE: config.INCIDENTS_FILE.name,
}


def main():
    missing = {k: v for k, v in NEEDED.items() if not (config.DATA_RAW / v).exists()}
    if not missing:
        print("Both datasets already present in", config.DATA_RAW)
        return
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("pip install kagglehub   (or download manually, see this file's docstring)")
    for slug, fname in missing.items():
        print(f"Downloading {slug} ...")
        path = Path(kagglehub.dataset_download(slug))
        src = next(path.rglob(fname), None)
        if src is None:
            raise SystemExit(f"{fname} not found in {path}")
        shutil.copy(src, config.DATA_RAW / fname)
        print(f"  -> {config.DATA_RAW / fname}")


if __name__ == "__main__":
    main()
