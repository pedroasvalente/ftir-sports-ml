import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
RESULTS_DIR = PROJ_ROOT / "results"
EXPERIMENTS_DIR = PROJ_ROOT / "experiments"
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"

TRAINING_DATA_PATH = os.environ.get(
    "TRAINING_DATA_PATH",
    str(DATA_DIR / "001_3_cleaned_FTIR.csv"),
)

random_seed = int(os.environ.get("RANDOM_SEED", 52))
global_threshold_acc = int(os.environ.get("GLOBAL_THRESHOLD_ACC", 70))

# FTIR atmospheric CO₂ / water vapour absorption region to exclude from analysis
WATER_REGION = (1850, 2500)  # wavenumbers (cm⁻¹)

try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except Exception:
    pass


def init_mlflow():
    """Initialise MLflow + DagsHub. Call this from training code only, not from the Streamlit app."""
    try:
        import mlflow

        _dagshub_token = os.environ.get("DAGSHUB_USER_TOKEN")
        if _dagshub_token:
            try:
                import dagshub
                dagshub.init(
                    repo_owner=os.environ.get("DAGSHUB_REPO_OWNER", "pedroasvalente"),
                    repo_name=os.environ.get("DAGSHUB_REPO_NAME", "ftir-sports-ml"),
                    mlflow=True,
                )
                logger.info("DagsHub initialized")
            except Exception as e:
                logger.warning(f"DagsHub init failed, using local MLflow: {e}")
        else:
            logger.info("No DAGSHUB_USER_TOKEN — using local MLflow tracking")

        mlflow_logger = logging.getLogger("mlflow")

        class _LoguruHandler(logging.Handler):
            def emit(self, record):
                if os.environ.get("MLFLOW_LOGGING_LEVEL") is None:
                    return
                log_entry = self.format(record)
                getattr(logger, record.levelname.lower(), logger.info)(log_entry)

        mlflow_logger.addHandler(_LoguruHandler())
        return mlflow

    except Exception as e:
        logger.warning(f"MLflow unavailable: {e}")
        return None
