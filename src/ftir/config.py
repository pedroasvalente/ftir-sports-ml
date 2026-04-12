import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

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

try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

try:
    import mlflow

    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logging_level = os.environ.get("MLFLOW_LOGGING_LEVEL", None)

    class _LoguruHandler(logging.Handler):
        def emit(self, record):
            if mlflow_logging_level is None:
                return
            log_entry = self.format(record)
            level = record.levelname.lower()
            getattr(logger, level, logger.info)(log_entry)

    mlflow_logger.addHandler(_LoguruHandler())
    for h in mlflow_logger.handlers[:-1]:
        mlflow_logger.removeHandler(h)
except ModuleNotFoundError:
    pass
