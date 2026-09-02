from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_PATH / 'Data' / 'Raw Data'

ARTIFACTS_PATH = PROJECT_PATH / 'artifacts'

REPORTS_PATH = ARTIFACTS_PATH / 'reports'

CHARTS_PATH = ARTIFACTS_PATH / 'charts'

MODEL_PATH = ARTIFACTS_PATH / 'models'
