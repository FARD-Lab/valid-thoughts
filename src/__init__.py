import os
from pathlib import Path

_env_loaded = False

if not _env_loaded:
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=False)

    if os.getenv("HF_DATASETS_CACHE"):
        os.environ.setdefault("HF_DATASETS_CACHE", os.getenv("HF_DATASETS_CACHE", ""))

    if os.getenv("HF_HOME"):
        os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", ""))

    _env_loaded = True
