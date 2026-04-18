from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
DATA_EXTERNAL = Path(getenv("DATA_EXTERNAL"))
DATA_INTERIM = Path(getenv("DATA_INTERIM"))
DATA_PROCESSED = Path(getenv("DATA_PROCESSED"))
DATA_RAW = Path(getenv("DATA_RAW"))