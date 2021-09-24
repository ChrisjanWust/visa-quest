from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "latest.pickle"
CACHE_DIR_PATH = Path(".scrapy") / "httpcache" / "generic"
GIBBERISH_CSV_PATH = Path("ml") / "gibberish_dataset" / "texts.csv"
