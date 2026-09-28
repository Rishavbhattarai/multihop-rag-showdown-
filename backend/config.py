from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"

LLM_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text"

SEED = 42
N_PER_HOPS = {"2hop": 24, "3hop": 20, "4hop": 16}
N_DEMO = 10  # held out of eval, shown in UI dropdown

TOP_K = 5
MAX_HOPS = 3
MAX_EDGES = 30
