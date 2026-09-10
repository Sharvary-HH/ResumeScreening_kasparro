"""Every tunable in the project lives here - thresholds, weights, keyword sets.

Nothing else in the codebase should contain a magic number.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LLM provider -----------------------------------------------------------
# Any OpenAI-compatible chat-completions endpoint works; switching provider is a
# base URL + model name change. HF_TOKEN is read as a fallback for older .env files.
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN") or "").strip()
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
).rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-lite-latest")
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "4"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))

# Backoff for 429 / 5xx. A server-supplied Retry-After overrides these, capped
# by HTTP_MAX_BACKOFF.
HTTP_RETRY_DELAYS = (2, 4, 8, 16)
HTTP_MAX_BACKOFF = 45.0

# --- GitHub -----------------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_API = "https://api.github.com"
GITHUB_TIMEOUT = 20.0

# --- Cache ------------------------------------------------------------------
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(PROJECT_ROOT / ".cache")))
LLM_CACHE_DIR = CACHE_DIR / "llm"
GITHUB_CACHE_DIR = CACHE_DIR / "github"

# --- Score weights (max points per category, 100 total) ---------------------
AI_MAX = 40
PYTHON_MAX = 30
CLOUD_MAX = 15
GITHUB_MAX = 10
ENGINEERING_MAX = 5

TOTAL_MAX = AI_MAX + PYTHON_MAX + CLOUD_MAX + GITHUB_MAX + ENGINEERING_MAX

# A penalty outside this range is almost certainly the model inventing a scale.
PENALTY_MIN = 5
PENALTY_MAX = 15

# --- Parsing ----------------------------------------------------------------
MIN_TEXT_CHARS = 300
# Above this ratio of non-alphanumerics the "text" is extraction garbage.
MAX_NON_ALNUM_RATIO = 0.60

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

# --- Eligibility keyword sets ----------------------------------------------
# These frameworks imply Python even when the word never appears in the skills list.
PYTHON_KEYWORDS = [
    "python",
    "django",
    "flask",
    "fastapi",
    "pandas",
    "numpy",
    "pytorch",
    "pytest",
]

# Tier 1: LLM / agentic work - what the role actually asks for.
AI_KEYWORDS_STRONG = [
    "llm",
    "rag",
    "retrieval augmented",
    "langchain",
    "langgraph",
    "llamaindex",
    "google adk",
    "crewai",
    "autogen",
    "semantic kernel",
    "agent",
    "agentic",
    "multi-agent",
    "tool calling",
    "function calling",
    "embedding",
    "vector search",
    "vector database",
    "faiss",
    "chroma",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "openai api",
    "anthropic",
    "claude",
    "gpt-4",
    "gpt-4o",
    "huggingface",
    "transformers",
    "fine-tun",
    "prompt engineering",
    "mcp",
]

# Tier 2: classical ML / DL. Passes the gate but is flagged and ranks low.
AI_KEYWORDS_WEAK = [
    "machine learning",
    "deep learning",
    "neural network",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "sklearn",
    "computer vision",
    "nlp",
    "opencv",
    "xgboost",
]

WEAK_AI_CONCERN = "AI evidence is classical ML, not LLM/agentic"

REJECT_NO_PYTHON = "No evidence of Python stack"
REJECT_NO_AI = "No AI/agentic project evidence"

# Reported as matched_skills. Display casing preserved; matching is case-insensitive.
RELEVANT_SKILLS = [
    "Python",
    "FastAPI",
    "Django",
    "Flask",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "Celery",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
    "CI/CD",
    "Git",
    "Linux",
    "REST API",
    "GraphQL",
    "LangChain",
    "LangGraph",
    "LlamaIndex",
    "CrewAI",
    "AutoGen",
    "RAG",
    "LLM",
    "OpenAI",
    "Hugging Face",
    "Transformers",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "Pandas",
    "NumPy",
    "FAISS",
    "Pinecone",
    "ChromaDB",
    "Qdrant",
    "Weaviate",
    "React",
    "Next.js",
    "Node.js",
    "TypeScript",
    "JavaScript",
    "Streamlit",
    "Gradio",
    "Pytest",
    "Kafka",
    "RabbitMQ",
]

# --- GitHub scoring thresholds ----------------------------------------------
# (max_days_since_last_push, score) - first match wins.
GITHUB_ACTIVITY_THRESHOLDS = [
    (30, 5),
    (90, 4),
    (180, 3),
    (365, 2),
]
GITHUB_ACTIVITY_STALE_SCORE = 1  # older than the last threshold above
GITHUB_ACTIVITY_NO_EVENTS_SCORE = 0

# (min_relevant_repo_count, score) - first match wins.
GITHUB_REPO_THRESHOLDS = [
    (5, 5),
    (3, 4),
    (2, 3),
    (1, 2),
]
GITHUB_REPO_HAS_REPOS_SCORE = 1  # repos exist, none relevant
GITHUB_REPO_NO_REPOS_SCORE = 0

# A repo counts as "recently maintained" within this many days.
GITHUB_REPO_FRESH_DAYS = 365

# Keywords that make a repo count as AI-relevant regardless of language.
GITHUB_AI_REPO_KEYWORDS = [
    "llm",
    "rag",
    "agent",
    "langchain",
    "langgraph",
    "openai",
    "gpt",
    "nlp",
    "ml",
    "machine-learning",
    "deep-learning",
    "transformer",
    "chatbot",
    "ai",
]

# --- Reporting --------------------------------------------------------------
DEFAULT_TOP_N = 10
