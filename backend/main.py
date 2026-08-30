from pathlib import Path
import os
import re
import time
import json
import requests

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoTokenizer


# ============================================================
# TruthLens AI Backend
# Version 7.3.2
#
# Evidence-First Misinformation Analysis
#
# Components:
#   1. DistilBERT initial assessment
#   2. Canonical evidence
#   3. Wikipedia search
#   4. Wikipedia page retrieval
#   5. Evidence relevance ranking
#   6. Deterministic verification
#   7. Optional NLI verification
#   8. Source reliability
#   9. Evidence quality
#  10. Evidence-first final decision
#
# IMPORTANT:
# NLI can be disabled on low-memory deployments:
#
#     DISABLE_NLI=true
#
# ============================================================


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="TruthLens AI",
    description=(
        "AI-Powered Misinformation Analysis "
        "and Evidence-Based Decision Support System"
    ),
    version="7.3.2",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    content: str


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens"
)

HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "neelu4304/TruthLensAI-distilbert",
)

HF_MODEL_SUBFOLDER = "distilbert_truthlens"

# ONNX Runtime deployment is CPU-only.
# PyTorch is imported only if optional NLI verification is enabled.
DEVICE = "cpu"

WIKIPEDIA_API = (
    "https://en.wikipedia.org/w/api.php"
)

NLI_MODEL_NAME = (
    "cross-encoder/nli-MiniLM2-L6-H768"
)

HEADERS = {
    "User-Agent": (
        "TruthLensAI/7.3 "
        "(educational misinformation analysis project)"
    ),
    "Accept": "application/json",
}

REQUEST_DELAY = 0.25

MIN_NLI_CONFIDENCE = 65.0

MIN_EVIDENCE_QUALITY = 45.0


# ============================================================
# SOURCE RELIABILITY
# ============================================================

SOURCE_RELIABILITY_SCORES = {

    "WIKIPEDIA": 85.0,

    "GOVERNMENT": 98.0,

    "ACADEMIC": 97.0,

    "PEER_REVIEWED": 98.0,

    "REPUTABLE_NEWS": 90.0,

    "ORGANIZATION": 88.0,

    "UNKNOWN": 50.0,
}


# ============================================================
# IN-MEMORY CACHES
# ============================================================

PAGE_CACHE = {}

SEARCH_CACHE = {}


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("TruthLens AI Backend v7.3.2")
print("=" * 70)

print("Device:", DEVICE)

print("Model path:", MODEL_PATH)

print("Hugging Face repository:", HF_MODEL_REPO)

print("Hugging Face subfolder:", HF_MODEL_SUBFOLDER)

print("=" * 70)


# ============================================================
# DISTILBERT MODEL LOADING - INT8 ONNX
# ============================================================
#
# The deployment model is the tested 64 MB INT8 ONNX model:
#
#   models/distilbert_truthlens_onnx/model.int8.onnx
#
# ONNX Runtime is used instead of PyTorch for the main model so
# the backend remains suitable for a small CPU-only deployment.
#
# NLI remains optional and disabled by default.
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens"
)

ONNX_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "distilbert_truthlens_onnx"
)

ONNX_MODEL_FILE = (
    ONNX_MODEL_PATH
    / "model.int8.onnx"
)

ONNX_TOKENIZER_FILE = (
    ONNX_MODEL_PATH
    / "tokenizer.json"
)

ONNX_TOKENIZER_CONFIG = (
    ONNX_MODEL_PATH
    / "tokenizer_config.json"
)

DEVICE = "cpu"

# Keep the process conservative for free/small CPU instances.
os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

print("-" * 70)
print("Loading TruthLens AI INT8 ONNX model...")
print("ONNX model path:", ONNX_MODEL_FILE)
print("Device:", DEVICE)

if not ONNX_MODEL_FILE.exists():

    raise FileNotFoundError(
        "INT8 ONNX model not found: "
        f"{ONNX_MODEL_FILE}"
    )

if not ONNX_TOKENIZER_FILE.exists():

    raise FileNotFoundError(
        "ONNX tokenizer.json not found: "
        f"{ONNX_TOKENIZER_FILE}"
    )

if not ONNX_TOKENIZER_CONFIG.exists():

    raise FileNotFoundError(
        "ONNX tokenizer_config.json not found: "
        f"{ONNX_TOKENIZER_CONFIG}"
    )

print("Loading ONNX tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(ONNX_MODEL_PATH),
    local_files_only=True,
)

print("ONNX tokenizer loaded successfully.")

print("Loading INT8 ONNX Runtime session...")

session_options = ort.SessionOptions()

session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

onnx_session = ort.InferenceSession(
    str(ONNX_MODEL_FILE),
    sess_options=session_options,
    providers=[
        "CPUExecutionProvider",
    ],
)

ONNX_INPUT_NAMES = {
    item.name
    for item in onnx_session.get_inputs()
}

ONNX_OUTPUT_NAMES = [
    item.name
    for item in onnx_session.get_outputs()
]

print(
    "ONNX inputs:",
    sorted(ONNX_INPUT_NAMES),
)

print(
    "ONNX outputs:",
    ONNX_OUTPUT_NAMES,
)

print(
    "INT8 ONNX model loaded successfully."
)

print(
    "INT8 ONNX model size:",
    round(
        ONNX_MODEL_FILE.stat().st_size
        / (1024 * 1024),
        2,
    ),
    "MB",
)

# ============================================================
# OPTIONAL NLI MODEL
# ============================================================

# Disable NLI by default for low-memory deployments (for example,
# Render free-tier instances). Set DISABLE_NLI=false only when
# sufficient memory is available.
DISABLE_NLI = (
    os.getenv(
        "DISABLE_NLI",
        "true",
    ).lower()
    == "true"
)

nli_tokenizer = None

nli_model = None

NLI_AVAILABLE = False

NLI_LABEL_IDS = {}

NLI_ENTAILMENT_ID = None

NLI_CONTRADICTION_ID = None

NLI_NEUTRAL_ID = None


if DISABLE_NLI:

    print("-" * 70)

    print(
        "NLI Verification: DISABLED"
    )

    print(
        "Reason: DISABLE_NLI=true"
    )

else:

    print("-" * 70)

    print(
        "Loading semantic verification model..."
    )

    print(
        "NLI model:",
        NLI_MODEL_NAME,
    )

    try:

        # Import PyTorch/AutoModel only when NLI is explicitly
        # enabled. This keeps the default deployment lightweight.
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
        )

        nli_tokenizer = (
            AutoTokenizer.from_pretrained(
                NLI_MODEL_NAME
            )
        )

        nli_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                NLI_MODEL_NAME
            )
        )

        nli_model.to(DEVICE)

        nli_model.eval()

        NLI_AVAILABLE = True

        print(
            "NLI model loaded successfully."
        )

        print(
            "NLI labels:",
            nli_model.config.id2label,
        )

    except Exception as error:

        print(
            "NLI model could not be loaded."
        )

        print(
            "NLI error:",
            repr(error),
        )

        print(
            "Continuing without NLI verification."
        )

        nli_tokenizer = None

        nli_model = None

        NLI_AVAILABLE = False


# ============================================================
# NLI LABEL MAPPING
# ============================================================

def build_nli_label_ids():

    if (
        not NLI_AVAILABLE
        or nli_model is None
    ):

        return (
            {},
            None,
            None,
            None,
        )

    labels = {}

    for index, label in (
        nli_model.config.id2label.items()
    ):

        labels[
            str(label)
            .lower()
            .strip()
        ] = int(index)

    def find_label(keyword):

        for label, index in labels.items():

            if keyword in label:

                return index

        return None

    entailment_id = find_label(
        "entail"
    )

    contradiction_id = find_label(
        "contrad"
    )

    neutral_id = find_label(
        "neutral"
    )

    # Some models expose LABEL_0,
    # LABEL_1 and LABEL_2.
    if (
        entailment_id is None
        or contradiction_id is None
        or neutral_id is None
    ):

        if len(
            nli_model.config.id2label
        ) == 3:

            contradiction_id = 0

            neutral_id = 1

            entailment_id = 2

    return (
        labels,
        entailment_id,
        contradiction_id,
        neutral_id,
    )


(
    NLI_LABEL_IDS,
    NLI_ENTAILMENT_ID,
    NLI_CONTRADICTION_ID,
    NLI_NEUTRAL_ID,
) = build_nli_label_ids()


if NLI_AVAILABLE:

    print(
        "NLI labels:",
        NLI_LABEL_IDS,
    )

    print(
        "NLI IDs:",
        {
            "entailment":
                NLI_ENTAILMENT_ID,

            "contradiction":
                NLI_CONTRADICTION_ID,

            "neutral":
                NLI_NEUTRAL_ID,
        },
    )


# ============================================================
# CANONICAL EVIDENCE
# ============================================================

CANONICAL_EVIDENCE = {

    "earth_flat": {

        "title": "Flat Earth",

        "text": (
            "Flat Earth is an archaic and "
            "scientifically disproven conception "
            "of the Earth's shape as a plane or disk. "
            "Scientific evidence shows that Earth "
            "is approximately spherical."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Flat_Earth",
    },


    "sun_star": {

        "title": "Sun",

        "text": (
            "The Sun is the star located at the "
            "centre of the Solar System. It is a "
            "massive sphere of hot plasma and is "
            "classified as a G-type main-sequence star."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Sun",
    },


    "india_country": {

        "title": "India",

        "text": (
            "India is a country in South Asia. "
            "It is a federal republic and one of "
            "the world's most populous countries."
        ),

        "source":
            "https://en.wikipedia.org/wiki/India",
    },


    "india_developed": {

        "title":
            "India — World Bank / IMF classification",

        "text": (
            "India is classified by the World Bank "
            "as a lower-middle-income economy. "
            "The IMF classifies India within emerging "
            "market and developing economies. Therefore, "
            "the claim that India is a developed country "
            "is not supported by these classifications."
        ),

        "source":
            "https://www.worldbank.org/en/country/india",
    },


    "earth_sun_orbit": {

        "title": "Earth",

        "text": (
            "Earth orbits the Sun. Earth completes "
            "one revolution around the Sun in "
            "approximately 365.25 days."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Earth",
    },


    "sun_earth_orbit": {

        "title": "Solar System",

        "text": (
            "Earth orbits the Sun, which is the "
            "central star of the Solar System."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Solar_System",
    },


    "water_freezing": {

        "title": "Water",

        "text": (
            "The freezing point of pure water is "
            "0 degrees Celsius at standard "
            "atmospheric pressure."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Freezing_point",
    },


    "water_boiling": {

        "title": "Water boiling point",

        "text": (
            "The normal boiling point of water is "
            "100 degrees Celsius at one standard "
            "atmosphere."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Boiling_point",
    },


    "water_survival": {

        "title": "Water and human body",

        "text": (
            "Water is essential for life. Humans "
            "need water for essential functions "
            "and must consume water to survive."
        ),

        "source": (
            "https://www.usgs.gov/"
            "water-science-school/"
            "science/water-you-water-and-human-body"
        ),
    },


    "vaccine_microchips": {

        "title":
            "COVID-19 vaccine ingredients",

        "text": (
            "COVID-19 vaccines do not contain "
            "microchips and are not designed to "
            "track people's movement."
        ),

        "source":
            "https://stacks.cdc.gov/view/cdc/109015",
    },


    "moon_cheese": {

        "title": "Moon",

        "text": (
            "The Moon is a rocky world with a crust, "
            "mantle, and core. Lunar rocks and minerals "
            "make up the Moon; it is not made entirely "
            "of cheese."
        ),

        "source":
            "https://science.nasa.gov/solar-system/moon/",
    },


    "oxygen_survival": {

        "title": "Oxygen",

        "text": (
            "Humans need oxygen for cellular respiration "
            "and normal survival. Oxygen is used by cells "
            "to release energy from nutrients during "
            "aerobic respiration."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Oxygen",
    },


    "pacific_larger_atlantic": {

        "title": "Pacific Ocean",

        "text": (
            "The Pacific Ocean is the largest and "
            "deepest of Earth's oceanic divisions. "
            "It is larger in area than the Atlantic Ocean."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Pacific_Ocean",
    },


    "exercise_cardiovascular": {

        "title": "Exercise",

        "text": (
            "Regular physical activity and exercise "
            "can improve cardiovascular health and "
            "reduce the risk of several chronic diseases."
        ),

        "source":
            "https://en.wikipedia.org/wiki/Exercise",
    },
}


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {

    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "and",
    "or",
    "but",
    "by",
    "as",
    "into",
    "about",
    "than",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "their",
    "there",
    "here",
    "they",
    "them",
    "then",
    "also",
    "very",
    "can",
    "completely",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace(
        "’",
        "'",
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def canonical_claim(claim):

    return (
        normalize_text(claim)
        .rstrip(".!?")
        .strip()
    )


def extract_terms(text):

    return [
        word
        for word in normalize_text(text).split()
        if (
            len(word) > 2
            and word not in STOPWORDS
        )
    ]


# ============================================================
# CANONICAL CLAIM TYPE
# ============================================================

def canonical_type(claim):

    c = canonical_claim(
        claim
    )

    if c in {
        "the earth is flat",
        "earth is flat",
    }:
        return "earth_flat"

    if c in {
        "the sun is a star",
        "sun is a star",
    }:
        return "sun_star"

    if c in {
        "india is a country",
        "india is a country in south asia",
    }:
        return "india_country"

    if (
        "india is a developed country"
        in c
    ):

        return "india_developed"

    if (
        "india is developed"
        in c
    ):

        return "india_developed"

    if c in {
        "the earth orbits the sun",
        "earth orbits the sun",
        "the earth revolves around the sun",
        "earth revolves around the sun",
    }:
        return "earth_sun_orbit"

    if c in {
        "the sun is orbited by the earth",
        "sun is orbited by earth",
    }:
        return "sun_earth_orbit"

    if (
        "water freezes at 0"
        in c
        or
        "water freezing point is 0"
        in c
        or
        "water freezes at zero"
        in c
    ):
        return "water_freezing"

    if (
        "water boils at 100"
        in c
        or
        "water boiling point is 100"
        in c
        or
        "water boils at one hundred"
        in c
    ):
        return "water_boiling"

    if (
        "water is essential for life"
        in c
        or
        "humans need water to survive"
        in c
    ):
        return "water_survival"

    if (
        "vaccine"
        in c
        and
        "microchip"
        in c
    ):
        return "vaccine_microchips"

    if (
        "moon"
        in c
        and
        "cheese"
        in c
    ):
        return "moon_cheese"

    if (
        "oxygen"
        in c
        and
        (
            "survive"
            in c
            or
            "human"
            in c
        )
    ):
        return "oxygen_survival"

    if (
        "pacific"
        in c
        and
        "atlantic"
        in c
    ):
        return "pacific_larger_atlantic"

    if (
        "exercise"
        in c
        and
        (
            "cardiovascular"
            in c
            or
            "heart"
            in c
        )
    ):
        return "exercise_cardiovascular"

    return "generic"


# ============================================================
# PRIORITY WIKIPEDIA TITLES
# ============================================================

def priority_titles(claim):

    kind = canonical_type(
        claim
    )

    mapping = {

        "earth_flat": [
            "Flat Earth",
            "Earth",
        ],

        "sun_star": [
            "Sun",
        ],

        "india_country": [
            "India",
        ],

        "india_developed": [
            "India",
        ],

        "earth_sun_orbit": [
            "Earth",
            "Solar System",
            "Sun",
        ],

        "sun_earth_orbit": [
            "Earth",
            "Solar System",
            "Sun",
        ],

        "water_freezing": [
            "Water",
        ],

        "water_boiling": [
            "Water",
        ],

        "water_survival": [
            "Water",
        ],

        "vaccine_microchips": [
            "COVID-19 vaccine",
            "Vaccine",
        ],

        "moon_cheese": [
            "Moon",
        ],

        "oxygen_survival": [
            "Oxygen",
        ],

        "pacific_larger_atlantic": [
            "Pacific Ocean",
            "Atlantic Ocean",
        ],

        "exercise_cardiovascular": [
            "Exercise",
        ],
    }

    return mapping.get(
        kind,
        [],
    )


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def generate_search_queries(
    claim
):

    kind = canonical_type(
        claim
    )

    queries = [
        claim
    ]

    terms = extract_terms(
        claim
    )

    if len(terms) >= 2:

        queries.append(
            " ".join(terms)
        )

    extra = {

        "earth_flat": [
            "Flat Earth",
            "Earth shape",
            "Earth spherical",
        ],

        "sun_star": [
            "Sun star",
            "Sun astronomy",
        ],

        "india_country": [
            "India country",
            "India South Asia",
        ],

        "india_developed": [
            "India developed country",
            "India World Bank classification",
            "India IMF developing economy",
        ],

        "earth_sun_orbit": [
            "Earth orbit Sun",
            "Earth revolves around Sun",
            "Earth Solar System",
        ],

        "sun_earth_orbit": [
            "Earth orbits Sun",
            "Solar System",
        ],

        "water_freezing": [
            "Water freezing point",
            "Water freezes 0 Celsius",
            "Freezing point water",
        ],

        "water_boiling": [
            "Water boiling point",
            "Water boils 100 Celsius",
            "Normal boiling point water",
        ],

        "water_survival": [
            "Water human survival",
            "Water essential human life",
            "Water human body",
        ],

        "vaccine_microchips": [
            "COVID-19 vaccine microchips",
            "vaccines microchips",
            "vaccine ingredients",
        ],

        "moon_cheese": [
            "Moon composition",
            "Moon made of cheese",
            "Moon rocks composition",
        ],

        "oxygen_survival": [
            "oxygen human survival",
            "oxygen cellular respiration",
        ],

        "pacific_larger_atlantic": [
            "Pacific Ocean Atlantic Ocean area",
            "largest ocean Pacific",
        ],

        "exercise_cardiovascular": [
            "exercise cardiovascular health",
            "physical activity heart health",
        ],

    }.get(
        kind,
        [],
    )

    queries.extend(
        extra
    )

    normalized = normalize_text(
        claim
    )

    if "herbal" in normalized:

        queries.extend([
            "herbal medicine",
            "herbal remedies",
        ])

    if "disease" in normalized:

        queries.extend([
            "disease prevention",
            "disease evidence",
        ])

    if "prevent" in normalized:

        queries.extend([
            "disease prevention",
        ])

    cleaned = []

    for query in queries:

        query = query.strip()

        if (
            query
            and
            query not in cleaned
        ):

            cleaned.append(
                query
            )

    return cleaned[:10]


# ============================================================
# RELEVANCE CALCULATION
# ============================================================

def calculate_relevance(
    claim,
    title,
    evidence,
):

    kind = canonical_type(
        claim
    )

    title_lower = (
        title
        .lower()
        .strip()
    )

    normalized = normalize_text(
        evidence
    )

    # --------------------------------------------------------
    # Exact canonical title matches
    # --------------------------------------------------------

    if (
        kind == "earth_flat"
        and
        title_lower == "flat earth"
    ):

        return 100.0

    if (
        kind == "sun_star"
        and
        title_lower == "sun"
    ):

        return 100.0

    if (
        kind == "india_country"
        and
        title_lower == "india"
    ):

        return 100.0

    if (
        kind == "india_developed"
        and
        title_lower == "india"
    ):

        return 100.0

    if (
        kind
        in {
            "earth_sun_orbit",
            "sun_earth_orbit",
        }
        and
        title_lower
        in {
            "earth",
            "solar system",
            "sun",
        }
    ):

        return 95.0

    if (
        kind
        in {
            "water_freezing",
            "water_boiling",
            "water_survival",
        }
        and
        title_lower == "water"
    ):

        return 95.0

    if (
        kind == "vaccine_microchips"
        and
        (
            "vaccine"
            in title_lower
            or
            "covid"
            in title_lower
        )
    ):

        return 95.0

    if (
        kind == "moon_cheese"
        and
        title_lower == "moon"
    ):

        return 95.0

    if (
        kind == "oxygen_survival"
        and
        title_lower == "oxygen"
    ):

        return 95.0

    if (
        kind == "pacific_larger_atlantic"
        and
        title_lower == "pacific ocean"
    ):

        return 95.0

    if (
        kind == "exercise_cardiovascular"
        and
        title_lower == "exercise"
    ):

        return 95.0

    # --------------------------------------------------------
    # Keyword overlap
    # --------------------------------------------------------

    claim_terms = set(
        extract_terms(claim)
    )

    title_terms = set(
        extract_terms(title)
    )

    evidence_terms = set(
        extract_terms(evidence)
    )

    if not claim_terms:

        return 0.0

    title_overlap = (
        len(
            claim_terms
            &
            title_terms
        )
        /
        max(
            len(claim_terms),
            1,
        )
    )

    evidence_overlap = (
        len(
            claim_terms
            &
            evidence_terms
        )
        /
        max(
            len(claim_terms),
            1,
        )
    )

    score = (
        title_overlap * 55.0
        +
        evidence_overlap * 45.0
    )

    # --------------------------------------------------------
    # Strong semantic patterns
    # --------------------------------------------------------

    patterns = {

        "earth_flat": [
            "scientifically disproven",
            "roughly spherical",
            "earth sphericity",
            "earth is spherical",
        ],

        "sun_star": [
            "sun is the star",
            "main sequence star",
            "g type star",
        ],

        "india_country": [
            "india is a country",
            "country in south asia",
        ],

        "india_developed": [
            "lower middle income",
            "emerging market",
            "developing economies",
            "developing economy",
        ],

        "earth_sun_orbit": [
            "earth orbits the sun",
            "earth revolves around the sun",
            "one revolution around the sun",
        ],

        "sun_earth_orbit": [
            "earth orbits the sun",
            "sun is the central star",
        ],

        "water_freezing": [
            "freezing point",
            "0 degrees celsius",
        ],

        "water_boiling": [
            "boiling point",
            "100 degrees celsius",
        ],

        "water_survival": [
            "water is essential for life",
            "water to survive",
            "essential functions",
        ],

        "vaccine_microchips": [
            "do not contain microchips",
            "microchips",
            "track people's movement",
        ],

        "moon_cheese": [
            "not made of cheese",
            "lunar rocks",
            "rocky world",
        ],

        "oxygen_survival": [
            "humans need oxygen",
            "cellular respiration",
            "normal survival",
        ],

        "pacific_larger_atlantic": [
            "largest",
            "larger in area",
            "atlantic ocean",
        ],

        "exercise_cardiovascular": [
            "cardiovascular health",
            "physical activity",
            "exercise",
        ],
    }

    for pattern in patterns.get(
        kind,
        [],
    ):

        if pattern in normalized:

            score = max(
                score,
                95.0,
            )

    return round(
        min(
            score,
            100.0,
        ),
        2,
    )


# ============================================================
# WIKIPEDIA REQUEST
# ============================================================

def wikipedia_get(
    params
):

    cache_key = str(
        sorted(
            params.items()
        )
    )

    if cache_key in SEARCH_CACHE:

        return SEARCH_CACHE[
            cache_key
        ]

    time.sleep(
        REQUEST_DELAY
    )

    try:

        response = requests.get(
            WIKIPEDIA_API,
            params=params,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        SEARCH_CACHE[
            cache_key
        ] = data

        return data

    except Exception as error:

        print(
            "Wikipedia request failed:",
            repr(error),
        )

        return None


# ============================================================
# WIKIPEDIA PAGE
# ============================================================

def wikipedia_page_by_title(
    title
):

    key = (
        title
        .lower()
        .strip()
    )

    if key in PAGE_CACHE:

        return PAGE_CACHE[
            key
        ]

    data = wikipedia_get({

        "action": "query",

        "format": "json",

        "prop": "extracts",

        "explaintext": 1,

        "exintro": 1,

        "redirects": 1,

        "titles": title,

    })

    if not data:

        return None

    pages = (
        data
        .get(
            "query",
            {},
        )
        .get(
            "pages",
            {},
        )
    )

    for page in pages.values():

        pageid = page.get(
            "pageid"
        )

        extract = (
            page.get(
                "extract",
                "",
            )
            or ""
        )

        if (
            pageid
            and
            extract.strip()
        ):

            result = {

                "pageid":
                    pageid,

                "title":
                    page.get(
                        "title",
                        title,
                    ),

                "extract":
                    extract,

                "cached":
                    False,
            }

            PAGE_CACHE[
                key
            ] = result

            return result

    return None


# ============================================================
# WIKIPEDIA SEARCH
# ============================================================

def wikipedia_search(
    query
):

    data = wikipedia_get({

        "action": "query",

        "format": "json",

        "list": "search",

        "srsearch": query,

        "srlimit": 8,

        "srprop": "snippet",

    })

    if not data:

        return []

    return (
        data
        .get(
            "query",
            {},
        )
        .get(
            "search",
            [],
        )
    )


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve_evidence(
    claim
):

    print("-" * 70)

    print(
        "Retrieving evidence for:",
        claim,
    )

    candidates = []

    seen_titles = set()

    kind = canonical_type(
        claim
    )

    # --------------------------------------------------------
    # 1. CANONICAL EVIDENCE
    # --------------------------------------------------------

    local = CANONICAL_EVIDENCE.get(
        kind
    )

    if local:

        candidates.append({

            "title":
                local["title"],

            "text":
                local["text"],

            "source":
                local["source"],

            "relevance":
                100.0,

            "cached":
                True,
        })

        seen_titles.add(
            local["title"]
            .lower()
        )

        print(
            "Canonical evidence:",
            local["title"],
        )

    # --------------------------------------------------------
    # 2. PRIORITY WIKIPEDIA PAGES
    # --------------------------------------------------------

    for title in priority_titles(
        claim
    ):

        if len(candidates) >= 8:

            break

        if (
            title.lower()
            in seen_titles
        ):

            continue

        page = (
            wikipedia_page_by_title(
                title
            )
        )

        if not page:

            continue

        relevance = (
            calculate_relevance(
                claim,
                page["title"],
                page["extract"],
            )
        )

        candidates.append({

            "title":
                page["title"],

            "text":
                page["extract"],

            "source": (
                "https://en.wikipedia.org/"
                f"?curid={page['pageid']}"
            ),

            "relevance":
                relevance,

            "cached":
                page.get(
                    "cached",
                    False,
                ),
        })

        seen_titles.add(
            page["title"]
            .lower()
        )

    # --------------------------------------------------------
    # 3. SEARCH
    # --------------------------------------------------------

    queries = (
        generate_search_queries(
            claim
        )
    )

    print(
        "Wikipedia queries:",
        queries,
    )

    for query in queries:

        if len(candidates) >= 10:

            break

        results = wikipedia_search(
            query
        )

        print(
            "Search:",
            query,
            "Results:",
            len(results),
        )

        for result in results:

            if len(candidates) >= 10:

                break

            title = (
                result
                .get(
                    "title",
                    "",
                )
                .strip()
            )

            if not title:

                continue

            if (
                title.lower()
                in seen_titles
            ):

                continue

            page = (
                wikipedia_page_by_title(
                    title
                )
            )

            if page:

                text = page[
                    "extract"
                ]

                pageid = page[
                    "pageid"
                ]

                final_title = page[
                    "title"
                ]

            else:

                snippet = (
                    result
                    .get(
                        "snippet",
                        "",
                    )
                )

                snippet = re.sub(
                    r"<[^>]+>",
                    "",
                    snippet,
                )

                snippet = snippet.strip()

                if not snippet:

                    continue

                text = snippet

                pageid = None

                final_title = title

            relevance = (
                calculate_relevance(
                    claim,
                    final_title,
                    text,
                )
            )

            if relevance < 8.0:

                continue

            if pageid:

                source = (
                    "https://en.wikipedia.org/"
                    f"?curid={pageid}"
                )

            else:

                source = (
                    "https://en.wikipedia.org/"
                )

            candidates.append({

                "title":
                    final_title,

                "text":
                    text,

                "source":
                    source,

                "relevance":
                    relevance,

                "cached":
                    False,
            })

            seen_titles.add(
                final_title.lower()
            )

    candidates.sort(
        key=lambda item:
            item.get(
                "relevance",
                0.0,
            ),
        reverse=True,
    )

    return candidates[:10]


# ============================================================
# SOURCE CLASSIFICATION
# ============================================================

def classify_source(
    source,
    title=""
):

    source = str(
        source
    ).lower()

    title = str(
        title
    ).lower()

    if any(
        x in source
        for x in (
            ".gov",
            ".gov.in",
            ".nic.in",
        )
    ):

        return "GOVERNMENT"

    if any(
        x in source
        for x in (
            ".edu",
            ".ac.uk",
            ".ac.in",
        )
    ):

        return "ACADEMIC"

    if any(
        x in source
        for x in (
            "pubmed",
            "ncbi.nlm.nih.gov",
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "ieee.org",
            "nih.gov",
        )
    ):

        return "PEER_REVIEWED"

    if (
        "wikipedia.org"
        in source
        or
        "wikipedia"
        in title
    ):

        return "WIKIPEDIA"

    if any(
        x in source
        for x in (
            "worldbank.org",
            "imf.org",
        )
    ):

        return "ORGANIZATION"

    if any(
        x in source
        for x in (
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "bbc.co.uk",
            "theguardian.com",
        )
    ):

        return "REPUTABLE_NEWS"

    return "UNKNOWN"


# ============================================================
# SOURCE RELIABILITY
# ============================================================

def calculate_source_reliability(
    source_url,
    title=""
):

    source_type = classify_source(
        source_url,
        title,
    )

    score = SOURCE_RELIABILITY_SCORES.get(
        source_type,
        SOURCE_RELIABILITY_SCORES[
            "UNKNOWN"
        ],
    )

    if score >= 85:

        level = "HIGH"

    elif score >= 65:

        level = "MEDIUM"

    else:

        level = "LOW"

    return {

        "type":
            source_type,

        "score":
            round(
                score,
                2,
            ),

        "level":
            level,
    }


# ============================================================
# EVIDENCE QUALITY
# ============================================================

def calculate_evidence_quality(
    relevance,
    verification,
    source_reliability,
):

    relevance_score = max(
        0.0,
        min(
            float(relevance),
            100.0,
        ),
    )

    verification_score = max(
        0.0,
        min(
            float(
                verification.get(
                    "confidence",
                    0.0,
                )
            ),
            100.0,
        ),
    )

    reliability_score = max(
        0.0,
        min(
            float(
                source_reliability.get(
                    "score",
                    50.0,
                )
            ),
            100.0,
        ),
    )

    quality = (
        verification_score * 0.50
        +
        relevance_score * 0.30
        +
        reliability_score * 0.20
    )

    return round(
        quality,
        2,
    )


# ============================================================
# DETERMINISTIC VERIFICATION
# ============================================================

def deterministic_verification(
    claim,
    evidence,
):

    c = canonical_claim(
        claim
    )

    e = normalize_text(
        evidence
    )

    # --------------------------------------------------------
    # INDIA IS A COUNTRY
    # --------------------------------------------------------

    if c in {
        "india is a country",
        "india is a country in south asia",
    }:

        if (
            "india is a country"
            in e
            or
            "country in south asia"
            in e
        ):

            return {

                "status":
                    "SUPPORTED",

                "confidence":
                    99.0,

                "supported_score":
                    99.0,

                "contradicted_score":
                    0.0,

                "neutral_score":
                    1.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # SUN IS A STAR
    # --------------------------------------------------------

    if c in {
        "the sun is a star",
        "sun is a star",
    }:

        if (
            "sun is the star"
            in e
            or
            "sun is a star"
            in e
            or
            "g type"
            in e
        ):

            return {

                "status":
                    "SUPPORTED",

                "confidence":
                    99.0,

                "supported_score":
                    99.0,

                "contradicted_score":
                    0.0,

                "neutral_score":
                    1.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # EARTH IS FLAT
    # --------------------------------------------------------

    if c in {
        "the earth is flat",
        "earth is flat",
    }:

        if (
            "scientifically disproven"
            in e
            or
            "roughly spherical"
            in e
            or
            "earth sphericity"
            in e
            or
            "earth is spherical"
            in e
        ):

            return {

                "status":
                    "CONTRADICTED",

                "confidence":
                    99.0,

                "supported_score":
                    0.0,

                "contradicted_score":
                    99.0,

                "neutral_score":
                    1.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # INDIA DEVELOPED
    # --------------------------------------------------------

    if (
        "india"
        in c
        and
        (
            "developed"
            in c
            or
            "developing"
            in c
        )
    ):

        if (
            "lower middle income"
            in e
            or
            "emerging market"
            in e
            or
            "developing economies"
            in e
            or
            "developing economy"
            in e
        ):

            return {

                "status":
                    "CONTRADICTED",

                "confidence":
                    95.0,

                "supported_score":
                    0.0,

                "contradicted_score":
                    95.0,

                "neutral_score":
                    5.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # EARTH ORBITS SUN
    # --------------------------------------------------------

    if (
        c
        in {
            "the earth orbits the sun",
            "earth orbits the sun",
            "the earth revolves around the sun",
            "earth revolves around the sun",
        }
    ):

        if (
            "earth orbits the sun"
            in e
            or
            "earth revolves around the sun"
            in e
            or
            "one revolution around the sun"
            in e
        ):

            return {

                "status":
                    "SUPPORTED",

                "confidence":
                    99.0,

                "supported_score":
                    99.0,

                "contradicted_score":
                    0.0,

                "neutral_score":
                    1.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # WATER FREEZING
    # --------------------------------------------------------

    if (
        canonical_type(
            claim
        )
        == "water_freezing"
    ):

        if (
            "freezing point"
            in e
            and
            "0 degrees celsius"
            in e
        ):

            return {

                "status":
                    "SUPPORTED",

                "confidence":
                    99.0,

                "supported_score":
                    99.0,

                "contradicted_score":
                    0.0,

                "neutral_score":
                    1.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # WATER BOILING
    # --------------------------------------------------------

    if (
        canonical_type(
            claim
        )
        == "water_boiling"
    ):

        if (
            "boiling point"
            in e
            and
            "100 degrees celsius"
            in e
        ):

            return {

                "status":
                    "SUPPORTED",

                "confidence":
                    99.0,

                "supported_score":
                    99.0,

                "contradicted_score":
                    0.0,

                "neutral_score":
                    1.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # VACCINE MICROCHIPS
    # --------------------------------------------------------

    if (
        canonical_type(
            claim
        )
        == "vaccine_microchips"
    ):

        if (
            "do not contain microchips"
            in e
            or
            "microchips"
            in e
            and
            "track"
            in e
        ):

            return {

                "status":
                    "CONTRADICTED",

                "confidence":
                    95.0,

                "supported_score":
                    0.0,

                "contradicted_score":
                    95.0,

                "neutral_score":
                    5.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # MOON CHEESE
    # --------------------------------------------------------

    if (
        canonical_type(
            claim
        )
        == "moon_cheese"
    ):

        if (
            "not made of cheese"
            in e
            or
            "lunar rocks"
            in e
        ):

            return {

                "status":
                    "CONTRADICTED",

                "confidence":
                    95.0,

                "supported_score":
                    0.0,

                "contradicted_score":
                    95.0,

                "neutral_score":
                    5.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    # --------------------------------------------------------
    # OXYGEN
    # --------------------------------------------------------

    if (
        canonical_type(
            claim
        )
        == "oxygen_survival"
    ):

        if (
            "humans need oxygen"
            in e
            or
            "cellular respiration"
            in e
        ):

            return {

                "status":
                    "SUPPORTED",

                "confidence":
                    95.0,

                "supported_score":
                    95.0,

                "contradicted_score":
                    0.0,

                "neutral_score":
                    5.0,

                "evidence_used":
                    evidence[:3500],

                "verification_method":
                    "deterministic_evidence_match",
            }

    return None


# ============================================================
# SELECT RELEVANT EVIDENCE
# ============================================================

def select_relevant_evidence(
    claim,
    evidence,
):

    text = re.sub(
        r"\s+",
        " ",
        str(evidence),
    ).strip()

    if not text:

        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    claim_terms = set(
        extract_terms(claim)
    )

    claim_norm = normalize_text(
        claim
    )

    scored = []

    for sentence in sentences:

        sentence_norm = (
            normalize_text(
                sentence
            )
        )

        sentence_terms = set(
            extract_terms(
                sentence
            )
        )

        score = len(
            claim_terms
            &
            sentence_terms
        )

        if (
            claim_norm
            and
            claim_norm
            in sentence_norm
        ):

            score += 100

        if (
            claim_norm
            == "india is a country"
            and
            "india is a country"
            in sentence_norm
        ):

            score += 100

        if (
            "sun"
            in claim_norm
            and
            "star"
            in claim_norm
            and
            (
                "sun is the star"
                in sentence_norm
                or
                "sun is a star"
                in sentence_norm
            )
        ):

            score += 100

        if (
            "earth"
            in claim_norm
            and
            "flat"
            in claim_norm
            and
            (
                "scientifically disproven"
                in sentence_norm
                or
                "roughly spherical"
                in sentence_norm
                or
                "earth is spherical"
                in sentence_norm
            )
        ):

            score += 100

        scored.append(
            (
                score,
                sentence,
            )
        )

    scored.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    selected = " ".join(
        sentence
        for _, sentence
        in scored[:4]
    )

    return selected[:3500]


# ============================================================
# NLI VERIFICATION
# ============================================================

def nli_verify(
    claim,
    evidence,
):

    if (
        not NLI_AVAILABLE
        or
        nli_model is None
        or
        nli_tokenizer is None
    ):

        return {

            "status":
                "NEUTRAL",

            "confidence":
                0.0,

            "supported_score":
                0.0,

            "contradicted_score":
                0.0,

            "neutral_score":
                100.0,

            "evidence_used":
                evidence[:3500],

            "verification_method":
                "NLI_DISABLED",
        }

    evidence_used = (
        select_relevant_evidence(
            claim,
            evidence,
        )
    )

    if not evidence_used:

        return {

            "status":
                "NEUTRAL",

            "confidence":
                0.0,

            "supported_score":
                0.0,

            "contradicted_score":
                0.0,

            "neutral_score":
                100.0,

            "evidence_used":
                "",

            "verification_method":
                "NLI",
        }

    try:

        # PyTorch is imported lazily because NLI is disabled by
        # default on low-memory deployments.
        import torch

        inputs = nli_tokenizer(
            evidence_used,
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        inputs = {
            key: value.to(
                DEVICE
            )
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            outputs = (
                nli_model(
                    **inputs
                )
            )

            probabilities = (
                torch.softmax(
                    outputs.logits,
                    dim=-1,
                )[0]
            )

        if (
            NLI_ENTAILMENT_ID
            is None
            or
            NLI_CONTRADICTION_ID
            is None
            or
            NLI_NEUTRAL_ID
            is None
        ):

            raise RuntimeError(
                "NLI labels could not "
                f"be resolved: {NLI_LABEL_IDS}"
            )

        entailment = (
            float(
                probabilities[
                    NLI_ENTAILMENT_ID
                ].item()
            )
            * 100
        )

        contradiction = (
            float(
                probabilities[
                    NLI_CONTRADICTION_ID
                ].item()
            )
            * 100
        )

        neutral = (
            float(
                probabilities[
                    NLI_NEUTRAL_ID
                ].item()
            )
            * 100
        )

        scores = {

            "SUPPORTED":
                entailment,

            "CONTRADICTED":
                contradiction,

            "NEUTRAL":
                neutral,
        }

        status = max(
            scores,
            key=scores.get,
        )

        return {

            "status":
                status,

            "confidence":
                round(
                    scores[status],
                    2,
                ),

            "supported_score":
                round(
                    entailment,
                    2,
                ),

            "contradicted_score":
                round(
                    contradiction,
                    2,
                ),

            "neutral_score":
                round(
                    neutral,
                    2,
                ),

            "evidence_used":
                evidence_used,

            "verification_method":
                "NLI",
        }

    except Exception as error:

        print(
            "NLI verification failed:",
            repr(error),
        )

        return {

            "status":
                "NEUTRAL",

            "confidence":
                0.0,

            "supported_score":
                0.0,

            "contradicted_score":
                0.0,

            "neutral_score":
                100.0,

            "evidence_used":
                evidence_used,

            "verification_method":
                "NLI_ERROR",
        }


# ============================================================
# VERIFY CLAIM AGAINST EVIDENCE
# ============================================================

def verify_claim_against_evidence(
    claim,
    evidence,
):

    deterministic = (
        deterministic_verification(
            claim,
            evidence,
        )
    )

    if deterministic is not None:

        return deterministic

    return nli_verify(
        claim,
        evidence,
    )


# ============================================================
# DISTILBERT ASSESSMENT - INT8 ONNX
# ============================================================

def run_model_assessment(
    content
):

    try:

        encoded = tokenizer(
            content,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {}

        if "input_ids" in ONNX_INPUT_NAMES:

            inputs["input_ids"] = (
                np.asarray(
                    encoded["input_ids"],
                    dtype=np.int64,
                )
            )

        if "attention_mask" in ONNX_INPUT_NAMES:

            inputs["attention_mask"] = (
                np.asarray(
                    encoded["attention_mask"],
                    dtype=np.int64,
                )
            )

        if "token_type_ids" in ONNX_INPUT_NAMES:

            if "token_type_ids" in encoded:

                inputs["token_type_ids"] = (
                    np.asarray(
                        encoded["token_type_ids"],
                        dtype=np.int64,
                    )
                )

            else:

                inputs["token_type_ids"] = (
                    np.zeros_like(
                        encoded["input_ids"],
                        dtype=np.int64,
                    )
                )

        outputs = onnx_session.run(
            None,
            inputs,
        )

        if not outputs:

            raise RuntimeError(
                "ONNX model returned no outputs."
            )

        logits = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        if logits.ndim == 1:

            logits = logits.reshape(
                1,
                -1,
            )

        if logits.ndim != 2:

            raise RuntimeError(
                "Unexpected ONNX logits shape: "
                f"{logits.shape}"
            )

        # Numerically stable softmax.
        shifted = (
            logits
            - np.max(
                logits,
                axis=-1,
                keepdims=True,
            )
        )

        exp_logits = np.exp(
            shifted
        )

        probabilities = (
            exp_logits
            /
            np.sum(
                exp_logits,
                axis=-1,
                keepdims=True,
            )
        )

        predicted_id = int(
            np.argmax(
                probabilities[0]
            )
        )

        confidence = (
            float(
                probabilities[
                    0,
                    predicted_id,
                ]
            )
            * 100.0
        )

        # Read the trained model's label mapping from config.json.
        label = str(
            predicted_id
        )

        config_file = (
            ONNX_MODEL_PATH
            / "config.json"
        )

        if config_file.exists():

            try:

                with open(
                    config_file,
                    "r",
                    encoding="utf-8",
                ) as file:

                    config = json.load(
                        file
                    )

                id2label = (
                    config.get(
                        "id2label",
                        {},
                    )
                )

                label = str(
                    id2label.get(
                        str(predicted_id),
                        id2label.get(
                            predicted_id,
                            label,
                        ),
                    )
                )

            except Exception as error:

                print(
                    "Could not read ONNX label map:",
                    repr(error),
                )

        return {

            "verdict":
                label,

            "confidence":
                round(
                    confidence,
                    2,
                ),

            "model":
                "DistilBERT",

            "runtime":
                "ONNX Runtime INT8",

        }

    except Exception as error:

        print(
            "DistilBERT ONNX assessment failed:",
            repr(error),
        )

        return {

            "verdict":
                "UNKNOWN",

            "confidence":
                0.0,

            "model":
                "DistilBERT",

            "runtime":
                "ONNX Runtime INT8",

        }


# ============================================================
# EVIDENCE ASSESSMENT
# ============================================================

def assess_evidence(
    claim,
    candidates,
):

    if not candidates:

        return {

            "status":
                "INSUFFICIENT_EVIDENCE",

            "score":
                0.0,

            "sources":
                [],

            "best_evidence":
                None,

            "verification":
                None,
        }

    verified_candidates = []

    for candidate in candidates:

        verification = (
            verify_claim_against_evidence(
                claim,
                candidate["text"],
            )
        )

        source_reliability = (
            calculate_source_reliability(
                candidate.get(
                    "source",
                    "",
                ),
                candidate.get(
                    "title",
                    "",
                ),
            )
        )

        evidence_quality = (
            calculate_evidence_quality(
                candidate.get(
                    "relevance",
                    0.0,
                ),
                verification,
                source_reliability,
            )
        )

        item = dict(
            candidate
        )

        item[
            "verification"
        ] = verification

        item[
            "source_reliability"
        ] = source_reliability

        item[
            "evidence_quality"
        ] = evidence_quality

        verified_candidates.append(
            item
        )

    # --------------------------------------------------------
    # Deterministic results
    # --------------------------------------------------------

    explicit_supported = [

        item

        for item
        in verified_candidates

        if (
            item[
                "verification"
            ].get(
                "verification_method"
            )
            ==
            "deterministic_evidence_match"

            and

            item[
                "verification"
            ].get(
                "status"
            )
            ==
            "SUPPORTED"
        )
    ]

    explicit_contradicted = [

        item

        for item
        in verified_candidates

        if (
            item[
                "verification"
            ].get(
                "verification_method"
            )
            ==
            "deterministic_evidence_match"

            and

            item[
                "verification"
            ].get(
                "status"
            )
            ==
            "CONTRADICTED"
        )
    ]

    # --------------------------------------------------------
    # Highest-priority deterministic evidence
    # --------------------------------------------------------

    if explicit_contradicted:

        best = max(

            explicit_contradicted,

            key=lambda item: (

                item.get(
                    "evidence_quality",
                    0.0,
                ),

                item[
                    "verification"
                ].get(
                    "confidence",
                    0.0,
                ),

                item.get(
                    "relevance",
                    0.0,
                ),
            ),
        )

        status = "CONTRADICTED"

    elif explicit_supported:

        best = max(

            explicit_supported,

            key=lambda item: (

                item.get(
                    "evidence_quality",
                    0.0,
                ),

                item[
                    "verification"
                ].get(
                    "confidence",
                    0.0,
                ),

                item.get(
                    "relevance",
                    0.0,
                ),
            ),
        )

        status = "SUPPORTED"

    else:

        usable = [

            item

            for item
            in verified_candidates

            if (
                item[
                    "verification"
                ].get(
                    "status"
                )
                in {
                    "SUPPORTED",
                    "CONTRADICTED",
                }

                and

                item[
                    "verification"
                ].get(
                    "confidence",
                    0.0,
                )
                >=
                MIN_NLI_CONFIDENCE
            )
        ]

        if usable:

            best = max(

                usable,

                key=lambda item: (

                    item.get(
                        "evidence_quality",
                        0.0,
                    ),

                    item[
                        "verification"
                    ].get(
                        "confidence",
                        0.0,
                    ),

                    item.get(
                        "relevance",
                        0.0,
                    ),
                ),
            )

            status = (
                best[
                    "verification"
                ][
                    "status"
                ]
            )

        else:

            best = max(

                verified_candidates,

                key=lambda item: (

                    item.get(
                        "evidence_quality",
                        0.0,
                    ),

                    item.get(
                        "relevance",
                        0.0,
                    ),
                ),
            )

            status = (
                "INSUFFICIENT_EVIDENCE"
            )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    if (
        status
        ==
        "INSUFFICIENT_EVIDENCE"
    ):

        score = 0.0

    else:

        score = float(
            best[
                "verification"
            ].get(
                "confidence",
                0.0,
            )
        )

    # --------------------------------------------------------
    # Sort evidence
    # --------------------------------------------------------

    verified_candidates.sort(

        key=lambda item: (

            item.get(
                "evidence_quality",
                0.0,
            ),

            item.get(
                "relevance",
                0.0,
            ),
        ),

        reverse=True,
    )

    # --------------------------------------------------------
    # Source cards
    # --------------------------------------------------------

    sources = []

    for candidate in (
        verified_candidates[:5]
    ):

        sources.append({

            "title":
                candidate["title"],

            "text":
                candidate[
                    "text"
                ][:3500],

            "source":
                candidate["source"],

            "relevance":
                candidate[
                    "relevance"
                ],

            "verification":
                candidate[
                    "verification"
                ],

            "source_reliability":
                candidate[
                    "source_reliability"
                ],

            "evidence_quality":
                candidate[
                    "evidence_quality"
                ],
        })

    # --------------------------------------------------------
    # Best evidence
    # --------------------------------------------------------

    best_evidence = {

        "title":
            best["title"],

        "text":
            best["text"][:3500],

        "source":
            best["source"],

        "relevance":
            best["relevance"],

        "verification":
            best["verification"],

        "source_reliability":
            best[
                "source_reliability"
            ],

        "evidence_quality":
            best[
                "evidence_quality"
            ],
    }

    print("-" * 70)

    print(
        "FINAL EVIDENCE STATUS:",
        status,
    )

    print(
        "BEST EVIDENCE:",
        best["title"],
    )

    print(
        "RELEVANCE:",
        best["relevance"],
    )

    print(
        "EVIDENCE QUALITY:",
        best[
            "evidence_quality"
        ],
    )

    print(
        "CONFIDENCE:",
        round(
            score,
            2,
        ),
    )

    print("-" * 70)

    return {

        "status":
            status,

        "score":
            round(
                max(
                    0.0,
                    min(
                        score,
                        100.0,
                    ),
                ),
                2,
            ),

        "sources":
            sources,

        "best_evidence":
            best_evidence,

        "verification":
            best[
                "verification"
            ],
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

@app.get("/diagnostics")
def diagnostics():

    return {

        "version":
            "7.3.2",

        "device":
            str(DEVICE),

        "distilbert_loaded":
            onnx_session is not None,

        "nli_loaded":
            NLI_AVAILABLE
            and
            nli_model is not None,

        "nli_disabled":
            DISABLE_NLI,

        "canonical_claim_types":
            sorted(
                CANONICAL_EVIDENCE.keys()
            ),

        "page_cache_size":
            len(
                PAGE_CACHE
            ),

        "search_cache_size":
            len(
                SEARCH_CACHE
            ),

        "nli_labels":
            NLI_LABEL_IDS,

        "source_reliability_enabled":
            True,

        "evidence_quality_enabled":
            True,
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    verification_methods = (
        "Canonical Evidence + Wikipedia + "
        "deterministic verification"
    )

    if NLI_AVAILABLE:

        verification_methods += (
            " + NLI"
        )

    return {

        "message":
            "Welcome to TruthLens AI",

        "status":
            "Backend is running",

        "version":
            "7.3.2",

        "model":
            "DistilBERT",

        "verification":
            verification_methods,

        "decision_policy":
            (
                "Evidence assessment determines "
                "the final verdict"
            ),

        "nli_enabled":
            NLI_AVAILABLE,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "healthy",

        "model_loaded":
            onnx_session is not None,

        "nli_loaded":
            NLI_AVAILABLE,

        "nli_disabled":
            DISABLE_NLI,

        "device":
            str(DEVICE),

        "evidence_enabled":
            True,

        "canonical_evidence":
            True,

        "wikipedia_search":
            True,

        "deterministic_verification":
            True,

        "semantic_verification":
            NLI_AVAILABLE,

        "source_reliability":
            True,

        "evidence_quality":
            True,
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze_content(
    request: AnalyzeRequest,
):

    content = (
        request.content
        .strip()
    )

    if not content:

        return {

            "success":
                False,

            "message":
                "No content was provided.",

            "content":
                "",
        }

    print("=" * 70)

    print(
        "ANALYZING CLAIM:",
        content,
    )

    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 — DistilBERT
    # --------------------------------------------------------

    model_assessment = (
        run_model_assessment(
            content
        )
    )

    # --------------------------------------------------------
    # STEP 2 — Evidence retrieval
    # --------------------------------------------------------

    candidates = (
        retrieve_evidence(
            content
        )
    )

    # --------------------------------------------------------
    # STEP 3 — Evidence assessment
    # --------------------------------------------------------

    evidence_assessment = (
        assess_evidence(
            content,
            candidates,
        )
    )

    evidence_status = (
        evidence_assessment[
            "status"
        ]
    )

    evidence_score = (
        evidence_assessment[
            "score"
        ]
    )

    # --------------------------------------------------------
    # STEP 4 — Final decision
    # --------------------------------------------------------

    if (
        evidence_status
        ==
        "SUPPORTED"
    ):

        final_verdict = (
            "SUPPORTED"
        )

        final_confidence = (
            evidence_score
        )

        basis = (
            "Relevant external evidence "
            "supports the claim."
        )

    elif (
        evidence_status
        ==
        "CONTRADICTED"
    ):

        final_verdict = (
            "CONTRADICTED"
        )

        final_confidence = (
            evidence_score
        )

        basis = (
            "Relevant external evidence "
            "contradicts the claim."
        )

    else:

        final_verdict = (
            "INSUFFICIENT EVIDENCE"
        )

        final_confidence = 0.0

        basis = (
            "Available evidence was not "
            "sufficiently relevant or conclusive."
        )

    # --------------------------------------------------------
    # Evidence basis
    # --------------------------------------------------------

    evidence_basis = (
        "Canonical evidence + Wikipedia "
        "search + relevance ranking + "
        "deterministic matching + "
        "source reliability"
    )

    if NLI_AVAILABLE:

        evidence_basis += (
            " + NLI verification"
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print("=" * 70)

    print(
        "FINAL VERDICT:",
        final_verdict,
    )

    print(
        "FINAL CONFIDENCE:",
        final_confidence,
    )

    print(
        "EVIDENCE SOURCES:",
        len(
            evidence_assessment[
                "sources"
            ]
        ),
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "success":
            True,

        "content":
            content,

        "model_assessment":
            model_assessment,

        "evidence_assessment":
            evidence_assessment,

        "final_assessment": {

            "verdict":
                final_verdict,

            "confidence":
                round(
                    final_confidence,
                    2,
                ),

            "basis":
                basis,
        },

        # ----------------------------------------------------
        # Frontend compatibility
        # ----------------------------------------------------

        "verdict":
            final_verdict,

        "confidence":
            round(
                final_confidence,
                2,
            ),

        "model":
            "DistilBERT",

        "decision_method":
            (
                "Final verdict is determined "
                "by evidence assessment, not "
                "by the DistilBERT ML signal alone."
            ),

        "evidence_basis":
            evidence_basis,

        "nli_enabled":
            NLI_AVAILABLE,
    }


# ============================================================
# STARTUP STATUS
# ============================================================

print("=" * 70)

print(
    "TruthLens AI backend ready."
)

print(
    "DistilBERT:",
    "ACTIVE"
    if onnx_session is not None
    else
    "INACTIVE",
)

print(
    "Model Runtime:",
    "ONNX Runtime INT8",
)

print(
    "Wikipedia Search:",
    "ACTIVE",
)

print(
    "Wikipedia Evidence:",
    "ACTIVE",
)

print(
    "Canonical Evidence:",
    "ACTIVE",
)

print(
    "Deterministic Verification:",
    "ACTIVE",
)

print(
    "NLI Verification:",
    "ACTIVE"
    if NLI_AVAILABLE
    else
    "DISABLED",
)

print(
    "Source Reliability:",
    "ACTIVE",
)

print(
    "Evidence Quality:",
    "ACTIVE",
)

print(
    "India Classification Contradiction Guard:",
    "ACTIVE",
)

print("=" * 70)