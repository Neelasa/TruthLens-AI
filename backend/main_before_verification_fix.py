"""
TruthLens AI Backend
Version 8.0.0

Evidence-first, dynamic misinformation analysis.

Design:
- DistilBERT INT8 ONNX = initial ML signal only.
- Dynamic evidence retrieval = Wikipedia + Wikidata + optional web search.
- Deterministic verifier = fast generic contradiction/support checks.
- Optional NLI = semantic claim/evidence verification when DISABLE_NLI=false.
- Evidence consensus = final verdict is based on evidence, not DistilBERT alone.

Optional environment variables:
    DISABLE_NLI=true|false
    SERPER_API_KEY=<optional>
    TAVILY_API_KEY=<optional>

The backend remains usable without either web-search API key.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer


# ============================================================
# APPLICATION
# ============================================================

APP_VERSION = "8.0.0"

app = FastAPI(
    title="TruthLens AI",
    description=(
        "AI-powered misinformation analysis and "
        "evidence-based decision support system."
    ),
    version=APP_VERSION,
)

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
# PATHS / CONFIG
# ============================================================

#BASE_DIR = Path(__file__).resolve().parent.parent
#BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent

ONNX_MODEL_PATH = BASE_DIR / "models" / "distilbert_truthlens_onnx"
ONNX_MODEL_FILE = ONNX_MODEL_PATH / "model.int8.onnx"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

DEVICE = "cpu"

REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

DISABLE_NLI = (
    os.getenv("DISABLE_NLI", "true").strip().lower() == "true"
)

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

MIN_RELEVANCE = 10.0
MIN_EVIDENCE_QUALITY = 45.0
MIN_NLI_CONFIDENCE = 65.0

HEADERS = {
    "User-Agent": (
        "TruthLensAI/8.0 "
        "(evidence-based misinformation analysis project)"
    ),
    "Accept": "application/json",
}

SOURCE_RELIABILITY_SCORES = {
    "GOVERNMENT": 98.0,
    "ACADEMIC": 97.0,
    "PEER_REVIEWED": 98.0,
    "WIKIDATA": 92.0,
    "REPUTABLE_NEWS": 90.0,
    "ORGANIZATION": 88.0,
    "WIKIPEDIA": 85.0,
    "SEARCH_ENGINE": 65.0,
    "UNKNOWN": 50.0,
}

PAGE_CACHE: Dict[str, Dict[str, Any]] = {}
SEARCH_CACHE: Dict[str, Any] = {}
WIKIDATA_CACHE: Dict[str, Any] = {}


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print(f"TruthLens AI Backend v{APP_VERSION}")
print("=" * 70)
print("Device:", DEVICE)
print("ONNX model:", ONNX_MODEL_FILE)
print("Dynamic Wikipedia:", True)
print("Dynamic Wikidata:", True)
print("Optional Serper search:", bool(SERPER_API_KEY))
print("Optional Tavily search:", bool(TAVILY_API_KEY))
print("NLI disabled:", DISABLE_NLI)
print("=" * 70)


# ============================================================
# DISTILBERT INT8 ONNX
# ============================================================

if not ONNX_MODEL_FILE.exists():
    raise FileNotFoundError(
        f"INT8 ONNX model not found: {ONNX_MODEL_FILE}"
    )

tokenizer = AutoTokenizer.from_pretrained(
    str(ONNX_MODEL_PATH),
    local_files_only=True,
)

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

onnx_session = ort.InferenceSession(
    str(ONNX_MODEL_FILE),
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)

ONNX_INPUT_NAMES = {
    item.name for item in onnx_session.get_inputs()
}

ONNX_OUTPUT_NAMES = [
    item.name for item in onnx_session.get_outputs()
]

print("ONNX inputs:", sorted(ONNX_INPUT_NAMES))
print("ONNX outputs:", ONNX_OUTPUT_NAMES)
print(
    "ONNX model size:",
    round(ONNX_MODEL_FILE.stat().st_size / (1024 * 1024), 2),
    "MB",
)


# ============================================================
# OPTIONAL NLI
# ============================================================

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
nli_tokenizer = None
nli_model = None
NLI_AVAILABLE = False
NLI_LABEL_IDS: Dict[str, int] = {}
NLI_ENTAILMENT_ID = None
NLI_CONTRADICTION_ID = None
NLI_NEUTRAL_ID = None

if not DISABLE_NLI:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification

        print("Loading NLI model:", NLI_MODEL_NAME)

        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME
        )
        nli_model.to(DEVICE)
        nli_model.eval()
        NLI_AVAILABLE = True

    except Exception as exc:
        print("NLI load failed:", repr(exc))
        nli_tokenizer = None
        nli_model = None
        NLI_AVAILABLE = False


def build_nli_label_ids():
    if not NLI_AVAILABLE or nli_model is None:
        return {}, None, None, None

    labels = {
        str(index): str(label).lower().strip()
        for index, label in nli_model.config.id2label.items()
    }

    entailment = None
    contradiction = None
    neutral = None

    for index, label in labels.items():
        if "entail" in label:
            entailment = int(index)
        elif "contrad" in label:
            contradiction = int(index)
        elif "neutral" in label:
            neutral = int(index)

    if None in (entailment, contradiction, neutral):
        # Common MNLI label order.
        if len(labels) == 3:
            contradiction, neutral, entailment = 0, 1, 2

    return labels, entailment, contradiction, neutral


(
    NLI_LABEL_IDS,
    NLI_ENTAILMENT_ID,
    NLI_CONTRADICTION_ID,
    NLI_NEUTRAL_ID,
) = build_nli_label_ids()

print(
    "NLI:",
    "ACTIVE" if NLI_AVAILABLE else "DISABLED",
)


# ============================================================
# TEXT HELPERS
# ============================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "of",
    "to", "in", "on", "at", "for", "from", "with", "and", "or",
    "but", "by", "as", "into", "about", "than", "this", "that",
    "these", "those", "it", "its", "their", "there", "here",
    "they", "them", "then", "also", "very", "can", "could",
    "would", "should", "completely", "currently", "current",
}


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\s.%/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_display_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonical_claim(claim: str) -> str:
    return normalize_text(claim).rstrip(".!?").strip()


def extract_terms(text: str) -> List[str]:
    return [
        w
        for w in normalize_text(text).split()
        if len(w) > 2 and w not in STOPWORDS
    ]


def sentence_split(text: str) -> List[str]:
    text = clean_display_text(text)
    if not text:
        return []
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ============================================================
# CLAIM STRUCTURE
# ============================================================

def extract_claim_entities(claim: str) -> List[str]:
    """
    Lightweight entity extraction. It intentionally does not
    pretend to be a full NER system. It extracts useful proper
    noun phrases for retrieval.
    """
    original = clean_display_text(claim)

    phrases = []

    # Capitalized multi-word names.
    for match in re.finditer(
        r"\b(?:[A-Z][A-Za-z.'-]*\s+){1,4}[A-Z][A-Za-z.'-]*\b",
        original,
    ):
        phrases.append(match.group(0).strip(" .,!?"))

    # Text after 'of' is often a location/entity.
    match = re.search(
        r"\b(?:of|in|from)\s+([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})",
        original,
    )
    if match:
        phrases.append(match.group(1).strip(" .,!?"))

    return unique_keep_order(phrases)[:8]


def detect_claim_type(claim: str) -> str:
    c = canonical_claim(claim)

    if re.search(r"\bcurrent\b.*\b(?:is|was)\b", c):
        return "CURRENT_FACT"

    if re.search(
        r"\b(?:who|which)\b.*\b(?:is|was)\b", c
    ):
        return "IDENTITY"

    if any(x in c for x in ["won", "winner", "champion"]):
        return "HISTORICAL_RESULT"

    if any(x in c for x in ["founded", "established", "created"]):
        return "DATE_OR_FOUNDING"

    if any(x in c for x in ["born", "birth"]):
        return "BIOGRAPHICAL"

    if any(x in c for x in ["died", "death"]):
        return "BIOGRAPHICAL"

    if any(x in c for x in ["is a", "is an", "was a", "was an"]):
        return "DEFINITION_OR_IDENTITY"

    if any(x in c for x in ["%", "percent", "million", "billion"]):
        return "NUMERIC"

    if any(x in c for x in ["causes", "prevents", "cures", "increases", "reduces"]):
        return "CAUSAL_OR_MEDICAL"

    return "GENERAL"


def parse_relation_claim(claim: str) -> Optional[Dict[str, str]]:
    """
    Extract generic subject/verb/object claims.

    Supports common factual relations such as:
      - "India is a country."
      - "Kurnool was the capital of Andhra State from 1953 to 1956."
      - "India won the 1983 Cricket World Cup."
      - "The company founded the project in 2020."

    This is intentionally generic and does not contain hard-coded facts.
    """
    original = clean_display_text(claim)

    # Remove discourse/filler words that should not become part of the
    # factual object, e.g. "was indeed the capital" -> "was the capital".
    filler_pattern = (
        r"\b(?:indeed|actually|reportedly|apparently|certainly|"
        r"clearly|obviously|supposedly|allegedly|truly)\b"
    )
    normalized_claim = re.sub(
        filler_pattern,
        " ",
        original,
        flags=re.I,
    )
    normalized_claim = re.sub(r"\s+", " ", normalized_claim).strip()

    # Ordered from more specific/multi-word verbs to simpler copulas.
    relation_verbs = [
        "was founded by",
        "were founded by",
        "was established by",
        "were established by",
        "was created by",
        "were created by",
        "was invented by",
        "were invented by",
        "was discovered by",
        "were discovered by",
        "was launched by",
        "were launched by",
        "is located in",
        "are located in",
        "was located in",
        "were located in",
        "is known as",
        "was known as",
        "are known as",
        "were known as",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "won",
        "wins",
        "defeated",
        "defeats",
        "hosted",
        "hosts",
        "founded",
        "invented",
        "discovered",
        "launched",
        "served",
        "became",
        "held",
        "ruled",
        "led",
        "created",
        "built",
        "established",
        "introduced",
        "declared",
        "appointed",
    ]

    # Longest phrases first prevents "was" from matching inside
    # "was founded by", etc.
    relation_verbs = sorted(
        relation_verbs,
        key=len,
        reverse=True,
    )
    verb_pattern = "|".join(
        re.escape(v) for v in relation_verbs
    )

    pattern = (
        rf"^(?:the\s+)?(.+?)\s+({verb_pattern})\s+"
        rf"(.+?)[.!?]?$"
    )
    match = re.match(
        pattern,
        normalized_claim,
        re.I,
    )

    if not match:
        return None

    subject = clean_display_text(match.group(1))
    verb = clean_display_text(match.group(2)).lower()
    obj = clean_display_text(match.group(3))

    if len(subject) < 2 or len(obj) < 2:
        return None

    return {
        "subject": subject,
        "verb": verb,
        "object": obj,
        "claim_type": detect_claim_type(claim),
    }


# ============================================================
# GENERIC SEARCH QUERY GENERATION
# ============================================================

def generate_search_queries(claim: str) -> List[str]:
    original = clean_display_text(claim)
    normalized = canonical_claim(claim)
    terms = extract_terms(claim)
    entities = extract_claim_entities(claim)
    relation = parse_relation_claim(claim)

    queries = [original]

    if relation:
        queries.extend(
            [
                f"{relation['subject']} {relation['object']}",
                f"{relation['subject']} {relation['object']} fact",
            ]
        )

    if entities:
        queries.extend(entities)

    if len(terms) >= 2:
        queries.append(" ".join(terms))

    if "current" in normalized:
        queries.extend(
            [
                f"{original} current",
                f"{' '.join(terms)} latest",
                f"{' '.join(terms)} official",
            ]
        )

    if "who" in normalized:
        queries.append(f"{' '.join(terms)} official")

    if any(
        word in normalized
        for word in ["health", "disease", "medicine", "vaccine", "drug"]
    ):
        queries.extend(
            [
                f"{original} scientific evidence",
                f"{original} study",
            ]
        )

    return unique_keep_order(queries)[:10]


# ============================================================
# HTTP HELPERS
# ============================================================

def cached_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    cache: Optional[Dict[str, Any]] = None,
):
    cache = cache if cache is not None else SEARCH_CACHE
    key = url + "|" + json.dumps(params or {}, sort_keys=True)

    if key in cache:
        return cache[key]

    try:
        time.sleep(REQUEST_DELAY)
        response = requests.get(
            url,
            params=params,
            headers=headers or HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        cache[key] = data
        return data
    except Exception as exc:
        print("HTTP GET failed:", url, repr(exc))
        return None


def cached_post_json(
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    cache: Optional[Dict[str, Any]] = None,
):
    cache = cache if cache is not None else SEARCH_CACHE
    key = url + "|" + json.dumps(data, sort_keys=True)

    if key in cache:
        return cache[key]

    try:
        time.sleep(REQUEST_DELAY)
        response = requests.post(
            url,
            json=data,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        cache[key] = result
        return result
    except Exception as exc:
        print("HTTP POST failed:", url, repr(exc))
        return None


# ============================================================
# WIKIPEDIA
# ============================================================

def wikipedia_search(query: str) -> List[Dict[str, Any]]:
    data = cached_get_json(
        WIKIPEDIA_API,
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 8,
            "srprop": "snippet",
        },
    )

    if not data:
        return []

    return data.get("query", {}).get("search", [])


def wikipedia_page_by_title(title: str) -> Optional[Dict[str, Any]]:
    key = normalize_text(title)

    if key in PAGE_CACHE:
        return PAGE_CACHE[key]

    data = cached_get_json(
        WIKIPEDIA_API,
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "explaintext": 1,
            "exchars": 9000,
            "inprop": "url",
            "redirects": 1,
            "titles": title,
        },
    )

    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})

    for page in pages.values():
        extract = clean_display_text(page.get("extract", ""))
        pageid = page.get("pageid")

        if not pageid or not extract:
            continue

        result = {
            "title": page.get("title", title),
            "extract": extract,
            "pageid": pageid,
            "url": (
                page.get("fullurl")
                or f"https://en.wikipedia.org/?curid={pageid}"
            ),
        }
        PAGE_CACHE[key] = result
        return result

    return None


def wikipedia_candidates(claim: str) -> List[Dict[str, Any]]:
    candidates = []
    seen = set()

    for query in generate_search_queries(claim):
        for result in wikipedia_search(query):
            title = clean_display_text(result.get("title", ""))
            if not title:
                continue

            key = normalize_text(title)
            if key in seen:
                continue

            seen.add(key)
            page = wikipedia_page_by_title(title)

            if page:
                text = page["extract"]
                source = page["url"]
            else:
                text = re.sub(
                    r"<[^>]+>",
                    "",
                    result.get("snippet", ""),
                )
                source = "https://en.wikipedia.org/"

            if not text:
                continue

            candidates.append(
                {
                    "title": title,
                    "text": text[:9000],
                    "source": source,
                    "source_kind": "WIKIPEDIA",
                    "retrieval": "wikipedia_search",
                }
            )

            if len(candidates) >= 12:
                return candidates

    return candidates


# ============================================================
# WIKIDATA
# ============================================================

def wikidata_search_entity(
    query: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    data = cached_get_json(
        WIKIDATA_API,
        {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "uselang": "en",
            "format": "json",
            "limit": limit,
        },
        cache=WIKIDATA_CACHE,
    )

    if not data:
        return []

    return data.get("search", [])


def wikidata_entity(qid: str) -> Optional[Dict[str, Any]]:
    data = cached_get_json(
        WIKIDATA_API,
        {
            "action": "wbgetentities",
            "ids": qid,
            "languages": "en",
            "format": "json",
            "props": "labels|descriptions|claims|sitelinks",
        },
        cache=WIKIDATA_CACHE,
    )

    if not data:
        return None

    return data.get("entities", {}).get(qid)


def wikidata_sparql(query: str) -> List[Dict[str, str]]:
    data = cached_get_json(
        WIKIDATA_SPARQL,
        {
            "query": query,
            "format": "json",
        },
        headers={
            **HEADERS,
            "Accept": "application/sparql-results+json",
        },
        cache=WIKIDATA_CACHE,
    )

    if not data:
        return []

    rows = []
    for row in data.get("results", {}).get("bindings", []):
        rows.append(
            {
                key: value.get("value", "")
                for key, value in row.items()
            }
        )
    return rows


def build_wikidata_evidence(
    title: str,
    text: str,
    url: str,
    retrieval: str,
) -> Dict[str, Any]:
    return {
        "title": title,
        "text": clean_display_text(text)[:9000],
        "source": url,
        "source_kind": "WIKIDATA",
        "retrieval": retrieval,
    }


def current_office_wikidata_candidates(
    claim: str,
) -> List[Dict[str, Any]]:
    """
    Handles claims such as:
      'The current Chief Minister of Andhra Pradesh is X.'
      'The current Prime Minister of India is X.'

    It does NOT contain a hard-coded person or office. It searches
    Wikidata for the office and then checks current P39/P1308/P6
    relationships.
    """
    normalized = canonical_claim(claim)

    if "current" not in normalized:
        return []

    relation = parse_relation_claim(claim)
    if not relation:
        return []

    subject = relation["subject"]
    obj = relation["object"]

    office_match = re.search(
        r"(?:current\s+)?(.+?)\s+of\s+(.+)$",
        subject,
        re.I,
    )

    if not office_match:
        return []

    office_phrase = clean_display_text(office_match.group(1))
    jurisdiction = clean_display_text(office_match.group(2))

    # "current" describes the temporal state, not the office name.
    office_phrase = re.sub(
        r"^current\s+",
        "",
        office_phrase,
        flags=re.I,
    ).strip()

    if len(office_phrase) < 3 or len(jurisdiction) < 2:
        return []

    office_queries = [
        f"{office_phrase} of {jurisdiction}",
        f"{office_phrase} {jurisdiction}",
    ]

    office_entities = []
    for q in office_queries:
        office_entities.extend(wikidata_search_entity(q, limit=5))

    office_qids = unique_keep_order(
        [item.get("id", "") for item in office_entities]
    )[:5]

    evidence = []

    # 1. Office item -> P1308 current office holder.
    for qid in office_qids:
        if not qid.startswith("Q"):
            continue

        sparql = f"""
        SELECT ?person ?personLabel WHERE {{
          wd:{qid} wdt:P1308 ?person .
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
          }}
        }}
        LIMIT 10
        """

        rows = wikidata_sparql(sparql)

        for row in rows:
            person = row.get("personLabel", "")
            if not person:
                continue

            evidence.append(
                build_wikidata_evidence(
                    title=(
                        f"Current holder of "
                        f"{office_phrase} of {jurisdiction}"
                    ),
                    text=(
                        f"Wikidata identifies {person} as a "
                        f"current holder of {office_phrase} of "
                        f"{jurisdiction}."
                    ),
                    url=f"https://www.wikidata.org/wiki/{qid}",
                    retrieval="wikidata_P1308",
                )
            )

    # 2. Jurisdiction item -> P6 head of government.
    jurisdiction_entities = []
    for q in [
        jurisdiction,
        f"{jurisdiction} government",
    ]:
        jurisdiction_entities.extend(
            wikidata_search_entity(q, limit=5)
        )

    jurisdiction_qids = unique_keep_order(
        [item.get("id", "") for item in jurisdiction_entities]
    )[:5]

    for qid in jurisdiction_qids:
        if not qid.startswith("Q"):
            continue

        sparql = f"""
        SELECT ?person ?personLabel WHERE {{
          wd:{qid} wdt:P6 ?person .
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
          }}
        }}
        LIMIT 10
        """

        rows = wikidata_sparql(sparql)

        for row in rows:
            person = row.get("personLabel", "")
            if not person:
                continue

            evidence.append(
                build_wikidata_evidence(
                    title=f"Head of government of {jurisdiction}",
                    text=(
                        f"Wikidata identifies {person} as the "
                        f"head of government of {jurisdiction}."
                    ),
                    url=f"https://www.wikidata.org/wiki/{qid}",
                    retrieval="wikidata_P6",
                )
            )

    # 3. Person -> current P39. This catches office records where
    # the office item is not linked through P1308.
    candidate_people = wikidata_search_entity(obj, limit=5)

    for person_entity in candidate_people:
        person_qid = person_entity.get("id", "")
        if not person_qid.startswith("Q"):
            continue

        person_data = wikidata_entity(person_qid)
        if not person_data:
            continue

        claims = person_data.get("claims", {})
        p39 = claims.get("P39", [])

        for statement in p39:
            mainsnak = statement.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})

            target_id = value.get("id")
            if not target_id:
                continue

            qualifiers = statement.get("qualifiers", {})
            end_values = qualifiers.get("P582", [])

            # No end time is the usual Wikidata representation for
            # a current position; also allow a future end time.
            currently_active = not end_values

            if not currently_active:
                for end in end_values:
                    raw = (
                        end.get("datavalue", {})
                        .get("value", {})
                        .get("time", "")
                    )
                    if raw and raw[:5].lstrip("+").isdigit():
                        try:
                            year = int(raw.lstrip("+")[:4])
                            if year >= datetime.now(timezone.utc).year:
                                currently_active = True
                        except ValueError:
                            pass

            if not currently_active:
                continue

            position_entity = wikidata_entity(target_id)
            position_label = (
                position_entity.get("labels", {})
                .get("en", {})
                .get("value", "")
                if position_entity
                else target_id
            )

            if (
                office_phrase.lower() not in position_label.lower()
                and jurisdiction.lower() not in position_label.lower()
            ):
                continue

            person_label = (
                person_data.get("labels", {})
                .get("en", {})
                .get("value", obj)
            )

            evidence.append(
                build_wikidata_evidence(
                    title=f"{person_label} — position held",
                    text=(
                        f"Wikidata records {person_label} as "
                        f"currently holding the position "
                        f"{position_label}."
                    ),
                    url=f"https://www.wikidata.org/wiki/{person_qid}",
                    retrieval="wikidata_P39",
                )
            )

    return evidence[:10]


def wikidata_general_candidates(
    claim: str,
) -> List[Dict[str, Any]]:
    """
    Adds structured entity evidence for arbitrary claims. This is
    deliberately conservative: it does not invent facts from a
    Wikidata label; it only uses labels/descriptions that are
    actually returned.
    """
    candidates = []
    for query in generate_search_queries(claim)[:4]:
        for entity in wikidata_search_entity(query, limit=3):
            qid = entity.get("id", "")
            if not qid.startswith("Q"):
                continue

            label = (
                entity.get("label")
                or entity.get("match", {}).get("text")
                or qid
            )
            description = clean_display_text(
                entity.get("description", "")
            )

            if not description:
                continue

            candidates.append(
                build_wikidata_evidence(
                    title=label,
                    text=f"{label}: {description}.",
                    url=f"https://www.wikidata.org/wiki/{qid}",
                    retrieval="wikidata_entity_search",
                )
            )

    return candidates[:8]


# ============================================================
# OPTIONAL WEB SEARCH
# ============================================================

def serper_candidates(claim: str) -> List[Dict[str, Any]]:
    if not SERPER_API_KEY:
        return []

    data = cached_post_json(
        "https://google.serper.dev/search",
        {
            "q": claim,
            "num": 8,
        },
        headers={
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        },
    )

    if not data:
        return []

    candidates = []
    for item in data.get("organic", []):
        title = clean_display_text(item.get("title", ""))
        snippet = clean_display_text(item.get("snippet", ""))
        link = item.get("link", "")

        if title and snippet and link:
            candidates.append(
                {
                    "title": title,
                    "text": snippet,
                    "source": link,
                    "source_kind": "SEARCH_ENGINE",
                    "retrieval": "serper",
                }
            )

    return candidates[:8]


def tavily_candidates(claim: str) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        return []

    data = cached_post_json(
        "https://api.tavily.com/search",
        {
            "api_key": TAVILY_API_KEY,
            "query": claim,
            "search_depth": "advanced",
            "max_results": 8,
            "include_answer": False,
        },
        headers={
            "Content-Type": "application/json",
        },
    )

    if not data:
        return []

    candidates = []
    for item in data.get("results", []):
        title = clean_display_text(item.get("title", ""))
        content = clean_display_text(item.get("content", ""))
        url = item.get("url", "")

        if title and content and url:
            candidates.append(
                {
                    "title": title,
                    "text": content,
                    "source": url,
                    "source_kind": "SEARCH_ENGINE",
                    "retrieval": "tavily",
                }
            )

    return candidates[:8]


# ============================================================
# SOURCE RELIABILITY
# ============================================================

def classify_source(
    source: str,
    title: str = "",
) -> str:
    source_l = str(source).lower()
    title_l = str(title).lower()

    if any(
        x in source_l
        for x in [
            ".gov",
            ".gov.in",
            ".nic.in",
        ]
    ):
        return "GOVERNMENT"

    if any(
        x in source_l
        for x in [
            ".edu",
            ".ac.uk",
            ".ac.in",
        ]
    ):
        return "ACADEMIC"

    if any(
        x in source_l
        for x in [
            "pubmed",
            "ncbi.nlm.nih.gov",
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "ieee.org",
            "nih.gov",
        ]
    ):
        return "PEER_REVIEWED"

    if "wikidata.org" in source_l:
        return "WIKIDATA"

    if "wikipedia.org" in source_l or "wikipedia" in title_l:
        return "WIKIPEDIA"

    if any(
        x in source_l
        for x in [
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "bbc.co.uk",
            "theguardian.com",
            "nytimes.com",
        ]
    ):
        return "REPUTABLE_NEWS"

    if any(
        x in source_l
        for x in [
            "who.int",
            "un.org",
            "worldbank.org",
            "imf.org",
            "nasa.gov",
            "cdc.gov",
            "fda.gov",
            "ec.europa.eu",
        ]
    ):
        return "ORGANIZATION"

    if title_l:
        return "UNKNOWN"

    return "UNKNOWN"


def calculate_source_reliability(
    source_url: str,
    title: str = "",
) -> Dict[str, Any]:
    source_type = classify_source(source_url, title)
    score = SOURCE_RELIABILITY_SCORES.get(
        source_type,
        SOURCE_RELIABILITY_SCORES["UNKNOWN"],
    )

    if score >= 90:
        level = "HIGH"
    elif score >= 70:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "type": source_type,
        "score": round(score, 2),
        "level": level,
    }


# ============================================================
# RELEVANCE
# ============================================================

def calculate_relevance(
    claim: str,
    title: str,
    evidence: str,
) -> float:
    claim_terms = set(extract_terms(claim))
    title_terms = set(extract_terms(title))
    evidence_terms = set(extract_terms(evidence))

    if not claim_terms:
        return 0.0

    title_overlap = (
        len(claim_terms & title_terms) / len(claim_terms)
    )
    evidence_overlap = (
        len(claim_terms & evidence_terms) / len(claim_terms)
    )

    score = (
        title_overlap * 60.0
        + evidence_overlap * 40.0
    )

    relation = parse_relation_claim(claim)
    normalized_evidence = normalize_text(evidence)

    if relation:
        subject_terms = set(extract_terms(relation["subject"]))
        object_terms = set(extract_terms(relation["object"]))

        subject_hit = bool(subject_terms & evidence_terms)
        object_hit = bool(object_terms & evidence_terms)

        if subject_hit:
            score += 20.0
        if object_hit:
            score += 20.0

    if canonical_claim(claim) in normalized_evidence:
        score = max(score, 98.0)

    return round(min(score, 100.0), 2)


# ============================================================
# DYNAMIC EVIDENCE RETRIEVAL
# ============================================================

def retrieve_evidence(claim: str) -> List[Dict[str, Any]]:
    print("-" * 70)
    print("DYNAMIC RETRIEVAL:", claim)

    candidates = []

    # Structured current-office lookup first.
    candidates.extend(
        current_office_wikidata_candidates(claim)
    )

    # General structured knowledge.
    candidates.extend(
        wikidata_general_candidates(claim)
    )

    # Reference evidence.
    candidates.extend(
        wikipedia_candidates(claim)
    )

    # Optional real web search if a provider key is configured.
    if SERPER_API_KEY:
        candidates.extend(serper_candidates(claim))

    if TAVILY_API_KEY:
        candidates.extend(tavily_candidates(claim))

    # De-duplicate by URL/title/text.
    unique = []
    seen = set()

    for candidate in candidates:
        key = (
            normalize_text(candidate.get("source", "")),
            normalize_text(candidate.get("title", "")),
            normalize_text(candidate.get("text", ""))[:300],
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(candidate)

    # Score relevance now, after all sources have been collected.
    for candidate in unique:
        candidate["relevance"] = calculate_relevance(
            claim,
            candidate.get("title", ""),
            candidate.get("text", ""),
        )

    unique = [
        item for item in unique
        if item["relevance"] >= MIN_RELEVANCE
    ]

    # Prefer high-reliability sources when relevance is similar.
    for candidate in unique:
        candidate["source_reliability"] = (
            calculate_source_reliability(
                candidate.get("source", ""),
                candidate.get("title", ""),
            )
        )

    unique.sort(
        key=lambda item: (
            item["relevance"],
            item["source_reliability"]["score"],
        ),
        reverse=True,
    )

    print("Evidence candidates:", len(unique))

    for item in unique[:10]:
        print(
            " -",
            item["title"],
            "| relevance=",
            item["relevance"],
            "| source=",
            item["source_reliability"]["type"],
        )

    return unique[:12]


# ============================================================
# GENERIC DETERMINISTIC VERIFICATION
# ============================================================

NEGATION_WORDS = {
    "not", "no", "never", "without", "cannot", "can't",
    "doesn't", "doesnt", "isn't", "isnt", "wasn't", "wasnt",
    "false", "incorrect", "disproven", "debunked",
}


def numbers_in(text: str) -> List[str]:
    return re.findall(
        r"\b\d+(?:\.\d+)?\b",
        normalize_text(text),
    )


def name_like_tokens(text: str) -> List[str]:
    words = re.findall(
        r"\b[A-Z][A-Za-z.'-]{2,}\b",
        clean_display_text(text),
    )
    return unique_keep_order(words)


def sentence_contains_claim_terms(
    claim: str,
    sentence: str,
    minimum_ratio: float = 0.45,
) -> bool:
    claim_terms = set(extract_terms(claim))
    sent_terms = set(extract_terms(sentence))

    if not claim_terms:
        return False

    ratio = len(claim_terms & sent_terms) / len(claim_terms)
    return ratio >= minimum_ratio


def deterministic_verification(
    claim: str,
    evidence: str,
) -> Optional[Dict[str, Any]]:
    """
    Generic rules, not a database of claims.

    It is intentionally conservative. Returning None means:
    'the rule system cannot safely decide this evidence.'
    The NLI verifier or consensus layer can then handle it.
    """
    c = canonical_claim(claim)
    e = normalize_text(evidence)

    # Exact sentence / phrase support.
    if c and c in e:
        return {
            "status": "SUPPORTED",
            "confidence": 98.0,
            "supported_score": 98.0,
            "contradicted_score": 1.0,
            "neutral_score": 1.0,
            "evidence_used": evidence[:3500],
            "verification_method": "generic_exact_match",
        }

    relation = parse_relation_claim(claim)

    if not relation:
        return None

    subject = normalize_text(relation["subject"])
    object_text = normalize_text(relation["object"])

    if len(subject) < 3 or len(object_text) < 2:
        return None

    # --------------------------------------------------------
    # Current office-holder claims
    # --------------------------------------------------------
    # Example:
    # "The current Chief Minister of Andhra Pradesh is X."
    # This rule is generic; it does not contain a hard-coded
    # politician, state, country, or office.
    if (
        relation["claim_type"] == "CURRENT_FACT"
        and " of " in subject
    ):
        office_match = re.search(
            r"(.+?)\s+of\s+(.+)$",
            subject,
            re.I,
        )

        if office_match:
            office = normalize_text(office_match.group(1))
            jurisdiction = normalize_text(office_match.group(2))
            sentences = sentence_split(evidence)

            for sentence in sentences:
                s = normalize_text(sentence)

                if (
                    office
                    and jurisdiction
                    and office in s
                    and jurisdiction in s
                ):
                    candidate_hit = object_text in s

                    if candidate_hit:
                        return {
                            "status": "SUPPORTED",
                            "confidence": 96.0,
                            "supported_score": 96.0,
                            "contradicted_score": 2.0,
                            "neutral_score": 2.0,
                            "evidence_used": sentence[:3500],
                            "verification_method": "generic_current_office_match",
                        }

                    # If the office and jurisdiction are present but
                    # the evidence names a different office-holder,
                    # treat it as contradiction only when a distinct
                    # person-like name is visible.
                    claim_names = {
                        normalize_text(x)
                        for x in name_like_tokens(relation["object"])
                    }
                    evidence_names = {
                        normalize_text(x)
                        for x in name_like_tokens(sentence)
                    }

                    if (
                        evidence_names
                        and not claim_names.intersection(evidence_names)
                    ):
                        return {
                            "status": "CONTRADICTED",
                            "confidence": 95.0,
                            "supported_score": 2.0,
                            "contradicted_score": 95.0,
                            "neutral_score": 3.0,
                            "evidence_used": sentence[:3500],
                            "verification_method": "generic_current_office_match",
                        }

    sentences = sentence_split(evidence)

    best_support = None
    best_contradiction = None

    object_terms = set(extract_terms(object_text))
    subject_terms = set(extract_terms(subject))

    for sentence in sentences:
        s = normalize_text(sentence)
        if not s:
            continue

        subject_hit = bool(subject_terms & set(extract_terms(s)))

        if not subject_hit:
            # Also allow exact subject phrase.
            subject_hit = subject in s

        if not subject_hit:
            continue

        # For a relation claim, prefer the predicate portion of the
        # evidence after the same verb (or a closely matching verb).
        # This supports both copular claims and action claims such as
        # "India won the 1983 Cricket World Cup."
        relation_verb = normalize_text(
            relation.get("verb", "")
        )
        verb_pattern = (
            re.escape(relation_verb)
            if relation_verb
            else r"(?:is|are|was|were|has|have|had|won|wins|defeated|defeats|hosted|hosts|founded|invented|discovered|launched|served|became|held|ruled|led|created|built|established|introduced|declared|appointed)"
        )
        verb_match = re.search(
            rf"\b{verb_pattern}\b",
            s,
            re.I,
        )
        predicate_text = (
            s[verb_match.end():].strip()
            if verb_match
            else s
        )

        object_hit = (
            object_text in predicate_text
            or bool(object_terms & set(extract_terms(predicate_text)))
        )

        if object_hit:
            score = 90.0

            local_negation = any(
                word in s for word in NEGATION_WORDS
            )

            if local_negation:
                best_contradiction = (
                    94.0,
                    sentence,
                )
            else:
                best_support = (
                    score,
                    sentence,
                )

        # Explicitly negative evidence should contradict a positive
        # "subject is object" claim even when the sentence is phrased
        # as "X is a disproven/false claim."
        if any(word in s for word in NEGATION_WORDS):
            if (
                subject in s
                and (
                    object_text in s
                    or bool(object_terms & set(extract_terms(s)))
                )
            ):
                best_contradiction = (
                    94.0,
                    sentence,
                )

        # If the claim says "subject is X", and the evidence says
        # "subject is Y", a different named entity is strong contradiction.
        if re.search(
            r"\b(?:is|was|are|were|has|have|had|won|wins|defeated|defeats|hosted|hosts|founded|invented|discovered|launched|served|became|held|ruled|led|created|built|established|introduced|declared|appointed)\b",
            s,
            re.I,
        ):
            before_after = re.split(
                r"\b(?:is|was|are|were|has|have|had|won|wins|defeated|defeats|hosted|hosts|founded|invented|discovered|launched|served|became|held|ruled|led|created|built|established|introduced|declared|appointed)\b",
                s,
                maxsplit=1,
                flags=re.I,
            )

            if len(before_after) == 2:
                evidence_subject = before_after[0].strip()
                evidence_object = before_after[1].strip()

                if (
                    subject in evidence_subject
                    or evidence_subject in subject
                ):
                    if object_text not in evidence_object:
                        # A different proper name is a strong signal.
                        evidence_names = set(
                            x.lower()
                            for x in name_like_tokens(sentence)
                        )
                        claim_names = set(
                            x.lower()
                            for x in name_like_tokens(claim)
                        )

                        if evidence_names and (
                            evidence_names != claim_names
                        ):
                            best_contradiction = (
                                92.0,
                                sentence,
                            )

    # Numeric comparison for claims like "X is 100 years old."
    claim_numbers = numbers_in(claim)
    if claim_numbers:
        for sentence in sentences:
            if subject not in normalize_text(sentence):
                continue
            evidence_numbers = numbers_in(sentence)
            if evidence_numbers and claim_numbers[0] not in evidence_numbers:
                best_contradiction = (
                    90.0,
                    sentence,
                )

    if best_contradiction:
        confidence, sentence = best_contradiction
        return {
            "status": "CONTRADICTED",
            "confidence": confidence,
            "supported_score": 3.0,
            "contradicted_score": confidence,
            "neutral_score": 100.0 - confidence,
            "evidence_used": sentence[:3500],
            "verification_method": "generic_relation_match",
        }

    if best_support:
        confidence, sentence = best_support
        return {
            "status": "SUPPORTED",
            "confidence": confidence,
            "supported_score": confidence,
            "contradicted_score": 3.0,
            "neutral_score": 100.0 - confidence,
            "evidence_used": sentence[:3500],
            "verification_method": "generic_relation_match",
        }

    return None


# ============================================================
# NLI VERIFICATION
# ============================================================

def select_relevant_evidence(
    claim: str,
    evidence: str,
    max_sentences: int = 6,
) -> str:
    sentences = sentence_split(evidence)
    claim_terms = set(extract_terms(claim))

    scored = []

    for sentence in sentences:
        sentence_terms = set(extract_terms(sentence))
        overlap = len(claim_terms & sentence_terms)

        if normalize_text(claim) in normalize_text(sentence):
            overlap += 100

        scored.append((overlap, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [
        sentence
        for score, sentence in scored[:max_sentences]
        if score > 0
    ]

    return " ".join(selected)[:3500]


def nli_verify(
    claim: str,
    evidence: str,
) -> Dict[str, Any]:
    if not NLI_AVAILABLE:
        return {
            "status": "NEUTRAL",
            "confidence": 0.0,
            "supported_score": 0.0,
            "contradicted_score": 0.0,
            "neutral_score": 100.0,
            "evidence_used": evidence[:3500],
            "verification_method": "NLI_DISABLED",
        }

    evidence_used = select_relevant_evidence(
        claim,
        evidence,
    )

    if not evidence_used:
        return {
            "status": "NEUTRAL",
            "confidence": 0.0,
            "supported_score": 0.0,
            "contradicted_score": 0.0,
            "neutral_score": 100.0,
            "evidence_used": "",
            "verification_method": "NLI",
        }

    try:
        import torch

        # Premise = evidence, hypothesis = user claim.
        inputs = nli_tokenizer(
            evidence_used,
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = nli_model(**inputs)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

        if None in (
            NLI_ENTAILMENT_ID,
            NLI_CONTRADICTION_ID,
            NLI_NEUTRAL_ID,
        ):
            raise RuntimeError(
                f"NLI labels unresolved: {NLI_LABEL_IDS}"
            )

        entailment = (
            float(probabilities[NLI_ENTAILMENT_ID]) * 100
        )
        contradiction = (
            float(probabilities[NLI_CONTRADICTION_ID]) * 100
        )
        neutral = (
            float(probabilities[NLI_NEUTRAL_ID]) * 100
        )

        scores = {
            "SUPPORTED": entailment,
            "CONTRADICTED": contradiction,
            "NEUTRAL": neutral,
        }

        status = max(scores, key=scores.get)

        return {
            "status": status,
            "confidence": round(scores[status], 2),
            "supported_score": round(entailment, 2),
            "contradicted_score": round(contradiction, 2),
            "neutral_score": round(neutral, 2),
            "evidence_used": evidence_used,
            "verification_method": "NLI",
        }

    except Exception as exc:
        print("NLI verification failed:", repr(exc))
        return {
            "status": "NEUTRAL",
            "confidence": 0.0,
            "supported_score": 0.0,
            "contradicted_score": 0.0,
            "neutral_score": 100.0,
            "evidence_used": evidence_used,
            "verification_method": "NLI_ERROR",
        }


def verify_claim_against_evidence(
    claim: str,
    evidence: str,
) -> Dict[str, Any]:
    deterministic = deterministic_verification(
        claim,
        evidence,
    )

    if deterministic is not None:
        return deterministic

    if NLI_AVAILABLE:
        return nli_verify(claim, evidence)

    return {
        "status": "NEUTRAL",
        "confidence": 0.0,
        "supported_score": 0.0,
        "contradicted_score": 0.0,
        "neutral_score": 100.0,
        "evidence_used": select_relevant_evidence(
            claim,
            evidence,
        ),
        "verification_method": "NO_SEMANTIC_VERIFIER",
    }


# ============================================================
# DISTILBERT ASSESSMENT
# ============================================================

def run_model_assessment(content: str) -> Dict[str, Any]:
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
            inputs["input_ids"] = np.asarray(
                encoded["input_ids"],
                dtype=np.int64,
            )

        if "attention_mask" in ONNX_INPUT_NAMES:
            inputs["attention_mask"] = np.asarray(
                encoded["attention_mask"],
                dtype=np.int64,
            )

        if "token_type_ids" in ONNX_INPUT_NAMES:
            inputs["token_type_ids"] = np.asarray(
                encoded.get(
                    "token_type_ids",
                    np.zeros_like(encoded["input_ids"]),
                ),
                dtype=np.int64,
            )

        outputs = onnx_session.run(
            None,
            inputs,
        )

        logits = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        shifted = logits - np.max(
            logits,
            axis=-1,
            keepdims=True,
        )

        exp_logits = np.exp(shifted)
        probabilities = exp_logits / np.sum(
            exp_logits,
            axis=-1,
            keepdims=True,
        )

        predicted_id = int(np.argmax(probabilities[0]))
        confidence = (
            float(probabilities[0, predicted_id]) * 100.0
        )

        label = str(predicted_id)

        config_file = ONNX_MODEL_PATH / "config.json"
        if config_file.exists():
            try:
                config = json.loads(
                    config_file.read_text(encoding="utf-8")
                )
                id2label = config.get("id2label", {})
                label = str(
                    id2label.get(
                        str(predicted_id),
                        id2label.get(predicted_id, label),
                    )
                )
            except Exception as exc:
                print("Label map read failed:", repr(exc))

        return {
            "verdict": label,
            "confidence": round(confidence, 2),
            "model": "DistilBERT",
            "runtime": "ONNX Runtime INT8",
        }

    except Exception as exc:
        print("DistilBERT assessment failed:", repr(exc))
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "model": "DistilBERT",
            "runtime": "ONNX Runtime INT8",
        }


# ============================================================
# EVIDENCE QUALITY / CONSENSUS
# ============================================================

def calculate_evidence_quality(
    relevance: float,
    verification: Dict[str, Any],
    source_reliability: Dict[str, Any],
) -> float:
    relevance_score = max(0.0, min(float(relevance), 100.0))
    verification_score = max(
        0.0,
        min(
            float(verification.get("confidence", 0.0)),
            100.0,
        ),
    )
    reliability_score = max(
        0.0,
        min(
            float(source_reliability.get("score", 50.0)),
            100.0,
        ),
    )

    quality = (
        verification_score * 0.50
        + relevance_score * 0.30
        + reliability_score * 0.20
    )

    return round(quality, 2)


def assess_evidence(
    claim: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not candidates:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "score": 0.0,
            "sources": [],
            "best_evidence": None,
            "verification": None,
            "consensus": {},
        }

    verified = []

    for candidate in candidates:
        verification = verify_claim_against_evidence(
            claim,
            candidate["text"],
        )

        reliability = candidate.get(
            "source_reliability"
        ) or calculate_source_reliability(
            candidate.get("source", ""),
            candidate.get("title", ""),
        )

        quality = calculate_evidence_quality(
            candidate.get("relevance", 0.0),
            verification,
            reliability,
        )

        item = dict(candidate)
        item["verification"] = verification
        item["source_reliability"] = reliability
        item["evidence_quality"] = quality

        verified.append(item)

    # Only confident semantic/deterministic decisions participate
    # in the final consensus.
    decisive = [
        item for item in verified
        if item["verification"].get("status")
        in {"SUPPORTED", "CONTRADICTED"}
        and (
            item["verification"].get("verification_method")
            in {
                "generic_exact_match",
                "generic_relation_match",
                "deterministic_evidence_match",
                "NLI",
            }
            or (
                item["verification"].get("confidence", 0.0)
                >= MIN_NLI_CONFIDENCE
            )
        )
    ]

    support_weight = 0.0
    contradiction_weight = 0.0

    for item in decisive:
        weight = (
            item["evidence_quality"]
            / 100.0
        )
        if item["verification"]["status"] == "SUPPORTED":
            support_weight += weight
        else:
            contradiction_weight += weight

    total = support_weight + contradiction_weight

    if total <= 0:
        final_status = "INSUFFICIENT_EVIDENCE"
        final_score = 0.0
    else:
        support_ratio = support_weight / total
        contradiction_ratio = contradiction_weight / total

        # Require meaningful consensus. This prevents one weak
        # result from forcing a verdict.
        if support_ratio >= 0.65 and support_weight >= 0.80:
            final_status = "SUPPORTED"
            final_score = support_ratio * 100.0
        elif (
            contradiction_ratio >= 0.65
            and contradiction_weight >= 0.80
        ):
            final_status = "CONTRADICTED"
            final_score = contradiction_ratio * 100.0
        else:
            final_status = "INSUFFICIENT_EVIDENCE"
            final_score = 0.0

    # If we have only one extremely strong authoritative source,
    # allow it to decide even if consensus count is small.
    if final_status == "INSUFFICIENT_EVIDENCE":
        strong = [
            item for item in decisive
            if (
                item["evidence_quality"] >= 82.0
                and item["source_reliability"]["score"] >= 88.0
                and item["verification"].get("confidence", 0.0) >= 90.0
            )
        ]

        if strong:
            strong.sort(
                key=lambda x: x["evidence_quality"],
                reverse=True,
            )
            strongest = strong[0]
            final_status = strongest["verification"]["status"]
            final_score = strongest["evidence_quality"]

    verified.sort(
        key=lambda item: (
            item["evidence_quality"],
            item["relevance"],
            item["source_reliability"]["score"],
        ),
        reverse=True,
    )

    best = verified[0] if verified else None

    sources = []
    for item in verified[:6]:
        sources.append(
            {
                "title": item["title"],
                "text": item["text"][:3500],
                "source": item["source"],
                "relevance": item["relevance"],
                "verification": item["verification"],
                "source_reliability": item["source_reliability"],
                "evidence_quality": item["evidence_quality"],
                "retrieval": item.get("retrieval", ""),
            }
        )

    best_evidence = None
    best_verification = None

    if best:
        best_verification = best["verification"]
        best_evidence = {
            "title": best["title"],
            "text": best["text"][:3500],
            "source": best["source"],
            "relevance": best["relevance"],
            "verification": best["verification"],
            "source_reliability": best["source_reliability"],
            "evidence_quality": best["evidence_quality"],
            "retrieval": best.get("retrieval", ""),
        }

    return {
        "status": final_status,
        "score": round(min(final_score, 100.0), 2),
        "sources": sources,
        "best_evidence": best_evidence,
        "verification": best_verification,
        "consensus": {
            "support_weight": round(support_weight, 3),
            "contradiction_weight": round(
                contradiction_weight,
                3,
            ),
            "decisive_sources": len(decisive),
        },
    }


# ============================================================
# EXPLANATION
# ============================================================

def build_explanation(
    claim: str,
    evidence_assessment: Dict[str, Any],
) -> str:
    status = evidence_assessment.get(
        "status",
        "INSUFFICIENT_EVIDENCE",
    )
    best = evidence_assessment.get("best_evidence")

    if not best:
        return (
            "TruthLens AI could not retrieve enough relevant "
            "evidence to make a reliable evidence-based decision."
        )

    title = best.get("title", "the retrieved source")
    reliability = best.get(
        "source_reliability",
        {},
    ).get("level", "UNKNOWN")

    if status == "SUPPORTED":
        return (
            f"Retrieved evidence from {title} ({reliability} "
            "source reliability) supports the claim. The final "
            "decision is based on evidence relevance, source "
            "reliability, and claim/evidence verification."
        )

    if status == "CONTRADICTED":
        return (
            f"Retrieved evidence from {title} ({reliability} "
            "source reliability) conflicts with the claim. "
            "The final decision is based on evidence rather than "
            "the DistilBERT signal alone."
        )

    return (
        "The system found relevant material, but the evidence "
        "did not reach the confidence and consensus threshold "
        "needed for a supported or contradicted verdict."
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    methods = [
        "Dynamic Wikipedia retrieval",
        "Dynamic Wikidata retrieval",
        "Generic evidence verification",
        "Source reliability",
        "Evidence consensus",
        "DistilBERT INT8 ONNX",
    ]

    if NLI_AVAILABLE:
        methods.append("NLI semantic verification")

    return {
        "message": "Welcome to TruthLens AI",
        "status": "Backend is running",
        "version": APP_VERSION,
        "verification_methods": methods,
        "decision_policy": (
            "Final verdict is determined from retrieved evidence "
            "and evidence consensus, not DistilBERT alone."
        ),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": onnx_session is not None,
        "nli_loaded": NLI_AVAILABLE,
        "nli_disabled": DISABLE_NLI,
        "device": DEVICE,
        "evidence_enabled": True,
        "dynamic_retrieval": True,
        "wikipedia_search": True,
        "wikidata_search": True,
        "optional_web_search": bool(
            SERPER_API_KEY or TAVILY_API_KEY
        ),
        "generic_verification": True,
        "semantic_verification": NLI_AVAILABLE,
        "source_reliability": True,
        "evidence_consensus": True,
    }


@app.get("/diagnostics")
def diagnostics():
    return {
        "version": APP_VERSION,
        "device": DEVICE,
        "onnx_model": str(ONNX_MODEL_FILE),
        "onnx_loaded": onnx_session is not None,
        "nli_loaded": NLI_AVAILABLE,
        "nli_disabled": DISABLE_NLI,
        "nli_labels": NLI_LABEL_IDS,
        "dynamic_retrieval": {
            "wikipedia": True,
            "wikidata": True,
            "serper": bool(SERPER_API_KEY),
            "tavily": bool(TAVILY_API_KEY),
        },
        "cache_sizes": {
            "pages": len(PAGE_CACHE),
            "search": len(SEARCH_CACHE),
            "wikidata": len(WIKIDATA_CACHE),
        },
    }


@app.post("/analyze")
def analyze_content(request: AnalyzeRequest):
    content = clean_display_text(request.content)

    if not content:
        return {
            "success": False,
            "message": "No content was provided.",
            "content": "",
        }

    print("=" * 70)
    print("ANALYZING CLAIM:", content)
    print("=" * 70)

    claim_type = detect_claim_type(content)
    entities = extract_claim_entities(content)
    search_queries = generate_search_queries(content)

    # 1. ML signal.
    model_assessment = run_model_assessment(content)

    # 2. Dynamic retrieval.
    candidates = retrieve_evidence(content)

    # 3. Evidence assessment.
    evidence_assessment = assess_evidence(
        content,
        candidates,
    )

    evidence_status = evidence_assessment["status"]
    evidence_score = evidence_assessment["score"]

    if evidence_status == "SUPPORTED":
        final_verdict = "SUPPORTED"
        final_confidence = evidence_score
        basis = (
            "Retrieved evidence supports the claim with sufficient "
            "relevance, source reliability, and verification strength."
        )

    elif evidence_status == "CONTRADICTED":
        final_verdict = "CONTRADICTED"
        final_confidence = evidence_score
        basis = (
            "Retrieved evidence contradicts the claim with sufficient "
            "relevance, source reliability, and verification strength."
        )

    else:
        final_verdict = "INSUFFICIENT EVIDENCE"
        final_confidence = 0.0
        basis = (
            "Available evidence was not sufficiently relevant, "
            "reliable, or consistent to make a safe binary decision."
        )

    evidence_basis = (
        "Dynamic Wikipedia + Wikidata retrieval + generic claim/evidence "
        "verification + source reliability + evidence consensus + "
        "DistilBERT INT8 initial assessment"
    )

    if NLI_AVAILABLE:
        evidence_basis += " + NLI semantic verification"

    explanation = build_explanation(
        content,
        evidence_assessment,
    )

    print("=" * 70)
    print("CLAIM TYPE:", claim_type)
    print("ENTITIES:", entities)
    print("FINAL VERDICT:", final_verdict)
    print("FINAL CONFIDENCE:", final_confidence)
    print(
        "EVIDENCE SOURCES:",
        len(evidence_assessment.get("sources", [])),
    )
    print("=" * 70)

    return {
        "success": True,
        "content": content,

        # Useful for debugging and UI.
        "claim_analysis": {
            "claim_type": claim_type,
            "entities": entities,
            "search_queries": search_queries,
        },

        "model_assessment": model_assessment,

        "evidence_assessment": evidence_assessment,

        "final_assessment": {
            "verdict": final_verdict,
            "confidence": round(final_confidence, 2),
            "basis": basis,
            "explanation": explanation,
        },

        # Frontend compatibility.
        "verdict": final_verdict,
        "confidence": round(final_confidence, 2),
        "model": "DistilBERT",
        "decision_method": (
            "Final verdict is determined by dynamically retrieved "
            "evidence and evidence consensus, not by the DistilBERT "
            "ML signal alone."
        ),
        "evidence_basis": evidence_basis,
        "nli_enabled": NLI_AVAILABLE,
    }


# ============================================================
# STARTUP STATUS
# ============================================================

print("=" * 70)
print("TruthLens AI backend ready.")
print("DistilBERT:", "ACTIVE")
print("Model Runtime:", "ONNX Runtime INT8")
print("Dynamic Wikipedia:", "ACTIVE")
print("Dynamic Wikidata:", "ACTIVE")
print(
    "Optional Web Search:",
    "ACTIVE" if (SERPER_API_KEY or TAVILY_API_KEY) else "NOT CONFIGURED",
)
print(
    "Generic Verification:",
    "ACTIVE",
)
print(
    "NLI Verification:",
    "ACTIVE" if NLI_AVAILABLE else "DISABLED",
)
print("Source Reliability:", "ACTIVE")
print("Evidence Consensus:", "ACTIVE")
print("=" * 70)
