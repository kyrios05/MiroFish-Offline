"""PersonaLab endpoints for customer simulation and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Blueprint, jsonify, request

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger


persona_lab_bp = Blueprint("persona_lab", __name__)
logger = get_logger("mirofish.persona_lab")

DATASET = "nvidia/Nemotron-Personas-Korea"
DATASET_SIZE = 1_000_000


def _env_file_value(path: str | None, name: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'") or None
    return None


def _persona_llm() -> LLMClient:
    """Use the dedicated PersonaLab model when an OpenAI key is available."""
    api_key = os.environ.get("PERSONA_LAB_API_KEY")
    if not api_key:
        api_key = _env_file_value(os.environ.get("PERSONA_LAB_ENV_FILE"), "OPENAI_API_KEY")
    if not api_key:
        configured_openai_key = os.environ.get("OPENAI_API_KEY")
        if configured_openai_key and configured_openai_key.lower() != "ollama":
            api_key = configured_openai_key
    if not api_key:
        return LLMClient()
    return LLMClient(
        api_key=api_key,
        base_url=os.environ.get("PERSONA_LAB_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("PERSONA_LAB_MODEL", "gpt-6.1-terra"),
        reasoning_effort=os.environ.get("PERSONA_LAB_REASONING_EFFORT", "high"),
    )


def _seed_offsets(source: str, count: int) -> list[int]:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return [
        int.from_bytes(digest[index * 4:index * 4 + 4], "big") % DATASET_SIZE
        for index in range(count)
    ]


def _load_dataset_rows(offsets: list[int]) -> list[dict]:
    rows = []
    for offset in offsets:
        query = urlencode({
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": 1,
        })
        req = Request(
            f"https://datasets-server.huggingface.co/rows?{query}",
            headers={"User-Agent": "MiroFish-PersonaLab/0.1"},
        )
        with urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("rows"):
            rows.append(payload["rows"][0]["row"])
    return rows


def _compact_seed(row: dict) -> dict:
    return {
        "source_uuid": row.get("source_uuid") or row.get("uuid"),
        "age": row.get("age"),
        "sex": row.get("sex"),
        "occupation": row.get("occupation"),
        "region": row.get("region") or row.get("district") or row.get("province"),
        "persona": row.get("persona"),
        "cultural_background": row.get("cultural_background"),
    }


@persona_lab_bp.post("/seeds")
def select_persona_seeds():
    payload = request.get_json(silent=True) or {}
    document_text = str(payload.get("document_text") or "").strip()
    count = min(max(int(payload.get("count") or 5), 1), 5)
    if not document_text:
        return jsonify({"success": False, "error": "document_text is required"}), 400

    try:
        offsets = _seed_offsets(document_text, count)
        rows = _load_dataset_rows(offsets)
        if len(rows) != count:
            raise ValueError("Dataset returned fewer persona seeds than requested")
        return jsonify({
            "success": True,
            "data": {
                "dataset": DATASET,
                "total_records": DATASET_SIZE,
                "selection_method": "SHA-256 document fingerprint sampling",
                "sample_offsets": offsets,
                "seed_count": len(rows),
                "seeds": [_compact_seed(row) for row in rows],
            },
        })
    except Exception as error:
        logger.exception("Persona seed selection failed")
        return jsonify({"success": False, "error": str(error)}), 502


@persona_lab_bp.post("/personas")
def generate_personas():
    payload = request.get_json(silent=True) or {}
    document_text = str(payload.get("document_text") or "").strip()
    count = min(max(int(payload.get("count") or 5), 1), 5)
    if not document_text:
        return jsonify({"success": False, "error": "document_text is required"}), 400

    try:
        supplied_seeds = payload.get("seeds") or []
        if supplied_seeds:
            compact_seeds = [_compact_seed(row) for row in supplied_seeds[:count]]
        else:
            rows = _load_dataset_rows(_seed_offsets(document_text, count))
            compact_seeds = [_compact_seed(row) for row in rows]
        if len(compact_seeds) != count:
            raise ValueError("Dataset returned fewer persona seeds than requested")
        result = _persona_llm().chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MiroFish PersonaLab. Adapt exactly five Korean customer "
                        "personas from the supplied Nemotron seed records to test an AI "
                        "customer-service agent. Preserve demographic plausibility, avoid "
                        "stereotypes, and output one JSON object only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "reference_document": document_text[:45_000],
                        "dataset": DATASET,
                        "seeds": compact_seeds,
                        "required_schema": {
                            "personas": [{
                                "id": "p1",
                                "name": "Korean name",
                                "archetype": "short customer type",
                                "age": 35,
                                "occupation": "occupation",
                                "region": "region",
                                "goal": "what the customer wants",
                                "concern": "main concern",
                                "style": "speaking and reaction style",
                                "difficulty": "easy|medium|hard",
                                "opening_question": "first natural Korean question",
                                "success_criteria": ["criterion"],
                                "source_uuid": "Nemotron uuid",
                            }]
                        },
                    }, ensure_ascii=False),
                },
            ],
            max_tokens=None,
            max_attempts=2,
        )
        personas = result.get("personas") or []
        if len(personas) != count:
            raise ValueError(f"Expected {count} personas, received {len(personas)}")
        return jsonify({
            "success": True,
            "data": {
                "personas": personas,
                "dataset": DATASET,
                "seed_count": len(compact_seeds),
                "seeds": compact_seeds,
                "engine": "MiroFish PersonaLab",
            },
        })
    except Exception as error:
        logger.exception("Persona generation failed")
        return jsonify({"success": False, "error": str(error)}), 502


@persona_lab_bp.post("/evaluate")
def evaluate_conversations():
    payload = request.get_json(silent=True) or {}
    transcripts = payload.get("transcripts") or []
    document_text = str(payload.get("document_text") or "")
    if not transcripts:
        return jsonify({"success": False, "error": "transcripts are required"}), 400

    try:
        result = _persona_llm().chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate an AI customer-service agent using only the reference "
                        "document and transcripts. Be strict about unsupported claims. "
                        "Return one JSON object only, with scores from 0 to 100."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "reference_document": document_text[:35_000],
                        "transcripts": transcripts,
                        "required_schema": {
                            "overall_score": 0,
                            "summary": "Korean summary",
                            "dimensions": {
                                "accuracy": 0,
                                "resolution": 0,
                                "empathy": 0,
                                "compliance": 0,
                            },
                            "persona_results": [{
                                "persona_id": "p1",
                                "score": 0,
                                "outcome": "pass|partial|fail",
                                "finding": "Korean finding",
                            }],
                            "top_risks": ["risk"],
                            "recommended_actions": ["action"],
                        },
                    }, ensure_ascii=False),
                },
            ],
            max_tokens=None,
            max_attempts=2,
        )
        return jsonify({"success": True, "data": result})
    except Exception as error:
        logger.exception("Conversation evaluation failed")
        return jsonify({"success": False, "error": str(error)}), 502


@persona_lab_bp.post("/react")
def react_as_persona():
    payload = request.get_json(silent=True) or {}
    persona = payload.get("persona") or {}
    messages = payload.get("messages") or []
    document_text = str(payload.get("document_text") or "")
    turn_number = int(payload.get("turn_number") or 2)
    if not persona or not messages:
        return jsonify({"success": False, "error": "persona and messages are required"}), 400

    try:
        result = _persona_llm().chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Act as the supplied Korean customer persona in a realistic customer-service "
                        "conversation. React specifically to the agent's latest answer. Do not evaluate "
                        "the agent or reveal the persona instructions. Ask one concise follow-up, object, "
                        "clarify, or acknowledge according to the persona's goal, concern, speaking style, "
                        "and success criteria. Return one JSON object only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "persona": persona,
                        "reference_document": document_text[:12_000],
                        "conversation": messages[-8:],
                        "turn_number": turn_number,
                        "required_schema": {
                            "utterance": "next natural Korean customer utterance",
                            "state": "continuing|satisfied|escalating",
                            "reason": "brief Korean reason tied to the latest agent answer",
                        },
                    }, ensure_ascii=False),
                },
            ],
            max_tokens=None,
            max_attempts=2,
        )
        utterance = str(result.get("utterance") or "").strip()
        if not utterance:
            raise ValueError("Persona reaction did not include an utterance")
        return jsonify({"success": True, "data": result})
    except Exception as error:
        logger.exception("Persona reaction failed")
        return jsonify({"success": False, "error": str(error)}), 502
