"""
AgriShield AI — LLM / Multimodal Reasoning Engine
IBM Granite Reasoning via a structured prompt orchestrator.

Responsibilities:
  • Accept multimodal context: visual metrics + crop metadata + weather telemetry
  • Construct a rich, structured RAG-style prompt
  • Return a fully parsed DiagnosticReport dataclass
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── IBM Granite / OpenAI-compatible client ───────────────────────────────────
# ── IBM watsonx.ai client ──────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

    _WATSONX_AVAILABLE = True
except ImportError:
    _WATSONX_AVAILABLE = False
    logger.warning("ibm-watsonx-ai SDK not found — install it: pip install ibm-watsonx-ai")

# ─────────────────────────────────────────────────────────────────────────────
# Input / Output contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CropContext:
    crop_type: str            # e.g. "Tomato", "Rice", "Cotton"
    growth_stage: str         # e.g. "Vegetative", "Flowering", "Fruiting"
    region: str               # e.g. "South Asia", "Sub-Saharan Africa"
    humidity_pct: float       # 0–100
    rainfall: str             # "None", "Low", "Moderate", "Heavy"
    temperature_c: float      # ambient °C


@dataclass
class DiagnosticReport:
    diagnosis: str
    etiology: str
    confidence_pct: float
    severity_label: str
    interventions_organic: list[str]
    interventions_biological: list[str]
    soil_microbiome_safeguards: list[str]
    pollinator_safeguards: list[str]
    irrigation_schedule: str
    sdg2_alignment: str
    sdg15_alignment: str
    disclaimer: str = (
        "AgriShield AI provides decision-support only. "
        "Consult a certified agronomist before applying treatments."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are AgriShield AI, a precision agriculture expert powered by IBM Granite Reasoning.
    You specialise in crop disease diagnosis, sustainable intervention design, and UN SDG alignment.

    Output Rules:
    — Respond ONLY in the structured JSON schema provided.
    — Interventions must be organic/biological; do NOT recommend persistent synthetic pesticides.
    — All recommendations must safeguard soil microbiome diversity and pollinator populations (SDG 15).
    — Cite evidence-based etiology (pathogen genus/species where applicable).
    — Confidence scores reflect visual evidence quality and contextual fit.
""")


def _build_user_prompt(
    chlorosis_pct: float,
    lesion_count: int,
    lesion_density: float,
    severity_score: float,
    severity_label: str,
    ctx: CropContext,
) -> str:
    return textwrap.dedent(f"""\
        ## Multimodal Diagnostic Context

        ### Visual Metrics (Computer Vision Pipeline)
        - Chlorosis / Yellowing      : {chlorosis_pct:.1f}%
        - Necrotic Lesion Count      : {lesion_count}
        - Lesion Density (per 1k px) : {lesion_density:.3f}
        - Composite Severity Score   : {severity_score:.1f}% ({severity_label})

        ### Crop Metadata
        - Crop Type      : {ctx.crop_type}
        - Growth Stage   : {ctx.growth_stage}
        - Region         : {ctx.region}

        ### Weather Telemetry
        - Humidity       : {ctx.humidity_pct:.0f}%
        - Rainfall       : {ctx.rainfall}
        - Temperature    : {ctx.temperature_c:.1f} °C

        ### Required JSON Output Schema
        {{
          "diagnosis": "<primary disease name or 'Healthy'>",
          "etiology": "<pathogen / abiotic cause with brief mechanism>",
          "confidence_pct": <0-100 float>,
          "severity_label": "<Healthy|Mild|Moderate|Severe|Critical>",
          "interventions_organic": ["<action 1>", "..."],
          "interventions_biological": ["<action 1>", "..."],
          "soil_microbiome_safeguards": ["<measure 1>", "..."],
          "pollinator_safeguards": ["<measure 1>", "..."],
          "irrigation_schedule": "<drip/cultural schedule description>",
          "sdg2_alignment": "<how this diagnosis/action supports Zero Hunger>",
          "sdg15_alignment": "<how recommendations protect Life on Land>"
        }}

        Produce ONLY valid JSON. No prose outside the JSON block.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# LLM client — IBM Granite via OpenAI-compatible endpoint
# ─────────────────────────────────────────────────────────────────────────────

def _llm_inference(
    chlorosis_pct: float,
    lesion_count: int,
    lesion_density: float,
    severity_score: float,
    severity_label: str,
    ctx: CropContext,
) -> Optional[DiagnosticReport]:
    """Run inference via the official IBM watsonx.ai SDK."""
    if not _WATSONX_AVAILABLE:
        raise RuntimeError(
            "ibm-watsonx-ai SDK is required. Install it: pip install ibm-watsonx-ai"
        )

    api_key = os.getenv("GRANITE_API_KEY")
    url = os.getenv("GRANITE_BASE_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    model_id = os.getenv("GRANITE_MODEL", "ibm/granite-3-8b-instruct")

    if not api_key:
        raise RuntimeError("No API key found. Set GRANITE_API_KEY in your environment.")
    if not project_id:
        raise RuntimeError("WATSONX_PROJECT_ID is missing. Add your watsonx Project ID to .env or secrets.")

    user_prompt = _build_user_prompt(
        chlorosis_pct, lesion_count, lesion_density,
        severity_score, severity_label, ctx,
    )
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"

    try:
        credentials = Credentials(url=url, api_key=api_key)
        params = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 1024,
            GenParams.TEMPERATURE: 0.2,
        }
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params=params,
        )
        response_text = model.generate_text(prompt=full_prompt)
        return _parse_llm_response(response_text, severity_label)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Watsonx call failed: {exc}") from exc

    api_key  = os.getenv("GRANITE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("GRANITE_BASE_URL", "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation")
    model    = os.getenv("GRANITE_MODEL", "ibm/granite-3-8b-instruct")

    if not api_key:
        raise RuntimeError(
            "No API key found. Set GRANITE_API_KEY (or OPENAI_API_KEY) in your environment."
        )

    try:
        client = _OpenAI(api_key=api_key, base_url=base_url)
        user_prompt = _build_user_prompt(
            chlorosis_pct, lesion_count, lesion_density,
            severity_score, severity_label, ctx,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        return _parse_llm_response(raw, severity_label)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def _parse_llm_response(raw: str, severity_label: str) -> DiagnosticReport:
    """Extract the JSON block from LLM output and map to DiagnosticReport."""
    import json

    # Pull out JSON block (the model may wrap with ```json ... ```)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise RuntimeError("LLM response contained no JSON block.")

    try:
        data = json.loads(match.group())
        return DiagnosticReport(
            diagnosis=data["diagnosis"],
            etiology=data["etiology"],
            confidence_pct=float(data["confidence_pct"]),
            severity_label=data.get("severity_label", severity_label),
            interventions_organic=data["interventions_organic"],
            interventions_biological=data["interventions_biological"],
            soil_microbiome_safeguards=data["soil_microbiome_safeguards"],
            pollinator_safeguards=data["pollinator_safeguards"],
            irrigation_schedule=data["irrigation_schedule"],
            sdg2_alignment=data["sdg2_alignment"],
            sdg15_alignment=data["sdg15_alignment"],
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Failed to parse LLM JSON response: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_diagnostic_report(
    chlorosis_pct: float,
    lesion_count: int,
    lesion_density: float,
    severity_score: float,
    severity_label: str,
    ctx: CropContext,
) -> DiagnosticReport:
    """
    Entry-point: run LLM inference via IBM Granite.

    Args:
        chlorosis_pct   : Chlorosis percentage from vision engine.
        lesion_count    : Raw lesion contour count.
        lesion_density  : Lesions per 1 000 px².
        severity_score  : Composite 0–100 score.
        severity_label  : Human label (Healthy/Mild/Moderate/Severe/Critical).
        ctx             : CropContext with crop metadata and weather telemetry.

    Returns:
        DiagnosticReport dataclass.

    Raises:
        RuntimeError: if the openai SDK is missing or no API key is configured.
    """
    return _llm_inference(
        chlorosis_pct, lesion_count, lesion_density,
        severity_score, severity_label, ctx,
    )
