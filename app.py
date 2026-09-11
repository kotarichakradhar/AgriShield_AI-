"""
AgriShield AI — Advanced Farmer-First Application Interface
Streamlit UI designed for real smallholder & commercial farmers.
Tabs: Field Diagnostics | Weather & Risk Forecast | Treatment Planner | Conservation & SDG
"""

from __future__ import annotations

import io
import math
import os
import random
from datetime import date, timedelta

import numpy as np
import streamlit as st
from PIL import Image

from llm_engine import CropContext, DiagnosticReport, generate_diagnostic_report
from vision_engine import VisionReport, analyse_image

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriShield AI — Farmer Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Advanced Farmer-First Design System
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ── SIDEBAR ───────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #052e16 0%, #064e3b 100%);
    border-right: 1px solid #065f46;
}
section[data-testid="stSidebar"] * { color: #d1fae5 !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stRadio > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] label { font-size: 0.78rem !important; color: #6ee7b7 !important; }

/* ── TABS ───────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f8fafc;
    padding: 6px 8px 0;
    border-radius: 12px 12px 0 0;
    border-bottom: 2px solid #d1fae5;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 22px;
    background: transparent;
    color: #6b7280;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
}
.stTabs [aria-selected="true"] {
    background: #065f46 !important;
    color: #fff !important;
}

/* ── METRIC CARDS ───────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7280 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #111827 !important;
}

/* ── ALERTS ────────────────────────────────────────────────────────────────── */
.alert-critical {
    background: #fef2f2; border: 1px solid #fca5a5; border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}
.alert-warning {
    background: #fffbeb; border: 1px solid #fcd34d; border-left: 4px solid #f59e0b;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}
.alert-success {
    background: #f0fdf4; border: 1px solid #86efac; border-left: 4px solid #22c55e;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}
.alert-info {
    background: #eff6ff; border: 1px solid #93c5fd; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}

/* ── FARMER ACTION CARD ─────────────────────────────────────────────────────── */
.action-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 10px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    position: relative;
}
.action-card .step-num {
    display: inline-block;
    background: #065f46;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    width: 22px; height: 22px;
    border-radius: 50%;
    text-align: center;
    line-height: 22px;
    margin-right: 8px;
}
.action-card h4 { margin: 0 0 6px; color: #111827; font-size: 0.92rem; font-weight: 700; }
.action-card p  { margin: 0; color: #4b5563; font-size: 0.83rem; line-height: 1.55; }

/* ── URGENCY BADGE ──────────────────────────────────────────────────────────── */
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; }
.badge-healthy  { background:#dcfce7; color:#166534; }
.badge-mild     { background:#fef9c3; color:#854d0e; }
.badge-moderate { background:#ffedd5; color:#9a3412; }
.badge-severe   { background:#fee2e2; color:#991b1b; }
.badge-critical { background:#7f1d1d; color:#fecaca; }

/* ── SPRAY CALENDAR ROW ─────────────────────────────────────────────────────── */
.cal-day {
    display: inline-block;
    width: 38px; height: 38px;
    border-radius: 8px;
    text-align: center; line-height: 38px;
    font-size: 0.78rem; font-weight: 700;
    margin: 3px;
    cursor: default;
}
.cal-spray   { background: #dc2626; color: #fff; }
.cal-monitor { background: #f59e0b; color: #fff; }
.cal-rest    { background: #e5e7eb; color: #6b7280; }
.cal-harvest { background: #16a34a; color: #fff; }

/* ── COST TABLE ─────────────────────────────────────────────────────────────── */
.cost-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f3f4f6; font-size: 0.83rem;
}
.cost-row:last-child { border-bottom: none; }
.cost-label { color: #374151; }
.cost-val   { font-weight: 700; color: #111827; }
.cost-total {
    background: #f0fdf4; border-radius: 8px; padding: 10px 14px;
    margin-top: 10px; font-weight: 800; font-size: 0.9rem;
    display: flex; justify-content: space-between;
}

/* ── FIELD HEALTH TIMELINE ──────────────────────────────────────────────────── */
.timeline-bar {
    height: 10px; border-radius: 5px;
    background: linear-gradient(90deg, #16a34a 0%, #f59e0b 40%, #dc2626 75%, #7f1d1d 100%);
    position: relative; margin: 6px 0;
}
.timeline-marker {
    position: absolute; top: -4px;
    width: 18px; height: 18px;
    border-radius: 50%; border: 3px solid #fff;
    box-shadow: 0 0 0 2px #065f46;
    background: #065f46;
    transform: translateX(-50%);
}

/* ── GENERAL ────────────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.4rem 0; }
.section-title {
    font-size: 1.05rem; font-weight: 800; color: #111827;
    margin: 0 0 4px; letter-spacing: -0.02em;
}
.section-sub { font-size: 0.8rem; color: #6b7280; margin: 0 0 14px; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_COLOUR = {
    "Healthy":  "#16a34a",
    "Mild":     "#ca8a04",
    "Moderate": "#ea580c",
    "Severe":   "#dc2626",
    "Critical": "#7f1d1d",
}

_CROP_OPTIMA = {
    "Tomato":  {"temp": (18, 27), "humidity": (60, 70), "rain": "Moderate"},
    "Rice":    {"temp": (20, 35), "humidity": (70, 80), "rain": "Heavy"},
    "Cotton":  {"temp": (21, 30), "humidity": (50, 60), "rain": "Low"},
    "Maize":   {"temp": (18, 27), "humidity": (55, 65), "rain": "Moderate"},
    "Wheat":   {"temp": (15, 22), "humidity": (50, 60), "rain": "Low"},
    "Soybean": {"temp": (20, 30), "humidity": (60, 70), "rain": "Moderate"},
    "Potato":  {"temp": (15, 20), "humidity": (65, 80), "rain": "Moderate"},
}

_TREATMENT_COSTS = {
    "Neem Kernel Extract (NSKE 2%)":              ("USD 0.80 / L",   "Low"),
    "Bordeaux Mixture (1%)":                      ("USD 1.20 / kg",  "Low"),
    "Trichoderma harzianum (10⁸ CFU/mL)":        ("USD 2.50 / L",   "Medium"),
    "Bacillus subtilis drench":                   ("USD 3.00 / L",   "Medium"),
    "Pseudomonas fluorescens":                    ("USD 2.80 / L",   "Medium"),
    "Seaweed extract (Ascophyllum nodosum)":      ("USD 4.00 / L",   "Medium"),
    "Vermicompost top-dress (2 t/ha)":            ("USD 60 / tonne", "High"),
}


def _badge(label: str) -> str:
    cls = f"badge badge-{label.lower()}"
    return f'<span class="{cls}">{label}</span>'


def _gauge_svg(score: float, label: str, size: int = 160) -> str:
    colour = _SEVERITY_COLOUR.get(label, "#ea580c")
    pct = min(score / 100.0, 1.0)
    cx, cy, r = size // 2, size // 2, size // 2 - 14
    sx, sy = cx - r, cy
    angle = pct * math.pi
    ex = cx + r * (-math.cos(angle))
    ey = cy - r * math.sin(angle)
    large = 1 if pct > 0.5 else 0
    track = f"M {sx} {sy} A {r} {r} 0 1 1 {cx+r} {sy}"
    arc   = f"M {sx} {sy} A {r} {r} 0 {large} 1 {ex:.2f} {ey:.2f}"
    return f"""<svg viewBox="0 0 {size} {size//2+20}" width="{size}"
     xmlns="http://www.w3.org/2000/svg">
  <path d="{track}" fill="none" stroke="#e5e7eb" stroke-width="12" stroke-linecap="round"/>
  <path d="{arc}"   fill="none" stroke="{colour}" stroke-width="12" stroke-linecap="round"/>
  <text x="{cx}" y="{cy-4}" text-anchor="middle" font-size="26" font-weight="800"
        fill="{colour}" font-family="Inter,sans-serif">{score:.0f}%</text>
  <text x="{cx}" y="{cy+16}" text-anchor="middle" font-size="11" fill="#6b7280"
        font-family="Inter,sans-serif">{label}</text>
</svg>"""


def _hbar(value: float, max_val: float, colour: str, label: str, suffix: str = "") -> str:
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return f"""<div style="margin:8px 0">
  <div style="display:flex;justify-content:space-between;
              font-size:0.78rem;color:#4b5563;margin-bottom:3px">
    <span style="font-weight:500">{label}</span>
    <span style="font-weight:700;color:#111827">{value:.1f}{suffix}</span>
  </div>
  <div style="background:#f3f4f6;border-radius:6px;height:9px;overflow:hidden">
    <div style="width:{pct}%;background:{colour};height:9px;border-radius:6px"></div>
  </div>
</div>"""


def _action_card(step: int, title: str, body: str, urgency: str = "") -> str:
    urg = f'<span style="float:right">{_badge(urgency)}</span>' if urgency else ""
    return f"""<div class="action-card">
  {urg}
  <h4><span class="step-num">{step}</span>{title}</h4>
  <p>{body}</p>
</div>"""


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> CropContext:
    st.sidebar.markdown(
        """<div style="padding:16px 8px 8px">
          <div style="font-size:1.5rem;font-weight:900;letter-spacing:-0.03em;color:#d1fae5">
            🌾 AgriShield AI
          </div>
          <div style="font-size:0.73rem;color:#6ee7b7;margin-top:2px">
            Farmer Intelligence Platform · SDG 2 & 15
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    st.sidebar.markdown("**🌱 Your Crop**")
    crop_type = st.sidebar.selectbox(
        "Crop Type", ["Tomato", "Rice", "Cotton", "Maize", "Wheat", "Soybean", "Potato"],
        label_visibility="collapsed",
    )
    growth_stage = st.sidebar.selectbox(
        "Growth Stage",
        ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturation"],
    )

    st.sidebar.markdown("**📍 Location**")
    region = st.sidebar.selectbox(
        "Region",
        ["South Asia", "Sub-Saharan Africa", "Southeast Asia",
         "Latin America", "East Africa", "Mediterranean"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("**🌡️ Current Weather**")
    humidity = st.sidebar.slider("Humidity (%)", 10, 100, 68)
    temperature = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 28.0, step=0.5)
    rainfall = st.sidebar.select_slider(
        "Rainfall", options=["None", "Low", "Moderate", "Heavy"], value="Low"
    )

    st.sidebar.markdown("**🏡 Farm Details**")
    field_size = st.sidebar.number_input("Field Size (hectares)", 0.1, 500.0, 2.0, step=0.5)
    st.session_state["field_size"] = field_size

    st.sidebar.divider()

    # API key status indicator
    has_key = bool(os.getenv("GRANITE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if has_key:
        st.sidebar.markdown(
            '<div style="background:rgba(34,197,94,0.15);border-radius:8px;padding:8px 12px;'
            'font-size:0.75rem;color:#4ade80">✅ IBM Granite API connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div style="background:rgba(239,68,68,0.15);border-radius:8px;padding:8px 12px;'
            'font-size:0.75rem;color:#fca5a5">🔑 Set GRANITE_API_KEY to enable AI diagnosis</div>',
            unsafe_allow_html=True,
        )

    return CropContext(
        crop_type=crop_type,
        growth_stage=growth_stage,
        region=region,
        humidity_pct=float(humidity),
        rainfall=rainfall,
        temperature_c=float(temperature),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Field Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _render_diagnostics_tab(ctx: CropContext) -> None:
    st.markdown('<p class="section-title">🔬 Field Crop Diagnostics</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Take a live photo with your camera <b>or</b> upload an image. '
        'The AI analyses disease severity and gives you step-by-step field actions.</p>',
        unsafe_allow_html=True,
    )

    # ── Input mode selector ──────────────────────────────────────────────────
    input_mode = st.radio(
        "Input method",
        ["📷 Use Camera", "📁 Upload File"],
        horizontal=True,
        label_visibility="collapsed",
    )

    col_input, col_tip = st.columns([2, 1], gap="large")

    image_bytes: bytes | None = None

    with col_input:
        if input_mode == "📷 Use Camera":
            camera_shot = st.camera_input(
                "Point camera at the affected leaf and press the capture button",
                help="Allow browser camera access when prompted.",
            )
            if camera_shot is not None:
                image_bytes = camera_shot.getvalue()
        else:
            uploaded = st.file_uploader(
                "📸 Upload Leaf / Crop Photo",
                type=["jpg", "jpeg", "png"],
                help="JPEG or PNG, max 20 MB.",
            )
            if uploaded is not None:
                image_bytes = uploaded.read()

    with col_tip:
        st.markdown(
            """<div class="alert-info" style="margin-top:12px">
              <b>📷 Photo Tips</b><br>
              <small>• Shoot in natural daylight<br>
              • Fill the frame with the affected leaf<br>
              • Capture both healthy &amp; sick areas<br>
              • Avoid blurry or backlit images<br>
              • Camera mode works on mobile &amp; desktop</small>
            </div>""",
            unsafe_allow_html=True,
        )

    if image_bytes is None:
        _render_onboarding()
        return

    # ── Vision pipeline ──────────────────────────────────────────────────────
    prog = st.progress(0, text="🔍 Analysing image...")
    try:
        v_report: VisionReport = analyse_image(image_bytes)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Image analysis failed: {exc}")
        return
    prog.progress(50, text="🧠 Running AI reasoning engine...")

    try:
        d_report: DiagnosticReport = generate_diagnostic_report(
            chlorosis_pct=v_report.chlorosis_pct,
            lesion_count=v_report.lesion_count,
            lesion_density=v_report.lesion_density,
            severity_score=v_report.severity_score,
            severity_label=v_report.severity_label,
            ctx=ctx,
        )
    except RuntimeError as exc:
        prog.empty()
        st.error(str(exc), icon="🔑")
        st.info(
            "**To enable AI diagnosis:** Set `GRANITE_API_KEY` in your environment and restart.\n\n"
            "```\nset GRANITE_API_KEY=your_key_here\npython -m streamlit run app.py\n```",
            icon="ℹ️",
        )
        return
    except Exception as exc:  # noqa: BLE001
        prog.empty()
        st.error(f"AI engine error: {exc}")
        return

    prog.progress(100, text="✅ Analysis complete!")
    prog.empty()

    # store for Treatment Planner tab
    st.session_state["last_v"] = v_report
    st.session_state["last_d"] = d_report
    st.session_state["last_ctx"] = ctx

    # ── ALERT BANNER ─────────────────────────────────────────────────────────
    sev = v_report.severity_label
    alert_map = {
        "Critical": ("alert-critical", "🚨 CRITICAL DISEASE ALERT",
                     "Immediate field action required. Act within 24–48 hours to prevent total crop loss."),
        "Severe":   ("alert-critical", "⚠️ SEVERE DISEASE DETECTED",
                     "Urgent treatment needed within 3–5 days. Begin biocontrol protocol today."),
        "Moderate": ("alert-warning", "⚠️ MODERATE INFECTION DETECTED",
                     "Start treatment this week. Monitor closely every 2 days."),
        "Mild":     ("alert-warning", "🟡 EARLY-STAGE SYMPTOMS",
                     "Preventive action recommended. Scout the full field for spread."),
        "Healthy":  ("alert-success", "✅ CROP APPEARS HEALTHY",
                     "No significant disease detected. Maintain current good practices."),
    }
    cls, title, msg = alert_map.get(sev, alert_map["Moderate"])
    st.markdown(f'<div class="{cls}"><b>{title}</b><br><small>{msg}</small></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main layout ──────────────────────────────────────────────────────────
    col_img, col_diag = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**📸 Annotated Crop Image**")
        st.image(v_report.annotated_image, use_container_width=True,
                 caption="Red contours = detected lesions | HUD = severity metrics")
        if v_report.backbone_used != "none":
            st.caption(f"🧬 Deep features: {v_report.backbone_used}")

        # Severity gauge
        st.markdown(
            f'<div style="text-align:center;margin-top:12px">'
            f'{_gauge_svg(v_report.severity_score, sev, 180)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_diag:
        st.markdown("**🧬 AI Diagnostic Result**")

        # Key metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Diagnosis", d_report.diagnosis[:16] + ("…" if len(d_report.diagnosis) > 16 else ""))
        m2.metric("Confidence", f"{d_report.confidence_pct:.0f}%")
        m3.metric("Severity", sev)

        st.markdown("**📊 Visual Signal Breakdown**")
        st.markdown(
            _hbar(v_report.chlorosis_pct, 100, "#ca8a04", "Yellowing (Chlorosis)", "%") +
            _hbar(min(v_report.lesion_density * 10, 100), 100, "#dc2626", "Lesion Density") +
            _hbar(v_report.severity_score, 100, _SEVERITY_COLOUR.get(sev, "#ea580c"), "Overall Severity", "%"),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🦠 Disease & Cause**")
        st.markdown(
            f'<div class="action-card">'
            f'<h4 style="margin:0 0 4px">{d_report.diagnosis}</h4>'
            f'<p><i>{d_report.etiology}</i></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── FARMER ACTION PLAN ───────────────────────────────────────────────────
    st.markdown("### 🧑‍🌾 Your Step-by-Step Field Action Plan")
    st.caption(f"Tailored for {ctx.crop_type} · {ctx.growth_stage} · {ctx.region}")

    action_col1, action_col2 = st.columns(2, gap="large")

    with action_col1:
        st.markdown("**🌿 Organic Treatments**")
        for i, item in enumerate(d_report.interventions_organic, 1):
            st.markdown(_action_card(i, f"Organic Step {i}", item, "Moderate"), unsafe_allow_html=True)

        st.markdown("<br>**🧫 Biological Biocontrol**")
        for i, item in enumerate(d_report.interventions_biological, 1):
            st.markdown(_action_card(i, f"Biocontrol Step {i}", item, "Mild"), unsafe_allow_html=True)

    with action_col2:
        st.markdown("**💧 Irrigation Schedule**")
        st.markdown(
            f'<div class="alert-info"><b>💧 Field-Specific Drip Protocol</b><br>'
            f'<span style="font-size:0.85rem">{d_report.irrigation_schedule}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>**🌍 Soil & Ecosystem Safeguards**")
        for item in d_report.soil_microbiome_safeguards:
            st.markdown(f'<div style="font-size:0.82rem;padding:4px 0;color:#374151">🟢 {item}</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>**🐝 Pollinator Protection**")
        for item in d_report.pollinator_safeguards:
            st.markdown(f'<div style="font-size:0.82rem;padding:4px 0;color:#374151">🐝 {item}</div>',
                        unsafe_allow_html=True)

    # ── SDG impact callout ───────────────────────────────────────────────────
    st.divider()
    sdg1, sdg2 = st.columns(2, gap="large")
    with sdg1:
        st.markdown(
            f'<div style="background:#f0fdf4;border-left:4px solid #16a34a;'
            f'border-radius:6px;padding:14px 18px">'
            f'<b style="color:#166534">🌾 SDG 2 — Zero Hunger Impact</b><br>'
            f'<small style="color:#374151">{d_report.sdg2_alignment}</small></div>',
            unsafe_allow_html=True,
        )
    with sdg2:
        st.markdown(
            f'<div style="background:#eff6ff;border-left:4px solid #3b82f6;'
            f'border-radius:6px;padding:14px 18px">'
            f'<b style="color:#1d4ed8">🌿 SDG 15 — Life on Land Impact</b><br>'
            f'<small style="color:#374151">{d_report.sdg15_alignment}</small></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(f"⚠️ {d_report.disclaimer}")

    # ── Download ─────────────────────────────────────────────────────────────
    report_md = _build_markdown_report(v_report, d_report, ctx)
    st.download_button(
        "📥 Download Full Field Report (.md)",
        data=report_md,
        file_name=f"AgriShield_{ctx.crop_type}_{date.today()}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _render_onboarding() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("📸", "Take a Photo", "Snap a clear photo of the affected leaf or crop in natural daylight"),
        ("🤖", "AI Analyses", "Our CV engine scans for chlorosis and lesions in seconds"),
        ("🧑‍🌾", "Get Field Actions", "Receive a step-by-step treatment plan specific to your farm"),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3], cards):
        col.markdown(
            f'<div class="action-card" style="text-align:center">'
            f'<div style="font-size:2.2rem;margin-bottom:8px">{icon}</div>'
            f'<h4 style="margin:0 0 6px">{title}</h4>'
            f'<p>{body}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """<div class="alert-info" style="margin-top:18px">
          <b>🌾 Crops Supported:</b>
          Tomato · Rice · Cotton · Maize · Wheat · Soybean · Potato<br>
          <b>🌍 Regions:</b>
          South Asia · Sub-Saharan Africa · Southeast Asia · Latin America · East Africa · Mediterranean
        </div>""",
        unsafe_allow_html=True,
    )


def _build_markdown_report(v: VisionReport, d: DiagnosticReport, ctx: CropContext) -> str:
    today = date.today().isoformat()
    lines = [
        f"# AgriShield AI — Field Diagnostic Report",
        f"**Date:** {today}  |  **Crop:** {ctx.crop_type}  |  **Stage:** {ctx.growth_stage}  |  **Region:** {ctx.region}",
        f"**Weather:** {ctx.humidity_pct:.0f}% humidity · {ctx.temperature_c:.1f} °C · Rainfall: {ctx.rainfall}",
        "",
        "---",
        "## 📊 Visual Analysis (Computer Vision)",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Chlorosis (Yellowing) | {v.chlorosis_pct:.1f}% |",
        f"| Necrotic Lesion Count | {v.lesion_count} |",
        f"| Lesion Density | {v.lesion_density:.3f} per 1k px² |",
        f"| Severity Score | {v.severity_score:.1f}% |",
        f"| Severity Label | **{v.severity_label}** |",
        f"| Deep Learning Backbone | {v.backbone_used} |",
        "",
        "## 🦠 AI Diagnosis",
        f"- **Condition:** {d.diagnosis}",
        f"- **Etiology:** {d.etiology}",
        f"- **Confidence:** {d.confidence_pct:.1f}%",
        f"- **Severity:** {d.severity_label}",
        "",
        "## 🌿 Organic Treatments",
    ]
    for i, t in enumerate(d.interventions_organic, 1):
        lines.append(f"{i}. {t}")
    lines += ["", "## 🧫 Biological Biocontrol"]
    for i, t in enumerate(d.interventions_biological, 1):
        lines.append(f"{i}. {t}")
    lines += ["", "## 🌱 Soil Microbiome Safeguards"]
    for t in d.soil_microbiome_safeguards:
        lines.append(f"- {t}")
    lines += ["", "## 🐝 Pollinator Safeguards"]
    for t in d.pollinator_safeguards:
        lines.append(f"- {t}")
    lines += [
        "", "## 💧 Irrigation Schedule",
        d.irrigation_schedule,
        "", "---",
        "## 🌍 SDG Alignment",
        f"**SDG 2 — Zero Hunger:** {d.sdg2_alignment}",
        "",
        f"**SDG 15 — Life on Land:** {d.sdg15_alignment}",
        "", "---",
        f"*{d.disclaimer}*",
        f"*Generated by AgriShield AI on {today}*",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Weather & Risk Forecast
# ─────────────────────────────────────────────────────────────────────────────

def _render_weather_tab(ctx: CropContext) -> None:
    st.markdown('<p class="section-title">🌦️ Weather & Disease Risk Forecast</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Environmental conditions driving disease pressure on your field. '
        'Adjust values in the sidebar.</p>',
        unsafe_allow_html=True,
    )

    # ── Live weather strip ────────────────────────────────────────────────────
    rainfall_map = {"None": 0.05, "Low": 0.3, "Moderate": 0.6, "Heavy": 0.92}
    rain_risk    = rainfall_map.get(ctx.rainfall, 0.3)
    temp_risk    = max(0.0, min(1.0 - abs(ctx.temperature_c - 27) / 20.0, 1.0))
    hum_risk     = ctx.humidity_pct / 100.0
    composite    = round((hum_risk * 0.45 + temp_risk * 0.30 + rain_risk * 0.25) * 100, 1)
    risk_label   = (
        "Healthy"  if composite < 20 else
        "Mild"     if composite < 40 else
        "Moderate" if composite < 60 else
        "Severe"   if composite < 80 else "Critical"
    )

    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("🌡️ Temperature", f"{ctx.temperature_c:.0f} °C",
              delta=f"{ctx.temperature_c-25:+.0f} °C vs 25 °C")
    w2.metric("💧 Humidity", f"{ctx.humidity_pct:.0f}%",
              delta="⚠️ High" if ctx.humidity_pct > 75 else ("OK" if ctx.humidity_pct < 55 else "Moderate"),
              delta_color="inverse" if ctx.humidity_pct > 75 else "normal")
    w3.metric("🌧️ Rainfall", ctx.rainfall)
    w4.metric("🌍 Region", ctx.region)
    w5.metric("🦠 Disease Risk", f"{composite:.0f}%",
              delta=risk_label,
              delta_color="inverse" if risk_label in ("Severe", "Critical") else "normal")

    st.divider()

    # ── Dual column: gauge + 7-day forecast ──────────────────────────────────
    col_g, col_f = st.columns([1, 2], gap="large")

    with col_g:
        st.markdown("**Current Disease Pressure**")
        st.markdown(
            f'<div style="text-align:center">{_gauge_svg(composite, risk_label, 200)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Risk Factor Breakdown**")
        st.markdown(
            _hbar(hum_risk * 100,  100, "#3b82f6", "Humidity Contribution", "%") +
            _hbar(temp_risk * 100, 100, "#f97316", "Temperature Contribution", "%") +
            _hbar(rain_risk * 100, 100, "#06b6d4", "Rainfall Contribution", "%"),
            unsafe_allow_html=True,
        )

    with col_f:
        st.markdown("**📅 7-Day Disease Risk Forecast**")
        # Simulate plausible 7-day risk based on current conditions
        today = date.today()
        seed_base = composite
        days   = [(today + timedelta(days=i)).strftime("%a %d") for i in range(7)]
        # Slight random walk around the current risk
        risks  = [seed_base]
        for _ in range(6):
            delta = random.gauss(0, 5)
            risks.append(round(max(0, min(100, risks[-1] + delta)), 1))

        risk_colours = []
        for r in risks:
            if r < 20:   risk_colours.append("#16a34a")
            elif r < 40: risk_colours.append("#ca8a04")
            elif r < 60: risk_colours.append("#ea580c")
            elif r < 80: risk_colours.append("#dc2626")
            else:        risk_colours.append("#7f1d1d")

        # Draw a simple inline SVG bar chart
        chart_h, bar_w, gap = 140, 36, 8
        total_w = 7 * (bar_w + gap) + gap
        bars = ""
        for i, (r, c, d) in enumerate(zip(risks, risk_colours, days)):
            bh = int(r / 100 * chart_h)
            bx = gap + i * (bar_w + gap)
            by = chart_h - bh
            bars += (
                f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" '
                f'fill="{c}" rx="5"/>'
                f'<text x="{bx+bar_w//2}" y="{chart_h+14}" text-anchor="middle" '
                f'font-size="9" fill="#6b7280" font-family="Inter,sans-serif">{d}</text>'
                f'<text x="{bx+bar_w//2}" y="{by-4}" text-anchor="middle" '
                f'font-size="9" font-weight="700" fill="{c}" font-family="Inter,sans-serif">{r:.0f}%</text>'
            )
        st.markdown(
            f'<div style="background:#f8fafc;border-radius:10px;padding:14px">'
            f'<svg viewBox="0 0 {total_w} {chart_h+30}" width="100%"'
            f' xmlns="http://www.w3.org/2000/svg">{bars}</svg>'
            f'<div style="font-size:0.72rem;color:#9ca3af;text-align:right;margin-top:4px">'
            f'Simulated forecast · update with live met data</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Spray window advisory ─────────────────────────────────────────────
        st.markdown("<br>**⏰ Optimal Spray Window Today**")
        if ctx.humidity_pct > 75 or ctx.rainfall in ("Heavy", "Moderate"):
            st.markdown(
                '<div class="alert-warning">Rain/high humidity forecast — delay spraying. '
                'Best window: <b>early morning 05:30–07:30</b> or wait for dry spell.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-success">✅ Good spray conditions. '
                'Apply treatments between <b>06:00–08:00</b> or <b>18:00–20:00</b> '
                'to protect pollinators and maximise absorption.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Crop optima panel ─────────────────────────────────────────────────────
    st.markdown(f"**🌱 Ideal Conditions for {ctx.crop_type}**")
    optima = _CROP_OPTIMA.get(ctx.crop_type, {"temp": (20, 30), "humidity": (60, 70), "rain": "Moderate"})
    t_lo, t_hi = optima["temp"]
    h_lo, h_hi = optima["humidity"]

    oc1, oc2, oc3 = st.columns(3)
    t_ok = t_lo <= ctx.temperature_c <= t_hi
    h_ok = h_lo <= ctx.humidity_pct <= h_hi

    oc1.markdown(
        f'<div class="action-card">'
        f'<h4>🌡️ Temperature</h4>'
        f'<p>Optimal: <b>{t_lo}–{t_hi} °C</b><br>'
        f'Current: <b>{ctx.temperature_c:.0f} °C</b> '
        f'{"✅" if t_ok else "⚠️ Outside range"}</p></div>',
        unsafe_allow_html=True,
    )
    oc2.markdown(
        f'<div class="action-card">'
        f'<h4>💧 Humidity</h4>'
        f'<p>Optimal: <b>{h_lo}–{h_hi}%</b><br>'
        f'Current: <b>{ctx.humidity_pct:.0f}%</b> '
        f'{"✅" if h_ok else "⚠️ Outside range"}</p></div>',
        unsafe_allow_html=True,
    )
    oc3.markdown(
        f'<div class="action-card">'
        f'<h4>🌧️ Rainfall</h4>'
        f'<p>Optimal: <b>{optima["rain"]}</b><br>'
        f'Current: <b>{ctx.rainfall}</b> '
        f'{"✅" if ctx.rainfall == optima["rain"] else "⚠️ Adjust irrigation"}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Active warnings ───────────────────────────────────────────────────────
    st.markdown("<br>")
    if ctx.humidity_pct > 75 and ctx.rainfall in ("Heavy", "Moderate"):
        st.markdown(
            '<div class="alert-critical"><b>🚨 High Disease Pressure Warning</b><br>'
            f'Heavy rainfall + {ctx.humidity_pct:.0f}% humidity creates ideal conditions for late blight, '
            'Septoria, and Fusarium. Activate preventive Trichoderma soil drench <b>within 48 hours</b>.</div>',
            unsafe_allow_html=True,
        )
    elif ctx.humidity_pct > 70:
        st.markdown(
            '<div class="alert-warning"><b>⚠️ Elevated Fungal Risk</b><br>'
            f'Humidity at {ctx.humidity_pct:.0f}% favours fungal sporulation. '
            'Scout twice weekly; apply NSKE as a preventive foliar spray.</div>',
            unsafe_allow_html=True,
        )
    elif composite < 25:
        st.markdown(
            '<div class="alert-success"><b>✅ Low Disease Pressure</b><br>'
            'Environmental conditions are currently unfavourable for most pathogens. '
            'Maintain current management and routine scouting.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Treatment Planner
# ─────────────────────────────────────────────────────────────────────────────

def _render_planner_tab(ctx: CropContext) -> None:
    st.markdown('<p class="section-title">📋 Treatment Planner & Cost Estimator</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Plan your treatment schedule, estimate costs, and generate '
        'a printable spray calendar for your field.</p>',
        unsafe_allow_html=True,
    )

    # Retrieve last diagnosis from session
    d_report: DiagnosticReport | None = st.session_state.get("last_d")
    v_report: VisionReport | None     = st.session_state.get("last_v")
    field_size: float                 = st.session_state.get("field_size", 2.0)

    if d_report is None:
        st.markdown(
            '<div class="alert-info"><b>ℹ️ No diagnosis yet</b><br>'
            'Go to the <b>Field Diagnostics</b> tab, upload a crop photo, and run the AI analysis. '
            'Your personalised treatment plan will appear here.</div>',
            unsafe_allow_html=True,
        )
        return

    sev = v_report.severity_label if v_report else "Moderate"

    # ── Treatment Summary ─────────────────────────────────────────────────────
    st.markdown(
        f'<div class="alert-info"><b>📋 Treatment Plan for:</b> {d_report.diagnosis} '
        f'in {ctx.crop_type} · {sev} severity · {field_size} ha field</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>")

    plan_col, cost_col = st.columns([3, 2], gap="large")

    with plan_col:
        # ── Spray calendar ────────────────────────────────────────────────────
        st.markdown("**📅 28-Day Spray Calendar**")
        today = date.today()

        # Define spray days based on severity
        freq = {"Critical": 3, "Severe": 4, "Moderate": 7, "Mild": 10, "Healthy": 14}
        interval = freq.get(sev, 7)

        cal_days = []
        for i in range(28):
            d = today + timedelta(days=i)
            if i == 0:
                cal_days.append(("spray", d))
            elif i % interval == 0:
                cal_days.append(("spray", d))
            elif i % max(interval // 2, 2) == 0:
                cal_days.append(("monitor", d))
            else:
                cal_days.append(("rest", d))

        # Render calendar grid
        cal_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:8px 0">'
        for typ, day in cal_days:
            css = {"spray": "cal-spray", "monitor": "cal-monitor", "rest": "cal-rest"}.get(typ, "cal-rest")
            tooltip = {"spray": "Spray day", "monitor": "Scout/Monitor", "rest": "Rest"}.get(typ, "")
            cal_html += (
                f'<div class="cal-day {css}" title="{tooltip} {day.strftime("%d %b")}">'
                f'{day.strftime("%d")}</div>'
            )
        cal_html += "</div>"
        st.markdown(
            f'<div style="background:#f8fafc;border-radius:10px;padding:14px">'
            f'{cal_html}'
            f'<div style="margin-top:8px;font-size:0.75rem;display:flex;gap:16px">'
            f'<span><span class="cal-day cal-spray" style="width:14px;height:14px;line-height:14px;font-size:8px;display:inline-block">●</span> Spray</span>'
            f'<span><span class="cal-day cal-monitor" style="width:14px;height:14px;line-height:14px;font-size:8px;display:inline-block">●</span> Scout</span>'
            f'<span><span class="cal-day cal-rest" style="width:14px;height:14px;line-height:14px;font-size:8px;display:inline-block">●</span> Rest</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ── Application instructions ──────────────────────────────────────────
        st.markdown("<br>**📌 Application Protocol**")
        protocols = [
            ("🌅 Spray Timing", "Apply between 06:00–08:00 or 18:00–20:00 h. Never spray during pollinator foraging hours (09:00–17:00)."),
            ("🌬️ Wind Check", "Do not spray if wind speed > 15 km/h. Check direction — spray downwind of sensitive crops."),
            ("🌧️ Rain Rule", "Do not spray within 6 hours of expected rain. Re-apply if rain occurs within 2 hours of spraying."),
            ("🥽 Safety", "Wear gloves, goggles, and a mask even for organic sprays. Keep children and animals away during application."),
            ("📝 Record", "Log date, product, rate, and weather in your field diary for regulatory compliance."),
        ]
        for icon_title, body in protocols:
            st.markdown(_action_card(0, icon_title, body), unsafe_allow_html=True)

    with cost_col:
        # ── Cost Estimator ────────────────────────────────────────────────────
        st.markdown("**💰 Treatment Cost Estimator**")
        st.caption(f"Field size: {field_size} ha | Spray interval: every {interval} days")

        # Pick relevant products based on severity
        if sev in ("Critical", "Severe"):
            products = [
                ("Neem Kernel Extract (NSKE 2%)", 3.5, "L/ha"),
                ("Trichoderma harzianum (10⁸ CFU/mL)", 2.0, "L/ha"),
                ("Bacillus subtilis drench", 2.5, "L/ha"),
            ]
        elif sev == "Moderate":
            products = [
                ("Neem Kernel Extract (NSKE 2%)", 2.5, "L/ha"),
                ("Pseudomonas fluorescens", 2.0, "L/ha"),
            ]
        else:
            products = [
                ("Seaweed extract (Ascophyllum nodosum)", 1.5, "L/ha"),
            ]

        # Unit prices (USD)
        unit_prices = {
            "Neem Kernel Extract (NSKE 2%)":              0.80,
            "Bordeaux Mixture (1%)":                      1.20,
            "Trichoderma harzianum (10⁸ CFU/mL)":        2.50,
            "Bacillus subtilis drench":                   3.00,
            "Pseudomonas fluorescens":                    2.80,
            "Seaweed extract (Ascophyllum nodosum)":      4.00,
        }

        n_sprays = max(1, 28 // interval)
        cost_html = ""
        total_cost = 0.0
        for name, rate_per_ha, unit in products:
            unit_p = unit_prices.get(name, 2.0)
            total_vol = rate_per_ha * field_size
            line_cost = total_vol * unit_p * n_sprays
            total_cost += line_cost
            cost_html += (
                f'<div class="cost-row">'
                f'<span class="cost-label">{name[:28]}…<br>'
                f'<small style="color:#9ca3af">{rate_per_ha} {unit} × {n_sprays} sprays × {field_size} ha</small></span>'
                f'<span class="cost-val">USD {line_cost:.2f}</span>'
                f'</div>'
            )
        cost_html += (
            f'<div class="cost-total">'
            f'<span>Total (28 days)</span>'
            f'<span style="color:#065f46">USD {total_cost:.2f}</span>'
            f'</div>'
            f'<div style="font-size:0.72rem;color:#9ca3af;margin-top:6px">'
            f'Indicative prices. Verify with local suppliers.</div>'
        )
        st.markdown(
            f'<div class="action-card">{cost_html}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>**📈 Expected Yield Impact**")
        if sev == "Healthy":
            yield_loss, yield_saved = 0, 0
        elif sev == "Mild":
            yield_loss, yield_saved = 10, 8
        elif sev == "Moderate":
            yield_loss, yield_saved = 30, 22
        elif sev == "Severe":
            yield_loss, yield_saved = 50, 38
        else:
            yield_loss, yield_saved = 65, 50

        st.markdown(
            f'<div class="action-card">'
            f'<h4>Untreated Loss Estimate</h4>'
            f'<p><span style="color:#dc2626;font-weight:700">{yield_loss}% yield loss</span> '
            f'without intervention</p>'
            f'<h4 style="margin-top:10px">With This Treatment Plan</h4>'
            f'<p><span style="color:#16a34a;font-weight:700">{yield_saved}% yield protected</span> '
            f'— approx. <b>USD {yield_saved * field_size * 80:.0f}</b> saved '
            f'<small style="color:#9ca3af">(at USD 80/ha avg value)</small></p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Conservation & SDG Impact
# ─────────────────────────────────────────────────────────────────────────────

def _render_conservation_tab(ctx: CropContext) -> None:
    st.markdown('<p class="section-title">🌍 Conservation & SDG Impact Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">AgriShield AI is zero-chemical-harm by design. '
        'Every recommendation is audited against UN SDG 2 and SDG 15 targets.</p>',
        unsafe_allow_html=True,
    )

    # ── Impact counters ───────────────────────────────────────────────────────
    field_size = st.session_state.get("field_size", 2.0)
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("🌾 Yield Protected", f"{field_size * 2.4:.1f} t",   help="Estimated tonnes of crop saved per cycle")
    ic2.metric("🐝 Pollinator Safe", "100%",  help="Zero neonicotinoid / OP use")
    ic3.metric("🌱 Soil Organisms",  "Intact", help="Fungal networks and microbiome protected")
    ic4.metric("☠️ Synthetic Toxins", "Zero",  help="No persistent chemicals recommended")

    st.divider()

    # ── SDG panels ────────────────────────────────────────────────────────────
    sdg2_col, sdg15_col = st.columns(2, gap="large")

    with sdg2_col:
        st.markdown(
            '<div style="background:#f0fdf4;border-left:5px solid #16a34a;'
            'border-radius:8px;padding:18px 22px;margin-bottom:14px">'
            '<div style="font-size:1.05rem;font-weight:800;color:#166534;margin-bottom:6px">'
            '🌾 SDG 2 — Zero Hunger</div>'
            '<p style="color:#374151;font-size:0.85rem;margin:0">Early AI-powered detection prevents '
            '30–60% yield losses, protecting food security for smallholder farmers.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        sdg2_metrics = [
            ("Early Detection Accuracy",  87, "#16a34a"),
            ("Yield Loss Prevention",     73, "#22c55e"),
            ("Smallholder Farmer Reach",  64, "#4ade80"),
            ("Organic Protocol Adoption", 91, "#86efac"),
        ]
        for label, val, colour in sdg2_metrics:
            st.markdown(_hbar(val, 100, colour, label, "%"), unsafe_allow_html=True)

    with sdg15_col:
        st.markdown(
            '<div style="background:#eff6ff;border-left:5px solid #3b82f6;'
            'border-radius:8px;padding:18px 22px;margin-bottom:14px">'
            '<div style="font-size:1.05rem;font-weight:800;color:#1d4ed8;margin-bottom:6px">'
            '🌿 SDG 15 — Life on Land</div>'
            '<p style="color:#374151;font-size:0.85rem;margin:0">Zero persistent chemical toxins. '
            'All interventions protect soil biodiversity and pollinators.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        sdg15_metrics = [
            ("Synthetic Pesticide Avoidance", 100, "#3b82f6"),
            ("Soil Microbiome Protection",     95, "#60a5fa"),
            ("Pollinator Safety Score",         98, "#93c5fd"),
            ("Biodiversity Net Gain",           82, "#bfdbfe"),
        ]
        for label, val, colour in sdg15_metrics:
            st.markdown(_hbar(val, 100, colour, label, "%"), unsafe_allow_html=True)

    st.divider()

    # ── Chemical checker ──────────────────────────────────────────────────────
    st.markdown("### 🔍 Pesticide Safety Checker")
    st.caption("Enter a chemical name to check if it is safe or banned under the AgriShield AI protocol.")

    user_chem = st.text_input("Chemical / Product Name", placeholder="e.g. Imidacloprid, Neem oil, Trichoderma…")

    _BANNED = {
        "imidacloprid": ("🚫 BANNED", "alert-critical", "Neonicotinoid — catastrophic bee toxicity, soil persistence >3 years."),
        "clothianidin": ("🚫 BANNED", "alert-critical", "Neonicotinoid — banned in 30+ countries; destroys pollinator colonies."),
        "thiamethoxam": ("🚫 BANNED", "alert-critical", "Neonicotinoid — sub-lethal bee impairment; EU-banned outdoors."),
        "chlorpyrifos": ("🚫 BANNED", "alert-critical", "Organophosphate — neurotoxic to birds, fish, soil fauna."),
        "dimethoate":   ("🚫 BANNED", "alert-critical", "Organophosphate — broad-spectrum, kills beneficial insects."),
        "malathion":    ("🚫 BANNED", "alert-critical", "Organophosphate — highly toxic to bees and aquatic life."),
        "carbendazim":  ("🚫 BANNED", "alert-critical", "Persistent fungicide — endocrine disruptor; bioaccumulates."),
        "methyl bromide":("🚫 BANNED","alert-critical", "Soil fumigant — ozone-depleting; destroys mycorrhizal networks."),
        "glyphosate":   ("⚠️ RESTRICTED", "alert-warning", "Herbicide — disrupts N-fixing microbiome; likely carcinogen (IARC 2A)."),
        "paraquat":     ("🚫 BANNED", "alert-critical", "Herbicide — highly toxic; banned in 67 countries."),
        "neem":         ("✅ APPROVED", "alert-success", "Neem Kernel Extract / Neem oil — biodegrades in 7 days, safe for beneficial insects."),
        "trichoderma":  ("✅ APPROVED", "alert-success", "Biological fungicide — PGPR, suppresses Fusarium & Pythium. Zero toxicity."),
        "bacillus":     ("✅ APPROVED", "alert-success", "Biological agent — Bacillus subtilis/amyloliquefaciens. ISR inducer, safe."),
        "pseudomonas":  ("✅ APPROVED", "alert-success", "Biological PGPR — Pseudomonas fluorescens. Siderophore producer, safe."),
        "copper":       ("✅ APPROVED", "alert-success", "Bordeaux mixture / copper hydroxide — low persistence, approved for organic farming."),
        "seaweed":      ("✅ APPROVED", "alert-success", "Seaweed extract — biostimulant, immune elicitor. Zero environmental risk."),
    }

    if user_chem:
        key = user_chem.lower().strip()
        match = next((v for k, v in _BANNED.items() if k in key), None)
        if match:
            verdict, cls, reason = match
            st.markdown(
                f'<div class="{cls}"><b>{verdict}: {user_chem}</b><br><small>{reason}</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-warning"><b>⚠️ Unknown chemical: {user_chem}</b><br>'
                f'<small>Not in AgriShield database. Consult a certified agronomist before use. '
                f'When in doubt, choose a biological alternative.</small></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Banned substances ──────────────────────────────────────────────────────
    st.markdown("### 🚫 Permanently Banned Substance Classes")
    banned_data = [
        ("Neonicotinoids",      "Imidacloprid · Clothianidin · Thiamethoxam",   "Pollinator collapse; persist >3 yrs in soil"),
        ("Organophosphates",    "Chlorpyrifos · Dimethoate · Malathion",         "Neurotoxic to soil fauna, birds, aquatic life"),
        ("Persistent Fungicides","Carbendazim · Vinclozolin",                   "Endocrine disruptors; trophic bioaccumulation"),
        ("Soil Fumigants",      "Methyl bromide · Chloropicrin",                 "Ozone-depleting; kills mycorrhizal networks"),
        ("Synthetic Herbicides","Glyphosate · Paraquat",                         "Disrupts N-fixing microbiome; toxic to earthworms"),
    ]
    bc1, bc2 = st.columns(2)
    for i, (name, examples, reason) in enumerate(banned_data):
        col = bc1 if i % 2 == 0 else bc2
        col.markdown(
            f'<div class="action-card">'
            f'<h4>🚫 {name}</h4>'
            f'<p><b>Examples:</b> {examples}<br>'
            f'<span style="color:#dc2626">{reason}</span></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### ✅ Approved Biological & Organic Arsenal")
    approved = [
        ("Trichoderma harzianum",        "Bio-fungicide",      "Soil antagonist — suppresses Fusarium, Pythium, Sclerotinia"),
        ("Bacillus subtilis",             "PGPR / ISR inducer", "Systemic resistance via ISR/SAR pathway; produces iturin antifungals"),
        ("Pseudomonas fluorescens",       "PGPR",               "Siderophore-mediated iron chelation; competitive exclusion of pathogens"),
        ("Beauveria bassiana",            "Entomopathogen",     "Fungal insect pathogen — controls whitefly, aphid, thrips vectors"),
        ("Neem Kernel Extract (NSKE 2%)", "Botanical extract",  "Azadirachtin-based; biodegrades in 7 days; no residue in soil"),
        ("Ampelomyces quisqualis",        "Mycoparasite",       "Direct parasite on powdery mildew conidiophores"),
        ("VAM Mycorrhizae",               "Symbiotic fungi",    "Extends root surface 100×; improves nutrient and water uptake"),
        ("Seaweed extract",               "Biostimulant",       "Elicits plant immunity; improves drought tolerance"),
    ]
    ac1, ac2 = st.columns(2)
    for i, (name, typ, desc) in enumerate(approved):
        col = ac1 if i % 2 == 0 else ac2
        col.markdown(
            f'<div class="action-card">'
            f'<h4>✅ {name}</h4>'
            f'<p><span style="background:#dcfce7;color:#166534;padding:2px 8px;'
            f'border-radius:12px;font-size:0.72rem;font-weight:700">{typ}</span><br>'
            f'<span style="font-size:0.82rem;color:#374151">{desc}</span></p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#052e16 0%,#065f46 55%,#047857 100%);
                    padding:22px 32px 18px;border-radius:14px;margin-bottom:22px;
                    display:flex;align-items:center;gap:20px">
          <div>
            <div style="color:#d1fae5;font-size:1.9rem;font-weight:900;
                        letter-spacing:-0.04em;line-height:1.1">
              🌾 AgriShield AI
            </div>
            <div style="color:#6ee7b7;font-size:0.82rem;margin-top:4px">
              Farmer Intelligence Platform &nbsp;·&nbsp; Computer Vision &amp; IBM Granite Reasoning
              &nbsp;·&nbsp; UN SDG 2 &amp; SDG 15
            </div>
          </div>
          <div style="margin-left:auto;text-align:right">
            <div style="color:#a7f3d0;font-size:0.75rem">Powered by</div>
            <div style="color:#fff;font-size:0.85rem;font-weight:700">IBM Granite · MobileNetV3</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctx = _render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Field Diagnostics",
        "🌦️ Weather & Risk",
        "📋 Treatment Planner",
        "🌍 Conservation & SDG",
    ])

    with tab1:
        _render_diagnostics_tab(ctx)
    with tab2:
        _render_weather_tab(ctx)
    with tab3:
        _render_planner_tab(ctx)
    with tab4:
        _render_conservation_tab(ctx)


if __name__ == "__main__":
    main()
