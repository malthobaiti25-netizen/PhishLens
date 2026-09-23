"""
Interactive Phishing Detection Interface
==========================================
Run with:  streamlit run App/app.py

Lets a user paste an email (subject, body, sender, links, attachment names)
and returns: predicted class, risk probability, plain-language explanation
of why, and concrete mitigation advice — built on top of the trained
XGBoost hybrid model (semantic + structural + psychological features).
"""

import email
import importlib.util
import re
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from gmail_integration import get_gmail_service, fetch_recent_emails, gmail_message_to_pipeline_input
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False

# Load the pipeline module directly from its file path in Src/, since its
# filename starts with a digit (e.g. "29_prediction_pipeline.py") and is
# not a valid Python import name.
PIPELINE_SCRIPT_NAME = "29_prediction_pipeline.py"  # adjust if your local filename differs
PIPELINE_PATH = Path(__file__).resolve().parent.parent / "Src" / PIPELINE_SCRIPT_NAME

_spec = importlib.util.spec_from_file_location("prediction_pipeline", PIPELINE_PATH)
pipeline_module = importlib.util.module_from_spec(_spec)
sys.modules["prediction_pipeline"] = pipeline_module
_spec.loader.exec_module(pipeline_module)


# ============================================================
# .eml parsing
# ============================================================

URL_PATTERN = re.compile(r"https?://[^\s\"'<>\)\]]+")

# Patterns that commonly mark the START of a quoted/forwarded message within
# an email body (Outlook, Gmail, Apple Mail, generic reply conventions).
QUOTE_MARKER_PATTERNS = [
    r"^_{10,}\s*$",                                   # Outlook separator line "________________________________"
    r"^From:\s.+\n(?:Sent|Date):\s.+\n(?:To|Cc):",     # Outlook-style quoted header block
    r"^On .+ wrote:\s*$",                              # Gmail/Apple Mail style "On [date], [name] wrote:"
    r"^-{2,}\s*Original Message\s*-{2,}\s*$",           # "---- Original Message ----"
    r"^>{1}",                                          # Classic ">" quote prefix (checked per-line separately)
]

COMBINED_QUOTE_PATTERN = re.compile(
    "|".join(f"(?:{p})" for p in QUOTE_MARKER_PATTERNS[:-1]),
    flags=re.IGNORECASE | re.MULTILINE,
)


def strip_quoted_reply(body_text: str) -> tuple[str, bool]:
    """Trim a reply-thread email body down to just the user's own most-recent
    message, cutting off at the first detected quoted/forwarded message marker.
    Returns (trimmed_text, was_trimmed)."""
    match = COMBINED_QUOTE_PATTERN.search(body_text)
    if match:
        return body_text[: match.start()].strip(), True
    return body_text.strip(), False


def parse_eml_file(uploaded_file, strip_quotes: bool = True) -> dict:
    """Parse an uploaded .eml file into subject/body/sender/urls/attachments.

    If strip_quotes is True (default), any quoted/forwarded prior message
    embedded in the body (e.g. an Outlook "From: ... Sent: ... To: ..." block)
    is removed, so only the most recent, actual message is analysed — this
    avoids conflating a short reply with a longer quoted original message.
    """
    raw_bytes = uploaded_file.read()
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    subject = msg.get("Subject", "") or ""
    sender = msg.get("From", "") or ""

    body_text = ""
    attachments = []       # filenames only (kept for display/backward compatibility)
    attachment_bytes = []  # (filename, raw_bytes) pairs — needed for content-based type detection

    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            content_type = part.get_content_type()

            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    attachments.append(filename)
                    raw = part.get_payload(decode=True)
                    attachment_bytes.append((filename, raw or b""))
                continue

            if content_type == "text/plain" and not body_text:
                try:
                    body_text = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)
                    body_text = payload.decode(errors="ignore") if payload else ""

        if not body_text:
            # Fall back to text/html, stripped of tags, if no text/plain part existed
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    try:
                        html = part.get_content()
                    except Exception:
                        payload = part.get_payload(decode=True)
                        html = payload.decode(errors="ignore") if payload else ""
                    body_text = re.sub(r"<[^>]+>", " ", html)
                    break
    else:
        try:
            body_text = msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True)
            body_text = payload.decode(errors="ignore") if payload else str(msg.get_payload())

    was_trimmed = False
    full_body_text = body_text.strip()
    if strip_quotes:
        body_text, was_trimmed = strip_quoted_reply(body_text)
    else:
        body_text = full_body_text

    urls = sorted(set(URL_PATTERN.findall(body_text)))

    return {
        "subject": subject.strip(),
        "body": body_text.strip(),
        "full_body": full_body_text,   # kept for reference/preview even when trimmed
        "was_trimmed": was_trimmed,
        "sender": sender.strip(),
        "urls": urls,
        "attachments": attachments,
        "attachment_bytes": attachment_bytes,
    }


# ============================================================
# Page config
# ============================================================

st.set_page_config(page_title="PhishLens — Phishing Risk Analyser", page_icon="🎣", layout="centered")

st.title("🎣 PhishLens")
st.caption("Hybrid Explainable AI for Phishing Email Detection — paste an email below to analyse it.")


# ============================================================
# Load the pipeline once (cached across interactions)
# ============================================================

@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_pipeline():
    return pipeline_module.PhishingPredictionPipeline()


# ============================================================
# Input: choose between uploading a .eml file or connecting Gmail
# ============================================================

parsed_email = None
submitted = False

input_tabs = st.tabs(["📁 Upload .eml file", "📧 Connect Gmail"])

# --- Tab 1: file upload (original behaviour, unchanged) ---
with input_tabs[0]:
    uploaded_eml = st.file_uploader(
        "Drag and drop or browse for a .eml file (exported from Outlook, Gmail, etc.)",
        type=["eml"],
    )

    strip_quotes = st.checkbox(
        "Analyse only the most recent message (remove any quoted reply/forwarded thread below it)",
        value=True,
        help="If this email is a reply or forward, this removes the older quoted message so only "
             "the newest message is analysed — recommended, since a long quoted thread can distort "
             "the analysis of a short reply.",
        key="strip_quotes_upload",
    )

    if uploaded_eml is not None:
        parsed_email = parse_eml_file(uploaded_eml, strip_quotes=strip_quotes)

        if parsed_email["was_trimmed"]:
            st.info(
                "📎 This email contains a quoted reply/forwarded thread. Only the most recent message "
                "(shown below) will be analysed. Uncheck the box above to analyse the full thread instead."
            )

        with st.expander("Parsed email preview (click to verify before analysing)"):
            st.markdown(f"**Subject:** {parsed_email['subject'] or '(none)'}")
            st.markdown(f"**From:** {parsed_email['sender'] or '(none)'}")
            st.markdown(f"**Links found:** {len(parsed_email['urls'])}")
            st.markdown(f"**Attachments found:** {len(parsed_email['attachments'])}")
            st.text_area("Body that will be analysed", parsed_email["body"][:2000], height=150, disabled=True)
            if parsed_email["was_trimmed"]:
                with st.expander("Show full original text (including the quoted thread)"):
                    st.text_area("Full body", parsed_email["full_body"][:4000], height=200, disabled=True)

        submitted = st.button("🔍 Analyse Email", use_container_width=True, type="primary", key="analyse_upload")
    else:
        st.info("Please upload a .eml file to begin analysis.")

# --- Tab 2: Gmail connection ---
with input_tabs[1]:
    if not GMAIL_AVAILABLE:
        st.error(
            "Gmail integration is not installed. Run: `pip install google-auth-oauthlib "
            "google-auth-httplib2 google-api-python-client`, then restart the app."
        )
    else:
        if "gmail_service" not in st.session_state:
            st.session_state.gmail_service = None
        if "gmail_emails" not in st.session_state:
            st.session_state.gmail_emails = []

        if st.session_state.gmail_service is None:
            st.info(
                "Connecting will open a Google sign-in window. PhishLens only requests "
                "read-only access to your inbox."
            )
            if st.button("Connect Gmail"):
                with st.spinner("Opening Google sign-in..."):
                    try:
                        st.session_state.gmail_service = get_gmail_service()
                        st.success("Connected!")
                        st.rerun()
                    except FileNotFoundError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                n_emails = st.slider("How many recent emails to load?", 5, 30, 10)
            with col2:
                st.write("")
                st.write("")
                if st.button("Refresh inbox"):
                    with st.spinner("Fetching emails..."):
                        st.session_state.gmail_emails = fetch_recent_emails(
                            st.session_state.gmail_service, max_results=n_emails
                        )

            if not st.session_state.gmail_emails:
                with st.spinner("Fetching emails..."):
                    st.session_state.gmail_emails = fetch_recent_emails(
                        st.session_state.gmail_service, max_results=n_emails
                    )

            if st.session_state.gmail_emails:
                options = [
                    f"{e['subject'][:70] or '(no subject)'}  —  {e['sender']}"
                    for e in st.session_state.gmail_emails
                ]
                selected_idx = st.selectbox(
                    "Select an email to analyse:", range(len(options)),
                    format_func=lambda i: options[i],
                )
                gmail_email = st.session_state.gmail_emails[selected_idx]

                # Build a parsed_email dict with the SAME keys the upload path produces
                parsed_email = {
                    "subject": gmail_email["subject"],
                    "sender": gmail_email["sender"],
                    "body": gmail_email["body"],
                    "full_body": gmail_email["body"],
                    "urls": gmail_email["urls"],
                    "attachments": gmail_email["attachments"],
                    "attachment_bytes": gmail_email["attachment_bytes"],
                    "was_trimmed": False,
                }

                with st.expander("Parsed email preview (click to verify before analysing)"):
                    st.markdown(f"**Subject:** {parsed_email['subject'] or '(none)'}")
                    st.markdown(f"**From:** {parsed_email['sender'] or '(none)'}")
                    st.markdown(f"**Links found:** {len(parsed_email['urls'])}")
                    st.markdown(f"**Attachments found:** {len(parsed_email['attachments'])}")
                    st.text_area("Body that will be analysed", parsed_email["body"][:2000], height=150, disabled=True, key="gmail_body_preview")

                submitted = st.button("🔍 Analyse Email", use_container_width=True, type="primary", key="analyse_gmail")


# ============================================================
# Run analysis
# ============================================================

if submitted:
    if not parsed_email or (not parsed_email["subject"] and not parsed_email["body"]):
        st.warning("The uploaded email appears to have no subject or body to analyse.")
        st.stop()

    pipeline = load_pipeline()

    with st.spinner("Analysing email (extracting features and running the model)..."):
        try:
            prediction = pipeline.predict(
                subject=parsed_email["subject"],
                body=parsed_email["body"],
                sender=parsed_email["sender"],
                urls=parsed_email["urls"],
                attachments=parsed_email["attachments"],
                attachment_bytes=parsed_email["attachment_bytes"],
            )
            dashboard = pipeline_module.generate_dashboard_report(pipeline, prediction)
            report = pipeline_module.generate_plain_language_report(pipeline, prediction)
        except Exception as e:
            st.error(f"Something went wrong during analysis: {e}")
            st.stop()

    st.markdown("---")

    # --- Compact headline: "🔴 Likely phishing · 94% probability" ---
    EMOJI_COLORS = {"🟢": "#2ecc71", "🟡": "#f39c12", "🔴": "#e74c3c"}
    accent_color = EMOJI_COLORS.get(dashboard["emoji"], "#999999")

    st.markdown(
        f"<h2 style='margin-bottom:2px;'>{dashboard['emoji']} "
        f"<span style='color:{accent_color};'>{dashboard['headline']}</span>"
        f"<span style='color:#888; font-weight:400; font-size:1.1rem;'> · {dashboard['probability']:.0%} probability</span></h2>",
        unsafe_allow_html=True,
    )
    st.caption("ⓘ This probability was independently verified to be well-calibrated (see 'Why did the AI decide this').")
    st.markdown(f"<p style='margin-top:8px; font-size:1rem;'>{dashboard['summary_line']}</p>", unsafe_allow_html=True)

    # --- Why? (top 3 features that actually drove this decision) ---
    st.markdown("#### Why?")
    for check in dashboard["checks"]:
        st.markdown(f"<p style='margin-bottom:6px; color:#333;'>• {check['detail']}</p>", unsafe_allow_html=True)

    # --- What should you do? ---
    st.markdown("#### What should you do?")
    st.info(dashboard["advice_line"])

    # --- Collapsible: full technical explanation (SHAP + LIME + tables) ---
    with st.expander("▾ Why did the AI decide this"):
        CLASS_COLORS = {"Phishing": "#e74c3c", "Spam": "#f39c12", "Valid": "#2ecc71"}
        class_probs = prediction["class_probabilities"]

        donut_fig = go.Figure(data=[go.Pie(
            labels=list(class_probs.keys()),
            values=list(class_probs.values()),
            hole=0.65,
            marker=dict(colors=[CLASS_COLORS.get(c, "#999999") for c in class_probs.keys()]),
            textinfo="label+percent",
            sort=False,
        )])
        donut_fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=240,
            annotations=[dict(
                text=f"<b>{report['label']}</b><br>{report['probability']:.0%}",
                x=0.5, y=0.5, font_size=18, showarrow=False,
            )],
        )
        st.plotly_chart(donut_fig, use_container_width=True)

        st.markdown("**What supports each possible classification?**")
        class_cols = st.columns(3)
        for col, class_name in zip(class_cols, ["Phishing", "Spam", "Valid"]):
            with col:
                color = CLASS_COLORS[class_name]
                st.markdown(f"<h5 style='color:{color}; margin-bottom:4px;'>{class_name}</h5>", unsafe_allow_html=True)
                table = report["reasons_by_class_table"].get(class_name)
                if table is not None and not table.empty:
                    st.dataframe(
                        table.style.background_gradient(subset=["Strength"], cmap="Greens", vmin=0, vmax=table["Strength"].max() or 1),
                        hide_index=True, use_container_width=True, height=180,
                    )
                else:
                    st.caption("No strong signal.")

        st.caption(f"ℹ️ {report['interpretation_note']}")

        st.markdown("**Which words influenced this decision?**")
        if not report["lime_word_table"].empty:
            lime_display = report["lime_word_table"].rename(columns={"Weight": "Influence"})
            st.dataframe(
                lime_display.style.bar(subset=["Influence"], align="mid", color=["#e74c3c", "#2ecc71"]),
                hide_index=True, use_container_width=True, height=min(300, 40 + 35 * len(lime_display)),
            )
        else:
            st.caption("Word-level explanation is not available for this email.")

        st.markdown("**Full feature breakdown (all 29 features)**")
        tech_table = report["technical_table"]
        st.dataframe(
            tech_table.style.bar(subset=["Contribution"], align="mid", color=["#e74c3c", "#2ecc71"]),
            hide_index=True, use_container_width=True, height=min(500, 40 + 35 * len(tech_table)),
        )

        st.markdown("**Raw feature values**")
        st.json(prediction["structural_features"])
        st.json(prediction["psychological_features"])

    st.caption(
        "This tool provides an automated risk assessment and does not replace your own judgement. "
        "When in doubt, verify the sender through an independent channel before acting on any email."
    )

st.markdown("---")
st.caption(
    "Developed as part of an MSc Cybersecurity dissertation: "
    "*Developing and Evaluating a Hybrid Explainable AI Framework Integrating Semantic, "
    "Structural, and Psychological Features for Phishing Email Detection and Interactive "
    "User Mitigation.*"
)