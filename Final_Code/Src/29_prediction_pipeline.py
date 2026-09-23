from __future__ import annotations
import ast
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
# CONFIGURATION AND CONSTANTS
MODELS_DIR = Path("Models")
MODEL_PATH = MODELS_DIR / "official_final_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
SCALER_PATH = MODELS_DIR / "official_final_scaler.joblib"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
STRUCTURAL_FEATURE_NAMES = [
    "has_url",
    "url_count",
    "url_length_max",
    "url_length_avg",
    "has_ip_url",
    "uses_https",
    "max_subdomain_count",
    "is_shortened_url",
    "sender_url_domain_mismatch",
    "has_attachment",
    "attachment_count",
    "has_sender_email",
    "sender_domain_length",
    "sender_domain_digit_count",
    "subject_length",
    "body_length",
    "body_word_count",
    "exclamation_count",
    "question_mark_count",
    "has_html_tag",
]
PSYCH_FEATURE_NAMES = [
    "authority",
    "scarcity_urgency",
    "commitment_consistency",
    "liking_rapport",
    "fear",
    "greed",
    "sensitive_information_request",
    "call_to_action",
]
ALL_FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES + PSYCH_FEATURE_NAMES
SHORTENED_URL_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "is.gd",
    "t.co",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "cutt.ly",
    "rb.gy",
    "rebrand.ly",
    "shorturl.at",
    "tiny.cc",
    "lnkd.in",
}
# TEXT CLEANING
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub("http\\S+|www\\.\\S+", " ", text)
    text = re.sub("\\S+@\\S+", " ", text)
    text = re.sub("<.*?>", " ", text)
    text = re.sub("[^a-zA-Z\\s]", " ", text) # Remove non-alphabetic characters
    text = re.sub("\\s+", " ", text).strip() # Remove extra whitespace
    return text
# STRUCTURAL FEATURE EXTRACTION
def extract_sender_email(sender: str) -> str:
    if not sender:
        return ""
    match = re.search("[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", str(sender))
    return match.group(0).lower() if match else ""


def extract_sender_domain(sender_email: str) -> str:
    if not sender_email or "@" not in sender_email:
        return ""
    return sender_email.split("@", 1)[1].lower().strip(".")


def extract_url_hostname(url: str) -> str:
    try:
        hostname = urlparse(str(url)).hostname
        return hostname.lower().strip(".") if hostname else ""
    except (TypeError, ValueError):
        return ""


def is_ipv4_address(hostname: str) -> int:
    if not hostname or not re.match("^(?:\\d{1,3}\\.){3}\\d{1,3}$", hostname):
        return 0
    try:
        sections = [int(s) for s in hostname.split(".")]
        return int(len(sections) == 4 and all((0 <= s <= 255 for s in sections)))
    except ValueError:
        return 0


def count_subdomains(url: str) -> int:
    hostname = extract_url_hostname(url)
    if not hostname or is_ipv4_address(hostname):
        return 0
    return max(len(hostname.split(".")) - 2, 0) # Subtract 2 for the main domain and TLD


def is_shortened_url(url: str) -> int:
    hostname = extract_url_hostname(url)
    if not hostname:
        return 0
    return int(any((hostname == d or hostname.endswith("." + d) for d in SHORTENED_URL_DOMAINS)))


def domains_match(sender_domain: str, url_hostname: str) -> bool:
    if not sender_domain or not url_hostname:
        return False
    sender_domain = sender_domain.lower().strip(".")
    url_hostname = url_hostname.lower().strip(".")
    return (
        sender_domain == url_hostname
        or url_hostname.endswith("." + sender_domain)
        or sender_domain.endswith("." + url_hostname)
    )


def calculate_sender_url_mismatch(sender_domain: str, urls: list[str]) -> int:
    if not sender_domain or not urls:
        return 0
    hostnames = [h for h in (extract_url_hostname(u) for u in urls) if h]
    if not hostnames:
        return 0
    return int(not any((domains_match(sender_domain, h) for h in hostnames)))

# Combines all the small helpers above into the final 20-feature
def extract_structural_features(
    subject: str,
    body: str,
    sender: str,
    urls: list[str],
    attachments: list[str],
    attachment_bytes: list[tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    subject = subject or "" #to avoid NoneType errors or crashes
    body = body or ""
    urls = urls or []
    attachments = attachments or []
    attachment_bytes = attachment_bytes or []

    sender_email = extract_sender_email(sender)
    sender_domain = extract_sender_domain(sender_email)
    url_lengths = [len(str(u)) for u in urls]
    uses_https = int(any((str(u).lower().startswith("https://") for u in urls)))
    has_ip_url = int(any((is_ipv4_address(extract_url_hostname(u)) for u in urls)))
    max_subdomain_count = max([count_subdomains(u) for u in urls], default=0)
    shortened = int(any((is_shortened_url(u) for u in urls)))
    return {
        "has_url": int(len(urls) > 0),
        "url_count": len(urls),
        "url_length_max": max(url_lengths, default=0),
        "url_length_avg": sum(url_lengths) / len(url_lengths) if url_lengths else 0.0,
        "has_ip_url": has_ip_url,
        "uses_https": uses_https,
        "max_subdomain_count": max_subdomain_count,
        "is_shortened_url": shortened,
        "sender_url_domain_mismatch": calculate_sender_url_mismatch(sender_domain, urls),
        "has_attachment": int(len(attachments) > 0),
        "attachment_count": len(attachments),
        "has_sender_email": int(bool(sender_email)),
        "sender_domain_length": len(sender_domain),
        "sender_domain_digit_count": sum((c.isdigit() for c in sender_domain)),
        "subject_length": len(subject),
        "body_length": len(body),
        "body_word_count": len(body.split()),
        "exclamation_count": body.count("!"),
        "question_mark_count": body.count("?"),
        "has_html_tag": int(bool(re.search("<[^>]+>", body))),
    }

# PSYCHOLOGICAL FEATURE EXTRACTION (GPT-4o-mini)
PSYCH_SYSTEM_PROMPT = '\nYou are a psychological feature extractor for academic phishing email\nresearch.\n\nYour task is to analyse the meaning, context, persuasive intent, and\ncommunicative purpose of an email.\n\nExtract only the following eight psychological and persuasive constructs:\n\n1. Authority\n2. Scarcity/Urgency\n3. Commitment and Consistency\n4. Liking/Rapport\n5. Fear\n6. Greed\n7. Sensitive Information Request\n8. Call-to-Action\n\nDo not classify the email as phishing, spam, legitimate, malicious, or safe.\nAnalyse the email semantically. Do not rely on predefined keywords, exact\nphrases, fixed word lists, or simple word matching.\n\nFor each construct, assign a continuous score between 0.00 and 1.00, where\n0.00 means absent and 1.00 means extremely strong and explicit. Provide\nshort evidence copied word-for-word from the email body. If the construct\nis absent, return a score of 0.00 and an empty evidence list.\n\nIn particular, distinguish Call-to-Action (the recipient is directed to\nperform an action) from Sensitive Information Request (the message\nexplicitly asks the recipient to disclose specific sensitive data such as\na password, code, or account number, in a reply or via a linked form). A\ngeneric "log in" or "verify your account" link is Call-to-Action only,\nunless the email explicitly names the sensitive data being requested.\n\nSubject-line wording alone is not sufficient evidence for a construct;\nevidence must come from the email body.\n\nReturn only valid JSON, with exactly these eight keys, each containing a\n"score" (float) and "evidence" (list of strings). Do not include Markdown\nor any text outside the JSON object.\n'

# ══════════════════════════════════════════════════════════════════
# SECTION 2: PSYCHOLOGICAL FEATURE EXTRACTION (LLM-BASED)

def extract_psychological_features(
    subject: str, body: str, api_key: str | None = None
) -> dict[str, float]:
    load_dotenv() 
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it in your .env file.")
    client = OpenAI(api_key=api_key)
    user_prompt = (
        f"Email subject:\n{subject}\n\nEmail body:\n{body}\n\nReturn the JSON object as specified."
    )
    response = client.responses.create(
        model="gpt-4o-mini", instructions=PSYCH_SYSTEM_PROMPT, input=user_prompt, temperature=0
    )
    parsed = json.loads(response.output_text.strip())

    # --- Normalise GPT's returned keys to our expected snake_case names ---
    def _normalise_key(key: str) -> str:
        key = key.lower()
        key = key.replace("/", "_").replace(" and ", "_").replace("-", "_").replace(" ", "_")
        return key
#verification: make sure all evidence items are actually present in the email body (after normalising whitespace and case)
    normalised_parsed = {_normalise_key(k): v for k, v in parsed.items()}
    parsed = normalised_parsed
    scores = {}
    normalised_body = " ".join(body.lower().split())
    for feat in PSYCH_FEATURE_NAMES:
        entry = parsed.get(feat, {"score": 0.0, "evidence": []})
        score = float(entry.get("score", 0.0))
        evidence = entry.get("evidence", [])
        verified = [e for e in evidence if " ".join(str(e).lower().split()) in normalised_body]
        
        if (
            feat in ("sensitive_information_request", "call_to_action")
            and evidence
            and (not verified)
        ):
            score = min(score, 0.2)
        scores[feat] = round(score, 3)
    return scores
# MAIN PREDICTION PIPELINE CLASS
# SECTION 3: THE PREDICTION PIPELINE (loads model, runs inference)
class PhishingPredictionPipeline:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self._embedding_model = None #lazy load the embedding model only when needed
        self._shap_explainer = None
        self._lime_explainer = None

    @property #lazy load the embedding model only when needed
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedding_model

    def predict(
        self,
        subject: str,
        body: str,
        sender: str = "",
        urls: list[str] | None = None,
        attachments: list[str] | None = None,
        attachment_bytes: list[tuple[str, bytes]] | None = None,
        openai_api_key: str | None = None,
    ) -> dict[str, Any]:
        urls = urls or []
        attachments = attachments or [] #to avoid NoneType errors or crashes
        structural = extract_structural_features(
            subject, body, sender, urls, attachments, attachment_bytes
        )
        psychological = extract_psychological_features(subject, body, api_key=openai_api_key)
        combined_text = f"{subject or ''} {body or ''}".strip()
        cleaned_text = clean_text(combined_text)
        embedding = self.embedding_model.encode([cleaned_text], convert_to_numpy=True) #convert the text to a 384-dimensional embedding vector
        feature_row = pd.DataFrame([{**structural, **psychological}])[ALL_FEATURE_NAMES] #ensure correct column order 28 features
        scaled_features = self.scaler.transform(feature_row)
        X = np.hstack([embedding, scaled_features]) #combine embeddings and scaled structural/psychological features
        proba = self.model.predict_proba(X)[0] #get the predicted probabilities for each class
        predicted_idx = int(np.argmax(proba)) #get the index of the class with the highest probability
        predicted_label = self.label_encoder.classes_[predicted_idx]
        predicted_score = float(proba[predicted_idx])
        return {
            "predicted_label": predicted_label,
            "predicted_probability": predicted_score,
            "class_probabilities": dict(zip(self.label_encoder.classes_, proba.tolist())), #rerurn a dictionary of class probabilities for all classes
            "structural_features": structural, #return the extracted structural features to be used in SHAP/LIME explainability and dashboard display
            "psychological_features": psychological,
            "feature_vector": X,
            "subject": subject,
            "body": body,
            "scaled_structural_features": scaled_features,
        }

# EXPLAINABILITY HELPERS (SHAP / LIME)
FEATURE_PHRASES = {
    "authority": (
        "invokes an authoritative or official-sounding source",
        "does not invoke any particular authority",
    ),
    "scarcity_urgency": (
        "creates time pressure or urgency to act quickly",
        "does not create any noticeable urgency",
    ),
    "commitment_consistency": (
        "appeals to a previous action, agreement, or ongoing relationship",
        "does not reference any prior action or agreement",
    ),
    "liking_rapport": (
        "uses a friendly, personal, or flattering tone to build trust",
        "does not use a notably friendly or personal tone",
    ),
    "fear": (
        "uses language designed to create fear, threat, or anxiety",
        "does not use fear-based or threatening language",
    ),
    "greed": (
        "offers a reward, prize, or financial gain",
        "does not offer any reward or financial gain",
    ),
    "sensitive_information_request": (
        "asks you to provide sensitive information such as a password or code",
        "does not ask you to provide sensitive information",
    ),
    "call_to_action": (
        "directs you to click a link, open an attachment, or reply",
        "does not clearly direct you to take any action",
    ),
    "sender_url_domain_mismatch": (
        "contains a link that does not match the sender's own domain",
        "contains links that match the sender's own domain",
    ),
    "has_ip_url": (
        "contains a link pointing directly to a numeric IP address",
        "does not contain any links pointing to a raw IP address",
    ),
    "is_shortened_url": (
        "contains a shortened link, which can hide the true destination",
        "does not contain any shortened links",
    ),
    "uses_https": (
        "uses a secure HTTPS connection for its link(s)",
        "does not use a secure HTTPS connection for its link(s)",
    ),
    "max_subdomain_count": (
        "contains a link with an unusually large number of subdomains",
        "does not contain links with an unusual number of subdomains",
    ),
    "has_url": ("contains one or more links", "does not contain any links"),
    "url_count": ("contains a notable number of links", "contains very few or no links"),
    "url_length_max": (
        "contains an unusually long link",
        "does not contain any unusually long links",
    ),
    "url_length_avg": (
        "contains links of a notable average length",
        "contains links of typical length",
    ),
    "has_attachment": ("contains an attachment", "does not contain any attachment"),
    "attachment_count": (
        "contains a notable number of attachments",
        "contains very few or no attachments",
    ),
    "has_sender_email": (
        "was sent from a detectable sender email address",
        "did not include a clearly detectable sender email address",
    ),
    "sender_domain_length": (
        "was sent from a domain of a notable length",
        "was sent from a domain of typical length",
    ),
    "sender_domain_digit_count": (
        "was sent from a domain containing numeric digits",
        "was sent from a domain without unusual digits",
    ),
    "subject_length": (
        "has a subject line of a notable length",
        "has a subject line of typical length",
    ),
    "body_length": ("has a body of a notable overall length", "has a body of typical length"),
    "body_word_count": (
        "has a body containing a notable number of words",
        "has a body of typical length",
    ),
    "exclamation_count": (
        "uses a notable number of exclamation marks",
        "does not use unusual punctuation",
    ),
    "question_mark_count": (
        "uses a notable number of question marks",
        "does not use an unusual number of question marks",
    ),
    "has_html_tag": ("contains HTML formatting in its body", "does not contain HTML formatting"),
}
FEATURE_DESCRIPTIONS = {feat: phrases[0] for feat, phrases in FEATURE_PHRASES.items()}
SHOW_ALL_FEATURES = True
ADVICE_RULES = [
    (
        "authority",
        "Verify the sender's identity independently (e.g. by calling the organisation using a number from their official website), rather than trusting the email alone.",
    ),
    (
        "scarcity_urgency",
        "Be cautious of messages that pressure you to act immediately — legitimate organisations rarely demand instant action.",
    ),
    (
        "commitment_consistency",
        "Be cautious of messages that reference a prior action or agreement you don't clearly remember — attackers sometimes fabricate this to seem legitimate.",
    ),
    (
        "liking_rapport",
        "A friendly or personal tone does not guarantee legitimacy — judge the email on its actual content and requests, not its tone.",
    ),
    (
        "fear",
        "Do not let alarming language rush your decision. Pause and verify the claim independently before responding.",
    ),
    (
        "greed",
        "Be skeptical of unsolicited offers of money, prizes, or rewards — if it seems too good to be true, it usually is.",
    ),
    (
        "sensitive_information_request",
        "Never enter your password, PIN, or verification code through a link in an email. Go to the official website or app directly instead.",
    ),
    (
        "call_to_action",
        "Before clicking any link, opening any attachment, or replying, pause and confirm the request is genuine through another channel.",
    ),
    (
        "has_url",
        "Check any links carefully before clicking, even if the email otherwise looks legitimate.",
    ),
    (
        "url_count",
        "Multiple links increase the chance one of them is malicious — check each one individually rather than assuming they are all safe.",
    ),
    (
        "url_length_max",
        "Very long links can be used to disguise the true destination — check where a link actually leads before clicking.",
    ),
    (
        "url_length_avg",
        "Review the links in this email carefully; unusually long or complex links can be used to obscure their real destination.",
    ),
    (
        "has_ip_url",
        "Be wary of links pointing to raw IP addresses instead of normal website names — this is rarely used by legitimate senders.",
    ),
    (
        "uses_https",
        "The presence of HTTPS alone does not guarantee a site is legitimate — attackers can also obtain HTTPS certificates.",
    ),
    (
        "max_subdomain_count",
        "Check the full domain name carefully — attackers sometimes bury the real destination behind many subdomains.",
    ),
    (
        "is_shortened_url",
        "Expand shortened links (using a link-preview tool) before clicking, so you can see the true destination first.",
    ),
    (
        "sender_url_domain_mismatch",
        "Hover over any links before clicking to check where they actually lead, and compare the link's domain to the sender's domain.",
    ),
    ("has_attachment", "Only open attachments you were expecting, from senders you can verify."),
    (
        "attachment_count",
        "Be cautious with emails containing multiple attachments from an unfamiliar or unverified sender.",
    ),
    (
        "has_sender_email",
        "If no clear sender address is shown, treat the email with extra caution and try to verify its origin.",
    ),
    (
        "sender_domain_length",
        "Review the sender's domain name carefully for anything unusual or unfamiliar.",
    ),
    (
        "sender_domain_digit_count",
        "Domains containing unusual numbers or digits can sometimes mimic a legitimate brand name — check it carefully.",
    ),
    (
        "subject_length",
        "Read the subject line critically — very short or very long subject lines can sometimes signal automated or mass-sent phishing content.",
    ),
    (
        "body_length",
        "Consider whether the length and level of detail in the message matches what you'd expect from the claimed sender.",
    ),
    (
        "body_word_count",
        "Consider whether the amount of content in the email matches what you'd expect from a genuine message of this type.",
    ),
    (
        "exclamation_count",
        "Excessive punctuation or exclamation marks can be a sign of manipulative or low-quality mass-sent email.",
    ),
    (
        "question_mark_count",
        "Consider whether the questions posed in the email are genuine, or designed to prompt an anxious, quick reply.",
    ),
    (
        "has_html_tag",
        "Be aware that HTML formatting can be used to disguise the true destination of a link (the visible text may not match where it actually leads).",
    ),
]
FEATURE_SHORT_LABELS = { #used in the ADVANCED technical tables
    "authority": "Authority language",
    "scarcity_urgency": "Urgency/Scarcity",
    "commitment_consistency": "Prior commitment",
    "liking_rapport": "Friendly tone",
    "fear": "Fear language",
    "greed": "Reward/Gain offer",
    "sensitive_information_request": "Sensitive info request",
    "call_to_action": "Call to action",
    "sender_url_domain_mismatch": "Sender/link domain mismatch",
    "has_ip_url": "IP-address link",
    "is_shortened_url": "Shortened link",
    "uses_https": "Uses HTTPS",
    "max_subdomain_count": "Subdomain count",
    "has_url": "Contains link(s)",
    "url_count": "Number of links",
    "url_length_max": "Longest link length",
    "url_length_avg": "Average link length",
    "has_attachment": "Has attachment",
    "attachment_count": "Number of attachments",
    "has_sender_email": "Sender address detected",
    "sender_domain_length": "Sender domain length",
    "sender_domain_digit_count": "Digits in sender domain",
    "subject_length": "Subject length",
    "body_length": "Body length",
    "body_word_count": "Body word count",
    "exclamation_count": "Exclamation marks",
    "question_mark_count": "Question marks",
    "has_html_tag": "HTML formatting",
}
DEFAULT_ADVICE = "Review this signal in context with the rest of the email before deciding whether it is a genuine concern."
LIME_NUM_SAMPLES = 300 #number of perturbed samples to generate for LIME explanations
LIME_NUM_FEATURES = 8 #number of top features to return in LIME explanations

# SECTION 4: EXPLAINABILITY -- LIME (word-level, offline analysis only)

def local_lime_for_prediction(
    pipeline: "PhishingPredictionPipeline",
    subject: str,
    body: str,
    fixed_structural_scaled: np.ndarray, #the structral features are fixed.
    predicted_class_idx: int,
) -> list[tuple[str, float]]:
    from lime.lime_text import LimeTextExplainer

    if pipeline._lime_explainer is None:
        pipeline._lime_explainer = LimeTextExplainer(
            class_names=list(pipeline.label_encoder.classes_), random_state=42, bow=True
        )
    combined_text = f"{subject or ''} {body or ''}".strip()
    cleaned_text = clean_text(combined_text)

    def classifier_function(texts):
        cleaned = [clean_text(t) for t in texts]
        embeddings = pipeline.embedding_model.encode(
            cleaned, convert_to_numpy=True, show_progress_bar=False, batch_size=64
        )
        repeated = np.repeat(fixed_structural_scaled, repeats=len(texts), axis=0)
        hybrid = np.hstack([embeddings, repeated])
        return pipeline.model.predict_proba(hybrid)

    explanation = pipeline._lime_explainer.explain_instance(
        cleaned_text,
        classifier_function,
        num_features=LIME_NUM_FEATURES, #8 features to return in the explanation
        num_samples=LIME_NUM_SAMPLES, #300 perturbed samples to generate for LIME explanations
        labels=[predicted_class_idx],
    )
    return explanation.as_list(label=predicted_class_idx)

# SECTION 5: EXPLAINABILITY -- SHAP (feature-level, used live)

def local_shap_all_classes(
    pipeline: "PhishingPredictionPipeline", X: np.ndarray
) -> dict[str, pd.DataFrame]:
    if pipeline._shap_explainer is None:
        import shap

        pipeline._shap_explainer = shap.TreeExplainer(pipeline.model)
    shap_values = pipeline._shap_explainer.shap_values(X)
    n_embed = 384
    result = {}
    for class_idx, class_name in enumerate(pipeline.label_encoder.classes_):
        if isinstance(shap_values, list):
            sv_full = shap_values[class_idx][0]
        else:
            sv_full = shap_values[0, :, class_idx]
        sv_struct = sv_full[n_embed : n_embed + len(ALL_FEATURE_NAMES)]
        df = (
            pd.DataFrame(
                {
                    "Feature": ALL_FEATURE_NAMES,
                    "Contribution": sv_struct,
                    "AbsContribution": np.abs(sv_struct),
                }
            )
            .sort_values("AbsContribution", ascending=False)
            .reset_index(drop=True)
        )
        result[class_name] = df
    return result


def local_shap_for_prediction(
    pipeline: "PhishingPredictionPipeline", X: np.ndarray, predicted_class_idx: int
) -> pd.DataFrame:
    if pipeline._shap_explainer is None:
        import shap

        pipeline._shap_explainer = shap.TreeExplainer(pipeline.model)
    shap_values = pipeline._shap_explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv_full = shap_values[predicted_class_idx][0]
    else:
        sv_full = shap_values[0, :, predicted_class_idx]
    n_embed = 384
    sv_struct = sv_full[n_embed : n_embed + len(ALL_FEATURE_NAMES)]
    return (
        pd.DataFrame(
            {
                "Feature": ALL_FEATURE_NAMES,
                "Contribution": sv_struct,
                "AbsContribution": np.abs(sv_struct),
            }
        )
        .sort_values("AbsContribution", ascending=False)
        .reset_index(drop=True)
    )
# Helper: formats one feature's raw value for the technical detail
# ----------------------------------------------------------------
def describe_feature_value(feature: str, structural: dict, psychological: dict) -> str:
    if feature in psychological:
        return f"(score: {psychological[feature]:.2f})"
    value = structural.get(feature)
    binary_features = {
        "has_url",
        "has_ip_url",
        "uses_https",
        "has_attachment",
            "has_sender_email",
        "has_html_tag",
        "is_shortened_url",
        "sender_url_domain_mismatch",
    }
    if feature in binary_features:
        return "(Yes)" if value else "(No)"
    return f"(value: {value})"

# SECTION 6: PLAIN-LANGUAGE TRANSLATION LAYER
# --- Step 6a: is the feature's raw value actually present (>0)? ---
def _feature_is_present(feat: str, structural: dict, psychological: dict) -> bool:
    if feat in psychological:
        return psychological[feat] > 0
    value = structural.get(feat, 0)
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return bool(value)
# --- Step 6b: turn (feature, present?, shap_positive?) into one sentence.
def _simple_feature_phrase(feat: str, present: bool, shap_positive: bool = True) -> str:
    affirmative, negative = FEATURE_PHRASES.get(
        feat, (feat.replace("_", " "), f"does not show '{feat.replace('_', ' ')}'")
    )
    if present:
        return affirmative
    if shap_positive:
        # The feature is absent, but its absence itself positively contributed to
        # the model's prediction. We can phrase this as a "negative signal" in plain language.
        short_label = FEATURE_SHORT_LABELS.get(feat, feat.replace("_", " "))
        return f"shows an absence of typical '{short_label}' patterns \u2014 a signal that contributed to this result"
    return negative
# SIMPLIFIED DASHBOARD REPORT (MAIN SCREEN)
VERDICT_BY_LABEL = {
    "Valid": {"emoji": "🟢", "headline": "Appears legitimate"},
    "Spam": {"emoji": "🟡", "headline": "Likely spam"},
    "Phishing": {"emoji": "🔴", "headline": "Likely phishing"},
}

SUMMARY_BY_LABEL = {
    "Valid": "No strong signs of phishing were found in this email.",
    "Spam": "This looks like unsolicited or promotional content rather than a targeted phishing attempt.",
    "Phishing": "This message shows signs commonly associated with phishing.",
}

DEFAULT_ADVICE_BY_LABEL = {
    "Valid": "You can proceed normally, but always stay alert to unexpected requests in any email.",
    "Spam": "This is likely spam. It's generally safe to ignore or delete, but avoid clicking links from unfamiliar senders.",
    "Phishing": "Treat this email with caution and verify the sender independently before acting on it.",
}
# SECTION 7: REPORT ASSEMBLY (what the interface actually displays)
def generate_dashboard_report(pipeline: "PhishingPredictionPipeline", prediction: dict) -> dict:
    label = prediction["predicted_label"]
    probability = prediction["predicted_probability"]
    structural = prediction["structural_features"]
    psychological = prediction["psychological_features"]

    verdict = VERDICT_BY_LABEL.get(label, VERDICT_BY_LABEL["Phishing"])

    predicted_idx = list(pipeline.label_encoder.classes_).index(label)
    shap_df = local_shap_for_prediction(pipeline, prediction["feature_vector"], predicted_idx)

    # Top 3 features that genuinely pushed the model TOWARD this specific
    # prediction (positive SHAP contribution). No arbitrary magnitude cutoff:
    # just the sign (did it support this class at all?) and the ranking itself.
    supporting = shap_df[shap_df["Contribution"] > 0].head(3)

    checks = []
    advice_features = []
    for _, row in supporting.iterrows():
        feat = row["Feature"]
        present = _feature_is_present(feat, structural, psychological)
        phrase = _simple_feature_phrase(feat, present, shap_positive=True)
        checks.append({"feature": feat, "detail": f"This email {phrase}."})
        advice_features.append(feat)

    summary_line = SUMMARY_BY_LABEL.get(label, SUMMARY_BY_LABEL["Phishing"])

    advice_lookup = dict(ADVICE_RULES)
    if label == "Phishing":
        advice_lines = [advice_lookup[f] for f in advice_features if f in advice_lookup]
        advice_lines = list(dict.fromkeys(advice_lines))  # de-duplicate, keep order
        advice_line = advice_lines[0] if advice_lines else DEFAULT_ADVICE_BY_LABEL["Phishing"]
    else:
        advice_line = DEFAULT_ADVICE_BY_LABEL.get(label, DEFAULT_ADVICE_BY_LABEL["Phishing"])

    return {
        "emoji": verdict["emoji"],
        "headline": verdict["headline"],
        "label": label,
        "probability": probability,
        "summary_line": summary_line,
        "checks": checks,
        "advice_line": advice_line,
    }

# FULL TECHNICAL REPORT (ADVANCED VIEW)
def generate_plain_language_report(
    pipeline: "PhishingPredictionPipeline", prediction: dict
) -> dict:
    label = prediction["predicted_label"]
    probability = prediction["predicted_probability"]
    structural = prediction["structural_features"]
    psychological = prediction["psychological_features"]
    risk_level = (
        "Phishing"
        if label == "Phishing"
        else (
            "N/A — classified as legitimate"
            if label == "Valid"
            else "N/A — classified as unwanted/spam"
        )
    )
    predicted_idx = list(pipeline.label_encoder.classes_).index(label)
    shap_df = local_shap_for_prediction(pipeline, prediction["feature_vector"], predicted_idx)
    shap_by_class = local_shap_all_classes(pipeline, prediction["feature_vector"])
    reasons_by_class: dict[str, list[str]] = {}
    reasons_by_class_table: dict[str, pd.DataFrame] = {}
    for class_name, class_shap_df in shap_by_class.items():
        supporting = class_shap_df[class_shap_df["Contribution"] > 0.01].head(5)
        class_reasons = []
        table_rows = []
        for _, row in supporting.iterrows():
            feat = row["Feature"]
            present = _feature_is_present(feat, structural, psychological)
            phrase = _simple_feature_phrase(feat, present, shap_positive=True)
            class_reasons.append(f"This email {phrase}.")
            table_rows.append(
                {
                    "Signal": FEATURE_SHORT_LABELS.get(feat, feat.replace("_", " ").title()),
                    "Strength": round(float(row["Contribution"]), 3),
                }
            )
        if not class_reasons:
            class_reasons.append("No strong signal was found in favour of this class.")
        reasons_by_class[class_name] = class_reasons
        reasons_by_class_table[class_name] = (
            pd.DataFrame(table_rows) if table_rows else pd.DataFrame(columns=["Signal", "Strength"])
        )
    try:
        lime_word_weights = local_lime_for_prediction(
            pipeline,
            prediction["subject"],
            prediction["body"],
            prediction["scaled_structural_features"],
            predicted_idx,
        )
    except Exception as error:
        lime_word_weights = []
        print(f"LIME explanation failed (word-level explanation will be omitted): {error}")
    lime_word_table = (
        pd.DataFrame(lime_word_weights, columns=["Word", "Weight"])
        if lime_word_weights
        else pd.DataFrame(columns=["Word", "Weight"])
    )
    if not lime_word_table.empty:
        lime_word_table = lime_word_table.sort_values(
            "Weight", key=abs, ascending=False
        ).reset_index(drop=True)
        lime_word_table["Weight"] = lime_word_table["Weight"].round(3)

    lime_simple_reasons = []
    for _, lime_row in lime_word_table.head(3).iterrows():
        word = lime_row["Word"]
        weight = lime_row["Weight"]
        direction = "supports" if weight > 0 else "goes against"
        lime_simple_reasons.append(
            f'The word "{word}" {direction} the \'{label}\' classification.'
        )

    supporting = shap_df[shap_df["Contribution"] > 0.01].head(3)
    simple_reasons = []
    for _, row in supporting.iterrows():
        feat = row["Feature"]
        present = _feature_is_present(feat, structural, psychological)
        phrase = _simple_feature_phrase(feat, present, shap_positive=True)
        simple_reasons.append(f"This email {phrase}.")
    if not simple_reasons:
        simple_reasons.append(
            "No single strong signal drove this decision — the model based it on a subtle combination of factors."
        )
    advice = []
    if label == "Phishing":
        for _, row in shap_df[shap_df["Contribution"] > 0.01].iterrows():
            feat = row["Feature"]
            advice.append(advice_rules_lookup_get(feat))
        advice = list(dict.fromkeys(advice))[:5]
    if not advice:
        if label == "Phishing":
            advice.append(
                "This email was flagged as Phishing based on a subtle combination of signals. Verify the sender through an independent channel before acting."
            )
        elif label == "Spam":
            advice.append(
                "This message was classified as unwanted/promotional content. It does not appear to pose a direct security risk, but avoid clicking links from unfamiliar senders."
            )
        else:
            advice.append(
                "No high-risk indicators were flagged. As a general precaution, always verify unexpected requests through an independent channel before acting."
            )
    technical_table_rows = []
    for _, row in shap_df.iterrows():
        feat = row["Feature"]
        contribution = row["Contribution"]
        present = _feature_is_present(feat, structural, psychological)
        value_str = describe_feature_value(feat, structural, psychological)
        technical_table_rows.append(
            {
                "Feature": FEATURE_SHORT_LABELS.get(feat, feat.replace("_", " ").title()),
                "Value": value_str.strip("()"),
                "Contribution": round(float(contribution), 3),
            }
        )
    technical_table = pd.DataFrame(technical_table_rows)
    summary = (
        f"This email was classified as **{label}** with a predicted probability of {probability:.0%}"
        + (f" ({risk_level} risk)." if label == "Phishing" else ".")
    )
    return {
        "label": label,
        "risk_level": risk_level,
        "probability": probability,
        "summary": summary,
        "reasons": simple_reasons,
        "advice": advice,
        "technical_table": technical_table,
        "lime_word_table": lime_word_table,
        "lime_simple_reasons": lime_simple_reasons,
        "reasons_by_class": reasons_by_class,
        "reasons_by_class_table": reasons_by_class_table,
        "full_shap_ranking": shap_df,
        "interpretation_note": "These signals are evaluated together, not in isolation. A feature that might seem risky on its own (such as a long link) can still support a 'legitimate' classification once combined with the rest of the email's characteristics — the model looks at the overall pattern, not any single signal alone.",
    }

def advice_rules_lookup_get(feat: str) -> str:
    return dict(ADVICE_RULES).get(feat, DEFAULT_ADVICE)
# MANUAL TEST
if __name__ == "__main__":
    pipeline = PhishingPredictionPipeline()
    result = pipeline.predict(
        subject="Urgent: Verify your account now",
        body="Dear customer, your account has been suspended. Click the link below and enter your password to restore access immediately.",
        sender="security@paypa1-verify.com",
        urls=["http://paypa1-verify.com/login"],
        attachments=[],
    )
    report = generate_plain_language_report(pipeline, result)
    print(report["summary"])
    print("\nWhy this decision was made (top contributing features, by real local SHAP value):")
    for r in report["reasons"]:
        print(f"  - {r}")
    print("\nRecommended actions:")
    for a in report["advice"]:
        print(f"  - {a}")