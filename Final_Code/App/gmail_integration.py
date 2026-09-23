"""
Gmail Integration for PhishLens
=================================
Connects the PhishLens interface to a user's Gmail inbox, fetches recent
emails, and formats them for the existing prediction pipeline
(31_prediction_pipeline.py) without modifying that file.

SETUP (one-time, manual steps you must do yourself before this works):

1. Go to https://console.cloud.google.com/
2. Create a new project (or use an existing one).
3. Enable the "Gmail API" for that project
   (APIs & Services -> Library -> search "Gmail API" -> Enable).
4. Go to APIs & Services -> Credentials -> Create Credentials ->
   OAuth client ID.
   - Application type: Desktop app
   - Give it any name (e.g. "PhishLens")
5. Download the resulting JSON file, rename it to `gmail_credentials.json`,
   and place it in your project's root folder (same folder as app.py).
6. Install the required packages:
       pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client

The first time you run this, a browser window will open asking you to
log into Google and approve access. This creates a `gmail_token.json`
file that caches your login for future runs (delete it to re-authenticate
with a different account).

USAGE (from app.py or standalone):
    from gmail_integration import get_gmail_service, fetch_recent_emails, gmail_message_to_pipeline_input

    service = get_gmail_service()
    emails = fetch_recent_emails(service, max_results=10)
    for email in emails:
        pipeline_input = gmail_message_to_pipeline_input(email)
        result = pipeline.predict(**pipeline_input)
"""

import base64
import os
import re
from email import message_from_bytes
from email.utils import parseaddr

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_PATH = "gmail_credentials.json"
TOKEN_PATH = "gmail_token.json"

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")


def get_gmail_service():
    """
    Authenticates with Gmail (opening a browser window on first run) and
    returns a Gmail API service object. Caches the login token so
    subsequent runs do not require re-authentication.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"'{CREDENTIALS_PATH}' not found. Follow the setup steps at the "
                    "top of this file (Google Cloud Console -> OAuth credentials -> "
                    "download JSON -> rename to gmail_credentials.json)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def fetch_recent_emails(service, max_results: int = 10) -> list[dict]:
    """
    Fetches the N most recent emails from the user's inbox.
    Returns a list of dicts with: id, subject, sender, body, urls,
    attachments (filenames only), attachment_bytes (list of (filename, bytes)).
    """
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["INBOX"]
    ).execute()
    message_refs = results.get("messages", [])

    emails = []
    for ref in message_refs:
        full_message = service.users().messages().get(
            userId="me", id=ref["id"], format="raw"
        ).execute()
        raw_bytes = base64.urlsafe_b64decode(full_message["raw"])
        mime_message = message_from_bytes(raw_bytes)
        emails.append(_parse_mime_message(mime_message))

    return emails


def _parse_mime_message(mime_message) -> dict:
    subject = mime_message.get("Subject", "") or ""
    sender_raw = mime_message.get("From", "") or ""
    _, sender_email = parseaddr(sender_raw)

    body_text = ""
    attachments = []
    attachment_bytes = []

    if mime_message.is_multipart():
        for part in mime_message.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            content_type = part.get_content_type()

            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    payload = part.get_payload(decode=True)
                    attachments.append(filename)
                    if payload:
                        attachment_bytes.append((filename, payload))
                continue

            if content_type == "text/plain" and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body_text = payload.decode(charset, errors="replace")
            elif content_type == "text/html" and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body_text = payload.decode(charset, errors="replace")
    else:
        payload = mime_message.get_payload(decode=True)
        if payload:
            charset = mime_message.get_content_charset() or "utf-8"
            body_text = payload.decode(charset, errors="replace")

    urls = list(dict.fromkeys(URL_PATTERN.findall(body_text)))  # de-duplicated, order-preserved

    return {
        "subject": subject,
        "sender": sender_email or sender_raw,
        "body": body_text,
        "urls": urls,
        "attachments": attachments,
        "attachment_bytes": attachment_bytes,
    }


def gmail_message_to_pipeline_input(email: dict) -> dict:
    """
    Converts a parsed Gmail message (from fetch_recent_emails) into the
    exact keyword arguments expected by
    PhishingPredictionPipeline.predict() in 31_prediction_pipeline.py.
    """
    return {
        "subject": email["subject"],
        "body": email["body"],
        "sender": email["sender"],
        "urls": email["urls"],
        "attachments": email["attachments"],
        "attachment_bytes": email["attachment_bytes"],
    }


if __name__ == "__main__":
    # Quick standalone test: list the 5 most recent inbox emails.
    print("Authenticating with Gmail (a browser window will open)...")
    service = get_gmail_service()
    print("Authenticated. Fetching 5 most recent emails...\n")
    recent = fetch_recent_emails(service, max_results=5)
    for i, email in enumerate(recent, 1):
        print(f"{i}. Subject: {email['subject'][:60]}")
        print(f"   From: {email['sender']}")
        print(f"   URLs found: {len(email['urls'])}")
        print(f"   Attachments: {email['attachments'] or 'none'}")
        print()