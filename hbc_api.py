#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "python-dotenv",
#   "requests",
#   "tqdm",
# ]
# ///

"""
hbc_api.py
----------
Python API client for the NSW Home Building Certificate (HBC Check) API.
Hosted at api.onegov.nsw.gov.au

Wraps three data endpoints:
  - get_access_token()  : authenticate and retrieve a Bearer token
  - browse()            : discover licences via text search
  - details()           : retrieve full licence profile by licenceID

Also includes:
  - verify()            : spot-check a specific licence number (utility only, not for bulk use)
  - HBCClient           : stateful client that manages token expiry automatically

CSV output functions:
  - save_browse_csv(results, path)   : save browse results to CSV
  - save_details_csv(details, path)  : save flattened details to CSV (appends rows)
  - run_and_save(client, terms, out_dir) : convenience — browse + details + save all CSVs

Output files (written to out_dir, default "./output"):
  - licences_raw.csv          : deduplicated browse results
  - licence_details.csv       : one row per licence, scalar fields flattened
  - normalised_classes.csv    : one row per licence class
  - normalised_conditions.csv : one row per condition
  - normalised_premises.csv   : one row per venue/premises
  - normalised_building_sites.csv  : one row per building site
  - normalised_disciplinary.csv    : one row per disciplinary event

Usage:
    from hbc_api import HBCClient, run_and_save

    client = HBCClient(api_key="your_key", api_secret="your_secret")

    # Browse + details + write all CSVs in one call:
    run_and_save(client, search_terms=["smith", "jones", "sydney"], out_dir="./output")

    # Or use the lower-level functions:
    results = client.browse("smith plumbing")
    save_browse_csv(results, "./output/licences_raw.csv")

Environment variables (recommended via .env + python-dotenv):
    API_KEY
    API_SECRET
"""

import base64
import csv
import logging
import os
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://api.onegov.nsw.gov.au"
TOKEN_ENDPOINT = "/oauth/client_credential/accesstoken"
BROWSE_ENDPOINT = "/hbccheckregister/v1/browse"
VERIFY_ENDPOINT = "/hbccheckregister/v1/verify"
DETAILS_ENDPOINT = "/hbccheckregister/v1/details"

# Public register endpoint (no auth required)
PUBLIC_REGISTER_BASE_URL = "https://verify.licence.nsw.gov.au/publicregisterapi/api/v1"
PUBLIC_REGISTER_LICENCE_TYPE = "Home%20Building%20Compensation%20Certificate"

# Minimum delay between public register requests (seconds).
# The endpoint sits behind Cloudflare and will 429 if hammered concurrently.
# Requests are always serialised (one at a time); this delay is applied between each.
PUBLIC_REGISTER_REQUEST_DELAY = 0.5

# Backoff schedule for public register 429s (seconds) — much longer than the
# OneGov schedule because Cloudflare's rate-limit windows are typically 30-60 s.
PUBLIC_REGISTER_429_BACKOFF = [30, 60, 120]


def _browser_headers() -> dict:
    """
    Return a randomised set of browser-like headers for each public register request.
    Varies User-Agent and Accept-Language to avoid Cloudflare fingerprinting.
    """

    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-GB,en-AU;q=0.9,en;q=0.8",
        "Referer": "https://verify.licence.nsw.gov.au/",
        "Origin": "https://verify.licence.nsw.gov.au",
        "User-Agent": "FYP-api",
    }


# Re-auth proactively when fewer than this many seconds remain on the token
TOKEN_EXPIRY_BUFFER_SECONDS = 60

# Retry config for transient server errors
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [2, 4, 8]  # exponential backoff, one entry per retry


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class HBCAuthError(Exception):
    """Raised when authentication fails (401 on token request)."""


class HBCAPIError(Exception):
    """Raised when the API returns a non-recoverable error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


# ---------------------------------------------------------------------------
# Token dataclass (plain dict-style, no external deps)
# ---------------------------------------------------------------------------
class Token:
    """Holds a Bearer token and its computed expiry time."""

    def __init__(self, access_token: str, issued_at_ms: int, expires_in_s: int):
        self.access_token = access_token
        # issued_at from the API is Unix milliseconds (returned as a string)
        self.issued_at = datetime.fromtimestamp(issued_at_ms / 1000, tz=timezone.utc)
        self.expires_at = datetime.fromtimestamp(
            (issued_at_ms / 1000) + expires_in_s, tz=timezone.utc
        )

    @property
    def seconds_remaining(self) -> float:
        delta = self.expires_at - datetime.now(tz=timezone.utc)
        return delta.total_seconds()

    @property
    def is_valid(self) -> bool:
        return self.seconds_remaining > TOKEN_EXPIRY_BUFFER_SECONDS

    def __repr__(self):
        return (
            f"<Token expires_at={self.expires_at.isoformat()} "
            f"remaining={self.seconds_remaining:.0f}s>"
        )


# ---------------------------------------------------------------------------
# Low-level functions (stateless)
# ---------------------------------------------------------------------------


def get_access_token(api_key: str, api_secret: str) -> Token:
    """
    Authenticate with the OneGov API using client credentials.

    Inputs:
        api_key    — OneGov-issued API key
        api_secret — OneGov-issued API secret

    Returns:
        _Token object containing access_token, issued_at, and expires_at.

    Raises:
        HBCAuthError  — on 401 (invalid credentials)
        HBCAPIError   — on other non-200 responses
        requests.RequestException — on network failure
    """
    credentials = f"{api_key}:{api_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()

    url = f"{BASE_URL}{TOKEN_ENDPOINT}"
    headers = {"Authorization": f"Basic {encoded}"}
    params = {"grant_type": "client_credentials"}

    logger.info("Requesting access token from %s", url)

    resp = requests.get(url, headers=headers, params=params, timeout=30)

    if resp.status_code == 401:
        body = safe_json(resp)
        raise HBCAuthError(f"Authentication failed: {body.get('Error', resp.text)}")
    if resp.status_code != 200:
        raise HBCAPIError(resp.status_code, resp.text)

    data = resp.json()

    # issued_at is a Unix millisecond timestamp returned as a string
    issued_at_ms = int(data["issued_at"])
    # expires_in is seconds remaining, also returned as a string
    expires_in_s = int(data["expires_in"])

    token = Token(
        access_token=data["access_token"],
        issued_at_ms=issued_at_ms,
        expires_in_s=expires_in_s,
    )

    logger.info(
        "Token acquired. Expires at %s (%ds remaining)",
        token.expires_at.isoformat(),
        token.seconds_remaining,
    )
    return token


def browse(token: Token, api_key: str, search_text: str) -> list[dict]:
    """
    Search for licences using partial text (name, suburb, business name, etc.).

    This is the discovery step for bulk collection. Note that the endpoint
    is NOT designed for enumeration — it returns an unspecified top-N result
    set with no pagination. Use varied search terms and deduplicate by licenceID
    to maximise coverage.

    Inputs:
        token       — valid _Token object
        api_key     — passed as 'apikey' request header
        search_text — minimum 2 characters; noise words and most punctuation
                      are stripped by the API

    Returns:
        List of dicts, each containing:
            licenceID, licenceNumber, licenceName, licensee, licenceType,
            status, suburb, postcode, businessNames, classes, categories

    Raises:
        HBCAPIError — on 400 (bad search text), 401 (expired token), 500
        requests.RequestException — on network failure

    Limitations:
        - No pagination or result count
        - Fuzzy matching is limited; some records may be missing
        - SIRA warns that results may be incomplete or misattributed
    """
    require_non_empty(search_text, "search_text")
    if len(search_text.strip()) < 2:
        raise ValueError("search_text must be at least 2 characters")

    url = f"{BASE_URL}{BROWSE_ENDPOINT}"
    headers = auth_headers(token, api_key)
    params = {"searchText": search_text}

    logger.debug("browse(search_text=%r)", search_text)

    resp = get_with_retry(
        url, headers=headers, params=params, token=token, api_key=api_key
    )
    return resp.json()


def verify(token: Token, api_key: str, licence_number: str) -> list[dict]:
    """
    Quickly validate a specific licence number and retrieve basic metadata.

    *** NOT for bulk pipeline use. ***
    browse() already returns the same fields (licenceID, status, licenceType,
    businessNames, categories, classes). Use verify() only for targeted
    spot-checks of individual licence numbers (e.g. user-initiated lookups).

    Inputs:
        token          — valid _Token object
        api_key        — passed as 'apikey' request header
        licence_number — must be exact (no fuzzy matching)

    Returns:
        List of dicts, each containing:
            licenceID, licenceNumber, status, startDate, expiryDate,
            refusedDate, licenceType, licenceName, licensee, address,
            vehicleRegistration, businessNames

    Raises:
        HBCAPIError — on 400 (invalid number), 401 (expired token), 500
        requests.RequestException — on network failure

    Limitations:
        - No fuzzy matching — licence number must be exact
        - Some licences may be missing or incorrectly linked
    """
    require_non_empty(licence_number, "licence_number")

    url = f"{BASE_URL}{VERIFY_ENDPOINT}"
    headers = auth_headers(token, api_key)
    params = {"licenceNumber": licence_number}

    logger.debug("verify(licence_number=%r)", licence_number)

    resp = get_with_retry(
        url, headers=headers, params=params, token=token, api_key=api_key
    )
    return resp.json()


def details(token: Token, api_key: str, licence_id: str) -> dict:
    """
    Retrieve the full licence profile for a given licenceID.

    licenceID must come from the licenceID field in browse() or verify() results.

    Inputs:
        token      — valid _Token object
        api_key    — passed as 'apikey' request header
        licence_id — unique licence identifier (e.g. '1-17YIOEU')

    Returns:
        Dict containing the full licence profile, including:
            - licence holder details
            - licence classes and conditions
            - venues (business premises)
            - buildingSites (with councilDANumber, complyingDCNumber,
              constructionCertificateNumber)
            - vehicle registrations
            - historical licence numbers
            - disciplinary history:
                infringementNotices, ctttOrders, publicWarnings,
                publicWarningsCount, cautionReprimandCount
            - insuranceClaimCounts (aggregate counts only — NOT individual records):
                claimCount, claimCountOldScheme,
                insuranceClaimsPaid, statutoryInsuranceClaimsPaid

    Raises:
        HBCAPIError — on 400 (invalid licenceID), 401 (expired token), 500
        requests.RequestException — on network failure

    Limitations (per SIRA's official warnings):
        - Some insurance records may be missing
        - Some records may be attributed to the wrong property
        - Unit/lot numbers may be incorrect (common in new developments)
        - Data may be incomplete due to insurer or SIRA system issues
        - Builders may enter incorrect information
        - Insurance purchased before subdivision may not map to final addresses
        - SIRA does NOT guarantee the policy covers the actual work done
        - Expect null fields, inconsistent formatting, and schema variations
        - insuranceClaimCounts is an aggregate summary — individual claim
          records are not available via this API
    """
    require_non_empty(licence_id, "licence_id")

    url = f"{BASE_URL}{DETAILS_ENDPOINT}"
    headers = auth_headers(token, api_key)
    params = {"licenceid": licence_id}

    logger.debug("details(licence_id=%r)", licence_id)

    resp = get_with_retry(
        url, headers=headers, params=params, token=token, api_key=api_key
    )
    return resp.json()


def public_register_details(licence_id: str) -> dict:
    """
    Retrieve full licence details from the NSW Public Register API.

    This endpoint requires NO authentication — it is publicly accessible.
    It tends to return more accurate or complete data than the OneGov
    details() endpoint for Home Building Compensation Certificates.

    Inputs:
        licence_id — the licenceID field from browse()/verify() results.
                     The OneGov licenceID is already in the pipe-separated
                     format the public register expects, e.g.:
                       "HBCF19046945|HBCF19046945"
                       "HBCF14866998|107-NSWDHIBHWI/189386"
                     Pass it directly — no transformation is performed.

    The endpoint sits behind Cloudflare. To avoid 429 rate-limit blocks:
      - Always call this function serially (one at a time), never concurrently.
      - Use run_and_save() which enforces serialisation automatically.
      - The PUBLIC_REGISTER_REQUEST_DELAY constant controls the inter-request gap.

    Returns:
        Dict containing the full licence profile from the public register.

    Raises:
        HBCAPIError — on 404 (not found) or unrecoverable errors after retries
        requests.RequestException — on network failure
    """
    require_non_empty(licence_id, "licence_id")

    # The licenceID from browse() is already the pipe-separated key the public
    # register expects. Pipe and slash must not be percent-encoded.
    encoded_id = requests.utils.quote(licence_id, safe="")
    url = (
        f"{PUBLIC_REGISTER_BASE_URL}/licence/search/details"
        f"/{PUBLIC_REGISTER_LICENCE_TYPE}/{encoded_id}"
    )

    logger.debug("public_register_details(licence_id=%r) -> %s", licence_id, url)

    last_exc = None
    max_attempts = len(PUBLIC_REGISTER_429_BACKOFF) + 1

    for attempt in range(max_attempts):
        try:
            resp = requests.get(
                url,
                headers=_browser_headers(),
                timeout=30,
            )
        except requests.RequestException as exc:
            last_exc = exc
            wait = PUBLIC_REGISTER_429_BACKOFF[
                min(attempt, len(PUBLIC_REGISTER_429_BACKOFF) - 1)
            ]
            logger.warning(
                "public_register_details: network error on attempt %d/%d — retrying in %ds: %s",
                attempt + 1,
                max_attempts,
                wait,
                exc,
            )
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 404:
            raise HBCAPIError(
                404, f"No public register record found for licenceID={licence_id!r}"
            )

        if resp.status_code == 429:
            if attempt < max_attempts - 1:
                wait = PUBLIC_REGISTER_429_BACKOFF[
                    min(attempt, len(PUBLIC_REGISTER_429_BACKOFF) - 1)
                ]
                logger.warning(
                    "public_register_details: Cloudflare 429 on attempt %d/%d "
                    "for licenceID=%r — waiting %ds before retry.",
                    attempt + 1,
                    max_attempts,
                    licence_id,
                    wait,
                )
                time.sleep(wait)
                continue
            raise HBCAPIError(
                429,
                f"Rate-limited after {max_attempts} attempts for licenceID={licence_id!r}",
            )

        if resp.status_code in (500, 502, 503, 504):
            wait = PUBLIC_REGISTER_429_BACKOFF[
                min(attempt, len(PUBLIC_REGISTER_429_BACKOFF) - 1)
            ]
            logger.warning(
                "public_register_details: server error %d on attempt %d/%d — retrying in %ds.",
                resp.status_code,
                attempt + 1,
                max_attempts,
                wait,
            )
            time.sleep(wait)
            last_exc = HBCAPIError(resp.status_code, resp.text[:200])
            continue

        # Any other status (400, 403, etc.) — not worth retrying
        raise HBCAPIError(resp.status_code, resp.text[:500])

    if last_exc:
        raise last_exc
    raise HBCAPIError(
        0, f"public_register_details: failed after {max_attempts} attempts"
    )


# ---------------------------------------------------------------------------
# Stateful client (recommended for pipeline use)
# ---------------------------------------------------------------------------


class HBCClient:
    """
    Stateful client that manages token lifecycle automatically.

    Proactively re-authenticates when the token has fewer than
    TOKEN_EXPIRY_BUFFER_SECONDS (60s) remaining, so long-running
    pipeline jobs never hit a mid-batch 401.

    Usage:
        client = HBCClient(api_key="...", api_secret="...")
        results = client.browse("smith plumbing")
        detail  = client.details("1-17YIOEU")
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        Inputs:
            api_key    — OneGov-issued API key (or set API_KEY in .env)
            api_secret — OneGov-issued API secret (or set API_SECRET in .env)
        """
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret must not be empty")
        self._api_key = api_key
        self._api_secret = api_secret
        self._token: Optional[Token] = None

    def ensure_token(self) -> Token:
        """Return a valid token, re-authenticating if necessary."""
        if self._token is None or not self._token.is_valid:
            logger.info(
                "Token %s — re-authenticating.",
                "absent" if self._token is None else "expiring soon",
            )
            self._token = get_access_token(self._api_key, self._api_secret)
        return self._token

    def browse(self, search_text: str) -> list[dict]:
        """See browse() for full documentation."""
        return browse(self.ensure_token(), self._api_key, search_text)

    def verify(self, licence_number: str) -> list[dict]:
        """See verify() for full documentation. Not for bulk pipeline use."""
        return verify(self.ensure_token(), self._api_key, licence_number)

    def details(self, licence_id: str) -> dict:
        """See details() for full documentation."""
        return details(self.ensure_token(), self._api_key, licence_id)

    def public_register_details(self, licence_id: str) -> dict:
        """
        See public_register_details() for full documentation.

        No authentication is required — this is a convenience wrapper
        so all licence lookups can go through the same client object.

        Inputs:
            licence_id — licenceID field from browse()/verify() results
                         (already pipe-separated, e.g. "HBCF14866998|107-NSWDHIBHWI/189386")
        """
        return public_register_details(licence_id)

    @property
    def token_status(self) -> str:
        """Human-readable token status string for logging."""
        if self._token is None:
            return "No token acquired yet"
        return repr(self._token)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def auth_headers(token: Token, api_key: str) -> dict:
    """Build the standard auth headers required by all data endpoints."""
    return {
        "Authorization": f"Bearer {token.access_token}",
        "apikey": api_key,
    }


def get_with_retry(
    url: str,
    headers: dict,
    params: dict,
    token: Token,
    api_key: str,
) -> requests.Response:
    """
    GET with retry logic:
      - 5xx: retry up to MAX_RETRIES times with exponential backoff
      - 401: log warning (caller should use HBCClient to auto-refresh)
      - 400: raise immediately (bad input, retrying won't help)
    """
    last_exc = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as exc:
            last_exc = exc
            maybe_backoff(attempt, f"Network error on attempt {attempt + 1}: {exc}")
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code == 400:
            body = safe_json(resp)
            raise HBCAPIError(400, body.get("message", resp.text))

        if resp.status_code == 404:
            # Record not found — not a transient error, retrying won't help.
            body = safe_json(resp)
            raise HBCAPIError(404, body.get("detail", resp.text))

        if resp.status_code == 408:
            # Traffic limit exceeded — retrying immediately will keep failing.
            body = safe_json(resp)
            raise HBCAPIError(408, body.get("message", resp.text))

        if resp.status_code == 401:
            # Token expired mid-call. Log and raise — HBCClient handles re-auth.
            logger.warning(
                "401 received. Token may have expired. Re-authenticate and retry."
            )
            raise HBCAPIError(401, "Invalid or expired access token")

        if resp.status_code == 429:
            wait = (
                RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)] * 2
            )
            logger.warning("429 rate limit hit. Waiting %ds before retry.", wait)
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            maybe_backoff(
                attempt, f"Server error {resp.status_code} on attempt {attempt + 1}"
            )
            last_exc = HBCAPIError(resp.status_code, resp.text)
            continue

        # Any other unexpected status
        raise HBCAPIError(resp.status_code, resp.text)

    if last_exc:
        raise last_exc
    raise HBCAPIError(0, f"Failed after {MAX_RETRIES} retries")


def maybe_backoff(attempt: int, message: str):
    """Log a warning and sleep if there are retries remaining."""
    if attempt < MAX_RETRIES:
        wait = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
        logger.warning("%s — retrying in %ds...", message, wait)
        time.sleep(wait)
    else:
        logger.error("%s — no retries remaining.", message)


def safe_json(resp: requests.Response) -> dict:
    """Parse response JSON safely, returning empty dict on failure."""
    try:
        return resp.json()
    except Exception:
        return {}


def require_non_empty(value: str, name: str):
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

# Scalar fields written to licence_details.csv (nested objects are
# flattened inline; arrays are written to their own normalised CSVs)
DETAILS_SCALAR_FIELDS = [
    "licenceID",
    "licenceNumber",
    "licenceName",
    "licensee",
    "licenceType",
    "status",
    "startDate",
    "expiryDate",
    "refusedDate",
    "address",
    "suburb",
    "postcode",
    "businessNames",
    "vehicleRegistration",
    "publicWarningsCount",
    "cautionReprimandCount",
    # insuranceClaimCounts sub-fields — flattened to top level
    "claimCount",
    "claimCountOldScheme",
    "insuranceClaimsPaid",
    "statutoryInsuranceClaimsPaid",
]


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def open_csv(path: str, fieldnames: list[str]):
    """Open a CSV for appending, writing header only if the file is new."""
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return f, writer


def save_browse_csv(results: list[dict], path: str = "output/licences_raw.csv"):
    """
    Write browse results to a CSV file.

    Each call appends to the file (creates it with a header if new).
    Deduplicate by licenceID before calling this if running multiple searches.

    Inputs:
        results — list of dicts returned by browse()
        path    — output file path (default: output/licences_raw.csv)
    """
    if not results:
        logger.info("save_browse_csv: no results to write.")
        return

    ensure_dir(path)
    fieldnames = list(results[0].keys())
    f, writer = open_csv(path, fieldnames)
    try:
        writer.writerows(results)
    finally:
        f.close()

    logger.info("Wrote %d browse rows to %s", len(results), path)


def save_details_csv(detail: dict, out_dir: str = "output", licence_id: str = ""):
    """
    Flatten a single details() response and write rows to multiple CSVs.

    Files written (all in out_dir):
        licence_details.csv          — one row, scalar fields only
        normalised_classes.csv       — one row per licence class
        normalised_conditions.csv    — one row per condition
        normalised_premises.csv      — one row per venue
        normalised_building_sites.csv — one row per building site
        normalised_disciplinary.csv  — one row per disciplinary event
          (infringement notices, CTTT orders, public warnings merged with a
           'event_type' discriminator column)

    Inputs:
        detail     — dict returned by details()
        out_dir    — directory to write CSV files into (default: output)
        licence_id — licenceID used to request this record; injected as the
                     primary key because the API response does not echo it back
    """
    os.makedirs(out_dir, exist_ok=True)

    # Bug fix 1: core licence fields are nested under "licenceDetail", not at the
    # top level of the response.
    # Bug fix 2: licenceID is not returned by the API — it must be passed in and
    # injected so that all output CSVs have a consistent primary key.
    core = detail.get("licenceDetail") or {}

    # --- licence_details.csv (scalar fields + flattened claim counts) ---
    row = {k: core.get(k) for k in DETAILS_SCALAR_FIELDS}
    row["licenceID"] = licence_id  # inject the key the API does not echo back

    # Bug fix 3: complianceActions sub-fields are nested under "complianceActions",
    # not at the top level of the response.
    compliance = detail.get("complianceActions") or {}
    row["publicWarningsCount"] = compliance.get("publicWarningsCount")
    row["cautionReprimandCount"] = compliance.get("cautionReprimandCount")

    counts = compliance.get("insuranceClaimCounts") or {}
    row["claimCount"] = counts.get("claimCount")
    row["claimCountOldScheme"] = counts.get("claimCountOldScheme")
    row["insuranceClaimsPaid"] = counts.get("insuranceClaimsPaid")
    row["statutoryInsuranceClaimsPaid"] = counts.get("statutoryInsuranceClaimsPaid")

    append_rows(
        os.path.join(out_dir, "licence_details.csv"),
        DETAILS_SCALAR_FIELDS,
        [row],
    )

    # --- normalised_classes.csv ---
    classes = detail.get("licenceClasses") or []
    if classes:
        fieldnames = ["licenceID"] + list(classes[0].keys())
        append_rows(
            os.path.join(out_dir, "normalised_classes.csv"),
            fieldnames,
            [{"licenceID": licence_id, **c} for c in classes],
        )

    # --- normalised_conditions.csv ---
    conditions = detail.get("conditions") or []
    if conditions:
        fieldnames = ["licenceID"] + list(conditions[0].keys())
        append_rows(
            os.path.join(out_dir, "normalised_conditions.csv"),
            fieldnames,
            [{"licenceID": licence_id, **c} for c in conditions],
        )

    # --- normalised_premises.csv ---
    venues = detail.get("venues") or []
    if venues:
        fieldnames = ["licenceID"] + list(venues[0].keys())
        append_rows(
            os.path.join(out_dir, "normalised_premises.csv"),
            fieldnames,
            [{"licenceID": licence_id, **v} for v in venues],
        )

    # --- normalised_building_sites.csv ---
    # buildingSites is a single object in the schema, not an array
    site = detail.get("buildingSites")
    if site:
        row_s = {"licenceID": licence_id, **site}
        append_rows(
            os.path.join(out_dir, "normalised_building_sites.csv"),
            list(row_s.keys()),
            [row_s],
        )

    # --- normalised_disciplinary.csv ---
    # Merge three disciplinary arrays into one table with an event_type column
    disc_rows = []

    for notice in compliance.get("infringementNotices") or []:
        disc_rows.append(
            {
                "licenceID": licence_id,
                "event_type": "infringement_notice",
                "date": notice.get("date"),
                "name": notice.get("name"),
                "section": notice.get("section"),
                "act": notice.get("act"),
                "penalty": notice.get("penalty"),
                "judgementDate": None,
                "complyByDate": None,
                "isActive": None,
                "publicWarningText": None,
            }
        )

    for order in compliance.get("ctttOrders") or []:
        disc_rows.append(
            {
                "licenceID": licence_id,
                "event_type": "cttt_order",
                "date": None,
                "name": None,
                "section": None,
                "act": None,
                "penalty": None,
                "judgementDate": order.get("judgementDate"),
                "complyByDate": order.get("complyByDate"),
                "isActive": None,
                "publicWarningText": None,
            }
        )

    for warning in compliance.get("publicWarnings") or []:
        disc_rows.append(
            {
                "licenceID": licence_id,
                "event_type": "public_warning",
                "date": None,
                "name": None,
                "section": None,
                "act": None,
                "penalty": None,
                "judgementDate": None,
                "complyByDate": None,
                "isActive": warning.get("isActive"),
                "publicWarningText": warning.get("publicWarningText"),
            }
        )

    if disc_rows:
        fieldnames = list(disc_rows[0].keys())
        append_rows(
            os.path.join(out_dir, "normalised_disciplinary.csv"),
            fieldnames,
            disc_rows,
        )

    logger.debug("Saved details for licenceID=%s to %s", licence_id, out_dir)


# Scalar fields written to public_register_details.csv
PUBLIC_REGISTER_SCALAR_FIELDS = [
    "licenceId",
    "licenceNumber",
    "licenceName",  # contractor licence number (e.g. "228617C")
    "licensee",
    "licenseeType",
    "licenceType",
    "licenceTypeFriendly",
    "licenceGroup",
    "status",
    "granted",
    "address",
    "addressType",
    "suburb",
    "state",
    "postcode",
    "latitude",
    "longitude",
    "businessNames",
    # derived / flattened from compliances list
    "total_claims_paid",
    "total_settlement_amount",
]


def save_public_register_csv(data: dict, out_dir: str = "output"):
    """
    Flatten a single public_register_details() response and write rows to
    multiple CSVs.

    The public register wraps everything under a "componentData" key, and its
    schema differs significantly from the OneGov details() response:
      - claims are in "compliances" (not complianceActions)
      - contractor and insurer details are in "associatedRoles"
      - the property address is in "locations[].premises[]"

    Files written (all in out_dir):
        public_register_details.csv        — one row per certificate, scalar fields
        normalised_pr_claims.csv           — one row per claim (type, amount, date)
        normalised_pr_associated_roles.csv — one row per party per role
        normalised_pr_contractor_licences.csv — one row per linked contractor licence
        normalised_pr_locations.csv        — one row per property address

    Inputs:
        data    — dict returned by public_register_details()
        out_dir — directory to write CSV files into (default: "output")
    """
    os.makedirs(out_dir, exist_ok=True)

    cd = data.get("componentData") or {}
    licence_id = cd.get("licenceId", "")

    # -------------------------------------------------------------------------
    # public_register_details.csv — one scalar row per certificate
    # -------------------------------------------------------------------------
    compliances = cd.get("compliances") or []
    total_settlement = 0.0
    for c in compliances:
        raw = (
            (c.get("settlementAmount") or "").replace("$", "").replace(",", "").strip()
        )
        try:
            total_settlement += float(raw)
        except ValueError:
            pass

    # businessNames may be a string or a list — normalise to a semicolon-joined string
    biz_raw = cd.get("businessNames")
    if isinstance(biz_raw, list):
        biz_str = "; ".join(biz_raw)
    else:
        biz_str = biz_raw or ""
    # Also check businessNameList for completeness
    biz_list = cd.get("businessNameList") or []
    if biz_list and not biz_str:
        biz_str = "; ".join(biz_list)

    scalar_row = {
        "licenceId": licence_id,
        "licenceNumber": cd.get("licenceNumber"),
        "licenceName": cd.get("licenceName"),
        "licensee": cd.get("licensee"),
        "licenseeType": cd.get("licenseeType"),
        "licenceType": cd.get("licenceType"),
        "licenceTypeFriendly": cd.get("licenceTypeFriendly"),
        "licenceGroup": cd.get("licenceGroup"),
        "status": cd.get("status"),
        "granted": cd.get("granted"),
        "address": cd.get("address"),
        "addressType": cd.get("addressType"),
        "suburb": cd.get("suburb"),
        "state": cd.get("state"),
        "postcode": cd.get("postcode"),
        "latitude": cd.get("latitude"),
        "longitude": cd.get("longitude"),
        "businessNames": biz_str,
        "total_claims_paid": len(compliances),
        "total_settlement_amount": (
            round(total_settlement, 2) if total_settlement else None
        ),
    }

    append_rows(
        os.path.join(out_dir, "public_register_details.csv"),
        PUBLIC_REGISTER_SCALAR_FIELDS,
        [scalar_row],
    )

    # -------------------------------------------------------------------------
    # normalised_pr_claims.csv — one row per claim paid
    # -------------------------------------------------------------------------
    if compliances:
        claim_rows = []
        for c in compliances:
            claim_rows.append(
                {
                    "licenceId": licence_id,
                    "licenceNumber": cd.get("licenceNumber"),
                    "sequence": c.get("sequence"),
                    "type": c.get("type"),
                    "description": c.get("description"),
                    "settlementAmount": c.get("settlementAmount"),
                    "dateClaimed": c.get("dateClaimed"),
                }
            )
        append_rows(
            os.path.join(out_dir, "normalised_pr_claims.csv"),
            list(claim_rows[0].keys()),
            claim_rows,
        )

    # -------------------------------------------------------------------------
    # normalised_pr_associated_roles.csv — one row per party per role
    # normalised_pr_contractor_licences.csv — one row per linked licence
    # -------------------------------------------------------------------------
    role_rows = []
    contractor_licence_rows = []

    for role in cd.get("associatedRoles") or []:
        role_name = role.get("name", "")
        for party in role.get("parties") or []:
            role_rows.append(
                {
                    "licenceId": licence_id,
                    "licenceNumber": cd.get("licenceNumber"),
                    "role": role_name,
                    "partyName": party.get("name"),
                    "partyRole": party.get("role"),
                    "suburb": party.get("suburb"),
                    "state": party.get("state"),
                    "postcode": party.get("postcode"),
                    "address": party.get("address"),
                    "website": party.get("website"),
                    "contact": party.get("contact"),
                    "email": party.get("email"),
                    "start": party.get("start"),
                }
            )
            for lic in party.get("licences") or []:
                contractor_licence_rows.append(
                    {
                        "licenceId": licence_id,
                        "licenceNumber": cd.get("licenceNumber"),
                        "partyName": party.get("name"),
                        "contractorLicenceNumber": lic.get("licenceNumber"),
                        "contractorLicenceType": lic.get("licenceType"),
                        "contractorLicenceID": lic.get("licenceID"),
                        "contractorStatus": lic.get("status"),
                        "contractorExpiry": lic.get("expiry"),
                    }
                )

    if role_rows:
        append_rows(
            os.path.join(out_dir, "normalised_pr_associated_roles.csv"),
            list(role_rows[0].keys()),
            role_rows,
        )

    if contractor_licence_rows:
        append_rows(
            os.path.join(out_dir, "normalised_pr_contractor_licences.csv"),
            list(contractor_licence_rows[0].keys()),
            contractor_licence_rows,
        )

    # -------------------------------------------------------------------------
    # normalised_pr_locations.csv — one row per property/premises address
    # -------------------------------------------------------------------------
    location_rows = []
    for loc_group in cd.get("locations") or []:
        loc_type = loc_group.get("type", "")
        for premise in loc_group.get("premises") or []:
            location_rows.append(
                {
                    "licenceId": licence_id,
                    "licenceNumber": cd.get("licenceNumber"),
                    "locationType": loc_type,
                    "address": premise.get("address"),
                    "suburb": premise.get("suburb"),
                    "state": premise.get("state"),
                    "postcode": premise.get("postcode"),
                    "latitude": premise.get("latitude"),
                    "longitude": premise.get("longitude"),
                    "dpNumber": premise.get("dpNumber"),
                    "lotNumber": premise.get("lotNumber"),
                }
            )

    if location_rows:
        append_rows(
            os.path.join(out_dir, "normalised_pr_locations.csv"),
            list(location_rows[0].keys()),
            location_rows,
        )

    logger.debug(
        "Saved public register details for licenceId=%s to %s", licence_id, out_dir
    )


# ---------------------------------------------------------------------------
# Search term generation — fuzz the browse endpoint for maximum coverage
# ---------------------------------------------------------------------------

# All NSW suburbs/localities. Broad coverage catches licensees by address.
NSW_SUBURBS = [
    "Abbotsford",
    "Abercrombie",
    "Aberdeen",
    "Abermain",
    "Acacia Gardens",
    "Adamstown",
    "Adamstown Heights",
    "Adelaide",
    "Adelong",
    "Agnes Banks",
    "Airds",
    "Airlie Beach",
    "Albion Park",
    "Albion Park Rail",
    "Albury",
    "Aldavilla",
    "Alexandria",
    "Alfords Point",
    "Allambie Heights",
    "Allawah",
    "Allendale",
    "Alstonville",
    "Altona",
    "Ambarvale",
    "Anambah",
    "Annandale",
    "Annangrove",
    "Appin",
    "Arcadia",
    "Ardlethan",
    "Armidale",
    "Arncliffe",
    "Arndell Park",
    "Arrawarra",
    "Artarmon",
    "Ashbury",
    "Ashcroft",
    "Ashfield",
    "Asquith",
    "Auburn",
    "Austinmer",
    "Avalon Beach",
    "Avoca Beach",
    "Avondale",
    "Balgowlah",
    "Balgowlah Heights",
    "Balgownie",
    "Ballina",
    "Balmain",
    "Balmain East",
    "Balmoral",
    "Bangor",
    "Banksia",
    "Banksmeadow",
    "Bankstown",
    "Barangaroo",
    "Bargo",
    "Barnsley",
    "Barraba",
    "Barrack Heights",
    "Bass Hill",
    "Batemans Bay",
    "Bathurst",
    "Baulkham Hills",
    "Bayview",
    "Beacon Hill",
    "Beecroft",
    "Bega",
    "Belfield",
    "Bella Vista",
    "Bellambi",
    "Bellevue Hill",
    "Belmont",
    "Belmore",
    "Beresfield",
    "Berkeley",
    "Berkeley Vale",
    "Bermagui",
    "Berridale",
    "Berry",
    "Bethungra",
    "Bexley",
    "Bexley North",
    "Bidwill",
    "Birrong",
    "Blackheath",
    "Blacksmiths",
    "Blacktown",
    "Blackwall",
    "Blair Athol",
    "Blakehurst",
    "Blayney",
    "Blenheim",
    "Blue Haven",
    "Bodalla",
    "Bolwarra",
    "Bomaderry",
    "Bombo",
    "Bondi",
    "Bondi Beach",
    "Bondi Junction",
    "Bonnet Bay",
    "Bonnyrigg",
    "Bonnyrigg Heights",
    "Booker Bay",
    "Boomerang Beach",
    "Bossley Park",
    "Botany",
    "Bourke",
    "Bowenfels",
    "Bowral",
    "Bowraville",
    "Box Hill",
    "Braidwood",
    "Branxton",
    "Breakfast Point",
    "Brighton-Le-Sands",
    "Bringelly",
    "Brisbane Water",
    "Broadmeadow",
    "Bronte",
    "Brooklyn",
    "Brookvale",
    "Broulee",
    "Bulli",
    "Bullaburra",
    "Bundanoon",
    "Bundeena",
    "Bundena",
    "Burraneer",
    "Burwood",
    "Bushy",
    "Buttaba",
    "Byron Bay",
    "Cabarita",
    "Cabramattta",
    "Casula",
    "Camden",
    "Campbelltown",
    "Camperdown",
    "Campsie",
    "Canada Bay",
    "Canley Heights",
    "Canley Vale",
    "Canowindra",
    "Canterbury",
    "Cardiff",
    "Carlingford",
    "Carlton",
    "Caringbah",
    "Caringbah South",
    "Carnes Hill",
    "Carramar",
    "Carss Park",
    "Cartwright",
    "Casino",
    "Castle Hill",
    "Castlecrag",
    "Castlereagh",
    "Casula",
    "Catalina",
    "Catho",
    "Caves Beach",
    "Cecil Hills",
    "Cessnock",
    "Charlestown",
    "Cheltenham",
    "Cherrybrook",
    "Chester Hill",
    "Chifley",
    "Chippendale",
    "Chipping Norton",
    "Chullora",
    "Churchill",
    "Circular Quay",
    "Claremont Meadows",
    "Clarence",
    "Claymore",
    "Clemton Park",
    "Clontarf",
    "Clovelly",
    "Cobar",
    "Cobbitty",
    "Coffs Harbour",
    "Collaroy",
    "Collaroy Plateau",
    "Colyton",
    "Como",
    "Concord",
    "Concord West",
    "Condell Park",
    "Coniston",
    "Constitution Hill",
    "Coogee",
    "Coolamon",
    "Coonamble",
    "Cootamundra",
    "Copacabana",
    "Corrimal",
    "Cosgrove",
    "Cotswold Hills",
    "Couridjah",
    "Cowra",
    "Cremorne",
    "Cremorne Point",
    "Cromer",
    "Cronulla",
    "Crows Nest",
    "Croydon",
    "Croydon Park",
    "Culburra Beach",
    "Curl Curl",
    "Currans Hill",
    "Daceyville",
    "Dapto",
    "Darkes Forest",
    "Darling Point",
    "Darlinghurst",
    "Darlington",
    "Davidson",
    "Davistown",
    "Dee Why",
    "Deniliquin",
    "Denistone",
    "Denistone East",
    "Denistone West",
    "Derrimut",
    "Doonside",
    "Double Bay",
    "Douglas Park",
    "Drummoyne",
    "Dubbo",
    "Duffys Forest",
    "Dulwich Hill",
    "Dundas",
    "Dundas Valley",
    "Dural",
    "Eagle Vale",
    "Earlwood",
    "East Gosford",
    "East Hills",
    "East Killara",
    "East Lindfield",
    "East Maitland",
    "East Ryde",
    "Eastern Creek",
    "Eastgardens",
    "Eastlakes",
    "Eastwood",
    "Edensor Park",
    "Edgeworth",
    "Edmonton",
    "Eight Mile Plains",
    "Elermore Vale",
    "Emerton",
    "Emu Heights",
    "Emu Plains",
    "Engadine",
    "Enmore",
    "Epping",
    "Ermington",
    "Erskineville",
    "Ettalong Beach",
    "Eungai",
    "Evans Head",
    "Fairfield",
    "Fairfield East",
    "Fairfield Heights",
    "Fairfield West",
    "Fairy Meadow",
    "Farmborough Heights",
    "Faulconbridge",
    "Figtree",
    "Fingal Bay",
    "Five Dock",
    "Flemington",
    "Fletcher",
    "Flinders",
    "Forbes",
    "Forest Lodge",
    "Forestville",
    "Forster",
    "Fountain Dale",
    "Freshwater",
    "Galston",
    "Georges Hall",
    "Gillieston Heights",
    "Girards Hill",
    "Girraween",
    "Gladesville",
    "Glebe",
    "Glen Alpine",
    "Glen Innes",
    "Glenbrook",
    "Glendenning",
    "Glenfield",
    "Glenhaven",
    "Glenmore Park",
    "Glenning Valley",
    "Glenorie",
    "Glenwood",
    "Gloucester",
    "Gordon",
    "Gosford",
    "Goulburn",
    "Grafton",
    "Granville",
    "Grose Vale",
    "Guildford",
    "Guildford West",
    "Gymea",
    "Gymea Bay",
    "Halekulani",
    "Hamlyn Terrace",
    "Hammondville",
    "Harrington Park",
    "Harris Park",
    "Hazelbrook",
    "Hebersham",
    "Helidon",
    "Helensburgh",
    "Henty",
    "Hillsdale",
    "Hinchinbrook",
    "Hoxton Park",
    "Hurstville",
    "Hurstville Grove",
    "Illawong",
    "Ingleburn",
    "Inverell",
    "Jannali",
    "Jerrabomberra",
    "Jervis Bay",
    "Jindabyne",
    "Kanwal",
    "Kariong",
    "Katoomba",
    "Kearns",
    "Kellyville",
    "Kellyville Ridge",
    "Kembla Grange",
    "Kempsey",
    "Kiama",
    "Killara",
    "Killarney Heights",
    "Kingsford",
    "Kingsgrove",
    "Kirrawee",
    "Kirribilli",
    "Kogarah",
    "Kogarah Bay",
    "Kooringal",
    "Ku-ring-gai",
    "Kurnell",
    "Kurrajong",
    "Lake Haven",
    "Lake Illawarra",
    "Lake Macquarie",
    "Lakemba",
    "Lansvale",
    "Laurieton",
    "Lavington",
    "Lawson",
    "Leeton",
    "Leichhardt",
    "Lennox Head",
    "Leppington",
    "Lidcombe",
    "Lilli Pilli",
    "Lindfield",
    "Lismore",
    "Lithgow",
    "Liverpool",
    "Loftus",
    "Longueville",
    "Lugarno",
    "Lurnea",
    "Macksville",
    "Macquarie Fields",
    "Macquarie Links",
    "Macquarie Park",
    "Maitland",
    "Manly",
    "Manly Vale",
    "Marayong",
    "Marrickville",
    "Marsden Park",
    "Mascot",
    "Matraville",
    "Mayfield",
    "McGraths Hill",
    "Meadowbank",
    "Medlow Bath",
    "Menai",
    "Merewether",
    "Merimbula",
    "Merrylands",
    "Minchinbury",
    "Miranda",
    "Mittagong",
    "Molong",
    "Mona Vale",
    "Moorebank",
    "Morisset",
    "Mortdale",
    "Mosman",
    "Mount Annan",
    "Mount Colah",
    "Mount Druitt",
    "Mount Hutton",
    "Mount Keira",
    "Mount Kembla",
    "Mount Ousley",
    "Mount Pritchard",
    "Mount Warrigal",
    "Mudgee",
    "Mullumbimby",
    "Mulgoa",
    "Mulgrave",
    "Murwillumbah",
    "Muswellbrook",
    "Narellan",
    "Narrabeen",
    "Narraweena",
    "Narromine",
    "Neutral Bay",
    "Newington",
    "Newport",
    "Newton",
    "Ngunnawal",
    "Niagara Park",
    "Noraville",
    "Normanhurst",
    "North Avoca",
    "North Curl Curl",
    "North Gosford",
    "North Manly",
    "North Narrabeen",
    "North Parramatta",
    "North Richmond",
    "North Rocks",
    "North Ryde",
    "North St Ives",
    "North Sydney",
    "North Turramurra",
    "North Wahroonga",
    "North Willoughby",
    "Northbridge",
    "Nowra",
    "Nundle",
    "Oakhurst",
    "Oakville",
    "Oatlands",
    "Oberon",
    "Old Toongabbie",
    "Orange",
    "Ourimbah",
    "Oxford Falls",
    "Oxley Park",
    "Padstow",
    "Padstow Heights",
    "Palm Beach",
    "Panania",
    "Parkes",
    "Parramatta",
    "Peakhurst",
    "Pendle Hill",
    "Penshurst",
    "Penrith",
    "Petersham",
    "Picnic Point",
    "Picton",
    "Plumpton",
    "Point Clare",
    "Point Frederick",
    "Port Macquarie",
    "Port Stephens",
    "Potts Point",
    "Punchbowl",
    "Pymble",
    "Pyrmont",
    "Quakers Hill",
    "Queanbeyan",
    "Randwick",
    "Ramsgate",
    "Ramsgate Beach",
    "Redfern",
    "Regents Park",
    "Revesby",
    "Revesby Heights",
    "Rhodes",
    "Richmond",
    "Riverstone",
    "Rockdale",
    "Rouse Hill",
    "Royal National Park",
    "Rozelle",
    "Ruse",
    "Rushcutters Bay",
    "Rutherford",
    "Ryde",
    "Salamander Bay",
    "Saratoga",
    "Schofields",
    "Seaforth",
    "Seven Hills",
    "Shellharbour",
    "Silverdale",
    "Silverwater",
    "Singleton",
    "Smeaton Grange",
    "Smithfield",
    "South Coogee",
    "South Granville",
    "South Hurstville",
    "South Penrith",
    "South Turramurra",
    "South Windsor",
    "Spring Farm",
    "Springfield",
    "St Clair",
    "St George",
    "St Helens Park",
    "St Ives",
    "St Ives Chase",
    "St Johns Park",
    "St Leonards",
    "St Marys",
    "St Peters",
    "Stanmore",
    "Strathfield",
    "Strathfield South",
    "Streaky Bay",
    "Summer Hill",
    "Sunnybank",
    "Sutherland",
    "Sydney",
    "Sylvania",
    "Sylvania Waters",
    "Tahmoor",
    "Tamworth",
    "Tarago",
    "Taree",
    "Taren Point",
    "Tempe",
    "Tenambit",
    "Terrigal",
    "The Entrance",
    "The Hills",
    "The Oaks",
    "Thornleigh",
    "Tingira Heights",
    "Toongabbie",
    "Toormina",
    "Toronto",
    "Toukley",
    "Tuggerah",
    "Tuggerawong",
    "Tumbi Umbi",
    "Tumut",
    "Turramurra",
    "Tweed Heads",
    "Ulladulla",
    "Umina Beach",
    "Uralla",
    "Urunga",
    "Vaucluse",
    "Villawood",
    "Vineyard",
    "Wahroonga",
    "Waitara",
    "Wakehurst",
    "Wallsend",
    "Warrawong",
    "Warwick Farm",
    "Warringah",
    "Waterfall",
    "Waterloo",
    "Watsons Bay",
    "Waverley",
    "Wentworthville",
    "Werrington",
    "West Gosford",
    "West Hoxton",
    "West Pennant Hills",
    "West Pymble",
    "West Ryde",
    "Westmead",
    "Wetherill Park",
    "Whalan",
    "Wiley Park",
    "Williamtown",
    "Windsor",
    "Windsor Downs",
    "Winston Hills",
    "Wollert",
    "Wollongong",
    "Wollstonecraft",
    "Woodbine",
    "Woodcroft",
    "Woodford",
    "Woollahra",
    "Woolwich",
    "Woy Woy",
    "Wyong",
    "Yamba",
    "Yagoona",
    "Yass",
    "Young",
]

# Two-letter alphabetic combos — catches names/business names the API fuzzy-matches
ALPHA_PAIRS = [a + b for a in "abcdefghijklmnoprstw" for b in "aeiou"]

# HBCF certificate number year prefixes (covers ~2000-2025 issuance years)
HBCF_PREFIXES = [f"HBCF{str(y)[2:]}" for y in range(2000, 2026)]


def generate_search_terms(n: int = 0) -> list[str]:
    """
    Generate a broad set of search terms for fuzzing the browse endpoint.

    Combines three dimensions to maximise certificate discovery:
      1. NSW suburbs — catches licensees by registered address
      2. Two-letter alpha pairs — catches partial name matches
      3. HBCF year prefixes — directly targets certificate number ranges

    Returns a deduplicated list of ~1 200 search terms by default.
    When n is greater than zero, returns up to n randomly ordered terms.
    The caller should pass these to run_and_save() as the search_terms argument.

    Usage:
        terms = generate_search_terms()
        run_and_save(client, search_terms=terms, out_dir="./output")
    """
    seen: set[str] = set()
    terms: list[str] = []

    for t in [*NSW_SUBURBS, *ALPHA_PAIRS, *HBCF_PREFIXES]:
        key = t.lower().strip()
        if key not in seen:
            seen.add(key)
            terms.append(t)

    if n > 0:
        random.shuffle(terms)
        return terms[:n]

    return terms


def run_and_save(
    client: "HBCClient",
    search_terms: list[str],
    out_dir: str = "output",
    request_delay: float = 0.5,
    browse_delay: float = 0.4,
    normalise_by_licence_name: bool = False,
    max_workers: int = 10,
):
    """
    Convenience function: browse all search terms, deduplicate, fetch details,
    and write all CSVs.

    Inputs:
        client                    — authenticated HBCClient instance
        search_terms              — list of search strings for the browse sweep
        out_dir                   — directory to write all CSVs (default: "output")
        request_delay             — minimum seconds between requests per worker (default: 0.5).
                                    With max_workers=10 this gives ~20 licences/s sustained.
        browse_delay              — seconds to sleep between browse() calls (default: 0.4).
                                    Prevents 408 traffic-limit errors on the apikeyed endpoint
                                    during large fuzz sweeps (~2.5 req/s sustained).
        normalise_by_licence_name — when True, deduplicate the details sweep by
                                    licenceName (the contractor licence number, e.g. "228617C")
                                    so that details() is called once per contractor rather than
                                    once per certificate. The first licenceID seen for each
                                    licenceName is used as the representative record.
                                    Default: False (fetch details for every licenceID).
        max_workers               — number of concurrent details() threads (default: 10).
                                    Throughput ≈ max_workers / request_delay licences/s.

    Output files written to out_dir:
        licences_raw.csv, licence_details.csv, normalised_classes.csv,
        normalised_conditions.csv, normalised_premises.csv,
        normalised_building_sites.csv, normalised_disciplinary.csv,
        public_register_details.csv, normalised_pr_claims.csv,
        normalised_pr_associated_roles.csv, normalised_pr_contractor_licences.csv,
        normalised_pr_locations.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Browse sweep + deduplication ---
    seen_ids: set[str] = set()
    seen_licence_names: set[str] = set()
    all_browse: list[dict] = []

    with tqdm(search_terms, desc="Browsing", unit="term", colour="cyan") as browse_bar:
        for term in browse_bar:
            browse_bar.set_postfix_str(term)
            try:
                results = client.browse(term)
            except HBCAPIError as e:
                logger.error("browse(%r) failed: %s", term, e)
                time.sleep(browse_delay)
                continue
            time.sleep(browse_delay)

            for r in results:
                lid = r.get("licenceID") or ""
                if lid and lid not in seen_ids:
                    seen_ids.add(lid)
                    all_browse.append(r)

    if all_browse:
        save_browse_csv(all_browse, os.path.join(out_dir, "licences_raw.csv"))

    # --- Build the details fetch list ---
    # With normalise_by_licence_name=True, keep only the first licenceID seen for
    # each unique licenceName (contractor licence number), skipping the rest.
    # With normalise_by_licence_name=False (default), fetch every licenceID.
    if normalise_by_licence_name:
        details_ids: list[str] = []
        for r in all_browse:
            licence_name = r.get("licenceName")
            lid = r.get("licenceID")
            if not lid:
                continue
            if licence_name and licence_name not in seen_licence_names:
                seen_licence_names.add(licence_name)
                details_ids.append(lid)
            elif not licence_name:
                # No licenceName — fall back to including the licenceID as-is
                details_ids.append(lid)
        logger.info(
            "normalise_by_licence_name: %d licenceIDs -> %d unique contractors",
            len(seen_ids),
            len(details_ids),
        )
    else:
        details_ids = list(seen_ids)

    # --- Details collection ---
    #
    # Two separate fetch strategies:
    #
    #  1. OneGov details()  — concurrent (ThreadPoolExecutor, max_workers threads)
    #     The OneGov API handles concurrency fine; use the same thread pool as before.
    #
    #  2. Public register   — strictly serialised, one request at a time.
    #     The endpoint is behind Cloudflare, which aggressively 429s concurrent
    #     requests even at low parallelism. We run PR fetches sequentially in the
    #     main thread, gated by pr_lock so worker threads can never overlap.
    #
    # csv_lock serialises all CSV writes so threads don't interleave rows.
    csv_lock = threading.Lock()
    pr_lock = threading.Lock()  # ensures only one PR request is in-flight at a time
    throttled: list[str] = []  # licenceIDs dropped due to OneGov 408 traffic limit

    def fetch_one(licence_id: str) -> bool:
        """Fetch from both OneGov and the public register, save both. Returns True on success."""
        time.sleep(request_delay)
        ok = True

        # --- OneGov details() (concurrent) ---
        try:
            detail = client.details(licence_id)
            with csv_lock:
                save_details_csv(detail, out_dir, licence_id=licence_id)
        except HBCAPIError as e:
            if e.status_code == 404:
                logger.warning("details(%r) skipped: no record found (404)", licence_id)
            elif e.status_code == 408:
                throttled.append(licence_id)
                ok = False
            else:
                logger.error("details(%r) failed: %s", licence_id, e)
                ok = False

        # --- Public register details() (serialised via pr_lock) ---
        # The licenceID from browse() is already the pipe-separated key the
        # public register expects — pass it directly.
        with pr_lock:
            try:
                pr_data = public_register_details(licence_id)
                with csv_lock:
                    save_public_register_csv(pr_data, out_dir)
            except HBCAPIError as e:
                if e.status_code == 404:
                    logger.warning(
                        "public_register_details(%r) skipped: not found (404)",
                        licence_id,
                    )
                elif e.status_code == 429:
                    logger.error(
                        "public_register_details(%r) still 429 after retries — skipped.",
                        licence_id,
                    )
                    ok = False
                else:
                    logger.error(
                        "public_register_details(%r) failed: %s", licence_id, e
                    )
                    ok = False
            finally:
                # Always pause before the next PR request, even on error,
                # to give Cloudflare's rate-limit window time to reset.
                time.sleep(PUBLIC_REGISTER_REQUEST_DELAY)

        return ok

    success = 0
    with tqdm(
        total=len(details_ids), desc="Fetching details", unit="licence", colour="green"
    ) as details_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_one, lid): lid for lid in details_ids}
            try:
                for future in as_completed(futures):
                    lid = futures[future]
                    details_bar.set_postfix_str(lid)
                    details_bar.update(1)
                    try:
                        if future.result():
                            success += 1
                    except Exception as e:
                        logger.error("Unexpected error for %r: %s", lid, e)
            except KeyboardInterrupt:
                logger.warning("Interrupted — cancelling pending tasks...")
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                details_bar.set_postfix_str("interrupted")
                print()  # newline after tqdm

    if throttled:
        logger.warning(
            "Traffic limit exceeded for %d licence(s) — skipped: %s",
            len(throttled),
            ", ".join(throttled),
        )
    logger.info(
        "%s. %d/%d records written to: %s",
        "Interrupted" if success < len(details_ids) else "Done",
        success,
        len(details_ids),
        os.path.abspath(out_dir),
    )


def append_rows(path: str, fieldnames: list[str], rows: list[dict]):
    """Write rows to a CSV, creating the file with a header if it doesn't exist."""
    if not rows:
        return
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    key = os.getenv("API_KEY")
    secret = os.getenv("API_SECRET")

    if not key or not secret:
        print("Set API_KEY and API_SECRET in your .env file before running.")
    else:
        client = HBCClient(api_key=key, api_secret=secret)

        print("\n--- Token status ---")
        client.ensure_token()
        print(client.token_status)

        # Full fuzz sweep — generates ~1 200 search terms across suburbs,
        # alpha pairs, and HBCF year prefixes to maximise certificate discovery.
        # Pass a custom search_terms list instead to target a specific subset.
        terms = generate_search_terms()
        print(f"\n--- Starting full fuzz sweep ({len(terms)} search terms) ---")
        run_and_save(
            client,
            search_terms=terms,
            out_dir="./output",
            request_delay=1,
            normalise_by_licence_name=False,
        )
        print("\nDone — check the ./output directory for CSV files.")
