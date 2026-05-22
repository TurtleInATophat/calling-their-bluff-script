"""
fetch_details.py
----------------
Resumable details fetcher for the NSW HBC Public Register.

Reads licences_raw.csv (and optionally other browse CSVs) to build a list of
licenceIDs, then hits both the OneGov details() endpoint and the public
register endpoint for each one — saving results incrementally so that any
interruption (Ctrl+C, crash, network drop) can be resumed cleanly.

Resume logic
~~~~~~~~~~~~
Before fetching, the script scans the existing output CSVs to find which
licenceIDs have already been processed. Only the remaining IDs are fetched,
so re-running the script after an interruption picks up exactly where it
left off with no duplicate rows.

Usage
~~~~~
    # Minimal — reads ./output/licences_raw.csv, writes back to ./output/
    python fetch_details.py

    # Full options
    python fetch_details.py \\
        --input  ./output/licences_raw.csv \\
        --extra  ./output/other_browse.csv \\
        --outdir ./output \\
        --workers 10 \\
        --delay  0.5 \\
        --pr-delay 0.5

Environment variables (via .env or shell):
    API_KEY     — OneGov API key
    API_SECRET  — OneGov API secret

Dependencies:
    pip install requests tqdm python-dotenv
    (hbc_api.py must be on the Python path or in the same directory)
"""

import argparse
import csv
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

# hbc_api.py must be importable (same directory or on PYTHONPATH)
from hbc_api import (
    HBCAPIError,
    HBCClient,
    PUBLIC_REGISTER_REQUEST_DELAY,
    save_details_csv,
    save_public_register_csv,
    public_register_details,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful-shutdown flag
# ---------------------------------------------------------------------------
_shutdown = threading.Event()


def _handle_signal(signum, frame):
    """Set the shutdown flag on SIGINT / SIGTERM so workers can drain cleanly."""
    if not _shutdown.is_set():
        print("\n[!] Interrupt received — finishing in-flight requests then saving...")
        _shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def read_licence_ids_from_csv(path: str) -> list[str]:
    """
    Return all non-empty licenceID values from a CSV file.
    Accepts files produced by save_browse_csv() (licences_raw.csv) or any
    other CSV that has a 'licenceID' column.
    """
    ids: list[str] = []
    if not os.path.exists(path):
        logger.warning("Input file not found, skipping: %s", path)
        return ids

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "licenceID" not in reader.fieldnames:
            logger.warning(
                "%s has no 'licenceID' column (columns: %s) — skipping.",
                path,
                reader.fieldnames,
            )
            return ids
        for row in reader:
            lid = (row.get("licenceID") or "").strip()
            if lid:
                ids.append(lid)

    logger.info("Read %d licenceIDs from %s", len(ids), path)
    return ids


def already_fetched_ids(out_dir: str) -> set[str]:
    """
    Scan the existing output CSVs to find licenceIDs that have already been
    processed.  We check both licence_details.csv (OneGov) and
    public_register_details.csv (public register) and treat a licenceID as
    done only when it appears in *both* files.

    If either file is missing we fall back to the IDs present in whichever
    file does exist, so a partial run (e.g. OneGov written but PR crashed)
    will still retry the missing half for those IDs.
    """

    def _read_id_col(path: str, col: str) -> set[str]:
        ids: set[str] = set()
        if not os.path.exists(path):
            return ids
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or col not in reader.fieldnames:
                return ids
            for row in reader:
                v = (row.get(col) or "").strip()
                if v:
                    ids.add(v)
        return ids

    onegov_done = _read_id_col(
        os.path.join(out_dir, "licence_details.csv"), "licenceID"
    )
    # Public register uses "licenceId" (lowercase 'd')
    pr_done = _read_id_col(
        os.path.join(out_dir, "public_register_details.csv"), "licenceId"
    )

    if onegov_done and pr_done:
        done = onegov_done & pr_done  # fully complete only
    else:
        done = onegov_done | pr_done  # at least one source present

    logger.info(
        "Resume check — OneGov done: %d, PR done: %d, intersection: %d",
        len(onegov_done),
        len(pr_done),
        len(done),
    )
    return done


# ---------------------------------------------------------------------------
# Core fetcher
# ---------------------------------------------------------------------------


def fetch_all(
    client: HBCClient,
    licence_ids: list[str],
    out_dir: str,
    request_delay: float = 0.5,
    pr_delay: float = PUBLIC_REGISTER_REQUEST_DELAY,
    max_workers: int = 10,
) -> dict:
    """
    Fetch OneGov details() and public_register_details() for each licenceID,
    saving results incrementally after every record.

    Returns a summary dict with counts: total, success, skipped, failed.

    Interrupt safety
    ~~~~~~~~~~~~~~~~
    The global _shutdown event is checked before each new task is submitted.
    In-flight requests are allowed to complete so their results are written
    before the process exits.  On the next run, already-written IDs are
    skipped automatically.
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_lock = threading.Lock()  # serialise CSV writes across threads
    pr_lock = threading.Lock()  # serialise public-register HTTP requests

    success = 0
    failed = 0
    skipped = 0
    total = len(licence_ids)

    def _fetch_one(lid: str) -> tuple[str, bool]:
        """Fetch one licenceID from both sources. Returns (lid, ok)."""
        if _shutdown.is_set():
            return lid, False  # don't start new work after interrupt

        time.sleep(request_delay)
        ok = True

        # ── OneGov details() (concurrent) ───────────────────────────────────
        try:
            detail = client.details(lid)
            with csv_lock:
                save_details_csv(detail, out_dir, licence_id=lid)
        except HBCAPIError as e:
            if e.status_code == 404:
                logger.warning(
                    "details(%r): no record (404) — skipping OneGov half.", lid
                )
            elif e.status_code == 408:
                logger.warning(
                    "details(%r): traffic limit (408) — will retry next run.", lid
                )
                ok = False
            else:
                logger.error("details(%r) failed: %s", lid, e)
                ok = False
        except Exception as e:
            logger.error("details(%r) unexpected error: %s", lid, e)
            ok = False

        # ── Public register (serialised) ────────────────────────────────────
        with pr_lock:
            if _shutdown.is_set():
                time.sleep(pr_delay)
                return lid, False

            try:
                pr_data = public_register_details(lid)
                with csv_lock:
                    save_public_register_csv(pr_data, out_dir)
            except HBCAPIError as e:
                if e.status_code == 404:
                    logger.warning(
                        "public_register_details(%r): not found (404) — skipping PR half.",
                        lid,
                    )
                elif e.status_code == 429:
                    logger.error(
                        "public_register_details(%r): still 429 after retries — will retry next run.",
                        lid,
                    )
                    ok = False
                else:
                    logger.error("public_register_details(%r) failed: %s", lid, e)
                    ok = False
            except Exception as e:
                logger.error("public_register_details(%r) unexpected error: %s", lid, e)
                ok = False
            finally:
                time.sleep(pr_delay)

        return lid, ok

    # ── Thread pool ─────────────────────────────────────────────────────────
    with tqdm(
        total=total,
        desc="Fetching details",
        unit="licence",
        colour="green",
        dynamic_ncols=True,
    ) as bar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Submit only if not already shutting down
            futures = {}
            for lid in licence_ids:
                if _shutdown.is_set():
                    logger.warning(
                        "Shutdown signalled — not submitting remaining tasks."
                    )
                    break
                futures[pool.submit(_fetch_one, lid)] = lid

            for future in as_completed(futures):
                lid = futures[future]
                bar.set_postfix_str(
                    f"{lid[-12:]:<12}"
                )  # fixed width, stops bar resizing
                bar.update(1)
                try:
                    _, ok = future.result()
                    if ok:
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error("Unhandled exception for %r: %s", lid, e)
                    failed += 1

            if _shutdown.is_set():
                bar.set_postfix_str("interrupted")
                pool.shutdown(wait=True)  # let in-flight finish writing

    # Count IDs that were never submitted due to early shutdown
    submitted = len(futures)
    skipped = total - submitted

    return {
        "total": total,
        "submitted": submitted,
        "success": success,
        "failed": failed,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Resumable HBC details fetcher. "
            "Reads licences_raw.csv (and optional extras), skips already-fetched "
            "licenceIDs, then hits OneGov + public register for everything remaining."
        )
    )
    p.add_argument(
        "--input",
        default="./output/licences_raw.csv",
        help="Primary browse CSV (licences_raw.csv). Default: ./output/licences_raw.csv",
    )
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="CSV",
        help="Additional browse CSVs to merge (any CSV with a licenceID column).",
    )
    p.add_argument(
        "--outdir",
        default="./output",
        help="Directory to write / append details CSVs. Default: ./output",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Concurrent OneGov detail workers. Default: 10",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between requests per worker. Default: 0.5",
    )
    p.add_argument(
        "--pr-delay",
        type=float,
        default=PUBLIC_REGISTER_REQUEST_DELAY,
        help=f"Seconds between public register requests. Default: {PUBLIC_REGISTER_REQUEST_DELAY}",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output CSVs and re-fetch everything.",
    )
    return p.parse_args()


def main():
    load_dotenv()

    args = parse_args()

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        print(
            "ERROR: API_KEY and API_SECRET must be set in your environment or .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Collect licenceIDs from input CSVs ───────────────────────────────────
    all_input_files = [args.input] + list(args.extra or [])
    raw_ids: list[str] = []
    seen: set[str] = set()

    for path in all_input_files:
        for lid in read_licence_ids_from_csv(path):
            if lid not in seen:
                seen.add(lid)
                raw_ids.append(lid)

    if not raw_ids:
        print("No licenceIDs found in the supplied input files. Nothing to do.")
        sys.exit(0)

    logger.info("Total unique licenceIDs from input: %d", len(raw_ids))

    # ── Resume: filter out already-processed IDs ────────────────────────────
    if args.no_resume:
        pending = raw_ids
        logger.info("--no-resume set: fetching all %d licenceIDs.", len(pending))
    else:
        done = already_fetched_ids(args.outdir)
        pending = [lid for lid in raw_ids if lid not in done]
        logger.info(
            "Already done: %d  |  Remaining: %d  |  Total: %d",
            len(done),
            len(pending),
            len(raw_ids),
        )

    if not pending:
        print("All licenceIDs have already been fetched. Nothing to do.")
        sys.exit(0)

    # ── Authenticate ─────────────────────────────────────────────────────────
    client = HBCClient(api_key=api_key, api_secret=api_secret)
    client.ensure_token()
    logger.info("Token acquired: %s", client.token_status)

    # ── Fetch ────────────────────────────────────────────────────────────────
    print(
        f"\nFetching details for {len(pending):,} licenceID(s) "
        f"({len(raw_ids) - len(pending):,} already done).\n"
        f"Output directory: {os.path.abspath(args.outdir)}\n"
        f"Workers: {args.workers}  |  Delay: {args.delay}s  |  PR delay: {args.pr_delay}s\n"
        f"Ctrl+C at any time — progress is saved after every record.\n"
    )

    summary = fetch_all(
        client=client,
        licence_ids=pending,
        out_dir=args.outdir,
        request_delay=args.delay,
        pr_delay=args.pr_delay,
        max_workers=args.workers,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    interrupted = _shutdown.is_set()
    print(
        f"\n{'[INTERRUPTED] ' if interrupted else ''}Done.\n"
        f"  Total to fetch : {summary['total']:>6,}\n"
        f"  Submitted      : {summary['submitted']:>6,}\n"
        f"  Success        : {summary['success']:>6,}\n"
        f"  Failed         : {summary['failed']:>6,}\n"
        f"  Not submitted  : {summary['skipped']:>6,}\n"
    )
    if interrupted or summary["failed"] > 0:
        print(
            "Re-run the script to pick up failed/skipped records — "
            "completed records will be skipped automatically."
        )

    sys.exit(0 if not interrupted and summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
