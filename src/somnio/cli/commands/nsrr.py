"""NSRR (National Sleep Research Resource) dataset download commands."""

from __future__ import annotations

import time
from pathlib import Path


from somnio.utils.imports import MissingOptionalDependency

import typer
from loguru import logger

try:
    import requests
    from requests.adapters import HTTPAdapter
    from tqdm import tqdm
    from urllib3.util.retry import Retry
except ModuleNotFoundError as e:
    if e.name not in ("requests", "tqdm", "urllib3"):
        raise
    raise MissingOptionalDependency(
        e.name, extra="nsrr", purpose="NSRR download"
    ) from e


DEFAULT_HTTP_RETRIES = 6
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY_SECONDS = 10
DOWNLOAD_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
)


def _build_session(http_retries: int = DEFAULT_HTTP_RETRIES) -> "requests.Session":
    """Create a requests Session with retries for transient NSRR issues.

    NSRR occasionally returns 502/503/504; we retry those with exponential backoff.
    """
    retry = Retry(
        total=http_retries,
        connect=http_retries,
        read=http_retries,
        status=http_retries,
        backoff_factor=1.0,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_directory_listing(
    session: "requests.Session",
    slug: str,
    token: str,
    path: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list:
    """Fetch one level of the dataset file listing from the NSRR API."""
    base_url = f"https://sleepdata.org/api/v1/datasets/{slug}/files.json"
    params: dict[str, str] = {"auth_token": token}
    if path:
        params["path"] = path
    response = session.get(base_url, params=params, timeout=timeout_seconds)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        snippet = (response.text or "")[:500].replace("\n", " ").strip()
        raise requests.HTTPError(
            f"{e} (status={response.status_code}) url={response.url} body_snippet={snippet!r}"
        ) from e
    return response.json()


def _collect_all_files(
    session: "requests.Session",
    slug: str,
    token: str,
    path: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list:
    """Recursively collect all file entries under the given path."""
    files: list[dict] = []
    items = _fetch_directory_listing(
        session, slug, token, path, timeout_seconds=timeout_seconds
    )
    for item in items:
        if item["is_file"]:
            files.append(item)
        else:
            files += _collect_all_files(
                session, slug, token, item["full_path"], timeout_seconds=timeout_seconds
            )
    return files


def _download_file(
    session: "requests.Session",
    slug: str,
    token: str,
    file_obj: dict,
    base_dir: Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Download a single file from NSRR, creating parent dirs and skipping if complete."""
    download_url = (
        f"https://sleepdata.org/datasets/{slug}/files/{file_obj['full_path']}"
        f"?auth_token={token}"
    )
    local_path = base_dir / file_obj["full_path"]
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        expected_size = file_obj.get("size")
        if expected_size is not None:
            if local_path.stat().st_size == expected_size:
                logger.debug("Skipping (exists, size match): {}", file_obj["full_path"])
                return
        else:
            logger.debug("Skipping (exists): {}", file_obj["full_path"])
            return

    logger.debug("Downloading {}...", file_obj["full_path"])
    with session.get(download_url, stream=True, timeout=timeout_seconds) as r:
        r.raise_for_status()
        with local_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.debug("Saved to {}", local_path)


def download(
    slug: str = typer.Argument(..., help="Dataset slug (e.g., sof, shhs, mesa)."),
    output_dir: Path = typer.Argument(
        ...,
        path_type=Path,
        help="Output directory; files are written to OUTPUT_DIR/SLUG/.",
    ),
    token: str | None = typer.Option(
        ...,
        "--token",
        "-t",
        envvar="NSRR_TOKEN",
        help="NSRR auth token.",
    ),
    path: str = typer.Option(
        None,
        "--path",
        "-p",
        help="Subpath to download (e.g., polysomnography). Default: entire dataset.",
    ),
    timeout_seconds: float = typer.Option(
        DEFAULT_TIMEOUT_SECONDS,
        "--timeout-seconds",
        help="Per-request timeout in seconds.",
    ),
    download_retries: int = typer.Option(
        DEFAULT_DOWNLOAD_RETRIES,
        "--download-retries",
        help="Retries per file on connection/read timeout.",
    ),
    http_retries: int = typer.Option(
        DEFAULT_HTTP_RETRIES,
        "--http-retries",
        help="Retries for HTTP requests.",
    ),
) -> None:
    """Download files from an NSRR dataset."""

    if not token:
        logger.error("Missing NSRR token. Set NSRR_TOKEN in .env or pass --token.")
        raise SystemExit(1)

    session = _build_session(http_retries=http_retries)
    target_path = path.strip() if path else None
    all_files = _collect_all_files(
        session, slug, token, target_path, timeout_seconds=timeout_seconds
    )
    logger.info("Found {} files under {!r}.", len(all_files), target_path or "(root)")

    out = output_dir / slug
    for file_obj in tqdm(all_files, desc="Downloading", unit="file"):
        for attempt in range(1, download_retries + 1):
            try:
                _download_file(
                    session,
                    slug,
                    token,
                    file_obj,
                    out,
                    timeout_seconds=timeout_seconds,
                )
                break
            except DOWNLOAD_RETRY_EXCEPTIONS as e:
                if attempt == download_retries:
                    raise

                delay = attempt * DOWNLOAD_RETRY_DELAY_SECONDS
                tqdm.write(
                    f"Timeout/connection error, retrying in {delay}s... "
                    f"(attempt {attempt}/{download_retries}, {e!r})"
                )
                time.sleep(delay)
