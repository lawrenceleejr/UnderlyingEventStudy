"""I/O helpers: discover CMS Open Data files and open them remotely with uproot."""
from __future__ import annotations

import functools
import os
import time
from typing import List

import requests
import uproot

from . import config


@functools.lru_cache(maxsize=8)
def list_files(record: int, https: bool = True) -> tuple:
    """Return the list of ROOT-file URLs for a CERN Open Data record.

    Uses the opendata.cern.ch records API to read the file index, then rewrites
    each eospublic URI to the reachable HTTPS front-end.
    """
    api = f"https://opendata.cern.ch/api/records/{record}"
    md = requests.get(api, timeout=60, verify=config.CA_BUNDLE).json()["metadata"]
    uris: List[str] = []
    for fi in md.get("_file_indices", []):
        for f in fi.get("files", []):
            uri = f.get("uri", "")
            if uri.endswith(".root"):
                uris.append(config.to_https(uri) if https else uri)
    return tuple(uris)


def download(url: str, dest_dir=None, chunk=4 << 20, retries=4) -> str:
    """Download a remote ROOT file to a local cache (idempotent). Returns path.

    Reading large jagged branches over many concurrent HTTP range requests is
    flaky against the opendata front-end, so the skim downloads whole files
    once, reads them locally, and deletes them to respect the disk budget.
    """
    dest_dir = dest_dir or (config.DATA / "cache")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(url))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp = path + ".part"
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120, verify=config.CA_BUNDLE) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for blk in r.iter_content(chunk_size=chunk):
                        fh.write(blk)
            os.replace(tmp, path)
            return path
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return path


def open_events(url: str):
    """Open the Events TTree of a (possibly remote) NanoAOD/PFNano file."""
    return uproot.open(url, ssl=config.ssl_context())["Events"]


def iterate_events(url: str, branches, step_size="50 MB"):
    """Yield awkward batches of the requested branches from a remote file."""
    f = uproot.open(url, ssl=config.ssl_context())
    yield from f["Events"].iterate(branches, step_size=step_size)
