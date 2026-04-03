"""crates.io documentation fetcher."""
import json
import urllib.request
import urllib.error
from typing import Optional
from datetime import datetime, timezone
from .base import PackageDoc

CRATES_USER_AGENT = "neural-memory-mcp (https://github.com/Yakoub-ai/neural-memory)"


class CratesFetcher:
    registry_name = "crates"

    def supports(self, package_name: str) -> bool:
        # Rust crate names: lowercase, hyphens/underscores, no slashes
        return (
            "/" not in package_name
            and not package_name.startswith("@")
            and not package_name.startswith("std")
        )

    def fetch(self, package_name: str) -> Optional[PackageDoc]:
        url = f"https://crates.io/api/v1/crates/{package_name}"
        req = urllib.request.Request(url, headers={"User-Agent": CRATES_USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            crate = data.get("crate", {})
            return PackageDoc(
                package_name=package_name,
                registry="crates",
                version=crate.get("max_stable_version") or crate.get("max_version", ""),
                summary=crate.get("description", "") or "",
                description=crate.get("description", "") or "",
                homepage_url=crate.get("homepage", "") or "",
                doc_url=crate.get("documentation", "") or f"https://docs.rs/{package_name}",
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            return None
