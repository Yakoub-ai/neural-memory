"""npm registry documentation fetcher."""
import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional
from datetime import datetime, timezone
from .base import PackageDoc


class NpmFetcher:
    registry_name = "npm"

    def supports(self, package_name: str) -> bool:
        # npm: scoped (@scope/name) or simple names
        return package_name.startswith("@") or (
            "/" not in package_name and not package_name.startswith("std/")
        )

    def fetch(self, package_name: str) -> Optional[PackageDoc]:
        encoded = package_name.replace("/", "%2F")
        url = f"https://registry.npmjs.org/{encoded}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            version = (data.get("dist-tags") or {}).get("latest", "")
            desc = data.get("description", "") or ""
            readme = data.get("readme", "") or ""
            return PackageDoc(
                package_name=package_name,
                registry="npm",
                version=version,
                summary=desc[:200],
                description=(desc + "\n\n" + readme)[:5000],
                homepage_url=data.get("homepage", "") or "",
                doc_url="",
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            return None
