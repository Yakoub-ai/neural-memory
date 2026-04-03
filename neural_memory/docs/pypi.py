"""PyPI documentation fetcher."""
import json
import urllib.request
import urllib.error
from typing import Optional
from datetime import datetime, timezone
from .base import PackageDoc


class PyPIFetcher:
    registry_name = "pypi"

    def supports(self, package_name: str) -> bool:
        # PyPI packages: no slashes, no @ prefix
        return "/" not in package_name and not package_name.startswith("@")

    def fetch(self, package_name: str) -> Optional[PackageDoc]:
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            info = data.get("info", {})
            desc = info.get("description", "") or ""
            return PackageDoc(
                package_name=package_name,
                registry="pypi",
                version=info.get("version", ""),
                summary=info.get("summary", ""),
                description=desc[:5000],
                homepage_url=info.get("home_page", "") or (info.get("project_urls") or {}).get("Homepage", ""),
                doc_url=(info.get("project_urls") or {}).get("Documentation", ""),
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            return None
