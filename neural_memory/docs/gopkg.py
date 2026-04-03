"""Go module proxy documentation fetcher."""
import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional
from datetime import datetime, timezone
from .base import PackageDoc


class GoPkgFetcher:
    registry_name = "go"

    def supports(self, package_name: str) -> bool:
        # Go import paths contain at least one /
        return "/" in package_name and not package_name.startswith("@")

    def fetch(self, package_name: str) -> Optional[PackageDoc]:
        encoded = urllib.parse.quote(package_name, safe="")
        version = ""
        try:
            proxy_url = f"https://proxy.golang.org/{encoded}/@latest"
            with urllib.request.urlopen(proxy_url, timeout=10) as resp:
                proxy_data = json.loads(resp.read())
            version = proxy_data.get("Version", "")
        except Exception:
            pass

        return PackageDoc(
            package_name=package_name,
            registry="go",
            version=version,
            summary=f"Go package: {package_name}",
            description="",
            homepage_url=f"https://pkg.go.dev/{package_name}",
            doc_url=f"https://pkg.go.dev/{package_name}",
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
