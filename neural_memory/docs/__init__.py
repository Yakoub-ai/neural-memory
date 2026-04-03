from .registry import register_fetcher
from .pypi import PyPIFetcher
from .npm import NpmFetcher
from .gopkg import GoPkgFetcher
from .crates import CratesFetcher

register_fetcher(PyPIFetcher())
register_fetcher(NpmFetcher())
register_fetcher(GoPkgFetcher())
register_fetcher(CratesFetcher())
