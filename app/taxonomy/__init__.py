from functools import lru_cache

from .local_taxonomy import LocalTaxonomy
from .provider import TaxonomyProvider


@lru_cache(maxsize=1)
def get_default_taxonomy_provider() -> TaxonomyProvider:
    return LocalTaxonomy()


__all__ = ["TaxonomyProvider", "LocalTaxonomy", "get_default_taxonomy_provider"]

