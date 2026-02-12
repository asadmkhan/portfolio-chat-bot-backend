import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.taxonomy.local_taxonomy import LocalTaxonomy  # noqa: E402


class TaxonomyTests(unittest.TestCase):
    def test_synonym_normalization_resolves_canonical_id(self):
        taxonomy = LocalTaxonomy()
        normalized, canonical_id = taxonomy.normalize_skill("Client Management")
        self.assertEqual(normalized, "client management")
        self.assertEqual(canonical_id, "skill_stakeholder_mgmt")


if __name__ == "__main__":
    unittest.main()

