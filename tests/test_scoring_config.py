import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config.scoring import get_scoring_config, get_scoring_value


class ScoringConfigTests(unittest.TestCase):
    def test_loader_and_value_lookup(self):
        config = get_scoring_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(get_scoring_value("matching.weights.must_have"), 0.75)


if __name__ == "__main__":
    unittest.main()
