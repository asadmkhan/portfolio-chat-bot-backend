import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.main  # noqa: F401
from app.core.config.scoring import get_scoring_value


class PipelineSmokeTests(unittest.TestCase):
    def test_safe_imports_and_scoring_config_lookup(self):
        self.assertEqual(get_scoring_value("matching.weights.must_have"), 0.75)


if __name__ == "__main__":
    unittest.main()

