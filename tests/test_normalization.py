import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.normalize.normalize_jd import normalize_jd  # noqa: E402
from app.normalize.normalize_resume import normalize_resume  # noqa: E402
from app.parsing.models import ParsedDoc  # noqa: E402


class NormalizationTests(unittest.TestCase):
    def test_resume_normalization_creates_stable_claim_ids(self):
        parsed = ParsedDoc(
            doc_id="resume-doc",
            source_type="txt",
            language="en",
            text=(
                "Jane Doe\n"
                "- Built APIs for payments\n"
                "* Reduced latency by 30%\n"
                "3. Led migration to cloud\n"
            ),
        )

        normalized = normalize_resume(parsed)
        self.assertEqual([claim.claim_id for claim in normalized.claims], ["c1", "c2", "c3"])

    def test_jd_normalization_creates_must_and_nice_requirements(self):
        parsed = ParsedDoc(
            doc_id="jd-doc",
            source_type="txt",
            language="en",
            text=(
                "Senior Backend Engineer\n"
                "Requirements:\n"
                "- Must have Python experience\n"
                "- Nice to have Kubernetes knowledge\n"
                "- Required cloud experience\n"
            ),
        )

        normalized = normalize_jd(parsed)
        priorities = [req.priority for req in normalized.requirements]
        self.assertIn("must", priorities)
        self.assertIn("nice", priorities)


if __name__ == "__main__":
    unittest.main()

