import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features.domain_classifier import classify_domain_from_jd  # noqa: E402
from app.schemas.normalized import EvidenceSpan, JDRequirement, NormalizedJD  # noqa: E402


class DomainClassifierTests(unittest.TestCase):
    def test_sales_jd_classifies_as_sales(self):
        jd = NormalizedJD(
            source_language="en",
            title="Senior Sales Account Executive",
            requirements=[
                JDRequirement(
                    req_id="r1",
                    text="Must own sales pipeline and hit quota targets",
                    priority="must",
                    evidence=EvidenceSpan(doc_id="jd-sales", line_start=1, line_end=1),
                ),
                JDRequirement(
                    req_id="r2",
                    text="Experience with CRM, prospecting, and deal closing",
                    priority="must",
                    evidence=EvidenceSpan(doc_id="jd-sales", line_start=2, line_end=2),
                ),
            ],
        )

        classification = classify_domain_from_jd(jd)
        self.assertEqual(classification.domain_primary, "sales")
        self.assertGreater(classification.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()

