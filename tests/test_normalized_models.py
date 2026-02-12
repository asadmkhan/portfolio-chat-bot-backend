import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.normalized import (  # noqa: E402
    EvidenceSpan,
    JDRequirement,
    NormalizedJD,
    NormalizedResume,
    ResumeClaim,
)


class NormalizedModelsTests(unittest.TestCase):
    def test_can_instantiate_normalized_resume_and_jd(self):
        resume = NormalizedResume(
            source_language="en",
            claims=[
                ResumeClaim(
                    claim_id="c1",
                    text="Reduced API latency by 20%",
                    evidence=EvidenceSpan(doc_id="resume-1", line_start=3, line_end=3),
                )
            ],
        )
        jd = NormalizedJD(
            source_language="en",
            title="Backend Engineer",
            requirements=[
                JDRequirement(
                    req_id="r1",
                    text="Must have Python experience",
                    priority="must",
                    evidence=EvidenceSpan(doc_id="jd-1", line_start=2, line_end=2),
                )
            ],
        )

        self.assertEqual(resume.claims[0].claim_id, "c1")
        self.assertEqual(jd.requirements[0].priority, "must")


if __name__ == "__main__":
    unittest.main()

