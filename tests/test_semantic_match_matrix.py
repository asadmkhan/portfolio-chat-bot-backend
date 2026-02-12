import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config.scoring import get_scoring_value  # noqa: E402
from app.schemas.normalized import (  # noqa: E402
    EvidenceSpan,
    JDRequirement,
    MatchMatrix,
    NormalizedJD,
    NormalizedResume,
    ResumeClaim,
)
from app.semantic.match_matrix import build_match_matrix  # noqa: E402


class SemanticMatchMatrixTests(unittest.TestCase):
    def test_match_matrix_builds_with_must_and_nice_ids(self):
        jd = NormalizedJD(
            source_language="en",
            title="Account Manager",
            requirements=[
                JDRequirement(
                    req_id="r1",
                    text="Experience with stakeholder management is required",
                    priority="must",
                    evidence=EvidenceSpan(doc_id="jd-1", line_start=1, line_end=1),
                ),
                JDRequirement(
                    req_id="r2",
                    text="Nice to have SEO campaign experience",
                    priority="nice",
                    evidence=EvidenceSpan(doc_id="jd-1", line_start=2, line_end=2),
                ),
            ],
        )
        resume = NormalizedResume(
            source_language="en",
            claims=[
                ResumeClaim(
                    claim_id="c1",
                    text="Owned client management for enterprise accounts and renewals.",
                    evidence=EvidenceSpan(doc_id="resume-1", line_start=5, line_end=5),
                ),
                ResumeClaim(
                    claim_id="c2",
                    text="Delivered SEO campaign strategy with 15% traffic growth.",
                    evidence=EvidenceSpan(doc_id="resume-1", line_start=8, line_end=8),
                ),
            ],
        )

        matrix = build_match_matrix(jd=jd, resume=resume)
        self.assertIsInstance(matrix, MatchMatrix)
        self.assertEqual(matrix.must_req_ids, ["r1"])
        self.assertEqual(matrix.nice_req_ids, ["r2"])

        weak_threshold = float(get_scoring_value("matching.similarity_thresholds.weak_match", 0.55))
        top_hit_for_r1 = matrix.matches["r1"][0]
        self.assertEqual(top_hit_for_r1.claim_id, "c1")
        self.assertGreaterEqual(top_hit_for_r1.similarity, weak_threshold)
        self.assertGreaterEqual(top_hit_for_r1.evidence_strength, 1)


if __name__ == "__main__":
    unittest.main()

