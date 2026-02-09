import os
import unittest

from fastapi.testclient import TestClient

from app.core.tools_rate_limit import clear_tools_rate_limit_events
from app.main import app


class RecruiterApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TOOLS_LLM_ENABLED"] = "0"
        os.environ["TOOLS_STRICT_LLM"] = "0"
        cls.client = TestClient(app)

    def setUp(self):
        clear_tools_rate_limit_events()
        self.common = {"locale": "en", "session_id": "session-abcdef123"}
        self.resume_text = (
            "Senior Backend Engineer\n"
            "- Built Python microservices for payments used by 1.2M users.\n"
            "- Reduced API latency by 38% and cut infra costs by $42,000 per year.\n"
            "- Led migration from monolith to event-driven architecture.\n"
        )
        self.jd_text = (
            "We need a Senior Backend Engineer with Python, distributed systems, and cloud architecture experience. "
            "Must collaborate with product and improve reliability metrics."
        )

    def test_resume_authenticity_contract(self):
        response = self.client.post(
            "/v1/recruiter/resume-authenticity",
            json={**self.common, "resume_text": self.resume_text, "jd_text": self.jd_text},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["risk_level"], {"Low", "Medium", "High"})
        self.assertTrue(body["overall_summary"])
        self.assertIsInstance(body["signals"], list)
        self.assertIsInstance(body["disclaimers"], list)

    def test_claim_verification_contract(self):
        response = self.client.post(
            "/v1/recruiter/claim-verification",
            json={**self.common, "resume_text": self.resume_text, "jd_text": self.jd_text},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["summary"])
        self.assertIsInstance(body["claims"], list)
        self.assertGreaterEqual(len(body["claims"]), 1)
        first = body["claims"][0]
        self.assertIn("claim_text", first)
        self.assertIn("questions", first)
        self.assertGreaterEqual(len(first["questions"]), 3)

    def test_jd_quality_contract(self):
        response = self.client.post(
            "/v1/recruiter/jd-quality",
            json={**self.common, "jd_text": self.jd_text},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["rating"], {"Clear", "Risky", "Problematic"})
        self.assertTrue(body["summary"])
        self.assertIn("role_level_inference", body)
        self.assertIsInstance(body["issues"], list)

    def test_ats_vs_human_contract(self):
        response = self.client.post(
            "/v1/recruiter/ats-vs-human",
            json={**self.common, "resume_text": self.resume_text, "jd_text": self.jd_text},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("ats_risk", body)
        self.assertIn("human_risk", body)
        self.assertIn("quick_wins", body)
        self.assertIn("where_to_focus", body)

    def test_resume_compare_contract(self):
        response = self.client.post(
            "/v1/recruiter/resume-compare",
            json={
                **self.common,
                "jd_text": self.jd_text,
                "resumes": [
                    {"candidate_label": "A", "resume_text": self.resume_text},
                    {
                        "candidate_label": "B",
                        "resume_text": (
                            "Backend Engineer\n"
                            "- Maintained APIs and production services.\n"
                            "- Worked with teams to ship features.\n"
                        ),
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["comparison_summary"])
        self.assertIsInstance(body["candidates"], list)
        self.assertEqual(len(body["candidates"]), 2)

    def test_resume_compare_requires_two_candidates(self):
        response = self.client.post(
            "/v1/recruiter/resume-compare",
            json={
                **self.common,
                "jd_text": self.jd_text,
                "resumes": [{"candidate_label": "A", "resume_text": self.resume_text}],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_share_create_and_get(self):
        create_response = self.client.post(
            "/v1/recruiter/share",
            json={
                "tool_slug": "resume-authenticity",
                "locale": "en",
                "result_payload": {"risk_level": "Medium", "overall_summary": "Sample"},
            },
        )
        self.assertEqual(create_response.status_code, 200)
        create_body = create_response.json()
        self.assertIn("share_id", create_body)
        share_id = create_body["share_id"]

        get_response = self.client.get(f"/v1/recruiter/share/{share_id}")
        self.assertEqual(get_response.status_code, 200)
        get_body = get_response.json()
        self.assertEqual(get_body["tool_slug"], "resume-authenticity")
        self.assertEqual(get_body["locale"], "en")
        self.assertIn("result_payload", get_body)

    def test_share_unknown_tool_rejected(self):
        response = self.client.post(
            "/v1/recruiter/share",
            json={
                "tool_slug": "unknown-tool",
                "locale": "en",
                "result_payload": {"foo": "bar"},
            },
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

