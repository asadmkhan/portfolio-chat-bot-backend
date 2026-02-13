import unittest
import os
from io import BytesIO
from unittest.mock import patch
from types import SimpleNamespace
from zipfile import ZipFile

# Keep API tests deterministic and fast by default.
os.environ.setdefault("TOOLS_LLM_ENABLED", "0")
os.environ.setdefault("TOOLS_STRICT_LLM", "0")

from fastapi.testclient import TestClient

from app.core.tools_rate_limit import clear_tools_rate_limit_events
from app.main import app


class ToolsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TOOLS_LLM_ENABLED"] = "0"
        cls.client = TestClient(app)
        cls.payload = {
            "locale": "en",
            "resume_text": (
                "John Doe\n"
                "Email john@example.com\n"
                "Phone +1 555 222 1111\n"
                "Python React SQL Docker AWS\n"
                "Built APIs for SaaS products and improved response time by 35%."
            ),
            "job_description_text": (
                "We need a Python backend engineer with SQL, Docker, and cloud experience. "
                "3+ years experience required."
            ),
            "candidate_profile": {"target_region": "US", "seniority": "mid"},
            "session_id": "session-abcdef123",
        }

    def setUp(self):
        clear_tools_rate_limit_events()

    def _get_ats_check(self, body: dict, category_id: str, check_id: str) -> dict:
        report = body.get("details", {}).get("ats_report", {})
        categories = report.get("categories", [])
        for category in categories:
            if category.get("id") != category_id:
                continue
            for check in category.get("checks", []):
                if check.get("id") == check_id:
                    return check
        raise AssertionError(f"ATS check not found: {category_id}/{check_id}")

    def test_job_match_contract_shape(self):
        response = self.client.post("/v1/tools/job-match", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn(body["recommendation"], {"apply", "fix", "skip"})
        self.assertIsInstance(body["confidence"], float)
        self.assertIn("job_match", body["scores"])
        self.assertIn("ats_readability", body["scores"])
        self.assertIsInstance(body["risks"], list)
        self.assertIsInstance(body["fix_plan"], list)
        self.assertIn("generated_at", body)

    def test_cover_letter_localized_modes(self):
        payload = dict(self.payload)
        payload["locale"] = "de"
        response = self.client.post("/v1/tools/cover-letter", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("letters", body["details"])
        self.assertIn("mode_labels", body["details"])
        self.assertEqual(body["details"]["mode_labels"]["hr"], "HR-Modus")

    def test_missing_keywords_localized_guidance(self):
        payload = dict(self.payload)
        payload["locale"] = "es"
        response = self.client.post("/v1/tools/missing-keywords", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        suggestions = body["details"]["insertion_suggestions"]
        self.assertTrue(len(suggestions) > 0)
        self.assertNotIn("Add '", suggestions[0]["guidance"])

    def test_tools_rate_limit_returns_429(self):
        status_codes = []
        for _ in range(22):
            response = self.client.post("/v1/tools/job-match", json=self.payload)
            status_codes.append(response.status_code)
        self.assertIn(429, status_codes)

    def test_lead_capture(self):
        response = self.client.post(
            "/v1/tools/lead-capture",
            json={
                "locale": "en",
                "session_id": "session-abcdef123",
                "email": "hello@example.com",
                "tool": "job-match",
                "consent": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["message"])

    def test_additional_tool_contract_shape(self):
        response = self.client.post("/v1/tools/resume-score", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["recommendation"], {"apply", "fix", "skip"})
        self.assertIn("score_breakdown", body["details"])
        self.assertIn("overall", body["details"]["score_breakdown"])

    def test_resume_optimization_report_has_quality_features(self):
        response = self.client.post("/v1/tools/resume-optimization-report", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("bullet_quality_analyzer", body["details"])
        self.assertIn("keyword_stuffing_detector", body["details"])
        self.assertIn("resume_credibility_score", body["details"])
        self.assertIn("merged_features", body["details"])

    def test_legacy_resume_tool_slug_maps_to_merged_report(self):
        response = self.client.post("/v1/tools/resume-score", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["details"]["tool"], "resume-optimization-report")
        self.assertEqual(body["details"]["requested_tool"], "resume-score")
        self.assertIn("score_breakdown", body["details"])
        self.assertIn("merged_features", body["details"])

    def test_ats_checker_contains_sectioned_report(self):
        response = self.client.post("/v1/tools/ats-checker", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        report = body["details"].get("ats_report") or {}
        self.assertIn("overall_score", report)
        self.assertIn("total_issues", report)
        self.assertIn("tier_scores", report)
        categories = report.get("categories") or []
        self.assertEqual(len(categories), 5)
        category_ids = {item.get("id") for item in categories}
        self.assertEqual(category_ids, {"content", "format", "skills_suggestion", "resume_sections", "style"})
        self.assertTrue(categories[0].get("checks"))

    def test_ats_checker_stream_emits_progress_stages_and_result(self):
        with self.client.stream(
            "POST",
            "/v1/tools/ats-checker/stream",
            json=self.payload,
            headers={"accept": "text/event-stream"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            stream_text = "".join(chunk for chunk in response.iter_text())

        self.assertIn("event: progress", stream_text)
        self.assertIn('"stage": "parsing_resume"', stream_text)
        self.assertIn('"stage": "analyzing_experience"', stream_text)
        self.assertIn('"stage": "extracting_skills"', stream_text)
        self.assertIn('"stage": "generating_recommendations"', stream_text)
        self.assertIn('"stage": "completed"', stream_text)
        self.assertIn("event: result", stream_text)

    def test_ats_quantifying_impact_excludes_dates_and_phone_noise(self):
        payload = dict(self.payload)
        payload["resume_text"] = (
            "Jane Doe\n"
            "Email jane@example.com\n"
            "Phone +49 152 34748915\n"
            "Senior Engineer\n"
            "2019-2024 Senior Engineer at Acme\n"
            "- Led backend migrations across regions\n"
            "- Built monitoring and deployment workflows\n"
            "- Partnered with product team for roadmap delivery\n"
        )
        response = self.client.post("/v1/tools/ats-checker", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        check = self._get_ats_check(body, "content", "quantifying_impact")
        self.assertGreater(check.get("issues", 0), 0)
        metrics = check.get("metrics") or {}
        self.assertGreaterEqual(metrics.get("excluded_numeric_tokens", 0), 1)
        self.assertTrue(check.get("issue_examples"))

    def test_ats_quantifying_impact_green_has_pass_rationale_and_metrics(self):
        payload = dict(self.payload)
        payload["resume_text"] = (
            "John Doe\n"
            "Email john@example.com\n"
            "Phone +1 555 222 1111\n"
            "Senior Backend Engineer\n"
            "- Reduced API latency by 38% across 120+ endpoints.\n"
            "- Cut cloud costs by $52,000 annually via autoscaling redesign.\n"
            "- Increased release throughput 2.4x while maintaining 99.95% uptime.\n"
            "- Improved conversion by 14% through checkout service refactor.\n"
            "- Scaled event processing to 6M messages/day with zero data loss.\n"
        )
        response = self.client.post("/v1/tools/ats-checker", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        check = self._get_ats_check(body, "content", "quantifying_impact")
        self.assertEqual(check.get("issues"), 0)
        self.assertTrue(check.get("pass_reasons"))
        metrics = check.get("metrics") or {}
        self.assertGreaterEqual(metrics.get("quantified_ratio", 0), 0.45)

    def test_ats_repetition_detects_near_duplicates(self):
        payload = dict(self.payload)
        payload["resume_text"] = (
            "John Doe\n"
            "Senior Engineer\n"
            "- Built scalable REST APIs for checkout reliability and performance.\n"
            "- Built scalable REST APIs to improve checkout reliability and performance.\n"
            "- Built scalable REST APIs focused on checkout reliability and performance.\n"
            "- Improved observability across services.\n"
        )
        response = self.client.post("/v1/tools/ats-checker", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        check = self._get_ats_check(body, "content", "repetition")
        metrics = check.get("metrics") or {}
        self.assertGreaterEqual(metrics.get("near_duplicate_pairs", 0), 1)
        self.assertTrue(check.get("issue_examples"))

    def test_ats_spelling_grammar_has_concrete_issue_examples(self):
        payload = dict(self.payload)
        payload["resume_text"] = (
            "john doe\n"
            "i built teh payment service,, and recieve alerts quickly.\n"
            "managed   pipelines across enviroment updates.\n"
            "responsiblity included incident followups.\n"
        )
        response = self.client.post("/v1/tools/ats-checker", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        check = self._get_ats_check(body, "content", "spelling_grammar")
        self.assertGreater(check.get("issues", 0), 0)
        issue_examples = check.get("issue_examples") or []
        self.assertTrue(issue_examples)
        first = issue_examples[0]
        self.assertTrue(first.get("text"))
        self.assertTrue(first.get("reason"))
        self.assertTrue(first.get("suggestion"))

    def test_every_ats_check_has_issue_or_pass_explanation(self):
        response = self.client.post("/v1/tools/ats-checker", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        report = body.get("details", {}).get("ats_report", {})
        categories = report.get("categories", [])
        for category in categories:
            for check in category.get("checks", []):
                issues = int(check.get("issues", 0))
                if issues > 0:
                    self.assertTrue(
                        check.get("issue_examples"),
                        msg=f"Missing issue examples for check {category.get('id')}/{check.get('id')}",
                    )
                else:
                    self.assertTrue(
                        check.get("pass_reasons"),
                        msg=f"Missing pass reasons for check {category.get('id')}/{check.get('id')}",
                    )
                self.assertTrue(check.get("rationale"), msg=f"Missing rationale for check {category.get('id')}/{check.get('id')}")

    def test_summarizer_contract_shape(self):
        response = self.client.post(
            "/v1/tools/summarize",
            json={
                "locale": "en",
                "source_type": "text",
                "content": (
                    "This release improves keyword grouping and decision output. "
                    "Next, review the changes with the hiring team."
                ),
                "mode": "summary",
                "output_language": "default",
                "length": "medium",
                "session_id": "session-abcdef123",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["summary"])
        self.assertIsInstance(body["key_points"], list)
        self.assertIsInstance(body["action_items"], list)
        self.assertIn("generated_at", body)

    def test_extract_text_from_upload(self):
        response = self.client.post(
            "/v1/tools/extract-text",
            files={"file": ("resume.txt", b"Python developer with ATS optimization experience.", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source_type"], "text")
        self.assertIn("Python developer", body["text"])
        self.assertIn("layout_profile", body["details"])
        self.assertIn("preview_meta", body["details"])

    def test_extract_text_preserves_line_breaks_and_layout_profile(self):
        response = self.client.post(
            "/v1/tools/extract-text",
            files={
                "file": (
                    "resume.txt",
                    b"Jane Doe\nSenior Engineer\nBuilt APIs that reduced latency by 35%\nLed migration to cloud\n",
                    "text/plain",
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("\n", body["text"])
        layout = body["details"].get("layout_profile") or {}
        self.assertIn(layout.get("detected_layout"), {"single_column", "hybrid", "multi_column", "unknown"})
        self.assertGreaterEqual(layout.get("column_count", 0), 1)

    def test_extract_text_from_docx_fallback(self):
        docx_bytes = BytesIO()
        with ZipFile(docx_bytes, "w") as archive:
            archive.writestr(
                "word/document.xml",
                (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    "<w:body><w:p><w:r><w:t>Docx fallback extraction works</w:t></w:r></w:p></w:body></w:document>"
                ),
            )
        response = self.client.post(
            "/v1/tools/extract-text",
            files={
                "file": (
                    "resume.docx",
                    docx_bytes.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source_type"], "word")
        self.assertIn("Docx fallback extraction works", body["text"])

    def test_extract_text_from_pptx_fallback(self):
        pptx_bytes = BytesIO()
        with ZipFile(pptx_bytes, "w") as archive:
            archive.writestr(
                "ppt/slides/slide1.xml",
                (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                    "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Pptx fallback extraction works</a:t></a:r></a:p>"
                    "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
                ),
            )
        response = self.client.post(
            "/v1/tools/extract-text",
            files={
                "file": (
                    "slides.pptx",
                    pptx_bytes.getvalue(),
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source_type"], "ppt")
        self.assertIn("Pptx fallback extraction works", body["text"])

    def test_extract_text_rejects_oversized_file(self):
        from app.api.v1.tools import MAX_UPLOAD_BYTES

        oversized_content = b"x" * (MAX_UPLOAD_BYTES + 1)
        response = self.client.post(
            "/v1/tools/extract-text",
            files={"file": ("huge.txt", oversized_content, "text/plain")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("too large", response.json()["detail"].lower())

    def test_extract_text_rejects_unsupported_file_type(self):
        response = self.client.post(
            "/v1/tools/extract-text",
            files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported", response.json()["detail"].lower())

    def test_extract_text_rejects_signature_mismatch(self):
        response = self.client.post(
            "/v1/tools/extract-text",
            files={"file": ("resume.pdf", b"plain text pretending to be a pdf", "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("signature", response.json()["detail"].lower())

    def test_extract_text_rejects_legacy_doc_with_conversion_message(self):
        response = self.client.post(
            "/v1/tools/extract-text",
            files={"file": ("legacy.doc", b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1", "application/msword")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("convert to .docx", response.json()["detail"].lower())

    def test_extract_job_from_url_success(self):
        html = (
            "<html><head><title>Backend Engineer</title></head><body>"
            "<main><h1>Senior Backend Engineer</h1><p>Build Python APIs with Docker and AWS.</p>"
            "<p>Requirements: 5+ years, SQL, cloud.</p></main></body></html>"
        )
        fake_response = SimpleNamespace(
            status_code=200,
            text=html,
            url="https://jobs.example.com/positions/123",
        )

        with patch("httpx.Client.get", return_value=fake_response), patch(
            "app.services.tools_service._readability_job_extract",
            return_value={
                "title": "Senior Backend Engineer",
                "description": (
                    "Build Python APIs with Docker and AWS in a production SaaS environment. "
                    "Partner with product managers and platform teams to deliver resilient services. "
                    "Requirements include 5+ years of backend engineering, SQL, cloud operations, and API design."
                ),
            },
        ):
            response = self.client.post(
                "/v1/tools/extract-job",
                json={
                    "locale": "en",
                    "session_id": "session-abcdef123",
                    "job_url": "https://jobs.example.com/positions/123",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["domain"], "jobs.example.com")
        self.assertFalse(body["blocked"])
        self.assertGreater(body["characters"], 20)
        self.assertTrue(body["job_description_text"])
        self.assertIn(body["extraction_mode"], {"json_ld", "domain_parser", "readability"})

    def test_extract_job_from_url_blocked_page(self):
        fake_response = SimpleNamespace(
            status_code=403,
            text="<html><body>Sign in to continue. Enable JavaScript.</body></html>",
            url="https://jobs.example.com/protected",
        )
        with patch("httpx.Client.get", return_value=fake_response):
            response = self.client.post(
                "/v1/tools/extract-job",
                json={
                    "locale": "en",
                    "session_id": "session-abcdef123",
                    "job_url": "https://jobs.example.com/protected",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["blocked"])
        self.assertTrue(body["warnings"])

    def test_extract_resume_from_url_success(self):
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        fake_response = SimpleNamespace(
            status_code=200,
            text="",
            content=pdf_bytes,
            headers={"content-type": "application/pdf"},
            url="https://files.example.com/resume.pdf",
        )
        with patch("httpx.Client.get", return_value=fake_response), patch(
            "app.services.tools_service.extract_text_from_file",
            return_value=SimpleNamespace(
                filename="resume.pdf",
                source_type="pdf",
                text="Extracted resume text from URL",
                characters=31,
                details={"layout_profile": {"detected_layout": "single_column"}},
            ),
        ):
            response = self.client.post(
                "/v1/tools/extract-resume-url",
                json={
                    "locale": "en",
                    "session_id": "session-abcdef123",
                    "resume_url": "https://files.example.com/resume.pdf",
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["domain"], "files.example.com")
        self.assertEqual(body["filename"], "resume.pdf")
        self.assertEqual(body["resume_text"], "Extracted resume text from URL")
        self.assertFalse(body["blocked"])
        self.assertTrue(body.get("content_base64"))

    def test_extract_resume_from_url_rejects_private_hosts(self):
        response = self.client.post(
            "/v1/tools/extract-resume-url",
            json={
                "locale": "en",
                "session_id": "session-abcdef123",
                "resume_url": "http://127.0.0.1/resume.pdf",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("private or local", response.json()["detail"].lower())

    def test_strict_llm_mode_returns_quality_error_when_llm_unavailable(self):
        with patch.dict(
            os.environ,
            {"TOOLS_STRICT_LLM": "true", "TOOLS_LLM_ENABLED": "0"},
            clear=False,
        ):
            response = self.client.post("/v1/tools/job-match", json=self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertIn("AI quality mode", response.json()["detail"])

    def test_strict_llm_mode_keeps_ats_checker_deterministic(self):
        with patch.dict(
            os.environ,
            {"TOOLS_STRICT_LLM": "true", "TOOLS_LLM_ENABLED": "0"},
            clear=False,
        ):
            response = self.client.post("/v1/tools/ats-checker", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        report = body.get("details", {}).get("ats_report", {})
        content_checks = []
        for category in report.get("categories") or []:
            if category.get("id") == "content":
                content_checks = category.get("checks") or []
                break
        grammar = next((item for item in content_checks if item.get("id") == "spelling_grammar"), {})
        metrics = grammar.get("metrics") or {}
        self.assertEqual(metrics.get("validation_mode"), "deterministic")

    def test_ats_layout_profile_impacts_score(self):
        base_payload = dict(self.payload)
        base_payload["candidate_profile"] = {"target_region": "US", "seniority": "mid"}

        single_payload = {
            **base_payload,
            "resume_layout_profile": {
                "detected_layout": "single_column",
                "column_count": 1,
                "confidence": 0.9,
                "table_count": 0,
                "header_link_density": 0.1,
                "complexity_score": 22,
                "source_type": "pdf",
                "signals": ["single_column_detected"],
            },
            "resume_file_meta": {"filename": "resume.pdf", "extension": "pdf", "source_type": "pdf"},
        }
        multi_payload = {
            **base_payload,
            "resume_layout_profile": {
                "detected_layout": "multi_column",
                "column_count": 2,
                "confidence": 0.9,
                "table_count": 2,
                "header_link_density": 0.7,
                "complexity_score": 74,
                "source_type": "pdf",
                "signals": ["multi_column_pattern_detected"],
            },
            "resume_file_meta": {"filename": "resume.pdf", "extension": "pdf", "source_type": "pdf"},
        }

        single_response = self.client.post("/v1/tools/ats-checker", json=single_payload)
        multi_response = self.client.post("/v1/tools/ats-checker", json=multi_payload)
        self.assertEqual(single_response.status_code, 200)
        self.assertEqual(multi_response.status_code, 200)

        single_body = single_response.json()
        multi_body = multi_response.json()
        self.assertLess(multi_body["scores"]["ats_readability"], single_body["scores"]["ats_readability"])
        multi_fit = multi_body["details"].get("layout_fit_for_target") or {}
        self.assertIn(multi_fit.get("fit_level"), {"moderate", "poor"})

    def test_ats_layout_region_weighting(self):
        layout_profile = {
            "detected_layout": "multi_column",
            "column_count": 2,
            "confidence": 0.88,
            "table_count": 1,
            "header_link_density": 0.5,
            "complexity_score": 68,
            "source_type": "pdf",
            "signals": ["multi_column_pattern_detected"],
        }

        us_payload = {
            **dict(self.payload),
            "candidate_profile": {"target_region": "US", "seniority": "mid"},
            "resume_layout_profile": layout_profile,
            "resume_file_meta": {"filename": "resume.pdf", "extension": "pdf", "source_type": "pdf"},
        }
        eu_payload = {
            **dict(self.payload),
            "candidate_profile": {"target_region": "EU", "seniority": "mid"},
            "resume_layout_profile": layout_profile,
            "resume_file_meta": {"filename": "resume.pdf", "extension": "pdf", "source_type": "pdf"},
        }

        us_response = self.client.post("/v1/tools/ats-checker", json=us_payload)
        eu_response = self.client.post("/v1/tools/ats-checker", json=eu_payload)
        self.assertEqual(us_response.status_code, 200)
        self.assertEqual(eu_response.status_code, 200)
        us_body = us_response.json()
        eu_body = eu_response.json()
        self.assertLessEqual(us_body["scores"]["ats_readability"], eu_body["scores"]["ats_readability"])

    def test_ats_does_not_force_multicol_for_weak_profile(self):
        payload = dict(self.payload)
        payload["resume_text"] = (
            "Jane Doe | Senior Backend Engineer | Berlin\n"
            "Email jane@example.com\n"
            "Phone +49 152 34748915\n"
            "- Built APIs that improved response times by 28%.\n"
            "- Reduced cloud cost by 17% through autoscaling updates.\n"
        )
        payload["resume_layout_profile"] = {
            "detected_layout": "multi_column",
            "column_count": 2,
            "confidence": 0.54,
            "table_count": 0,
            "header_link_density": 0.1,
            "complexity_score": 26,
            "source_type": "text",
            "signals": [],
        }
        response = self.client.post("/v1/tools/ats-checker", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        recommendation = (body.get("details", {}).get("format_recommendation") or "").lower()
        self.assertNotIn("multi-column resume detected", recommendation)
        parse_check = self._get_ats_check(body, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        self.assertNotEqual((metrics.get("layout_type") or "").lower(), "multi column")

    def test_ats_pipe_separator_header_does_not_trigger_table_or_multicol_flags(self):
        payload = dict(self.payload)
        payload["resume_text"] = (
            "ASAD MATEEN KHAN | Senior Full Stack Developer | AI Engineer | .NET | Node.js\n"
            "Email contact@example.com\n"
            "Phone +49 15234748915\n"
            "- Improved release cycle by 32%.\n"
            "- Reduced incident recovery time by 41%.\n"
        )
        response = self.client.post("/v1/tools/ats-checker", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        parsing_flags = body.get("details", {}).get("parsing_flags") or []
        lowered = " ".join(flag.lower() for flag in parsing_flags)
        self.assertNotIn("table", lowered)
        self.assertNotIn("column", lowered)

    def test_ats_parse_rate_metrics_use_scalar_reason_summary_and_reasoned_examples(self):
        payload = dict(self.payload)
        payload["resume_text"] = (
            "Jane Doe\n"
            "Email jane@example.com\n"
            "Phone +1 555 100 2200\n"
            "Skill | Level | Years\n"
            "Python | Advanced | 8\n"
            "SQL | Intermediate | 5\n"
        )
        response = self.client.post("/v1/tools/ats-checker", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        parse_check = self._get_ats_check(body, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        self.assertIsInstance(metrics.get("parsing_penalty_reason_count"), int)
        self.assertIsInstance(metrics.get("parsing_penalty_reason_titles"), str)
        reasons = metrics.get("parsing_penalty_reasons") or []
        self.assertTrue(all(isinstance(item, str) for item in reasons))
        reason_details = metrics.get("parsing_penalty_reasons_detail") or []
        self.assertTrue(all(isinstance(item, dict) for item in reason_details))
        issue_examples = parse_check.get("issue_examples") or []
        self.assertTrue(issue_examples)
        self.assertFalse(
            any(
                (item.get("reason") or "").strip().lower() == "issue detected by ats heuristic evaluation."
                for item in issue_examples
            )
        )

    def test_vpn_tool_endpoint(self):
        response = self.client.post(
            "/v1/tools/vpn/is-my-vpn-working",
            json={
                "locale": "en",
                "session_id": "session-abcdef123",
                "input": {
                    "expected_country": "Germany",
                    "detected_country": "Germany",
                    "ip_changed": True,
                    "dns_leak": False,
                    "webrtc_leak": False,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["tool"], "is-my-vpn-working")
        self.assertIn(body["verdict"], {"good", "attention", "critical"})
        self.assertIsInstance(body["cards"], list)
        self.assertTrue(len(body["actions"]) >= 1)

    def test_vpn_probe_enrich_endpoint(self):
        response = self.client.post(
            "/v1/tools/vpn/probe-enrich",
            json={
                "locale": "en",
                "session_id": "session-abcdef123",
                "public_ip": "8.8.8.8",
                "webrtc_ips": ["8.8.4.4", "192.168.1.2"],
                "dns_resolver_ips": ["1.1.1.1"],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("records", body)
        self.assertIsInstance(body["records"], list)
        self.assertTrue(len(body["records"]) >= 1)
        self.assertIn("summary", body)

    def test_vpn_quiz_recommendations_shape(self):
        response = self.client.post(
            "/v1/tools/vpn/find-best-vpn-for-me-quiz",
            json={
                "locale": "en",
                "session_id": "session-abcdef123",
                "input": {
                    "country": "UAE",
                    "use_case": "privacy",
                    "budget": "medium",
                    "devices": 3,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["tool"], "find-best-vpn-for-me-quiz")
        recommendations = body["details"].get("recommendations") or []
        self.assertGreaterEqual(len(recommendations), 2)
        self.assertTrue(recommendations[0].get("provider"))
        self.assertTrue(recommendations[0].get("reason"))
        self.assertIn(body["details"].get("generation_scope"), {"heuristic", "full-analysis"})

    def test_new_standalone_tools_contract_shape(self):
        slugs = [
            "job-application-roi-calculator",
            "seniority-calibration-tool",
            "rejection-reason-classifier",
            "cv-region-translator",
        ]
        for slug in slugs:
            response = self.client.post(
                f"/v1/tools/{slug}",
                json={
                    **self.payload,
                    "tool_inputs": {
                        "target_role": "Senior Backend Engineer",
                        "country": "Germany",
                        "years_experience": 8,
                        "minutes_per_application": 35,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn("details", body)
            self.assertEqual(body["details"]["tool"], slug)

    @patch("app.services.tools_service._youtube_transcript", return_value="")
    @patch(
        "app.services.tools_service._youtube_metadata_text",
        return_value="Video title: Test title. Channel: Test channel. Transcript was unavailable.",
    )
    def test_youtube_summarizer_uses_metadata_fallback(self, *_mocks):
        response = self.client.post(
            "/v1/tools/summarize",
            json={
                "locale": "en",
                "source_type": "youtube",
                "content": "",
                "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "mode": "summary",
                "output_language": "default",
                "length": "short",
                "session_id": "session-abcdef123",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["summary"])
        self.assertTrue(body["metadata"].get("youtube_fallback_used"))


if __name__ == "__main__":
    unittest.main()
