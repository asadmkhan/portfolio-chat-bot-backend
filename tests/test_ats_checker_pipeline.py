import sys
import tempfile
import unittest
import json
import re
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.models import ParsedBlock, ParsedDoc  # noqa: E402
from app.parsing.parse import parse_document  # noqa: E402
from app.normalize.normalize_resume import normalize_resume  # noqa: E402
from app.schemas.tools import ATSCheckerOutput, ResumeLayoutProfile, ToolRequest  # noqa: E402
from app.services.tools_service import run_ats_checker  # noqa: E402


class ATSCheckerPipelineTests(unittest.TestCase):
    WRITER_LIKE_RESUME_TEXT = (
        "Freelance Content Writer\n"
        "Email writer@example.com\n"
        "Phone +1 555 111 2222\n"
        "Objective\n"
        "A green card holder with over four years of experience in online content writing and print media.\n"
        "Work Experience\n"
        "- Research, create and edit SEO blogs for a variety of websites.\n"
        "- Worked closely with content editors to adhere to strict deadlines.\n"
        "- Wrote content for clients on multiple social networking sites.\n"
        "- Wrote original and engaging search-engine-optimized content for digital marketing initiatives.\n"
        "- Researched web for related information and relevant keywords for SEO quality.\n"
        "- Framed content for web pages to explain services and products clearly.\n"
        "- Created promotional articles for industry websites.\n"
        "- Researched and edited SEO blogs for multiple sites and audiences.\n"
        "- Worked with content editors on publishing timelines and revisions.\n"
        "- Wrote social content and campaign copy for distributed channels.\n"
        "- Built article drafts with audience-first structure and clear CTA placement.\n"
        "- Delivered copy packages in US and UK writing styles.\n"
        "Skills\n"
        "Macintosh & PC platforms, Photoshop, Adobe Acrobat Pro, content management systems\n"
        "Education\n"
        "M.A. in Political Science\n"
    )
    WRITER_LIKE_JD_TEXT = (
        "Qualifications\n"
        "Must-Have\n"
        "Strong familiarity with professional software tools including Windows, MacOS, and Linux.\n"
        "Detail-oriented and capable of following precise instructions.\n"
        "Comfortable working independently and meeting tight deadlines.\n"
        "Access to a physical Mac and ability to create a fresh macOS user profile if required.\n"
        "Prior experience with data collection, annotation, or QA work.\n"
        "Experience recording or documenting workflows.\n"
        "Role Responsibilities\n"
        "Record screen sessions demonstrating specific tasks with clear verbal narration.\n"
        "Annotate screenshots by drawing bounding boxes around relevant UI elements.\n"
        "Follow staging instructions and use a custom capture tool to record workflows.\n"
    )

    @staticmethod
    def _get_check(response, category_id: str, check_id: str) -> dict:
        report = (response.details or {}).get("ats_report") or {}
        for category in report.get("categories") or []:
            if category.get("id") != category_id:
                continue
            for check in category.get("checks") or []:
                if check.get("id") == check_id:
                    return check
        raise AssertionError(f"check not found: {category_id}/{check_id}")

    def test_ats_checker_temp_txt_pipeline_returns_stable_blockers(self):
        content = (
            "JOHN DOE\n"
            "SUMMARY\n"
            "| Skill | Level |\n"
            "| Python | Advanced |\n"
            "EXPERIENCE:\n"
            "- Improved release quality by 20%.\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(content)
            tmp.close()

            payload = ToolRequest(
                locale="en",
                session_id="session-abcdef123",
                resume_text="placeholder fallback text",
                job_description_text="Backend engineer role",
                tool_inputs={"uploaded_resume_path": str(tmp_path)},
                resume_layout_profile=ResumeLayoutProfile(
                    detected_layout="multi_column",
                    column_count=2,
                    confidence=0.9,
                    table_count=1,
                    header_link_density=0.8,
                    source_type="text",
                    signals=["icon_detected"],
                ),
            )

            response = run_ats_checker(payload)
            ats_payload = response.details.get("ats_checker")
            self.assertIsNotNone(ats_payload)

            ats_output = ATSCheckerOutput.model_validate(ats_payload)
            blocker_ids = [blocker.id for blocker in ats_output.blockers]
            self.assertEqual(
                blocker_ids,
                ["multi_column", "tables", "icons_graphics", "missing_contact", "heading_inconsistency"],
            )
            self.assertEqual(ats_output.ats_risk_level, "medium")
            self.assertFalse(ats_output.needs_user_input)
            self.assertTrue(all(blocker.evidence.spans or blocker.evidence.claim_ids for blocker in ats_output.blockers))
            legacy_report = response.details.get("ats_report") or {}
            self.assertTrue(legacy_report.get("categories"))
            self.assertEqual(len(legacy_report.get("categories", [])), 5)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_ats_report_contains_frontend_required_checks_and_no_unavailable_status(self):
        content = (
            "JOHN DOE\n"
            "john@example.com | +1 555 111 2222\n"
            "SUMMARY\n"
            "Backend engineer with measurable delivery outcomes.\n"
            "EXPERIENCE\n"
            "- Reduced API latency by 35% across critical endpoints.\n"
            "- Improved deployment throughput by 2.1x using CI/CD automation.\n"
            "- Cut cloud costs by $30,000 annually with autoscaling redesign.\n"
            "- Increased conversion by 14% through checkout refactor.\n"
            "- Scaled event processing to 6M messages/day with zero data loss.\n"
            "- Led cross-functional incident response and reduced MTTR by 41%.\n"
            "SKILLS\n"
            "Python, SQL, Docker, AWS, FastAPI\n"
            "EDUCATION\n"
            "BSc Computer Science\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(content)
            tmp.close()

            payload = ToolRequest(
                locale="en",
                session_id="session-ats-report-0001",
                resume_text=content,
                job_description_text=(
                    "We need a Python backend engineer with SQL, Docker, AWS, communication skills, "
                    "and measurable impact in production systems."
                ),
                candidate_profile={"target_region": "US", "seniority": "mid"},
                tool_inputs={"uploaded_resume_path": str(tmp_path)},
            )
            response = run_ats_checker(payload)
            details = response.details or {}
            normalized_summary = details.get("normalized_resume") or {}
            self.assertGreater(normalized_summary.get("claim_count", 0), 0)
            self.assertIn("domain_classification", details)
            self.assertIn("analysis_units_summary", details)

            report = details.get("ats_report") or {}
            categories = report.get("categories") or []
            category_map = {item.get("id"): item for item in categories}
            self.assertEqual(set(category_map.keys()), {"content", "format", "skills_suggestion", "resume_sections", "style"})

            def check_ids(category_id: str) -> set[str]:
                checks = (category_map.get(category_id) or {}).get("checks") or []
                return {check.get("id") for check in checks}

            self.assertIn("file_format_size", check_ids("format"))
            self.assertIn("resume_length", check_ids("format"))
            self.assertIn("long_bullet_points", check_ids("format"))
            self.assertIn("essential_sections", check_ids("resume_sections"))
            self.assertIn("design", check_ids("style"))

            for category in categories:
                for check in category.get("checks") or []:
                    self.assertIn(check.get("status"), {"ok", "issue"})
                    if int(check.get("issues") or 0) > 0:
                        self.assertTrue(check.get("issue_examples"))
                    else:
                        self.assertTrue(check.get("pass_reasons"))

            format_checks = {check.get("id"): check for check in (category_map["format"].get("checks") or [])}
            resume_length = format_checks.get("resume_length") or {}
            metrics = resume_length.get("metrics") or {}
            self.assertGreater(metrics.get("word_count", 0), 0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_ats_checker_falls_back_to_resume_text_when_parser_returns_empty(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-ats-fallback-0001",
            resume_text=(
                "JOHN DOE\n"
                "Email john@example.com\n"
                "Phone +1 555 111 2222\n"
                "- Built APIs that improved response time by 35%.\n"
                "- Reduced infra cost by $22,000 annually.\n"
                "- Improved release cadence by 2x with automation.\n"
            ),
            job_description_text="Backend role needing Python, SQL, cloud, and measurable impact.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )

        with patch(
            "app.services.tools_service.parse_document",
            return_value=ParsedDoc(
                doc_id="empty-doc",
                source_type="txt",
                language=None,
                text="",
                blocks=[],
                parsing_warnings=["simulated parser failure"],
                layout_flags={},
            ),
        ):
            response = run_ats_checker(payload)

        normalized_summary = (response.details or {}).get("normalized_resume") or {}
        self.assertGreater(normalized_summary.get("claim_count", 0), 0)
        parsed_summary = (response.details or {}).get("parsed_doc") or {}
        self.assertGreater(parsed_summary.get("text_length", 0), 0)
        warnings = parsed_summary.get("parsing_warnings") or []
        self.assertTrue(any("fallback" in warning.lower() for warning in warnings))

    def test_pdf_fixture_parses_with_non_trivial_claim_coverage(self):
        fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "resumes" / "sample_resume.pdf"
        self.assertTrue(fixture_path.exists(), msg=f"Missing fixture: {fixture_path}")

        parsed = parse_document(str(fixture_path))
        normalized = normalize_resume(parsed)

        self.assertEqual(parsed.source_type, "pdf")
        self.assertGreater(len(parsed.text), 200)
        self.assertGreater(len(normalized.claims), 5)

    def test_grammar_fragment_detected_without_false_lowercase_start(self):
        content = (
            "A green card holder with 8+ years of experience in print media.\n"
            "Have completed nationwide campaign launches with measurable uplift.\n"
            "Work Experience\n"
            "- Led campaign analysis and delivery planning across stakeholders.\n"
            "Email jane@example.com\n"
            "Phone +1 555 222 3333\n"
        )
        payload = ToolRequest(
            locale="en",
            session_id="session-grammar-fragment-0001",
            resume_text=content,
            job_description_text="Marketing role requiring communication and campaign execution.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        grammar_check = self._get_check(response, "content", "spelling_grammar")
        issue_examples = grammar_check.get("issue_examples") or []

        fragment_hit = any(
            "fragment" in (example.get("reason") or "").lower()
            and "Have completed" in (example.get("text") or "")
            for example in issue_examples
        )
        lowercase_false_positive = any(
            "lowercase" in (example.get("reason") or "").lower()
            and "Have completed" in (example.get("text") or "")
            for example in issue_examples
        )
        self.assertTrue(fragment_hit)
        self.assertFalse(lowercase_false_positive)

    def test_wrapped_bullet_continuation_does_not_trigger_lowercase_start_issue(self):
        content = (
            "EXPERIENCE\n"
            "- Designed and built scalable services for payment reconciliation and\n"
            "managed cross-functional rollout across three regions.\n"
            "Email john@example.com\n"
            "Phone +1 555 111 2222\n"
        )
        payload = ToolRequest(
            locale="en",
            session_id="session-bullet-wrap-0001",
            resume_text=content,
            job_description_text="Backend role with distributed systems and service ownership.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        grammar_check = self._get_check(response, "content", "spelling_grammar")
        issue_examples = grammar_check.get("issue_examples") or []
        self.assertFalse(
            any(
                "lowercase" in (example.get("reason") or "").lower()
                and "managed cross-functional rollout" in (example.get("text") or "").lower()
                for example in issue_examples
            )
        )

    def test_contact_or_url_lines_do_not_trigger_lowercase_start_issue(self):
        content = (
            "JANE DOE\n"
            "linkedin.com/in/jane-doe\n"
            "github.com/jane\n"
            "Email jane@example.com\n"
            "Phone +1 555 111 2222\n"
            "- Led campaign delivery for multi-channel launches.\n"
        )
        payload = ToolRequest(
            locale="en",
            session_id="session-contact-lowercase-0001",
            resume_text=content,
            job_description_text="Content writer role requiring communication and campaign delivery.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        grammar_check = self._get_check(response, "content", "spelling_grammar")
        issue_examples = grammar_check.get("issue_examples") or []
        self.assertFalse(
            any(
                "lowercase" in (example.get("reason") or "").lower()
                and "linkedin.com" in (example.get("text") or "").lower()
                for example in issue_examples
            )
        )

    def test_unicode_spacing_artifacts_are_normalized_before_spacing_rule(self):
        content = (
            "ACME\u00adCorp\u00a0\u00a0Senior Engineer\u00a02020\u20132024\n"
            "Email jane@example.com\n"
            "Phone +1 555 444 3333\n"
            "- Improved reliability by 22%.\n"
        )
        payload = ToolRequest(
            locale="en",
            session_id="session-unicode-spacing-0001",
            resume_text=content,
            job_description_text="Engineering role requiring reliability and delivery metrics.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        grammar_check = self._get_check(response, "content", "spelling_grammar")
        issue_examples = grammar_check.get("issue_examples") or []
        self.assertFalse(any("repeated spacing" in (example.get("reason") or "").lower() for example in issue_examples))

    def test_issue_quality_non_zero_when_issue_examples_have_evidence(self):
        content = (
            "JOHN DOE\n"
            "Email john@example.com\n"
            "Phone +1 555 222 1111\n"
            "Have completed migration projects under strict timelines.\n"
            "Work Experience\n"
            "- Delivered release-readiness planning for platform migrations.\n"
        )
        payload = ToolRequest(
            locale="en",
            session_id="session-issue-quality-0001",
            resume_text=content,
            job_description_text="Program manager role requiring delivery ownership.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        report = (response.details or {}).get("ats_report") or {}
        tier_scores = report.get("tier_scores") or {}
        self.assertGreater(tier_scores.get("issue_impact_score", 0), 0)
        self.assertEqual(tier_scores.get("issue_quality"), tier_scores.get("issue_quality_score"))

        grammar_check = self._get_check(response, "content", "spelling_grammar")
        issue_examples = grammar_check.get("issue_examples") or []
        evidenced_examples = [
            example
            for example in issue_examples
            if isinstance(example.get("evidence"), dict)
            and isinstance((example.get("evidence") or {}).get("spans"), list)
            and len((example.get("evidence") or {}).get("spans")) > 0
        ]
        self.assertTrue(evidenced_examples)

    def test_hard_soft_term_denominator_consistency(self):
        payload_with_terms = ToolRequest(
            locale="en",
            session_id="session-terms-with-total-0001",
            resume_text=(
                "Email john@example.com\n"
                "Phone +1 555 000 1111\n"
                "Work Experience\n"
                "- Built Python engineer workflows with SQL and Docker delivery outcomes.\n"
            ),
            job_description_text="Python SQL Docker backend engineer role.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response_with_terms = run_ats_checker(payload_with_terms)
        hard_check = self._get_check(response_with_terms, "skills_suggestion", "hard_skills")
        hard_metrics = hard_check.get("metrics") or {}
        hard_match = re.search(r"Matched hard terms=(\d+)/(\d+)", hard_check.get("rationale") or "")
        self.assertIsNotNone(hard_match)
        self.assertEqual(int(hard_match.group(2)), hard_metrics.get("hard_terms_total"))
        self.assertEqual(int(hard_match.group(1)), hard_metrics.get("hard_terms_matched"))

        payload_zero_terms = ToolRequest(
            locale="en",
            session_id="session-terms-zero-total-0001",
            resume_text=(
                "Email jane@example.com\n"
                "Phone +1 555 777 8888\n"
                "Work Experience\n"
                "- Managed operations reporting and stakeholder coordination.\n"
            ),
            job_description_text="Role requires ownership, communication, and adaptability.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response_zero_terms = run_ats_checker(payload_zero_terms)
        hard_zero_check = self._get_check(response_zero_terms, "skills_suggestion", "hard_skills")
        hard_zero_metrics = hard_zero_check.get("metrics") or {}
        hard_zero_match = re.search(r"Matched hard terms=(\d+)/(\d+)", hard_zero_check.get("rationale") or "")
        self.assertIsNotNone(hard_zero_match)
        self.assertEqual(int(hard_zero_match.group(2)), hard_zero_metrics.get("hard_terms_total"))
        self.assertEqual(int(hard_zero_match.group(1)), hard_zero_metrics.get("hard_terms_matched"))

    def test_hard_skills_low_confidence_when_jd_is_insufficient(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-hard-skills-lowconf-0001",
            resume_text=(
                "Email john@example.com\n"
                "Phone +1 555 110 2200\n"
                "Experience\n"
                "- Built backend services and scaled API throughput by 35%.\n"
                "- Improved deployment reliability with CI/CD automation.\n"
            ),
            job_description_text="N/A",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        hard_check = self._get_check(response, "skills_suggestion", "hard_skills")
        metrics = hard_check.get("metrics") or {}
        self.assertTrue(metrics.get("low_confidence"))
        self.assertEqual(metrics.get("hard_terms_total"), 0)
        self.assertEqual(metrics.get("display_denominator"), 0)
        self.assertLessEqual(int(hard_check.get("score") or 100), 79)
        pass_reasons = " ".join(hard_check.get("pass_reasons") or []).lower()
        self.assertIn("insufficient", pass_reasons)

    def test_writer_like_jd_produces_non_zero_hard_skill_denominator(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-writer-hard-terms-0001",
            resume_text=(
                "Email writer@example.com\n"
                "Phone +1 555 101 2020\n"
                "- Wrote long-form blog posts and website copy for campaigns.\n"
                "- Edited marketing drafts for clarity and tone consistency.\n"
                "- Collaborated with SEO and social teams on editorial calendars.\n"
            ),
            job_description_text=(
                "Requirements:\n"
                "- Content writing for web and print media\n"
                "- Copy editing and proofreading\n"
                "- SEO optimization and keyword research\n"
                "- Stakeholder communication\n"
            ),
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        hard_check = self._get_check(response, "skills_suggestion", "hard_skills")
        metrics = hard_check.get("metrics") or {}
        self.assertGreater(metrics.get("hard_terms_total", 0), 0)

    def test_grammar_issue_count_stability_upper_bound(self):
        fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "resumes" / "sample_resume.pdf"
        parsed = parse_document(str(fixture_path))
        payload = ToolRequest(
            locale="en",
            session_id="session-grammar-stability-0001",
            resume_text=parsed.text,
            job_description_text="Backend role needing cloud, SQL, and measurable impact.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
            tool_inputs={"uploaded_resume_path": str(fixture_path)},
        )
        response = run_ats_checker(payload)
        grammar_check = self._get_check(response, "content", "spelling_grammar")
        metrics = grammar_check.get("metrics") or {}
        self.assertLessEqual(metrics.get("validated_issues", 0), 8)

    def test_no_unavailable_payload_strings_for_normal_inputs(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-no-unavailable-0001",
            resume_text=(
                "Email ana@example.com\n"
                "Phone +1 555 678 1234\n"
                "Summary\n"
                "- Delivered analytics improvements by 18% in campaign performance.\n"
                "- Reduced reporting cycle time by 35% through automation.\n"
                "Skills\n"
                "SQL, Python, Tableau\n"
            ),
            job_description_text="Analytics role requiring SQL, reporting, and stakeholder communication.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        payload_text = json.dumps(response.model_dump(mode="json")).lower()
        self.assertNotIn("unavailable in the current payload", payload_text)

    def test_all_flagged_issue_examples_include_evidence_spans(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-issue-evidence-0001",
            resume_text=(
                "John Doe\n"
                "Phone +1 555 000 1111\n"
                "Have completed migrations for multiple systems.\n"
                "teh deployment process was updated for reliability improvements.\n"
            ),
            job_description_text="Backend role requiring ownership, communication, and measurable outcomes.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        report = (response.details or {}).get("ats_report") or {}

        for category in report.get("categories") or []:
            for check in category.get("checks") or []:
                if int(check.get("issues") or 0) <= 0:
                    continue
                for example in check.get("issue_examples") or []:
                    evidence = example.get("evidence") or {}
                    spans = evidence.get("spans") or []
                    self.assertTrue(spans, msg=f"missing evidence spans for {category.get('id')}/{check.get('id')}")
                    first_span = spans[0]
                    self.assertTrue(first_span.get("text_snippet"))
                    self.assertIn("line_start", first_span)
                    self.assertIn("line_end", first_span)

    def test_active_voice_examples_do_not_use_header_or_contact_lines(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-active-voice-scope-0001",
            resume_text=(
                "JANE DOE\n"
                "Email jane@example.com\n"
                "Phone +1 555 901 8888\n"
                "linkedin.com/in/jane-doe\n"
                "EXPERIENCE\n"
                "- Managed content workflows across editorial and product teams.\n"
                "- Built publication calendar and improved turnaround time.\n"
                "- Responsible for campaign messaging alignment across channels.\n"
            ),
            job_description_text="Writer role requiring collaboration and content delivery.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        active_voice = self._get_check(response, "style", "active_voice")
        issue_examples = active_voice.get("issue_examples") or []
        self.assertFalse(
            any(
                "email" in (example.get("text") or "").lower()
                or "phone" in (example.get("text") or "").lower()
                or "linkedin.com" in (example.get("text") or "").lower()
                for example in issue_examples
            )
        )

    def test_writer_like_fixture_active_voice_evidence_comes_from_experience_bullets(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-writer-active-evidence-0001",
            resume_text=self.WRITER_LIKE_RESUME_TEXT,
            job_description_text=self.WRITER_LIKE_JD_TEXT,
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        active_voice = self._get_check(response, "style", "active_voice")
        evidence_lines = [str(item) for item in (active_voice.get("evidence") or [])]
        self.assertTrue(evidence_lines)
        self.assertFalse(any("objective" in line.lower() for line in evidence_lines))
        self.assertFalse(any("email" in line.lower() or "phone" in line.lower() for line in evidence_lines))

    def test_writer_like_fixture_quantifying_scans_more_than_ten_bullets(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-writer-quant-bullets-0001",
            resume_text=self.WRITER_LIKE_RESUME_TEXT,
            job_description_text=self.WRITER_LIKE_JD_TEXT,
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        quantifying = self._get_check(response, "content", "quantifying_impact")
        metrics = quantifying.get("metrics") or {}
        self.assertGreater(metrics.get("experience_bullets_scanned", 0), 10)

    def test_writer_like_fixture_hard_skills_excludes_noise_terms(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-writer-hard-noise-0001",
            resume_text=self.WRITER_LIKE_RESUME_TEXT,
            job_description_text=self.WRITER_LIKE_JD_TEXT,
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        hard_skills = self._get_check(response, "skills_suggestion", "hard_skills")
        hard_payload_text = json.dumps(hard_skills).lower()
        self.assertNotIn('"text": "resume"', hard_payload_text)
        self.assertNotIn('"text": "access"', hard_payload_text)
        self.assertNotIn('"text": "comfort"', hard_payload_text)
        self.assertNotIn('"text": "profile"', hard_payload_text)
        self.assertNotIn("professional software tools including windows", hard_payload_text)
        self.assertIn("windows", hard_payload_text)
        self.assertIn("macos", hard_payload_text)
        metrics = hard_skills.get("metrics") or {}
        self.assertGreater(metrics.get("hard_terms_total", 0), 0)

    def test_objective_counts_as_summary_for_essential_sections(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-objective-summary-0001",
            resume_text=self.WRITER_LIKE_RESUME_TEXT,
            job_description_text=self.WRITER_LIKE_JD_TEXT,
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        essential_sections = self._get_check(response, "resume_sections", "essential_sections")
        metrics = essential_sections.get("metrics") or {}
        self.assertTrue(metrics.get("summary_present"))

    def test_writer_like_fixture_active_voice_scans_experience_bullets_only(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-writer-active-scope-0001",
            resume_text=self.WRITER_LIKE_RESUME_TEXT,
            job_description_text=self.WRITER_LIKE_JD_TEXT,
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        active_voice = self._get_check(response, "style", "active_voice")
        metrics = active_voice.get("metrics") or {}
        self.assertGreaterEqual(metrics.get("action_units_scanned", 0), 5)

    def test_tailored_title_is_low_confidence_when_jd_role_terms_missing(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-tailored-title-lowconf-0001",
            resume_text=(
                "ASAD MATEEN KHAN\n"
                "Senior Full Stack Engineer\n"
                "Email asad@example.com\n"
                "Phone +1 555 222 4444\n"
                "Work Experience\n"
                "- Led full-stack delivery across distributed teams.\n"
            ),
            job_description_text="N/A",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        tailored = self._get_check(response, "style", "tailored_title")
        metrics = tailored.get("metrics") or {}
        self.assertEqual(tailored.get("issues"), 0)
        self.assertTrue(metrics.get("title_check_low_confidence"))
        self.assertTrue(metrics.get("low_confidence"))
        self.assertLessEqual(int(tailored.get("score") or 100), 79)
        evidence = tailored.get("evidence") or []
        self.assertTrue(any("Senior Full Stack Engineer" in item for item in evidence))

    def test_parse_rate_score_and_penalty_stay_consistent(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-parse-penalty-consistency-0001",
            resume_text=(
                "JANE DOE\n"
                "Email jane@example.com\n"
                "Phone +1 555 300 4000\n"
                "Experience\n"
                "- Improved release reliability by 28%.\n"
                "- Reduced support incidents by 19%.\n"
            ),
            job_description_text="Backend role requiring service reliability and delivery ownership.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
            resume_layout_profile={
                "detected_layout": "multi_column",
                "column_count": 2,
                "confidence": 0.91,
                "table_count": 1,
                "header_link_density": 0.6,
                "complexity_score": 70,
                "source_type": "pdf",
                "signals": ["multi_column_pattern_detected"],
            },
            resume_file_meta={"filename": "resume.pdf", "extension": "pdf", "source_type": "pdf"},
        )
        response = run_ats_checker(payload)
        parse_check = self._get_check(response, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        parsing_penalty = int(metrics.get("parsing_penalty") or 0)
        score = int(parse_check.get("score") or 0)
        score_from_metrics = int(metrics.get("parse_rate_score") or 0)
        self.assertEqual(score, score_from_metrics)
        if score >= 90:
            self.assertLessEqual(parsing_penalty, int(metrics.get("max_penalty_for_great") or 5))

    def test_clean_single_column_resume_has_great_parse_rate_without_penalty_drift(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-parse-penalty-clean-0001",
            resume_text=(
                "Jane Doe\n"
                "Email jane@example.com\n"
                "Phone +1 555 123 9900\n"
                "Summary\n"
                "Backend engineer focused on reliable APIs.\n"
                "Experience\n"
                "- Reduced API latency by 35%.\n"
                "- Improved deployment stability by 25%.\n"
            ),
            job_description_text="Backend engineer role requiring API reliability and measurable impact.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
            resume_layout_profile={
                "detected_layout": "single_column",
                "column_count": 1,
                "confidence": 0.94,
                "table_count": 0,
                "header_link_density": 0.1,
                "complexity_score": 18,
                "source_type": "pdf",
                "signals": ["single_column_detected"],
            },
            resume_file_meta={"filename": "resume.pdf", "extension": "pdf", "source_type": "pdf"},
        )
        response = run_ats_checker(payload)
        parse_check = self._get_check(response, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        self.assertGreaterEqual(int(parse_check.get("score") or 0), 90)
        self.assertLessEqual(int(metrics.get("parsing_penalty") or 0), int(metrics.get("max_penalty_for_great") or 5))

    def test_header_pipe_contact_line_does_not_trigger_table_penalty(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-header-pipe-no-table-0001",
            resume_text=(
                "Asad Mateen Khan\n"
                "Bavaria, Germany | +49 15234748915 | asadmateenkhan@gmail.com\n"
                "Senior Software Engineer\n"
                "Experience\n"
                "- Improved release reliability by 28%.\n"
                "- Reduced support incidents by 19%.\n"
            ),
            job_description_text="Backend role requiring reliability and delivery ownership.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        parse_check = self._get_check(response, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        reason_details = metrics.get("parsing_penalty_reasons_detail") or []
        reason_ids = {item.get("id") for item in reason_details if isinstance(item, dict)}
        self.assertNotIn("table_like_structure", reason_ids)
        self.assertNotIn("table_count_penalty", reason_ids)

    def test_true_table_lines_still_trigger_table_penalty(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-true-table-penalty-0001",
            resume_text=(
                "Jane Doe\n"
                "Email jane@example.com\n"
                "Phone +1 555 111 2222\n"
                "Skills Matrix\n"
                "Skill | Level | Years\n"
                "Python | Advanced | 8\n"
                "SQL | Intermediate | 5\n"
                "Experience\n"
                "- Improved query performance by 32%.\n"
            ),
            job_description_text="Data role requiring SQL and measurable impact.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        parse_check = self._get_check(response, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        reason_details = metrics.get("parsing_penalty_reasons_detail") or []
        reason_ids = {item.get("id") for item in reason_details if isinstance(item, dict)}
        self.assertTrue({"table_like_structure", "table_count_penalty"} & reason_ids)
        self.assertGreater(int(parse_check.get("issues") or 0), 0)
        recommendation = (parse_check.get("recommendation") or "").lower()
        self.assertTrue("table" in recommendation or "pipe" in recommendation or "grid" in recommendation)

    def test_pdf_bbox_signal_can_promote_multi_column_layout(self):
        parsed_doc = ParsedDoc(
            doc_id="pdf-bbox-multi-001",
            source_type="pdf",
            language="en",
            text=(
                "John Doe\n"
                "Experience\n"
                "- Led backend modernization across distributed teams.\n"
                "- Reduced latency by 35% and improved reliability.\n"
            ),
            blocks=[
                ParsedBlock(page=1, bbox=[40.0, 70.0, 260.0, 95.0], text="Left col header"),
                ParsedBlock(page=1, bbox=[42.0, 108.0, 265.0, 132.0], text="Left col line 1"),
                ParsedBlock(page=1, bbox=[44.0, 145.0, 270.0, 168.0], text="Left col line 2"),
                ParsedBlock(page=1, bbox=[350.0, 72.0, 560.0, 96.0], text="Right col header"),
                ParsedBlock(page=1, bbox=[352.0, 110.0, 558.0, 134.0], text="Right col line 1"),
                ParsedBlock(page=1, bbox=[355.0, 148.0, 562.0, 172.0], text="Right col line 2"),
                ParsedBlock(page=2, bbox=[41.0, 70.0, 262.0, 94.0], text="Left p2 header"),
                ParsedBlock(page=2, bbox=[351.0, 72.0, 561.0, 96.0], text="Right p2 header"),
            ],
            parsing_warnings=[],
            layout_flags={},
        )

        payload = ToolRequest(
            locale="en",
            session_id="session-bbox-multi-column-0001",
            resume_text=parsed_doc.text,
            job_description_text="Software engineering role with ATS-heavy requirements.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        with patch("app.services.tools_service.parse_document", return_value=parsed_doc):
            response = run_ats_checker(payload)

        parse_check = self._get_check(response, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        self.assertEqual((metrics.get("layout_type") or "").lower(), "multi column")
        reason_details = metrics.get("parsing_penalty_reasons_detail") or []
        reason_ids = {item.get("id") for item in reason_details if isinstance(item, dict)}
        self.assertIn("multi_column_layout", reason_ids)
        layout_analysis = (response.details or {}).get("layout_analysis") or {}
        self.assertTrue(layout_analysis.get("pdf_bbox_multicolumn"))

    def test_parse_rate_issue_examples_are_reason_grounded(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-parse-reason-grounded-0001",
            resume_text=(
                "Jane Doe\n"
                "Email jane@example.com\n"
                "Phone +1 555 111 2222\n"
                "Skill | Level | Years\n"
                "Python | Advanced | 8\n"
                "SQL | Intermediate | 5\n"
            ),
            job_description_text="Data role requiring SQL and measurable impact.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        parse_check = self._get_check(response, "content", "ats_parse_rate")
        issue_examples = parse_check.get("issue_examples") or []
        self.assertTrue(issue_examples)
        self.assertFalse(
            any(
                (example.get("reason") or "").strip().lower() == "issue detected by ats heuristic evaluation."
                for example in issue_examples
            )
        )
        for example in issue_examples:
            evidence = example.get("evidence") or {}
            spans = evidence.get("spans") or []
            self.assertTrue(spans)
            self.assertTrue((spans[0] or {}).get("text_snippet"))

    def test_parse_rate_metrics_include_scalar_and_detail_reason_fields(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-parse-metrics-safe-0001",
            resume_text=(
                "Jane Doe\n"
                "Email jane@example.com\n"
                "Phone +1 555 111 2222\n"
                "Skill | Level | Years\n"
                "Python | Advanced | 8\n"
                "SQL | Intermediate | 5\n"
            ),
            job_description_text="Data role requiring SQL and measurable impact.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        parse_check = self._get_check(response, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        self.assertIsInstance(metrics.get("parsing_penalty_reason_count"), int)
        self.assertIsInstance(metrics.get("parsing_penalty_reason_titles"), str)
        reasons = metrics.get("parsing_penalty_reasons") or []
        self.assertTrue(all(isinstance(reason, str) for reason in reasons))
        reason_details = metrics.get("parsing_penalty_reasons_detail") or []
        self.assertTrue(all(isinstance(reason, dict) for reason in reason_details))

    def test_parse_rate_recommendation_matches_top_penalty_reason(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-parse-recommendation-consistency-0001",
            resume_text=(
                "Asad Mateen Khan\n"
                "https://portfolio.example.com\n"
                "https://github.com/asadmkhan\n"
                "https://linkedin.com/in/asadmkhan\n"
                "https://codedbyasad.com\n"
                "Email asad@example.com\n"
                "Phone +1 555 222 1111\n"
                "Senior Software Engineer\n"
                "Experience\n"
                "- Improved deployment reliability by 30%.\n"
            ),
            job_description_text="Backend role requiring reliability and delivery ownership.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
            resume_layout_profile={
                "detected_layout": "single_column",
                "column_count": 1,
                "confidence": 0.88,
                "table_count": 0,
                "header_link_density": 0.9,
                "complexity_score": 20,
                "source_type": "pdf",
                "signals": ["single_column_detected"],
            },
            resume_file_meta={"filename": "resume.pdf", "extension": "pdf", "source_type": "pdf"},
        )
        response = run_ats_checker(payload)
        parse_check = self._get_check(response, "content", "ats_parse_rate")
        metrics = parse_check.get("metrics") or {}
        reason_details = metrics.get("parsing_penalty_reasons_detail") or []
        reason_ids = {item.get("id") for item in reason_details if isinstance(item, dict)}
        self.assertIn("header_link_density", reason_ids)
        recommendation = (parse_check.get("recommendation") or "").lower()
        self.assertTrue("header" in recommendation or "link" in recommendation)

    def test_lowercase_continuation_clause_is_not_flagged_as_sentence_start_error(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-lowercase-continuation-0001",
            resume_text=(
                "Architecture + delivery: Designed and led the end-to-end delivery of a multi-tenant SAP Business One "
                "Web Client used by 50+\n"
                "companies to process thousands of transactions remotely additionally implemented pricing/tax/localization "
                "rules ensuring compliance across diverse EU and global markets, multi-language UI, and Crystal Reports layouts.\n"
                "Email asad@example.com\n"
                "Phone +1 555 111 7777\n"
            ),
            job_description_text="Enterprise software role requiring delivery ownership and system integration.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        grammar_check = self._get_check(response, "content", "spelling_grammar")
        issue_examples = grammar_check.get("issue_examples") or []
        self.assertFalse(
            any(
                "lowercase" in (example.get("reason") or "").lower()
                and "companies to process thousands of transactions remotely additionally implemented"
                in (example.get("text") or "").lower()
                for example in issue_examples
            )
        )

    def test_quantifying_counts_plus_volume_bullets_as_quantified(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-quant-plus-volume-0001",
            resume_text=(
                "Experience\n"
                "- Designed and led the end-to-end delivery of a multi-tenant SAP Business One Web Client used by "
                "50+ companies to process thousands of transactions remotely.\n"
                "Email asad@example.com\n"
                "Phone +1 555 333 9999\n"
            ),
            job_description_text="Software engineering role requiring delivery ownership and measurable impact.",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        quantifying = self._get_check(response, "content", "quantifying_impact")
        metrics = quantifying.get("metrics") or {}
        self.assertGreaterEqual(metrics.get("quantified_bullets", 0), 1)
        issue_examples = quantifying.get("issue_examples") or []
        self.assertFalse(
            any(
                "50+ companies" in (example.get("text") or "").lower()
                for example in issue_examples
            )
        )

    def test_no_experience_bullets_sets_needs_user_input_and_low_confidence_quantifying(self):
        resume_text = (
            "Professional Profile\n"
            "Experienced content specialist with strong cross-functional collaboration and editorial quality focus.\n"
            "Worked with distributed teams, campaign managers, and designers to deliver production-ready copy under strict timelines.\n"
            "Managed publication reviews, revisions, and handoffs with quality-control checkpoints across multiple channels.\n"
            "Partnered with stakeholders to shape messaging and support product narratives with clear documentation.\n"
            "Skills\n"
            "Content strategy, editing, proofreading, collaboration, writing\n"
        )
        payload = ToolRequest(
            locale="en",
            session_id="session-no-bullets-needs-input-0001",
            resume_text=resume_text,
            job_description_text=self.WRITER_LIKE_JD_TEXT,
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        ats_checker = (response.details or {}).get("ats_checker") or {}
        self.assertTrue(ats_checker.get("needs_user_input"))
        errors = " ".join(ats_checker.get("errors") or []).lower()
        self.assertIn("experience bullets", errors)

    def test_short_pdf_like_parse_sets_needs_user_input_with_clear_error(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-short-pdf-0001",
            resume_text="tiny",
            job_description_text="Backend role",
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        with patch(
            "app.services.tools_service.parse_document",
            return_value=ParsedDoc(
                doc_id="short-pdf-doc",
                source_type="pdf",
                language=None,
                text="short text",
                blocks=[],
                parsing_warnings=[],
                layout_flags={},
            ),
        ):
            response = run_ats_checker(payload)

        ats_checker = (response.details or {}).get("ats_checker") or {}
        self.assertTrue(ats_checker.get("needs_user_input"))
        errors = ats_checker.get("errors") or []
        self.assertTrue(any("Insufficient parsed text" in error for error in errors))

    def test_writer_like_end_to_end_regression_has_populated_checks_and_confidence(self):
        payload = ToolRequest(
            locale="en",
            session_id="session-writer-e2e-0001",
            resume_text=(
                "A green card holder with over four years of experience in online content writing and print media.\n"
                "Have completed campaign launches with editorial deadlines and stakeholder reviews.\n"
                "Email writer@example.com\n"
                "Phone +1 555 123 9876\n"
                "EXPERIENCE\n"
                "- Wrote and edited web content for product campaigns.\n"
                "- Coordinated with design and marketing teams to ship copy updates.\n"
                "- Delivered long-form articles aligned to SEO briefs and publication calendars.\n"
                "SKILLS\n"
                "Content writing, Copy editing, SEO, Stakeholder communication\n"
            ),
            job_description_text=(
                "Requirements:\n"
                "- Content writing for web and print\n"
                "- Editing and proofreading\n"
                "- SEO optimization\n"
                "- Stakeholder collaboration\n"
            ),
            candidate_profile={"target_region": "US", "seniority": "mid"},
        )
        response = run_ats_checker(payload)
        self.assertGreater(response.confidence, 0.0)
        report = (response.details or {}).get("ats_report") or {}
        categories = report.get("categories") or []
        self.assertEqual(len(categories), 5)
        payload_text = json.dumps(response.model_dump(mode="json")).lower()
        self.assertNotIn("unavailable in the current payload", payload_text)
        self.assertNotIn("unavailable payload", payload_text)


if __name__ == "__main__":
    unittest.main()
