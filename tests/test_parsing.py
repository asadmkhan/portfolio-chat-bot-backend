import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.parse import parse_document  # noqa: E402


class ParsingFacadeTests(unittest.TestCase):
    def test_parse_txt_returns_stable_parsed_doc(self):
        content = "Line one\n- Bullet item\nLine three"
        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(content)
            tmp_file.close()

            parsed = parse_document(str(tmp_path))
            self.assertEqual(parsed.source_type, "txt")
            self.assertEqual(parsed.text, content)
            self.assertTrue(parsed.doc_id)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

