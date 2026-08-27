"""أسماء الجلسات العربية يجب أن تبقى في الـ commit (core.quotepath)."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path_is_allowed(f, files=("outcomes.csv", "sessions/")):
    for a in files:
        prefix = a.rstrip("/")
        if f == prefix or f.startswith(prefix + "/"):
            return True
    return False


class TestPushCsvArabicPaths(unittest.TestCase):
    def test_ledger_note_is_allowed(self):
        self.assertTrue(path_is_allowed("sessions/السجل.md"))
        self.assertTrue(path_is_allowed("sessions/2026-08-26.md"))
        self.assertFalse(path_is_allowed("cheap_options_screener_v3.py"))
        self.assertFalse(path_is_allowed('"sessions/السجل.md"'))

    def test_quotepath_false_lists_arabic_without_quotes(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["git", "init"], cwd=td, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t"], cwd=td, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=td, check=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "init"],
                cwd=td, check=True, capture_output=True,
            )
            sess = Path(td) / "sessions"
            sess.mkdir()
            (sess / "السجل.md").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "add", "sessions"], cwd=td, check=True)
            quoted = subprocess.check_output(
                ["git", "-c", "core.quotepath=true", "diff", "--cached", "--name-only"],
                cwd=td, text=True,
            ).strip()
            plain = subprocess.check_output(
                ["git", "-c", "core.quotepath=false", "diff", "--cached", "--name-only"],
                cwd=td, text=True,
            ).strip()
            self.assertIn("السجل.md", plain)
            self.assertFalse(plain.startswith('"'))
            self.assertTrue(quoted.startswith('"') or "\\" in quoted)
            bad = subprocess.run(
                ["git", "restore", "--staged", "--", quoted],
                cwd=td, capture_output=True, text=True,
            )
            self.assertNotEqual(bad.returncode, 0, bad.stderr)
            good = subprocess.run(
                ["git", "restore", "--staged", "--", plain],
                cwd=td, capture_output=True, text=True,
            )
            self.assertEqual(good.returncode, 0, good.stderr)


if __name__ == "__main__":
    unittest.main()
