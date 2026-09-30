# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for the offline diff viewer."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from rdd.review import git, render, snapshot


class ReviewTests(unittest.TestCase):
    @staticmethod
    def baseline(root):
        git(root, "init", "-q")
        (root / "tracked.txt").write_text("base\n", encoding="utf-8")
        (root / "deleted.txt").write_text("remove in snapshot\n", encoding="utf-8")
        (root / ".gitignore").write_text("cache/\n", encoding="utf-8")
        git(root, "add", ".")
        git(root, "-c", "user.name=Review Test", "-c", "user.email=review@example.invalid",
            "commit", "-qm", "baseline")
        git(root, "tag", "fastvideo-base")

    def test_complete_patch_without_staging(self):
        with tempfile.TemporaryDirectory(prefix="rdd-review-test-") as directory:
            root = Path(directory)
            self.baseline(root)
            (root / "tracked.txt").write_text("staged\n", encoding="utf-8")
            git(root, "add", "tracked.txt")
            (root / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
            (root / "deleted.txt").unlink()
            (root / "notes & cafe.md").write_text("new text\n", encoding="utf-8")
            (root / "media.bin").write_bytes(bytes(range(256)))
            (root / "cache").mkdir()
            (root / "cache/ignored.txt").write_text("excluded\n", encoding="utf-8")
            index_before = (root / ".git/index").read_bytes()
            head_before = git(root, "rev-parse", "HEAD")
            entries, patch = snapshot(root)
            self.assertEqual(index_before, (root / ".git/index").read_bytes())
            self.assertEqual(head_before, git(root, "rev-parse", "HEAD"))
            files = {item["path"]: item for item in entries}
            self.assertEqual(set(files), {"tracked.txt", "deleted.txt", "notes & cafe.md", "media.bin"})
            self.assertEqual(files["tracked.txt"]["status"], "M")
            self.assertEqual(files["deleted.txt"]["status"], "D")
            self.assertTrue(files["media.bin"]["binary"])
            self.assertTrue(files["notes & cafe.md"]["untracked"])
            self.assertIn("+unstaged", files["tracked.txt"]["diff"])
            page = render(entries, "example", "test")
            self.assertIn("notes &amp; cafe.md", page)
            self.assertIn("Binary media", page)
            with tempfile.TemporaryDirectory(prefix="rdd-review-apply-test-") as apply_directory:
                apply_root = Path(apply_directory)
                self.baseline(apply_root)
                env = os.environ.copy()
                subprocess.run(["git", "apply", "--check", "--binary", "-"],
                               input=patch, cwd=apply_root, env=env, check=True,
                               capture_output=True)


if __name__ == "__main__":
    unittest.main()
