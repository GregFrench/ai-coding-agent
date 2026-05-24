import tempfile
import unittest
from pathlib import Path

from local_coding_agent.tools import ShellPolicy, WorkspaceTools


class WorkspaceToolsTests(unittest.TestCase):
    def test_write_read_and_replace_file(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = WorkspaceTools(directory, shell_policy=ShellPolicy(enabled=False))

            write = tools.write_file("example.txt", "hello\nworld\n")
            self.assertTrue(write.ok)

            read = tools.read_file("example.txt")
            self.assertEqual(read.output, "hello\nworld\n")

            replace = tools.replace_in_file("example.txt", "world", "agent")
            self.assertTrue(replace.ok)
            self.assertEqual(Path(directory, "example.txt").read_text(encoding="utf-8"), "hello\nagent\n")

    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = WorkspaceTools(directory, shell_policy=ShellPolicy(enabled=False))

            result = tools.run("read_file", {"path": "../outside.txt"})

            self.assertFalse(result.ok)
            self.assertIn("Path escapes workspace", result.output)

    def test_shell_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = WorkspaceTools(directory, shell_policy=ShellPolicy(enabled=False))

            result = tools.run("run_shell", {"command": "echo hi"})

            self.assertFalse(result.ok)
            self.assertIn("disabled", result.output)


if __name__ == "__main__":
    unittest.main()
