import unittest

from local_coding_agent.protocol import ProtocolError, parse_action


class ParseActionTests(unittest.TestCase):
    def test_parses_plain_json(self):
        action = parse_action('{"action":"list_files","path":"."}')

        self.assertEqual(action.action, "list_files")
        self.assertEqual(action.arguments, {"path": "."})

    def test_parses_fenced_json(self):
        action = parse_action('```json\n{"thought":"look","action":"read_file","path":"README.md"}\n```')

        self.assertEqual(action.action, "read_file")
        self.assertEqual(action.thought, "look")
        self.assertEqual(action.arguments["path"], "README.md")

    def test_rejects_missing_action(self):
        with self.assertRaises(ProtocolError):
            parse_action('{"path":"."}')

    def test_repairs_triple_quoted_json_string(self):
        action = parse_action(
            '```json\n'
            '{\n'
            '  "action": "write_file",\n'
            '  "path": "hello.py",\n'
            '  "content": """print("hi")\n"""\n'
            '}\n'
            '```'
        )

        self.assertEqual(action.action, "write_file")
        self.assertEqual(action.arguments["path"], "hello.py")
        self.assertEqual(action.arguments["content"], 'print("hi")\n')

    def test_accepts_unescaped_newline_inside_json_string(self):
        action = parse_action(
            '```json\n'
            '{"action":"write_file","path":"hello.py","content":"print(\\"hi\\")\n"}\n'
            '```'
        )

        self.assertEqual(action.arguments["content"], 'print("hi")\n')


if __name__ == "__main__":
    unittest.main()
