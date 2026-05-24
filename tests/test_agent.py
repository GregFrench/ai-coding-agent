import unittest
from dataclasses import dataclass

from local_coding_agent.agent import CodingAgent
from local_coding_agent.llm import ChatMessage
from local_coding_agent.tools import ShellPolicy, WorkspaceTools


@dataclass
class FakeLLM:
    responses: list[str]

    def chat(self, messages: list[ChatMessage]) -> str:
        if not self.responses:
            raise AssertionError("No fake responses left.")
        return self.responses.pop(0)


class CodingAgentTests(unittest.TestCase):
    def test_agent_runs_tool_then_finishes(self):
        llm = FakeLLM(
            responses=[
                '{"action":"list_files","path":"."}',
                '{"action":"finish","message":"All done."}',
            ]
        )
        tools = WorkspaceTools(".", shell_policy=ShellPolicy(enabled=False))
        agent = CodingAgent(llm=llm, tools=tools, max_steps=3)

        result = agent.run("look around")

        self.assertTrue(result.ok)
        self.assertEqual(result.message, "All done.")
        self.assertEqual(result.steps, 2)

    def test_agent_does_not_accept_finish_after_protocol_error_without_tool_use(self):
        llm = FakeLLM(
            responses=[
                "not json",
                '{"action":"finish","message":"The provided input was not a valid JSON object."}',
            ]
        )
        tools = WorkspaceTools(".", shell_policy=ShellPolicy(enabled=False))
        agent = CodingAgent(llm=llm, tools=tools, max_steps=3)

        result = agent.run("create a file")

        self.assertFalse(result.ok)
        self.assertIn("did not take any action", result.message)

    def test_agent_observes_shell_stdout(self):
        llm = FakeLLM(
            responses=[
                '{"action":"run_shell","command":"printf \'1, 1, 2\'","timeout_seconds":5}',
                '{"action":"finish","message":"Done."}',
            ]
        )
        observed = []
        tools = WorkspaceTools(".", shell_policy=ShellPolicy(enabled=True, auto_approve=True))
        agent = CodingAgent(llm=llm, tools=tools, max_steps=3, observer=observed.append)

        result = agent.run("print numbers")

        self.assertTrue(result.ok)
        self.assertIn("command stdout:\n1, 1, 2", observed)


if __name__ == "__main__":
    unittest.main()
