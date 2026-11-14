"""Adapters for the upstream Function Calling run loop."""

import ast

from appworld_agents.code.common.usage_tracker import Usage
from appworld_agents.code.simplified.agent import Status
from appworld_agents.code.simplified.function_calling_agent import SimplifiedFunctionCallingAgent


def parse_execution(execution):
    """Decode only the literal call wrapper produced by the official agent."""
    expression = ast.parse(execution.content, mode="eval").body
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "print"
        and len(expression.args) == 1
    ):
        raise ValueError("Invalid official execution wrapper")
    call = expression.args[0]
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == execution.metadata["function_name"]
        and not call.args
        and len(call.keywords) == 1
        and call.keywords[0].arg is None
    ):
        raise ValueError("Invalid official tool call")
    args = ast.literal_eval(call.keywords[0].value)
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    return call.func.id, args


class InteractiveOfficialAgent(SimplifiedFunctionCallingAgent):
    def __init__(self, *, interaction_tools, **kwargs):
        super().__init__(**kwargs)
        self.interaction_tools = interaction_tools

    def first_execution_inputs_usage_and_status(self):
        self.functions = list(self.interaction_tools)
        return [], Usage(), Status(failed=False)
