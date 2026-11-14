"""Connect AppWorld's ReAct code loop to the shared interactive test APIs."""

import ast
from types import SimpleNamespace

from appworld_agents.code.simplified.agent import ExecutionIO

BUILTINS = {
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "next",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
METHODS = {
    "append",
    "casefold",
    "count",
    "endswith",
    "extend",
    "find",
    "get",
    "index",
    "items",
    "join",
    "keys",
    "lower",
    "lstrip",
    "replace",
    "rsplit",
    "rstrip",
    "split",
    "splitlines",
    "startswith",
    "strip",
    "upper",
    "values",
}
STATEMENTS = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
)
EXPRESSIONS = (
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Call,
    ast.keyword,
    ast.Starred,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.IfExp,
    ast.JoinedStr,
    ast.FormattedValue,
)
API_DOCS = {"show_app_descriptions", "show_api_descriptions", "show_api_doc"}


class ReActEnvironment:
    def __init__(self, world, broker, interaction_tools, tool_docs):
        self.world = world
        self.variables = set()
        self.docs = {app: {} for app in sorted({name.split("__", 1)[0] for name in broker.allowed})}
        self.docs["user"] = {}
        functions = {app: SimpleNamespace() for app in self.docs}
        self.paths = {f"apis.api_docs.{name}" for name in API_DOCS}
        self.attributes = {"apis.api_docs", *(f"apis.{app}" for app in self.docs)}
        for name in sorted(broker.allowed):
            app, api = name.split("__", 1)
            self.docs[app][api] = (
                tool_docs[name]
            )
            self.paths.add(f"apis.{app}.{api}")
            setattr(functions[app], api, self.tool(broker, name))
        for spec in interaction_tools:
            function = spec["function"]
            name = function["name"]
            api = name.split("__", 1)[1]
            self.docs["user"][api] = function
            self.paths.add(f"apis.user.{api}")
            setattr(functions["user"], api, self.tool(broker, name))
        functions["api_docs"] = SimpleNamespace(
            show_app_descriptions=lambda: [
                {
                    "name": app,
                    "description": world.task.app_descriptions.get(app, "User interaction"),
                }
                for app in self.docs
            ],
            show_api_descriptions=self.show_api_descriptions,
            show_api_doc=self.show_api_doc,
        )
        world.shell.user_ns["apis"] = SimpleNamespace(**functions)

    @staticmethod
    def tool(broker, name):
        def invoke(**args):
            if broker.claimed_success or broker.explicit_failure:
                raise ValueError("The task has ended.")
            return broker.call(name, args)

        return invoke

    def show_api_descriptions(self, app_name):
        return [
            {"name": name, "description": doc["description"]}
            for name, doc in self.docs.get(app_name, {}).items()
        ]

    def show_api_doc(self, app_name, api_name):
        return self.docs.get(app_name, {}).get(api_name, {"error": "API outside the pilot scope."})

    def execute(self, execution):
        code = execution.content
        try:
            nodes = list(ast.walk(ast.parse(code)))
            assigned = {
                node.id
                for node in nodes
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
            available = {"apis"} | BUILTINS | self.variables
            if any(name.startswith("_") or name in ({"apis"} | BUILTINS) for name in assigned):
                raise ValueError("Use a new variable name for assignments.")
            for node in nodes:
                if not isinstance(
                    node,
                    (
                        *STATEMENTS,
                        *EXPRESSIONS,
                        ast.operator,
                        ast.unaryop,
                        ast.boolop,
                        ast.cmpop,
                        ast.expr_context,
                    ),
                ):
                    raise ValueError(f"{type(node).__name__} is outside the pilot REPL subset.")
                if isinstance(node, ast.Name) and node.id not in available | assigned:
                    raise ValueError(f"Unknown variable: {node.id}")
                if isinstance(node, ast.Attribute) and (
                    not isinstance(node.ctx, ast.Load)
                    or (
                        node.attr not in METHODS
                        and ast.unparse(node) not in self.paths | self.attributes
                    )
                ):
                    raise ValueError("Use a documented API or a supported data method.")
                if isinstance(node, ast.Call) and not (
                    isinstance(node.func, ast.Name)
                    and node.func.id in BUILTINS
                    or isinstance(node.func, ast.Attribute)
                    and (node.func.attr in METHODS or ast.unparse(node.func) in self.paths)
                ):
                    raise ValueError("Call a documented API, supported data method or built-in.")
            if any(
                name in self.world.shell.user_ns and name not in self.variables for name in assigned
            ):
                raise ValueError(
                    "Choose a variable name that does not shadow the REPL environment."
                )
        except (SyntaxError, ValueError) as error:
            return ExecutionIO(content=f"Execution rejected: {error}")
        result = self.world.execute(code)
        self.variables.update(name for name in assigned if name in self.world.shell.user_ns)
        return ExecutionIO(content=result)
