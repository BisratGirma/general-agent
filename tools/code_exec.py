"""Safe Python code execution in an isolated subprocess."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap

from tools.common import _format_error

# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------

_BLOCKED_MODULES = frozenset({
    "os", "subprocess", "sys", "socket", "requests", "shutil",
    "pathlib", "ctypes", "pickle", "urllib", "http", "ssl",
    "multiprocessing", "importlib", "builtins",
})

_BLOCKED_CALLS = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "exit", "quit",
})


class _SecurityVisitor(ast.NodeVisitor):
    """AST visitor that raises ``ValueError`` on forbidden imports or calls."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in _BLOCKED_MODULES:
                raise ValueError(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in _BLOCKED_MODULES:
            raise ValueError(f"Forbidden import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name: str | None = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name in _BLOCKED_CALLS:
            raise ValueError(f"Forbidden call: {func_name}")
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

def execute_python_code(code: str, timeout: int = 30) -> str:
    """Execute *code* in an isolated subprocess and return its output.

    The code is first parsed and statically checked for forbidden imports and
    function calls before any execution takes place.

    Args:
        code: Python source code to run.
        timeout: Maximum execution time in seconds.

    Returns:
        Captured stdout/stderr prefixed with ``"Output:\\n"``, or a formatted
        error string describing what went wrong.
    """
    code = (code or "").strip()
    if not code:
        return _format_error("Code Interpreter", "empty code")

    # --- Static security check ---
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return _format_error("Code Interpreter", f"syntax error: {exc}")

    try:
        _SecurityVisitor().visit(tree)
    except ValueError as exc:
        return _format_error("Code Interpreter", f"Security violation - {exc}")

    # --- Execute in subprocess ---
    script = textwrap.dedent(code)
    kwargs: dict = {"capture_output": True, "text": True, "timeout": timeout}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        result = subprocess.run([sys.executable, "-c", script], **kwargs)
    except subprocess.TimeoutExpired:
        return _format_error("Code Interpreter", "execution timed out")
    except Exception as exc:
        return _format_error("Code Interpreter", str(exc))

    output_parts: list[str] = []
    if result.stdout.strip():
        output_parts.append(result.stdout.strip())
    if result.stderr.strip():
        output_parts.append(f"STDERR:\n{result.stderr.strip()}")

    if not output_parts:
        return "Output:\n<no output>"
    return "Output:\n" + "\n".join(output_parts)
