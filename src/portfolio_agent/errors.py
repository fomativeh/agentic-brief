from __future__ import annotations


class ToolError(RuntimeError):
    def __init__(self, message: str, *, tool: str, kind: str, status_code: int | None = None):
        super().__init__(message)
        self.tool = tool
        self.kind = kind
        self.status_code = status_code

    def to_error(self) -> dict:
        return {
            "tool": self.tool,
            "kind": self.kind,
            "status_code": self.status_code,
            "message": str(self),
        }
