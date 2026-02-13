from getattr_pkg.mod import dynamic_a as dynamic_a

__all__ = ["dynamic_a", "dynamic_b"]


def __getattr__(name: str) -> int:
    ...
