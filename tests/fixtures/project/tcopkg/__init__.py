from typing import type_check_only

from tcopkg.mod import PublicClass, public_func


@type_check_only
class _Proto:
    x: int


def visible() -> None: ...


__all__ = ["PublicClass", "_Proto", "public_func", "visible"]
