from typing import type_check_only


def public_func(x: int) -> str:
    return str(x)


@type_check_only
def _checker() -> None: ...


@type_check_only
class _InternalProto:
    y: str


class PublicClass:
    z: int
