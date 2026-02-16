from typing import Any

from anypkg._defs import Chained, NotAny, Remote, Unknown

any_var: Any = None
unknown_var: Unknown = None
chained_var: Chained = None
remote_var: Remote = None
normal_var: int = 1
not_any_alias_var: NotAny = 1


def annotated_func(a: Unknown, b: int) -> str:
    return str(a)
