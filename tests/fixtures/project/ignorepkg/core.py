x: int = 1  # type: ignore[assignment]
y = "hello"  # pyright: ignore[reportGeneralClassIssue]


def value() -> str:  # ty: ignore
    return "ok"
