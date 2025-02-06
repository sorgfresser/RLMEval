import re


def compress_newlines(lean_code: str):
    # compress lines containing only whitespaces
    lean_code = re.sub(r"^\s+$", "", lean_code, flags=re.MULTILINE)
    # Compress multiple consecutive newlines
    lean_code = re.sub(r"\n\n+", "\n\n", lean_code)
    lean_code = lean_code.lstrip()
    if lean_code.endswith("\n"):
        lean_code = lean_code.rstrip() + "\n"
    return lean_code
