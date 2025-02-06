def message_intersects_code(message, start_line, end_line):
    res = True
    if start_line is not None:
        if message["endPos"]:
            res = res and message["endPos"]["line"] >= start_line
    if end_line is not None:
        if message["startPos"]:
            res = res and message["startPos"]["line"] <= end_line
    return res


def is_valid_lean(
    lean_output: dict, start_line: int | None = None, end_line: int | None = None, allow_sorry: bool = True
):
    # check only the messages intersecting the code
    errors = [
        message
        for message in lean_output.get("messages", [])
        if message_intersects_code(message, start_line, end_line)
        and message["severity"] == "error"
        and not message["data"] == "no goals to be solved"  # goal is solved but useless tactics were applied at the end
    ]
    sorries = [
        message for message in lean_output.get("sorries", []) if message_intersects_code(message, start_line, end_line)
    ]
    return not errors and (allow_sorry or not sorries) and "env" in lean_output
