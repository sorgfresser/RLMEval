import re
from typing import Any

from rlm_eval.tools.string import compress_newlines


def indent_code(code: str, nb_spaces: int = 2) -> str:
    return "\n".join(" " * nb_spaces + line for line in code.split("\n"))


def lean_comments_ranges(
    lean_code: str, multiline_comment_suffix: str = "", remove_single_line_comments: bool = True
) -> list[tuple[int, int]]:
    """Extract the ranges of Lean comments from a Lean code snippet."""
    # multiline comments
    open_comment_indices = [m.start() for m in re.finditer(r"/-" + multiline_comment_suffix, lean_code)]
    close_comment_indices = [
        m.start() + len(multiline_comment_suffix) + 2 for m in re.finditer(multiline_comment_suffix + r"-/", lean_code)
    ]

    if len(open_comment_indices) == len(close_comment_indices) + 1:
        # the last comment has probably not been closed due to partial code
        close_comment_indices.append(len(lean_code))

    elif len(open_comment_indices) + 1 == len(close_comment_indices):
        # the first comment has probably been opened before the code snippet
        open_comment_indices.insert(0, 0)

    elif len(open_comment_indices) != len(close_comment_indices):
        raise ValueError("Mismatched open and close comment indices.")

    # trick to handle nested comments in a simple way
    multiline_comment_ranges = list(zip(open_comment_indices, close_comment_indices))

    if remove_single_line_comments:
        # single line comments
        single_line_comment_ranges = [
            (m.start(), lean_code.find("\n", m.start())) for m in re.finditer(r"--", lean_code)
        ]
        multiline_comment_ranges += single_line_comment_ranges

    # merge potential overlapping ranges
    comment_ranges = sorted(multiline_comment_ranges, key=lambda x: x[0])
    merged_comment_ranges = []
    for start, end in comment_ranges:
        if merged_comment_ranges and start <= merged_comment_ranges[-1][1]:
            merged_comment_ranges[-1] = (merged_comment_ranges[-1][0], max(merged_comment_ranges[-1][1], end))
        else:
            merged_comment_ranges.append((start, end))

    return merged_comment_ranges


def remove_lean_comments(lean_code: str) -> str | None:
    try:
        comment_ranges = lean_comments_ranges(lean_code)

        new_lean_code = ""
        prev_start = 0
        for start, end in comment_ranges:
            new_lean_code += lean_code[prev_start:start]
            prev_start = end

        new_lean_code += lean_code[prev_start:]
        return new_lean_code

    except Exception:
        return None


def split_implementation(declaration: str, start: int = 0):
    # for a theorem, an implementation is the proof
    if ":=" in declaration:
        # we have to be careful here as ":=" can be used inside the declaration itself
        indices = set([m.start() for m in re.finditer(r":=", declaration)])

        # we remove the ones related to "let", "haveI", ... declarations
        for keyword in ["let", "haveI"]:
            regex = rf"{keyword}\s+\S*?\s*(:=)"
            decl_indices = set([m.start(1) for m in re.finditer(regex, declaration)])
            indices = indices - decl_indices

        # implementation using pcre2 blows up the memory, and it turns out it is faster to use a python loop
        counters = {"(": 0, "{": 0, "[": 0}
        closing = {")": "(", "}": "{", "]": "["}
        for i, c in enumerate(declaration[start:]):
            if c in counters:
                counters[c] += 1
            elif c in [")", "}", "]"]:
                counters[closing[c]] -= 1
            if all([v == 0 for v in counters.values()]) and (i + start) in indices:
                return i + start
    return None


def split_conclusion(declaration: str, start: int = 0) -> int | None:
    counters = {"(": 0, "{": 0, "[": 0}
    closing = {")": "(", "}": "{", "]": "["}
    for i, c in enumerate(declaration[start:]):
        if c in counters:
            counters[c] += 1
        elif c in [")", "}", "]"]:
            counters[closing[c]] -= 1
        if all([v == 0 for v in counters.values()]) and c == ":":
            return i + start
    return None


def clean_theorem_string(theorem_string: str, new_theorem_name: str = "dummy", add_sorry: bool = True) -> str | None:
    """Clean a theorem string by removing the proof, comments, and updating the theorem name.
    This method assumes that no other declarations are present in the theorem string."""
    try:
        # clean the theorem string
        clean_formal = remove_lean_comments(theorem_string)
        if clean_formal is None:
            raise ValueError("Comment removal failed.")
        # clean_formal = re.sub(r"\s+", " ", clean_formal).strip()
        clean_formal = clean_formal.strip()

        # we remove the first part of the string until the first "theorem" or "lemma" keyword
        theorem_decl_keywords = "|".join(["theorem", "lemma"])
        re_match = re.search(rf"\b{theorem_decl_keywords}\s", clean_formal)
        if re_match is None:
            return None
        idx_theorem = re_match.start()
        clean_formal = clean_formal[idx_theorem:]

        # if a proof is provided we remove it
        idx_implement = split_implementation(clean_formal)
        if idx_implement is not None:
            clean_formal = clean_formal[:idx_implement].strip()

        # remove "theorem"/"lemma" and the theorem name
        clean_formal = re.sub(r"^[^\s]+", "", clean_formal).strip()
        clean_formal = re.sub(r"^[^\s:({\[]+", "", clean_formal).strip()

        clean_formal = f"theorem {new_theorem_name} " + clean_formal
        if add_sorry:
            clean_formal += " := sorry"
        return clean_formal
    except Exception:
        return None


def extract_last_theorem(lean_code: str) -> int:
    """Extract the last theorem from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    comments_ranges = lean_comments_ranges(lean_code)

    # find last theorem by looking for `theorem` keyword surrounded by whitespaces, or by being at the beginning of the string
    theorem_decl_keywords = ["theorem", "lemma"]
    theorem_indices = []
    for keyword in theorem_decl_keywords:
        theorem_indices += [m.start() for m in re.finditer(rf"\b{keyword}\s", lean_code)]

    if not theorem_indices:
        raise ValueError(f"No theorem found in the provided Lean code:\n{lean_code}")

    # remove matches that are inside comments
    theorem_indices = [idx for idx in theorem_indices if not any(start <= idx <= end for start, end in comments_ranges)]

    return theorem_indices[-1]


def extract_last_theorem_name(lean_code: str) -> str:
    """Extract the last theorem name from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    idx_last_theorem = extract_last_theorem(lean_code)
    theorem_decl_keywords = ["theorem", "lemma"]
    for keyword in theorem_decl_keywords:
        if lean_code[idx_last_theorem:].startswith(keyword):
            idx_theorem_name = idx_last_theorem + len(keyword)
            return lean_code[idx_theorem_name:].split()[0]

    raise ValueError(f"Theorem name extraction failed for the following Lean code:\n{lean_code}")


def clean_last_theorem_string(lean_code: str, new_theorem_name: str = "dummy", add_sorry: bool = False) -> str:
    """Clean the last theorem string from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    idx_last_theorem = extract_last_theorem(lean_code)
    clean_thm = clean_theorem_string(lean_code[idx_last_theorem:], new_theorem_name, add_sorry=add_sorry)
    if clean_thm is not None:
        return lean_code[:idx_last_theorem] + clean_thm

    raise ValueError(f"Theorem extraction failed for the following Lean code:\n{lean_code}")


def remove_ranges(content: str, ranges: list[tuple[int, int]]) -> str:
    ranges = sorted(ranges, key=lambda x: x[0])

    # merge potential overlapping ranges
    merged_ranges = []
    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    new_content = ""
    prev_start = 0
    for start, end in merged_ranges:
        if start >= len(content):
            break
        if end >= len(content):
            end = len(content)
        new_content += content[prev_start:start]
        prev_start = end
    new_content += content[prev_start:]

    return new_content


def replace_ranges(content: str, ranges: list[tuple[int, int, str]]) -> str:
    ranges = sorted(ranges, key=lambda x: x[0])

    # check potential overlapping ranges
    for i in range(len(ranges) - 1):
        if ranges[i][1] > ranges[i + 1][0]:
            raise ValueError("Overlapping ranges")

    new_content = ""
    prev_start = 0
    for start, end, replacement in ranges:
        if start >= len(content):
            break
        if end >= len(content):
            end = len(content)
        new_content += content[prev_start:start] + replacement
        prev_start = end
    new_content += content[prev_start:]

    return new_content


def trim_comments_end(lean_code: str) -> str:
    """Trim the end of a Lean code snippet by removing everything that is commented or whitespace.
    We keep newlines at the end of the code snippet."""
    comment_ranges = lean_comments_ranges(lean_code)
    idx = len(lean_code)

    while idx > 0:
        idx -= 1
        if lean_code[idx].isspace():
            continue
        # Check if idx is inside a comment range
        in_comment = False
        for start, end in comment_ranges:
            if start <= idx < end:
                in_comment = True
                idx = start
                break
        if not in_comment:
            # Found non-comment, non-whitespace character
            idx += 1
            while idx < len(lean_code) and lean_code[idx] == "\n":
                idx += 1
            break
    return lean_code[:idx]


class LeanFilesProcessor:
    def __init__(self, lean_declarations: list[dict[str, Any]]):
        self.lean_declarations = lean_declarations

    def compress(self, file_content: str, remove_top_comment: bool = True) -> str:
        """Compress extra lines / whitespaces and remove top comments at the beginning of the file as \
        they often only contain information about the author and file license."""
        # remove top comments
        if remove_top_comment:
            comment_ranges = lean_comments_ranges(file_content)
            if len(comment_ranges) > 0 and comment_ranges[0][0] == 0:
                file_content = remove_ranges(file_content, [comment_ranges[0]])

        # remove all `#align` lines
        file_content = re.sub(r"^\s*#align.*$", "", file_content, flags=re.MULTILINE)

        # remove comments using /-%% format
        comment_ranges = lean_comments_ranges(
            file_content, multiline_comment_suffix="%", remove_single_line_comments=False
        )

        file_content = remove_ranges(file_content, comment_ranges)

        return compress_newlines(file_content)

    def remove_theorems(
        self,
        lean_file_path: str,
        lean_file_content: str,
        whitelist: set[str] | None = None,
        compress_newlines_comments: bool = True,
        remove_top_comment: bool = True,
    ) -> str:
        """Remove all theorems from a Lean file.
        Caution: this will very likely produce invalid Lean code"""
        if whitelist is None:
            whitelist = set()

        # collect all theorem declarations
        theorem_decls_ranges = []
        for decl in self.lean_declarations:
            if decl["file"] == lean_file_path and "theorem_info" in decl and decl["full_name"] not in whitelist:
                # before removing the theorem, we have to check if the previous non-empty line is an "open ... in"
                start_idx = decl["start_idx"]
                lines_before = lean_file_content[:start_idx].split("\n")
                i = len(lines_before) - 1
                while i >= 0 and not lines_before[i].strip():
                    i -= 1
                if i >= 0 and lines_before[i].rstrip().endswith(" in"):
                    nb_char = sum(len(line) + 1 for line in lines_before[i + 1 :])
                    nb_char += len(lines_before[i]) - lines_before[i].rfind(" in")
                    start_idx = start_idx - nb_char

                theorem_decls_ranges.append((start_idx, decl["end_idx"]))

        new_content = remove_ranges(lean_file_content, theorem_decls_ranges)

        if compress_newlines_comments:
            new_content = self.compress(new_content, remove_top_comment=remove_top_comment)

        return new_content

    def remove_proofs(
        self,
        lean_file_path: str,
        lean_file_content: str,
        whitelist: set[str] | None = None,
        compress_newlines: bool = True,
        remove_top_comment: bool = True,
    ) -> str:
        """Remove all proofs from a Lean file.
        Caution: this will very likely produce invalid Lean code"""
        if whitelist is None:
            whitelist = set()

        # collect all theorem declarations
        proof_ranges = []
        for decl in self.lean_declarations:
            if decl["file"] == lean_file_path and "theorem_info" in decl and decl["full_name"] not in whitelist:
                proof_ranges.append((decl["theorem_info"]["proof_start_idx"], decl["end_idx"], " := sorry"))

        new_content = replace_ranges(lean_file_content, proof_ranges)

        if compress_newlines:
            new_content = self.compress(new_content, remove_top_comment=remove_top_comment)

        return new_content
