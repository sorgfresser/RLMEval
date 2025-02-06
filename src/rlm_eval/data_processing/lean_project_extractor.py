import json
import os

import jsonlines
from lean_dojo import LeanFile, LeanGitRepo, TracedFile, TracedRepo, trace
from lean_dojo.data_extraction.ast import (
    CommandDeclarationNode,
    CommandDeclsigNode,
    IdentNode,
    LemmaNode,
    Node,
    Pos,
)
from lean_dojo.data_extraction.traced_data import (
    get_code_without_comments,
    is_mutual_lean4,
    is_potential_premise_lean4,
)
from rlm_eval.utils import logger


def pos_to_index(lean_file: LeanFile, pos: tuple[int, int] | Pos) -> int:
    lines = lean_file.code
    line, col = pos
    return sum(len(line) + 1 for line in lines[: line - 1]) + col - 1


def index_to_pos(lean_file: LeanFile, idx: int) -> Pos:
    lines = lean_file.code
    line = 0
    col = idx
    while col >= len(lines[line]):
        col -= len(lines[line]) + 1
        line += 1
    return Pos(line + 1, col + 1)


def get_declaration_dependencies(traced_file: TracedFile) -> list[dict]:
    results = []

    def _callback(node: Node, _) -> None:
        if is_potential_premise_lean4(node):
            start, end = node.get_closure()
            assert start is not None and end is not None
            res_dict = {
                "kind": node.kind(),
                "file": traced_file.path.as_posix(),
                "start": list(start),
                "end": list(end),
                "start_idx": pos_to_index(traced_file.lean_file, start),
                "end_idx": pos_to_index(traced_file.lean_file, end),
            }

            local_results = []

            def _sub_callback(node: Node, _) -> None:
                if (
                    isinstance(node, IdentNode)
                    and node.full_name is not None
                    and node.mod_name is not None
                    and node.def_start is not None
                    and node.def_end is not None
                ):
                    local_results.append(
                        {
                            "full_name": node.full_name,
                            "module": node.mod_name,
                            "file": node.def_path,
                            "start": list(node.def_start),
                            "end": list(node.def_end),
                            "start_idx": pos_to_index(traced_file.lean_file, node.def_start),
                            "end_idx": pos_to_index(traced_file.lean_file, node.def_end),
                        }
                    )

            def deduplicate(local_results):
                seen = set()
                deduplicated = []
                for result in local_results:
                    if result["full_name"] + result["file"] not in seen:
                        seen.add(result["full_name"] + result["file"])
                        deduplicated.append(result)
                return deduplicated

            is_theorem = False
            if isinstance(node, LemmaNode):
                is_theorem = True
                thm_node = node.children[1]
                proof_node = node.children[1].children[3]
            if isinstance(node, CommandDeclarationNode) and node.is_theorem:
                is_theorem = True
                thm_node = node.get_theorem_node()
                proof_node = thm_node.get_proof_node()

            node.traverse_preorder(_sub_callback, node_cls=None)
            res_dict["dependencies"] = deduplicate(local_results)
            res_dict["decl"] = traced_file.lean_file[start:end]
            res_dict["decl_no_comments"] = get_code_without_comments(
                traced_file.lean_file, start, end, traced_file.comments
            )

            if is_theorem:
                # theorem-specific information
                proof_start, _ = proof_node.get_closure()
                assert proof_start is not None

                # we adjust the beginning of the proof to be the last non-whitespace character before ":="
                tmp_declsig = traced_file.lean_file[start:proof_start]
                proof_start_idx = pos_to_index(traced_file.lean_file, proof_start)
                if tmp_declsig.rstrip().endswith(":="):
                    proof_start_idx -= len(tmp_declsig) - len(tmp_declsig[: tmp_declsig.rfind(":=")])
                    tmp_declsig = tmp_declsig[: tmp_declsig.rfind(":=")]
                proof_start_idx -= len(tmp_declsig) - len(tmp_declsig.rstrip())
                proof_start = index_to_pos(traced_file.lean_file, proof_start_idx)

                res_dict["theorem_info"] = {
                    "proof_start": list(proof_start),
                    "proof_start_idx": proof_start_idx,
                }
                res_dict["theorem_info"]["declsig"] = traced_file.lean_file[start:proof_start].strip()
                res_dict["theorem_info"]["proof"] = traced_file.lean_file[proof_start:end].strip()
                res_dict["theorem_info"]["declsig_no_comments"] = get_code_without_comments(
                    traced_file.lean_file, start, proof_start, traced_file.comments
                )
                res_dict["theorem_info"]["proof_no_comments"] = get_code_without_comments(
                    traced_file.lean_file, proof_start, end, traced_file.comments
                )

                # extract dependencies separately for theorem statement and proof
                assert isinstance(thm_node.children[2], CommandDeclsigNode)
                thm_node.children[2].traverse_preorder(_sub_callback, node_cls=None)
                res_dict["theorem_info"]["declsig_dependencies"] = deduplicate(local_results)

                local_results = []
                proof_node.traverse_preorder(_sub_callback, node_cls=None)
                res_dict["theorem_info"]["proof_dependencies"] = deduplicate(local_results)

            assert isinstance(node.name, str)  # type: ignore
            res_dict["name"] = node.name  # type: ignore
            if is_mutual_lean4(node):
                for s in node.full_name:  # type: ignore
                    assert isinstance(node.full_name, str)  # type: ignore
                    res_dict["full_name"] = s
                    results.append(res_dict)
            else:
                assert isinstance(node.full_name, str)  # type: ignore
                res_dict["full_name"] = node.full_name  # type: ignore
                results.append(res_dict)

    traced_file.traverse_preorder(_callback, node_cls=None)
    return results


def extract_declaration_dependencies(traced_repo: TracedRepo) -> tuple[list[dict], list[dict[str, str]]]:
    """Extract all Lean declaration identifiers from the traced repo and their dependencies."""
    declaration_dependencies = []
    files_content = []
    for traced_file in traced_repo.traced_files:
        declaration_dependencies.extend(get_declaration_dependencies(traced_file))
        files_content.append({"file": traced_file.path.as_posix(), "content": "\n".join(traced_file.lean_file.code)})
    return declaration_dependencies, files_content


def trace_repo(output_dir: str, git_url: str, commit: str, build_deps: bool = True) -> tuple[TracedRepo, list[dict]]:
    repo = LeanGitRepo(git_url, commit)
    if os.path.exists(output_dir):
        logger.info(f"Skipping tracing of {git_url} at commit {commit} as output directory already exists.")
        traced_repo = trace(repo, build_deps=build_deps)
    else:
        logger.info(f"Tracing {git_url} at commit {commit} to {output_dir}")
        traced_repo = trace(repo, dst_dir=output_dir, build_deps=build_deps)

    all_declarations, files_content = extract_declaration_dependencies(traced_repo)
    with jsonlines.open(os.path.join(output_dir, "lean_declarations.jsonl"), "w") as writer:
        writer.write_all(all_declarations)
    with jsonlines.open(os.path.join(output_dir, "lean_files.jsonl"), "w") as writer:
        writer.write_all(files_content)

    # get lean version
    with open(traced_repo.root_dir / "lean-toolchain", "r") as f:
        lean_toolchain = f.read().strip()
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({"git_url": git_url, "commit": commit, "lean_toolchain": lean_toolchain}, f)

    return traced_repo, all_declarations
