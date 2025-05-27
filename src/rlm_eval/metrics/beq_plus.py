import json
from dataclasses import dataclass

from lean_interact import AutoLeanServer, Command
from lean_interact.interface import (
    CommandResponse,
    LeanError,
    Pos,
    message_intersects_code,
)
from lean_interact.utils import (
    clean_last_theorem_string,
    indent_code,
    split_conclusion,
)
from lean_pool_models import RunRequest, RunResponse
import requests
from pydantic import ValidationError

UVICORN_PORT = 8020
TIMEOUT = 90

def try_repeatedly(req: RunRequest, project: str, allow_error: bool = False, retry_count: int = 2) -> RunResponse:
    response = None
    for j in range(retry_count):
        resp = None
        for i in range(retry_count):
            try:
                resp = requests.post(f"http://localhost:{UVICORN_PORT}/{project}/run", data=req.model_dump_json(), timeout=TIMEOUT)
                break
            except requests.exceptions.Timeout:
                pass
        if resp is None:
            raise RuntimeError("Repeatedly timed out!")
        response = RunResponse.model_validate(resp.json())
        # Check whether is valid
        try:
            CommandResponse.model_validate(response.result)
        except ValidationError:
            if allow_error:
                try:
                    LeanError.model_validate(response.result)
                except ValidationError:
                    response = None
            else:
                response = None
        if response:
            break
    if response is None:
        raise RuntimeError("Parsing failed!")
    return response

@dataclass
class BEqCPUResult:
    beql_unidirections: tuple[str | None, str | None] = (None, None)
    beq_plus_unidirections: tuple[str | None, str | None] = (None, None)

    def beql(self) -> bool:
        return all(proof is not None for proof in self.beql_unidirections)

    def beq_plus(self) -> bool:
        return all(proof is not None for proof in self.beq_plus_unidirections)


def update_tuple(t: tuple, idx: int, value) -> tuple:
    t_list = list(t)
    t_list[idx] = value
    return tuple(t_list)


def extract_exact_proof(lean_output: CommandResponse, proof_start_line: int | None = None) -> str | None:
    # check only the messages intersecting the proof
    start = Pos(line=proof_start_line, column=0) if proof_start_line else None
    for message in lean_output.messages:
        if message_intersects_code(message, start, None):
            if message.severity == "error":
                return None
            if message.severity == "info" and message.data.startswith("Try this:"):
                return message.data.split("Try this:")[1].strip()
    return None


def check_theorem_equivalence(
        theorem1: str, theorem2: str, lean_server: AutoLeanServer, context_env: int, timeout_per_proof: int
) -> BEqCPUResult:
    base_thm_name = "base_theorem"
    reformulated_thm_name = "reformulated_theorem"

    def prove_all(tactics: list[str]) -> str:
        prove_independent = " ; ".join([f"(all_goals try {t})" for t in tactics])
        prove_combined = "all_goals (" + " ; ".join([f"(try {t})" for t in tactics]) + ")"
        return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

    solver_tactics_apply = ["tauto", "simp_all_arith!", "noncomm_ring", "exact?"]
    solver_tactics_have = ["tauto", "simp_all_arith!", "exact? using this"]
    proof_all_apply = prove_all(solver_tactics_apply)
    proof_all_have = prove_all(solver_tactics_have)

    res = BEqCPUResult()
    for i, (base_thm, reform_thm) in enumerate([(theorem1, theorem2), (theorem2, theorem1)]):
        try:
            formal_1_code = clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True) + "\n\n"
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name, add_sorry=False)} := by"
        except ValueError:
            # we cannot change one of the theorems name, probably because it is invalid, so we skip this pair
            break

        def check_proof_sub(proof: str, formal_code: str = formal_1_code + formal_2_code) -> str | None:
            prepended_proof = "\nintros\nsymm_saturate\n"
            try:
                lean_output = lean_server.run(
                    Command(cmd=formal_code + indent_code(prepended_proof + proof, 2), env=context_env),
                    timeout=timeout_per_proof,
                )

                if isinstance(lean_output, LeanError):
                    return None

                if proof == "sorry":
                    # sorry is used on purpose to check if the formalization is well-typed
                    if lean_output.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0)):
                        return proof
                    return None

                if lean_output.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0), allow_sorry=False):
                    if proof == "exact?":
                        return extract_exact_proof(lean_output, proof_start_line=formal_2_start_line)
                    return proof
            except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
                pass
            return None

        # to avoid losing time, we first check if the formalizations are well-typed
        if check_proof_sub("sorry") is None:
            break

        # 1. Use `exact?` tactic. We check if it is using the base theorem in the proof.
        proof_exact = check_proof_sub("exact?")
        if proof_exact and base_thm_name in proof_exact:
            res.beql_unidirections = update_tuple(res.beql_unidirections, i, proof_exact)
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_exact)
            continue

        # 2. try to apply the base theorem directly
        proof_apply = check_proof_sub(f"apply {base_thm_name}\n" + proof_all_apply)
        if proof_apply:
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_apply)
            continue

        # 3. try to add the conlusion of the base theorem as hypothesis
        # sanity check: if we can prove `reform_thm` using a tactic in `solver_tactics_have` without introducing the hypothesis,
        # then we should skip this `have` step as it may introduce a false positive
        # drawback of `have` strategy: variable names/types must match exactly
        provable_without_have = False
        try:
            res_without_have = lean_server.run(
                Command(cmd=formal_2_code + proof_all_have, env=context_env), timeout=timeout_per_proof
            )
            if isinstance(res_without_have, CommandResponse):
                provable_without_have = res_without_have.lean_code_is_valid(allow_sorry=False)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
            pass

        if not provable_without_have:
            idx_conclusion = split_conclusion(formal_1_code)
            if idx_conclusion:
                idx_end_conclusion = formal_1_code.rfind(":=")
                conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
                have_stmt_proof = (
                        f"have {conclusion} := by\n"
                        + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                        + "\n"
                )
                proof_have = check_proof_sub(have_stmt_proof + proof_all_have)
                if proof_have:
                    res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_have)
                    continue

        # 4. try to apply the base theorem with some tolerance on the differences in the conclusion
        for max_step in range(0, 5):
            proof_convert = check_proof_sub(
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply
            )
            if proof_convert:
                res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_convert)
                break

        if not res.beq_plus_unidirections[i]:
            break

    return res


def check_theorem_eq_server(
        theorem1: str, theorem2: str, context: str, project: str
) -> BEqCPUResult:
    base_thm_name = "base_theorem"
    reformulated_thm_name = "reformulated_theorem"

    def prove_all(tactics: list[str]) -> str:
        prove_independent = " ; ".join([f"(all_goals try {t})" for t in tactics])
        prove_combined = "all_goals (" + " ; ".join([f"(try {t})" for t in tactics]) + ")"
        return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

    solver_tactics_apply = ["tauto", "simp_all_arith!", "noncomm_ring", "exact?"]
    solver_tactics_have = ["tauto", "simp_all_arith!", "exact? using this"]
    proof_all_apply = prove_all(solver_tactics_apply)
    proof_all_have = prove_all(solver_tactics_have)

    res = BEqCPUResult()
    for i, (base_thm, reform_thm) in enumerate([(theorem1, theorem2), (theorem2, theorem1)]):
        try:
            formal_1_code = clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True) + "\n\n"
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name, add_sorry=False)} := by"
        except ValueError:
            # we cannot change one of the theorems name, probably because it is invalid, so we skip this pair
            break

        def check_proof_sub(proof: str, formal_code: str = formal_1_code + formal_2_code) -> str | None:
            prepended_proof = "\nintros\nsymm_saturate\n"

            try:
                req = RunRequest(
                    data=Command(cmd=formal_code + indent_code(prepended_proof + proof, 2)), context=context)
                response = try_repeatedly(req, project, True)
                try:
                    lean_output = CommandResponse.model_validate(response.result)
                except ValidationError:
                    lean_output = LeanError.model_validate(response.result)

                if isinstance(lean_output, LeanError):
                    return None
                if proof == "sorry":
                    # sorry is used on purpose to check if the formalization is well-typed
                    if lean_output.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0)):
                        return proof
                    return None

                if lean_output.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0), allow_sorry=False):
                    if proof == "exact?":
                        return extract_exact_proof(lean_output, proof_start_line=formal_2_start_line)
                    return proof
            except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
                pass
            return None

        # to avoid losing time, we first check if the formalizations are well-typed
        if check_proof_sub("sorry") is None:
            break

        # 1. Use `exact?` tactic. We check if it is using the base theorem in the proof.
        proof_exact = check_proof_sub("exact?")
        if proof_exact and base_thm_name in proof_exact:
            res.beql_unidirections = update_tuple(res.beql_unidirections, i, proof_exact)
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_exact)
            continue

        # 2. try to apply the base theorem directly
        proof_apply = check_proof_sub(f"apply {base_thm_name}\n" + proof_all_apply)
        if proof_apply:
            res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_apply)
            continue

        # 3. try to add the conlusion of the base theorem as hypothesis
        # sanity check: if we can prove `reform_thm` using a tactic in `solver_tactics_have` without introducing the hypothesis,
        # then we should skip this `have` step as it may introduce a false positive
        # drawback of `have` strategy: variable names/types must match exactly
        provable_without_have = False
        try:
            req = RunRequest(data=Command(cmd=formal_2_code + proof_all_have), context=context)
            result = try_repeatedly(req, project, True)
            try:
                res_without_have = CommandResponse.model_validate(result.result)
            except ValidationError:
                res_without_have = LeanError.model_validate(result.result)
            if isinstance(res_without_have, CommandResponse):
                provable_without_have = res_without_have.lean_code_is_valid(allow_sorry=False)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError):
            pass

        if not provable_without_have:
            idx_conclusion = split_conclusion(formal_1_code)
            if idx_conclusion:
                idx_end_conclusion = formal_1_code.rfind(":=")
                conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
                have_stmt_proof = (
                        f"have {conclusion} := by\n"
                        + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                        + "\n"
                )
                proof_have = check_proof_sub(have_stmt_proof + proof_all_have)
                if proof_have:
                    res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_have)
                    continue

        # 4. try to apply the base theorem with some tolerance on the differences in the conclusion
        for max_step in range(0, 5):
            proof_convert = check_proof_sub(
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply
            )
            if proof_convert:
                res.beq_plus_unidirections = update_tuple(res.beq_plus_unidirections, i, proof_convert)
                break

        if not res.beq_plus_unidirections[i]:
            break

    return res
