from typing import Optional
from pydantic import BaseModel
from lean_interact import Command, FileCommand, ProofStep, PickleProofState, UnpickleProofState
from lean_interact.interface import BaseREPLQuery


class RunRequest(BaseModel):
    data: Command | FileCommand | ProofStep | PickleProofState | UnpickleProofState | BaseREPLQuery
    context: Optional[str] = None


class RunResponse(BaseModel):
    server_id: int
    result: dict
