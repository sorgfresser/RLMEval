from typing import Optional
from pydantic import BaseModel
from lean_interact import Command, FileCommand, PickleEnvironment, UnpickleEnvironment, ProofStep, PickleProofState, \
    UnpickleProofState
from lean_interact.interface import BaseREPLQuery


class RunRequest(BaseModel):
    data: Command | FileCommand | PickleEnvironment | UnpickleEnvironment \
          | ProofStep | PickleProofState | UnpickleProofState | BaseREPLQuery
    server_id: Optional[int] = None


class ContextRequest(BaseModel):
    context: str
    server_id: Optional[int] = None


class ContextResponse(BaseModel):
    server_id: int
    env_id: int


class RunResponse(BaseModel):
    server_id: int
    result: dict
