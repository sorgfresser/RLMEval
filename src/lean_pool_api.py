"""
FastAPI service that multiplexes many AutoLeanServer instances over many
servers and shares context files between them.
"""
from __future__ import annotations

import asyncio
import hashlib
from itertools import count
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from filelock import FileLock

from lean_interact import AutoLeanServer, LeanREPLConfig, LocalProject
from lean_interact.interface import (
    Command,
    FileCommand,
    PickleEnvironment,
    UnpickleEnvironment,
    ProofStep,
    PickleProofState,
    UnpickleProofState,
    BaseREPLQuery,
    CommandResponse,
    ProofStepResponse,
    LeanError,
)

REPOS: dict[str, Path] = {
    "PFR": Path('/home/tss52/PycharmProjects/RLMEval/traced_repos/pfr_f6bdcac2365623d3667d3ff8fd8ddb7f95ce2313/pfr'),
    "FLT3": Path('/home/tss52/PycharmProjects/RLMEval/traced_repos/FLT3_a199fa0467f86504a9d2f6164b0456608e586821/FLT3'),
    "Carleson": Path(
        '/home/tss52/PycharmProjects/RLMEval/traced_repos/carleson_ec175b9008144d009269ce427b9ad43dbd70d0a5/carleson'),
    "FLT": Path('/home/tss52/PycharmProjects/RLMEval/traced_repos/FLT_fed5e57b05e232f3bfe24d24098111e9dcd7bcd1/FLT'),
    "TestingLowerBounds": Path(
        '/home/tss52/PycharmProjects/RLMEval/traced_repos/testing-lower-bounds_0f09ff100a06a5e4542181514bfff74213ae126b/testing-lower-bounds'),
    "PromeNumberTheoremAnd": Path(
        '/home/tss52/PycharmProjects/RLMEval/traced_repos/PrimeNumberTheoremAnd_6101a4b1f0cd4096b0c41cc90c7ba89f7593ef77/PrimeNumberTheoremAnd'),
}


def _build_config(repo_root: Path) -> LeanREPLConfig:
    return LeanREPLConfig(
        project=LocalProject(str(repo_root)),
        memory_hard_limit_mb=25_000,
    )


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


class LoadedContext:
    def __init__(self, pickle_path: Path, env_id: int):
        self.pickle_path = pickle_path
        self.env_id = env_id


class _ServerWrapper:
    """A single AutoLeanServer and asyncio.Lock for mutual exclusion."""

    def __init__(self, cfg: LeanREPLConfig):
        self.server = AutoLeanServer(cfg)
        self.lock = asyncio.Lock()
        self.loaded_contexts: dict[str, LoadedContext] = {}

    async def get_env(self, context: str, context_lock: asyncio.Lock, context_paths: dict[str, Path],
                      context_locks: dict[str, asyncio.Lock]) -> LoadedContext:
        """Return the environment associated with a context.

        If the context does not exist yet, load it.
        Two locking mechanisms are in place here. Context_lock assures that only one new context is created at a time,
        and context_locks ensures that only one server wrapper reads a context at the same time.
        """
        key = hashlib.sha256(context.encode()).hexdigest()
        if key in self.loaded_contexts:
            return self.loaded_contexts[key]
        # Load context
        async with self.lock:
            await context_lock.acquire()
            # Never before, create from scratch
            if key not in context_paths:
                pickle_dir = Path(self.server.config.working_dir) / "context_cache"
                pickle_dir.mkdir(parents=True, exist_ok=True)
                pickle_path = pickle_dir / f"{key}.olean"
                context_paths[key] = pickle_path
                context_locks[key] = asyncio.Lock()
                # Get the lock for the specific context, release the overall one
                await context_locks[key].acquire()
                context_lock.release()
                response = await self.server.async_run(Command(cmd=context), add_to_session_cache=False)
                assert not isinstance(response, LeanError)
                self.loaded_contexts[key] = LoadedContext(pickle_path=pickle_path, env_id=response.env)
                # Pickle the specific context, then release
                response = await self.server.async_run(PickleEnvironment(env=response.env, pickle_to=str(pickle_path)))
                assert not isinstance(response, LeanError)
                context_locks[key].release()
            else:
                context_lock.release()
                async with context_locks[key]:
                    response = await self.server.async_run(
                        UnpickleEnvironment(unpickle_env_from=str(context_paths[key])))
                assert not isinstance(response, LeanError)
                self.loaded_contexts[key] = LoadedContext(pickle_path=context_paths[key], env_id=response.env)
        return self.loaded_contexts[key]


class _ProjectPool:
    """Pool of AutoLeanServer instances for one repository.
    
    Uses an asyncio.Lock to protect its map while picking/creating servers.
    Holds a context cache, i.e. sha256(context) to pickle_path.
    """

    def __init__(self, cfg: LeanREPLConfig):
        self.cfg = cfg
        self.servers: dict[int, _ServerWrapper] = {}
        self._id_gen = count()
        self._registry_lock = asyncio.Lock()
        self._context_cache: dict[str, Path] = {}
        self._context_lock = asyncio.Lock()
        self._context_locks: dict[str, asyncio.Lock] = {}

    async def acquire_server(
            self, explicit_sid: Optional[int]
    ) -> Tuple[int, _ServerWrapper]:
        async with self._registry_lock:
            if explicit_sid is not None:
                try:
                    return explicit_sid, self.servers[explicit_sid]
                except KeyError:
                    raise HTTPException(404, f"Server {explicit_sid} not found")
            # Any free one
            for sid, wrap in self.servers.items():
                if not wrap.lock.locked():
                    return sid, wrap

            # all busy, start new server
            sid = next(self._id_gen)
            self.servers[sid] = _ServerWrapper(self.cfg)
            return sid, self.servers[sid]

    async def get_or_create_env(
            self, ctx: str, wrapper: _ServerWrapper
    ) -> int:
        """Returns an env_id that is present in the server"""
        return (await wrapper.get_env(ctx, self._context_lock, self._context_cache, self._context_locks)).env_id


_POOLS: dict[str, _ProjectPool] = {
    name: _ProjectPool(_build_config(path)) for name, path in REPOS.items()
}

app = FastAPI(title="Lean-Interact Pool API")


@app.post(
    "/{project}/context",
    response_model=ContextResponse,
    summary="Compile a context and get back (server_id, env_id)"
)
async def load_context(project: str, req: ContextRequest):
    pool = _POOLS.get(project)
    if pool is None:
        raise HTTPException(404, f"Unknown project {project!r}")
    if not req.context:
        raise HTTPException(400, "No context provided")

    sid, wrapper = await pool.acquire_server(req.server_id)

    env_id = await pool.get_or_create_env(req.context, wrapper)

    return ContextResponse(server_id=sid, env_id=env_id)


@app.post(
    "/{project}/run",
    response_model=RunResponse,
    summary="Run an arbitrary Lean-REPL request"
)
async def run(project: str, req: RunRequest):
    pool = _POOLS.get(project)
    if pool is None:
        raise HTTPException(404, f"Unknown project {project!r}")

    sid, wrapper = await pool.acquire_server(req.server_id)

    async with wrapper.lock:
        result = await wrapper.server.async_run(req.data, timeout=300, )
        # Ensure JSON-serialisable
        if hasattr(result, "model_dump"):
            result = result.model_dump(by_alias=True)

    return RunResponse(server_id=sid, result=result)
