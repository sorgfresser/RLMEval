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

from lean_interact import AutoLeanServer, LeanREPLConfig, LocalProject
from lean_interact.interface import (
    Command,
    PickleEnvironment,
    UnpickleEnvironment,
    LeanError,
)
import logging
import psutil
import os
from lean_pool_models import RunResponse, RunRequest

logger = logging.getLogger(__name__)

REPOS: dict[str, Path] = {
    "PFR": Path(__file__).parent.parent / 'traced_repos/pfr_f6bdcac2365623d3667d3ff8fd8ddb7f95ce2313/pfr',
    "FLT3": Path(__file__).parent.parent / 'traced_repos/FLT3_a199fa0467f86504a9d2f6164b0456608e586821/FLT3',
    "Carleson": Path(
        __file__).parent.parent / 'traced_repos/carleson_ec175b9008144d009269ce427b9ad43dbd70d0a5/carleson',
    "FLT": Path(__file__).parent.parent / 'traced_repos/FLT_fed5e57b05e232f3bfe24d24098111e9dcd7bcd1/FLT',
    "TestingLowerBounds": Path(
        __file__).parent.parent / 'traced_repos/testing-lower-bounds_0f09ff100a06a5e4542181514bfff74213ae126b/testing-lower-bounds',
    "PrimeNumberTheoremAnd": Path(
        __file__).parent.parent / 'traced_repos/PrimeNumberTheoremAnd_6101a4b1f0cd4096b0c41cc90c7ba89f7593ef77/PrimeNumberTheoremAnd',
}

RETRIES = 10

def _build_config(repo_root: Path) -> LeanREPLConfig:
    return LeanREPLConfig(project=LocalProject(str(repo_root), build=False), memory_hard_limit_mb=None)


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

    async def restart(self):
        self.server.restart()
        new_contexts = {}
        for key, context in self.loaded_contexts.items():
            response = await self.server.async_run(
                UnpickleEnvironment(unpickle_env_from=str(context.pickle_path)))
            if isinstance(response, LeanError):
                logger.warning("UnpickleEnvironment failed with %s", response.message)
                raise HTTPException(status_code=500,
                                    detail=f"UnpickleEnvironment failed with {response.message}")
            new_contexts[key] = LoadedContext(pickle_path=context.pickle_path, env_id=response.env)
        self.loaded_contexts = new_contexts


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
        pickle_dir = Path(self.server.config.working_dir) / "context_cache"
        pickle_dir.mkdir(parents=True, exist_ok=True)
        pickle_path = pickle_dir / f"{key}.olean"
        # Load context
        await context_lock.acquire()
        # Never before, create from scratch
        if key not in context_paths:
            context_paths[key] = pickle_path
            context_locks[key] = asyncio.Lock()
            # Get the lock for the specific context, release the overall one
            await context_locks[key].acquire()
            context_lock.release()
            # Pickle path might still exist, e.g. after a restart
            if not pickle_path.exists():
                context_response = await self.server.async_run(Command(cmd=context), add_to_session_cache=False)
                if isinstance(context_response, LeanError):
                    logger.warning("Context Command failed with %s", context_response.message)
                    raise HTTPException(status_code=400,
                                        detail=f"Command for context failed with {context_response.message}")
                assert not isinstance(context_response, LeanError)
                if not context_response.lean_code_is_valid(allow_sorry=True):
                    logger.warning("Context Command has errors! Response: %s", context_response)
                    raise HTTPException(status_code=400,
                                        detail=f"Command for context failed with errors: {context_response.get_errors()}")
                self.loaded_contexts[key] = LoadedContext(pickle_path=pickle_path, env_id=context_response.env)
                # Pickle the specific context, then release
                request = PickleEnvironment(env=context_response.env, pickle_to=str(pickle_path))
                response = await self.server.async_run(request)
                context_locks[key].release()
                if isinstance(response, LeanError):
                    logger.warning("PickleEnvironment failed with %s.\nRequest: %s.\nContext response: %s",
                                   response.message, request, context_response)
                    raise HTTPException(status_code=500, detail=f"PickleEnvironment failed with {response.message}")
            else:
                context_locks[key].release()
                try:
                    response = await self.server.async_run(
                        UnpickleEnvironment(unpickle_env_from=str(context_paths[key])))
                # If unpickling fails, delete the element from cache
                except (ConnectionAbortedError, TimeoutError):
                    async with context_lock:
                        async with context_locks[key]:
                            del context_paths[key]
                            del context_locks[key]
                            pickle_path.unlink()
                    raise
                if isinstance(response, LeanError):
                    logger.warning("UnpickleEnvironment failed with %s", response.message)
                    raise HTTPException(status_code=500,
                                        detail=f"UnpickleEnvironment failed with {response.message}")
                self.loaded_contexts[key] = LoadedContext(pickle_path=context_paths[key], env_id=response.env)
        else:
            context_lock.release()
            await context_locks[key].acquire()  # this is only needed to wait if we're still pickling
            context_locks[key].release()
            response = await self.server.async_run(
                UnpickleEnvironment(unpickle_env_from=str(context_paths[key])), verbose=False)
            assert not isinstance(response, LeanError)
            assert response.lean_code_is_valid(allow_sorry=True)
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
        env_id = None
        for _ in range(RETRIES):
            try:
                env_id = (await wrapper.get_env(ctx, self._context_lock, self._context_cache, self._context_locks)).env_id
                break
            except (TimeoutError, ConnectionAbortedError):
                await wrapper.restart()
        if env_id is None:
            raise RuntimeError("Could not get env!")
        return env_id


_POOLS: dict[str, _ProjectPool] = {
    name: _ProjectPool(_build_config(path)) for name, path in REPOS.items()
}
resetting = False
resetting_lock = asyncio.Lock()

def _kill_child_repls() -> None:
    """Terminate any descendant process whose executable path ends in 'lake/build/bin/repl'."""
    root = psutil.Process(os.getpid())
    repl_children = []
    for child in root.children(recursive=True):
        try:
            # Skip if already a zombie
            if child.status() == psutil.STATUS_ZOMBIE:
                logger.debug("Skipping zombie pid %s", child.pid)
                continue
            if child.exe().endswith(".lake/build/bin/repl"):
                repl_children.append(child)
        except (psutil.ZombieProcess, psutil.NoSuchProcess, psutil.AccessDenied):
            # Already dead or inaccessible, just ignore
            continue

    for p in repl_children:
        logger.debug("Sending SIGTERM to repl pid %s (ppid %s)", p.pid, p.ppid())
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    gone, alive = psutil.wait_procs(repl_children, timeout=10)

    for p in alive:
        logger.info("Process %s still alive – sending SIGKILL", p.pid)
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

app = FastAPI(title="Lean-Interact Pool API")

@app.post(
    "/{project}/run",
    response_model=RunResponse,
    summary="Run an arbitrary Lean-REPL request"
)
async def run(project: str, req: RunRequest):
    if resetting:
        logger.warning("Is currently resetting!")
        return HTTPException(status_code=503, detail="Currently resetting!")

    pool = _POOLS.get(project)
    if pool is None:
        raise HTTPException(404, f"Unknown project {project!r}")

    sid, wrapper = await pool.acquire_server(None)
    async with wrapper.lock:
        env_id = None
        if req.context is not None:
            if not req.context:
                raise HTTPException(400, "Empty context provided, pass None or a string!")
            env_id = await pool.get_or_create_env(req.context, wrapper)
        data = req.data
        if isinstance(data, Command):
            data = data.model_copy(deep=True, update={"env": env_id})

        result = None
        for _ in range(2):
            try:
                result = await wrapper.server.async_run(data, timeout=20, verbose=False)
            except (TimeoutError, ConnectionAbortedError) as e:
                logger.warning(e)
                await wrapper.restart()
                env_id = await pool.get_or_create_env(req.context, wrapper)
                if isinstance(data, Command):
                    data = data.model_copy(deep=True, update={"env": env_id})
    if result is None:
        raise HTTPException(400, "Data did not return valid result!")
    # Ensure JSON-serialisable
    if hasattr(result, "model_dump"):
        result = result.model_dump(by_alias=True)
    return RunResponse(server_id=sid, result=result)


@app.post("/reset")
async def reset():
    global _POOLS
    global resetting
    global resetting_lock
    async with resetting_lock:
        resetting = True
        _kill_child_repls()
        _POOLS = {
            name: _ProjectPool(_build_config(path)) for name, path in REPOS.items()
        }
        resetting = False
    return
