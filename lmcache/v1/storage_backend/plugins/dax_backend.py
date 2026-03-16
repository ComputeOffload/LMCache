# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence
import asyncio
import ctypes
import mmap
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import (
    StoragePluginInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


@dataclass
class _Entry:
    """In-memory index entry for a stored chunk."""

    offset: int
    size: int
    meta: DiskCacheMetadata
    slot_id: int
    generation: int


@dataclass
class _Inflight:
    """In-progress put operation tracking."""

    offset: int
    meta: DiskCacheMetadata
    slot_id: int
    generation: int
    canceled: bool = False


@dataclass
class _SlotState:
    """Slot state for a stored DAX chunk."""

    generation: int
    committed: bool = False
    borrow_count: int = 0
    pending_free: bool = False


@dataclass(frozen=True)
class _ArenaHandle:
    """Borrow token used while reading from a DAX slot."""

    slot_id: int
    generation: int


class _ArenaState:
    """Owns the DAX mmap and releases it when the backend closes."""

    def __init__(
        self,
        fd: int,
        mmap_obj: mmap.mmap,
        base_ptr: int,
        arena_view: memoryview,
        arena_tensor: torch.Tensor,
    ) -> None:
        self._fd: Optional[int] = fd
        self._mmap_obj: Optional[mmap.mmap] = mmap_obj
        self._base_ptr = base_ptr
        self._arena_view: Optional[memoryview] = arena_view
        self._arena_tensor: Optional[torch.Tensor] = arena_tensor
        self._released = False
        self._lock = threading.Lock()

    def release_owner(self) -> None:
        with self._lock:
            if self._released:
                return
            fd = self._fd
            mmap_obj = self._mmap_obj
            arena_view = self._arena_view
            self._fd = None
            self._mmap_obj = None
            self._arena_view = None
            self._arena_tensor = None
            self._base_ptr = 0
            self._released = True

        resources = (fd, mmap_obj, arena_view)
        if resources is not None:
            self._cleanup_resources(*resources)

    def snapshot(self) -> tuple[
        Optional[int],
        Optional[mmap.mmap],
        int,
        Optional[memoryview],
        Optional[torch.Tensor],
    ]:
        with self._lock:
            return (
                self._fd,
                self._mmap_obj,
                self._base_ptr,
                self._arena_view,
                self._arena_tensor,
            )

    @staticmethod
    def _cleanup_resources(
        fd: Optional[int],
        mmap_obj: Optional[mmap.mmap],
        arena_view: Optional[memoryview],
    ) -> None:
        if arena_view is not None:
            try:
                arena_view.release()
            except Exception as e:
                logger.warning("Failed to release DAX memoryview: %s", e)

        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception as e:
                logger.warning("Failed to close DAX mmap: %s", e)

        if fd is not None:
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Failed to close DAX fd: %s", e)


@dataclass
class _BackendOpToken:
    """Tracks an in-flight backend operation that must quiesce during close."""

    backend: "DaxBackend"
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.backend._finish_backend_op()


class DaxBackend(StoragePluginInterface):
    """Storage plugin backend for /dev/dax mmap-backed KV cache."""

    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
        local_cpu_backend: Optional[LocalCPUBackend] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cpu",
    ) -> None:
        """Initialize a DAX-backed storage backend."""
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        if self.config is None:
            raise ValueError("DaxBackend requires config")
        if self.metadata is None:
            raise ValueError("DaxBackend requires metadata")

        if self.metadata.world_size != 1:
            raise ValueError(
                "DaxBackend currently only supports TP=1 "
                f"(world_size={self.metadata.world_size})"
            )
        if self.metadata.get_num_groups() != 1:
            raise ValueError(
                "DaxBackend currently supports only single-group KV layout"
            )

        extra = self.config.extra_config or {}
        self.device_path = str(extra.get("dax.device_path", "")).strip()
        if not self.device_path:
            raise ValueError("extra_config['dax.device_path'] is required")

        # Optional async put path. Disabled by default to avoid loop/thread
        # coupling issues on platforms where cross-thread loop wakeups are limited.
        self.async_put = _to_bool(extra.get("dax.async_put", False))
        if self.async_put and self.loop is None:
            raise ValueError("DaxBackend async_put=true requires an asyncio event loop")

        self.arena_size_gb = float(extra.get("dax.arena_size_gb", 0))
        if self.arena_size_gb <= 0:
            raise ValueError("extra_config['dax.arena_size_gb'] must be > 0")

        if self.local_cpu_backend is None:
            raise ValueError("DaxBackend requires local_cpu_backend")

        self._arena_bytes = int(self.arena_size_gb * 1024**3)
        if self._arena_bytes <= 0:
            raise ValueError("dax.arena_size_gb results in zero-sized arena")

        self._fd: Optional[int] = None
        self._mmap_obj: Optional[mmap.mmap] = None
        self._base_ptr: int = 0
        self._arena_view: Optional[memoryview] = None
        self._arena_tensor: Optional[torch.Tensor] = None
        self._arena_state: Optional[_ArenaState] = None
        self._open_arena()
        try:
            assert self.local_cpu_backend is not None
            full_chunk_size = int(self.local_cpu_backend.get_full_chunk_size_bytes())
            self.slot_bytes = max(1, int(full_chunk_size))
            self._max_slots = self._arena_bytes // self.slot_bytes
            if self._max_slots <= 0:
                raise RuntimeError("DAX arena too small for configured chunk slot size")

            self._state_lock = threading.RLock()
            self._state_condition = threading.Condition(self._state_lock)

            self._index: dict[CacheEngineKey, _Entry] = {}
            self._pinned: set[CacheEngineKey] = set()
            self._inflight: dict[CacheEngineKey, _Inflight] = {}
            self._lru: "OrderedDict[CacheEngineKey, None]" = OrderedDict()
            self._slot_states: dict[int, _SlotState] = {}
            self._slot_generations: dict[int, int] = {}

            self._next_slot = 0
            self._free_slots: list[int] = []
            self._reserved_slots: set[int] = set()
            self._put_tasks: set[CacheEngineKey] = set()
            self._async_futures: set[Future] = set()
            self._active_ops = 0
            self._active_puts = 0
            self._closing = False
            self._closed = False

            logger.info(
                "DaxBackend init: device=%s arena=%d slot=%d max_slots=%d",
                self.device_path,
                self._arena_bytes,
                self.slot_bytes,
                self._max_slots,
            )
        except Exception:
            self._cleanup_arena()
            raise

    def __str__(self) -> str:
        return "DaxBackend"

    def _cleanup_arena(self) -> None:
        arena_state = self._arena_state
        self._detach_backend_arena()
        self._arena_state = None
        if arena_state is not None:
            arena_state.release_owner()

    def _bind_backend_arena(self, arena_state: _ArenaState) -> None:
        (
            self._fd,
            self._mmap_obj,
            self._base_ptr,
            self._arena_view,
            self._arena_tensor,
        ) = arena_state.snapshot()

    def _detach_backend_arena(self) -> None:
        self._fd = None
        self._mmap_obj = None
        self._base_ptr = 0
        self._arena_view = None
        self._arena_tensor = None

    def _next_generation_locked(self, slot_id: int) -> int:
        generation = self._slot_generations.get(slot_id, 0) + 1
        self._slot_generations[slot_id] = generation
        return generation

    def _reserve_slot_state_locked(self, slot_id: int) -> int:
        generation = self._next_generation_locked(slot_id)
        self._slot_states[slot_id] = _SlotState(generation=generation)
        return generation

    def _begin_backend_op(self) -> Optional[_BackendOpToken]:
        with self._state_lock:
            if self._closing:
                return None
            self._active_ops += 1
        return _BackendOpToken(self)

    def _finish_backend_op(self) -> None:
        with self._state_lock:
            if self._active_ops > 0:
                self._active_ops -= 1
            else:
                logger.warning("DaxBackend active op count underflow")
            self._state_condition.notify_all()

    def _mark_slot_committed_locked(self, slot_id: int, generation: int) -> None:
        state = self._slot_states.get(slot_id)
        if state is None or state.generation != generation:
            return
        state.committed = True
        state.pending_free = False

    def _schedule_slot_reclaim_locked(self, slot_id: int, generation: int) -> None:
        state = self._slot_states.get(slot_id)
        if state is None or state.generation != generation:
            return
        state.committed = False
        if state.borrow_count == 0:
            self._slot_states.pop(slot_id, None)
            self._free_slot_locked(slot_id)
        else:
            state.pending_free = True

    def _acquire_borrow_handle_locked(self, entry: _Entry) -> Optional[_ArenaHandle]:
        state = self._slot_states.get(entry.slot_id)
        if state is None or state.generation != entry.generation or not state.committed:
            return None
        state.borrow_count += 1
        return _ArenaHandle(slot_id=entry.slot_id, generation=entry.generation)

    def _finalize_inflight_locked(
        self,
        key: CacheEngineKey,
        write_failed: bool,
    ) -> bool:
        inflight = self._inflight.pop(key, None)
        if inflight is None:
            return False
        if inflight.canceled or write_failed:
            self._schedule_slot_reclaim_locked(inflight.slot_id, inflight.generation)
            return False
        self._reserved_slots.discard(inflight.slot_id)
        self._mark_slot_committed_locked(inflight.slot_id, inflight.generation)
        self._index[key] = _Entry(
            offset=inflight.offset,
            size=inflight.meta.size,
            meta=inflight.meta,
            slot_id=inflight.slot_id,
            generation=inflight.generation,
        )
        self._touch_locked(key)
        return True

    def _release_arena_handle_locked(self, handle: _ArenaHandle) -> None:
        state = self._slot_states.get(handle.slot_id)
        if state is None or state.generation != handle.generation:
            return

        if state.borrow_count > 0:
            state.borrow_count -= 1
        if state.pending_free and state.borrow_count == 0:
            self._slot_states.pop(handle.slot_id, None)
            self._free_slot_locked(handle.slot_id)

    def _touch_if_current_locked(
        self,
        key: CacheEngineKey,
        slot_id: int,
        generation: int,
    ) -> None:
        current = self._index.get(key)
        if (
            current is not None
            and current.slot_id == slot_id
            and current.generation == generation
        ):
            self._touch_locked(key)

    def _discard_async_future(self, future: Future) -> None:
        with self._state_lock:
            self._async_futures.discard(future)

    def _begin_put_task(self, key: CacheEngineKey) -> bool:
        with self._state_lock:
            if self._closing:
                raise RuntimeError("DaxBackend is closing")
            if key in self._put_tasks:
                return False
            self._put_tasks.add(key)
            self._active_puts += 1
            return True

    def _finish_put_task(self, key: CacheEngineKey) -> None:
        with self._state_lock:
            self._put_tasks.discard(key)
            if self._active_puts > 0:
                self._active_puts -= 1
            else:
                logger.warning("DaxBackend active put count underflow for key %s", key)
            self._state_condition.notify_all()

    def _invoke_on_complete_callback(
        self,
        key: CacheEngineKey,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        if on_complete_callback is None:
            return
        try:
            on_complete_callback(key)
        except Exception as e:
            logger.warning("on_complete_callback failed for key %s: %s", key, e)

    def _open_arena(self) -> None:
        fd: Optional[int] = None
        mmap_obj: Optional[mmap.mmap] = None
        arena_view: Optional[memoryview] = None
        try:
            fd = os.open(self.device_path, os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                f"Failed to open dax device {self.device_path}: {e}"
            ) from e
        try:
            try:
                # Best effort capacity check for file-backed tests and many dax setups.
                capacity_bytes = os.fstat(fd).st_size
                if capacity_bytes > 0 and self._arena_bytes > capacity_bytes:
                    raise RuntimeError(
                        f"dax.arena_size_gb ({self._arena_bytes} bytes) exceeds "
                        f"device capacity ({capacity_bytes} bytes)"
                    )
            except OSError:
                # Some dax devices may not report size via fstat.
                pass

            mmap_obj = mmap.mmap(
                fd,
                self._arena_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mmap_obj))
            arena_view = memoryview(mmap_obj)
            if hasattr(torch, "frombuffer"):
                arena_tensor = torch.frombuffer(arena_view, dtype=torch.uint8)
            else:
                # Third Party
                import numpy as np

                arr = np.frombuffer(arena_view, dtype=np.uint8)
                arena_tensor = torch.from_numpy(arr)
            self._arena_state = _ArenaState(
                fd=fd,
                mmap_obj=mmap_obj,
                base_ptr=base_ptr,
                arena_view=arena_view,
                arena_tensor=arena_tensor,
            )
            self._bind_backend_arena(self._arena_state)
        except Exception as e:
            _ArenaState._cleanup_resources(fd, mmap_obj, arena_view)
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(
                f"Failed to mmap dax arena ({self._arena_bytes} bytes) from "
                f"{self.device_path}: {e}"
            ) from e

    def _allocate_slot_locked(self) -> int:
        if self._free_slots:
            slot = self._free_slots.pop()
            self._reserved_slots.add(slot)
            return slot * self.slot_bytes
        if self._next_slot < self._max_slots:
            slot = self._next_slot
            self._next_slot += 1
            self._reserved_slots.add(slot)
            return slot * self.slot_bytes
        raise RuntimeError("No free slots available; eviction required")

    def _free_slot_locked(self, slot_id: int) -> None:
        if slot_id < 0:
            return
        self._reserved_slots.discard(slot_id)
        if slot_id not in self._free_slots:
            self._free_slots.append(slot_id)

    def _touch_locked(self, key: CacheEngineKey) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None

    def _evict_one_locked(self) -> bool:
        for victim in list(self._lru.keys()):
            if victim in self._pinned or victim in self._inflight:
                continue
            entry = self._index.get(victim)
            if entry is None:
                continue
            state = self._slot_states.get(entry.slot_id)
            if (
                state is None
                or state.generation != entry.generation
                or state.borrow_count > 0
            ):
                continue
            self._index.pop(victim, None)
            self._lru.pop(victim, None)
            self._pinned.discard(victim)
            self._schedule_slot_reclaim_locked(entry.slot_id, entry.generation)
            return True
        return False


    def _do_write(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(self._base_ptr + offset),
            ctypes.c_void_p(memory_obj.data_ptr),
            ctypes.c_size_t(size),
        )

    def _do_read(self, offset: int, memory_obj: MemoryObj, size: int) -> None:
        ctypes.memmove(
            ctypes.c_void_p(memory_obj.data_ptr),
            ctypes.c_void_p(self._base_ptr + offset),
            ctypes.c_size_t(size),
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self._state_lock:
            ok = key in self._index
            if ok and pin:
                self._pinned.add(key)
            return ok

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._state_lock:
            return key in self._put_tasks

    def pin(self, key: CacheEngineKey) -> bool:
        with self._state_lock:
            if key in self._index:
                self._pinned.add(key)
                return True
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self._state_lock:
            if key in self._pinned:
                self._pinned.remove(key)
                return True
            return key in self._index

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        del force
        with self._state_lock:
            existed = key in self._index or key in self._inflight
            entry = self._index.pop(key, None)
            inflight = self._inflight.get(key)
            self._pinned.discard(key)
            self._lru.pop(key, None)
            if entry is not None:
                self._schedule_slot_reclaim_locked(entry.slot_id, entry.generation)
            if inflight is not None:
                inflight.canceled = True
            return existed

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        """Store a batch of memory objects in the DAX arena."""
        del transfer_spec
        if len(keys) != len(objs):
            raise ValueError(
                "keys and objs must have the same length, "
                f"got {len(keys)} and {len(objs)}"
            )
        futures: List[Future] = []

        for key, obj in zip(keys, objs, strict=True):
            task_started = self._begin_put_task(key)
            if not task_started:
                continue

            should_finish_task = True
            try:
                # Reject multi-tensor objects explicitly
                num_shapes = len(obj.get_shapes())
                if num_shapes > 1:
                    logger.error(
                        "DaxBackend does not support multi-tensor allocations: "
                        "key=%s has %d tensors. "
                        "Use single-tensor format or extend metadata.",
                        key,
                        num_shapes,
                    )
                    continue
                size = int(obj.get_size())
                obj_metadata = obj.metadata
                shape = obj_metadata.shape
                dtype = obj_metadata.dtype
                cached_positions = obj_metadata.cached_positions
                fmt = obj_metadata.fmt

                with self._state_lock:
                    if key in self._index or key in self._inflight:
                        continue

                    if size > self.slot_bytes:
                        logger.warning(
                            "Skipping DAX put for key %s: object size %d exceeds "
                            "slot size %d",
                            key,
                            size,
                            self.slot_bytes,
                        )
                        continue
                    while True:
                        try:
                            offset = self._allocate_slot_locked()
                            break
                        except RuntimeError:
                            if not self._evict_one_locked():
                                raise
                    slot_id = int(offset // self.slot_bytes)
                    generation = self._reserve_slot_state_locked(slot_id)

                    meta = DiskCacheMetadata(
                        path=f"{self.device_path}@{offset}",
                        size=size,
                        shape=shape,
                        dtype=dtype,
                        cached_positions=cached_positions,
                        fmt=fmt,
                        pin_count=0,
                    )

                    self._inflight[key] = _Inflight(
                        offset=offset,
                        meta=meta,
                        slot_id=slot_id,
                        generation=generation,
                        canceled=False,
                    )

                if self.async_put and self.loop is not None and self.loop.is_running():
                    obj.ref_count_up()
                    try:
                        fut = asyncio.run_coroutine_threadsafe(
                            self._submit_write(
                                key=key,
                                offset=offset,
                                size=size,
                                memory_obj=obj,
                                on_complete_callback=on_complete_callback,
                            ),
                            self.loop,
                        )
                    except Exception:
                        with self._state_lock:
                            self._finalize_inflight_locked(key, write_failed=True)
                        obj.ref_count_down()
                        raise
                    with self._state_lock:
                        self._async_futures.add(fut)
                    fut.add_done_callback(self._discard_async_future)
                    futures.append(fut)
                    should_finish_task = False
                    continue

                try:
                    self._do_write(offset, obj, size)
                except Exception as e:
                    with self._state_lock:
                        self._finalize_inflight_locked(key, write_failed=True)
                    raise RuntimeError(
                        f"DaxBackend write failed for key {key}: {e}"
                    ) from e

                with self._state_lock:
                    should_invoke_callback = self._finalize_inflight_locked(
                        key,
                        write_failed=False,
                    )

                if should_invoke_callback:
                    self._invoke_on_complete_callback(key, on_complete_callback)
            finally:
                if task_started and should_finish_task:
                    self._finish_put_task(key)

        return futures or None

    async def _submit_write(
        self,
        key: CacheEngineKey,
        offset: int,
        size: int,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        write_error: Optional[Exception] = None
        should_invoke_callback = False
        try:
            try:
                await asyncio.to_thread(self._do_write, offset, memory_obj, size)
            except Exception as e:
                write_error = e
            finally:
                with self._state_lock:
                    should_invoke_callback = self._finalize_inflight_locked(
                        key,
                        write_failed=write_error is not None,
                    )

            if write_error is not None:
                raise RuntimeError(
                    f"DaxBackend write failed for key {key}: {write_error}"
                ) from write_error

            if should_invoke_callback:
                self._invoke_on_complete_callback(key, on_complete_callback)
        finally:
            memory_obj.ref_count_down()
            self._finish_put_task(key)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Return the memory object for a key, or ``None`` if unavailable."""
        op_token = self._begin_backend_op()
        if op_token is None:
            return None

        borrow_handle: Optional[_ArenaHandle] = None
        touch_slot_id: Optional[int] = None
        touch_generation: Optional[int] = None
        try:
            with self._state_lock:
                if self._closing:
                    return None
                entry = self._index.get(key)
                if entry is None:
                    return None

                meta = entry.meta
                if meta.shape is None or meta.dtype is None:
                    return None

                borrow_handle = self._acquire_borrow_handle_locked(entry)
                if borrow_handle is None:
                    return None
                touch_slot_id = entry.slot_id
                touch_generation = entry.generation
                shape = meta.shape
                dtype = meta.dtype
                fmt = meta.fmt
                cached_positions = meta.cached_positions
                offset = entry.offset
                size = int(meta.size)

            assert self.local_cpu_backend is not None
            should_touch = False
            memory_obj: Optional[MemoryObj] = None
            try:
                memory_obj = self.local_cpu_backend.allocate(
                    shape,
                    dtype,
                    fmt,
                )
                if memory_obj is None:
                    return None
                self._do_read(offset, memory_obj, size)
                memory_obj.metadata.cached_positions = cached_positions
                should_touch = True
            except Exception:
                if memory_obj is not None:
                    memory_obj.ref_count_down()
                raise
            finally:
                if borrow_handle is not None:
                    with self._state_lock:
                        if (
                            should_touch
                            and touch_slot_id is not None
                            and touch_generation is not None
                        ):
                            self._touch_if_current_locked(
                                key,
                                touch_slot_id,
                                touch_generation,
                            )
                        self._release_arena_handle_locked(borrow_handle)
            assert memory_obj is not None
            return memory_obj
        finally:
            op_token.release()

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        del lookup_id
        hit = 0
        with self._state_lock:
            for key in keys:
                if key not in self._index:
                    break
                if pin:
                    self._pinned.add(key)
                hit += 1
        return hit

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        del lookup_id, transfer_spec
        results: list[MemoryObj] = []
        for key in keys:
            mem_obj = await asyncio.to_thread(self.get_blocking, key)
            if mem_obj is None:
                break
            results.append(mem_obj)
        return results

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        hit = 0
        with self._state_lock:
            for key in keys:
                if key not in self._index:
                    break
                if pin:
                    self._pinned.add(key)
                hit += 1
        return hit

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        removed = 0
        for key in keys:
            removed += int(self.remove(key, force=force))
        return removed

    def get_allocator_backend(self) -> LocalCPUBackend:
        if self.local_cpu_backend is None:
            raise RuntimeError("DaxBackend has no allocator backend available")
        return self.local_cpu_backend

    def close(self) -> None:
        """Quiesce outstanding operations and release the mapped DAX arena."""
        with self._state_lock:
            if self._closed:
                return
            self._closing = True
            while self._active_puts > 0 or self._active_ops > 0:
                self._state_condition.wait()
            if self._closed:
                return
            self._closed = True
            futures = list(self._async_futures)
            self._index.clear()
            self._inflight.clear()
            self._lru.clear()
            self._pinned.clear()
            self._slot_states.clear()
            self._slot_generations.clear()
            self._reserved_slots.clear()
            self._free_slots.clear()
            self._put_tasks.clear()
            self._async_futures.clear()
            arena_state = self._arena_state
            self._arena_state = None
            self._detach_backend_arena()

        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.warning("In-flight DAX async write failed during close: %s", e)

        if arena_state is not None:
            arena_state.release_owner()
