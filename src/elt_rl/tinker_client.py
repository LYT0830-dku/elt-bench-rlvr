"""Pinned SDK 0.39.2 dispatch policy and stock HTTPX transport.

Apply to a service holder BEFORE creating a model client. Service-holder metadata
bootstrap may already have used the SDK's retry policy; no sampling/optimizer
submission is made by this adapter. Pin/version-test these private APIs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
import math
import types

import httpx
from tinker._base_client import AsyncHttpxClientWrapper
from tinker._client import AsyncTinker
from tinker.lib.retry_handler import RetryConfig


class DispatchBackoffRejected(RuntimeError):
    """The SDK requested a billing/capacity backoff; no resubmission is allowed."""


def _holder_pools(holder):
    pools = list(holder._client_pools.values())
    if holder._auth_pool is not None:
        pools.append(holder._auth_pool)
    return list(dict.fromkeys(pools))


def _is_stock_httpx(client):
    wrapped = client._client
    if type(wrapped) is not AsyncHttpxClientWrapper or wrapped.is_closed:
        return False
    transports = [wrapped._transport, *wrapped._mounts.values()]
    return all(
        transport is None
        or (type(transport) is httpx.AsyncHTTPTransport and transport._pool._retries == 0)
        for transport in transports
    )


async def _finish_transport_cleanup(holder):
    cleanup = getattr(holder, "_elt_transport_cleanup", None)
    if cleanup is None:
        return
    # The holder stores this cleanup future so close can finish if this caller
    # is cancelled. A later harden call must observe any cleanup failure.
    results = await asyncio.shield(cleanup)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError("Tinker HTTP client cleanup failed; do not dispatch") from failures[0]
    holder._elt_transport_cleanup = None


async def harden_holder_on_own_loop(holder, *, read_timeout_seconds=None):
    if importlib.metadata.version("tinker") != "0.39.2":
        raise RuntimeError("No-retry adapter is only validated for tinker==0.39.2")
    loop = asyncio.get_running_loop()
    if loop is not holder.get_loop():
        raise RuntimeError("Harden the Tinker holder only on its own event loop")
    if holder._constructor_kwargs.get("http_client") is not None:
        raise RuntimeError("Expected SDK-owned HTTP clients; do not dispatch")

    async def once(self, function, *args, **kwargs):
        return await function(*args, **kwargs)

    def never_pause(self, *args, **kwargs):
        return False

    timeout = httpx.Timeout(None)
    if read_timeout_seconds is not None:
        if not math.isfinite(read_timeout_seconds) or read_timeout_seconds <= 0:
            raise ValueError("HTTP read timeout must be positive and finite")
        timeout = httpx.Timeout(60.0, connect=5.0, read=float(read_timeout_seconds))
    holder._constructor_kwargs["timeout"] = timeout

    holder.execute_with_retries = types.MethodType(once, holder)
    holder._should_pause_on_billing = types.MethodType(never_pause, holder)
    holder._constructor_kwargs["max_retries"] = 0
    holder._client_config = holder._client_config.model_copy(
        update={"use_pyqwest_transport": False}
    )
    holder._constructor_kwargs["_client_config"] = holder._client_config
    lock = getattr(holder, "_elt_transport_harden_lock", None)
    if lock is None:
        lock = holder._elt_transport_harden_lock = asyncio.Lock()
    async with lock:
        await _finish_transport_cleanup(holder)
        construction_error = None
        while True:
            # close() can mark the holder closed from another thread. Hold its
            # own close lock through the final idle check, construction and swap.
            with holder._close_lock:
                if holder._closed:
                    raise RuntimeError("Cannot harden a closed Tinker holder")
                pools = _holder_pools(holder)
                for pool in pools:
                    if pool._closed:
                        raise RuntimeError("Cannot harden a closed Tinker connection pool")
                    if pool._constructor_kwargs.get("http_client") is not None:
                        raise RuntimeError("Expected SDK-owned HTTP clients; do not dispatch")
                    pool._constructor_kwargs["max_retries"] = 0
                    pool._constructor_kwargs["timeout"] = timeout
                    config = pool._constructor_kwargs.get("_client_config", holder._client_config)
                    pool._constructor_kwargs["_client_config"] = config.model_copy(
                        update={"use_pyqwest_transport": False}
                    )
                    if len(pool._clients) != len(pool._client_active_refcount):
                        raise RuntimeError("Tinker pool reference counts do not match clients")
                    for client in pool._clients:
                        client.max_retries = 0
                        client.timeout = timeout
                        client._client.timeout = timeout
                targets = [
                    (pool, index, client)
                    for pool in pools
                    for index, client in enumerate(pool._clients)
                    if not _is_stock_httpx(client)
                ]
                if all(pool._client_active_refcount[index] == 0 for pool, index, _ in targets):
                    replacements = []
                    try:
                        for pool, index, old in targets:
                            new = AsyncTinker(**pool._constructor_kwargs)
                            replacements.append((pool, index, old, new))
                            if not _is_stock_httpx(new):
                                raise RuntimeError("SDK did not construct stock HTTPX")
                    except BaseException as exc:
                        construction_error = exc
                    if construction_error is None:
                        # No await between checking idle and committing every swap.
                        for pool, index, _old, new in replacements:
                            pool._clients[index] = new
                        for pool in pools:
                            for client in pool._clients:
                                client._client_config = pool._constructor_kwargs["_client_config"]
                    break
            await asyncio.sleep(0.05)
        clients_to_close = [
            new if construction_error is not None else old for _, _, old, new in replacements
        ]
        holder._elt_transport_cleanup = asyncio.gather(
            *(client.close() for client in clients_to_close), return_exceptions=True
        )
        await _finish_transport_cleanup(holder)
        if construction_error is not None:
            raise construction_error
        with holder._close_lock:
            if holder._closed:
                raise RuntimeError("Tinker holder closed during transport cleanup; do not dispatch")
    return {
        "sdk": "0.39.2",
        "holder_dispatch_retries": 0,
        "low_level_http_retries": 0,
        "billing_pause": False,
        "http_transport": "stock_httpx",
        "http_transport_retries": 0,
        "http_clients_rebuilt": len(replacements),
        "http_read_timeout_seconds": read_timeout_seconds,
    }


async def harden_service(service, *, read_timeout_seconds=None):
    holder = service.holder
    return await holder.run_coroutine_threadsafe(
        harden_holder_on_own_loop(holder, read_timeout_seconds=read_timeout_seconds)
    )


async def harden_sampler_on_own_loop(sampler, *, read_timeout_seconds=None):
    await harden_holder_on_own_loop(sampler.holder, read_timeout_seconds=read_timeout_seconds)
    sampler.retry_handler.config = dataclasses.replace(
        sampler.retry_handler.config, enable_retry_logic=False, enable_stuck_detection=False
    )
    if not getattr(sampler, "_elt_no_retry_wrapped", False):
        original = sampler._send_asample_request

        async def send_once(*args, **kwargs):
            result = await original(*args, **kwargs)
            if result is None:
                raise DispatchBackoffRejected(
                    "Sampling dispatch asked for capacity/billing backoff; stop and reconcile before another request"
                )
            return result

        sampler._send_asample_request = send_once
        sampler._elt_no_retry_wrapped = True
    return {
        "sampling_retry_handler": False,
        "sampling_backoff_resubmission": False,
        "sampling_stuck_detection": False,
    }


def no_retry_config():
    return RetryConfig(enable_retry_logic=False, enable_stuck_detection=False)


async def harden_sampler(sampler, *, read_timeout_seconds=None):
    return await sampler.holder.run_coroutine_threadsafe(
        harden_sampler_on_own_loop(sampler, read_timeout_seconds=read_timeout_seconds)
    )
