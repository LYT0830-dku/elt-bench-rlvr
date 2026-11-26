"""Offline transport/no-retry regression tests using real SDK client construction."""

import asyncio
import threading
import types

import httpx
import pytest
from tinker._client import AsyncTinker
from tinker.lib._auth_token_provider import ApiKeyAuthProvider
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType as PoolType
from tinker.lib.internal_client_holder import InternalClientHolder
from tinker.types.client_config_response import ClientConfigResponse

from elt_rl import tinker_client as adapter


@pytest.fixture(autouse=True)
def forbid_requests(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("All HTTP requests are forbidden in this offline proof")

    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(AsyncTinker, "request", forbidden)


class FakeHolder:
    _create_client_connection_pool = InternalClientHolder._create_client_connection_pool
    _get_client_connection_pool = InternalClientHolder._get_client_connection_pool

    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._client_config = ClientConfigResponse(sample_max_concurrent_requests=321)
        self._constructor_kwargs = {
            "base_url": "https://example.invalid",
            "default_headers": {"X-Offline-Proof": "preserve"},
            "_strict_response_validation": True,
            "_client_config": self._client_config,
        }
        self._default_auth = ApiKeyAuthProvider("tml-offline-constructor-only")
        self._client_pools = {}
        self._auth_pool = None
        self._close_lock = threading.Lock()
        self._closed = False

    def get_loop(self):
        return self._loop

    def pool(self, kind):
        pool = self._get_client_connection_pool(kind)
        with pool.aclient() as client:
            pass
        return pool, client

    async def cleanup(self):
        pending = getattr(self, "_elt_transport_cleanup", None)
        if pending is not None:
            await pending
        for pool in adapter._holder_pools(self):
            await pool.close()


def test_all_existing_pools_and_auth_rebuilt_once():
    async def case():
        holder = FakeHolder()
        original = {}
        for kind in PoolType:
            pool, client = holder.pool(kind)
            original[pool] = client
        holder._auth_pool = holder._create_client_connection_pool(1, holder._default_auth)
        with holder._auth_pool.aclient() as client:
            original[holder._auth_pool] = client
        try:
            result = await adapter.harden_holder_on_own_loop(holder)
            # Redirect support is already stock in the unmodified SDK.
            assert result["http_clients_rebuilt"] == len(original) - 1
            assert result["holder_dispatch_retries"] == result["low_level_http_retries"] == 0
            identities = []
            for pool, old in original.items():
                new = pool._clients[0]
                if old is not new:
                    assert old._client.is_closed
                assert adapter._is_stock_httpx(new)
                assert new.max_retries == pool._constructor_kwargs["max_retries"] == 0
                assert new._auth is holder._default_auth
                assert new._custom_headers["X-Offline-Proof"] == "preserve"
                assert not new._client_config.use_pyqwest_transport
                assert (
                    new._client_config.sample_max_concurrent_requests
                    == pool._constructor_kwargs["_client_config"].sample_max_concurrent_requests
                )
                identities.append(id(new))
            # Reentry must not wait for or close stock clients with active long
            # polls (retrieve_futures) or metadata requests.
            for pool in original:
                pool._client_active_refcount[0] = 1
            second = await adapter.harden_holder_on_own_loop(holder)
            assert second["http_clients_rebuilt"] == 0
            assert identities == [id(pool._clients[0]) for pool in original]
            for pool in original:
                assert not pool._clients[0]._client.is_closed
                pool._client_active_refcount[0] = 0
            calls = []

            async def fails():
                calls.append(1)
                raise RuntimeError("one dispatch")

            with pytest.raises(RuntimeError, match="one dispatch"):
                await holder.execute_with_retries(fails)
            assert calls == [1]
            assert holder._should_pause_on_billing(402, "billing") is False
        finally:
            await holder.cleanup()

    asyncio.run(case())


def test_future_pool_types_and_new_clients_use_stock():
    async def case():
        holder = FakeHolder()
        try:
            holder.pool(PoolType.SESSION)
            await adapter.harden_holder_on_own_loop(holder)
            assert holder._client_config.sample_max_concurrent_requests == 321
            for kind in PoolType:
                pool, client = holder.pool(kind)
                assert adapter._is_stock_httpx(client)
                assert client.max_retries == 0
                assert not pool._constructor_kwargs["_client_config"].use_pyqwest_transport
        finally:
            await holder.cleanup()

    asyncio.run(case())


def test_cancelled_idle_wait_never_closes_or_replaces_inflight():
    async def case():
        holder = FakeHolder()
        pool, old = holder.pool(PoolType.SAMPLE)
        pool._client_active_refcount[0] = 1
        pending = asyncio.create_task(adapter.harden_holder_on_own_loop(holder))
        try:
            await asyncio.sleep(0)
            assert not pending.done()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert pool._clients == [old]
            assert not old._client.is_closed
            assert old.max_retries == 0
            assert not pool._constructor_kwargs["_client_config"].use_pyqwest_transport
        finally:
            if not pending.done():
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            pool._client_active_refcount[0] = 0
            await holder.cleanup()

    asyncio.run(case())


def test_wait_rescans_pools_created_by_background_work():
    async def case():
        holder = FakeHolder()
        pool, old = holder.pool(PoolType.SAMPLE)
        pool._client_active_refcount[0] = 1
        added = []

        async def background():
            await asyncio.sleep(0.01)
            added.append(holder.pool(PoolType.TELEMETRY))
            pool._client_active_refcount[0] = 0

        work = asyncio.create_task(background())
        try:
            await adapter.harden_holder_on_own_loop(holder)
            await work
            assert old._client.is_closed
            assert adapter._is_stock_httpx(added[0][1])
            assert added[0][0]._constructor_kwargs["max_retries"] == 0
        finally:
            await work
            await holder.cleanup()

    asyncio.run(case())


def test_constructor_failure_keeps_originals_and_closes_uncommitted(monkeypatch):
    async def case():
        holder = FakeHolder()
        pool, old = holder.pool(PoolType.SAMPLE)
        pool._clients.append(AsyncTinker(**pool._constructor_kwargs))
        pool._client_active_refcount.append(0)
        originals = list(pool._clients)
        created = []

        def constructor(**kwargs):
            if created:
                raise RuntimeError("offline constructor failure")
            new = AsyncTinker(**kwargs)
            created.append(new)
            return new

        monkeypatch.setattr(adapter, "AsyncTinker", constructor)
        try:
            with pytest.raises(RuntimeError, match="offline constructor failure"):
                await adapter.harden_holder_on_own_loop(holder)
            assert pool._clients == originals
            assert all(not client._client.is_closed for client in originals)
            assert created[0]._client.is_closed
            assert all(client.max_retries == 0 for client in originals)
        finally:
            await holder.cleanup()

    asyncio.run(case())


def test_nonstock_mount_is_rebuilt():
    class ForeignTransport(httpx.AsyncBaseTransport):
        closed = False

        async def aclose(self):
            self.closed = True

    async def case():
        holder = FakeHolder()
        try:
            pool, _ = holder.pool(PoolType.SAMPLE)
            await adapter.harden_holder_on_own_loop(holder)
            old = pool._clients[0]
            mount = ForeignTransport()
            old._client._mounts[object()] = mount
            assert not adapter._is_stock_httpx(old)
            result = await adapter.harden_holder_on_own_loop(holder)
            assert result["http_clients_rebuilt"] == 1
            assert mount.closed
            assert adapter._is_stock_httpx(pool._clients[0])
        finally:
            await holder.cleanup()

    asyncio.run(case())


def test_cancelled_cleanup_finishes_and_is_observed_on_reentry():
    async def case():
        holder = FakeHolder()
        pool, old = holder.pool(PoolType.SAMPLE)
        entered, release = asyncio.Event(), asyncio.Event()
        close = old.close

        async def delayed_close(self):
            entered.set()
            await release.wait()
            await close()

        old.close = types.MethodType(delayed_close, old)
        task = asyncio.create_task(adapter.harden_holder_on_own_loop(holder))
        try:
            await entered.wait()
            assert pool._clients[0] is not old
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not holder._elt_transport_cleanup.done()
            release.set()
            result = await adapter.harden_holder_on_own_loop(holder)
            assert old._client.is_closed
            assert result["http_clients_rebuilt"] == 0
        finally:
            release.set()
            await holder.cleanup()

    asyncio.run(case())


def test_close_during_idle_wait_is_rejected():
    async def case():
        holder = FakeHolder()
        pool, old = holder.pool(PoolType.SAMPLE)
        pool._client_active_refcount[0] = 1

        async def close_marker():
            await asyncio.sleep(0.01)
            with holder._close_lock:
                holder._closed = True

        marker = asyncio.create_task(close_marker())
        try:
            with pytest.raises(RuntimeError, match="closed Tinker holder"):
                await adapter.harden_holder_on_own_loop(holder)
            assert pool._clients[0] is old and not old._client.is_closed
        finally:
            await marker
            pool._client_active_refcount[0] = 0
            await holder.cleanup()

    asyncio.run(case())


def test_wrong_loop_and_sdk_version_rejected(monkeypatch):
    async def case():
        holder = FakeHolder()
        holder._loop = object()
        with pytest.raises(RuntimeError, match="own event loop"):
            await adapter.harden_holder_on_own_loop(holder)
        holder._loop = asyncio.get_running_loop()
        monkeypatch.setattr(adapter.importlib.metadata, "version", lambda name: "0.40.0")
        with pytest.raises(RuntimeError, match="0.39.2"):
            await adapter.harden_holder_on_own_loop(holder)

    asyncio.run(case())


def test_sampler_hardening_keeps_retry_handler_and_backoff_disabled():
    async def case():
        holder = FakeHolder()
        calls = []

        async def backoff():
            calls.append(1)
            return None

        sampler = types.SimpleNamespace(
            holder=holder,
            retry_handler=types.SimpleNamespace(config=adapter.RetryConfig()),
            _send_asample_request=backoff,
        )
        try:
            await adapter.harden_sampler_on_own_loop(sampler)
            assert sampler.retry_handler.config.enable_retry_logic is False
            assert adapter.no_retry_config().enable_retry_logic is False
            wrapped = sampler._send_asample_request
            await adapter.harden_sampler_on_own_loop(sampler)
            assert sampler._send_asample_request is wrapped
            with pytest.raises(adapter.DispatchBackoffRejected):
                await sampler._send_asample_request()
            assert calls == [1]
        finally:
            await holder.cleanup()

    asyncio.run(case())


def test_close_during_cleanup_does_not_report_ready():
    async def case():
        holder = FakeHolder()
        pool, old = holder.pool(PoolType.SAMPLE)
        close = old.close

        async def close_holder_marker(self):
            with holder._close_lock:
                holder._closed = True
            await close()

        old.close = types.MethodType(close_holder_marker, old)
        try:
            with pytest.raises(RuntimeError, match="closed during transport cleanup"):
                await adapter.harden_holder_on_own_loop(holder)
            assert old._client.is_closed
            assert adapter._is_stock_httpx(pool._clients[0])
        finally:
            await holder.cleanup()

    asyncio.run(case())
