"""``GET/PUT /api/config/default-template``: the template a session runs when nothing names one.

The route writes ``agent.default_agent`` and nothing else. The rules under test
are the ones that keep it from silently pointing every future plain session at
something that is not there: only an installed GLOBAL template qualifies; a
crewmate's private copy, a background-only runtime spec and an app's
materialized agent are refused with their own codes; an unknown name is 404; the
write is owner-only and goes through the locked config read-modify-write so no
sibling setting is reverted.
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew import agent_state
from kiro_crew.agent_files import LITE_AGENT_FILENAME
from kiro_crew.config.loader import KiroCrewConfig, config_path
from kiro_crew.dashboard.routes import agents as agents_routes

pytestmark = pytest.mark.asyncio


class _FakeState:
    """No owner configured: only the signed local bootstrap subjects pass."""

    owner_id = ""

    async def read_folders(self, read):
        # The roster snapshots the chat-folder store for ``used_by``; none here.
        return read([])


def _build_app() -> web.Application:
    @web.middleware
    async def _identity(request, handler):
        request["user"] = request.headers.get("X-Test-User", "local-app")
        request["app"] = request.headers.get("X-Test-App", "")
        return await handler(request)

    app = web.Application(middlewares=[_identity])
    app["state"] = _FakeState()
    agents_routes.register(app)
    return app


@pytest.fixture(autouse=True)
def _owner_caller(monkeypatch):
    """Past the owner boundary by default; the 403 case flips it off."""
    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
        lambda request: True,
    )


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    d = tmp_path / "agents"
    d.mkdir()
    monkeypatch.setattr("kiro_crew.agent.KIRO_AGENTS_DIR", d)
    return d


def _write(agents_dir, filename: str, **spec) -> None:
    data = {"name": filename.rsplit(".", 1)[0], "tools": ["fs_read"], **spec}
    (agents_dir / filename).write_text(json.dumps(data), encoding="utf-8")


def _seed_config(default_template: str = "", **extra) -> None:
    cfg = KiroCrewConfig()
    cfg.agent.default_agent = default_template
    cfg.save()
    if extra:
        doc = json.loads(config_path().read_text(encoding="utf-8"))
        doc.update(extra)
        config_path().write_text(json.dumps(doc), encoding="utf-8")


def _stored() -> dict:
    return json.loads(config_path().read_text(encoding="utf-8"))


async def _put(client: TestClient, template, **headers):
    return await client.put(
        "/api/config/default-template", json={"template": template}, headers=headers
    )


async def test_get_reports_stored_and_effective(agents_dir):
    """Unset stores "" but STARTS the runtime template; the picker needs both."""
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await client.get("/api/config/default-template")
        assert resp.status == 200
        assert await resp.json() == {"default_template": "", "effective": "kirocrew"}

        _seed_config("reviewer")
        resp = await client.get("/api/config/default-template")
        assert await resp.json() == {"default_template": "reviewer", "effective": "reviewer"}


async def test_put_writes_agent_default_agent_and_nothing_else(agents_dir):
    _write(agents_dir, "reviewer.json")
    # A crewmate default that is NOT the template being picked, so the two
    # settings can be told apart in the written document.
    _seed_config(
        sidecar_setting={"kept": True},
        agents={"pr-bot": {"kiro_agent": "reviewer"}},
        default_agent="pr-bot",
    )
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, "reviewer")
        assert resp.status == 200
        assert await resp.json() == {
            "ok": True,
            "default_template": "reviewer",
            "effective": "reviewer",
        }
    doc = _stored()
    assert doc["agent"]["default_agent"] == "reviewer"
    # A locked read-modify-write, not a whole-document publish: the unrelated
    # key written beside it survives.
    assert doc["sidecar_setting"] == {"kept": True}
    # The CREWMATE default is a different setting and is untouched.
    assert doc["default_agent"] == "pr-bot"


async def test_put_accepts_a_package_template(agents_dir):
    """A package template is read-only to EDIT, but every session start can reach it."""
    _write(agents_dir, "SomePkg-atlas.json", name="atlas")
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, "atlas")
        assert resp.status == 200
    assert _stored()["agent"]["default_agent"] == "atlas"


async def test_put_empty_clears_back_to_the_runtime_default(agents_dir):
    _seed_config("reviewer")
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, "")
        assert resp.status == 200
        assert await resp.json() == {"ok": True, "default_template": "", "effective": "kirocrew"}
    assert _stored()["agent"]["default_agent"] == ""


async def test_put_unknown_template_is_404_and_writes_nothing(agents_dir):
    _write(agents_dir, "reviewer.json")
    _seed_config("reviewer")
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, "ghost")
        assert resp.status == 404
        assert (await resp.json())["code"] == "template_not_found"
    assert _stored()["agent"]["default_agent"] == "reviewer"


async def test_put_refuses_a_private_copy(agents_dir):
    """A crewmate's copy is deleted by that crewmate's publish/reset cleanup."""
    _write(agents_dir, "pr-bot-copy.json")
    agent_state.set_fork_info("pr-bot-copy", forked_from="reviewer", private_to="pr-bot")
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, "pr-bot-copy")
        assert resp.status == 409
        body = await resp.json()
        assert body["code"] == "template_private_copy"
        assert "pr-bot" in body["error"]
    assert _stored()["agent"]["default_agent"] == ""


async def test_put_refuses_a_background_only_runtime_spec(agents_dir):
    _write(agents_dir, LITE_AGENT_FILENAME)
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, LITE_AGENT_FILENAME.rsplit(".", 1)[0])
        assert resp.status == 409
        assert (await resp.json())["code"] == "template_background_only"


async def test_put_refuses_an_app_registered_agent(agents_dir):
    """``<app>--<agent>.json`` is unlinked when the app is disabled."""
    _write(agents_dir, "notes--scribe.json", name="scribe")
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, "scribe")
        assert resp.status == 409
        assert (await resp.json())["code"] == "app_registered_template"


@pytest.mark.parametrize("bad", [["reviewer"], 7, None])
async def test_put_non_string_is_400(agents_dir, bad):
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, bad)
        assert resp.status == 400
        assert (await resp.json())["code"] == "invalid_template_type"


@pytest.mark.parametrize("bad", ["../escape", "has space", "a" * 64])
async def test_put_malformed_name_is_400_before_any_scan(agents_dir, bad):
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await _put(client, bad)
        assert resp.status == 400
        assert (await resp.json())["code"] == "invalid_template_name"


async def test_put_is_owner_only(agents_dir, monkeypatch):
    """Refused before the body is read: a 403, not a 400 for the missing body."""
    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
        lambda request: False,
    )
    _write(agents_dir, "reviewer.json")
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await client.put("/api/config/default-template")
        assert resp.status == 403
        assert (await resp.json())["code"] == "owner_only"
        # GET stays open: the picker's readout is not a mutation.
        resp = await client.get("/api/config/default-template")
        assert resp.status == 200
    assert _stored()["agent"]["default_agent"] == ""


async def test_roster_flags_which_rows_the_picker_may_offer(agents_dir):
    """The PUT's refusals, as one display flag per row, so the picker lists only what would land."""
    _write(agents_dir, "reviewer.json")
    _write(agents_dir, "pr-bot-copy.json")
    agent_state.set_fork_info("pr-bot-copy", forked_from="reviewer", private_to="pr-bot")
    _write(agents_dir, LITE_AGENT_FILENAME)
    _write(agents_dir, "notes--scribe.json", name="scribe")
    _seed_config()
    async with TestClient(TestServer(_build_app())) as client:
        resp = await client.get("/api/agents/templates")
        assert resp.status == 200
        rows = {r["name"]: r["default_eligible"] for r in (await resp.json())["templates"]}
    assert rows["reviewer"] is True
    assert rows["pr-bot-copy"] is False
    assert rows[LITE_AGENT_FILENAME.rsplit(".", 1)[0]] is False
    assert rows["scribe"] is False
