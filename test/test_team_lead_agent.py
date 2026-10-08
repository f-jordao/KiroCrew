"""The team-lead crewmate: its spec, its charter, its provisioning and its skill.

Follows ``test_security_conductor_agent.py``'s installer half -- stub the agents
dir and ``build_agent_config``, run the installer, assert on the JSON it wrote --
and adds the three halves that agent does not have: a charter whose contract
sentences are pinned because the capability is the sentence, a crewmate record
whose provisioning must be safe to repeat on every boot, and a skill that reuses
another skill's scripts rather than carrying a copy.

Each class is named so it can be selected alone with ``-k``.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from kiro_crew import agent, agent_state, subagent
from kiro_crew.agent_files import (
    OWNED_KIRO_AGENT_FILES,
    REQUIRED_KIRO_AGENT_FILES,
    TEAM_LEAD_AGENT_FILENAME,
)
from kiro_crew.kiro_cli import SPEC_PERMISSIONS_MIN_VERSION

_ACCEPTS = SPEC_PERMISSIONS_MIN_VERSION
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SKILLS = _REPO_ROOT / "src" / "kiro_crew" / "builtin_skills"


class _Recorder:
    """A SEL sink that accepts every record and keeps it, for a test that needs neither.

    Stands in for the real ``SecurityEventLog`` so a test observing something else -- the
    spec lock's critical section -- is not also exercising SEL init, which reads and
    writes the trust directory. The installer's own audits are fenced, so this is a
    convenience rather than a requirement; the tests that assert ON the records define
    their own local recorder beside the assertions that read it.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def log_api_access(self, **kwargs):
        self.records.append(kwargs)


def _stub_environment(tmp_path, monkeypatch, *, may_auto_approve=None) -> None:
    """Pin the agents dir, the template and the writer's version gate.

    The template mirrors what ``config/defaults.json`` ships in the one respect
    these tests turn on: ``fs_write`` mounted and ungranted.
    """
    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    monkeypatch.setattr("kiro_crew.kiro_cli.installed_kiro_cli_version", lambda: _ACCEPTS)
    monkeypatch.setattr(
        agent,
        "build_agent_config",
        lambda: {
            "name": "kirocrew",
            "prompt": "file://x",
            "mcpServers": {
                "kirocrew-core": {"command": "/resolved/kirocrew", "args": ["mcp-core"]},
                "builder-mcp": {"command": "/x/builder", "args": []},
            },
            "tools": ["fs_write", "code", "execute_bash", "fs_read", "@kirocrew-core"],
            "allowedTools": ["fs_read", "@kirocrew-core"],
        },
    )
    monkeypatch.setattr(
        agent, "_kirocrew_mcp_invocation", lambda sub: ("/resolved/kirocrew", [sub])
    )
    monkeypatch.setattr(agent, "_may_auto_approve", may_auto_approve or (lambda ref: True))


def _install(tmp_path, monkeypatch, *, may_auto_approve=None) -> dict:
    _stub_environment(tmp_path, monkeypatch, may_auto_approve=may_auto_approve)
    assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
    return json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))


def _charter(tmp_path, monkeypatch) -> str:
    """The emitted prompt with its whitespace normalized, so a reflow cannot
    break an assertion about a sentence the charter actually makes."""
    return " ".join(_install(tmp_path, monkeypatch)["prompt"].split())


class TestTeamLeadSpecIsMaterializedByProductCode:
    def test_the_installer_writes_the_spec(self, tmp_path, monkeypatch):
        """The whole point of a shipped crewmate: the file is produced by an
        installer the rebuild calls, never by a hand that drops it in."""
        data = _install(tmp_path, monkeypatch)
        assert data["name"] == "kirocrew-team-lead"
        assert (tmp_path / TEAM_LEAD_AGENT_FILENAME).is_file()

    def test_the_rebuild_calls_it(self):
        """Eager, beside the other generated specs. ``session_create`` resolves an
        agent from a boot-time in-memory snapshot that no spec write refreshes, so
        a spec materialized later cannot be dispatched by name at all."""
        source = (_REPO_ROOT / "src" / "kiro_crew" / "agent.py").read_text(encoding="utf-8")
        assert "team_lead_agent._install_team_lead_agent(clean=clean)" in source

    def test_only_a_transient_hold_enters_the_retry_set(self):
        """``_conductor_spec_held`` feeds an hourly retry sweep, so a PERMANENT
        decline must not be counted there: the file is still somebody else's on
        every later pass, which makes it a retry that can never succeed and a log
        line reading as a failure for a correct decision."""
        source = (_REPO_ROOT / "src" / "kiro_crew" / "agent.py").read_text(encoding="utf-8")
        head, _, rest = source.partition("team_lead_agent._install_team_lead_agent(clean=clean)")
        assert rest, "the rebuild no longer calls the installer the way this test reads"
        assert "InstallOutcome.HELD" in rest[:200]
        # Not the collapsed form, which would count DECLINED as held.
        assert "not team_lead_agent._install_team_lead_agent" not in source

    def test_filename_is_owned_but_not_required(self):
        """Owned, so the convergence sweep rewrites it when the Playwright servers
        move. NOT required: that list fails every turn when its file is absent,
        and this spec's absence disables one feature."""
        assert TEAM_LEAD_AGENT_FILENAME in OWNED_KIRO_AGENT_FILES
        assert TEAM_LEAD_AGENT_FILENAME not in REQUIRED_KIRO_AGENT_FILES

    def test_the_agent_is_advertised(self):
        """The opposite of every conductor, and for the reason that set exists: a
        conductor is withheld from the roster because work handed to it cannot be
        done, having no file-writing tool. This one can do the work."""
        assert "kirocrew-team-lead" not in subagent.UNADVERTISED_AGENTS


class TestTeamLeadInstallerDeclinesAForeignSpec:
    """Provenance, not existence. The spec id makes a collision unlikely rather
    than impossible, and an operator's own agent at this filename is not an
    out-of-date spec: replacing it destroys whatever put it there."""

    def test_a_crew_of_its_own_at_this_filename_is_left_alone(self, tmp_path, monkeypatch):
        """Left alone means BYTE-IDENTICAL. The install declines and changes nothing --
        no grant removed, no key added, no reformat. What keeps the grants on it from
        mattering is that a session on this spec is REFUSED, not that the file was
        sanitized: sanitizing needs a correct judgement about every field it touches,
        and refusing needs none."""
        foreign = {
            "name": "my-own-lead",
            "prompt": "mine",
            "tools": ["fs_write"],
            "mcpServers": {},
        }
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        target.write_text(json.dumps(foreign), encoding="utf-8")
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        assert json.loads(target.read_text(encoding="utf-8")) == foreign, (
            "the install changed a spec it declined; it may read that file and refuse to "
            "run it, and nothing else"
        )

    def test_declining_is_reported_at_error_level(self, tmp_path, monkeypatch):
        """The installer's own caller swallows at debug, so without this the one
        event an operator needs is invisible at any ordinary level."""
        (tmp_path / TEAM_LEAD_AGENT_FILENAME).write_text(
            json.dumps({"name": "my-own-lead"}), encoding="utf-8"
        )
        _stub_environment(tmp_path, monkeypatch)
        errors: list[str] = []
        monkeypatch.setattr(
            agent.logger, "error", lambda msg, *a, **kw: errors.append(msg % a if a else msg)
        )
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        assert errors and "my-own-lead" in errors[0]

    def test_its_own_previous_output_is_rewritten(self, tmp_path, monkeypatch):
        """Idempotence depends on this: the marks are what the installer itself
        writes, so a second rebuild re-derives rather than declining forever."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN

    def test_a_missing_defining_server_is_named_in_the_reason(self):
        """A spec sharing the name but not the mounts is somebody else's. The
        reason says which mark failed, so the log tells an operator what to look
        at rather than only that something is in the way."""
        for absent in agent._DEFINING_SERVERS:
            present = [s for s in agent._DEFINING_SERVERS if s != absent]
            spec = {
                "name": "kirocrew-team-lead",
                "mcpServers": {s: {} for s in present},
            }
            reason = agent._foreign_team_lead_spec_reason(spec)
            assert reason is not None and absent in reason, absent

    def test_only_the_shape_this_release_writes_is_accepted(self):
        """Nothing is healed, because there is no earlier release of this spec to
        heal. The mount must be in ``mcpServers``; the same reference reached
        through ``tools`` or a per-tool grant is somebody else's file."""
        assert (
            agent._foreign_team_lead_spec_reason(
                {
                    "name": "kirocrew-team-lead",
                    "mcpServers": {s: {} for s in agent._DEFINING_SERVERS},
                }
            )
            is None
        )
        reason = agent._foreign_team_lead_spec_reason(
            {
                "name": "kirocrew-team-lead",
                "tools": ["@kirocrew-dashboard"],
                "allowedTools": ["@kirocrew-work/work_ledger_read"],
            }
        )
        assert reason == "it declares no mcpServers map"

    def test_a_foreign_spec_with_a_malformed_grant_list_is_still_declined(
        self, tmp_path, monkeypatch
    ):
        """Attribution runs BEFORE grant-shape validation, and this is the case
        that proves it. The hardened reader rejects a non-list ``allowedTools`` as
        bytes it cannot use, which a hand-edit typo produces on a file that is
        otherwise entirely the operator's. Reading that as "no spec here" would
        skip attribution and overwrite the prompt, servers and model pin whole."""
        foreign = {
            "name": "my-own-lead",
            "prompt": "file:///somewhere/my-own-lead.md",
            "model": "my-pinned-model",
            "mcpServers": {"my-server": {"command": "x", "args": []}},
            "allowedTools": None,
        }
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        raw = json.dumps(foreign)
        target.write_text(raw, encoding="utf-8")
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        # Byte-identical, typo included: the install does not repair a hand-edit and does
        # not sanitize one either.
        assert target.read_text(encoding="utf-8") == raw, "the operator's file was rewritten"

    def test_a_transient_read_in_the_attribution_gap_holds_rather_than_writes(
        self, tmp_path, monkeypatch
    ):
        """The narrow case the attribution read opens, and why it is a HOLD.

        The file parses, so the first read answers "cannot use these bytes" on
        grant shape alone; the attribution read then hits an I/O error. Treating
        that as "nothing to attribute" would overwrite a file that had just
        demonstrated it is somebody's. A transient failure is the reader's own
        other class, so it holds, and a hold costs one rebuild.
        """
        foreign = {
            "name": "my-own-lead",
            "prompt": "file:///somewhere/my-own-lead.md",
            "allowedTools": None,
        }
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        raw = json.dumps(foreign)
        target.write_text(raw, encoding="utf-8")
        _stub_environment(tmp_path, monkeypatch)

        # ONLY the attribution read fails. Failing both would hold for the first
        # read's own reason and prove nothing about this gap, which is the trap
        # this test exists inside: the two reads are told apart by the
        # ``operation`` each passes, so the real reader still serves the first.
        from kiro_crew import agent_discovery

        real = agent_discovery.read_agent_spec_strict
        calls: list[str] = []

        def _reader(path, *, operation, source):
            calls.append(operation)
            if operation == "team_lead_spec_attribution":
                raise OSError("input/output error")
            return real(path, operation=operation, source=source)

        monkeypatch.setattr(agent_discovery, "read_agent_spec_strict", _reader)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.HELD
        assert target.read_text(encoding="utf-8") == raw, "the operator's file was rewritten"
        assert calls == [
            "conductor_spec_regeneration",
            "team_lead_spec_attribution",
        ], f"the first read must succeed and the second must be reached: {calls}"

    def test_a_file_that_is_present_and_unparseable_is_declined(self, tmp_path, monkeypatch):
        """One missing brace is the ordinary result of a hand edit, and those bytes
        are the operator's whether or not they parse. Declining does not strand the
        agent: the remedy is documented and costs one rename, which is the
        dashboard-author contract rather than the derived worker's."""
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        raw = '{ "name": "my-own-lead", "prompt": "file:///mine.md"'
        target.write_text(raw, encoding="utf-8")
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        assert target.read_text(encoding="utf-8") == raw, "the operator's file was rewritten"

    def test_a_present_name_with_no_readable_bytes_is_declined(self, tmp_path, monkeypatch):
        """A dangling link is not an absent file. Only ABSENT installs without a
        name to check, so a name that still exists behind the read failure is
        present with nothing to attribute."""
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        target.symlink_to(tmp_path / "nowhere.json")
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        assert target.is_symlink(), "the link was replaced"

    def test_only_an_absent_path_installs_without_attribution(self, tmp_path, monkeypatch):
        """The control for both declines above, and the third of the three states.
        A guard that refused everything present would still have to install on an
        empty agents directory, which is every first install."""
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert not target.exists()
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert json.loads(target.read_text(encoding="utf-8"))["name"] == "kirocrew-team-lead"

    def test_the_three_outcomes_are_told_apart_on_the_same_path(self, tmp_path, monkeypatch):
        """The control for the two decline tests above. A guard that refused
        everything would pass them both, so this requires all three answers from
        the same path: a malformed foreign spec declined, a well-formed foreign
        spec declined, and a spec this installer wrote replaced.

        The third case is the installer's OWN write rather than a hand-made file
        carrying the right marks, because the marks are a shape and not authorship:
        only bytes reproducing the recorded ownership digest are replaced."""
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        _stub_environment(tmp_path, monkeypatch)

        target.write_text(json.dumps({"name": "theirs", "allowedTools": None}), encoding="utf-8")
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED

        target.write_text(
            json.dumps({"name": "theirs", "mcpServers": {}, "allowedTools": []}),
            encoding="utf-8",
        )
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED

        target.unlink()
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert json.loads(target.read_text(encoding="utf-8"))["prompt"].startswith(
            "# Kiro Crew Team Lead"
        )

    def test_the_global_mcp_merge_is_pinned_off(self, tmp_path, monkeypatch):
        """``includeMcpJson`` tells kiro-cli to merge the global ``mcp.json`` ON TOP of
        this spec, and the operator's own ``agent.json`` can set it, because
        ``build_agent_config`` deep-merges that file. Every filter in this installer reads
        ``config["mcpServers"]`` and nothing else, so a server arriving from the global
        file is governed by none of them: its ``autoApprove`` is never seen by the ceiling
        strip, and kiro-cli approves such a tool locally with no permission request. The
        pin is what makes the filtered map the only map.

        Control: a template that sets neither field still gets the pin written rather than
        left absent, because absent is not the same answer to kiro-cli as false."""
        _stub_environment(tmp_path, monkeypatch)
        base = agent.build_agent_config()
        base["includeMcpJson"] = True
        base["useLegacyMcpJson"] = True
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))

        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))
        assert spec["includeMcpJson"] is False, (
            "the global mcp.json merge stayed on, so a server from that file reaches "
            "kiro-cli past every filter in this installer"
        )
        assert "useLegacyMcpJson" not in spec, "the retired spelling can re-open the merge"

        # Control: with neither field in the template the pin is still written, and the
        # spec this installer produces is otherwise the ordinary one.
        clean = agent.build_agent_config()
        clean.pop("includeMcpJson", None)
        clean.pop("useLegacyMcpJson", None)
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(clean)))
        (tmp_path / TEAM_LEAD_AGENT_FILENAME).unlink()
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))
        assert spec["includeMcpJson"] is False
        assert "useLegacyMcpJson" not in spec
        assert "@kirocrew-work/work_brief" in spec["allowedTools"]

    def test_every_grant_this_installer_withholds_is_audited(self, tmp_path, monkeypatch):
        """Withholding a grant is a permission DECISION, and an operator reads the SEL
        feed to find out why a grant they can see in the template is not on the spec. The
        whole-server narrowing is the one that fires on an ORDINARY rebuild -- the
        template's bare ``@kirocrew-core`` is subtracted every time -- and the ceiling pass
        after it sees only the survivors, so an unaudited subtraction there leaves no trace
        anywhere.

        All three of this installer's own withholds are required in ONE run, so a feed
        that recorded two of them does not pass."""
        records: list[dict] = []

        class _Recorder:
            def log_api_access(self, **kwargs):
                records.append(kwargs)

        _stub_environment(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "sel", lambda: _Recorder())
        base = agent.build_agent_config()
        base["mcpServers"] = {**(base.get("mcpServers") or {}), "kirocrew-panel": {"command": "/p"}}
        base["allowedTools"] = [*base.get("allowedTools", []), "session_*"]
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))

        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        withheld = [
            str(r.get("resources"))
            for r in records
            if r.get("operation") == "mcp_auto_approve_withheld"
        ]
        # The whole-server narrowing: present on every rebuild, and the gap this pins.
        assert any(
            "@kirocrew-core" in r and "verb by verb" in r for r in withheld
        ), f"the whole-server narrowing left no audit record: {withheld}"
        # The two that already audited, required here so this cannot pass on one event.
        assert any("session_*" in r for r in withheld), f"no record names the pattern: {withheld}"
        assert any("kirocrew-panel" in r for r in withheld), f"no record names it: {withheld}"

    def test_the_whole_read_attribute_write_runs_under_the_spec_lock(self, tmp_path, monkeypatch):
        """This installer is a read-modify-writer and was the one that took no lock.
        ``agents_spec_lock`` is the template-spec writer lock every other one holds --
        the worker installer, the dashboard-author installer, the reset path, the fork
        refresh, the dashboard PATCH -- so two overlapping rebuilds could interleave
        their reads and writes here and silently revert each other.

        Pinned as ONE critical section rather than merely as "a lock is taken", because
        splitting it is the bug and not a smaller version of it: attributing under the
        lock and then writing outside it is the same race with a narrower window. The
        read, the attribution and the write are each required to happen while it is
        held, and the lock is required to be taken exactly once -- it is a cross-process
        ``flock`` on a sidecar path, so a second acquire from this process would wait on
        itself."""
        events: list[str] = []
        real_lock = agent.agents_spec_lock

        import contextlib

        @contextlib.contextmanager
        def _watched(agents_dir):
            events.append("lock-enter")
            with real_lock(agents_dir):
                yield
            events.append("lock-exit")

        # Watched on the owning module rather than the facade: the attribution read is
        # this installer's own helper, which reaches the conductor installers' hardened
        # reader, so the facade carries no name for it.
        from kiro_crew.agent_materialization import team_lead_agent

        real_read = team_lead_agent._existing_spec_for_attribution
        real_write = agent._atomic_json_write

        _stub_environment(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "sel", lambda: _Recorder())
        monkeypatch.setattr(agent, "agents_spec_lock", _watched)
        monkeypatch.setattr(
            team_lead_agent,
            "_existing_spec_for_attribution",
            lambda *a, **k: (events.append("read"), real_read(*a, **k))[1],
        )
        monkeypatch.setattr(
            agent,
            "_atomic_json_write",
            lambda *a, **k: (events.append("write"), real_write(*a, **k))[1],
        )

        # A file already present, so the attribution read actually runs rather than
        # short-circuiting on an absent path.
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        events.clear()
        assert target.is_file()
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN

        assert events.count("lock-enter") == 1, (
            f"the lock was taken {events.count('lock-enter')} times; it is a "
            f"cross-process flock, so a second acquire would wait on itself: {events}"
        )
        enter, exit_ = events.index("lock-enter"), events.index("lock-exit")
        assert "read" in events and "write" in events, f"neither ran: {events}"
        for stage in ("read", "write"):
            assert enter < events.index(stage) < exit_, (
                f"the {stage} happened outside the critical section, so another writer "
                f"can land between it and the rest: {events}"
            )

    def test_an_audit_failure_never_breaks_the_install(self, tmp_path, monkeypatch):
        """Same footing as every other audit here: an unwritable SEL sink costs one
        record, not the spec. The control for the test above -- an installer that raised
        out of its own audit would fail this one instead of recording anything."""

        class _Broken:
            def log_api_access(self, **kwargs):
                raise OSError("audit sink unavailable")

        _stub_environment(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "sel", lambda: _Broken())
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert (tmp_path / TEAM_LEAD_AGENT_FILENAME).is_file()

    def test_an_inherited_wildcard_grant_is_not_carried(self, tmp_path, monkeypatch):
        """The ceiling judges an ``allowedTools`` entry by its EXACT NAME, so a PATTERN
        escapes it silently: it is in no table and names no server, passes as an unmapped
        name, and kiro-cli then expands it over every builtin and every verb of every
        mounted server -- including ``@kirocrew-dashboard``, which this same installer
        mounts. ``session_?end`` is the sharp case: the exact spelling is withheld by name
        while the pattern that expands onto it would be auto-approved, on the one path
        that never reaches the approval gate.

        Both controls the ceiling's own behaviour depends on are asserted here, so a drop
        that took everything would fail: an exact grant the ceiling allows survives, and a
        bare ``@server`` entry is still narrowed by its own separate pass.
        """
        _stub_environment(tmp_path, monkeypatch)
        base = agent.build_agent_config()
        base["allowedTools"] = [
            *base.get("allowedTools", []),
            "session_*",
            "*_send",
            "@*",
            "@kirocrew-dashboard/session_?end",
        ]
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))
        monkeypatch.setattr(agent, "_may_auto_approve", lambda ref: "session_send" not in ref)

        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))
        allowed = spec["allowedTools"]

        for pattern in ("session_*", "*_send", "@*", "@kirocrew-dashboard/session_?end"):
            assert pattern not in allowed, (
                f"{pattern} reached the spec, so kiro-cli would expand it over a verb the "
                "ceiling withholds by name"
            )
        # Control: the exact grants this installer ships are judged by name and kept, so
        # the drop is "patterns" and not "anything the operator touched".
        assert "fs_read" in allowed
        assert "@kirocrew-work/work_brief" in allowed
        # Control: the bare ``@server`` pass still does its own separate job, which a
        # wildcard filter neither replaces nor is replaced by.
        assert "@kirocrew-core" not in allowed

    def test_a_governed_auto_approve_on_an_inherited_server_is_stripped(
        self, tmp_path, monkeypatch
    ):
        """``autoApprove`` on an ``mcpServers`` entry is the SECOND channel a call skips
        the PreToolUse gate through: kiro-cli approves such a tool locally and emits no
        permission request, so no amount of ``allowedTools`` filtering reaches it. This
        map is ADDITIVE and ``build_agent_config`` deep-merges the operator's own
        ``agent.json``, so an entry they wrote arrives with whatever ``autoApprove`` they
        gave it -- a grant no ceiling has read.

        Driven through the real governance pass by its own predicate rather than a stubbed
        strip, so the two answers are the ceiling's. Both controls are here: a governed
        server keeps nothing, an ungoverned one the operator wrote survives, and the
        ``allowedTools`` filter still does its own separate job.
        """
        from kiro_crew.platform import governance

        monkeypatch.setattr(
            governance, "may_skip_gate_now", lambda ref: not ref.startswith("@governed")
        )
        _stub_environment(tmp_path, monkeypatch)
        base = agent.build_agent_config()
        base["mcpServers"] = {
            **(base.get("mcpServers") or {}),
            "governed": {"command": "/x/g", "args": [], "autoApprove": ["delete"]},
            "friendly": {"command": "/x/f", "args": [], "autoApprove": ["read"]},
        }
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))

        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))
        servers = spec["mcpServers"]

        assert "autoApprove" not in servers["governed"], (
            "a ceiling-governed autoApprove reached the spec, so a native chat would run "
            "the denied tool without a permission request"
        )
        # The server itself stays mounted: its tools go through the approval gate, which
        # is where a per-argument ceiling rule is actually applied.
        assert "governed" in servers
        # Control: the ceiling is silent about this one and the operator wrote it, so
        # their own statement about their own tools survives. Without this a pass that
        # stripped every autoApprove would pass the assertion above.
        assert servers["friendly"]["autoApprove"] == ["read"]
        # Control: the other channel still behaves as before -- the shipped grants are
        # on the list and the template's whole-server entry is still subtracted.
        assert "@kirocrew-work/work_brief" in spec["allowedTools"]
        assert "@kirocrew-core" not in spec["allowedTools"]

    def test_an_opt_in_set_the_operator_mounted_is_not_inherited(self, tmp_path, monkeypatch):
        """An ``opt_in`` server is an assignable SET, not an always-on capability: the
        agents that need one hand-build the entry, and that IS the per-agent assignment.
        An operator mounting one on their personal ``agent.json`` has assigned it to that
        agent, so it must not ride through ``build_agent_config`` into this spec -- least
        of all ``kirocrew-panel``, which this agent's own charter says no spec may emit.

        All three surfaces, because a server reaches a session through any of them."""
        _stub_environment(tmp_path, monkeypatch)
        base = agent.build_agent_config()
        base["mcpServers"] = {**(base.get("mcpServers") or {}), "kirocrew-panel": {"command": "/p"}}
        base["tools"] = [*base.get("tools", []), "@kirocrew-panel"]
        base["allowedTools"] = [*base.get("allowedTools", []), "@kirocrew-panel"]
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))

        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))
        assert "kirocrew-panel" not in spec["mcpServers"]
        assert "@kirocrew-panel" not in spec["tools"]
        assert "@kirocrew-panel" not in spec["allowedTools"]
        # Control: the drop set is "the opt-in sets nobody assigned HERE", not "every
        # opt-in set". Asserted on the set itself, because the two this installer assigns
        # are re-added after the drop -- so their presence in the written spec would hold
        # even if the drop had taken them, and would prove nothing.
        unassignable = agent._team_lead_unassignable_servers()
        assert "kirocrew-panel" in unassignable
        for server in agent._DEFINING_SERVERS:
            assert server not in unassignable, server
        # And the entries that survive are the installer's hand-built ones: an operator
        # who mounts their own `kirocrew-work` does not get to decide how it launches.
        assert spec["mcpServers"]["kirocrew-work"]["args"] == ["mcp-work"]

    def test_an_in_place_edit_is_declined_rather_than_silently_reverted(
        self, tmp_path, monkeypatch
    ):
        """An operator who opens the installed file and changes its prompt or its model
        leaves the declared name and both mounted servers exactly as they were, and holds
        no lineage record. On the marks alone that file reads as this installer's own and
        is rewritten whole on the next rebuild, so EVERY in-place edit would be silently
        reverted -- ordinary use, not a forge.

        Authorship is therefore the digest the installer records for the bytes it lands.
        The control that matters most is the second one: the installer's own unmodified
        write must still be replaceable, because an over-tight rule here means the agent
        can never be updated again."""
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        _stub_environment(tmp_path, monkeypatch)

        # Control: an absent path installs and records its ownership digest.
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert agent_state.get_managed_digest("kirocrew-team-lead"), "no ownership recorded"

        # Control: the installer's own unmodified write is replaced on the next rebuild, so
        # the digest does not freeze the spec against future releases.
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN

        # The finding: an in-place edit that leaves both marks intact.
        spec = json.loads(target.read_text(encoding="utf-8"))
        assert spec["name"] == "kirocrew-team-lead"
        for server in agent._DEFINING_SERVERS:
            assert server in spec["mcpServers"], server
        spec["prompt"] = "my own charter"
        edited = json.dumps(spec, indent=2) + "\n"
        target.write_text(edited, encoding="utf-8")

        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        assert target.read_text(encoding="utf-8") == edited, "the operator's edit was reverted"

    def test_a_recorded_private_copy_is_declined_though_both_marks_hold(
        self, tmp_path, monkeypatch
    ):
        """Both marks can arrive without anyone forging them. A private copy is named after
        the crew that owns it and its declared ``name`` is set to that stem, so a crew named
        for this agent lands a file declaring this agent's name; copy any conductor and that
        file mounts both defining servers too. The lineage record is the one signal no spec
        field carries, so it is read first and the crew's own bytes are left alone.

        Driven against a file this installer DID write, so the record is the only thing
        that changes the answer and is shown to outrank even a confirmed ownership digest:
        a crew's lineage on this name means hands off whoever wrote the bytes."""
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        _stub_environment(tmp_path, monkeypatch)

        # Control: our own write, replaced on the next rebuild, so the decline below is the
        # record's doing and not some other rejection of this file.
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        raw = target.read_text(encoding="utf-8")

        agent_state.set_fork_info(
            "kirocrew-team-lead", forked_from="kirocrew-conductor", private_to="a-crew"
        )
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        # Byte-identical, and this is the ONE declined class the strip leaves alone. A
        # recorded fork is already governed -- ``fork_refresh`` projects the ceiling onto
        # each recorded fork's grants -- and stripping it would break that: these bytes
        # are this installer's own write, with a matching ownership digest, so rewriting
        # them under the record leaves a digest that can never match again and the
        # control below could never pass.
        assert target.read_text(encoding="utf-8") == raw, "the crew's private copy was rewritten"

        # Control: clearing the record restores the write, so the decline tracks the record
        # rather than latching on the first refusal.
        agent_state.clear_fork_info("kirocrew-team-lead")
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN

    def test_a_lineage_read_that_may_succeed_later_holds_rather_than_writes(
        self, tmp_path, monkeypatch
    ):
        """The record is a SECOND read, so it has the spec read's two classes. A sidecar that
        cannot be read is not a file nobody claims: collapsing the two would replace a crew's
        copy on one failed read, so it holds and the next rebuild asks again."""
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        raw = json.dumps(
            {
                "name": "kirocrew-team-lead",
                "mcpServers": {s: {} for s in agent._DEFINING_SERVERS},
                "allowedTools": [],
            }
        )
        target.write_text(raw, encoding="utf-8")
        _stub_environment(tmp_path, monkeypatch)

        def _unreadable(_name, *, strict=False):
            raise OSError("input/output error")

        monkeypatch.setattr(agent_state, "get_fork_info", _unreadable)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.HELD
        assert target.read_text(encoding="utf-8") == raw, "the operator's file was rewritten"

    def test_an_absent_path_is_the_one_state_with_nothing_to_attribute(self):
        """``None`` means absent, and absent is the only answer that writes. A
        present file that does not parse never reaches this function: the caller
        declines it before asking, because unreadable bytes at this name are still
        somebody's."""
        assert agent._foreign_team_lead_spec_reason(None) is None

    def test_a_hand_edited_container_does_not_raise_out_of_the_check(self):
        """The map is shape-checked before it is read, so a wrong-typed value is
        a declined write rather than an error out of an attribution."""
        for bad in (1, "kirocrew-dashboard", ["kirocrew-dashboard"]):
            reason = agent._foreign_team_lead_spec_reason(
                {"name": "kirocrew-team-lead", "mcpServers": bad}
            )
            assert reason == "it declares no mcpServers map", bad
        # A map that IS one, missing a mount, names the mount.
        reason = agent._foreign_team_lead_spec_reason(
            {"name": "kirocrew-team-lead", "mcpServers": {"kirocrew-work": {}}}
        )
        assert reason is not None and "kirocrew-dashboard" in reason


class TestAHandEditedSpecIsRefusedRatherThanRewritten:
    """The decline governs by REFUSING THE START, not by sanitizing the file.

    Rewriting somebody's spec to make it safe needs a correct judgement about every
    field it touches, and ``permissions`` is the field that shows why that is the
    wrong shape: ``kas_permissions._EFFECTS`` admits ``deny`` and ``ask`` beside
    ``allow``, so a pass emptying the block to remove GRANTS could delete a DENY the
    operator wrote and leave them less restricted than they asked to be. Refusing
    needs no such judgement and changes nothing on disk.

    Selectable alone with ``-k HandEdited``."""

    def test_an_edited_spec_is_untouched_and_its_start_is_refused(self, tmp_path, monkeypatch):
        """Both halves in one run, because either alone is the bug: untouched without a
        refusal leaves the edited grants live, and a refusal that rewrote the file first
        is the thing this replaces."""
        _stub_environment(tmp_path, monkeypatch)
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN

        spec = json.loads(target.read_text(encoding="utf-8"))
        spec["prompt"] = "my own charter"
        spec["allowedTools"] = [*spec["allowedTools"], "web_*"]
        spec["permissions"] = {"rules": [{"capability": "shell", "effect": "deny"}]}
        edited = json.dumps(spec, indent=2) + "\n"
        target.write_text(edited, encoding="utf-8")

        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        assert target.read_text(encoding="utf-8") == edited, (
            "the install rewrote a spec it declined; the operator's own `deny` rule is "
            "exactly what a sanitizing pass would have removed"
        )
        refusal = agent.spec_start_refusal("kirocrew-team-lead")
        assert refusal == (
            "kirocrew-team-lead.json was edited by hand; Kiro Crew will not run it. "
            "Rename it to keep your copy, or delete it to get the shipped one back."
        ), refusal

    def test_the_refusal_reaches_every_start_through_the_shared_gate(self, tmp_path, monkeypatch):
        """Wired into ``require_fork_governance``, not into the three harnesses. The KAS
        harness, the kiro harness and the ACP client all call that one function, so a
        rule copied into each is a rule that will be in two of them after the next
        change. Raised as ``ForkGovernanceUnresolved``, which is what all three already
        catch, so the message reaches the operator unchanged."""
        _stub_environment(tmp_path, monkeypatch)
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads(target.read_text(encoding="utf-8"))
        spec["prompt"] = "mine"
        target.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

        with pytest.raises(agent.ForkGovernanceUnresolved) as caught:
            agent.require_fork_governance("kirocrew-team-lead", tmp_path)
        assert "will not run it" in str(caught.value)
        # Control: the gate admits every OTHER non-fork agent exactly as before, so this
        # is one refusal and not a new class of them.
        agent.require_fork_governance("kirocrew-worker", tmp_path)
        agent.require_fork_governance("some-operator-agent", tmp_path)

    def test_deleting_the_file_restores_the_shipped_spec(self, tmp_path, monkeypatch):
        """The remedy the message names has to work. Renaming frees the name the same
        way; deleting is the half that must also put the shipped spec back."""
        _stub_environment(tmp_path, monkeypatch)
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads(target.read_text(encoding="utf-8"))
        spec["prompt"] = "mine"
        target.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        assert agent._install_team_lead_agent() is agent.InstallOutcome.DECLINED
        assert agent.spec_start_refusal("kirocrew-team-lead") is not None

        target.unlink()
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert json.loads(target.read_text(encoding="utf-8"))["prompt"] != "mine"
        assert agent.spec_start_refusal("kirocrew-team-lead") is None, (
            "the shipped spec is back and this installer confirms it, so nothing is left "
            "to refuse"
        )

    def test_a_recorded_fork_still_starts(self, tmp_path, monkeypatch):
        """THE control for the fork lane. A recorded fork is a supported feature with its
        own governance -- the refresh re-filters its grants and the gate's own wait holds
        its starts -- so refusing it here would break that lane outright, which is a worse
        regression than the finding this closes.

        It cannot happen by construction rather than by care: the refusal is reached only
        on the gate's own NON-fork path, keyed on the ``is_fork`` the gate already
        resolved, so there is one sidecar read and one notion of "is a fork" rather than
        two that can disagree. Driven by SETTING a real lineage record, not by stubbing a
        predicate."""
        _stub_environment(tmp_path, monkeypatch)
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        # Edited AND recorded as a crew's private copy: on bytes alone this is the
        # refusable case, so only the record can be what admits it.
        spec = json.loads(target.read_text(encoding="utf-8"))
        spec["prompt"] = "the crew's own charter"
        target.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        assert agent.spec_start_refusal("kirocrew-team-lead") is not None

        agent_state.set_fork_info(
            "kirocrew-team-lead", forked_from="kirocrew-conductor", private_to="a-crew"
        )
        import kiro_crew.agent_materialization.fork_refresh as fr

        fr._fork_refresh_settled.set()
        monkeypatch.setattr(fr, "_fork_refresh_failed", frozenset())
        # Admitted: the fork path runs its own checks and reaches no refusal here.
        agent.require_fork_governance("kirocrew-team-lead", tmp_path)

        # Control: clear the record and the SAME bytes are refused again, so the record
        # is the only thing that moves the answer.
        agent_state.clear_fork_info("kirocrew-team-lead")
        with pytest.raises(agent.ForkGovernanceUnresolved):
            agent.require_fork_governance("kirocrew-team-lead", tmp_path)

    def test_a_rebuild_that_dies_before_this_installer_still_refuses_the_start(
        self, tmp_path, monkeypatch
    ):
        """THE case a marker cannot catch, and the reason the check is stateless.

        ``rebuild_agent_config`` writes the default spec BEFORE it reaches this
        installer, so a failure there ends the pass without this installer running at
        all -- and any marker the installer would have set is left exactly as a healthy
        boot leaves it. Asking the question at admission needs no cooperation from the
        rebuild: the file's own grants are compared with what the current ceiling would
        produce, so a rebuild that never ran is indistinguishable from one that ran and
        could not write, which is the point."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert agent.spec_start_refusal("kirocrew-team-lead") is None

        # The ceiling tightens against a verb the file already auto-approves, and the
        # rebuild that would re-derive it dies on the FIRST spec it writes.
        monkeypatch.setattr(
            agent, "_may_auto_approve", lambda r: not r.startswith("@kirocrew-core/")
        )
        from kiro_crew.agent_materialization import default_spec_commit

        monkeypatch.setattr(
            default_spec_commit,
            "write_default_spec",
            lambda *a, **k: (_ for _ in ()).throw(OSError("read-only file system")),
        )
        with pytest.raises(Exception):
            agent.rebuild_agent_config()

        refusal = agent.spec_start_refusal("kirocrew-team-lead")
        assert refusal is not None, (
            "the rebuild died before this installer ran, so its grants were never "
            "re-derived under the tightened ceiling, and a session would consume them"
        )
        assert "earlier governance policy" in refusal, refusal
        assert "edited by hand" not in refusal, "an operator who edited nothing was told they did"
        with pytest.raises(agent.ForkGovernanceUnresolved, match="earlier governance policy"):
            agent.require_fork_governance("kirocrew-team-lead", tmp_path)

    def test_a_failed_team_lead_write_under_a_tightened_ceiling_refuses_the_start(
        self, tmp_path, monkeypatch
    ):
        """The same answer when the install DOES run and cannot write. The file keeps
        the previous ceiling's list, and the check reads that off the file rather than
        off any record of what the install did."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN

        monkeypatch.setattr(
            agent, "_may_auto_approve", lambda r: not r.startswith("@kirocrew-core/")
        )
        with monkeypatch.context() as broken:
            broken.setattr(
                agent,
                "_atomic_json_write",
                lambda *a, **k: (_ for _ in ()).throw(OSError("no space left on device")),
            )
            with pytest.raises(OSError):
                agent._install_team_lead_agent()

        refusal = agent.spec_start_refusal("kirocrew-team-lead")
        assert refusal is not None and "earlier governance policy" in refusal, refusal

    def test_autoapprove_drift_alone_refuses_the_start(self, tmp_path, monkeypatch):
        """The half that goes inert if the pair is not run, and it needs its own case.
        ``_apply_allowed_tools_ceiling`` returns early on its own key and never looks at
        ``mcpServers``, so a check that ran only the ceiling and then compared
        ``autoApprove`` would compare the file against itself and always pass.

        Driven with ``allowedTools`` DELIBERATELY left alone: the ceiling permits every
        grant on the list, and only the server-level auto-approve is governed away. If
        this reddens only because the grant list also drifted, it would prove nothing
        about the strip pass."""
        from kiro_crew.platform import governance

        _stub_environment(tmp_path, monkeypatch)
        base = agent.build_agent_config()
        base["mcpServers"] = {
            **(base.get("mcpServers") or {}),
            "vendor": {"command": "/v", "autoApprove": ["delete"]},
        }
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))
        monkeypatch.setattr(governance, "may_skip_gate_now", lambda r: True)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))
        assert spec["mcpServers"]["vendor"].get("autoApprove") == ["delete"], (
            "the fixture never got an autoApprove onto the file, so this test cannot "
            "be about the strip pass"
        )
        assert agent.spec_start_refusal("kirocrew-team-lead") is None

        # The ceiling tightens against THAT SERVER only. Every allowedTools entry is
        # still permitted, so the grant list does not drift and the ceiling pass alone
        # would see nothing.
        monkeypatch.setattr(governance, "may_skip_gate_now", lambda r: r != "@vendor")
        from kiro_crew.agent_materialization import team_lead_agent as tla

        probe = json.loads(json.dumps(spec))
        tla.auto_approve._apply_allowed_tools_ceiling(probe, source="t")
        assert (
            probe["allowedTools"] == spec["allowedTools"]
        ), "the grant list drifted too, so this case no longer isolates autoApprove"
        refusal = agent.spec_start_refusal("kirocrew-team-lead")
        assert refusal is not None and "earlier governance policy" in refusal, (
            "a server-level autoApprove the ceiling now denies stayed live on a spec "
            "kiro-cli loads, and that channel never reaches the approval gate"
        )

    def test_a_clean_rebuild_admits_the_start(self, tmp_path, monkeypatch):
        """The over-tight control, and the one a too-strict comparison breaks. A spec
        this installer just wrote under the CURRENT ceiling must admit -- the passes run
        on the copy have to produce exactly what is already on the file. Comparing more
        than the governed fields reddens this while every refusal test above stays
        green, which is how a comparison that refuses forever would otherwise ship."""
        _stub_environment(tmp_path, monkeypatch)
        # SCOPED, so the restrictive ceiling comes off without an undo(): the global
        # form would also drop the ``_stub_environment`` this test stands on, and the
        # second half below would then run against the real environment.
        with monkeypatch.context() as tightened:
            tightened.setattr(
                agent, "_may_auto_approve", lambda r: not r.startswith("@kirocrew-core/")
            )
            assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
            assert agent.spec_start_refusal("kirocrew-team-lead") is None, (
                "a spec written under this very ceiling reads as drifted, so this agent "
                "can never start"
            )
            agent.require_fork_governance("kirocrew-team-lead", tmp_path)

        # And again with the ordinary ceiling, so the admit is not an artifact of the
        # one stub above.
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert agent.spec_start_refusal("kirocrew-team-lead") is None

    def test_a_recorded_fork_still_starts_under_a_tightened_ceiling(self, tmp_path, monkeypatch):
        """The fork lane stays out, third round running. A recorded fork has its own
        governance -- the refresh re-filters its grants and the gate's own wait holds
        its starts -- and the drift check is reached only on the gate's non-fork path,
        so drift on the TEMPLATE cannot block a fork."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        monkeypatch.setattr(
            agent, "_may_auto_approve", lambda r: not r.startswith("@kirocrew-core/")
        )
        assert agent.spec_start_refusal("kirocrew-team-lead") is not None

        agent_state.set_fork_info(
            "kirocrew-team-lead", forked_from="kirocrew-conductor", private_to="a-crew"
        )
        import kiro_crew.agent_materialization.fork_refresh as fr

        fr._fork_refresh_settled.set()
        monkeypatch.setattr(fr, "_fork_refresh_failed", frozenset())
        agent.require_fork_governance("kirocrew-team-lead", tmp_path)

        # Control: clear the record and the same drift refuses again.
        agent_state.clear_fork_info("kirocrew-team-lead")
        with pytest.raises(agent.ForkGovernanceUnresolved, match="earlier governance policy"):
            agent.require_fork_governance("kirocrew-team-lead", tmp_path)

    def test_the_drift_check_does_no_io_and_emits_no_audit_on_a_healthy_start(
        self, tmp_path, monkeypatch
    ):
        """It runs on EVERY session start, so its cost is part of the contract. The
        installer's passes are being run outside the installer, and two of them emit an
        install-time SEL record by default -- a per-start copy of that event would be a
        lie in the feed. The strip takes ``audit=False``; the ceiling pass has no such
        switch and emits only when it withholds, which is the refusal path."""
        import subprocess

        _stub_environment(tmp_path, monkeypatch)
        records: list[dict] = []

        class _Recording:
            def log_api_access(self, **kwargs):
                records.append(kwargs)

        monkeypatch.setattr(agent, "sel", lambda: _Recording())
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads((tmp_path / TEAM_LEAD_AGENT_FILENAME).read_text(encoding="utf-8"))
        from kiro_crew.agent_materialization import team_lead_agent as tla

        tla._ceiling_drift_reason(spec)  # warm any caches the config loader holds
        records.clear()
        spawns: list[object] = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: spawns.append(a), raising=False)
        assert tla._ceiling_drift_reason(spec) is None
        assert not spawns, f"the drift check spawned a subprocess per start: {spawns}"
        assert not records, f"a healthy start emitted install-time audit records: {records}"

    def test_a_model_reset_renews_the_digest_and_leaves_the_agent_startable(
        self, tmp_path, monkeypatch
    ):
        """An ORDINARY user action must not brick the agent. ``reset_agent_model`` edits
        the installed spec in place to drop the model pin, so without renewing the
        recorded ownership digest the file stops matching it, the next rebuild declines
        it as somebody else's, and every later start is refused -- for a reset the user
        was invited to perform, with nothing telling them why.

        The renewal is the mechanism dashboard-author already had; this spec is a second
        entry in one table rather than a second branch beside it."""
        _stub_environment(tmp_path, monkeypatch)
        # A model PIN is the precondition a reset exists for, and the stub template
        # carries none, so it is set here rather than assumed.
        base = agent.build_agent_config()
        base["model"] = "my-pinned-model"
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        assert json.loads(target.read_text(encoding="utf-8")).get("model"), (
            "the fixture wrote no model pin, so a reset would strip nothing and this "
            "test could not be about the reset"
        )
        assert agent.spec_start_refusal("kirocrew-team-lead") is None

        path, previous = agent.reset_agent_model("kirocrew-team-lead")
        assert path == target and previous
        after = json.loads(target.read_text(encoding="utf-8"))
        assert not after.get("model"), "the reset did not remove the model pin"
        # The finding: the agent is still startable afterwards.
        assert agent.spec_start_refusal("kirocrew-team-lead") is None, (
            "an authorized model reset left the spec unconfirmable, so the user who "
            "performed it can never run this agent again"
        )
        agent.require_fork_governance("kirocrew-team-lead", tmp_path)
        # And it is a real renewal rather than the refusal being skipped: the sidecar
        # now records the digest of the NEW bytes.
        assert agent_state.managed_digest_matches(
            "kirocrew-team-lead", agent_state.spec_digest(after), strict=True
        )

    def test_a_model_reset_on_a_hand_edited_spec_does_not_launder_it(self, tmp_path, monkeypatch):
        """THE control, and why the renewal is keyed on the PRE-EDIT bytes. A reset that
        renewed unconditionally would be a way to LAUNDER a user-authored spec into a
        confirmed one, handing it exactly the provenance the installer refused it. The
        reset may still strip the model; what it must not do is stamp the result ours."""
        _stub_environment(tmp_path, monkeypatch)
        # A model PIN is the precondition a reset exists for, and the stub template
        # carries none, so it is set here rather than assumed.
        base = agent.build_agent_config()
        base["model"] = "my-pinned-model"
        monkeypatch.setattr(agent, "build_agent_config", lambda: json.loads(json.dumps(base)))
        target = tmp_path / TEAM_LEAD_AGENT_FILENAME
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        spec = json.loads(target.read_text(encoding="utf-8"))
        spec["prompt"] = "my own charter"
        target.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        assert agent.spec_start_refusal("kirocrew-team-lead") is not None

        agent.reset_agent_model("kirocrew-team-lead")
        after = json.loads(target.read_text(encoding="utf-8"))
        assert after["prompt"] == "my own charter", "the reset rewrote the operator's edit"
        refusal = agent.spec_start_refusal("kirocrew-team-lead")
        assert (
            refusal is not None and "edited by hand" in refusal
        ), f"a model reset laundered a hand-edited spec into a confirmed one: {refusal}"
        assert not agent_state.managed_digest_matches(
            "kirocrew-team-lead", agent_state.spec_digest(after), strict=True
        ), "the digest was renewed for bytes this installer never wrote"

    def test_a_project_shadow_refuses_the_start_and_names_the_project_file(
        self, tmp_path, monkeypatch
    ):
        """We validated one file while kiro-cli would run another. It resolves
        ``--agent`` against ``<cwd>/.kiro/agents`` BEFORE the global directory, so a
        checkout shipping its own ``kirocrew-team-lead.json`` is the file that starts --
        and every other check in this module reads the GLOBAL spec, which in that
        situation nothing will execute. The global spec here is clean and CONFIRMED, so
        attribution and drift both pass: only the shadow check can refuse."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        project = tmp_path / "checkout"
        (project / ".kiro" / "agents").mkdir(parents=True)
        (project / ".kiro" / "agents" / TEAM_LEAD_AGENT_FILENAME).write_text(
            json.dumps(
                {
                    "name": "kirocrew-team-lead",
                    "tools": ["execute_bash"],
                    "allowedTools": ["execute_bash"],
                }
            ),
            encoding="utf-8",
        )
        # Control: the global spec on its own admits, so the refusal below is the
        # shadow and not some other complaint about the installed file.
        assert agent.spec_start_refusal("kirocrew-team-lead") is None

        refusal = agent.spec_start_refusal("kirocrew-team-lead", project_dir=project)
        assert refusal is not None, (
            "the project copy is what kiro-cli would run, and it was admitted on the "
            "strength of a different file"
        )
        assert TEAM_LEAD_AGENT_FILENAME in refusal and ".kiro/agents" in refusal, refusal
        assert (
            "edited by hand" not in refusal and "earlier governance policy" not in refusal
        ), f"the shadow reused another refusal's sentence, whose remedy is wrong: {refusal}"
        with pytest.raises(agent.ForkGovernanceUnresolved, match="project declares its own"):
            agent.require_fork_governance("kirocrew-team-lead", project)

    def test_a_project_shadow_is_refused_under_the_declared_name_too(self, tmp_path, monkeypatch):
        """Both spellings, for the reason the fork check already gives: a binding can
        carry the file STEM where the declared name differs, and the backend resolves
        either against the project directory. Checking only the binding name would let
        the other spelling through."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        project = tmp_path / "checkout"
        (project / ".kiro" / "agents").mkdir(parents=True)
        (project / ".kiro" / "agents" / TEAM_LEAD_AGENT_FILENAME).write_text(
            json.dumps({"name": "kirocrew-team-lead", "allowedTools": ["execute_bash"]}),
            encoding="utf-8",
        )
        # The BINDING is some other spelling; the DECLARED name is ours.
        refusal = agent.spec_start_refusal(
            "some-binding-name", "kirocrew-team-lead", project_dir=project
        )
        assert refusal is not None and "project declares its own" in refusal, refusal

    def test_no_project_shadow_admits_the_start(self, tmp_path, monkeypatch):
        """The over-tight control. A refusal that did not actually test the project
        directory would refuse every start in every checkout, which is a total outage
        rather than a guard. Driven with a project dir that EXISTS and declares other
        agents, so the admit is not an artifact of there being nothing to scan."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        project = tmp_path / "checkout"
        (project / ".kiro" / "agents").mkdir(parents=True)
        (project / ".kiro" / "agents" / "some-other-agent.json").write_text(
            json.dumps({"name": "some-other-agent"}), encoding="utf-8"
        )
        assert agent.spec_start_refusal("kirocrew-team-lead", project_dir=project) is None, (
            "a project declaring unrelated agents refused this one, so no checkout can " "start it"
        )
        agent.require_fork_governance("kirocrew-team-lead", project)
        # And with no project dir at all.
        assert agent.spec_start_refusal("kirocrew-team-lead", project_dir=None) is None

    def test_a_recorded_fork_keeps_its_own_shadow_refusal(self, tmp_path, monkeypatch):
        """The fork lane stays out, fourth round running. A shadowed FORK is refused by
        the check that was already there, with its own fork-specific sentence -- this
        one must not take that case over, because the remedies differ: a fork can also
        be repaired by dropping the fork record."""
        _stub_environment(tmp_path, monkeypatch)
        assert agent._install_team_lead_agent() is agent.InstallOutcome.WRITTEN
        project = tmp_path / "checkout"
        (project / ".kiro" / "agents").mkdir(parents=True)
        (project / ".kiro" / "agents" / TEAM_LEAD_AGENT_FILENAME).write_text(
            json.dumps({"name": "kirocrew-team-lead"}), encoding="utf-8"
        )
        agent_state.set_fork_info(
            "kirocrew-team-lead", forked_from="kirocrew-conductor", private_to="a-crew"
        )
        import kiro_crew.agent_materialization.fork_refresh as fr

        fr._fork_refresh_settled.set()
        monkeypatch.setattr(fr, "_fork_refresh_failed", frozenset())
        with pytest.raises(agent.ForkGovernanceUnresolved, match="private template copy"):
            agent.require_fork_governance("kirocrew-team-lead", project)

    def test_a_crewmate_bound_to_the_template_runs_the_template_file(self):
        """Trap two, answered by reading rather than assumed. ``kirocrew agent create
        --kiro-agent kirocrew-team-lead`` stores a NAME, not a copy of the bytes:
        ``KiroCrewAgentConfig.kiro_agent`` is documented as the "Kiro agent name (modeId
        for session/set_mode)" at ``config/sections.py``, so the crewmate resolves the
        template file at start time.

        So there is no crewmate-with-its-own-bytes case to protect on this path, and
        refusing member starts is not collateral damage -- those sessions are precisely
        the ones that would run the hand-edited file. A private COPY is a different
        mechanism: fork/publish lands its own bytes under its own name with a lineage
        record, which is the lane the test above protects."""
        from kiro_crew.config.sections import KiroCrewAgentConfig

        cfg = KiroCrewAgentConfig(kiro_agent="kirocrew-team-lead")
        assert cfg.kiro_agent == "kirocrew-team-lead"
        assert not hasattr(cfg, "spec"), "a crewmate carrying its own spec bytes"
        fields = {f for f in vars(cfg)}
        assert not any("prompt" in f or "allowedTools" in f for f in fields), (
            f"the crewmate record holds spec content, so it is a copy and not a "
            f"reference: {sorted(fields)}"
        )


class TestTeamLeadCanDoTheWorkItself:
    """Capability 1, which is the gap the crewmate exists to close. Asserted
    against the emitted tool list, never against a sentence about it."""

    def test_the_writers_and_shell_are_mounted(self, tmp_path, monkeypatch):
        data = _install(tmp_path, monkeypatch)
        for tool in ("fs_write", "execute_bash", "code"):
            assert tool in data["tools"], tool

    def test_the_tool_list_is_not_a_conductor_literal(self, tmp_path, monkeypatch):
        """Each conductor installer OVERWRITES the template's list with a literal
        that withholds the writers. This installer appends to the template, which
        is what makes the do-it-yourself half true of the spec."""
        data = _install(tmp_path, monkeypatch)
        assert "fs_write" in data["tools"]
        assert "@kirocrew-dashboard" in data["tools"]
        assert "@kirocrew-work" in data["tools"]

    def test_writing_a_file_still_passes_the_gate(self, tmp_path, monkeypatch):
        """Mounted and ungranted, which is the default agent's own posture:
        ``fs_write`` and ``execute_bash`` reach the PreToolUse gate rather than
        skipping it through ``allowedTools``, whose entries are name-scoped with
        no argument matching."""
        allowed = set(_install(tmp_path, monkeypatch)["allowedTools"])
        assert "fs_write" not in allowed
        assert "execute_bash" not in allowed


class TestTeamLeadGrantsAreExactlyTheIntendedTuple:
    def test_every_grant_comes_from_an_existing_tuple(self, tmp_path, monkeypatch):
        """No grant name of this spec's own. Each tuple keeps its own invariant
        comment as the justification, and a reviewer asking what a new
        auto-approve widens has nothing to be shown."""
        data = _install(tmp_path, monkeypatch)
        shipped = {ref for tuple_ in agent._TEAM_LEAD_SHIPPED_GRANTS for ref in tuple_}
        # ``fs_read`` is the template's, and survives; its ``@kirocrew-core`` does
        # not, because this spec grants that server verb by verb.
        assert set(data["allowedTools"]) == shipped | {"fs_read"}
        # Named explicitly, because the comparison above reads the shipped list and
        # so cannot notice a tuple being added to it. ``work_report`` writes into a
        # PARENT's record across a dispatch relationship, which is why every
        # conductor withholds it, and this agent is the root of its own goal and has
        # no parent to report to. ``work_brief`` is granted and is the control: it
        # proves the probe reads the work server's refs rather than matching nothing.
        assert "@kirocrew-work/work_report" not in data["allowedTools"]
        assert "@kirocrew-work/work_brief" in data["allowedTools"]

    def test_the_grant_list_documents_exactly_the_tuples_it_ships(self):
        """A tuple added to the shipped list without a line in the comment above it
        is how a withheld verb arrives unnoticed: the enumeration test reads that
        same list, so only the documentation disagrees. Keep the two in step."""
        path = _REPO_ROOT / "src" / "kiro_crew" / "agent_materialization" / "team_lead_agent.py"
        source = path.read_text(encoding="utf-8")
        # Every lookup below is checked before it is used. This test reads SOURCE
        # TEXT, so a reword of either anchor is a likely future edit, and an
        # unchecked ``index`` answers that edit with a ValueError traceback naming
        # nothing -- which reads as a broken test rather than as a renamed anchor.
        decl = "_TEAM_LEAD_SHIPPED_GRANTS: tuple"
        comment_head = "#: The grant tuples this spec ships"
        head, found, rest = source.partition(decl)
        assert found, f"{path.name} no longer declares {decl!r}; update this test's anchor"
        assert comment_head in head, (
            f"{path.name} no longer opens its grant-list comment with {comment_head!r}; "
            "update this test's anchor"
        )
        comment = head[head.rindex(comment_head) :]
        open_at = rest.find("(")
        close_at = rest.find(")")
        assert 0 <= open_at < close_at, (
            f"the {decl!r} assignment in {path.name} is not the parenthesised tuple "
            "this test reads; update this test"
        )
        listed = [
            line.strip().rstrip(",").removeprefix("agent_mod.")
            for line in rest[open_at:close_at].splitlines()
            if line.strip().startswith("agent_mod.")
        ]
        assert len(listed) == 3, f"the shipped list names {len(listed)} tuples: {listed}"
        for name in listed:
            assert f"``{name}``" in comment, f"{name} is shipped and not documented"

    def test_the_whole_server_grant_does_not_survive_beside_the_named_verbs(
        self, tmp_path, monkeypatch
    ):
        """The narrowing has to be real. Both backends resolve a whole-server
        reference before a per-tool one, so a bare ``@kirocrew-core`` left beside
        the named verbs would auto-approve every verb on that server -- including
        the ones that START work from ingested context -- and would survive the
        governance ceiling's strip of any single verb."""
        data = _install(tmp_path, monkeypatch)
        allowed = set(data["allowedTools"])
        assert "@kirocrew-core" not in allowed
        assert "@kirocrew-core/monitor_start" in allowed
        for never in (
            "@kirocrew-core/task_run",
            "@kirocrew-core/workflow_run",
            "@kirocrew-core/spawn_run",
            "@kirocrew-core/cron_add",
        ):
            assert never not in allowed, never
        # Mounted, so a governed verb still works through the approval gate.
        assert "@kirocrew-core" in data["tools"]

    def test_narrowing_keeps_a_per_verb_entry_on_the_same_server(self):
        """Only an EXACT whole-server match is dropped. A control, because a pass
        that dropped every ref containing the server name would silently remove
        the grants this spec exists to ship."""
        kept = agent._narrow_whole_server_grants(
            ["@kirocrew-core", "@kirocrew-core/monitor_start", "fs_read", "@builder-mcp"]
        )
        assert kept == ["@kirocrew-core/monitor_start", "fs_read", "@builder-mcp"]

    def test_the_dispatch_and_patrol_verbs_are_granted(self, tmp_path, monkeypatch):
        """An unattended patrol cycle must not stall on an approval nobody is
        there to give."""
        allowed = set(_install(tmp_path, monkeypatch)["allowedTools"])
        for ref in (
            "@kirocrew-dashboard/session_create",
            "@kirocrew-dashboard/chat_folder_file_self",
            "@kirocrew-core/monitor_start",
            "@kirocrew-core/resource_status",
            "@kirocrew-core/session_ledger_record",
            "@kirocrew-work/work_ledger_record",
            "@kirocrew-work/work_brief",
        ):
            assert ref in allowed, ref

    def test_the_peer_mutating_verbs_stay_gated_on_the_spec(self, tmp_path, monkeypatch):
        """The three verbs that MUTATE a peer session are earned by an ownership
        fence a spec does not have: ``authorize_target`` refuses a MEMBER caller
        on a session it did not create. A crewmate's own thread is granted them
        from that fence at session establishment, so the spec withholds them."""
        allowed = set(_install(tmp_path, monkeypatch)["allowedTools"])
        for ref in (
            "@kirocrew-dashboard/session_send",
            "@kirocrew-dashboard/session_broadcast",
            "@kirocrew-dashboard/session_stop",
            "@kirocrew-dashboard",
        ):
            assert ref not in allowed, ref
        assert "@kirocrew-dashboard" in _install(tmp_path, monkeypatch)["tools"]

    def test_the_panel_server_is_never_emitted(self, tmp_path, monkeypatch):
        """A spec may not carry it at all: it is ``opt_in`` and host-injected per
        session, and its verbs resolve their crew from the session. A spec that
        listed them would ship tools answering ``no_crew``."""
        data = _install(tmp_path, monkeypatch)
        assert "@kirocrew-panel" not in data["tools"]
        assert "kirocrew-panel" not in data["mcpServers"]
        assert not [ref for ref in data["allowedTools"] if "kirocrew-panel" in ref]

    def test_the_ceiling_filters_the_grants_and_the_derived_rules(self, tmp_path, monkeypatch):
        """``allowedTools`` is the one path that never reaches the gate, so a host
        governing a verb gets a prompt rather than a bypass -- and the KAS rules
        are derived from the FILTERED list, so a stripped grant loses its rule."""
        data = _install(
            tmp_path,
            monkeypatch,
            may_auto_approve=lambda ref: ref != "@kirocrew-core/monitor_start",
        )
        assert "@kirocrew-core/monitor_start" not in data["allowedTools"]
        rules = json.dumps(data["permissions"])
        assert "monitor_start" not in rules
        assert "monitor_update" in rules

    def test_the_two_opt_in_servers_are_assigned_by_hand(self, tmp_path, monkeypatch):
        """Neither spec-writing loop emits an ``opt_in`` set, so naming them here
        IS the per-agent assignment. The template's own servers survive, because
        an agent that runs a build may need them."""
        mcp = _install(tmp_path, monkeypatch)["mcpServers"]
        assert mcp["kirocrew-dashboard"]["args"] == ["mcp-dashboard"]
        assert mcp["kirocrew-work"]["args"] == ["mcp-work"]
        assert "kirocrew-core" in mcp


class TestTeamLeadCharterStatesEachCapability:
    """One assertion per capability the crewmate ships, because for a charter the
    sentence IS the mechanism: an agent that is not told to call the evaluator
    reads the claim instead."""

    def test_it_decides_between_doing_and_dispatching(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        assert "ONE acceptance condition" in charter
        assert "Everything else is dispatched." in charter

    def test_it_registers_the_work_before_starting(self, tmp_path, monkeypatch):
        """Generic by design: a step the owner configures, never a named crew."""
        charter = _charter(tmp_path, monkeypatch)
        assert "run the owner's intake step" in charter

    def test_it_dispatches_in_the_mandated_order(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        create = charter.index("`work_ledger_record` `action=create`")
        dispatch = charter.index("`session_create` with a title")
        bind = charter.index("`action=bind`")
        seed = charter.index("`session_send` the seed")
        assert create < dispatch < bind < seed
        assert "Bind before you seed." in charter
        assert "ONE brief file that every seed names by path" in charter

    def test_nesting_is_the_default_and_is_one_level(self, tmp_path, monkeypatch):
        """The promise is graded against what the server permits: a ledger item at
        the third conducting level is refused, so a charter promising more would
        promise what the server refuses."""
        charter = _charter(tmp_path, monkeypatch)
        assert "dispatch a conductor rather than a worker" in charter
        assert "That nesting is one level" in charter

    def test_patrol_is_event_driven(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        assert 'watch="work-ledger"' in charter

    def test_loop_health_is_a_runtime_reading_not_a_memory(self, tmp_path, monkeypatch):
        """A dead loop reads exactly like a working one, so the charter names the
        two fields the runtime DERIVES -- the ``bind`` reply's ``patrol`` and the
        compact row's ``unpatrolled`` -- and keeps ``monitor_inspect`` as the
        fallback. A rule that tells the agent to remember having armed a loop is
        the failure it is written against."""
        charter = _charter(tmp_path, monkeypatch)
        assert "Loop health is checked, never remembered" in charter
        assert "`unpatrolled`" in charter
        assert "a `patrol` field" in charter
        assert "`monitor_inspect` is the fallback" in charter

    def test_acceptance_is_the_evaluator_and_never_a_claim(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        assert "scripts/accept_eval.py" in charter
        assert "`action=decide`" in charter
        assert "`action=accept`" in charter
        assert "a CLAIM, never an acceptance" in charter
        assert "Nothing a child can write reaches `verdict`" in charter

    def test_the_head_sha_is_read_in_the_turn_it_is_reported(self, tmp_path, monkeypatch):
        """A child's green is a reading of some head, and a rebase moves it."""
        charter = _charter(tmp_path, monkeypatch)
        assert "names the head sha you read from git in that same turn" in charter

    def test_a_stalled_fleet_is_not_read_as_a_busy_one(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        assert "`stale` at once is a STOPPED fleet" in charter

    def test_the_seed_quotes_the_ask_and_the_child_echoes_it(self, tmp_path, monkeypatch):
        """Both halves, because one without the other does not catch the failure:
        a verbatim seed nobody reads back is still a seed nobody read."""
        charter = _charter(tmp_path, monkeypatch)
        assert "restates the owner's ask VERBATIM" in charter
        assert "Require the echo." in charter

    def test_it_drives_the_dashboard_and_writes_no_numbers(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        assert "`dashboard_fields`" in charter
        assert "`dashboard_write`" in charter
        assert "`verdict`" in charter
        assert "`for_you`" in charter
        assert "Write judgement, never arithmetic." in charter

    def test_its_own_state_survives_a_restart(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        assert "`session_ledger_record`" in charter
        assert "a compaction or a restart resumes the patrol" in charter

    def test_capacity_is_read_and_never_a_number_it_holds(self, tmp_path, monkeypatch):
        charter = _charter(tmp_path, monkeypatch)
        assert "`resource_status`" in charter
        assert "hold no count of your own for how many sessions a goal may run" in charter

    def test_the_charter_carries_no_polling_prose(self, tmp_path, monkeypatch):
        """An event-driven patrol described as a timer is a charter that teaches
        the opposite of what it mounts. The control proves the probe reads."""
        charter = _charter(tmp_path, monkeypatch).lower()
        for banned in ("poll", "on a timer", "re-check in"):
            assert banned not in charter, banned
        assert "work-ledger" in charter

    def test_the_charter_names_its_own_skill_as_the_procedure(self, tmp_path, monkeypatch):
        """Wiring, not wording. A charter pointing at a conductor's skill would
        send this agent to a procedure whose first rule is that it does no work
        itself, which is the one capability it exists to add. The shipped skill
        directory and the name the charter reads must be the same string."""
        charter = _charter(tmp_path, monkeypatch)
        assert "The `team-lead` skill carries the operating procedure" in charter
        assert (_SKILLS / "team-lead" / "SKILL.md").is_file()
        # The two scripts still come from where they are maintained.
        assert "`goal-conductor` skill's `scripts/accept_eval.py`" in charter

    def test_the_retired_verbosity_token_is_absent(self, tmp_path, monkeypatch):
        """Reply style arrives as session-context chrome for every agent, so a
        token left here reaches the model as a literal."""
        assert "{{VERBOSITY_BLOCK}}" not in _install(tmp_path, monkeypatch)["prompt"]


class TestTheCrewmateIsMadeByOneDocumentedCommand:
    """The product ships the TEMPLATE. Binding a crewmate to it is the operator's
    one command, and these tests pin that the command documented is the command
    the CLI actually accepts -- a doc naming a flag the parser does not take is
    worse than no doc, because it fails only when somebody follows it."""

    def test_the_cli_accepts_the_documented_command(self):
        """Read off the parser rather than trusted: ``--name`` is required and
        ``--kiro-agent`` takes the template name."""
        cli = (_REPO_ROOT / "src" / "kiro_crew" / "cli.py").read_text(encoding="utf-8")
        assert 'agent_sub.add_parser("create"' in cli
        assert 'agent_create.add_argument("--kiro-agent"' in cli
        assert '"--name",\n        required=True,' in cli

    def test_the_rfc_and_the_skill_both_name_it(self):
        """Both readers of this feature need it, and neither should invent its own
        spelling: an operator reading the RFC and an agent reading the skill."""
        command = "kirocrew agent create --name <name> --kiro-agent kirocrew-team-lead"
        rfc = (_REPO_ROOT / "docs" / "request-for-change" / "rfc-lead-crewmate.md").read_text(
            encoding="utf-8"
        )
        skill = (_SKILLS / "team-lead" / "SKILL.md").read_text(encoding="utf-8")
        assert command in rfc
        assert command in skill

    def test_no_config_switch_and_no_boot_path_provisioning_remain(self):
        """The product creates no crewmate, so nothing may read a switch for one
        and nothing may run on the boot path for one. Asserted against the three
        files that carried it, with a control proving each probe reads."""
        for rel, control in (
            (("src", "kiro_crew", "config", "sections.py"), "crew_panel"),
            (("src", "kiro_crew", "config", "loader.py"), "crew_panel"),
            (("src", "kiro_crew", "dashboard", "server.py"), "_kick_crewmate_prune"),
        ):
            text = (_REPO_ROOT.joinpath(*rel)).read_text(encoding="utf-8")
            assert control in text, f"{rel[-1]}: the probe does not read"
            assert "team_lead_crewmate" not in text, rel[-1]


class TestTeamLeadSkillReusesRatherThanCopies:
    def test_the_skill_ships_in_the_tree(self):
        assert (_SKILLS / "team-lead" / "SKILL.md").is_file()

    def test_it_delegates_the_shared_procedure_instead_of_restating_it(self):
        """The skill is a DELTA. Dispatch order, patrol, acceptance, stop
        conditions, durable state and capacity live in ``goal-conductor``, and a
        procedure stated twice is one that drifts until a reader follows neither
        copy. So the test is that the pointer is there and the restatement is
        not."""
        skill = (_SKILLS / "team-lead" / "SKILL.md").read_text(encoding="utf-8")
        assert "goal-conductor" in skill
        assert "It is your procedure" in skill
        # The sections it defers rather than repeats, each named by a phrase the
        # long form used. A control follows, so an empty match cannot pass.
        for restated in ("`action=bind`", "`monitor_start`", "`action=verdict`"):
            assert restated not in skill, restated
        assert "do-it-yourself test" in skill

    def test_no_copy_of_either_script_ships_with_it(self):
        """The reuse is the point. A copy is a second file whose later divergence
        from the original nothing in the tree would detect."""
        for script in ("accept_eval.py", "patrol_budget.py"):
            assert not (_SKILLS / "team-lead" / "scripts" / script).exists(), script

    def test_the_skill_carries_no_polling_prose(self):
        """The control is a term this delta carries itself. Patrol belongs to
        ``goal-conductor``, so a term from the patrol section would prove nothing
        about whether this probe reads."""
        skill = (_SKILLS / "team-lead" / "SKILL.md").read_text(encoding="utf-8").lower()
        for banned in ("on a timer", "re-check in", "poll"):
            assert banned not in skill, banned
        assert "goal-conductor" in skill
