"""The team-lead crewmate spec: the one generated agent that both leads and works.

Its own module because it shares neither sibling's shape. Every conductor spec in
:mod:`conductor_agents` overwrites the template's tool list with a literal that
withholds ``fs_write`` and ``code``, which is what makes "never does a work item's
work itself" true against the tool list rather than only against the charter. The
dispatched worker in :mod:`worker_agent` keeps the writers and mirrors the default
spec on disk, but an ``opt_in`` set is assigned per agent, so that installer
deliberately strips ``kirocrew-dashboard`` from the mirror and a worker cannot
dispatch at all.

This spec is the union those two leave empty: the template's whole toolset, so a
small focused item it keeps for itself is one it can finish, plus session control
and the work ledger, so everything else is a fleet it dispatches, patrols and
decides for.

EVERY path that writes this spec's filename, and the one invariant each is held
to: never write there when the existing content was not written by this
installer, whether or not that content parses.

* :func:`_install_team_lead_agent` is the only writer in this package, and the
  only one that replaces the file WHOLE with nothing carried forward. Three
  callers reach it and all three are the same function -- the boot rebuild, the
  hourly retry of held specs, and ``kirocrew setup --agent-only --clean`` -- so
  the attribution in one place covers all of them.
* Two shared sweeps edit an owned spec IN PLACE rather than replacing it: the
  retired-key repair that walks ``OWNED_KIRO_AGENT_FILES`` in ``agent.py``, and
  the Playwright convergence in ``browser/setup.py``. Both are keyed on the
  filename rather than on provenance, which is a property every owned spec has
  shared since before this one existed; neither discards a file, and changing
  that shape for one filename is not this installer's to do.

The spec is half of the capability, and the half it is not is worth stating here
rather than leaving to the charter. Driving the owner's dynamic dashboard needs
``dashboard_fields`` / ``dashboard_write``, those verbs resolve the calling crew
from the SESSION and not from an argument, and the server behind them is
``opt_in`` with no spec allowed to emit it at all
(``members.member_panel_session_server``). So they answer in a crewmate's own
thread and nowhere else: a spec cannot confer that on itself, and this installer
does not pretend to. What the spec carries is everything that works without it.
"""

from __future__ import annotations

import copy
import enum
import os
from pathlib import Path
from typing import Any

from kiro_crew import agent as agent_mod
from kiro_crew import agent_discovery, agent_state
from kiro_crew.agent_files import TEAM_LEAD_AGENT_FILENAME as _TEAM_LEAD_AGENT_FILENAME
from kiro_crew.agent_materialization import auto_approve, conductor_agents, managed_mcp

#: This spec's agent NAME, and also the key its lineage is recorded under: kiro-cli
#: resolves an agent by reading ``<agents dir>/<name>.json``, and ``agent_state`` keys
#: a template's fork record by that same name, so the filename and the record share
#: one string by construction rather than by a second spelling.
_TEAM_LEAD_AGENT_NAME = Path(_TEAM_LEAD_AGENT_FILENAME).stem


def _team_lead_mcp_servers(config: dict[str, Any]) -> dict[str, Any]:
    """The template's own ``mcpServers`` map plus this spec's two explicit assignments.

    ADDITIVE, where :func:`conductor_agents._conductor_mcp_servers` is subtractive,
    and the difference follows from the charter: a conductor keeps ``kirocrew-core``
    and drops the rest because it does no work, while this agent runs a build and
    drives git, so a server the template assembles is a server its own half of the
    charter may need.

    ADDITIVE, so what bounds it is stated here rather than assumed. Not mirroring the
    default spec on disk removes one inheritance channel and not the other: this builds
    from ``build_agent_config``, which deep-merges the operator's own ``agent.json`` --
    ``mcpServers`` included -- so an entry they wrote arrives through the template
    itself. Two passes in :func:`_install_team_lead_agent` are what bound it, and
    neither is optional for an additive map: the unassigned ``opt_in`` sets are dropped
    (:func:`_team_lead_unassignable_servers`), and every surviving entry's
    ``autoApprove`` goes through the governance ceiling, which the ``allowedTools``
    filter cannot see. A conductor needs neither, because
    ``conductor_agents._conductor_mcp_servers`` is SUBTRACTIVE and keeps nothing it did
    not put there; this map keeps what it is given, so it filters what it keeps.

    ``kirocrew-dashboard`` and ``kirocrew-work`` are hand-built because both are
    ``opt_in``: neither spec-writing loop emits one, and this installer naming them
    IS the explicit per-agent assignment such a set requires. The dashboard entry
    carries the two fields a copy forgets -- without ``"type": "registry"`` a
    registry-mode client silently DROPS it, so the granted session-control tools
    never launch and the dispatch half of this charter is dead with no local error;
    without the ``KIROCREW_HOME`` pin the shim reads the default data home while the
    gateway runs under an override, so dispatch would act on a different session
    store than the one it reports on. Both helpers return empty on a default
    install, so the emitted spec is unchanged there.
    """
    mcp = dict(config.get("mcpServers") or {})
    dash_cmd, dash_args = agent_mod._kirocrew_mcp_invocation("mcp-dashboard")
    dash_entry: dict[str, Any] = {"command": dash_cmd, "args": dash_args}
    if managed_mcp._mcp_registry_mode():
        dash_entry["type"] = managed_mcp._MCP_REGISTRY_TYPE
    dash_env = managed_mcp._managed_mcp_env()
    if dash_env:
        dash_entry["env"] = dash_env
    mcp["kirocrew-dashboard"] = dash_entry
    mcp["kirocrew-work"] = managed_mcp._managed_opt_in_entry("mcp-work")
    return mcp


#: The grant tuples this spec ships, in the order they are appended. Every name is
#: an existing tuple reused VERBATIM, so this installer introduces no grant name of
#: its own -- which is the strongest answer to a reviewer asking what a new
#: auto-approve widens, and it keeps each tuple's own invariant comment as the
#: justification rather than restating it here.
#:
#: ``_CONDUCTOR_CORE_GRANTS`` is the patrol loop's own lifecycle, the capacity reads
#: this charter uses instead of holding a session count, the durable store it
#: resumes a goal from, skill lookup for seeding a child, and the person-facing
#: reports. ``_CONDUCTOR_DASHBOARD_GRANTS`` is the create-and-read half of session
#: control, and the half this spec may grant: the three verbs that MUTATE a peer
#: session stay out, because they are earned by an ownership fence a spec does not
#: have -- ``authorize_target`` refuses a MEMBER caller on any session it did not
#: itself create, and a crewmate's own thread is granted them from that fence at
#: session establishment rather than from here. ``_LEDGER_CONDUCTOR_WORK_GRANTS``
#: is the ledger every dispatched item is recorded in, plus ``work_brief``, whose
#: cost the tuple's own comment states: a nested lead's mandated FIRST call, in a
#: child session nobody opened, is an approval stall before any planning happens.
#:
#: Three tuples, and ``_WORKER_WORK_GRANTS`` is deliberately not a fourth. It would
#: contribute exactly one verb this list does not already hold -- ``work_report`` --
#: and that verb is withheld from every conductor on a rule its own tuple states:
#: it WRITES into the parent's record, across a dispatch relationship. The crewmate
#: is the root of its own goal and has no parent to report to, so the grant would
#: widen an auto-approve across a trust boundary for a call that never comes.
_TEAM_LEAD_SHIPPED_GRANTS: tuple[tuple[str, ...], ...] = (
    agent_mod._CONDUCTOR_CORE_GRANTS,
    agent_mod._CONDUCTOR_DASHBOARD_GRANTS,
    agent_mod._LEDGER_CONDUCTOR_WORK_GRANTS,
)


#: The two servers that together ARE the leading half of this agent, and so the
#: provenance marks :func:`_foreign_team_lead_spec_reason` reads. Every release that
#: writes this spec writes both, by construction rather than by convention: without
#: ``kirocrew-dashboard`` it cannot dispatch and without ``kirocrew-work`` it cannot
#: record what it dispatched, and either absence makes the charter false.
_DEFINING_SERVERS: tuple[str, ...] = ("kirocrew-dashboard", "kirocrew-work")


def _team_lead_unassignable_servers() -> frozenset[str]:
    """Managed ``opt_in`` servers this spec must not carry in from the template.

    An ``opt_in`` server is an ASSIGNABLE SET rather than an always-on capability:
    neither spec-writing loop emits one, and an agent that needs one hand-builds the
    entry, which IS the per-agent assignment such a set requires. ``build_agent_config``
    deep-merges the operator's own ``agent.json``, ``mcpServers`` included, so a set they
    mounted on their personal agent would otherwise ride into this spec -- assigned to
    that agent, never to this one. ``kirocrew-panel`` is why this is load-bearing rather
    than tidy: the charter's own account of this agent says no spec may emit that server,
    and inheriting it would make that true only of hosts whose operator mounted nothing.

    The exclusion is ``opt_in`` MINUS :data:`_DEFINING_SERVERS`, and the reuse is exact
    rather than convenient: the two servers this installer hand-builds are the two it
    assigns, so every other opt-in set is one nobody assigned here. Derived from the
    registry rather than listed, so an opt-in server added tomorrow is withheld by
    default instead of riding in until someone notices.
    """
    return frozenset(
        name
        for name, spec in agent_mod._MANAGED_MCP_SERVERS.items()
        if spec.get("opt_in") and name not in _DEFINING_SERVERS
    )


class InstallOutcome(enum.Enum):
    """What one install attempt did to the spec on disk.

    Three values because a rebuild asks two different questions of a file it did
    not write: is this spec's own write still owed, and is another attempt worth
    making. ``WRITTEN`` settles both no. ``HELD`` is the only one that answers
    the second yes, which is why it is not folded into either of the others: it
    means this installer could not write ITS OWN spec and a later pass should
    try again. ``DECLINED`` is permanent and touches nothing.

    What governs the grants on a declined file is NOT any of these values. The
    install does not sanitize a spec it cannot attribute; the session start
    refuses to run it (:func:`spec_start_refusal`). So a declined file
    carries whatever the operator wrote, and carries it nowhere.
    """

    WRITTEN = "written"
    HELD = "held"
    DECLINED = "declined"


def _foreign_team_lead_spec_reason(spec: dict[str, Any] | None) -> str | None:
    """Why *spec*, read from this installer's path, is not its own, or ``None``.

    PROVENANCE, not existence, taking
    ``worker_agent._foreign_worker_spec_reason``'s rule: a file at this path the
    installer did not write is not an out-of-date spec, and replacing it destroys
    whatever put it there. The spec id makes a collision unlikely rather than
    impossible, so the guard is what covers the operator who happens to hold a
    hand-authored agent at this name.

    Takes a parsed object, or ``None`` for an ABSENT path. ``None`` -- "ours to
    write" -- is the only state that writes without a name to check, and a file that
    is present and does not parse never reaches here: the caller declines it on
    :data:`_UNATTRIBUTABLE`, because unreadable bytes at this name are still
    somebody's.

    The marks are the declared ``name`` and an ``mcpServers`` entry for each of
    :data:`_DEFINING_SERVERS` -- the shape THIS release writes, and only that one.
    A looser test that also accepted the reference from ``tools`` or a per-tool
    grant was considered and dropped: there is no earlier release of this spec to
    heal, so the leniency would have bought nothing and widened what counts as
    this installer's own file.

    The bound is worth stating exactly, because the marks do not carry it alone.
    Neither mark is a secret, so a file holding both is indistinguishable BY CONTENT
    from this installer's own output. Two different files reach that description and
    they are not the same case:

    * a DELIBERATE forge, which volunteers the forger's own file and costs them only
      what they chose to lose;
    * a crew's PRIVATE COPY, which arrives at both marks with nobody choosing
      anything. A copy is named after the crew that owns it and its declared ``name``
      is set equal to that stem, so a crew named for this agent lands a file declaring
      this agent's name; copy a conductor and that file mounts both
      :data:`_DEFINING_SERVERS` too, because every conductor spec mounts them.

    A third file reaches it too, and it is the common one: the installed spec after an
    operator edits its prompt or its model in place. That edit changes neither mark.

    So the marks are not the ownership signal and this function does not decide a
    replace. It is a cheap PRE-FILTER with a useful error message -- it names which
    mark a file fails, which a digest comparison cannot -- and
    :func:`_attribution_reason` requires :func:`_unconfirmed_digest_reason` after it.
    Authorship is the digest the installer recorded for its own last write; a mark is
    a shape anything can hold.

    What the marks still rule out early, before any sidecar read, is the reverse -- an
    ordinary crew, with its own charter and its own servers, being read as this spec
    because it occupies the filename.
    """
    if not isinstance(spec, dict):
        return None
    declared = spec.get("name")
    if declared != "kirocrew-team-lead":
        return f"it declares the agent name {declared!r}"
    # Shape-checked before it is read, not only its members: a hand-edited
    # ``"mcpServers": 1`` would otherwise raise out of an attribution, which
    # reaches the installer's caller as an error rather than as a declined write.
    servers = spec.get("mcpServers")
    if not isinstance(servers, dict):
        return "it declares no mcpServers map"
    for server in _DEFINING_SERVERS:
        if server not in servers:
            return f"it does not mount the {server} server this agent leads through"
    return None


def _recorded_fork_reason() -> str | None:
    """Why this installer's name holds a crew's private copy, or ``None``.

    The one signal the bytes cannot carry. A private copy is recorded in the lineage
    sidecar when it is published -- Crew writes ``forked_from`` / ``private_to``
    there because a spec file cannot hold them (kiro-cli rejects unknown fields) --
    so the record says whose the file is where the marks only say what shape it has.

    Reachable for a copy published BEFORE this stem became an owned filename.
    ``fork_publish`` counts every ``OWNED_KIRO_AGENT_FILES`` stem as taken, so no new
    copy lands here while that list holds this one; a record at this name is what
    remains of one that landed while the name was still free.

    Read STRICT, so an unreadable sidecar raises the reader's transient class and the
    caller HOLDS rather than writes. Degrading it to "no record" is the hole this
    closes: one failed read in the gap would be indistinguishable from a file nobody
    claims, and the copy would be replaced on that guess.
    """
    try:
        fork = agent_state.get_fork_info(_TEAM_LEAD_AGENT_NAME, strict=True)
    except (OSError, ValueError) as exc:
        raise conductor_agents._SpecUnusable(
            f"the lineage sidecar could not be read to tell whether this name holds a"
            f" crew's private copy ({exc}); the file may be that copy",
            replace=False,
        ) from exc
    if fork is None:
        return None
    return (
        f"the lineage sidecar records it as {fork['private_to']!r}'s private copy of"
        f" {fork['forked_from']!r}"
    )


def _attribution_reason(spec: dict[str, Any] | _Unattributable | None) -> str | None:
    """Why the file at this installer's path is not its own to replace, or ``None``.

    The WHOLE attribution, and the order is the invariant: the record before the
    marks. The marks describe the shape this release writes and the fork feature can
    put that shape on a crew's file innocently, so a recorded copy has to be rejected
    before any mark test gets the chance to accept it.

    Raises the reader's transient class when the lineage sidecar cannot be read, which
    the caller turns into a HOLD, for the same reason the spec read does: a check that
    could not run is not a check that passed.
    """
    if isinstance(spec, _Unattributable):
        return "its bytes are not a spec this installer can attribute to anyone"
    if spec is None:
        # ABSENT: the one state with nothing to destroy, and so the one that writes
        # without a name to check. A record naming this name with no file at it has no
        # bytes to keep, and the publish that records lineage before its destination
        # file exists passes through exactly this state.
        return None
    return (
        _recorded_fork_reason()
        or _foreign_team_lead_spec_reason(spec)
        or _unconfirmed_digest_reason(spec)
    )


def _unconfirmed_digest_reason(spec: dict[str, Any]) -> str | None:
    """Why *spec*'s bytes are not this installer's own last write, or ``None``.

    THE ownership signal. The marks above are a shape, and a shape is not authorship:
    an operator who opens the installed file and edits its prompt or its model leaves
    the declared name and both mounted servers exactly as they were, and holds no
    lineage record either. On the marks alone that file reads as this installer's and
    is replaced whole on the next rebuild -- so every in-place edit would be silently
    reverted, which is ordinary use rather than a forge.

    What the marks cannot carry, the sidecar can: the installer records the digest of
    the exact bytes it lands (:func:`agent_state.spec_digest` over the canonical form
    ``_atomic_json_write`` writes), so only a file reproducing that value is its own
    last write. Any edit changes the digest, which is the point.

    Requiring it cannot freeze a managed install, because there is no earlier release
    of this spec to heal: the filename becomes owned in the release that adds it, so a
    file at this name is either absent, this installer's own recorded write, or
    somebody else's. A stem with prior releases could not take this rule without a
    heal path for the files they wrote.

    Read STRICT, so an unreadable sidecar raises the reader's transient class and the
    caller HOLDS rather than writes, for the reason :func:`_recorded_fork_reason`
    gives: a check that could not run is not a check that passed.
    """
    try:
        confirmed = agent_state.managed_digest_matches(
            _TEAM_LEAD_AGENT_NAME, agent_state.spec_digest(spec), strict=True
        )
    except (OSError, ValueError) as exc:
        raise conductor_agents._SpecUnusable(
            f"the ownership sidecar could not be read to confirm this file as this"
            f" installer's own write ({exc}); its bytes may be yours",
            replace=False,
        ) from exc
    if confirmed:
        return None
    return (
        "its bytes do not reproduce the digest this installer recorded for its own last"
        " write, so it has been edited or was written by something else"
    )


#: The one refusal, worded for the person who edited the file. It names what to do
#: with their copy FIRST, because the file is intact and renaming it keeps it; the
#: shipped spec comes back only when the name is free.
_HAND_EDITED_REFUSAL = (
    f"{_TEAM_LEAD_AGENT_FILENAME} was edited by hand; Kiro Crew will not run it. "
    "Rename it to keep your copy, or delete it to get the shipped one back."
)

#: The THIRD fact, and the one that blames nobody. The file is ours, unedited, and
#: still must not run: its auto-approvals were derived under an older ceiling and the
#: rebuild that would re-derive them has not succeeded since. Nothing for the operator
#: to undo, so the sentence says what clears it instead of what they did.
_CEILING_DRIFT_REFUSAL = (
    f"{_TEAM_LEAD_AGENT_FILENAME} holds auto-approvals from an earlier governance "
    "policy and Kiro Crew will not run it until a rebuild brings them up to date. "
    "Nothing is wrong with the file; the last rebuild did not finish. It is retried "
    "on the next one, or run `kirocrew setup --agent-only` to do it now."
)


def _governed_fields(spec: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The fields the ceiling decides, and ONLY those, for comparison.

    Scoped deliberately. Comparing whole dicts would read a template prompt or model
    change between releases as governance drift and refuse every start until a rebuild
    ran, which is an outage rather than a guard. These two are what the ceiling moves:
    ``allowedTools``, and the per-server ``autoApprove`` that is the other path never
    reaching the PreToolUse gate.

    ``includeMcpJson`` is OUT because it is pinned to a constant ``False`` and so
    cannot drift with a ceiling; an edit to it is a byte change the attribution catches
    first. ``permissions`` is out for a stronger reason, in :func:`_ceiling_drift_reason`.
    """
    servers = spec.get("mcpServers")
    auto = {
        name: entry["autoApprove"]
        for name, entry in (servers or {}).items()
        if isinstance(entry, dict) and "autoApprove" in entry
    }
    return spec.get("allowedTools"), auto


def _ceiling_drift_reason(spec: dict[str, Any]) -> str | None:
    """Would this installer write different grants than the file holds? Then refuse.

    A DRIFT DETECTOR, not a filter. The passes run on a deep COPY and their result is
    compared with the file; nothing is written, no operator ``deny`` is removed, and no
    judgement about an individual grant is enacted. That distinction is what makes this
    safe where sanitizing was not -- the earlier attempt KEPT what it judged permitted,
    so being wrong about one entry shipped a wrong spec, while being wrong here can
    only refuse a start that a rebuild then clears.

    Stateless on purpose. Every marker the rebuild sets shares one shape: the rebuild
    can die before reaching it. ``write_default_spec`` runs first, so a failure there
    ends the pass before this installer is called and leaves the marker exactly as a
    healthy boot would. Asking the question at admission instead needs no cooperation
    from the rebuild at all.

    THREE passes, which is the installer's own post-append sequence minus one, in its
    order: the whole-server narrowing, the ``allowedTools`` ceiling, the ``autoApprove``
    strip. The pairing matters -- ``_apply_allowed_tools_ceiling`` returns early on its
    own key and never looks at ``mcpServers``, so running it alone and then comparing
    ``autoApprove`` would compare the file against itself and that half would always
    pass. The ceiling plus the strip is the pair ``fork_refresh`` runs for the same
    reason.

    ``_write_derived_permissions`` is the pass left OUT, and not for cost alone.
    ``permissions`` is a PURE FUNCTION of ``allowedTools``
    (``kas_permissions.allowed_tools_to_permissions``, a list plus an agent id), so it
    drifts if and only if ``allowedTools`` drifts -- which is already refused above, so
    including it adds no detection. It would also introduce a FALSE positive this check
    must not have: that writer is version-gated on ``installed_kiro_cli_version()``, so
    an operator who merely upgraded kiro-cli would see the derived block differ from
    their file and be refused for something that is not drift at all.

    Audit side effects are suppressed where a switch exists and left honest where one
    does not. The strip takes ``audit=False``, because its record names
    ``strip_ungoverned_auto_approve`` as the source and a per-start copy of an
    install-time event would be a lie in the feed. The ceiling pass has no such switch
    and emits only when it actually withholds -- that is, only on the refusal path and
    never on a healthy start -- so it is given a SOURCE naming this check, where the
    record reads as what it is: a start refused on drifted grants.
    """
    from kiro_crew.platform import governance

    probe = copy.deepcopy(spec)
    granted = [ref for ref in (probe.get("allowedTools") or []) if isinstance(ref, str)]
    probe["allowedTools"] = _narrow_whole_server_grants(granted)
    auto_approve._apply_allowed_tools_ceiling(probe, source="team-lead-start-drift")
    servers = probe.get("mcpServers")
    if isinstance(servers, dict):
        probe["mcpServers"] = dict(governance.strip_ungoverned_auto_approve(servers, audit=False))
    if _governed_fields(probe) == _governed_fields(spec):
        return None
    agent_mod.logger.warning(
        "%s: its auto-approvals differ from what the current governance ceiling would "
        "produce, so a session on it is refused until a rebuild re-derives them",
        _TEAM_LEAD_AGENT_FILENAME,
    )
    return _CEILING_DRIFT_REFUSAL


def _project_shadow_refusal(
    project_dir: str | Path | None, names: tuple[str | None, ...]
) -> str | None:
    """Refuse when the session's PROJECT declares this agent's name.

    kiro-cli resolves ``--agent`` against ``<cwd>/.kiro/agents`` BEFORE the global
    directory, so a checkout shipping ``.kiro/agents/kirocrew-team-lead.json`` is the
    file that actually runs. Every other check in this module reads the GLOBAL spec,
    which in that situation is a file nothing will execute -- so without this the
    attribution and the drift check both pass on bytes that are not the ones starting.

    A REFUSAL, deliberately, and not attribution-or-drift run against the project copy.
    Two reasons, and both are why re-adding that validation later would be wrong.
    Nothing in this product ever writes into a project directory, so a project spec can
    never be re-derived or brought under the ceiling by any rebuild -- its attribution
    could only ever end in refusal, and a check whose every answer is "refuse" is more
    honestly written as a refusal. And this gate has already made that decision once: a
    shadowed FORK is refused outright rather than validated, so this mirrors the
    posture beside it instead of introducing a second one.

    Its own sentence rather than the fork path's, because that one says "is a private
    template copy" -- true there, false here -- and the two remedies differ: a fork is
    repaired by renaming the project spec OR dropping the fork, while this one has only
    the project file to move.

    BOTH spellings are tested, for the reason the fork check gives: a binding can carry
    the file stem where the declared name differs, and the backend resolves either
    against the project directory.
    """
    if project_dir is None:
        return None
    shadow = agent_discovery.project_agent_names(
        project_dir, operation="team_lead_start_refusal", source="unknown"
    )
    if not any(name in shadow for name in names if name):
        return None
    return (
        f"the session's project declares its own {_TEAM_LEAD_AGENT_FILENAME}, and "
        "kiro-cli resolves a project spec before the installed one, so that copy is "
        "what would run -- ungoverned, because Kiro Crew never writes into a project "
        "directory and cannot bring it under the governance ceiling. Rename or remove "
        "the project's own .kiro/agents spec for this name to run the installed agent."
    )


def spec_start_refusal(*names: str | None, project_dir: str | Path | None = None) -> str | None:
    """Why a session on *names* must not start, or ``None`` to admit it.

    FOUR refusals with four sentences, because they are four different facts about who
    did what, and each names a different remedy. A PROJECT SHADOW means the checkout
    ships its own copy at this name and that copy is what would run; the remedy is to
    move the project file. A hand-edited global file is the operator's own change
    (rename or delete). An unreadable one is the same answer failing closed. DRIFTED
    grants are nobody's mistake -- the file is ours and untouched, the ceiling moved
    under it, and the remedy is a rebuild. An operator who edited nothing and broke
    nothing must not be told otherwise.

    Takes the names the caller is about to start rather than being asked only
    about this spec, so the admission gate does not need a copy of this spec's
    filename: the gate offers every spelling it resolved and this answers for the
    ones that are ours. Any name that is not this spec's is admitted untouched,
    which is what keeps a shared gate from becoming a list of special cases.

    The replacement for sanitizing a declined spec, and a narrower claim than the
    sanitizing was. Rewriting somebody's file to make it safe needs a correct
    judgement about every field it touches, and that judgement was wrong in a new
    way each time it was made -- most sharply on ``permissions``, which
    :data:`kiro_crew.acp.kas_permissions._EFFECTS` defines over ``allow``, ``deny``
    AND ``ask``, so a pass that emptied the block to remove grants could delete a
    DENY the operator wrote and leave them with less restriction than they asked
    for. Refusing needs no such judgement: a spec whose authorship cannot be
    established does not run, and nothing on disk changes.

    Called from the session-start admission gate rather than from the installer,
    because the install already happened (or declined) hours earlier and the file
    can be edited at any point after it. The gate is the only place that sees
    every start.

    ABSENT is admitted, and that is not an oversight: with no file at this name
    there is nothing to refuse, and the caller's own spec resolution answers it.
    A file this installer CONFIRMS as its own is admitted too, by definition.

    A RECORDED FORK never reaches this. The gate calls it only on its own non-fork
    path, so the fork lane -- which has its own governance, a refresh that
    re-filters its grants and a wait that gates its starts -- is untouched.
    """
    if _TEAM_LEAD_AGENT_NAME not in names:
        return None
    # The PROJECT shadow first, because it decides which file the question is even
    # about: when the project declares this name, the global spec below is one nothing
    # will execute, and passing it would be answering about the wrong bytes.
    shadowed = _project_shadow_refusal(project_dir, names)
    if shadowed:
        return shadowed
    path = agent_mod.kiro_agents_dir_path() / _TEAM_LEAD_AGENT_FILENAME
    try:
        existing = _existing_spec_for_attribution(path)
    except conductor_agents._SpecUnusable:
        # The bytes are there and cannot be read deterministically. Fail CLOSED:
        # admitting on a failed read is admitting a spec nobody examined, and the
        # read that failed is the only thing that could have cleared it.
        return _HAND_EDITED_REFUSAL
    if existing is None:
        return None
    if _attribution_reason(existing) is not None or not isinstance(existing, dict):
        # The second clause is a FAIL-CLOSED floor, not a type ceremony. Attribution
        # answers with a reason for an unattributable file, so reaching here with
        # anything but a parsed object would mean that contract changed -- and the
        # safe reading of "I cannot tell what this is" is the same refusal.
        return _HAND_EDITED_REFUSAL
    # CONFIRMED as our own write, and still not necessarily safe to run: the bytes
    # match what this installer last wrote, which says nothing about whether the
    # ceiling has moved since. Asked STATELESSLY rather than from a marker the
    # rebuild sets, because a marker is only as good as the rebuild reaching it --
    # and the rebuild writes the default spec first, so one failure there kills the
    # pass before this installer runs at all and leaves any such marker clear.
    return _ceiling_drift_reason(existing)


def _managed_team_lead_install_lands(path: Path) -> bool:
    """Would the managed install WRITE at *path* -- does this installer own that file?

    The fail-closed companion to :func:`_attribution_reason`, for the callers that must
    never read a crew's file as machine-maintained: fork governance deciding it may
    SKIP a spec because an owned writer re-filters its grants every rebuild, the owned
    ORIGIN test behind a plumbing refresh, and the capability-parent classification.

    Those callers ask "is this confirmed OURS?", so every uncertain answer is False.
    An ABSENT file is confirmed as nothing -- the installer does write there, but no
    fork's grants are maintained by a file that does not exist -- and a read that
    failed transiently is not a confirmation either. The installer keeps its own
    three-way answer, where absent WRITES and a transient failure HOLDS; collapsing
    both into this predicate is what would turn a hold into a skip.
    """
    try:
        existing = _existing_spec_for_attribution(path)
        if existing is None:
            return False
        return _attribution_reason(existing) is None
    except conductor_agents._SpecUnusable:
        return False


#: The servers this spec grants VERB BY VERB. A bare ``@server`` entry for one of
#: them is dropped from the assembled list, and that subtraction is what makes the
#: narrowing real rather than cosmetic.
_VERB_GRANTED_SERVERS: frozenset[str] = frozenset(
    {"@kirocrew-core", "@kirocrew-dashboard", "@kirocrew-work"}
)


def _narrow_whole_server_grants(granted: list[str]) -> list[str]:
    """Drop a bare ``@server`` grant for a server granted verb by verb.

    The template auto-approves ``@kirocrew-core`` as a WHOLE SERVER, and this
    installer appends to the template's list rather than replacing it, so without
    this pass the named verbs would sit beside a wildcard that already covers every
    verb on that server -- ``task_run``, ``workflow_run``, ``spawn_run`` and the rest
    included. The per-verb list would then describe a narrowing the emitted spec
    does not have.

    Worse than cosmetic, and the conductor spec's own comment names the mechanism:
    both backends resolve a whole-server reference before the per-tool one, so a
    bare entry "would have survived the filter on the KAS backend" -- a verb the
    governance ceiling strips from the named list is still reached through the
    wildcard, and the ceiling's decision is silently undone.

    Subtractive and ordered LAST of the grant passes, exactly as the worker
    installer subtracts scheduling from its own assembled list, so it applies to
    every source at once: the template's entry, the appended tuples, and anything a
    later tuple adds. Only an EXACT ``@server`` match is dropped -- a per-verb entry
    on the same server is what this keeps.
    """
    return [ref for ref in granted if ref not in _VERB_GRANTED_SERVERS]


class _Unattributable:
    """A file is PRESENT at the spec's name and cannot be attributed to anyone.

    Its own type rather than ``None``, because the two answers lead to opposite
    actions and sharing one value is what let a write through once already:
    ``None`` means there is nothing at this path, which is the only state in which
    this installer writes without a name to check, while this means there IS
    something and its bytes do not say whose it is.
    """


#: The one instance; compared by identity.
_UNATTRIBUTABLE = _Unattributable()


def _readable_object_at(path: Path) -> dict[str, Any] | _Unattributable | None:
    """The JSON object at *path* for ATTRIBUTION only.

    Asked after the hardened reader has already answered ``replace=True``, and it
    answers a narrower question than that reader does: not "is this a spec this
    release can use", but "is there a readable JSON object here at all". Those
    differ in exactly the case that matters -- a document that parses and is then
    rejected on grant shape is still somebody's file, and attribution has to see
    it.

    Read through the same hardened reader (``read_agent_spec_strict``), so the
    no-follow path fence, the size cap and the sensitive-target refusal all still
    apply; what this drops is only the shape verdict layered on top of it.

    A failure here is sorted into the reader's OWN two classes rather than
    collapsed, because the two call for opposite answers:

    * a plain ``OSError`` is a read that MAY succeed next time -- a permission, an
      I/O error, a refused open. Collapsing it to "nothing to attribute" is a hole
      the size of the whole guard: a file that parsed a moment ago, and so is
      somebody's, would be attributed against nothing and overwritten because one
      read in the gap failed. It is raised as the reader's transient class, which
      the caller turns into a HOLD, and a hold costs one rebuild.
    * a ``ValueError`` is the content not being a spec, and a re-read will not
      change that. :data:`_UNATTRIBUTABLE`, so the caller DECLINES: a file with one
      missing brace is the ordinary result of a hand edit, and its bytes are the
      operator's whether or not they parse.

    Declining a file that cannot be made into a spec does not strand this agent,
    which is what makes the trade here different from ``worker_agent``'s. That spec
    is DERIVED from the default and re-checked before every worker session, so
    refusing it would break dispatch with no way out. This one is optional and
    selected by name, exactly like the dashboard-author spec, and it follows that
    spec's documented contract: the file is left untouched, no backup is written,
    and the agent stays unselectable under this name until the operator removes or
    renames it.

    An ABSENT file is the one ``None``, and the only state in which the caller
    writes without a name to check. ``FileNotFoundError`` is an ``OSError`` so it is
    caught first, and a name that still exists behind that error is a dangling link
    -- present, with nothing to attribute.
    """
    try:
        data = agent_discovery.read_agent_spec_strict(
            path, operation="team_lead_spec_attribution", source="unknown"
        )
    except FileNotFoundError:
        return _UNATTRIBUTABLE if os.path.lexists(path) else None
    except OSError as exc:
        raise conductor_agents._SpecUnusable(
            f"its bytes could not be read for attribution ({exc}); they may be your" " intact spec",
            replace=False,
        ) from exc
    except ValueError:
        return _UNATTRIBUTABLE
    return data if isinstance(data, dict) else _UNATTRIBUTABLE


def _existing_spec_for_attribution(path: Path) -> dict[str, Any] | _Unattributable | None:
    """The spec on disk at *path* to attribute before writing.

    One reader for the installer, so the transient class has ONE answer however it
    is reached. ``_spec_to_replace`` decides whether this release can USE the file;
    when it answers "no, and the bytes will not improve" (``replace=True``), the
    narrower attribution read runs, because that verdict includes a document that
    parsed. Either read raising the transient class propagates it, and the caller
    holds.

    ``None`` means the path is ABSENT, which is the only state in which this
    installer writes without a name to check. A file that is present and cannot be
    attributed answers :data:`_UNATTRIBUTABLE` instead, and the caller declines it.
    """
    try:
        return conductor_agents._spec_to_replace(path)
    except conductor_agents._SpecUnusable as exc:
        if not exc.replace:
            raise
        return _readable_object_at(path)


def _install_team_lead_agent(*, clean: bool = False) -> InstallOutcome:
    """Generate and install the kirocrew-team-lead agent config.

    The template PLUS session control PLUS the work ledger, and the first of those
    three is the one that matters: ``fs_write``, ``code``, ``grep``, ``glob`` and
    shell arrive because this installer never overwrites ``config["tools"]`` with a
    literal, the way each conductor installer does. So the charter's do-it-yourself
    half is true against the emitted spec and not only against its prose, and the
    gap this agent exists to close -- a conductor handed a one-file fix cannot make
    it, and a worker handed a goal cannot split it -- is closed in the tool list.

    ``fs_write`` needs no grant to be mounted and gets none: it is in the template's
    ``tools`` and absent from its ``allowedTools``, which is the default agent's own
    posture and the one a reviewer can check against. ``execute_bash`` is the same,
    for the reason the conductor installers record: ``allowedTools`` is name-scoped
    with no argument matching, so trusting the bundled acceptance evaluator cannot be
    told apart from trusting arbitrary shell. The charter's answer is to batch those
    calls, not to widen the grant.

    ``clean`` is accepted for one reason: every installer in this package takes it,
    and the rebuild passes it to each. It changes nothing here, because nothing on
    the previous file is carried forward -- the spec is a pure function of the
    template, the charter and the tuples above, so there is no user entry to drop.
    The conductor specs merge the user's own grants under a provenance table and
    need the flag; this one has no such merge to reset.

    Returns an :class:`InstallOutcome`, and the THREE values are the point: a
    rebuild needs to tell a list it re-derived from one it left alone, and among
    the ones it left alone it needs to tell a hold that may clear from a decline
    that never will.

    ``HELD`` is work on THIS INSTALLER'S OWN spec that did not happen for a reason
    the next pass may not meet: a read that failed in the reader's transient class,
    or a write that raised. Retrying is exactly right -- the same read or the same
    write a moment later is what clears an I/O error.

    ``DECLINED`` is a spec this installer did not write
    (:func:`_foreign_team_lead_spec_reason`) and will not rewrite, and does not
    touch at all. Retrying is exactly wrong. The file will still be somebody
    else's on the next pass and on every pass after it, so counting it as held
    puts the install in a retry set it can never leave -- every hourly sweep
    re-running a full rebuild and logging a spec that "could not be written", for
    a decision this installer made on purpose. What keeps the grants on such a
    file from mattering is not this value: the session start refuses to run it
    (:func:`spec_start_refusal`).
    """
    config = agent_mod.build_agent_config()
    # Applied to the INHERITED material, before this installer adds its own two servers
    # and their grants: an ``opt_in`` set is assigned per agent, and the operator
    # mounting one on their personal ``agent.json`` is not assigning it to every spec
    # built from that template. All three surfaces go at once (the entry, the ``@server``
    # ref in ``tools``, any grant in ``allowedTools``), because a server reaches a
    # session through any of them.
    # Imported from the owner that DEFINES it rather than read off the facade, so the
    # cross-owner dependency is visible; function-local to avoid an import cycle at load.
    from kiro_crew.agent_materialization import worker_agent

    unassigned = worker_agent._drop_servers(config, _team_lead_unassignable_servers())
    if unassigned:
        # Same footing as every other permission decision here: a set the operator's own
        # agent holds and this one does not is something they have to be able to find.
        try:
            agent_mod.sel().log_api_access(
                caller="system",
                operation="mcp_auto_approve_withheld",
                outcome="ok",
                source="_install_team_lead_agent",
                resources=(
                    f"{', '.join(unassigned)} not carried onto the team lead "
                    "(an opt-in set is assigned per agent, not inherited)"
                ),
            )
        except Exception:  # noqa: BLE001 — the audit must not break the install
            agent_mod.logger.debug("SEL audit unavailable for unassigned server", exc_info=True)
    config["name"] = "kirocrew-team-lead"
    config["description"] = (
        "Owns a goal end to end and runs a team on it: registers the work, "
        "splits it into ledger items, does the small focused ones itself with "
        "the full default toolset, dispatches a session for every other one, "
        "patrols that fleet event-driven, and decides what its children cannot."
    )
    config["prompt"] = agent_mod._TEAM_LEAD_SYSTEM_PROMPT

    tools = [ref for ref in (config.get("tools") or []) if isinstance(ref, str)]
    for server in ("@kirocrew-dashboard", "@kirocrew-work"):
        if server not in tools:
            tools.append(server)
    config["tools"] = tools

    granted = [ref for ref in (config.get("allowedTools") or []) if isinstance(ref, str)]
    # A PATTERN is not an approval of anything the ceiling can name. The ceiling judges a
    # ref by its exact name -- a builtin by its own, an ``@server`` ref by its server --
    # and a pattern is in no table and names no server, so it passes as an unmapped name
    # while kiro-cli expands it over every builtin and every verb of every mounted server.
    # This installer mounts the very server such a pattern expands onto, so a kept
    # ``session_*`` would auto-approve a verb the ceiling withholds by name, on the one
    # path that never reaches the approval gate. Dropped BEFORE the ceiling pass, so it is
    # never offered to it as though it approved something in particular. Only
    # ``allowedTools`` is filtered: ``tools`` mounts, and mounting is not auto-approving.
    wildcards = [ref for ref in granted if conductor_agents._is_wildcard(ref)]
    if wildcards:
        granted = [ref for ref in granted if ref not in wildcards]
        # WARNING with the one remedy, because this drops something the operator wrote and
        # a silent drop leaves them no way to tell a withheld pattern from a typo.
        agent_mod.logger.warning(
            "%s: allowedTools entries %s are patterns, which cannot be judged against the "
            "governance ceiling and are not carried onto this spec -- name tools exactly",
            _TEAM_LEAD_AGENT_FILENAME,
            ", ".join(wildcards),
        )
        try:
            agent_mod.sel().log_api_access(
                caller="system",
                operation="mcp_auto_approve_withheld",
                outcome="ok",
                source="_install_team_lead_agent",
                resources=(
                    f"{', '.join(wildcards)} not carried onto the team lead "
                    "(a pattern cannot be judged against the ceiling); name tools exactly"
                ),
            )
        except Exception:  # noqa: BLE001 — the audit must not break the install
            agent_mod.logger.debug("SEL audit unavailable for withheld pattern", exc_info=True)
    for tuple_ in _TEAM_LEAD_SHIPPED_GRANTS:
        granted.extend(ref for ref in tuple_ if ref not in granted)
    config["allowedTools"] = _narrow_whole_server_grants(granted)
    # Audited HERE rather than inside the helper, which stays a pure function of its
    # list so the enumeration test can call it directly. Subtracting a whole-server
    # grant is a permission decision like every other one in this installer, and it is
    # the one that fires on an ORDINARY rebuild: the template's bare ``@kirocrew-core``
    # is removed from the assembled list every time. The ceiling pass that follows sees
    # only the survivors, so without an event here the subtraction leaves no trace in
    # the feed an operator reads to find out why a grant they can see in the template is
    # not on the spec.
    narrowed_away = [ref for ref in granted if ref not in config["allowedTools"]]
    if narrowed_away:
        try:
            agent_mod.sel().log_api_access(
                caller="system",
                operation="mcp_auto_approve_withheld",
                outcome="ok",
                source="_install_team_lead_agent",
                resources=(
                    f"{', '.join(narrowed_away)} dropped from the team lead's auto-approve "
                    "list (this spec grants those servers verb by verb, and a whole-server "
                    "entry resolves first); their tools go through the approval gate"
                ),
            )
        except Exception:  # noqa: BLE001 — the audit must not break the install
            agent_mod.logger.debug(
                "SEL audit unavailable for narrowed whole-server grant", exc_info=True
            )
    config["mcpServers"] = _team_lead_mcp_servers(config)

    # The spec's server map is the WHOLE server map, pinned here rather than inherited.
    # ``includeMcpJson`` tells kiro-cli to merge the global ``mcp.json`` on top of this
    # spec, and the operator's own ``agent.json`` can set it: ``build_agent_config``
    # deep-merges that file, so a ``true`` there would ride through. Every filter in this
    # installer reads ``config["mcpServers"]`` and nothing else, so a server arriving from
    # the global file is governed by none of them -- its ``autoApprove`` is never seen by
    # the ceiling strip above, and kiro-cli approves such a tool locally with no
    # permission request. Pinned False so the filtered map is the only one, and the
    # retired spelling is dropped with it so a stale key cannot re-open the merge.
    config.pop("useLegacyMcpJson", None)
    config["includeMcpJson"] = False

    # ONE ceiling pass, over the whole assembled list and after every append, which
    # is what keeps the ceiling authoritative over this installer rather than beside
    # it: ``allowedTools`` is the one path that never reaches the PreToolUse gate, so
    # a host governing a verb must get a prompt here instead of a bypass.
    auto_approve._apply_allowed_tools_ceiling(config, source="_install_team_lead_agent")
    # The SECOND channel a call skips the PreToolUse gate through, and the pass above
    # cannot see it: ``autoApprove`` on an ``mcpServers`` entry is approved locally by
    # kiro-cli, which emits no permission request at all. This map is additive, so an
    # entry the operator wrote in their own ``agent.json`` arrives with whatever
    # ``autoApprove`` they gave it -- a grant no ceiling has read. Ordered after the
    # grant filter for the same reason that one is ordered last: both channels are
    # filtered once, over everything assembled, so the ceiling stays authoritative over
    # this installer rather than beside it.
    config["mcpServers"] = auto_approve._strip_ungoverned_auto_approve(config["mcpServers"])
    # Derived from the FILTERED list rather than restated, so a ceiling that strips a
    # grant strips its KAS rule with it. The shared writer version-gates the field.
    auto_approve._write_derived_permissions(
        config, config["allowedTools"], _TEAM_LEAD_AGENT_FILENAME
    )

    agents_dir = agent_mod.kiro_agents_dir_path()
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / _TEAM_LEAD_AGENT_FILENAME

    # ATTRIBUTION FIRST, then grant-shape validation. The order is the invariant:
    # never write to this path when its existing content was not written by this
    # installer, whether or not that content parses as a usable spec.
    #
    # Reusing the conductor installers' hardened reader rather than a plain load,
    # because it keeps the failure CLASS and a read that failed TRANSIENTLY must
    # leave the file alone -- holding it costs one rebuild, and replacing it on a
    # bad read costs the file.
    #
    # What that reader cannot decide for this caller is the other class. It answers
    # ``replace=True`` for several different things, and one of them is a document
    # it PARSED as an object and then rejected on grant SHAPE: an ``allowedTools``
    # that is not a list. Treating that as "no spec here" would skip attribution
    # altogether, so a hand-authored agent with one typo in that field -- the most
    # ordinary way for it to be malformed -- would be overwritten whole, which is
    # precisely the harm the attribution exists to prevent. So every
    # ``replace=True`` goes through one more read whose only job is attribution,
    # and a read that fails TRANSIENTLY in that second step holds rather than
    # writes: a file that parsed a moment ago is somebody's, so one I/O error in
    # the gap must not become permission to replace it.
    # ONE CRITICAL SECTION from the existing-file read through the finalize, under
    # ``agents_spec_lock`` -- the template-spec writer lock every other
    # read-modify-writer of this directory holds (the worker installer, the
    # dashboard-author installer, the reset path, the fork refresh, the dashboard
    # PATCH). This installer is a read-modify-writer like all of them and was the
    # exception.
    #
    # The whole sequence, not the read alone, and splitting it is the bug rather than
    # a smaller version of it: attributing under the lock and then writing outside it
    # is the same race with a narrower window -- another writer lands between the
    # attribution and the replace, and this pass overwrites a file it never examined.
    # The declined strip is inside for the same reason, since it reads those bytes and
    # writes them back, and so is the two-phase digest commit, whose whole invariant is
    # that the recorded digest and the bytes on disk cannot disagree.
    #
    # ``agents_spec_lock`` ALONE, and the sibling worker installer is the contrast that
    # shows why. It holds ``bridges._mcp_lock`` as well because it MIRRORS
    # ``kirocrew.json`` onto its spec -- two files, each with its own independent
    # writers, so a deregistration landing after its mirror snapshot would leave a
    # removed server's grant auto-approved. This spec deliberately does not mirror the
    # default: it is built from the in-memory template, and the only file it touches is
    # its own. The dashboard-author installer is the closer sibling and takes this lock
    # by itself for exactly that reason.
    #
    # Nothing inside takes this lock again, which matters because the lock is a
    # cross-process ``flock`` on a sidecar path rather than a re-entrant one: a second
    # acquire from the same process on a new descriptor would wait on itself.
    # ``agent_state``, ``auto_approve``, ``conductor_agents``, ``managed_mcp`` and
    # ``owned_provenance`` hold no reference to it, and the one ``worker_agent`` function
    # reached from here is ``_drop_servers``, a pure in-place filter that runs during
    # assembly above rather than in here.
    with agent_mod.agents_spec_lock(agents_dir):
        reason: str | None
        try:
            existing = _existing_spec_for_attribution(path)
            # Inside the same guard as the spec read, because attribution has a SECOND
            # read that can fail transiently -- the lineage sidecar -- and both failures
            # mean the same thing: the check did not run, so this pass must not write.
            reason = _attribution_reason(existing)
        except conductor_agents._SpecUnusable as exc:
            agent_mod.logger.warning(
                "%s: the spec on disk could not be read (%s); left in place, so the "
                "agent is installed from the next rebuild rather than from this one",
                _TEAM_LEAD_AGENT_FILENAME,
                exc.cause,
            )
            return InstallOutcome.HELD
        if reason is not None:
            # ERROR, and the level is the point: this installer's caller swallows at debug,
            # so without this line the one event an operator needs -- "your own agent is
            # occupying this filename, and the shipped one is therefore absent" -- would be
            # invisible at any ordinary level. Declining is not silent and not destructive.
            agent_mod.logger.error(
                "Refusing to overwrite %s: %s, so it was not written by this installer. "
                "The shipped team-lead agent is NOT installed while that file is there; "
                "move or rename it to let the agent be installed.",
                path,
                reason,
            )
            # Nothing is written and nothing is edited. The file is left exactly as it
            # is, bytes for bytes, and the grants on it are not narrowed, filtered or
            # removed -- a spec this installer cannot attribute is not one it may
            # sanitize either. What governs those grants is the REFUSAL at session
            # start (:func:`spec_start_refusal`): a spec nobody can vouch for
            # does not run, which is a decision that needs no judgement about any
            # individual grant and leaves the operator's file recoverable by renaming
            # it. Rewriting it to make it safe was the thing that kept being wrong --
            # most sharply on ``permissions``, where ``_EFFECTS`` admits ``deny`` as
            # well as ``allow``, so emptying the block could DELETE a restriction the
            # operator wrote rather than only a grant.
            return InstallOutcome.DECLINED

        # Two-phase digest commit, so the bytes on disk and the recorded ownership never
        # disagree permanently: record the NEW bytes' digest as PENDING, replace the file,
        # then promote it. At every instant between those steps the file reproduces one of
        # the two recorded values -- the old finalized digest before the replace, the
        # pending one after -- so a crash in any window leaves the spec still confirmable
        # and the next rebuild refreshes it instead of reading its own file as foreign and
        # freezing it. ``current`` is the digest of the bytes just confirmed above, which
        # is what covers a replace that fails after the pending record lands.
        new_digest = agent_state.spec_digest(config)
        current_digest = agent_state.spec_digest(existing) if isinstance(existing, dict) else None
        agent_state.begin_managed_write(_TEAM_LEAD_AGENT_NAME, new_digest, current=current_digest)
        agent_mod._atomic_json_write(path, config)
        agent_state.finalize_managed_write(_TEAM_LEAD_AGENT_NAME)
        agent_mod.logger.info("Installed team-lead agent config: %s", path)
        return InstallOutcome.WRITTEN
