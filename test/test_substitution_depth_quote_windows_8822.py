"""Tests for the quote-aware argv windows in the self-protection floor.

The self-protection argv windows bound the current command with
``_substitution_depth_delta``, which counts parens on DE-QUOTED tokens, so a
QUOTED ``)`` nested in a command substitution reads as a real closer and the
window ends early at a separator that is still inside the substitution.  For
``_is_credential_mint`` that drops the ``token`` verb from a command bash
actually mints (the correctness defect fixed here).  The sibling operand/kill
windows are pinned too, with the verdict for each.

Program and verb literals are assembled from pieces so the mint shape only
exists at runtime, never as a literal substring of this source or of any
command line.
"""

import pytest

from kiro_crew.security import is_denied
from kiro_crew.security import shell_normalizer as sn

# Assembled from pieces so the credential-mint shape exists only at runtime,
# never as a literal substring of this file or of any command line.
_N = "kiro" + "crew"  # product CLI program name
_MOD = "kiro" + "_crew"  # product import/module name
_T = "to" + "ken"  # the credential-minting verb


class TestRawSimpleCommands:
    """The quote-aware neutralizer collapses each substitution span to one inert
    placeholder word and copies everything else verbatim."""

    def _ph(self):
        return sn._SUBSTITUTION_PLACEHOLDER

    def test_command_substitution_span_collapses_to_placeholder(self):
        # The decoy shape: the quoted ')' and nested ';' live inside the span, so
        # the whole span -- including them -- becomes one inert word.
        assert sn._neutralize_substitution_spans("foo $(true ')' ; true) baz") == (
            f"foo {self._ph()} baz"
        )

    def test_backtick_span_collapses(self):
        assert sn._neutralize_substitution_spans("foo `printf ';'` baz") == (
            f"foo {self._ph()} baz"
        )

    def test_process_substitution_span_collapses(self):
        assert sn._neutralize_substitution_spans("cat <(echo ';') x") == (f"cat {self._ph()} x")

    def test_nested_spans_collapse_to_one_placeholder(self):
        assert sn._neutralize_substitution_spans("foo $(a ; $(b ; c) ; d) baz") == (
            f"foo {self._ph()} baz"
        )

    def test_quoted_dollar_paren_is_literal_not_a_span(self):
        # Inside single quotes ``$(`` is data, so nothing is neutralized.
        src = "echo '$(true)' x"
        assert sn._neutralize_substitution_spans(src) == src

    def test_text_without_substitutions_is_unchanged(self):
        src = 'kirocrew restart; echo "a|b" \\$kept'
        assert sn._neutralize_substitution_spans(src) == src

    def test_surrounding_quotes_and_escapes_preserved_verbatim(self):
        # A STANDALONE span (spaces around it) collapses to the placeholder while
        # the surrounding text is copied byte for byte.
        assert sn._neutralize_substitution_spans("x a $(b) c y") == (f"x a {self._ph()} c y")
        # A span inside a quoted word is part of that word, so it collapses to
        # NOTHING (its empty-output case fuses the word), surroundings verbatim.
        assert sn._neutralize_substitution_spans('x "a$(b)c" y') == 'x "ac" y'


class TestCredentialMintQuoteBlindWindow:
    """Site 1 (the correctness defect this PR fixes).

    The de-quoted mint-verb window closes at a separator NESTED in a command
    substitution once a QUOTED ``)`` has driven its depth counter non-positive,
    so the verb that follows the substitution is never seen -- while bash
    expands the substitution to nothing and runs the mint.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            # The core case: $(true ')' ; true) expands to NOTHING (true
            # ignores its ')' arg), so bash runs ``<cli> <verb>`` -- a real mint
            # -- while the quoted ')' + nested ';' close the de-quoted window early.
            "{n} $(true ')' ; true) {t}",
            # The backtick spelling of the same decoy.
            "{n} `true ')' ; true` {t}",
            # A redirection interposed between the decoy and the verb is still
            # the same simple command: bash runs ``<cli> <verb>`` with stderr
            # redirected.
            "{n} $(true ')' ; true) 2>/tmp/out {t}",
            # The module form shares the identical window, so it must be caught too.
            "python -m {mod} $(true ')' ; true) {t}",
            # Nested one wrapper deep: the floor descends shell payloads.
            "bash -c \"{n} $(true ')' ; true) {t}\"",
            # A quoted ')' with a nested '|' is the same mechanism (| ends the
            # de-quoted window just as ';' does).
            "{n} $(true ')' | true) {t}",
        ],
    )
    def test_decoyed_mint_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N, mod=_MOD, t=_T))

    @pytest.mark.parametrize(
        "cmd",
        [
            # The undecorated mint the floor has always caught.
            "{n} {t}",
            # Flags interposed, no decoy -- the plain window already catches it.
            "{n} -v {t}",
        ],
    )
    def test_control_mints_remain_denied(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    @pytest.mark.parametrize(
        "cmd",
        [
            # The verb is an ARGUMENT of a data-consumer command, not the CLI's
            # own argv -- bash prints it, nothing mints.  The fix must not widen
            # the window into denying these.
            "echo {n} $(true ')' ; true) {t}",
            # The name is a path and the verb is a different program's argument.
            "cd /workplace/user/{n}-wt && pytest test/test_{t}_auth.py",
            # The verb lands in grep's argv across a pipe, not the CLI's.
            "{n} doctor | grep {t}",
            # A redirection to a FILE named like the verb is not the verb word.
            "{n} >{t}",
            "{n} 2>{t}.log doctor",
            # The decoy belongs to a DIFFERENT, non-self command on the line.
            "{n} doctor; echo $(true ')' ; true) {t}",
        ],
    )
    def test_scoped_non_detections_stay_allowed(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T)) is None


class TestSelfKillByNameQuoteBlindWindow:
    """Site 3 -- the pkill/killall BY-NAME window, also fixed here.

    The issue judged this low-reachability assuming a decoy that OUTPUTS text
    (a second pattern pkill rejects).  But an EMPTY-output generator makes
    ``pkill -f $(true ')' ; true) <name>`` expand to ``pkill -f <name>`` -- a
    real self-kill -- while the quoted ')' closes the de-quoted window at the
    nested ';' before the name.  So it is the same early-close defect, caught the
    same way as the mint window.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            "pkill -f $(true ')' ; true) {n}",
            "killall $(true ')' ; true) {n}",
            "pkill -f `true ')' ; true` {n}",
            "bash -c \"pkill -f $(true ')' ; true) {n}\"",
        ],
    )
    def test_decoyed_self_kill_by_name_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N))

    @pytest.mark.parametrize(
        "cmd",
        [
            # The undecorated control the floor has always caught.
            "pkill -f {n}",
            "pkill -f '[k]irocrew'",
        ],
    )
    def test_control_self_kills_remain_denied(self, cmd):
        assert is_denied(cmd.format(n=_N))

    @pytest.mark.parametrize(
        "cmd",
        [
            # pkill named as a data-consumer's argument prints words; nothing dies.
            "echo pkill $(true ')' ; true) {n}",
            # the pkill and the name are in DIFFERENT commands.
            "pkill other; echo {n}",
        ],
    )
    def test_scoped_non_detections_stay_allowed(self, cmd):
        assert is_denied(cmd.format(n=_N)) is None


class TestSiblingSiteNotTheSameDefect:
    """Site 2 -- the self-SUBCOMMAND operand window -- LOOKS like a copy of the
    quote-blind defect but is not independently fixable by it, so it is PINNED.

    The subcommand floor is PRECISE, not conservative: it fires only when the
    destructive subcommand is the LEADING operand.  When a substitution precedes
    the subcommand, bash's real leading operand is the substitution's OUTPUT,
    which a static scan cannot evaluate -- so the lead is unprovable and the floor
    must allow, exactly as it already allows the plain ``<cli> $(cmd) restart``.
    A quote-aware window yields the SAME allow (operands ``['<span>', 'restart']``,
    lead is the opaque span), so the quoted-paren window closing early flips no
    deny into an allow here.

    KNOWN RESIDUAL (pinned below): an EMPTY-output generator before the
    subcommand (``<cli> $(true ')' ; true) restart`` -> bash ``<cli> restart``)
    is a real self-restart the floor misses -- but it is the pre-existing
    static-analysis limitation shared with every ``<cli> $(cmd) restart``, NOT the
    quote-blind-window defect, and closing it requires making the subcommand floor
    conservative (a policy change tracked as a follow-up), not a quote-aware read.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            # The decoyed form and the plain-substitution form share one verdict:
            # the lead is an unevaluatable substitution either way.
            "{n} $(true ')' ; true) restart",
            "{n} $(true) restart",
        ],
    )
    def test_subcommand_behind_substitution_is_allowed_either_way(self, cmd):
        assert is_denied(cmd.format(n=_N)) is None

    def test_plain_restart_with_no_hidden_lead_still_denied(self):
        assert is_denied("{n} restart".format(n=_N))


class TestUnionExercisesTheRawPath:
    """Prove the decoyed denials come from the NEUTRALIZED pass, not the plain
    de-quoted window: with neutralization disabled (``_allow_neutralize=False``)
    the window truncates on the decoy and misses it, while the full call catches
    it.  This is the mutation check -- disabling the pass is exactly what the fix
    would revert to."""

    def test_disabling_neutralize_misses_the_decoyed_mint(self):
        from kiro_crew.security import argv_floor as af

        cmd = "{n} $(true ')' ; true) {t}".format(n=_N, t=_T).lower()
        # The quote-blind window truncates at the nested ';', so disabling the
        # neutralized pass misses it.
        assert af._is_credential_mint(cmd, _allow_neutralize=False) is False
        # The full call's neutralized pass restores the detection.
        assert af._is_credential_mint(cmd) is True
        # The undecoyed control is caught by the window directly.
        assert af._is_credential_mint("{n} {t}".format(n=_N, t=_T)) is True

    def test_disabling_neutralize_misses_the_decoyed_self_kill(self):
        from kiro_crew.security import argv_floor as af

        cmd = "pkill -f $(true ')' ; true) {n}".format(n=_N).lower()
        assert af._is_self_kill(cmd, _allow_neutralize=False) is False
        assert af._is_self_kill(cmd) is True
        assert af._is_self_kill("pkill -f {n}".format(n=_N)) is True


class TestRawWindowResolvesShellWordsNotRawBytes:
    """The raw window must match what bash PASSES, not the raw spelling: a
    backslash escape is decoded, a redirect target is skipped, and a
    substitution span's interior is opaque.  Each case here is one bash runs the
    way the assertion says, confirmed on this checkout.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            # A backslash before an ordinary letter is removed by bash, so
            # ``kiro\\crew`` is the program/pattern ``kirocrew``.  The de-quoted
            # window truncates on the decoy, so only the raw window sees these --
            # it must decode the backslash before matching the name/verb.
            "pkill -f $(true ')' ; true) kiro\\crew",
            "{n} $(true ')' ; true) to\\ken",
            "{n} $(true ')' ; true) kiro\\crew to\\ken",
        ],
    )
    def test_backslash_decoded_decoy_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N))

    @pytest.mark.parametrize(
        "benign",
        [
            # A decoy before a BENIGN subcommand stays allowed: bash runs
            # ``<cli> doctor``, no mint verb, no destructive subcommand.
            "{n} $(true ')' ; true) doctor",
            "{n} $(true ')' ; true) --help",
        ],
    )
    def test_decoy_before_benign_subcommand_stays_allowed(self, benign):
        assert is_denied(benign.format(n=_N)) is None

    @pytest.mark.parametrize(
        "tail",
        [
            "doctor > token",
            "doctor 2> token",
        ],
    )
    def test_redirect_decoy_matches_the_decoy_free_verdict(self, tail):
        # ``> token`` as a SEPARATE word is a redirect to a file named ``token``.
        # Whatever the core decides for the decoy-FREE command, the neutralized
        # decoy form must decide identically -- the decoy expands to nothing, so
        # it cannot change the verdict.  (The core's own treatment of a file
        # literally named ``token`` is pre-existing and not this PR's to change;
        # this pins that the fix introduces no NEW divergence.)
        decoyed = "{n} $(true ')' ; true) {tail}".format(n=_N, tail=tail)
        plain = "{n} {tail}".format(n=_N, tail=tail)
        assert (is_denied(decoyed) is None) == (is_denied(plain) is None)

    @pytest.mark.parametrize(
        "cmd",
        [
            # The name appears only INSIDE a substitution whose OUTPUT is an
            # ordinary word, so bash passes ``pkill -f ordinary`` -- not a
            # self-kill.  The neutralized pass must treat the span as opaque, not
            # literal-match its interior source.  (A ``;`` before the name keeps
            # the public regex tier -- whose ``[^;&#>]*`` class stops at it --
            # out of the verdict, so this exercises the structural floor alone.)
            "pkill -f $(true ')' ; true {n}; printf ordinary)",
        ],
    )
    def test_opaque_substitution_span_interior_is_not_the_target(self, cmd):
        assert is_denied(cmd.format(n=_N)) is None

    def test_empty_substitution_glue_still_resolves_the_name(self):
        # An empty substitution vanishes, so ``kiro$()crew`` IS the name
        # ``kirocrew`` -- the core's own empty-substitution resolution catches it
        # (the neutralized pass need not, and must not swallow it).
        assert is_denied("pkill -f kiro$()crew")


class TestNeutralizeHandlesNestingEmptyAndComments:
    """The neutralized pass must reach nested payloads, collapse an EMPTY
    substitution to nothing, and skip a commented opener -- the three ways the
    first neutralize cut missed a command bash still runs."""

    @pytest.mark.parametrize(
        "cmd",
        [
            # F1 -- the decoy is in a NESTED bash -c payload (quote-protected at
            # the outer level), so the outer-only neutralize missed it; each
            # payload source must be neutralized.
            "bash -c 'pkill -f $(true \")\" ; true) {n}'",
            "bash -c \"{n} $(true ')' ; true) {t}\"",
            "sh -c 'pkill -f $(true \")\" ; true) {n}'",
        ],
    )
    def test_nested_payload_decoy_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    @pytest.mark.parametrize(
        "cmd",
        [
            # F2 -- the name is AFTER the decoy (so the de-quoted window truncates)
            # AND glued to an empty ``$()`` (so a placeholder would hide it); the
            # empty substitution must collapse to nothing: ``kiro$()crew`` ->
            # ``kirocrew``.
            "pkill -f $(true ')' ; true) kiro$()crew",
            "{n} $(true ')' ; true) to$()ken",
        ],
    )
    def test_empty_substitution_after_decoy_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N))

    def test_empty_substitution_collapses_to_nothing(self):
        # Direct check on the neutralizer: an empty span rejoins its neighbours.
        assert sn._neutralize_substitution_spans("kiro$()crew") == "kirocrew"
        assert sn._neutralize_substitution_spans("to`  `ken") == "token"
        # A NON-empty span GLUED to a word also collapses to nothing (its
        # empty-output case fuses the word), while a STANDALONE one is the
        # inert placeholder.
        assert sn._neutralize_substitution_spans("kiro$(x)crew") == "kirocrew"
        assert sn._neutralize_substitution_spans("a $(x) b") == (
            f"a {sn._SUBSTITUTION_PLACEHOLDER} b"
        )

    @pytest.mark.parametrize(
        "cmd",
        [
            # F3 -- a commented ``$(`` with no close must not be read as an
            # unterminated span that swallows the real command after the newline.
            "# $(\npkill -f $(true ')' ; true) {n}",
            "# $(\n{n} $(true ')' ; true) {t}",
        ],
    )
    def test_commented_opener_does_not_swallow_the_next_line(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_commented_opener_is_skipped_by_the_neutralizer(self):
        # The comment is copied verbatim and the real command on the next line
        # survives (its span neutralized), rather than the whole tail vanishing.
        out = sn._neutralize_substitution_spans("# $(\npkill -f $(x) foo")
        assert out.splitlines()[0] == "# $("
        assert "pkill -f" in out and sn._SUBSTITUTION_PLACEHOLDER in out


class TestNeutralizeFoldsContinuationsAndTracksWordBoundaries:
    """The neutralizer must fold a line-continuation before scanning openers,
    and treat ``#`` as a comment only at a genuine word boundary -- the two ways
    a decoy slipped the first comment/span handling."""

    @pytest.mark.parametrize(
        "cmd",
        [
            # F1 -- ``$`` + backslash-newline + ``(`` folds to ``$(`` in bash, so
            # the opener must be seen after folding.
            "pkill -f $\\\n(true ')' ; true) {n}",
            "{n} $\\\n(true ')' ; true) {t}",
        ],
    )
    def test_line_continuation_opener_decoy_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_neutralizer_folds_continuation_before_scanning(self):
        # The folded span collapses; without folding the ``$\<LF>(`` is invisible.
        assert sn._neutralize_substitution_spans("a $\\\n(x) b") == (
            f"a {sn._SUBSTITUTION_PLACEHOLDER} b"
        )

    @pytest.mark.parametrize(
        "cmd",
        [
            # F2 -- ``$(true)#x`` is ONE word to bash (the ``#`` is glued to the
            # substitution's close), so it is NOT a comment, and the decoyed
            # self-kill after the ``;`` must still be seen.
            "echo $(true)#x; pkill -f $(true ')' ; true) {n}",
            "echo $(true)#x; {n} $(true ')' ; true) {t}",
        ],
    )
    def test_glued_hash_after_substitution_is_not_a_comment(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_hash_glued_to_a_span_stays_in_the_word(self):
        # The ``#x`` is word data after the span, not a comment, so the ``;`` and
        # the command after it survive neutralization.
        out = sn._neutralize_substitution_spans("echo $(true)#x; pkill -f $(y) foo")
        assert "#x" in out
        assert "pkill -f" in out

    def test_hash_at_a_real_word_boundary_is_still_a_comment(self):
        # A ``#`` after whitespace IS a comment: the span after it is dropped.
        out = sn._neutralize_substitution_spans("echo a # $(x) rest")
        assert out == "echo a # $(x) rest" or out.startswith("echo a # ")


class TestGluedSpanCollapsesToNothing:
    """A substitution span GLUED to the verb/name (no separating space) must
    collapse to nothing, because an empty-output generator fuses the neighbours
    in bash -- ``$(true)token`` -> ``token``.  A STANDALONE span stays a
    placeholder so the subcommand hidden-lead allowance is preserved."""

    @pytest.mark.parametrize(
        "cmd",
        [
            # the verb glued to the RIGHT of the decoy span
            "{n} $(true ')' ; true)token",
            # backtick spelling, glued
            "{n} `true ')' ; true`token",
            # double-quoted span glued to the verb
            "{n} \"$(true ')' ; true)\"token",
            # module form, span glued to the verb
            "python -m {mod} $(true ')' ; true)token",
            # kill: span glued to the LEFT of the name (also exercises decode)
            "pkill -f $(true ')' ; true)kiro{bs}crew",
            "pkill -f $(true ')' ; true)kirocrew",
        ],
    )
    def test_glued_decoy_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N, mod=_MOD, bs=chr(92)))

    def test_neutralizer_collapses_glued_span_to_nothing(self):
        # right-glued
        assert sn._neutralize_substitution_spans("x $(y)token") == "x token"
        # left-glued
        assert sn._neutralize_substitution_spans("kiro$(y)crew z") == "kirocrew z"
        # standalone stays a placeholder (keeps the operand slot)
        assert sn._neutralize_substitution_spans("x $(y) z") == (
            f"x {sn._SUBSTITUTION_PLACEHOLDER} z"
        )

    def test_standalone_decoy_before_subcommand_stays_allowed(self):
        # The hidden-lead pin: a STANDALONE span before ``restart`` is the lead
        # operand (not ``restart``), so this stays allowed -- the glued-span fix
        # must not collapse a standalone span and flip this to a deny.
        assert is_denied("{n} $(true ')' ; true) restart".format(n=_N)) is None
        assert is_denied("{n} restart".format(n=_N))


class TestNestedAndQuotedSpanEarlyClose:
    """Round-5 findings: a double-quoted STANDALONE span must stay a placeholder
    (not collapse to nothing and stop the scan), and a nested substitution inside
    double quotes must not close the outer span early."""

    @pytest.mark.parametrize(
        "cmd",
        [
            # Opus F2 -- the span is the whole content of a double-quoted word.
            # It must become a placeholder INSIDE the quotes, so the verb/name
            # after it is still scanned.  (echo -v / echo -f are flags bash passes
            # through, so bash runs ``kirocrew -v token`` / ``pkill -f kirocrew``.)
            "{n} \"$(true ')' ; echo -v)\" token",
            "pkill \"$(true ')' ; echo -f)\" {n}",
            "{n} \"$(true ')' ; true)\" token",
        ],
    )
    def test_double_quoted_standalone_span_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N))

    @pytest.mark.parametrize(
        "cmd",
        [
            # GPT F1 -- a nested ``$(`` inside double quotes within the outer
            # ``$(...)`` must not close the outer span at the inner quoted ``)``.
            'pkill -f $(true "$(true ")" ; true)" ; true) {n}',
            '{n} $(true "$(true ")" ; true)" ; true) token',
        ],
    )
    def test_nested_double_quoted_substitution_is_denied(self, cmd):
        assert is_denied(cmd.format(n=_N))

    def test_matching_close_paren_consumes_nested_quoted_substitution(self):
        # The shared scanner now spans the WHOLE outer substitution, not stopping
        # at the inner quoted ``)``.
        body = 'true "$(true ")" ; true)" ; true) rest'
        rel, proven = sn._matching_close_paren(body, 0)
        assert proven
        # everything up to and including the outer ``)`` is the span; ``rest`` is after
        assert body[rel:].strip() == "rest"


class TestWordStartAndHeredocBoundaries:
    """A word-start marker for the ``#`` rule and heredoc-body removal keep the
    neutralizer walk from being derailed before it reaches a decoyed span.

    Two ways the flat quote walk could skip the decoy and copy the tail verbatim
    (so the neutralized copy equalled the source and the pass was a no-op):
    an EMPTY quote delimiter before a ``#`` made the ``#`` read as a comment, and
    a quote character inside a HEREDOC body leaked its quote state past the
    terminator so a later ``$(`` looked quoted.
    """

    @pytest.mark.parametrize(
        "prefix",
        [
            # Empty double/single quotes START a word, so bash reads ``""#x`` /
            # ``''#x`` as the word ``#x`` -- NOT a comment -- and runs the tail.
            'echo ""#x; ',
            ": ''#x; ",
        ],
    )
    def test_empty_quote_before_hash_is_not_a_comment(self, prefix):
        cmd = prefix + "{n} $(true ')' ; true) {t}".format(n=_N, t=_T)
        assert is_denied(cmd)

    def test_hash_at_a_real_word_start_still_starts_a_comment(self):
        # A ``#`` at a genuine word start IS a comment: the mint on the commented
        # line never runs, so nothing is denied on that account.  (A leading
        # ``true`` keeps the first word off ``#`` so the line itself parses.)
        src = "true # {n} $(x) {t}".format(n=_N, t=_T)
        assert sn._neutralize_substitution_spans(src) == src

    def test_glued_hash_after_span_stays_data(self):
        # ``$(true)#x`` is one word: the ``#`` is glued to the consumed span, so
        # it is data, not a comment -- the word boundary is preserved.
        out = sn._neutralize_substitution_spans("$(true)#x rest")
        assert "#x rest" in out

    def test_heredoc_body_quote_does_not_leak_past_terminator(self):
        # A lone ``'`` inside the heredoc body must not leave the later ``$(``
        # looking single-quoted; the decoyed mint after the terminator is denied.
        cmd = "cat <<E\ndon't\nE\n{n} $(true ')' ; true) {t}".format(n=_N, t=_T)
        assert is_denied(cmd)

    def test_quoted_heredoc_tag_body_is_stripped(self):
        # ``<<'E'`` -- a quoted tag -- still bounds a body that is removed.
        cmd = "cat <<'E'\n\"unterminated\nE\n{n} $(true ')' ; true) {t}".format(n=_N, t=_T)
        assert is_denied(cmd)

    def test_strip_heredoc_bodies_removes_body_and_tag_line(self):
        assert sn._strip_heredoc_bodies("cat <<E\nbody line\nE\nrest") == "cat <<E\nrest"

    def test_strip_heredoc_bodies_dash_strips_tab_before_terminator(self):
        assert sn._strip_heredoc_bodies("cat <<-E\n\tbody\n\tE\nrest") == "cat <<-E\nrest"

    def test_strip_heredoc_bodies_handles_two_heredocs_on_one_line(self):
        src = "cat <<A <<B\na-body\nA\nb-body\nB\nrest"
        assert sn._strip_heredoc_bodies(src) == "cat <<A <<B\nrest"

    def test_strip_heredoc_bodies_ignores_quoted_and_here_string_operators(self):
        # ``<<`` inside quotes is data; ``<<<`` is a here-string, not a heredoc.
        assert sn._strip_heredoc_bodies("echo '<<E' rest") == "echo '<<E' rest"
        assert sn._strip_heredoc_bodies("cmd <<<word\nrest") == "cmd <<<word\nrest"


class TestArithmeticAndDeepNestingAreNotMisread:
    """Round-6 findings: ``$(( … ))`` arithmetic must not be read as a heredoc
    (its ``<<`` is a shift) nor collapsed as a command substitution (its result
    is a number spliced into the word), and a deeply nested double-quoted
    substitution must yield a verdict, never a RecursionError.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            # F1 -- ``$((1 << 2))`` was read as a heredoc tagged ``2``, so the
            # following line (a real self-kill) was stripped and allowed.
            "echo $((1 << 2))\npkill -f $(true ')' ; true) {n}",
            "echo $((3<<1)) ; pkill -f $(true ')' ; true) {n}",
            "x=$((1<<4)); {n} $(true ')' ; true) {t}",
            # ANSI-C / locale quoted heredoc delimiter must resolve to its literal
            # text so the terminator matches and the tail is NOT erased.
            "cat <<$'E'\nbody\nE\npkill -f $(true ')' ; true) {n}",
            "cat <<$\"E\"\nbody\nE\npkill -f $(true ')' ; true) {n}",
        ],
    )
    def test_arithmetic_or_quoted_tag_does_not_erase_the_tail(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_arithmetic_shift_is_not_a_heredoc(self):
        assert sn._heredoc_tags_on_line("echo $((1 << 2))")[0] == []
        # a real heredoc on the same line is still found
        assert sn._heredoc_tags_on_line("cat <<E $((1 << 2))")[0] == [(False, "E")]

    @pytest.mark.parametrize(
        "delim,tag",
        [("<<$'E'", "E"), ('<<$"E"', "E"), ("<<-$'E'", "E"), ("<<'E'", "E"), ("<<E", "E")],
    )
    def test_quote_introducer_stripped_from_heredoc_tag(self, delim, tag):
        tags = sn._heredoc_tags_on_line("cat " + delim)[0]
        assert tags and tags[0][1] == tag

    @pytest.mark.parametrize(
        "cmd",
        [
            # F3 -- ``kiro$((2))crew`` is arithmetic glued into the name, so bash
            # runs ``kiro2crew`` which the base permits.  The neutralizer must
            # keep the arithmetic verbatim, not collapse it and manufacture
            # ``kirocrew``.
            "pkill -f kiro$((2))crew",
            "pkill -f kiro$((1+1))crew",
        ],
    )
    def test_arithmetic_does_not_manufacture_a_protected_target(self, cmd):
        assert is_denied(cmd) is None
        # the literal spelling the arithmetic evaluates to is likewise permitted
        assert is_denied("pkill -f kiro2crew") is None

    def test_neutralizer_keeps_arithmetic_verbatim(self):
        assert sn._neutralize_substitution_spans("kiro$((2))crew") == "kiro$((2))crew"
        assert sn._neutralize_substitution_spans("a $((1+1)) b") == "a $((1+1)) b"
        # a true command substitution next to arithmetic still collapses
        assert sn._neutralize_substitution_spans("a $((1+1)) $(x) b") == (
            f"a $((1+1)) {sn._SUBSTITUTION_PLACEHOLDER} b"
        )

    def test_deeply_nested_double_quoted_substitution_does_not_crash(self):
        # F2 -- a deeply nested double-quoted substitution must not raise
        # RecursionError out of the scanner.  The recursion budget makes the
        # span scanner itself bounded, which is what the crash depended on; the
        # cost of the FULL floor on a huge pathological command is the
        # pre-existing super-linear self-kill walk tracked separately, so this
        # asserts the budget directly (cheap) plus one modest end-to-end call.
        deep = 'true "$(' * 2000 + "x" + ')"' * 2000
        rel, proven = sn._matching_close_paren(deep, 0)  # no RecursionError
        assert proven is False and rel == len(deep)
        # a modest nested command still returns a verdict end to end
        modest = "kirocrew doctor $(" + 'true "$(' * 30 + "true" + ')"' * 30 + ")"
        assert is_denied(modest) is None or isinstance(is_denied(modest), str)

    def test_matching_close_paren_handles_deep_nesting_without_recursion(self):
        # The nested double-quoted substitution walk is iterative, so a deeply
        # nested span neither raises RecursionError nor runs to the end of the
        # text: a balanced span resolves to its REAL closing boundary.
        inner = 'true "$(' * 200 + "x" + ')"' * 200
        text = inner + ") rest"
        rel, proven = sn._matching_close_paren(text, 0)
        assert proven is True
        assert text[rel:].strip() == "rest"

    @pytest.mark.parametrize(
        "cmd",
        [
            # F1(round7) -- a BARE ((1 << 2)) arithmetic command (no $ prefix) was
            # read as a heredoc tagged '2' and erased the following self-kill.
            "((1 << 2))\npkill -f $(true ')' ; true) {n}",
            "((3<<1)); {n} $(true ')' ; true) {t}",
        ],
    )
    def test_bare_arithmetic_command_is_not_a_heredoc(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_bare_arithmetic_shift_is_not_a_heredoc_tag(self):
        assert sn._heredoc_tags_on_line("((1 << 2))")[0] == []
        assert sn._strip_heredoc_bodies("((1 << 2))\npkill -f x kirocrew") == (
            "((1 << 2))\npkill -f x kirocrew"
        )

    @pytest.mark.parametrize(
        "cmd",
        [
            # F2(round7) -- a quoted '(' inside a nested $() within arithmetic must
            # not stretch the arithmetic span over a later real substitution.
            "echo $(( $(true '(' ; printf 0) + 1)); pkill -f $(true ')' ; true) {n}",
            "echo $(( `true '('` + 1)); {n} $(true ')' ; true) {t}",
        ],
    )
    def test_quoted_paren_in_nested_sub_does_not_swallow_later_span(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_arith_expansion_end_is_quote_aware(self):
        # The nested $() carries a quoted '(' / ')'; the arithmetic span must end
        # at its own ')' pair, not be extended by the quoted parens.
        src = "$(( $(true '(' ; printf 0) + 1)) tail"
        end = sn._arith_expansion_end(src, 3)
        assert src[:end] == "$(( $(true '(' ; printf 0) + 1))"
        assert src[end:] == " tail"


class TestCrossLineQuoteAnsiCAndNestedArithmetic:
    """Round-8 findings: the heredoc scan must carry quote state ACROSS lines,
    decode an ANSI-C delimiter, and the arithmetic span must not be stretched by
    a substitution opened inside double quotes.
    """

    def test_multiline_quoted_double_angle_is_not_a_heredoc(self):
        # F1 -- a ``<<`` inside a MULTI-LINE double-quoted string is data, not a
        # heredoc opener; the quote opened on an earlier line must be carried.
        cmd = "echo \"a\n<<E\nb\"\npkill -f $(true ')' ; true) {n}".format(n=_N)
        assert is_denied(cmd)

    def test_strip_carries_quote_state_across_lines(self):
        # the open double quote on line 1 means line 2's ``<<E`` opens no heredoc,
        # so nothing after it is erased.
        src = 'echo "a\n<<E\nb"\npkill -f x kirocrew'
        assert sn._strip_heredoc_bodies(src) == src

    @pytest.mark.parametrize(
        "cmd",
        [
            # F2 -- the ANSI-C delimiter ``$'\x45'`` decodes to ``E``; the stripper
            # must match the ``E`` terminator and keep the command after it.
            "cat <<$'\\x45'\nbody\nE\npkill -f $(true ')' ; true) {n}",
            "cat <<$'\\x45'\nbody\nE\n{n} $(true ')' ; true) {t}",
        ],
    )
    def test_ansi_c_heredoc_delimiter_is_decoded(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_ansi_c_tag_decodes_to_literal(self):
        tags, _, _, _ = sn._heredoc_tags_on_line("cat <<$'\\x45'")
        assert tags == [(False, "E")]

    @pytest.mark.parametrize(
        "cmd",
        [
            # F3 -- a ``$(`` opened INSIDE double quotes within arithmetic must be
            # consumed, so its interior ``(`` does not stretch the arithmetic span
            # over the later self-kill.
            'echo $(( "$(true "(" ; printf 0)" + 1 )); pkill -f $(true \')\' ; true) {n}',
            'echo $(( "$(true "(" ; printf 0)" + 1 )); {n} $(true \')\' ; true) {t}',
        ],
    )
    def test_quoted_sub_in_arithmetic_does_not_swallow_tail(self, cmd):
        assert is_denied(cmd.format(n=_N, t=_T))

    def test_arith_expansion_end_handles_quoted_nested_sub(self):
        src = '$(( "$(true "(" ; printf 0)" + 1 )) tail'
        end = sn._arith_expansion_end(src, 3)
        assert src[:end] == '$(( "$(true "(" ; printf 0)" + 1 ))'
        assert src[end:] == " tail"


class TestGluedSpanConservativelyRefusedNoResolver:
    """A glued substitution span collapses to the empty string (fail-closed), so
    the floor leans toward DETECTION.

    This is a deliberate, KNOWN OVER-REFUSAL: a glued span whose generator emits a
    non-empty literal (``kiro$(printf 2)crew`` -> bash runs ``kiro2crew``, a
    different program) is refused even though it is not a self-kill.  Statically
    modelling the generator's output to avoid that is not attempted, because
    guessing empty where bash emits text would HIDE a real self-kill; refusal is
    the safe default.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            # KNOWN OVER-REFUSAL (safe failure): bash would run ``kiro2crew`` (a
            # different program), but the glued span collapses to empty so the
            # floor refuses rather than risk hiding a self-kill.
            "pkill -f kiro$(printf 2)crew",
            'pkill -f "kiro$(printf 2)crew"',
            "pkill -f kiro$(echo 2)crew",
        ],
    )
    def test_glued_nonempty_generator_is_conservatively_refused(self, cmd):
        assert is_denied(cmd)

    @pytest.mark.parametrize(
        "cmd",
        [
            # Empty-output and opaque generators glued into the name fuse to the
            # self-name, so these are correctly denied.
            "pkill -f kiro$()crew",
            "pkill -f kiro$(true)crew",
            "pkill -f kiro$(x)crew",
        ],
    )
    def test_empty_or_opaque_glue_still_denied(self, cmd):
        assert is_denied(cmd)

    def test_neutralizer_collapses_every_glued_span_to_empty(self):
        # All glued spans -- empty, opaque, or non-empty-generator -- collapse to
        # the empty string (fail-closed); none is resolved.
        assert sn._neutralize_substitution_spans("kiro$()crew") == "kirocrew"
        assert sn._neutralize_substitution_spans("kiro$(x)crew") == "kirocrew"
        assert sn._neutralize_substitution_spans("kiro$(true)crew") == "kirocrew"
        assert sn._neutralize_substitution_spans("kiro$(printf 2)crew") == "kirocrew"
        assert sn._neutralize_substitution_spans("kiro$(echo foo)crew") == "kirocrew"


class TestHeredocDelimiterAndMultilineArithmeticEdges:
    """Round-10 findings: the heredoc scan must handle a multi-line arithmetic
    span, an empty quoted delimiter and a control-character delimiter without
    erasing the command after it, and the glued-span resolver must not resolve a
    quoted flag.
    """

    def test_quoted_flag_generator_is_conservatively_refused(self):
        # F1 (round-4) -- the earlier literal resolver that could be fooled by a
        # quoted flag was WITHDRAWN; a glued span now always collapses to empty,
        # so a would-be ``kiro$(printf '-n')crew`` is simply refused (fail-closed),
        # with no resolver to mis-model the generator's output.
        assert is_denied("pkill -f kiro$(printf '-n')crew")
        assert is_denied("pkill -f kiro$(echo '-n')crew")

    def test_multiline_arithmetic_shift_is_not_a_heredoc(self):
        # F2 -- a $((...)) spanning lines: the << on the later line is a shift,
        # not a heredoc, so the self-kill after it is still seen.
        cmd = "x=$((1 +\n2 << 2))\npkill -f $(true ')' ; true) {n}".format(n=_N)
        assert is_denied(cmd)

    def test_empty_quoted_heredoc_delimiter_body_is_stripped(self):
        # F3 -- <<'' closes on a blank line; the body (with its apostrophe) must
        # not leak into command scanning.
        cmd = "cat <<''\n'\n\npkill -f $(true ')' ; true) {n}".format(n=_N)
        assert is_denied(cmd)
        # the empty delimiter is recorded (not dropped) so the body is consumed
        assert sn._strip_heredoc_bodies("cat <<''\nbody\n\nrest") == "cat <<''\nrest"

    def test_control_char_heredoc_delimiter_decodes_losslessly(self):
        # F4 -- <<$'\t' has a TAB delimiter; the lossless decode matches a tab
        # terminator line so the command after it is not erased.
        assert sn._decode_heredoc_tag_escapes("\\t") == "\t"
        assert sn._decode_heredoc_tag_escapes("\\x45") == "E"
        cmd = "cat <<$'\\t'\nbody\n\t\npkill -f $(true ')' ; true) {n}".format(n=_N)
        assert is_denied(cmd)

    @pytest.mark.parametrize(
        "cmd",
        [
            # none of these edges may reopen a prior High self-kill:
            "pkill -f kiro$()crew",
            "pkill -f kiro$(x)crew",
            "pkill -f kiro$(true)crew",
            "cat <<E\nbody\nE\npkill -f $(true ')' ; true) {n}",
            "echo \"a\n<<E\nb\"\npkill -f $(true ')' ; true) {n}",
        ],
    )
    def test_no_prior_high_reopened(self, cmd):
        assert is_denied(cmd.format(n=_N))


class TestArithmeticQuoteStateAndUnmatchedDelimiterFailSafe:
    """Round-11 findings: the arithmetic branch must carry the surrounding quote
    state (the ``$(`` branch does), and an UNMATCHED heredoc delimiter must erase
    nothing rather than consume every remaining line.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            # F1 -- after a ``$(( ))`` the quote state must be preserved, or a
            # decoy that follows hides the self-kill.
            "echo \"$((1))\" ; pkill -f $(true ')' ; true) {n}",
            "x=$((1)); pkill -f $(true ')' ; true) {n}",
        ],
    )
    def test_arithmetic_branch_preserves_quote_state(self, cmd):
        assert is_denied(cmd.format(n=_N))

    def test_unmatched_heredoc_delimiter_erases_nothing(self):
        # F2 -- a delimiter whose terminator never appears must NOT consume the
        # rest of the input (that would drop a real command); fail safe.
        src = "cat <<$'\\u0045'\nbody\npkill -f x kirocrew"
        out = sn._strip_heredoc_bodies(src)
        assert "pkill -f x kirocrew" in out
        assert "body" in out  # nothing erased

    def test_unmatched_delimiter_self_kill_is_denied(self):
        cmd = "cat <<$'\\u0045'\nbody\npkill -f $(true ')' ; true) {n}".format(n=_N)
        assert is_denied(cmd)

    def test_matched_heredoc_still_strips(self):
        # the fail-safe must not stop a genuine, matchable heredoc from stripping.
        assert sn._strip_heredoc_bodies("cat <<E\nb\nE\nrest") == "cat <<E\nrest"
        cmd = "cat <<E\nbody\nE\npkill -f $(true ')' ; true) {n}".format(n=_N)
        assert is_denied(cmd)


class TestEveryNeutralizerExitOverMatches:
    """Round-12 ruling: audit EVERY exit of the neutralizer so none ERASES a
    protected argv suffix.  When a span's extent is unknown (unterminated, or a
    nested walk that cannot resolve the boundary), it must over-match (inert
    placeholder) or the surrounding scan must keep the text, never collapse the
    rest to nothing.  One test per exit.
    """

    def _ph(self):
        return sn._SUBSTITUTION_PLACEHOLDER

    def test_depth_exhaustion_preserves_the_suffix(self):
        # The nested walk is iterative now, so a deeply nested span resolves its
        # real boundary and the trailing kill target survives neutralization.
        nested = 'true "$(' * 121 + "true" + ')"' * 121
        cmd = "pkill -f $(true ')' ; " + nested + ") kirocrew"
        assert sn._neutralize_substitution_spans(cmd).endswith("kirocrew")
        assert is_denied(cmd)

    def test_unterminated_command_substitution_over_matches(self):
        # No closing ``)``: the span has no known end, so it becomes a placeholder
        # (never the empty string that would erase the rest).
        out = sn._neutralize_substitution_spans("a $(true ; echo b")
        assert self._ph() in out

    def test_unterminated_backtick_over_matches(self):
        out = sn._neutralize_substitution_spans("a `true ; echo b")
        assert self._ph() in out

    def test_unterminated_process_substitution_over_matches(self):
        out = sn._neutralize_substitution_spans("a <(true ; echo b")
        assert self._ph() in out

    def test_glued_unterminated_substitution_is_not_erased(self):
        # Even GLUED, an unterminated span must NOT collapse to empty (that would
        # erase the trailing text); it over-matches to the placeholder.
        out = sn._neutralize_substitution_spans("kiro$(true ; echo crew")
        assert out != "kiro" and self._ph() in out

    def test_unclosed_arithmetic_at_eof_is_copied_verbatim(self):
        # An unterminated ``$((`` copies the remainder verbatim (no erasure).
        out = sn._neutralize_substitution_spans("x=$((1 + kirocrew")
        assert "kirocrew" in out

    def test_unmatched_heredoc_delimiter_erases_nothing_exit(self):
        src = "cat <<$'\\u0041'\nbody\npkill -f x kirocrew"
        out = sn._strip_heredoc_bodies(src)
        assert "pkill -f x kirocrew" in out and "body" in out

    def test_proven_empty_glued_span_still_collapses(self):
        # The ONLY empty-collapse that survives: a PROVEN span (cleanly closed)
        # glued to a word, where bash fuses the neighbours.
        assert sn._neutralize_substitution_spans("kiro$()crew") == "kirocrew"
        assert sn._neutralize_substitution_spans("kiro$(x)crew") == "kirocrew"
