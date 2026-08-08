from __future__ import annotations

from typing import Any

from agents.common import reasoning_excerpt, run_agent as _run_agent


V4_1_MAX_INPUT_TOKENS = 4096
EVIDENCE_AUDITOR_MAX_TOKENS = 1280
DISAGREEMENT_RESOLVER_MAX_TOKENS = 4096
COMMIT_FINALIZER_MAX_TOKENS = 768


def _candidate_packet(candidates: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, call in enumerate(candidates, start=1):
        excerpt = reasoning_excerpt(
            call.get("raw_response", ""),
            max_tokens=650,
            head_ratio=0.2,
        )
        answer = call.get("declared_answer") or "MISSING"
        blocks.append(
            f"SC_CANDIDATE_{index}\n"
            f"FORMAL_ANSWER: {answer}\n"
            f"REASONING_EXCERPT:\n{excerpt}"
        )
    return "\n\n".join(blocks)


def run_evidence_auditor_v4_1(
    problem: str,
    candidates: list[dict[str, Any]],
    vote_summary: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Convert divergent SC traces into a task-grounded evidence ledger."""

    packet = _candidate_packet(candidates)
    return _run_agent(
        role="evidence_auditor",
        seed=seed,
        max_new_tokens=EVIDENCE_AUDITOR_MAX_TOKENS,
        max_input_tokens=V4_1_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Evidence Auditor, not a fourth solver and not a "
                    "voter. Three independent CoT agents disagreed or failed to "
                    "commit. Extract only reusable facts and the earliest claim "
                    "that separates their answers. Treat all formal answers as "
                    "fallible. Preserve a supported minority instead of following "
                    "frequency."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

VOTE_SUMMARY:
{vote_summary}

INDEPENDENT_TRACES:
{packet}

Return exactly these compact fields and do not solve beyond the audit:
TARGET: exact requested quantity and answer type
GIVENS: equations, domains, multiplicities, signs, and geometric relations
AGREED_FACTS: facts supported by a concrete calculation
DISPUTED_STEP: earliest mathematical step separating the answers
CHECK_REQUIRED: one decisive recomputation or counterexample
SUPPORTED_MINORITY: answer plus evidence, or NONE
COMMIT_RULE: condition an answer must satisfy before submission

Audit symbol fidelity, endpoint inclusion, parity/sign constraints, feasibility
witnesses, indistinguishable-object multiplicities, and diagram labels whenever
they are relevant. Illustrative diagram coordinates are not givens.
""".strip(),
            },
        ],
    )


def run_disagreement_resolver_v4_1(
    problem: str,
    candidate_answers: list[str | None],
    evidence_ledger: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Resolve the one audited disagreement and stop after a checked commitment."""

    ledger = reasoning_excerpt(evidence_ledger, max_tokens=1050, head_ratio=0.75)
    options = "\n".join(
        f"OPTION_{index}: {answer or 'MISSING'}"
        for index, answer in enumerate(candidate_answers, start=1)
    )
    return _run_agent(
        role="disagreement_resolver",
        seed=seed,
        max_new_tokens=DISAGREEMENT_RESOLVER_MAX_TOKENS,
        max_input_tokens=V4_1_MAX_INPUT_TOKENS,
        do_sample=False,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the independent Disagreement Resolver. Solve only "
                    "the audited decisive step, then verify the original target. "
                    "Do not choose by vote count. A minimum/maximum needs both a "
                    "feasible witness and a bound; a probability needs elementary "
                    "outcomes with multiplicity; a diagram needs explicit angle or "
                    "incidence equations. Once a conclusion passes its check, "
                    "commit it immediately and stop. Do not reopen a verified result."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

SUBMITTED_OPTIONS:
{options}

EVIDENCE_LEDGER:
{ledger}

Recompute CHECK_REQUIRED from the original problem. Keep the response focused.
End immediately after exactly one final line:
FINAL_ANSWER: \\boxed{{verified answer}}
""".strip(),
            },
        ],
    )


def run_commit_finalizer_v4_1(
    problem: str,
    evidence_ledger: str,
    resolver_response: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Commit only a conclusion that survives the ledger's proof obligation."""

    ledger = reasoning_excerpt(evidence_ledger, max_tokens=700, head_ratio=0.75)
    resolver = reasoning_excerpt(
        resolver_response,
        max_tokens=1800,
        head_ratio=0.25,
    )
    return _run_agent(
        role="commit_finalizer",
        seed=seed,
        max_new_tokens=COMMIT_FINALIZER_MAX_TOKENS,
        max_input_tokens=V4_1_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a constraint-aware Commit Finalizer, not a new full "
                    "solver. The Resolver was truncated or failed to emit a final "
                    "field. Select a conclusion only if an explicit computation "
                    "satisfies the Evidence Ledger's COMMIT_RULE. Recency is not "
                    "evidence: reject a later guess that violates an earlier sign, "
                    "domain, multiplicity, target, or feasibility check."
                ),
            },
            {
                "role": "user",
                "content": f"""
Original problem:
{problem}

EVIDENCE_LEDGER:
{ledger}

RESOLVER_EXCERPT:
{resolver}

Check the target and commit rule. Return no explanation and exactly one line:
FINAL_ANSWER: \\boxed{{supported answer}}

If no conclusion is fully supported, return exactly:
FINAL_ANSWER: MISSING
""".strip(),
            },
        ],
    )
