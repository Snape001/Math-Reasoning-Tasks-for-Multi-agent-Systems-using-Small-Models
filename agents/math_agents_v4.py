from __future__ import annotations

from typing import Any

from agents.common import reasoning_excerpt, run_agent as _run_agent


V4_MAX_INPUT_TOKENS = 4096
CONSTRUCTIVE_SOLVER_MAX_TOKENS = 3072
ALTERNATIVE_SOLVER_MAX_TOKENS = 3072
LOGIC_REVIEWER_MAX_TOKENS = 1280
CALCULATION_REVIEWER_MAX_TOKENS = 2048
DISAGREEMENT_ARBITER_MAX_TOKENS = 2048
DECISION_FINALIZER_MAX_TOKENS = 384


def run_constructive_solver_v4(
    problem: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Produce a complete solution from a direct constructive representation."""

    return _run_agent(
        role="constructive_solver",
        seed=seed,
        max_new_tokens=CONSTRUCTIVE_SOLVER_MAX_TOKENS,
        max_input_tokens=V4_MAX_INPUT_TOKENS,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Constructive Solver in a mathematics team. "
                    "Work independently: no other candidate is visible. Build a "
                    "direct derivation from the givens, track the exact requested "
                    "quantity, and check signs, domains, counting conditions, and "
                    "units. Do not imitate a generic persona; use explicit equations."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Your first line must be:
PROPOSED_ANSWER: \\boxed{{your preliminary answer}}

Then give a complete decisive derivation and one falsifiable check. If the
derivation changes the preliminary result, end with:
FINAL_ANSWER: \\boxed{{corrected answer}}
""".strip(),
            },
        ],
    )


def run_alternative_solver_v4(
    problem: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Independently solve with a deliberately different representation."""

    return _run_agent(
        role="alternative_solver",
        seed=seed,
        max_new_tokens=ALTERNATIVE_SOLVER_MAX_TOKENS,
        max_input_tokens=V4_MAX_INPUT_TOKENS,
        do_sample=True,
        temperature=0.65,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Alternative Solver in a mathematics team. Work "
                    "independently and seek a representation that is less likely "
                    "to share a standard-solution failure: invariants, coordinates, "
                    "a small exact case, complementary counting, substitution, or "
                    "another appropriate method. Still solve the full original "
                    "problem and answer its exact target."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Your first line must be:
PROPOSED_ANSWER: \\boxed{{your preliminary answer}}

Develop an independent solution. Explicitly test the step most likely to fail.
If that test changes the answer, end with:
FINAL_ANSWER: \\boxed{{corrected answer}}
""".strip(),
            },
        ],
    )


def _candidate_packet(
    candidates: list[dict[str, Any]],
    *,
    order: list[int],
) -> str:
    blocks: list[str] = []
    labels = "ABC"
    for display_index, candidate_index in enumerate(order):
        call = candidates[candidate_index]
        excerpt = reasoning_excerpt(
            call.get("raw_response", ""),
            max_tokens=680,
            head_ratio=0.28,
        )
        blocks.append(f"CANDIDATE_{labels[display_index]}:\n{excerpt}")
    return "\n\n".join(blocks)


def run_logic_reviewer_v4(
    problem: str,
    candidates: list[dict[str, Any]],
    vote_summary: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Audit target fidelity, logical validity, and minority evidence."""

    packet = _candidate_packet(candidates, order=[0, 1, 2])
    return _run_agent(
        role="logic_reviewer",
        seed=seed,
        max_new_tokens=LOGIC_REVIEWER_MAX_TOKENS,
        max_input_tokens=V4_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Logic and Target Reviewer. The three proposals "
                    "come from fallible agents. Majority is a prior, not proof. "
                    "Preserve a majority only when its derivation answers the exact "
                    "target and survives a concrete check. Protect a well-supported "
                    "minority when it exposes a shared error. Do not average answers."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

ANONYMOUS_VOTE_SUMMARY:
{vote_summary}

PROPOSALS_AND_EVIDENCE:
{packet}

Return exactly these compact fields:
VERDICT: SUPPORT_MAJORITY / OVERRIDE / UNCERTAIN
VERIFIED_ANSWER: \\boxed{{answer}} if logically established, otherwise MISSING
TARGET_CHECK: exact requested quantity, answer type, and whether it is satisfied
EARLIEST_DECISIVE_ERROR: NONE or the first concrete invalid step
DECISIVE_EVIDENCE: one equation, substitution, invariant, or counterexample

Do not support an answer merely because two candidates state it.
""".strip(),
            },
        ],
    )


def run_calculation_reviewer_v4(
    problem: str,
    candidate_answers: list[str | None],
    vote_summary: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Recompute the decisive mathematics without reading candidate rationales."""

    answer_lines = "\n".join(
        f"OPTION_{index}: {answer if answer is not None else 'MISSING'}"
        for index, answer in enumerate(candidate_answers, start=1)
    )
    return _run_agent(
        role="calculation_reviewer",
        seed=seed,
        max_new_tokens=CALCULATION_REVIEWER_MAX_TOKENS,
        max_input_tokens=V4_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the independent Calculation Reviewer. You do not see "
                    "the other agents' reasoning, only their submitted answers, so "
                    "recompute the decisive mathematics from the original problem. "
                    "Majority is not evidence. Check whether a minority option is "
                    "correct before accepting consensus."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

ANONYMOUS_VOTE_SUMMARY:
{vote_summary}

SUBMITTED_OPTIONS:
{answer_lines}

Recompute using a short independent derivation. Return exactly:
VERDICT: SUPPORT_MAJORITY / OVERRIDE / UNCERTAIN
VERIFIED_ANSWER: \\boxed{{answer}} if established, otherwise MISSING
INDEPENDENT_CHECK: the decisive recomputation
OPTION_COMPARISON: which submitted option, if any, matches the recomputation

Never select an option from frequency alone.
""".strip(),
            },
        ],
    )


def run_disagreement_arbiter_v4(
    problem: str,
    candidates: list[dict[str, Any]],
    vote_summary: str,
    logic_review: str,
    calculation_review: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Resolve only a genuine second-round disagreement."""

    packet = _candidate_packet(candidates, order=[2, 0, 1])
    logic_excerpt = reasoning_excerpt(
        logic_review, max_tokens=520, head_ratio=0.65
    )
    calculation_excerpt = reasoning_excerpt(
        calculation_review, max_tokens=650, head_ratio=0.65
    )
    return _run_agent(
        role="disagreement_arbiter",
        seed=seed,
        max_new_tokens=DISAGREEMENT_ARBITER_MAX_TOKENS,
        max_input_tokens=V4_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the conditional Disagreement Arbiter. The first-round "
                    "solvers or second-round reviewers disagree. Locate the smallest "
                    "mathematical claim that separates the answers and independently "
                    "recompute that claim. Do not conduct another open-ended debate "
                    "and do not treat majority as correctness."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

VOTE_SUMMARY:
{vote_summary}

REORDERED_PROPOSALS:
{packet}

LOGIC_REVIEW:
{logic_excerpt}

CALCULATION_REVIEW:
{calculation_excerpt}

Your first line must be:
PROPOSED_ANSWER: \\boxed{{best answer}}

Then state the decisive disputed claim and recompute it. End with exactly:
FINAL_ANSWER: \\boxed{{verified answer}}
""".strip(),
            },
        ],
    )


def run_decision_finalizer_v4(
    problem: str,
    arbiter_response: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Commit a supported conclusion from a truncated arbiter without re-solving."""

    arbiter_tail = reasoning_excerpt(
        arbiter_response,
        max_tokens=1200,
        head_ratio=0.05,
    )
    return _run_agent(
        role="decision_finalizer",
        seed=seed,
        max_new_tokens=DECISION_FINALIZER_MAX_TOKENS,
        max_input_tokens=V4_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Decision Finalizer, not another solver. The "
                    "Disagreement Arbiter was truncated after doing the main "
                    "mathematics. Recover only a conclusion that is explicitly "
                    "supported by its equations. Never copy an earlier preliminary "
                    "candidate when the later calculation corrected it, and never "
                    "invent a result that the arbiter did not establish."
                ),
            },
            {
                "role": "user",
                "content": f"""
Original problem:
{problem}

TAIL OF TRUNCATED ARBITER:
{arbiter_tail}

Check the requested target against the arbiter's last decisive calculation.
Return no derivation and exactly one line:
FINAL_ANSWER: \\boxed{{supported answer}}

If the tail contains no supported conclusion, return exactly:
FINAL_ANSWER: MISSING
""".strip(),
            },
        ],
    )
