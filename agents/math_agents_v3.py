from __future__ import annotations

from typing import Any

from agents.common import reasoning_excerpt, run_agent as _run_agent


PLANNER_MAX_TOKENS = 512
PRIMARY_SOLVER_MAX_TOKENS = 4096
ADVERSARIAL_VERIFIER_MAX_TOKENS = 1280
REVISER_MAX_TOKENS = 4096
FINAL_CHECKER_MAX_TOKENS = 768
RESCUE_SOLVER_MAX_TOKENS = 4096
RESCUE_FINALIZER_MAX_TOKENS = 512
V3_MAX_INPUT_TOKENS = 4096

# v3.1: evidence-grounded disagreement jury.  Every solver declares a
# preliminary answer before reasoning, so a long derivation cannot turn a
# usable attempt into MISSING merely because generation reaches its cap.
EVIDENCE_LEDGER_MAX_TOKENS = 512
STRUCTURAL_SOLVER_MAX_TOKENS = 3072
INDEPENDENT_SOLVER_MAX_TOKENS = 3072
DISAGREEMENT_ARBITER_MAX_TOKENS = 1536
MICRO_RESCUE_MAX_TOKENS = 2048
JURY_FINALIZER_MAX_TOKENS = 768
V3_JURY_MAX_INPUT_TOKENS = 4096
VISUAL_INTERPRETER_MAX_TOKENS = 768
TAIL_COMMIT_MAX_TOKENS = 192
COMPACT_SPECIALIST_MAX_TOKENS = 2048
DENSE_VISUAL_AUDITOR_MAX_TOKENS = 1024
EMERGENCY_ANSWERER_MAX_TOKENS = 384

# V3: role-specialized, sequential collaboration. These agents do not
# generate independent full solutions for voting.  Each owns a different
# transformation of a shared blackboard, so the gain is attributable to
# division of labour and cross-agent error correction rather than SC.
BLACKBOARD_DECOMPOSER_MAX_TOKENS = 640
BLACKBOARD_PLAN_AUDITOR_MAX_TOKENS = 640
BLACKBOARD_WORKER_MAX_TOKENS = 1152
BLACKBOARD_INTEGRATOR_MAX_TOKENS = 1280
BLACKBOARD_VERIFIER_MAX_TOKENS = 1024
BLACKBOARD_REPAIRER_MAX_TOKENS = 1536
TABLE_TRANSCRIBER_MAX_TOKENS = 640
TABLE_MAPPING_AUDITOR_MAX_TOKENS = 512
TABLE_CALCULATOR_MAX_TOKENS = 640


def run_planner(problem: str, *, seed: int = 43) -> dict[str, Any]:
    """Decompose the task and define checks without proposing an answer."""

    return _run_agent(
        role="planner",
        seed=seed,
        max_new_tokens=PLANNER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Planner in a multi-agent mathematics team. "
                    "Decompose the problem and define falsifiable checks. Do "
                    "not solve it and do not propose a numeric final answer."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Return exactly six compact fields:
TARGET_QUANTITY: the exact mathematical quantity requested by the last question
ANSWER_TYPE: use scalar for volume/area/count/probability; otherwise expression / equation / set / ordered tuple / geometric object
METHOD: the main representation or theorem to use
SUBGOALS: ordered subgoals separated by semicolons
RISK_POINTS: likely sign, domain, counting, or generalization mistakes
CHECKLIST: include one mathematical check and one target/answer-type check
""".strip(),
            },
        ],
    )


def run_primary_solver(
    problem: str,
    planner_response: str,
    *,
    seed: int = 44,
) -> dict[str, Any]:
    """Execute the shared plan with a full reasoning budget."""

    plan = reasoning_excerpt(planner_response, max_tokens=420, head_ratio=0.7)
    return _run_agent(
        role="primary_solver",
        seed=seed,
        max_new_tokens=PRIMARY_SOLVER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Primary Solver. Execute the Planner's useful "
                    "subgoals, but correct the plan if necessary. Complete the "
                    "derivation and explicitly run its checklist. The final "
                    "answer must be the requested quantity, not a description "
                    "of an intermediate geometric object."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

TEAM_PLAN:
{plan}

First identify TARGET_QUANTITY and ANSWER_TYPE from the plan and final question.
Solve completely. Do not stop at a special case or intermediate value. If the
problem asks for a volume, area, count, or probability, submit that scalar rather
than merely naming the region or object.
End with exactly:
FINAL_ANSWER: \\boxed{{your answer}}
""".strip(),
            },
        ],
    )


def run_adversarial_verifier(
    problem: str,
    planner_response: str,
    solver_response: str,
    *,
    seed: int = 45,
) -> dict[str, Any]:
    """Try to falsify the proposed derivation and identify the earliest issue."""

    plan = reasoning_excerpt(planner_response, max_tokens=300, head_ratio=0.7)
    solution = reasoning_excerpt(solver_response, max_tokens=1250, head_ratio=0.3)
    return _run_agent(
        role="adversarial_verifier",
        seed=seed,
        max_new_tokens=ADVERSARIAL_VERIFIER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an adversarial mathematical Verifier. Your role "
                    "is not to restate the solution or vote. Attempt to falsify "
                    "the earliest decisive step using substitution, boundary "
                    "cases, independent arithmetic, or a small exact example."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

PLAN_AND_CHECKLIST:
{plan}

PRIMARY_SOLUTION_EXCERPT:
{solution}

Return exactly these fields:
VERDICT: PASS / REVISE / UNCERTAIN
TARGET_TYPE_CHECK: whether the candidate answers the exact requested quantity
DECISIVE_CHECK: the concrete check and its computed result
EARLIEST_ISSUE: NONE, or the first invalid/incomplete step
REQUIRED_FIX: NONE, or the exact step to recompute
VERIFIED_ANSWER: \\boxed{{answer}} only if established, otherwise MISSING

Use PASS only when the proposed answer has the required type and its decisive
derivation survives the check. A correct object description is not a valid
answer when the problem asks for its area, volume, or another scalar.
""".strip(),
            },
        ],
    )


def run_reviser(
    problem: str,
    planner_response: str,
    solver_response: str,
    verifier_response: str,
    *,
    seed: int = 46,
) -> dict[str, Any]:
    """Repair the challenged step and produce a complete revised answer."""

    plan = reasoning_excerpt(planner_response, max_tokens=260, head_ratio=0.7)
    solution = reasoning_excerpt(solver_response, max_tokens=900, head_ratio=0.25)
    verification = reasoning_excerpt(
        verifier_response, max_tokens=750, head_ratio=0.55
    )
    return _run_agent(
        role="reviser",
        seed=seed,
        max_new_tokens=REVISER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Reviser in a closed-loop mathematics team. "
                    "Treat all prior messages as fallible evidence. Recompute "
                    "the challenged step, repair the solution, and verify the "
                    "general answer rather than copying a check value. Match "
                    "the exact target quantity and answer type."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

PLAN:
{plan}

PRIMARY_SOLUTION_EXCERPT:
{solution}

ADVERSARIAL_REVIEW:
{verification}

Produce a corrected, self-contained decisive derivation. Before submitting,
re-read the final question and compute the requested scalar or expression; do
not substitute a description of the underlying object. End with exactly:
FINAL_ANSWER: \\boxed{{your answer}}
""".strip(),
            },
        ],
    )


def run_final_checker(
    problem: str,
    planner_response: str,
    candidate_response: str,
    verifier_response: str,
    *,
    seed: int = 47,
) -> dict[str, Any]:
    """Accept only a candidate that survives two short independent checks."""

    plan = reasoning_excerpt(planner_response, max_tokens=230, head_ratio=0.7)
    candidate = reasoning_excerpt(
        candidate_response, max_tokens=950, head_ratio=0.2
    )
    # The checker is intentionally independent of the adversarial review. This
    # prevents an erroneous critique from anchoring both revision and checking.
    _ = verifier_response
    return _run_agent(
        role="final_checker",
        seed=seed,
        max_new_tokens=FINAL_CHECKER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the final independent Checker, not a stylistic "
                    "judge. Run the requested checks on the candidate. Never "
                    "mark PASS merely because its derivation looks plausible."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

CHECKLIST:
{plan}

CANDIDATE_SOLUTION:
{candidate}

Return exactly five compact fields, with the first two lines in this order:
CHECK_STATUS: PASS / FAIL / UNCERTAIN
CHECKED_ANSWER: \\boxed{{same candidate answer}} only on PASS, otherwise MISSING
TARGET_CHECK: state the requested quantity/type and whether the candidate matches it
CHECK_1: one short independent substitution, recomputation, or boundary check
CHECK_2: one short constraint, completeness, or generality check

FAIL any answer that describes an object when the problem asks for its volume,
area, count, probability, or another scalar quantity.
Use at most ten short lines total; never repeat the candidate derivation.
""".strip(),
            },
        ],
    )


def run_rescue_solver(
    problem: str,
    planner_response: str,
    solver_response: str,
    verifier_response: str,
    candidate_response: str,
    checker_response: str,
    *,
    seed: int = 48,
) -> dict[str, Any]:
    """Independently resolve a failed or truncated closed-loop attempt."""

    plan = reasoning_excerpt(planner_response, max_tokens=220, head_ratio=0.7)
    solver = reasoning_excerpt(solver_response, max_tokens=420, head_ratio=0.2)
    verifier = reasoning_excerpt(verifier_response, max_tokens=420, head_ratio=0.5)
    candidate = reasoning_excerpt(candidate_response, max_tokens=550, head_ratio=0.15)
    checker = reasoning_excerpt(checker_response, max_tokens=450, head_ratio=0.55)
    return _run_agent(
        role="rescue_solver",
        seed=seed,
        max_new_tokens=RESCUE_SOLVER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        do_sample=True,
        temperature=0.55,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the independent Rescue Solver. The previous team "
                    "failed a final check or was truncated. Re-solve the core "
                    "mathematics using a different representation. Use the "
                    "failure evidence only to avoid repeating its mistake. "
                    "Re-read the final question and match its answer type."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

PLAN_CAPSULE:
{plan}

FAILED_PRIMARY_CAPSULE:
{solver}

VERIFIER_CAPSULE:
{verifier}

FAILED_CANDIDATE_CAPSULE:
{candidate}

FINAL_CHECK_FAILURE:
{checker}

Start by stating the exact TARGET_QUANTITY and ANSWER_TYPE requested. Solve
independently, complete the derivation, and run a final sanity check. If the
target is a volume, area, count, or probability, calculate that scalar instead
of returning only an object description.
End with exactly:
FINAL_ANSWER: \\boxed{{your answer}}
""".strip(),
            },
        ],
    )


def run_rescue_finalizer(
    problem: str,
    rescue_response: str,
    checker_response: str,
    *,
    seed: int = 49,
) -> dict[str, Any]:
    """Continue a truncated rescue derivation without restarting or guessing."""

    rescue_tail = reasoning_excerpt(
        rescue_response,
        max_tokens=1200,
        head_ratio=0.08,
    )


def run_blackboard_decomposer_v3(
    problem: str,
    *,
    seed: int = 50,
) -> dict[str, Any]:
    """Create three ordered, dependent work packages without solving them."""

    return _run_agent(
        role="blackboard_decomposer",
        seed=seed,
        max_new_tokens=BLACKBOARD_DECOMPOSER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the task Decomposer for a hierarchical mathematics "
                    "team. Design a dependency graph, not a solution. Divide the "
                    "work so each specialist computes a small reusable artifact. "
                    "Do not calculate the final answer and do not infer metric "
                    "facts from illustrative Asymptote coordinates."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Fill exactly these six fields with problem-specific content:
TARGET: [quote the requested quantity and state its answer type]
REPRESENTATION: [name concrete variables and equations/theorems]
SUBGOAL_1: [one concrete calculation producing a named lemma]
SUBGOAL_2: [one concrete calculation that explicitly consumes RESULT_1]
SUBGOAL_3: [one concrete calculation that consumes earlier results and yields TARGET]
ASSEMBLY_CHECK: [write one actual equation/invariant to recompute]

Replace every bracketed instruction. Never copy phrases such as "exact requested
quantity", "first reusable lemma", or "derive the target candidate" as values.
For a polygon from computed points, make one subgoal establish the cyclic convex
order and use shoelace by default; do not assume perpendicular diagonals,
rectangle, kite, or rhombus without proof. Do not carry out calculations here.
""".strip(),
            },
        ],
    )


def run_blackboard_plan_auditor_v3(
    problem: str,
    plan_response: str,
    *,
    seed: int = 51,
) -> dict[str, Any]:
    """Audit the mathematical specification before any worker consumes it."""

    plan = reasoning_excerpt(plan_response, max_tokens=620, head_ratio=0.8)
    return _run_agent(
        role="blackboard_plan_auditor",
        seed=seed,
        max_new_tokens=BLACKBOARD_PLAN_AUDITOR_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Constraint Auditor, not a solver. Compare the "
                    "proposed plan line-by-line with the original problem. Catch "
                    "lost scale factors, coefficient ratios, signs, parity, "
                    "domains, distinctness, units, target changes, and unjustified "
                    "diagram measurements before workers inherit them."
                ),
            },
            {
                "role": "user",
                "content": f"""
Original problem:
{problem}

PROPOSED_PLAN:
{plan}

Return exactly these compact fields:
AUDIT_STATUS: PASS / REVISE
OMITTED_OR_DISTORTED_GIVENS: NONE or exact items
AUDITED_REPRESENTATION: corrected equations with every required scale/sign
AUDITED_SUBGOAL_1: corrected concrete task
AUDITED_SUBGOAL_2: corrected concrete task consuming RESULT_1
AUDITED_SUBGOAL_3: corrected concrete task yielding the original target
AUDITED_ASSEMBLY_CHECK: exact invariant including scale factors and constraints

For a polynomial, explicitly audit Vieta ratios such as a0/an and
a_(n-1)/an. For a diagram, coordinates are topology-only unless declared exact.
For one line tangent to two curves, each contact point must satisfy its own
parent curve and the common line; it normally does NOT lie on the other curve.
For polygon area, require cyclic vertex order and shoelace unless the exact
conditions of a special area formula are proved. Never invent a shape class.
Do not enumerate cases or compute the target answer.
""".strip(),
            },
        ],
    )


def run_blackboard_worker_v3(
    problem: str,
    plan_response: str,
    subgoal: str,
    prior_results: str,
    *,
    worker_id: int,
    seed: int,
) -> dict[str, Any]:
    """Solve one bounded work package using the accumulated blackboard."""

    plan = reasoning_excerpt(plan_response, max_tokens=500, head_ratio=0.8)
    prior = reasoning_excerpt(prior_results, max_tokens=850, head_ratio=0.25)
    return _run_agent(
        role=f"blackboard_worker_{worker_id}",
        seed=seed,
        max_new_tokens=BLACKBOARD_WORKER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are Worker {worker_id} in a sequential blackboard "
                    "team. You own only the assigned subgoal. Use earlier "
                    "entries as fallible inputs, correct a concrete upstream "
                    "error if found, and publish a reusable mathematical result. "
                    "Do not restart the entire problem or discuss team voting."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

TEAM_PLAN:
{plan}

YOUR_SUBGOAL:
{subgoal}

PRIOR_BLACKBOARD:
{prior if prior_results.strip() else 'EMPTY'}

Return at most ten short lines in this order:
DERIVATION: decisive equations only
LOCAL_CHECK: substitution, exhaustiveness, units, or domain check
UPSTREAM_CORRECTION: NONE, or a corrected earlier entry with evidence
RESULT_{worker_id}: the exact reusable lemma/value/constraint

If this is the final subgoal, RESULT_{worker_id} must be the requested quantity,
but do not use FINAL_ANSWER syntax; the Integrator owns final submission.
For a computed contact point, show substitution into its parent curve and
tangent line. For polygon area, explicitly order vertices around the convex
boundary before applying shoelace; a bow-tie ordering is invalid.
""".strip(),
            },
        ],
    )


def run_blackboard_integrator_v3(
    problem: str,
    plan_response: str,
    worker_responses: list[str],
    *,
    seed: int = 54,
) -> dict[str, Any]:
    """Assemble worker artifacts into one answer without re-solving globally."""

    plan = reasoning_excerpt(plan_response, max_tokens=400, head_ratio=0.8)
    blackboard = "\n\n".join(
        f"WORKER_{index}:\n{reasoning_excerpt(response, max_tokens=620, head_ratio=0.3)}"
        for index, response in enumerate(worker_responses, start=1)
    )
    return _run_agent(
        role="blackboard_integrator",
        seed=seed,
        max_new_tokens=BLACKBOARD_INTEGRATOR_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Integrator of a hierarchical mathematics team. "
                    "Assemble the dependency chain and resolve explicit upstream "
                    "corrections. Do not create a separate full solution. Reject "
                    "an entry that lacks the domain, completeness, or target-type "
                    "condition required by the plan."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

PLAN:
{plan}

SHARED_BLACKBOARD:
{blackboard}

Return compactly:
DEPENDENCY_AUDIT: RESULT_1 -> RESULT_2 -> RESULT_3, noting any correction used
ASSEMBLY_CHECK: evaluate the plan's exact invariant
TARGET_CHECK: confirm requested quantity, answer type, and units
FINAL_ANSWER: \\boxed{{answer}}

Use MISSING as the answer if the entries cannot support a checked conclusion.
If the target is polygon area from coordinates, independently enforce cyclic
boundary order and run shoelace; ignore any unproved diagonal formula.
""".strip(),
            },
        ],
    )


def run_blackboard_verifier_v3(
    problem: str,
    plan_response: str,
    worker_responses: list[str],
    integrator_response: str,
    *,
    seed: int = 55,
) -> dict[str, Any]:
    """Independently test the weakest bridge in the assembled dependency chain."""

    plan = reasoning_excerpt(plan_response, max_tokens=300, head_ratio=0.8)
    entries = "\n".join(
        reasoning_excerpt(response, max_tokens=300, head_ratio=0.25)
        for response in worker_responses
    )
    integrated = reasoning_excerpt(
        integrator_response, max_tokens=650, head_ratio=0.25
    )
    return _run_agent(
        role="blackboard_verifier",
        seed=seed,
        max_new_tokens=BLACKBOARD_VERIFIER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an independent blackboard Verifier. Recompute the "
                    "single weakest dependency bridge or invariant from the "
                    "original givens. Do not vote and do not reward agreement."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

PLAN:
{plan}

BLACKBOARD_ENTRIES:
{entries}

INTEGRATOR:
{integrated}

Return exactly five compact fields:
VERDICT: PASS / FAIL / UNCERTAIN
CHECKED_ANSWER: \\boxed{{answer}} only if independently established, else MISSING
FIRST_BAD_ENTRY: NONE or RESULT_1 / RESULT_2 / RESULT_3 / INTEGRATOR
DECISIVE_CHECK: explicit recomputation with its numerical/algebraic result
REPAIR_INSTRUCTION: NONE or the exact entry and constraint to repair

Reject a geometric contact point unless direct substitution into its own curve
and the tangent line succeeds. For polygon area, recompute from cyclically
ordered vertices using shoelace unless a special formula's hypotheses are proved.
""".strip(),
            },
        ],
    )


def run_blackboard_repairer_v3(
    problem: str,
    plan_response: str,
    worker_responses: list[str],
    integrator_response: str,
    verifier_response: str,
    *,
    seed: int = 56,
) -> dict[str, Any]:
    """Repair the first bad blackboard entry identified by the verifier."""

    plan = reasoning_excerpt(plan_response, max_tokens=260, head_ratio=0.8)
    # Deliberately hide the contradictory derivation corpus.  The repairer is
    # assigned the verifier's first bad bridge and receives only short result
    # capsules, preventing repeated anchoring on already rejected algebra.
    entries = "\n".join(
        reasoning_excerpt(response, max_tokens=110, head_ratio=0.08)
        for response in worker_responses
    )
    integrated = reasoning_excerpt(
        integrator_response, max_tokens=180, head_ratio=0.08
    )
    verification = reasoning_excerpt(
        verifier_response, max_tokens=520, head_ratio=0.6
    )
    return _run_agent(
        role="blackboard_repairer",
        seed=seed,
        max_new_tokens=BLACKBOARD_REPAIRER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Repairer in a hierarchical blackboard team. "
                    "Recompute the verifier's first bad entry from the original "
                    "givens, propagate only that correction downstream, and "
                    "preserve sound entries. Rejected derivations are hidden "
                    "on purpose; do not reconstruct or defend them."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

PLAN:
{plan}

RESULT_CAPSULES_ONLY:
{entries}

INTEGRATOR_CONCLUSION_CAPSULE:
{integrated}

VERIFIER:
{verification}

Return compactly:
REPAIRED_ENTRY: corrected lemma/equation with no more than eight equations
PROPAGATION: compute the requested target from the repaired dependency chain
FINAL_CHECK: one exact substitution/invariant plus answer-type/unit check
FINAL_ANSWER: \\boxed{{answer}}
""".strip(),
            },
        ],
    )


def run_table_transcriber_v3(
    problem: str,
    *,
    seed: int = 50,
) -> dict[str, Any]:
    """Map a dense table's requested slice; deliberately perform no arithmetic."""

    return _run_agent(
        role="table_transcriber",
        seed=seed,
        max_new_tokens=TABLE_TRANSCRIBER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a literal table Transcriber, not a solver. Decode "
                    "Asymptote label coordinates or LaTeX rows/columns. Select "
                    "only the slice requested in the final question. Never "
                    "average, total, estimate, or infer the answer."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem with table code:
{problem}

Return exactly these fields:
TARGET_SLICE: requested row/column/category
HEADERS: ordered row and column headers
VALUE_MAP: selected category -> exact table count/value
UNIT_MAP: category -> price/rate/unit supplied by prose
EXCLUDED_DATA: table cells deliberately not used
DATA_CHECK: reread coordinates/headers and confirm every selected mapping

Do not calculate the requested answer.
""".strip(),
            },
        ],
    )


def run_table_calculator_v3(
    problem: str,
    transcript_response: str,
    *,
    seed: int = 51,
) -> dict[str, Any]:
    """Compute only from the transcriber's explicit, auditable data slice."""

    transcript = reasoning_excerpt(
        transcript_response, max_tokens=700, head_ratio=0.8
    )
    return _run_agent(
        role="table_calculator",
        seed=seed,
        max_new_tokens=TABLE_CALCULATOR_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Calculator in a two-agent data pipeline. Use "
                    "only the audited explicit selected mappings. Check "
                    "the denominator, units, and requested rounding. If a needed "
                    "mapping is absent or inconsistent, return MISSING."
                ),
            },
            {
                "role": "user",
                "content": f"""
Original question (for target and units only):
{problem}

AUDITED_TRANSCRIPT_AND_MAPPING_AUDIT:
{transcript}

Return exactly four compact fields:
FORMULA: requested computation; for an average price/rate, explicitly use
sum(VALUE_MAP[category] * UNIT_MAP[category]) / sum(VALUE_MAP[category])
ARITHMETIC_CHECK: independently recompute numerator and denominator
UNIT_AND_ROUNDING_CHECK: requested output unit and applied rounding
FINAL_ANSWER: \\boxed{{answer}}
""".strip(),
            },
        ],
    )


def run_table_mapping_auditor_v3(
    problem: str,
    transcript_response: str,
    *,
    seed: int = 51,
) -> dict[str, Any]:
    """Cross-check every transcribed category against prose and table position."""

    transcript = reasoning_excerpt(
        transcript_response, max_tokens=700, head_ratio=0.8
    )
    return _run_agent(
        role="table_mapping_auditor",
        seed=seed,
        max_new_tokens=TABLE_MAPPING_AUDITOR_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data Mapping Auditor, not a calculator. Compare "
                    "each category's selected table value and prose-supplied "
                    "unit/rate against the original text and exact table "
                    "coordinate. Correct swaps or copied neighboring values."
                ),
            },
            {
                "role": "user",
                "content": f"""
Original problem:
{problem}

TRANSCRIBER_OUTPUT:
{transcript}

Return exactly these fields:
AUDIT_STATUS: PASS / REVISE
MAPPING_ERRORS: NONE or category -> wrong value -> corrected value
AUDITED_VALUE_MAP: category -> exact selected count/value
AUDITED_UNIT_MAP: category -> exact prose price/rate/unit
PAIRWISE_CHECK: one category per clause, confirming value and unit together

Do not total, average, or compute the requested answer.
""".strip(),
            },
        ],
    )
    checker = reasoning_excerpt(
        checker_response,
        max_tokens=260,
        head_ratio=0.55,
    )
    return _run_agent(
        role="rescue_finalizer",
        seed=seed,
        max_new_tokens=RESCUE_FINALIZER_MAX_TOKENS,
        max_input_tokens=V3_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Rescue Finalizer. Continue the truncated "
                    "rescue derivation from its exact endpoint. Do not restart "
                    "the problem, introduce a new method, or copy an earlier "
                    "unverified candidate."
                ),
            },
            {
                "role": "user",
                "content": f"Problem:\n{problem}\n\nFinal check context:\n{checker}",
            },
            {
                "role": "assistant",
                "content": rescue_tail,
            },
            {
                "role": "user",
                "content": (
                    "Finish only the remaining simplification and sanity check. "
                    "Output the requested quantity, not an intermediate object. "
                    "End with exactly:\nFINAL_ANSWER: \\boxed{your answer}"
                ),
            },
        ],
    )


def run_visual_interpreter_v3(
    problem: str,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Reconstruct diagram topology and table mappings without doing algebra."""

    return _run_agent(
        role="visual_interpreter",
        seed=seed,
        max_new_tokens=VISUAL_INTERPRETER_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a literal Asymptote/table interpreter, not a math "
                    "solver. Reconstruct only visual topology and data mapping. "
                    "At every intersection or named point, split each full drawn "
                    "line into both incident rays, even when those rays have no "
                    "named endpoint. Sort all incident rays by direction from "
                    "the given coordinates. A label's coordinate is its absolute "
                    "anchor and its N/S/E/W argument is an offset from that anchor; "
                    "never relocate it to a different vertex. Assign an angle "
                    "label to the drawn sector that contains its displayed anchor "
                    "plus offset. For tables, map every "
                    "cell to its exact row and column. Coordinates may be "
                    "illustrative for lengths but are authoritative for topology, "
                    "ray order, and label placement."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem containing visual code:
{problem}

Return a compact literal transcript only:
POINTS_AND_LINES: coordinates, collinearities, parallel lines, drawn segments
RAY_ORDER: clockwise ray order at each labeled vertex
ANGLE_LABEL_SECTORS: each label -> its two boundary rays
TABLE_MAP: each relevant row/column/value mapping, or NONE
TOPOLOGY_WARNING: any label whose anchor is offset for readability

Sanity rule: a label displayed on a top line cannot denote an angle at a bottom
vertex unless an explicit leader says so. Include unnamed left/right/up/down
rays created by full lines. Do not calculate the requested answer or infer an
unstated length.
""".strip(),
            },
        ],
    )


def run_evidence_ledger(
    problem: str,
    visual_response: str | None = None,
    *,
    seed: int = 43,
) -> dict[str, Any]:
    """Extract auditable givens and diagram/table facts without solving."""

    visual = reasoning_excerpt(
        visual_response or "NONE", max_tokens=680, head_ratio=0.8
    )
    return _run_agent(
        role="evidence_ledger",
        seed=seed,
        max_new_tokens=EVIDENCE_LEDGER_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Evidence Keeper for a mathematics jury. "
                    "Transcribe givens exactly and expose interpretation risks; "
                    "do not solve or guess the final answer. For Asymptote, "
                    "tables, charts, or diagrams, reconstruct labels, row/column "
                    "meaning, incidences, parallel lines, and marked angles from "
                    "the code before writing the ledger. A label near a vertex "
                    "must be assigned to the angular sector that physically "
                    "contains the label, not to the most convenient segment. "
                    "Never rename a marked target angle as a standard triangle "
                    "angle unless their two boundary rays are identical."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

INDEPENDENT_VISUAL_TRANSCRIPT:
{visual}

Return only these compact fields:
TARGET_SYMBOL_OR_PHRASE: quote the exact target symbol or words from the final
question without translating them into new mathematical notation
TARGET_BOUNDARY_RAYS: for a marked angle, copy its two rays from VISUAL_DATA;
otherwise NOT_APPLICABLE
ANSWER_TYPE: scalar / expression / set / ordered tuple
GIVENS: semicolon-separated exact facts and numerical mappings
VISUAL_DATA: reconstructed diagram/table facts, or NONE; for every marked angle
name its two boundary rays explicitly, and for every table map each value to its
row and column explicitly
CONSTRAINTS: domains, distinctness, positivity, ordering, units, conventions;
Asymptote coordinates define topology only and are not exact metric data unless
the problem statement explicitly assigns those coordinates
DATA_CHECK: one rereading check designed to catch a swapped label, row, column,
angle, side, price, or endpoint
METHOD_OPTIONS: two genuinely different promising representations

Use at most 12 short lines. Do not begin solving inside DATA_CHECK.
""".strip(),
            },
        ],
    )


def run_tail_commit_v3(
    problem: str,
    truncated_response: str,
    *,
    commit_role: str,
    seed: int = 49,
) -> dict[str, Any]:
    """Continue a reasoning tail only long enough to commit its resolved answer."""

    tail = reasoning_excerpt(
        truncated_response,
        max_tokens=1400,
        head_ratio=0.04,
    )
    return _run_agent(
        role=commit_role,
        seed=seed,
        max_new_tokens=TAIL_COMMIT_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "Continue the exact reasoning endpoint below. Do not restart, "
                    "reconsider earlier preliminary answers, or introduce a new "
                    "method. Commit only a conclusion logically resolved in the "
                    "tail; if it remains contradictory or unfinished, submit "
                    "MISSING rather than a preliminary guess."
                ),
            },
            {
                "role": "user",
                "content": f"Problem:\n{problem}",
            },
            {
                "role": "assistant",
                "content": tail,
            },
            {
                "role": "user",
                "content": (
                    "Output exactly one line and nothing else:\n"
                    "FINAL_ANSWER: \\boxed{answer concluded above, or MISSING}"
                ),
            },
        ],
    )


def run_structural_solver_v3(
    problem: str,
    ledger_response: str,
    *,
    seed: int = 44,
) -> dict[str, Any]:
    """Solve through structure/invariants while declaring an answer first."""

    ledger = reasoning_excerpt(ledger_response, max_tokens=700, head_ratio=0.8)
    return _run_agent(
        role="structural_solver",
        seed=seed,
        max_new_tokens=STRUCTURAL_SOLVER_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Structural Solver A. Prefer invariants, symmetry, "
                    "factorization, inequalities, geometric transformations, "
                    "or generating functions. Treat the ledger as fallible and "
                    "check it against the original problem. Commit to a best "
                    "answer before explaining so truncation cannot erase it. "
                    "Never calculate exact lengths or angles from illustrative "
                    "Asymptote coordinates unless the text declares them."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

EVIDENCE_LEDGER:
{ledger}

Re-read the original final question. Use TARGET_SYMBOL_OR_PHRASE and
TARGET_BOUNDARY_RAYS literally; never replace a marked angle by a different
triangle angle.
Your FIRST line must be:
PROPOSED_ANSWER: \\boxed{{your best answer}}
Then give a concise decisive derivation, explicitly verify DATA_CHECK, and test
one boundary/small case. If reasoning changes the answer, end with:
FINAL_ANSWER: \\boxed{{revised answer}}
Never spend tokens restarting the solution.
""".strip(),
            },
        ],
    )


def run_independent_solver_v3(
    problem: str,
    ledger_response: str,
    *,
    seed: int = 45,
) -> dict[str, Any]:
    """Independently solve with a complementary constructive/computational path."""

    ledger = reasoning_excerpt(ledger_response, max_tokens=620, head_ratio=0.8)
    return _run_agent(
        role="independent_solver",
        seed=seed,
        max_new_tokens=INDEPENDENT_SOLVER_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        do_sample=True,
        temperature=0.45,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Independent Solver B. Use a complementary route: "
                    "coordinates, direct enumeration, interpolation, numerical "
                    "sanity checks, or explicit construction. You do not see "
                    "Solver A or the router. Re-read all labels and constraints. "
                    "Commit to a best answer before the derivation. Never treat "
                    "illustrative Asymptote coordinates as exact metric givens."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

EVIDENCE_LEDGER:
{ledger}

Re-read the original final question. Use TARGET_SYMBOL_OR_PHRASE and
TARGET_BOUNDARY_RAYS literally; never replace a marked angle by a different
triangle angle.
Your FIRST line must be:
PROPOSED_ANSWER: \\boxed{{your best answer}}
Then derive it independently and perform one exact substitution, enumeration,
or numerical check. If it changes, end with:
FINAL_ANSWER: \\boxed{{revised answer}}
Keep moving forward; do not restart.
""".strip(),
            },
        ],
    )


def run_disagreement_arbiter_v3(
    problem: str,
    ledger_response: str,
    structural_response: str,
    independent_response: str,
    router_response: str | None,
    *,
    seed: int = 46,
) -> dict[str, Any]:
    """Locate the first disagreement and decide by an explicit decisive check."""

    ledger = reasoning_excerpt(ledger_response, max_tokens=560, head_ratio=0.75)
    solver_a = reasoning_excerpt(
        structural_response, max_tokens=760, head_ratio=0.18
    )
    solver_b = reasoning_excerpt(
        independent_response, max_tokens=760, head_ratio=0.18
    )
    router = reasoning_excerpt(
        router_response or "MISSING", max_tokens=240, head_ratio=0.25
    )
    return _run_agent(
        role="disagreement_arbiter",
        seed=seed,
        max_new_tokens=DISAGREEMENT_ARBITER_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Disagreement Arbiter. The router and both "
                    "solvers are fallible witnesses, not votes. Identify the "
                    "first fact or operation on which their answers diverge, "
                    "then decide it with one fresh exact calculation. Consensus "
                    "without a decisive check is not evidence. Diagram coordinates "
                    "establish topology and label sectors, not exact measures."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

EVIDENCE_LEDGER:
{ledger}

ROUTER_WITNESS:
{router}

STRUCTURAL_SOLVER:
{solver_a}

INDEPENDENT_SOLVER:
{solver_b}

Return these fields, putting the answer first so it survives truncation:
FINAL_ANSWER: \\boxed{{best supported answer, or MISSING}}
DECISION: CONSENSUS_CHECKED / VERIFIED_SELECTION / VERIFIED_REPAIR / UNRESOLVED
FIRST_DISAGREEMENT: exact conflicting fact or operation
DECISIVE_CHECK: fresh arithmetic, substitution, construction, or theorem check
REJECTED_CLAIMS: concise reasons unsupported candidates fail
CONFIDENCE: HIGH / MEDIUM / LOW

For diagrams/tables, recompute from VISUAL_DATA. For inverse functions, state
the principal-value convention. For a set, verify completeness, not one root.
""".strip(),
            },
        ],
    )


def run_micro_rescue_v3(
    problem: str,
    ledger_response: str,
    structural_response: str,
    independent_response: str,
    arbiter_response: str,
    *,
    seed: int = 47,
) -> dict[str, Any]:
    """Resolve only the disputed bottleneck instead of restarting the problem."""

    ledger = reasoning_excerpt(ledger_response, max_tokens=500, head_ratio=0.75)
    solver_a = reasoning_excerpt(
        structural_response, max_tokens=380, head_ratio=0.12
    )
    solver_b = reasoning_excerpt(
        independent_response, max_tokens=380, head_ratio=0.12
    )
    arbiter = reasoning_excerpt(arbiter_response, max_tokens=720, head_ratio=0.7)
    return _run_agent(
        role="micro_rescue",
        seed=seed,
        max_new_tokens=MICRO_RESCUE_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        do_sample=True,
        temperature=0.35,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Micro-Rescue specialist. Do not restart or "
                    "survey many methods. Resolve the arbiter's FIRST_DISAGREEMENT "
                    "with the smallest exact lemma, computation, enumeration, "
                    "or diagram reconstruction that settles the final answer. "
                    "If a diagram/table label is disputed, ignore the ledger's "
                    "interpretation and reconstruct the labeled sector or cell "
                    "directly from the original code."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

LEDGER:
{ledger}

SOLVER_A_CAPSULE:
{solver_a}

SOLVER_B_CAPSULE:
{solver_b}

ARBITER_REPORT:
{arbiter}

Your FIRST line must be:
PROPOSED_ANSWER: \\boxed{{best answer}}
Then resolve only the disputed bottleneck and give one exact verification.
If the answer changes, end with FINAL_ANSWER: \\boxed{{revised answer}}.
""".strip(),
            },
        ],
    )


def run_jury_finalizer_v3(
    problem: str,
    ledger_response: str,
    arbiter_response: str,
    micro_response: str,
    *,
    seed: int = 48,
) -> dict[str, Any]:
    """Choose between arbiter and micro-rescue using a short target check."""

    ledger = reasoning_excerpt(ledger_response, max_tokens=360, head_ratio=0.8)
    arbiter = reasoning_excerpt(arbiter_response, max_tokens=580, head_ratio=0.75)
    micro = reasoning_excerpt(micro_response, max_tokens=720, head_ratio=0.2)
    return _run_agent(
        role="jury_finalizer",
        seed=seed,
        max_new_tokens=JURY_FINALIZER_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Jury Foreperson. Select the answer supported "
                    "by the decisive check, not by verbosity or majority. "
                    "Return an explicit answer even if confidence is limited."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

LEDGER:
{ledger}

ARBITER:
{arbiter}

MICRO_RESCUE:
{micro}

Return exactly:
FINAL_ANSWER: \\boxed{{selected answer}}
FINAL_CHECK: one short calculation supporting the selection
TARGET_CHECK: confirm exact requested quantity and completeness
""".strip(),
            },
        ],
    )


def run_compact_specialist_rescue_v3(
    problem: str,
    *,
    seed: int = 61,
) -> dict[str, Any]:
    """Last-chance independent solve for cases where the stable chain has no answer."""

    return _run_agent(
        role="compact_specialist_rescue",
        seed=seed,
        max_new_tokens=COMPACT_SPECIALIST_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compact olympiad specialist called only after a "
                    "team failed to finish. Solve independently; you do not see "
                    "their attempts. Select the shortest exact representation. "
                    "For algebra consider Vieta, interpolation, symmetric sums, "
                    "or equality cases of moment inequalities. For geometry "
                    "consider similarity, reflection, distance-to-line, or area "
                    "coordinates, but never use illustrative diagram coordinates "
                    "as metric givens. For counting/probability consider a "
                    "bijection, symmetry, complement, or geometric-series "
                    "parameterization. Reject a method quickly if it stalls."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Use at most 18 compact mathematical lines. Start with:
METHOD_CHOICE: one theorem or representation
Then carry it through without restarting. Re-read the final question, check the
answer type and one exact condition, and end with exactly:
FINAL_ANSWER: \\boxed{{your answer}}
""".strip(),
            },
        ],
    )


def run_dense_visual_auditor_v3(
    problem: str,
    router_response: str,
    team_response: str,
    *,
    seed: int = 62,
) -> dict[str, Any]:
    """Recompute a dense chart/table question when two witnesses disagree."""

    router = reasoning_excerpt(router_response, max_tokens=180, head_ratio=0.1)
    team = reasoning_excerpt(team_response, max_tokens=260, head_ratio=0.1)
    return _run_agent(
        role="dense_visual_auditor",
        seed=seed,
        max_new_tokens=DENSE_VISUAL_AUDITOR_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You audit dense tables/charts. Two witnesses disagree, so "
                    "neither answer is evidence. Transcribe only the row/column "
                    "needed by the final question, map every value to its label, "
                    "and independently recompute the requested aggregate."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem with chart/table code:
{problem}

ROUTER_WITNESS:
{router}

TEAM_WITNESS:
{team}

Return at most 12 short lines:
TARGET_ROW_OR_COLUMN: exact requested slice
TRANSCRIPTION: label=value pairs used
RECOMPUTATION: one explicit formula with substituted values
DATA_CHECK: reread one price/unit/header mapping
FINAL_ANSWER: \\boxed{{independently recomputed answer}}
""".strip(),
            },
        ],
    )


def run_emergency_answerer_v3(
    problem: str,
    *,
    seed: int = 71,
) -> dict[str, Any]:
    """Cheap answer-only fallback used exclusively when every prior answer is missing."""

    return _run_agent(
        role="emergency_answerer",
        seed=seed,
        max_new_tokens=EMERGENCY_ANSWERER_MAX_TOKENS,
        max_input_tokens=V3_JURY_MAX_INPUT_TOKENS,
        do_sample=True,
        temperature=0.2,
        top_p=0.85,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the emergency final-answer specialist. Solve the "
                    "problem internally using the shortest exact theorem. Do not "
                    "show reasoning, alternatives, checks, or commentary."
                ),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Output exactly one line and nothing else:
FINAL_ANSWER: \\boxed{{answer to the exact final question}}
""".strip(),
            },
        ],
    )
