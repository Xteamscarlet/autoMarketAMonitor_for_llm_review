---
description: Review current session and propose evolution changes
---

You are the Hermes Evolution reviewer for this OpenCode project.

## Task

Review the current session conversation and identify evolution-worthy moments. Look for:

1. **Corrections** - Where the user explicitly corrected you or said "don't do X" or "always do Y"
2. **Preferences** - Where the user expressed a stable preference (coding style, output format, workflow preference)
3. **Workflows** - Where a reusable multi-step procedure emerged
4. **Struggles** - Where you failed/retried multiple times (>=3 retries or >=8 tool calls) before succeeding

## Focus Type

If $ARGUMENTS is provided, only look for that type: $ARGUMENTS
Otherwise, look for all types.

## Evolution Intensity

Read `EVOLUTION.md` in the project root to determine the current intensity level:
- **100% (aggressive)**: Propose on strong signal OR two medium signals
- **50% (cautious)**: Only propose on strong signals, skip medium signals
- **0% (off)**: Do nothing, report "Evolution is disabled"

If `EVOLUTION.md` does not exist, default to **100%**.

## Signal Strength

**Strong signals:**
- User explicitly corrected you ("no, don't do that", "do it this way instead")
- User said "always/never" or "from now on"
- User taught a reusable new workflow
- You clearly failed and retried before succeeding

**Medium signals:**
- Clear step pattern emerged for a type of task
- A tool/environment pitfall worth remembering long-term
- Same preference mentioned more than once

## Process

1. Read `EVOLUTION.md` for intensity setting
2. Review the current conversation for signals
3. If no strong or qualifying signals found, respond: "No evolution-worthy moments found in this session."
4. If signals found:
   a. Read existing `AGENTS.md`, `MEMORY.md`, `TOOLS.md` (create if missing)
   b. Check `.opencode/evolution-drafts/pending/` and `.opencode/evolution-drafts/approved/` for duplicates
   c. Check `.opencode/evolution-drafts/rejected/` for recently rejected (within 14 days) proposals
   d. If no duplicate found, create a draft file

## Draft Format

Create file: `.opencode/evolution-drafts/pending/<signature>.md`

Where `<signature>` is a lowercase hyphenated string combining target and normalized rule, e.g.:
- `agents-research-before-writing`
- `memory-prefer-concise-replies`
- `tools-npm-run-needs-prefix`

Draft content:

```markdown
# Evolution Proposal: <short title>

- Proposal-ID: evo-<YYYY-MM-DD>-<signature>
- Status: pending
- Signature: <signature>
- Created-At: <YYYY-MM-DD HH:mm>
- Target-File: <AGENTS.md | MEMORY.md | TOOLS.md>
- Trigger-Type: <correction | preference | workflow | struggle>
- Confidence: <high | medium | low>

## Why This Matters
- <why this should persist>

## Evidence
- "<exact user quote or precise summary>"

## Proposed Change
<the exact content to append to the target file>

## Apply Plan
1. <which file to modify>
2. <where to insert or append>
3. <what the final result should look like>
```

## Target File Selection

- **AGENTS.md** → behavior rules, workflow rules, safety boundaries
- **MEMORY.md** → stable preferences, long-term facts, communication style
- **TOOLS.md** → tool pitfalls, environment config, command experience

## Duplicate Detection

Before creating a draft:
1. Check if the target file already contains a similar rule
2. Check `.opencode/evolution-drafts/pending/` for same signature
3. Check `.opencode/evolution-drafts/rejected/` for recently rejected (within 14 days)

If duplicate found:
- Already covered in target file → skip, tell user "Already covered"
- Pending draft exists → update it, don't create new
- Recently rejected without new evidence → skip

## User Approval

After creating the draft, present it to the user:

"I found an evolution-worthy moment in this session:

**[Title]**
- Target: [file]
- Type: [correction/preference/workflow/struggle]
- Confidence: [high/medium/low]

**Why:** [one sentence]

**Proposed change:** [the actual content]

Reply `approve` to apply, `reject` to dismiss, or `revise: <feedback>` to modify."

## If Approved

1. Apply the proposed change to the target file
2. Move draft from `pending/` to `approved/`
3. Confirm: "Evolution applied"

## If Rejected

1. Move draft from `pending/` to `rejected/`
2. Confirm: "Proposal rejected, draft archived"

## If Revised

1. Update the draft based on feedback
2. Present the revised version again for approval

## Quality Standard

Only propose changes that are:
- **Specific** - not vague generalities
- **Actionable** - can be followed by future sessions
- **Durable** - will remain useful long-term
- **Non-one-off** - applies to more than just this moment
- **Helpful** - likely to improve future interactions

When in doubt, skip. The default answer is: no proposal needed.