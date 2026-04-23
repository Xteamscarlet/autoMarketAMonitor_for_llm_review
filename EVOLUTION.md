# Evolution Configuration

This file controls the Hermes Evolution intensity for this OpenCode project.

## Intensity Level

<!-- Change this value to adjust how aggressively evolution is triggered -->
<!-- 100% = aggressive (strong + medium signals) -->
<!-- 50% = cautious (strong signals only) -->
<!-- 0% = disabled -->

**Current: 100% (aggressive)**

## What This Means

- **100%**: Any strong signal OR two medium signals will trigger an evolution proposal
- **50%**: Only strong signals (explicit corrections, "always/never", taught workflows) will trigger
- **0%**: No evolution proposals will be created

## Target Files

Evolution proposals can write to:

| File | Purpose |
|------|---------|
| `AGENTS.md` | Behavior rules, workflow patterns, safety boundaries |
| `MEMORY.md` | Stable preferences, long-term facts |
| `TOOLS.md` | Tool pitfalls, environment experience, command notes |

## Draft Workflow

1. Draft created in `.opencode/evolution-drafts/pending/`
2. User reviews and approves/rejects/revises
3. Approved drafts are applied to target files and moved to `approved/`
4. Rejected drafts are moved to `rejected/`

## How to Change

Edit the intensity level above. For example, to switch to cautious mode:

```markdown
**Current: 50% (cautious)**
```