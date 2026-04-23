# Evolution Proposal: Diagnose-first workflow for training/backtest debugging requests

- Proposal-ID: evo-2026-04-20-agents-diagnose-first-before-editing
- Status: pending
- Signature: agents-diagnose-first-before-editing
- Created-At: 2026-04-20 14:00
- Target-File: AGENTS.md
- Trigger-Type: workflow
- Confidence: high

## Why This Matters
The user's established workflow for this autotrade project is: paste a training/backtest log → ask me to review → I enumerate **all** problems first without touching code → user picks which to fix via the `question` tool → then I modify code. This happened twice in a single session (the initial 7-item fix list, then the second round of 4 additional fixes). The user explicitly chose "先全部列出详细修复方案，不动代码" (list all detailed fix plans, don't modify code) on the first round. Documenting this workflow avoids wasting context on premature edits and keeps the user in control.

## Evidence
- First round, user's option selection included: **"先全部列出详细修复方案，不动代码"** (list all detailed fix plans, do not modify code)
- Then user responded: **"开始全部修复"** (start fixing all of them) only after seeing the plan
- Second round, same pattern: I diagnosed the val_loss=11 anomaly, used `question` to present 6 possible causes with recommendations, waited for user to select 4 items, then started fixing
- Twice in one session confirms this is a stable workflow, not a one-off preference

## Proposed Change
Append to `AGENTS.md`:

```markdown
## Workflow: Diagnose-first for ML training/backtest debugging

When the user pastes a training log, backtest log, or similar runtime output and asks "这个结果对吗 / 看看有什么问题" (is this result correct / what's wrong), follow this workflow:

1. **Read the relevant code first.** Use `read` on the entry script (`run_train.py` / `run_backtest.py` / `run_predict.py`) and its core module (`model/trainer.py`, `model/predictor.py`, `backtest/engine.py` etc.) before drawing any conclusions. Use `task` with an explore agent for unfamiliar codebases.

2. **Enumerate ALL issues, do not modify code yet.** Produce a prioritized list covering:
   - Immediate red flags in the log (loss not decreasing, NaN/Inf, early stopping)
   - Config vs actual-behavior mismatches (e.g. configured lr ≠ logged lr → resume pollution)
   - Silent bugs (equity_curve single-step multiplication, time_weight inconsistency)
   - Data/feature engineering risks (inf from pct_change on halted stocks, scaler clip loss)
   - Metric-comparison validity (train/val loss actually comparable?)

3. **Use the `question` tool with checkbox options** to let the user pick which issues to fix. Mark recommended ones with "(Recommended)". Do not assume; the user may intentionally want to defer some fixes.

4. **Only after explicit confirmation, modify code.** Use `todowrite` to track multi-file fixes.

5. **After each fix batch:**
   - Run `python -c "import ast; ast.parse(...)"` for syntax validation
   - Run a smoke import test (`from module import ...`) to catch runtime import errors
   - Run `python run_*.py --dry-run` if the script supports it

### Anti-patterns to avoid

- Do **not** start editing code on a log-review request without first presenting the diagnosis.
- Do **not** squash multiple unrelated fixes into one commit without listing them separately in the diagnosis.
- Do **not** assume "最佳模型保存了就是训练成功" (best model saved = training succeeded) — read the loss curve logic; EMA lag can make val_loss look like it's improving when it's just the EMA catching up to the online model.
```

## Apply Plan
1. Create `AGENTS.md` at project root (does not currently exist).
2. Add the content above as the first section.
3. Final result: `AGENTS.md` contains a single "Workflow: Diagnose-first for ML training/backtest debugging" section that future sessions can follow when the user pastes a run log.
