# Evolution Proposal: Fall back to byte-level Python script when Edit fails on mojibake/CRLF files

- Proposal-ID: evo-2026-04-20-tools-edit-fallback-byte-script
- Status: pending
- Signature: tools-edit-fallback-byte-script
- Created-At: 2026-04-20 14:00
- Target-File: TOOLS.md
- Trigger-Type: struggle
- Confidence: high

## Why This Matters
In this session, editing `model/trainer.py` repeatedly failed because the file is UTF-8-declared but contains GBK-mojibake comments and uses CRLF line endings. The `edit` tool's `oldString` whitespace/encoding match failed 5+ times. Each failure wasted a tool round-trip. The working solution was to write a small throwaway Python script that does exact byte-level `str.find`/`str.replace` on the file contents and write back in binary mode. Recording this fallback saves future sessions from re-discovering the pattern.

## Evidence
- Multiple `edit` failures in sequence on `model/trainer.py`:
  - `Could not find oldString in the file. It must match exactly, including whitespace, indentation, and line endings.` (x5)
- Eventually solved by writing `__fix_trainer.py`, `__fix_trainer2.py`, `__fix_val_print.py`, `__fix_retnan.py` which all open the file in `'rb'` mode, do byte-slice/replace, write back in `'wb'` mode.
- Root cause confirmed: file has GBK-mojibake bytes (e.g., `\xe9\x94\x9f?\xe6\xa3\xb0...`) mixed into what the editor treats as UTF-8, plus `\r\n` line endings; this makes the `edit` tool's string matching brittle.

## Proposed Change
Append to `TOOLS.md`:

```markdown
## Edit tool: fallback for mojibake / CRLF files

When `edit` repeatedly fails with "Could not find oldString" on the same file, **stop retrying after 2 attempts** and switch to a byte-level Python script instead. Typical root causes:

1. File was saved as GBK/GB2312 but declared `# -*- coding: utf-8 -*-` — comments appear as mojibake in UTF-8 view; the exact bytes shown by `read` may not round-trip through `edit`'s string matching.
2. Windows CRLF line endings (`\r\n`) mixed with LF matches in the edit payload.
3. Whitespace at end-of-line (trailing spaces) not visible in the read output.

### Fallback pattern

Write a throwaway `__fix_<topic>.py` script:

```python
with open("path/to/file.py", "rb") as f:
    data = f.read()

start_marker = b"<unique byte prefix>"  # use `grep` first to confirm uniqueness
end_marker   = b"<unique byte suffix>"
si = data.find(start_marker)
ei = data.find(end_marker, si)
assert si >= 0 and ei > si, f"markers not found: si={si}, ei={ei}"

new_block = (
    b"new content line 1\r\n"
    b"new content line 2\r\n"
)
new_data = data[:si] + new_block + data[ei:]
with open("path/to/file.py", "wb") as f:
    f.write(new_data)
```

Then run `python __fix_<topic>.py`, run `ast.parse` to verify syntax, and delete the script.

**Tips:**
- Use `grep` first on a unique substring to confirm the marker appears exactly once.
- Preserve `\r\n` line endings explicitly in byte literals.
- After the fix, always run `python -c "import ast; ast.parse(open('file.py','rb').read().decode('utf-8','replace'))"` to validate.
- Clean up the `__fix_*.py` files at the end of the task.
```

## Apply Plan
1. Create `TOOLS.md` at project root (does not currently exist).
2. Add the content above as the first (and currently only) section.
3. Final result: `TOOLS.md` with a single "Edit tool: fallback for mojibake / CRLF files" section that future sessions can reference before wrestling with `edit`.
