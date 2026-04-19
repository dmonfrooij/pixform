# Resume Tomorrow (2026-04-19)

## Current State
- Runtime device preference: `cuda` (from `.pixform_device`)
- Selected installed model(s): `trellis2` only (from `.pixform_models.json`)
- Worktree has many local changes; do **not** reset blindly.

## Saved Backups
- `.copilot-backups/status_20260419_023142.txt`
- `.copilot-backups/working_20260419_023142.patch`
- `.copilot-backups/staged_20260419_023142.patch`
- Legacy app snapshot: `backend/app_old.py`

## Restore Exact Working State
Use this if anything gets overwritten before tomorrow.

```powershell
cd C:\Users\Eiboer\PycharmProjects\pixform
git apply --index .copilot-backups\staged_20260419_023142.patch
git apply .copilot-backups\working_20260419_023142.patch
git --no-pager status --short
```

## Continue Quickly Tomorrow
```powershell
cd C:\Users\Eiboer\PycharmProjects\pixform
.\PIXFORM.bat nvidia 4
```

## Last Known TRELLIS.2 Situation
- Loader compatibility patches were added in `backend/trellis2/models/__init__.py`.
- TRELLIS.2 still depends on checkpoint/runtime class compatibility for all required decoder classes.
- If startup fails again, capture and share only the final line:
  - `TRELLIS.2 failed to load: ...`

## Optional Safety Commit (Before New Edits)
```powershell
cd C:\Users\Eiboer\PycharmProjects\pixform
git checkout -b backup/trellis2-20260419
git add -A
git commit -m "WIP backup TRELLIS.2 session 2026-04-19"
```

