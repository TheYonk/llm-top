# Log View Feature Design

## Overview

Add `L` / `D` keyboard shortcuts to toggle between the dashboard view and a full-screen live log view for inference server containers.

## Decisions

- **Log target**: Auto-select first discovered inference container
- **Log behavior**: Live tail via `docker logs --tail N` each refresh cycle
- **Layout**: Full-screen logs replacing dashboard entirely
- **Approach**: Subprocess per refresh (Approach A) — matches existing architecture patterns

## Design

### View Mode State

Add a `view_mode` variable to the main loop: `"dashboard"` (default) or `"logs"`.

### Key Handling

- `q` — quit (unchanged)
- `l` / `L` — switch to logs view, latch onto first inference container
- `d` / `D` — switch back to dashboard view

### New Function: `get_container_logs(container_name, lines)`

- Runs `docker logs --tail <lines> <container_name>` via subprocess
- `lines` defaults to terminal height minus header/footer
- Returns list of strings

### New Function: `build_log_view(container_name, refresh_sec)`

- Header: `"llm-top — LOGS: <container_name>    HH:MM:SS"`
- Body: log lines filling the screen
- Footer: `"Press D for dashboard | q to quit"`
- Returns a `Table` (same type as `build_dashboard`)

### Main Loop Changes

```python
if view_mode == "dashboard":
    live.update(build_dashboard(...))
elif view_mode == "logs":
    live.update(build_log_view(log_container, ...))
```

### Edge Cases

- No inference containers found when `L` pressed → stay on dashboard
- Latched container disappears → fall back to dashboard

## TODO

- [x] Design approved
- [x] Add `get_container_logs()` function
- [x] Add `build_log_view()` function
- [x] Add view mode state and key handling to main loop
- [x] Update footer text on dashboard to show L/D keys
- [x] Syntax verified

## Implementation Notes

- `get_container_logs()` added at line ~377, uses `stderr=subprocess.STDOUT` to capture both stdout and stderr from docker logs
- `build_log_view()` added at line ~791, uses `shutil.get_terminal_size()` to fill the screen with log lines
- Key handling in the sleep loop includes `break` to trigger immediate re-render on view switch
- Dashboard footer updated from "Press q to quit" to "L logs | q quit"
- Log view footer shows "D dashboard | q quit"
