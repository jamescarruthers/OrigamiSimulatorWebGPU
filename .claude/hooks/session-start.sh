#!/bin/bash
# SessionStart hook for Claude Code on the web.
# Installs the toolchain needed by the WebGL->WebGPU migration regression
# harness (see tests/README.md): npm deps (Vite + Playwright) and a headless
# Chromium so `npm run test:regression` / `npm run regression:update` work.
set -euo pipefail

# Only run inside the Claude Code remote (web) environment; never touch a
# developer's local machine.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Resolve repo root (CLAUDE_PROJECT_DIR is set by the harness; fall back to the
# script location so the hook also works when run manually for validation).
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

# 1. Node dependencies. `npm install` (not `ci`) so the cached container layer
#    is reused across sessions and this stays idempotent.
npm install

# 2. Headless browser for the Playwright fold-regression harness.
#    Best-effort: this needs the environment's network policy to allow the
#    Playwright browser CDN (cdn.playwright.dev). If the policy blocks it (the
#    default curated allowlist does), we DON'T fail session startup — the rest
#    of the toolchain still works and the harness simply can't run until a
#    browser is available.
#
#    Fast pre-check: when the CDN host is blocked by the allowlist the proxy
#    returns a "Host not in allowlist" body. Detect that and skip the
#    (otherwise 6x-retried) download so session startup stays quick. Where the
#    CDN is reachable this probe passes and we install normally.
PW_CDN="${PLAYWRIGHT_DOWNLOAD_HOST:-https://cdn.playwright.dev}"
probe="$(curl -sS --max-time 8 "$PW_CDN" 2>/dev/null || true)"
if printf '%s' "$probe" | grep -qiE 'not in allowlist|blocked'; then
  echo "session-start: WARN Playwright CDN ($PW_CDN) is blocked by the network policy; skipping browser install." >&2
  echo "session-start: WARN The fold regression harness needs a browser. Allow cdn.playwright.dev in the" >&2
  echo "session-start: WARN environment's network policy, or run 'npm run regression:update' locally." >&2
elif timeout 180 bash -c 'npx playwright install --with-deps chromium 2>/dev/null || npx playwright install chromium'; then
  echo "session-start: Playwright Chromium ready."
else
  echo "session-start: WARN Playwright Chromium install did not complete; the fold regression harness may be unavailable." >&2
fi
