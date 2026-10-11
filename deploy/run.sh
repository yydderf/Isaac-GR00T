#!/usr/bin/env bash
# deploy/run.sh: auto-select platform, overridable with PLATFORM=...

if [ -z "$PLATFORM" ]; then
  if [ -f /etc/nv_tegra_release ]; then PLATFORM=edge; else PLATFORM=workstation; fi
fi

exec "$(dirname "$0")/$PLATFORM/run_container.sh" "$@"

