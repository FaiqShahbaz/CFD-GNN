#!/usr/bin/env bash
# ==============================================================================
# Run OpenFOAM cases in parallel and convert to VTU/VTP files in structured folders
# Portable: macOS + Linux
# ==============================================================================
set -euo pipefail

# --- Resolve project paths (works no matter where you run it from) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# === Configuration ===
MAX_JOBS="${MAX_JOBS:-6}"
CASEDIR="$ROOT_DIR/data/cases"
VTUDIR="$ROOT_DIR/data/vtu"
LOGDIR="$ROOT_DIR/data/logs"

# === Color Output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
  echo -e "${1}${2}${NC}"
}

# === Validation ===
validate_setup() {
  print_status "$YELLOW" "[INFO] Validating setup..."

  command -v simpleFoam >/dev/null || { print_status "$RED" "[ERROR] OpenFOAM not sourced (simpleFoam not found)"; exit 1; }
  command -v foamToVTK  >/dev/null || { print_status "$RED" "[ERROR] foamToVTK not found in PATH"; exit 1; }
  [[ -d "$CASEDIR" ]]   || { print_status "$RED" "[ERROR] '$CASEDIR' directory not found"; exit 1; }

  mkdir -p "$VTUDIR" "$LOGDIR"

  if command -v parallel >/dev/null 2>&1; then
    print_status "$GREEN" "[INFO] GNU parallel detected ✓"
  else
    print_status "$YELLOW" "[WARN] GNU parallel not installed; will use xargs -P fallback"
  fi

  print_status "$GREEN" "[INFO] Validation passed ✓"
}

# === Run a single OpenFOAM case ===
run_case() {
  local CASE="$1"
  local CASEPATH="$CASEDIR/$CASE"
  local LOGFILE="$CASEPATH/log.simpleFoam"

  print_status "$BLUE" "[INFO] Running $CASE..."
  if simpleFoam -case "$CASEPATH" > "$LOGFILE" 2>&1; then
    print_status "$GREEN" "[INFO] ✓ Completed $CASE"
    return 0
  else
    print_status "$RED" "[ERROR] ✗ Failed $CASE (see $LOGFILE)"
    return 1
  fi
}

# === Convert OpenFOAM output to VTU/VTP ===
convert_to_vtu() {
  local CASE="$1"
  local CASEPATH="$CASEDIR/$CASE"
  local LOGFILE="$CASEPATH/log.foamToVTK"
  local VTUCASE="$VTUDIR/$CASE"
  local VTKDIR="$CASEPATH/VTK"

  mkdir -p "$VTUCASE"
  print_status "$BLUE" "[INFO] Converting $CASE to VTU..."

  if foamToVTK -case "$CASEPATH" -latestTime -fields '(U p)' > "$LOGFILE" 2>&1; then
    # Find most recent VTK subfolder (e.g., VTK/U10_A0_1000/)
    local SUBDIR=""
    if [[ -d "$VTKDIR" ]]; then
      # Portable way: list directories, sort, take last
      SUBDIR="$(find "$VTKDIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
    fi

    if [[ -n "$SUBDIR" && -d "$SUBDIR" ]]; then
      # Copy any .vtu/.vtp files without relying on brace globbing
      local copied_any=false

      if find "$SUBDIR" -maxdepth 1 -type f -name '*.vtu' -print -quit | grep -q .; then
        cp "$SUBDIR"/*.vtu "$VTUCASE/" 2>/dev/null || true
        copied_any=true
      fi
      if find "$SUBDIR" -maxdepth 1 -type f -name '*.vtp' -print -quit | grep -q .; then
        cp "$SUBDIR"/*.vtp "$VTUCASE/" 2>/dev/null || true
        copied_any=true
      fi

      # Verify output contains at least one vtu or vtp
      if find "$VTUCASE" -maxdepth 1 -type f \( -name '*.vtu' -o -name '*.vtp' \) -print -quit | grep -q .; then
        [[ -f "$CASEPATH/case_info.txt" ]] && cp "$CASEPATH/case_info.txt" "$VTUCASE/"

        cat > "$VTUCASE/conversion_info.txt" <<EOF
Case: $CASE
Converted: $(date)
Source: $SUBDIR
EOF

        print_status "$GREEN" "[INFO] ✓ VTU/VTP files exported from $SUBDIR"
        return 0
      else
        print_status "$RED" "[ERROR] ✗ No VTU/VTP files found in $SUBDIR"
        return 1
      fi
    else
      print_status "$RED" "[ERROR] ✗ No subdirectory found in $VTKDIR"
      return 1
    fi
  else
    print_status "$RED" "[ERROR] ✗ foamToVTK failed for $CASE"
    tail -n 5 "$LOGFILE" || true
    return 1
  fi
}

# === Process each case ===
process_case() {
  local CASE="$1"
  run_case "$CASE" && convert_to_vtu "$CASE"
}

# === Main Script ===
main() {
  print_status "$GREEN" "====== OpenFOAM Runner + VTU Export ======"
  validate_setup

  # Portable case listing (no GNU find -printf)
  local cases
  cases="$(find "$CASEDIR" -maxdepth 1 -type d -name 'U*_A*' -exec basename {} \; | sort)"

  local total
  total="$(echo "$cases" | awk 'NF{c++} END{print c+0}')"

  print_status "$YELLOW" "[INFO] Found $total cases"
  print_status "$YELLOW" "[INFO] Running with up to $MAX_JOBS parallel jobs"

  local start_time end_time
  start_time="$(date +%s)"

  if command -v parallel >/dev/null 2>&1; then
    # Export functions and vars for GNU parallel
    export -f process_case run_case convert_to_vtu print_status
    export CASEDIR VTUDIR LOGDIR RED GREEN YELLOW BLUE NC
    echo "$cases" | parallel -j "$MAX_JOBS" process_case {}
  else
    # Fallback: xargs parallelism (portable)
    export CASEDIR VTUDIR LOGDIR
    echo "$cases" | xargs -I{} -P "$MAX_JOBS" bash -c 'process_case "$@"' _ {}
  fi

  end_time="$(date +%s)"
  local duration=$((end_time - start_time))

  local vtu_count
  vtu_count="$(find "$VTUDIR" -type f \( -name '*.vtu' -o -name '*.vtp' \) 2>/dev/null | wc -l | tr -d ' ')"

  print_status "$GREEN" "[INFO] All tasks completed in ${duration}s"
  print_status "$YELLOW" "[INFO] Total VTU/VTP files: $vtu_count"
  print_status "$GREEN" "[DONE] See '$VTUDIR/<case>/' for outputs"
}

main "$@"