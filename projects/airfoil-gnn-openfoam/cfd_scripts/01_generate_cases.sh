#!/usr/bin/env bash
# ==============================================================================
# Generate OpenFOAM cases with correct velocity, drag, and lift directions
# (Portable: macOS + Linux)
# ==============================================================================
set -euo pipefail

# --- Resolve project paths (works no matter where you run it from) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# === User Configuration ===
U_LIST=(1 2 3 4 5 7 10)         # Velocities in m/s
A_LIST=(-4 0 2 4 6 8 10)        # Angles of attack in degrees
BASE="$ROOT_DIR/cfd/base_case"  # Base case directory (template)
OUTROOT="$ROOT_DIR/data/cases"  # Output directory for generated cases

# === Colored Output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${1}${2}${NC}"
}

# --- Cross-platform sed -i wrapper (GNU sed vs BSD sed) ---
sed_inplace() {
    # Usage: sed_inplace 's/a/b/g' file
    # or:    sed_inplace -e 's/a/b/g' -e 's/c/d/g' file
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

# === Validation ===
validate_setup() {
    print_status "$YELLOW" "[INFO] Validating setup..."

    [[ -d "$BASE" ]] || { print_status "$RED" "[ERROR] Base case '$BASE' not found."; exit 1; }

    local required_files=("0/U" "system/controlDict")
    for file in "${required_files[@]}"; do
        [[ -f "$BASE/$file" ]] || { print_status "$RED" "[ERROR] Missing: $BASE/$file"; exit 1; }
    done

    grep -q '__UX__\|__UY__' "$BASE/0/U" || {
        print_status "$RED" "[ERROR] Placeholders __UX__/__UY__ not found in 0/U"; exit 1; }

    grep -q '__U__\|__DRAGX__\|__DRAGY__\|__LIFTX__\|__LIFTY__' "$BASE/system/controlDict" || {
        print_status "$RED" "[ERROR] Force placeholder tags missing in controlDict"; exit 1; }

    mkdir -p "$OUTROOT"
    print_status "$GREEN" "[INFO] Validation passed ✓"
}

# === Generate one case ===
generate_case() {
    local U="$1"
    local A="$2"

    if [[ -z "${U:-}" || -z "${A:-}" ]]; then
        print_status "$RED" "[ERROR] generate_case: missing arguments"
        exit 1
    fi

    local CASE="U${U}_A${A}"
    local CASEDIR="${OUTROOT}/${CASE}"
    print_status "$YELLOW" "[INFO] Generating $CASE"

    mkdir -p "$CASEDIR"
    rsync -a "$BASE/" "$CASEDIR/"

    # AOA in radians (positive = CCW)
    local alpha_rad cosA sinA
    alpha_rad=$(awk "BEGIN{print ($A)*4*atan2(1,1)/180}")
    cosA=$(awk "BEGIN{printf \"%.8f\", cos($alpha_rad)}")
    sinA=$(awk "BEGIN{printf \"%.8f\", sin($alpha_rad)}")

    # Velocity (positive AOA = counter-clockwise rotation)
    local UX UY DRAGX DRAGY LIFTX LIFTY
    UX=$(awk "BEGIN{printf \"%.8f\", $U * $cosA}")
    UY=$(awk "BEGIN{printf \"%.8f\", $U * $sinA}")

    DRAGX="$cosA"
    DRAGY="$sinA"

    LIFTX=$(awk "BEGIN{printf \"%.8f\", -1 * $sinA}")
    LIFTY="$cosA"

    # Replace placeholders
    sed_inplace "s/__UX__/${UX}/g; s/__UY__/${UY}/g" "$CASEDIR/0/U"
    sed_inplace \
        -e "s/__U__/${U}/g" \
        -e "s/__DRAGX__/${DRAGX}/g" -e "s/__DRAGY__/${DRAGY}/g" \
        -e "s/__LIFTX__/${LIFTX}/g" -e "s/__LIFTY__/${LIFTY}/g" \
        "$CASEDIR/system/controlDict"

    # Case metadata
    cat > "$CASEDIR/case_info.txt" << EOF
Case: $CASE
Velocity magnitude: $U m/s
AOA: $A degrees (${alpha_rad} radians)
Velocity vector: (${UX}, ${UY}, 0)
Drag direction: (${DRAGX}, ${DRAGY}, 0)
Lift direction: (${LIFTX}, ${LIFTY}, 0)
Generated: $(date)
EOF

    print_status "$GREEN" "[INFO] ✓ Created $CASE"
}

# === Main Execution ===
main() {
    print_status "$GREEN" "====== OpenFOAM Case Generator ======"
    validate_setup

    local total=$(( ${#U_LIST[@]} * ${#A_LIST[@]} ))
    local count=0

    for U in "${U_LIST[@]}"; do
        for A in "${A_LIST[@]}"; do
            count=$((count + 1))
            print_status "$YELLOW" "[INFO] Progress: $count / $total"
            generate_case "$U" "$A"
        done
    done

    print_status "$GREEN" "[SUCCESS] $total cases created in '$OUTROOT'"
}

main "$@"