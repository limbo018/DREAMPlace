#!/usr/bin/env bash
# =============================================================================
# run_mac.sh — DREAMPlace Docker runner for macOS Apple Silicon (M-series)
# =============================================================================
#
# USAGE
#   ./run_mac.sh <command> [args]
#
# COMMANDS
#   build-image          Build (or rebuild) the Docker image
#   install              CMake-configure, compile, and install DREAMPlace
#   reinstall            Clean build artifacts, then install fresh
#   run <config.json>    Run a full placement from the install directory
#   shell                Open an interactive shell inside the container
#   clean                Remove build/ and install/ directories
#   nuke                 clean + remove the Docker image
#   status               Show current environment state
#   help                 Show this message
#
# TYPICAL FIRST-TIME FLOW
#   ./run_mac.sh build-image      # ~10-20 min (downloads deps)
#   ./run_mac.sh install          # ~10-30 min (compiles DREAMPlace CPU-only)
#   ./run_mac.sh run test/ispd2005/adaptec1.json
#
# REINSTALL FLOW (wipes compiled artifacts, keeps Docker image)
#   ./run_mac.sh reinstall
#
# FULL RESET (removes everything including the Docker image)
#   ./run_mac.sh nuke && ./run_mac.sh build-image && ./run_mac.sh install
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="dreamplace-mac"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

DOCKERFILE="${SCRIPT_DIR}/Dockerfile.mac"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

# Container path where the host repo is mounted
CONTAINER_ROOT="/DREAMPlace"

# Target platform: Apple M-series runs linux/arm64 natively in Docker Desktop
PLATFORM="linux/arm64"

# ─── Colors & Logging ─────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
  C_RED='\033[0;31m'
  C_GREEN='\033[0;32m'
  C_YELLOW='\033[1;33m'
  C_BLUE='\033[0;34m'
  C_CYAN='\033[0;36m'
  C_BOLD='\033[1m'
  C_RESET='\033[0m'
else
  C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_CYAN=''; C_BOLD=''; C_RESET=''
fi

log_info()    { echo -e "${C_BLUE}[INFO]${C_RESET}  $*"; }
log_ok()      { echo -e "${C_GREEN}[OK]${C_RESET}    $*"; }
log_warn()    { echo -e "${C_YELLOW}[WARN]${C_RESET}  $*"; }
log_error()   { echo -e "${C_RED}[ERROR]${C_RESET} $*" >&2; }
log_step()    { echo -e "${C_CYAN}${C_BOLD}  ==> $*${C_RESET}"; }
log_section() { echo -e "\n${C_BOLD}${C_BLUE}=== $* ===${C_RESET}\n"; }
log_divider() { echo -e "${C_BLUE}─────────────────────────────────────────────${C_RESET}"; }

# ─── Prerequisite Checks ──────────────────────────────────────────────────────

check_docker() {
  if ! command -v docker &>/dev/null; then
    log_error "Docker not found in PATH."
    log_error "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
  fi
  if ! docker info &>/dev/null 2>&1; then
    log_error "Docker daemon is not running. Please start Docker Desktop."
    exit 1
  fi
}

# Ensure all git submodules (Limbo, pybind11, OpenTimer, etc.) are present.
# This runs on the HOST so that git credentials and network are available.
check_submodules() {
  log_step "Checking git submodules..."
  cd "${SCRIPT_DIR}"

  # Check for actual file content rather than git registration status.
  # git submodule status only shows '-' for *unregistered* submodules; it
  # passes silently when a directory is registered but was never checked out
  # (empty directory, partial clone, previous git submodule init without update).
  # pybind11/CMakeLists.txt is the earliest thing CMake needs — its absence
  # causes every subsequent CMake error in the build.
  if [[ ! -f "${SCRIPT_DIR}/thirdparty/pybind11/CMakeLists.txt" ]]; then
    log_warn "Submodule content is missing — running 'git submodule update --init --recursive'."
    log_warn "This clones several repos and may take a few minutes on first run."
    git submodule update --init --recursive
    log_ok "Submodules initialized."
  else
    log_ok "All git submodules are present."
  fi
}

# Auto-build the image if it doesn't exist yet.
ensure_image() {
  if ! docker image inspect "${FULL_IMAGE}" &>/dev/null 2>&1; then
    log_warn "Docker image '${FULL_IMAGE}' not found — building it now."
    cmd_build_image
  fi
}

# ─── Docker Run Helper ────────────────────────────────────────────────────────
# Mounts the repo directory into the container, removes container on exit.

docker_run() {
  docker run --rm \
    --platform "${PLATFORM}" \
    -v "${SCRIPT_DIR}:${CONTAINER_ROOT}" \
    -w "${CONTAINER_ROOT}" \
    "${FULL_IMAGE}" \
    "$@"
}

# Interactive version (with -it for TTY)
docker_run_interactive() {
  docker run --rm -it \
    --platform "${PLATFORM}" \
    -v "${SCRIPT_DIR}:${CONTAINER_ROOT}" \
    -w "${CONTAINER_ROOT}" \
    "${FULL_IMAGE}" \
    "$@"
}

# ─── Commands ─────────────────────────────────────────────────────────────────

cmd_build_image() {
  log_section "Building Docker Image"
  check_docker

  local extra_flags=()
  if [[ "${1:-}" == "--no-cache" ]]; then
    extra_flags+=("--no-cache")
    log_info "Cache disabled — performing a fully fresh image build."
  fi

  log_info "Image: ${FULL_IMAGE}"
  log_info "Platform: ${PLATFORM} (Apple Silicon native)"
  log_info "Dockerfile: ${DOCKERFILE}"
  log_divider

  docker build \
    --platform "${PLATFORM}" \
    "${extra_flags[@]+"${extra_flags[@]}"}" \
    -f "${DOCKERFILE}" \
    -t "${FULL_IMAGE}" \
    "${SCRIPT_DIR}"

  log_divider
  log_ok "Image '${FULL_IMAGE}' is ready."
}

# ---------------------------------------------------------------------------

cmd_clean() {
  log_section "Cleaning Build Artifacts"
  local cleaned=0

  if [[ -d "${BUILD_DIR}" ]]; then
    log_step "Removing build/ ..."
    rm -rf "${BUILD_DIR}"
    cleaned=1
  fi

  if [[ -d "${INSTALL_DIR}" ]]; then
    log_step "Removing install/ ..."
    rm -rf "${INSTALL_DIR}"
    cleaned=1
  fi

  if [[ "${cleaned}" -eq 0 ]]; then
    log_info "Nothing to clean (build/ and install/ do not exist)."
  else
    log_ok "Build artifacts removed."
  fi
}

# ---------------------------------------------------------------------------

cmd_nuke() {
  log_section "Full Reset (nuke)"
  cmd_clean

  check_docker
  if docker image inspect "${FULL_IMAGE}" &>/dev/null 2>&1; then
    log_step "Removing Docker image '${FULL_IMAGE}' ..."
    docker image rm "${FULL_IMAGE}"
    log_ok "Docker image removed."
  else
    log_info "Docker image '${FULL_IMAGE}' not found — skipping."
  fi

  log_ok "Full reset complete. Run './run_mac.sh build-image && ./run_mac.sh install' to start fresh."
}

# ---------------------------------------------------------------------------

cmd_install() {
  log_section "Installing DREAMPlace (CPU-only)"
  check_docker
  check_submodules
  ensure_image

  log_info "Repo:    ${SCRIPT_DIR}"
  log_info "Install: ${INSTALL_DIR}/"
  log_divider

  # nproc is evaluated inside the container (reflects Docker CPU allocation)
  docker_run bash -c "
    set -e

    # git 2.35.2+ refuses to operate on directories owned by a different user
    # (macOS host mount appears as a different UID to the container root).
    git config --global --add safe.directory '${CONTAINER_ROOT}'

    # Initialize submodules INSIDE the container.
    # The host-side check can silently return 0 without cloning anything when
    # submodule dirs already exist (even empty) due to a prior 'git submodule
    # init' without a matching 'update'.  Running inside Docker guarantees a
    # clean git environment and confirmed internet access.
    echo '==> Initializing git submodules...'
    if [[ ! -f thirdparty/pybind11/CMakeLists.txt ]]; then
      git submodule update --init --recursive --jobs 4
    else
      echo '    Already present, skipping.'
    fi

    # Hard-stop with an actionable message if cloning still failed.
    if [[ ! -f thirdparty/pybind11/CMakeLists.txt ]]; then
      echo ''
      echo 'ERROR: thirdparty/pybind11/CMakeLists.txt is still missing.'
      echo 'Possible causes:'
      echo '  1. No internet access inside Docker (check Docker Desktop network settings).'
      echo '  2. The repo was downloaded as a ZIP — re-clone with:'
      echo '     git clone --recursive https://github.com/limbo018/DREAMPlace.git'
      exit 1
    fi

    echo ''
    echo '==> Detecting PyTorch C++ ABI...'
    # FIX: auto-detect _GLIBCXX_USE_CXX11_ABI from the installed PyTorch.
    # PyTorch 2.x on ARM64 Ubuntu 22.04 uses ABI=1 (new ABI). The CMakeLists
    # default is 0 (old ABI), which causes undefined-symbol linker errors when
    # the ABI is mismatched. We query torch directly to get the right value.
    TORCH_CXX_ABI=\$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
    echo \"    TORCH_CXX_ABI=\${TORCH_CXX_ABI}\"

    echo ''
    echo '==> Configuring with CMake (CPU-only)...'
    mkdir -p build && cd build
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=${CONTAINER_ROOT}/install \
      -DCMAKE_CXX_ABI=\${TORCH_CXX_ABI} \
      -DPython_EXECUTABLE=\$(which python3)

    echo ''
    echo '==> Compiling (using \$(nproc) cores)...'
    make -j\$(nproc)

    echo ''
    echo '==> Installing to ${CONTAINER_ROOT}/install ...'
    make install

    echo ''
    echo 'Build complete!'
  "

  log_divider
  log_ok "DREAMPlace installed to: ${INSTALL_DIR}/"
  log_info "Run a benchmark with:"
  log_info "  ./run_mac.sh run test/ispd2005/adaptec1.json"
}

# ---------------------------------------------------------------------------

cmd_reinstall() {
  log_section "Reinstalling DREAMPlace"
  log_info "This will wipe build/ and install/, then rebuild from source."
  cmd_clean
  cmd_install
}

# ---------------------------------------------------------------------------

cmd_run() {
  local config="${1:-}"
  if [[ -z "${config}" ]]; then
    log_error "No JSON config file specified."
    log_error "Usage:   ./run_mac.sh run <path/to/config.json>"
    log_error "Example: ./run_mac.sh run test/ispd2005/adaptec1.json"
    log_error ""
    log_error "Available benchmark configs (relative to install/):"
    if [[ -d "${INSTALL_DIR}/test" ]]; then
      find "${INSTALL_DIR}/test" -name "*.json" | sed "s|${INSTALL_DIR}/||" | sort | head -20
    fi
    exit 1
  fi

  check_docker
  ensure_image

  if [[ ! -d "${INSTALL_DIR}" ]]; then
    log_error "DREAMPlace is not installed yet. Run './run_mac.sh install' first."
    exit 1
  fi

  log_section "Running DREAMPlace (CPU-only)"
  log_info "Config: ${config}"
  log_info "Running from: ${INSTALL_DIR}/"
  log_divider

  # FIX: All shipped test configs contain "gpu": 1. On a CPU-only build,
  # Placer.py asserts CUDA_FOUND == 'TRUE' when gpu=1 and immediately aborts.
  # We create a temporary copy of the config with gpu forced to 0 so the user
  # does not have to manually edit every benchmark JSON file.
  docker_run bash -c "
    set -e
    cd ${CONTAINER_ROOT}/install

    # Build a temporary CPU-only config (gpu=0) from the user-supplied file.
    TMP_CFG=\$(mktemp /tmp/dreamplace_cpu_XXXXXX.json)
    python3 - \"\$TMP_CFG\" <<'PYEOF'
import json, sys

cfg_path = '${config}'
tmp_path = sys.argv[1]

with open(cfg_path) as f:
    cfg = json.load(f)

if cfg.get('gpu', 0):
    print('[INFO] CPU-only mode: overriding gpu=1 -> gpu=0 in config')

cfg['gpu'] = 0

with open(tmp_path, 'w') as f:
    json.dump(cfg, f, indent=2)
PYEOF

    python3 dreamplace/Placer.py \"\$TMP_CFG\"
    rm -f \"\$TMP_CFG\"
  "
}

# ---------------------------------------------------------------------------

cmd_shell() {
  log_section "Opening Interactive Shell"
  check_docker
  ensure_image

  log_info "Container: ${FULL_IMAGE}"
  log_info "Repo mounted at: ${CONTAINER_ROOT}"
  log_info "Type 'exit' or press Ctrl-D to leave."
  log_divider

  docker_run_interactive bash
}

# ---------------------------------------------------------------------------

cmd_status() {
  log_section "DREAMPlace Environment Status"
  check_docker

  # Docker image
  log_step "Docker image"
  if docker image inspect "${FULL_IMAGE}" &>/dev/null 2>&1; then
    local created size
    created=$(docker image inspect "${FULL_IMAGE}" --format '{{.Created}}' | cut -dT -f1)
    size=$(docker image inspect "${FULL_IMAGE}" --format '{{.Size}}' | awk '{printf "%.0f MB", $1/1024/1024}')
    log_ok "${FULL_IMAGE}  (created ${created}, ${size})"
  else
    log_warn "${FULL_IMAGE} — not built yet  (run './run_mac.sh build-image')"
  fi

  # Build directory
  log_step "Build directory"
  if [[ -d "${BUILD_DIR}" ]]; then
    log_ok "build/  exists"
  else
    log_warn "build/  does not exist"
  fi

  # Install directory
  log_step "Install directory"
  if [[ -d "${INSTALL_DIR}" ]]; then
    if [[ -f "${INSTALL_DIR}/dreamplace/Placer.py" ]]; then
      log_ok "install/  exists with Placer.py"
    else
      log_warn "install/  exists but Placer.py not found (incomplete install?)"
    fi
  else
    log_warn "install/  does not exist  (run './run_mac.sh install')"
  fi

  # Submodule status
  log_step "Git submodules"
  cd "${SCRIPT_DIR}"
  if git submodule status 2>/dev/null | grep -q '^-'; then
    log_warn "Some submodules are NOT initialized"
  else
    log_ok "All submodules present"
  fi
}

# ─── Help ─────────────────────────────────────────────────────────────────────

show_help() {
  echo -e "${C_BOLD}${C_BLUE}DREAMPlace Mac Runner${C_RESET}"
  echo -e "Docker-based build & run for Apple Silicon (M-series, CPU-only)"
  echo ""
  echo -e "${C_BOLD}USAGE${C_RESET}"
  echo "  ./run_mac.sh <command> [args]"
  echo ""
  echo -e "${C_BOLD}COMMANDS${C_RESET}"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "build-image"         "Build the Docker image (ARM64, CPU-only)"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "build-image --no-cache" "Force a completely fresh image build"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "install"             "CMake + compile + install DREAMPlace"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "reinstall"           "Remove build artifacts, then install fresh"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "run <config.json>"   "Run placement from install directory"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "shell"               "Open interactive shell inside container"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "clean"               "Remove build/ and install/ directories"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "nuke"                "clean + remove the Docker image"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "status"              "Show Docker image and install state"
  printf "  ${C_CYAN}%-22s${C_RESET} %s\n" "help"                "Show this message"
  echo ""
  echo -e "${C_BOLD}FIRST-TIME SETUP${C_RESET}"
  echo "  ./run_mac.sh build-image          # build Docker image (~10-20 min)"
  echo "  ./run_mac.sh install              # compile DREAMPlace (~10-30 min)"
  echo "  ./run_mac.sh run test/ispd2005/adaptec1.json"
  echo ""
  echo -e "${C_BOLD}REINSTALL (keep Docker image, rebuild DREAMPlace)${C_RESET}"
  echo "  ./run_mac.sh reinstall"
  echo ""
  echo -e "${C_BOLD}FULL RESET${C_RESET}"
  echo "  ./run_mac.sh nuke"
  echo "  ./run_mac.sh build-image && ./run_mac.sh install"
  echo ""
  echo -e "${C_BOLD}NOTES${C_RESET}"
  echo "  - Requires Docker Desktop with Apple Silicon support."
  echo "  - CPU-only execution (CUDA / NVIDIA GPU not supported on Apple Silicon)."
  echo "  - Multi-threaded via OpenMP; Docker CPU allocation controls thread count."
  echo "  - 'run' auto-patches gpu=0 in the config — no manual JSON edits needed."
  echo "  - Docker image: ${FULL_IMAGE}"
  echo "  - Install dir:  ./install/"
  echo "  - Build dir:    ./build/"
}

# ─── Main Dispatcher ──────────────────────────────────────────────────────────

main() {
  local cmd="${1:-help}"
  shift || true

  case "${cmd}" in
    build-image) cmd_build_image "$@" ;;
    install)     cmd_install     "$@" ;;
    reinstall)   cmd_reinstall   "$@" ;;
    run)         cmd_run         "$@" ;;
    shell)       cmd_shell              ;;
    clean)       cmd_clean       "$@" ;;
    nuke)        cmd_nuke        "$@" ;;
    status)      cmd_status             ;;
    help|--help|-h) show_help          ;;
    *)
      log_error "Unknown command: '${cmd}'"
      echo ""
      show_help
      exit 1
      ;;
  esac
}

main "$@"
