#!/bin/bash

# Exit on any error
set -e

# Default configuration
CONFIG_FILE="${CONFIG_FILE:-$HOME/devops-testing/config/validation_config.yaml}"
LOG_DIR="${CBW_LOG_DIR:-$HOME/devops-testing/reports}"
RETRY_COUNT=3
RETRY_DELAY=5
PARALLEL_JOBS=4

# Load configuration if yq is available
if command -v yq >/dev/null 2>&1 && [ -f "$CONFIG_FILE" ]; then
    RETRY_COUNT=$(yq eval '.retry_count // 3' "$CONFIG_FILE")
    RETRY_DELAY=$(yq eval '.retry_delay // 5' "$CONFIG_FILE")
    PARALLEL_JOBS=$(yq eval '.parallel_jobs // 4' "$CONFIG_FILE")
fi

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/validation_$(date +%Y%m%d_%H%M%S).log"

# Logging function with mutex
log() {
    flock -x "$LOG_FILE" echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Retry mechanism
retry_function() {
    local func=$1
    local description=$2
    local retry_count=0
    local success=false

    while [ $retry_count -lt $RETRY_COUNT ] && [ "$success" = false ]; do
        if $func; then
            success=true
            log "✓ $description succeeded on attempt $((retry_count + 1))"
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $RETRY_COUNT ]; then
                log "⚠ $description failed, retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi
    done

    if [ "$success" = false ]; then
        log "✗ $description failed after $RETRY_COUNT attempts"
        return 1
    fi
    return 0
}

# Directory structure validation
validate_directory_structure() {
    local required_dirs
    if command -v yq >/dev/null 2>&1 && [ -f "$CONFIG_FILE" ]; then
        required_dirs=($(yq eval '.required_directories[]' "$CONFIG_FILE"))
    else
        required_dirs=(
            "$HOME/devops-testing/tests"
            "$HOME/devops-testing/config"
            "$HOME/devops-testing/scripts"
            "$HOME/devops-testing/reports"
        )
    fi

    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            return 1
        fi
    done
    return 0
}

# Symlink verification with improved error handling
verify_symlinks() {
    local symlinks=()
    while IFS= read -r -d '' link; do
        symlinks+=("$link")
    done < <(find "$HOME/devops-testing" -type l -print0 2>/dev/null)

    for link in "${symlinks[@]}"; do
        target=$(readlink -f "$link" 2>/dev/null)
        if [ ! -e "$target" ]; then
            return 1
        fi
    done
    return 0
}

# Permission checks with configuration support
check_permissions() {
    local paths_to_check
    if command -v yq >/dev/null 2>&1 && [ -f "$CONFIG_FILE" ]; then
        while IFS= read -r line; do
            paths_to_check+=("$line")
        done < <(yq eval '.permissions | to_entries | .[] | .key + " " + .value' "$CONFIG_FILE")
    else
        paths_to_check=(
            "$HOME/devops-testing 755"
            "$HOME/devops-testing/config 750"
            "$HOME/devops-testing/reports 755"
        )
    fi

    for check in "${paths_to_check[@]}"; do
        read -r path perm <<< "$check"
        if [ ! -e "$path" ] || [ "$(stat -c "%a" "$path")" != "$perm" ]; then
            return 1
        fi
    done
    return 0
}

# Package management tests with version checking
test_package_management() {
    local required_packages
    if command -v yq >/dev/null 2>&1 && [ -f "$CONFIG_FILE" ]; then
        required_packages=($(yq eval '.required_packages[]' "$CONFIG_FILE"))
    else
        required_packages=(
            "curl"
            "git"
            "yq"
        )
    fi

    for pkg in "${required_packages[@]}"; do
        if ! command -v "$pkg" >/dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

# Security mechanism validation
validate_security() {
    # Check SSH configuration
    if [ -f "/etc/ssh/sshd_config" ]; then
        if ! grep -q "^PermitRootLogin no" "/etc/ssh/sshd_config"; then
            return 1
        fi
    fi

    # Check sensitive file permissions
    local sensitive_files=(
        "/etc/shadow:640"
        "/etc/passwd:644"
        "/etc/group:644"
    )

    for entry in "${sensitive_files[@]}"; do
        IFS=':' read -r file expected_perm <<< "$entry"
        if [ -f "$file" ]; then
            actual_perm=$(stat -c "%a" "$file")
            if [ "$actual_perm" != "$expected_perm" ]; then
                return 1
            fi
        fi
    done
    return 0
}

# Parallel test execution
run_tests_parallel() {
    local pids=()
    local results=()

    # Array of test functions and their descriptions
    declare -A tests=(
        ["validate_directory_structure"]="Directory structure validation"
        ["verify_symlinks"]="Symlink verification"
        ["check_permissions"]="Permission checks"
        ["test_package_management"]="Package management tests"
        ["validate_security"]="Security mechanism validation"
    )

    # Run tests in parallel with max jobs limit
    for test_func in "${!tests[@]}"; do
        while [ $(jobs -p | wc -l) -ge $PARALLEL_JOBS ]; do
            sleep 1
        done

        {
            if retry_function "$test_func" "${tests[$test_func]}"; then
                echo "$test_func:0"
            else
                echo "$test_func:1"
            fi
        } &
        pids+=($!)
    done

    # Wait for all tests to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# Main execution
main() {
    log "Starting system validation tests with parallel execution..."
    run_tests_parallel
    log "System validation tests completed"
}

main

