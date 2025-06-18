#!/bin/bash

# Configuration
MONITOR_INTERVAL=60  # Seconds between checks
LOG_DIR="${HOME}/devops-testing/reports/monitoring"
ERROR_LOG="${LOG_DIR}/errors.log"
ALERT_LOG="${LOG_DIR}/alerts.log"
CONFIG_FILE="${HOME}/devops-testing/config/monitor_config.yaml"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Initialize log files
touch "$ERROR_LOG" "$ALERT_LOG"

log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$ERROR_LOG"
}

monitor_permissions() {
    while IFS= read -r line; do
        path=$(echo "$line" | yq eval '.path' -)
        expected_perm=$(echo "$line" | yq eval '.permission' -)
        actual_perm=$(stat -c "%a" "$path" 2>/dev/null)
        
        if [ "$actual_perm" != "$expected_perm" ]; then
            log "WARNING" "Permission mismatch: $path (expected: $expected_perm, got: $actual_perm)"
            echo "{\"type\": \"permission\", \"path\": \"$path\", \"expected\": \"$expected_perm\", \"actual\": \"$actual_perm\"}" >> "$ALERT_LOG"
        fi
    done < <(yq eval '.permissions[]' "$CONFIG_FILE")
}

monitor_configuration() {
    local config_dir="$HOME/devops-testing/config"
    local config_hash_file="$LOG_DIR/config_hash"
    local current_hash

    current_hash=$(find "$config_dir" -type f -exec sha256sum {} \; | sort | sha256sum)

    if [ -f "$config_hash_file" ]; then
        local stored_hash
        stored_hash=$(cat "$config_hash_file")
        
        if [ "$current_hash" != "$stored_hash" ]; then
            log "INFO" "Configuration changes detected"
            echo "{\"type\": \"config_change\", \"timestamp\": \"$(date -Iseconds)\"}" >> "$ALERT_LOG"
        fi
    fi

    echo "$current_hash" > "$config_hash_file"
}

monitor_security() {
    # Check for failed SSH attempts
    if [ -f "/var/log/auth.log" ]; then
        grep "Failed password" /var/log/auth.log | tail -n 5 | while read -r line; do
            echo "{\"type\": \"security\", \"event\": \"failed_login\", \"details\": \"$line\"}" >> "$ALERT_LOG"
        done
    fi

    # Check for modified system files
    while IFS= read -r file; do
        if [ ! -f "$file.sha256" ]; then
            sha256sum "$file" > "$file.sha256"
        else
            if ! sha256sum -c "$file.sha256" >/dev/null 2>&1; then
                log "ALERT" "System file modified: $file"
                echo "{\"type\": \"security\", \"event\": \"file_modified\", \"file\": \"$file\"}" >> "$ALERT_LOG"
            fi
        fi
    done < <(yq eval '.system_files[]' "$CONFIG_FILE")
}

monitor_packages() {
    # Check for available updates
    if ! apt list --upgradable 2>/dev/null | grep -q "^Listing..."; then
        local updates
        updates=$(apt list --upgradable 2>/dev/null | grep -v "^Listing...")
        if [ -n "$updates" ]; then
            echo "{\"type\": \"package\", \"event\": \"updates_available\", \"details\": \"$updates\"}" >> "$ALERT_LOG"
        fi
    fi

    # Check installed packages against required list
    while IFS= read -r pkg; do
        if ! dpkg -l "$pkg" >/dev/null 2>&1; then
            log "WARNING" "Required package missing: $pkg"
            echo "{\"type\": \"package\", \"event\": \"missing_package\", \"package\": \"$pkg\"}" >> "$ALERT_LOG"
        fi
    done < <(yq eval '.required_packages[]' "$CONFIG_FILE")
}

process_alerts() {
    if [ -s "$ALERT_LOG" ]; then
        # Convert alerts to structured format
        jq -s '.' "$ALERT_LOG" > "${ALERT_LOG}.json"
        
        # Trigger AI agent for resolution
        if [ -f "${HOME}/devops-testing/scripts/ai_resolver.py" ]; then
            python3 "${HOME}/devops-testing/scripts/ai_resolver.py" "${ALERT_LOG}.json"
        fi
        
        # Archive processed alerts
        local timestamp=$(date +%Y%m%d_%H%M%S)
        mv "$ALERT_LOG" "${ALERT_LOG}.${timestamp}"
        touch "$ALERT_LOG"
    fi
}

main() {
    log "INFO" "Starting system monitoring"
    
    while true; do
        monitor_permissions
        monitor_configuration
        monitor_security
        monitor_packages
        process_alerts
        sleep "$MONITOR_INTERVAL"
    done
}

main

