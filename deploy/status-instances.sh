#!/bin/bash

# Status Script for OSINT Instances
# This script shows the status of all instances

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIDS_FILE="$SCRIPT_DIR/pids.txt"
LOGS_DIR="$SCRIPT_DIR/logs"

# Hardcoded instance configuration
INSTANCE_PORTS=(12410 12411 12412)
INSTANCE_NAMES=("OSINT_Instance_1" "OSINT_Instance_2" "OSINT_Instance_3")
DB_NAMES=("osint_db1" "osint_db2" "osint_db3")

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

get_pid_for_port() {
    local port=$1
    lsof -t -i:$port 2>/dev/null || echo "N/A"
}

check_health() {
    local port=$1
    if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
        return 0  # Healthy
    else
        return 1  # Not healthy
    fi
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║             OSINT INSTANCES STATUS REPORT                      ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""

    local running_count=0
    local healthy_count=0

    for i in "${!INSTANCE_PORTS[@]}"; do
        local port="${INSTANCE_PORTS[$i]}"
        local name="${INSTANCE_NAMES[$i]}"
        local instance_id=$((i+1))
        local db_name="${DB_NAMES[$i]}"

        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Instance $instance_id:${NC} $name"
        echo -e "  Port:     $port"
        echo -e "  Database: $db_name"
        echo -e "  URL:      http://localhost:$port"

        if check_port $port; then
            local pid=$(get_pid_for_port $port)
            echo -e "  PID:      $pid"
            echo -e "  Status:   ${GREEN}RUNNING${NC}"
            running_count=$((running_count + 1))

            if check_health $port; then
                echo -e "  Health:   ${GREEN}HEALTHY${NC}"
                healthy_count=$((healthy_count + 1))
            else
                echo -e "  Health:   ${YELLOW}STARTING/UNHEALTHY${NC}"
            fi
        else
            echo -e "  PID:      N/A"
            echo -e "  Status:   ${RED}STOPPED${NC}"
            echo -e "  Health:   ${RED}DOWN${NC}"
        fi

        # Show log file location
        local log_file="$LOGS_DIR/instance_$instance_id/server.log"
        echo -e "  Log:      $log_file"

        # Show last few log lines if file exists
        if [ -f "$log_file" ]; then
            echo -e "  Last log:"
            tail -n 2 "$log_file" 2>/dev/null | sed 's/^/    /' || echo "    (no recent logs)"
        else
            echo -e "  Last log: (log file not found)"
        fi
        echo ""
    done

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    echo -e "  Total instances:   3"
    echo -e "  Running:           $running_count"
    echo -e "  Healthy:           $healthy_count"
    echo ""

    if [ $running_count -eq 3 ] && [ $healthy_count -eq 3 ]; then
        print_success "All instances are running and healthy!"
    elif [ $running_count -eq 3 ]; then
        print_warning "All instances are running but some are not healthy yet"
    elif [ $running_count -eq 0 ]; then
        print_error "No instances are running"
    else
        print_warning "Some instances are not running"
    fi

    echo ""
    echo -e "${YELLOW}Shamir Secret Sharing Setup:${NC}"
    echo -e "  Instance 1: Share 1 → http://localhost:12410 (DB: ${DB_NAMES[0]})"
    echo -e "  Instance 2: Share 2 → http://localhost:12411 (DB: ${DB_NAMES[1]})"
    echo -e "  Instance 3: Share 3 → http://localhost:12412 (DB: ${DB_NAMES[2]})"
    echo ""

    echo -e "${BLUE}Management Commands:${NC}"
    echo -e "  Start:   ./deploy/start-instances.sh"
    echo -e "  Stop:    ./deploy/stop-instances.sh"
    echo -e "  Logs:    tail -f deploy/logs/instance_*/server.log"
    echo -e "  Test:    ./deploy/test-client.sh"
    echo ""
}

main
