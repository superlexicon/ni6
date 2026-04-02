#!/bin/bash

# Stop Script for OSINT Instances
# This script stops all running instances

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
INSTANCE_PORTS=(12410 12411 12412)

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

# Main execution
main() {
    echo -e "${YELLOW}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║   Stopping OSINT Instances                     ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"

    local stopped_count=0

    # Kill processes from PID file
    if [ -f "$PIDS_FILE" ]; then
        print_info "Killing processes from PID file..."
        while IFS= read -r pid; do
            if ps -p $pid > /dev/null 2>&1; then
                print_info "Killing PID: $pid"
                kill -9 $pid 2>/dev/null || true
                stopped_count=$((stopped_count + 1))
                print_success "Killed PID: $pid"
            else
                print_warning "PID $pid is not running"
            fi
        done < "$PIDS_FILE"
        rm "$PIDS_FILE"
        print_success "Removed PID file"
    else
        print_warning "No PID file found at $PIDS_FILE"
    fi

    # Kill processes using the ports
    print_info "Checking ports..."
    for port in "${INSTANCE_PORTS[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            pid=$(lsof -t -i:$port)
            print_warning "Port $port is still in use by PID $pid. Killing..."
            kill -9 $pid 2>/dev/null || true
            stopped_count=$((stopped_count + 1))
            print_success "Killed process on port $port"
        fi
    done

    # Verify all ports are free
    print_info "Verifying all ports are free..."
    all_free=true
    for port in "${INSTANCE_PORTS[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            print_error "Port $port is still in use!"
            all_free=false
        else
            print_success "Port $port is free"
        fi
    done

    echo ""
    if [ "$all_free" = true ]; then
        print_success "All instances stopped successfully! ($stopped_count processes killed)"
    else
        print_error "Some ports are still in use. Please check manually."
        exit 1
    fi
}

main
