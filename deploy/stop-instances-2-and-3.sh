#!/bin/bash

# Stop only instances 2 and 3, keeping instance 1 running
# This is for testing GPU resource contention issues

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration - only stop instances 2 and 3
STOP_PORTS=(12411 12412)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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

main() {
    echo -e "${YELLOW}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║   Stopping Instances 2 & 3 (keeping Instance 1) ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"

    local stopped_count=0

    # Kill processes using ports 12411 and 12412
    for port in "${STOP_PORTS[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            pid=$(lsof -t -i:$port)
            print_warning "Stopping instance on port $port (PID: $pid)..."
            kill -9 $pid 2>/dev/null || true
            stopped_count=$((stopped_count + 1))
            sleep 1
            
            # Verify it's stopped
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
                print_error "Port $port is still in use!"
            else
                print_success "Stopped instance on port $port"
            fi
        else
            print_warning "Port $port is not in use (instance already stopped)"
        fi
    done

    # Check instance 1 status
    echo ""
    if lsof -Pi :12410 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        print_success "Instance 1 (port 12410) is still RUNNING"
    else
        print_warning "Instance 1 (port 12410) is NOT running"
    fi

    echo ""
    print_success "Stopped $stopped_count instance(s). Only Instance 1 should now be running."
    print_info "You can now test postal code extraction with reduced GPU contention."
}

main
