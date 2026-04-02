#!/bin/bash

# Localhost Deployment Script for 3 OSINT Instances
# This script starts 3 instances on ports 12410, 12411, 12412

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$SCRIPT_DIR/logs"
PIDS_FILE="$SCRIPT_DIR/pids.txt"
ENV_FILE="$SCRIPT_DIR/.env"

# Load database connection configuration from .env file
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Database configuration (with defaults if not in .env)
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-3306}"
DB_USER="${DB_USER:-osint_user}"
DB_PASSWORD="${DB_PASSWORD:-osint_password}"

# Hardcoded instance configuration
INSTANCE_PORTS=(12410 12411 12412)
INSTANCE_NAMES=("OSINT_Instance_1" "OSINT_Instance_2" "OSINT_Instance_3")
DB_NAMES=("osint_db1" "osint_db2" "osint_db3")

# Function to print colored output
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

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill existing instances
kill_existing_instances() {
    print_info "Checking for existing instances..."

    if [ -f "$PIDS_FILE" ]; then
        while IFS= read -r pid; do
            if ps -p $pid > /dev/null 2>&1; then
                print_warning "Killing existing instance with PID: $pid"
                kill -9 $pid 2>/dev/null || true
            fi
        done < "$PIDS_FILE"
        rm "$PIDS_FILE"
    fi

    # Also check ports and kill processes using them
    for port in "${INSTANCE_PORTS[@]}"; do
        if check_port $port; then
            pid=$(lsof -t -i:$port)
            print_warning "Port $port is in use by PID $pid. Killing..."
            kill -9 $pid 2>/dev/null || true
            sleep 1
        fi
    done
}

# Function to setup directories
setup_directories() {
    print_info "Setting up directories..."

    mkdir -p "$LOGS_DIR"

    for i in "${!INSTANCE_PORTS[@]}"; do
        mkdir -p "$LOGS_DIR/instance_$((i+1))"
    done

    print_success "Directories created"
}

# Function to compute seeds and public keys from Python
# This ensures the PEER_PUBLIC_KEYS match the actual keys derived from SEED
compute_keys() {
    print_info "Computing deterministic keys for 3 instances..."

    cd "$PROJECT_ROOT"
    # Get space-separated seeds and public keys from Python
    eval $(poetry run python3 deploy/compute_keys.py)

    # Convert space-separated strings to arrays
    GENERATED_SEEDS=($GENERATED_SEEDS)
    GENERATED_PUBLIC_KEYS=($GENERATED_PUBLIC_KEYS)

    print_success "Keys computed successfully"
    print_info "  Instance 1: SEED=${GENERATED_SEEDS[0]:0:20}... PUBKEY=${GENERATED_PUBLIC_KEYS[0]:0:32}..."
    print_info "  Instance 2: SEED=${GENERATED_SEEDS[1]:0:20}... PUBKEY=${GENERATED_PUBLIC_KEYS[1]:0:32}..."
    print_info "  Instance 3: SEED=${GENERATED_SEEDS[2]:0:20}... PUBKEY=${GENERATED_PUBLIC_KEYS[2]:0:32}..."
}

# Function to start an instance
start_instance() {
    local instance_id=$1
    local port=$2
    local name=$3
    local log_file="$LOGS_DIR/instance_$instance_id/server.log"
    local db_name="${DB_NAMES[$((instance_id-1))]}"  # Use hardcoded DB name

    # Use computed SEED for this instance (index: instance_id-1)
    local unique_key="${GENERATED_SEEDS[$((instance_id-1))]}"

    # Build this instance's URL
    local this_instance_url="http://localhost:$port"

    # Build peer instances list (all other instances)
    local peer_instances=""
    for i in "${!INSTANCE_PORTS[@]}"; do
        if [ $((i+1)) -ne $instance_id ]; then
            local peer_port="${INSTANCE_PORTS[$i]}"
            if [ -z "$peer_instances" ]; then
                peer_instances="http://localhost:$peer_port"
            else
                peer_instances="$peer_instances,http://localhost:$peer_port"
            fi
        fi
    done

    # Build PEER_PUBLIC_KEYS JSON mapping (peer URL -> computed public key)
    local peer_public_keys="{"
    local first=true
    for i in "${!INSTANCE_PORTS[@]}"; do
        if [ $((i+1)) -ne $instance_id ]; then
            local peer_port="${INSTANCE_PORTS[$i]}"
            local peer_pubkey="${GENERATED_PUBLIC_KEYS[$i]}"
            if [ "$first" = true ]; then
                first=false
            else
                peer_public_keys="$peer_public_keys,"
            fi
            peer_public_keys="$peer_public_keys\"http://localhost:$peer_port\":\"$peer_pubkey\""
        fi
    done
    peer_public_keys="$peer_public_keys}"

    print_info "Starting $name on port $port (Database: $db_name)"
    echo ""
    echo -e "${YELLOW}Environment Variables for $name:${NC}"
    echo -e "  ${BLUE}SEED:${NC}                $unique_key"
    echo -e "  ${BLUE}INSTANCE_URL:${NC}        $this_instance_url"
    echo -e "  ${BLUE}PEER_INSTANCES:${NC}      $peer_instances"
    echo -e "  ${BLUE}PEER_PUBLIC_KEYS:${NC}    $peer_public_keys"
    echo ""

    cd "$PROJECT_ROOT"

    # Start the instance in background with environment variables
    nohup env \
        INSTANCE_ID=$instance_id \
        INSTANCE_NAME="$name" \
        PORT=$port \
        DB_HOST="$DB_HOST" \
        DB_PORT="$DB_PORT" \
        DB_NAME="$db_name" \
        DB_USER="$DB_USER" \
        DB_PASSWORD="$DB_PASSWORD" \
        INSTANCE_URL="$this_instance_url" \
        PEER_INSTANCES="$peer_instances" \
        PEER_PUBLIC_KEYS="$peer_public_keys" \
        STARTUP_SYNC_ENABLED="${STARTUP_SYNC_ENABLED:-true}" \
        STARTUP_SYNC_TIMEOUT="${STARTUP_SYNC_TIMEOUT:-30}" \
        PEER_REQUEST_TIMEOUT="${PEER_REQUEST_TIMEOUT:-10}" \
        PYTORCH_ENABLE_MPS_FALLBACK=1 \
        TF_USE_LEGACY_KERAS=1 \
        SEED="$unique_key" \
        OSINT_FACE_VERIFIED_MAX_IMAGES=0 \
        JOB_TIMEOUT_SECONDS=600 \
        VERIFICATION_NAME_MATCH_THRESHOLD=60.0 \
        poetry run app > "$log_file" 2>&1 &

    local pid=$!
    echo $pid >> "$PIDS_FILE"

    print_success "$name started with PID: $pid (Port: $port, DB: $db_name)"

    # Wait a bit to check if process is still running
    sleep 2
    if ps -p $pid > /dev/null; then
        print_success "$name is running successfully"
    else
        print_error "$name failed to start. Check logs at: $log_file"
        return 1
    fi
}

# Function to wait for instance to be ready
wait_for_instance() {
    local port=$1
    local instance_name=$2
    local max_attempts=30
    local attempt=0

    print_info "Waiting for $instance_name to be ready..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/api/health" > /dev/null 2>&1; then
            print_success "$instance_name is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
        echo -n "."
    done

    echo ""
    print_warning "$instance_name health check timeout (not critical if logs show it's running)"
    return 1
}

# Function to display instance status
show_status() {
    echo ""
    print_info "============================================"
    print_info "         OSINT INSTANCES STATUS            "
    print_info "============================================"
    echo ""

    for i in "${!INSTANCE_PORTS[@]}"; do
        local port="${INSTANCE_PORTS[$i]}"
        local name="${INSTANCE_NAMES[$i]}"
        local instance_id=$((i+1))
        local db_name="${DB_NAMES[$i]}"

        echo -e "${BLUE}Instance $instance_id:${NC} $name"
        echo -e "  Port: $port"
        echo -e "  Database: $db_name"
        echo -e "  URL: http://localhost:$port"
        echo -e "  Logs: $LOGS_DIR/instance_$instance_id/server.log"

        if check_port $port; then
            echo -e "  Status: ${GREEN}RUNNING${NC}"
        else
            echo -e "  Status: ${RED}STOPPED${NC}"
        fi
        echo ""
    done

    echo -e "${YELLOW}Shamir Secret Sharing:${NC}"
    echo -e "  Client should send 1 share to each instance"
    echo -e "  Instance 1: Share 1 -> http://localhost:12410 (DB: ${DB_NAMES[0]})"
    echo -e "  Instance 2: Share 2 -> http://localhost:12411 (DB: ${DB_NAMES[1]})"
    echo -e "  Instance 3: Share 3 -> http://localhost:12412 (DB: ${DB_NAMES[2]})"
    echo ""

    print_info "============================================"
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║   OSINT Localhost Deployment Script           ║"
    echo "║   Starting 3 Instances for Testing            ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Step 1: Kill existing instances
    kill_existing_instances

    # Step 2: Setup directories
    setup_directories

    # Step 3: Compute seeds and public keys
    compute_keys

    # Step 4: Start instances
    print_info "Starting instances..."
    for i in "${!INSTANCE_PORTS[@]}"; do
        start_instance $((i+1)) "${INSTANCE_PORTS[$i]}" "${INSTANCE_NAMES[$i]}"
        sleep 3  # Give each instance time to start before starting the next
    done

    # Step 4: Wait for instances to be ready
    echo ""
    print_info "Waiting for all instances to be ready..."
    for i in "${!INSTANCE_PORTS[@]}"; do
        wait_for_instance "${INSTANCE_PORTS[$i]}" "${INSTANCE_NAMES[$i]}" || true
    done

    # Step 5: Show status
    show_status

    print_success "All instances started successfully!"
    print_info "To stop all instances, run: ./deploy/stop-instances.sh"
    print_info "To view logs, run: tail -f deploy/logs/instance_*/server.log"
}

# Run main function
main
