#!/bin/bash

# Database Setup Script for OSINT Testing
# This script initializes the MySQL database for all 3 instances

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
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

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCHEMA_FILE="$PROJECT_ROOT/schema/schema.sql"
ENV_FILE="$SCRIPT_DIR/.env"

# Load database connection configuration from .env file
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    print_info "Loaded configuration from .env file"
else
    print_warning ".env file not found, using defaults"
fi

# Database configuration (with defaults if not in .env)
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-3306}"
DB_USER="${DB_USER:-osint_user}"
DB_PASSWORD="${DB_PASSWORD:-osint_password}"
DB_ROOT_PASSWORD="${DB_ROOT_PASSWORD:-}"

# OSSPEP database configuration
OSSPEP_DB_NAME="${OSSPEP_DB_NAME:-osspep}"

# RethinkDB configuration
RETHINKDB_HOST="${RETHINKDB_HOST:-localhost}"
RETHINKDB_PORT="${RETHINKDB_PORT:-28015}"
RETHINKDB_DB="${RETHINKDB_DB:-im_osint_sync}"
RETHINKDB_TABLE="${RETHINKDB_TABLE:-otp_events}"

# Hardcoded database names
DB_NAMES=("osint_db1" "osint_db2" "osint_db3")

check_mysql() {
    if ! command -v mysql &> /dev/null; then
        print_error "MySQL client not found. Please install MySQL."
        exit 1
    fi
    print_success "MySQL client found"
}

check_mysql_server() {
    local mysql_cmd="mysql -h $DB_HOST -P $DB_PORT"

    if [ -n "$DB_ROOT_PASSWORD" ]; then
        mysql_cmd="$mysql_cmd -u root -p$DB_ROOT_PASSWORD"
    else
        mysql_cmd="$mysql_cmd -u $DB_USER"
        if [ -n "$DB_PASSWORD" ]; then
            mysql_cmd="$mysql_cmd -p$DB_PASSWORD"
        fi
    fi

    if ! $mysql_cmd -e "SELECT 1" --silent 2>/dev/null; then
        print_error "Cannot connect to MySQL server at $DB_HOST:$DB_PORT"
        print_info "Please check your credentials and ensure MySQL server is running"
        exit 1
    fi
    print_success "MySQL server is running and accessible"
}

create_databases() {
    print_info "Setting up 3 separate databases: ${DB_NAMES[@]}..."
    print_info "Note: Databases are assumed to already exist. Only tables will be dropped and recreated."
    print_info "For OSSPEP database ($OSSPEP_DB_NAME), tables will only be created if they don't exist."

    local mysql_cmd="mysql -h $DB_HOST -P $DB_PORT -u $DB_USER"
    if [ -n "$DB_PASSWORD" ]; then
        mysql_cmd="$mysql_cmd -p$DB_PASSWORD"
    fi

    # For main app databases: drop and recreate all tables
    for db_name in "${DB_NAMES[@]}"; do
        print_info "Setting up database: $db_name"

        # Check if database exists
        if ! $mysql_cmd -e "USE $db_name" 2>/dev/null; then
            print_warning "Database $db_name does not exist. Please create it first."
            print_info "You can create it with: CREATE DATABASE $db_name CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            exit 1
        fi

        # Drop all existing tables
        print_info "Dropping existing tables in $db_name..."
        $mysql_cmd "$db_name" -e "SET FOREIGN_KEY_CHECKS=0;" 2>/dev/null || true
        TABLES=$($mysql_cmd "$db_name" -e "SHOW TABLES;" 2>/dev/null | tail -n +2)
        if [ -n "$TABLES" ]; then
            echo "$TABLES" | while read TABLE; do
                $mysql_cmd "$db_name" -e "DROP TABLE IF EXISTS \`$TABLE\`;" 2>/dev/null || true
            done
        fi
        $mysql_cmd "$db_name" -e "SET FOREIGN_KEY_CHECKS=1;" 2>/dev/null || true

        print_success "Database ready: $db_name"
    done

    # Check OSSPEP database exists
    print_info "Checking OSSPEP database: $OSSPEP_DB_NAME"
    if ! $mysql_cmd -e "USE $OSSPEP_DB_NAME" 2>/dev/null; then
        print_warning "OSSPEP database $OSSPEP_DB_NAME does not exist. Please create it first."
        print_info "You can create it with: CREATE DATABASE $OSSPEP_DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        exit 1
    fi

    print_success "All databases accessible"
}

load_schemas() {
    print_info "Loading schema from: $SCHEMA_FILE"

    if [ ! -f "$SCHEMA_FILE" ]; then
        print_error "Schema file not found: $SCHEMA_FILE"
        exit 1
    fi

    # Load schema into each database using hardcoded names
    for db_name in "${DB_NAMES[@]}"; do
        print_info "Loading schema into $db_name..."

        # Use mysql with password option, show actual error if it fails
        if [ -n "$DB_PASSWORD" ]; then
            mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$db_name" < "$SCHEMA_FILE" || {
                print_error "Failed to load schema into $db_name"
                print_error "Check the error message above for details"
                exit 1
            }
        else
            mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "$db_name" < "$SCHEMA_FILE" || {
                print_error "Failed to load schema into $db_name"
                print_error "Check the error message above for details"
                exit 1
            }
        fi

        print_success "Schema loaded into $db_name"
    done
}

verify_tables() {
    print_info "Verifying tables in all databases..."

    for db_name in "${DB_NAMES[@]}"; do
        print_info "Verifying $db_name..."

        local tables=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$db_name" -e "SHOW TABLES;" 2>/dev/null | tail -n +2)

        if [ -z "$tables" ]; then
            print_error "No tables found in $db_name"
            exit 1
        fi

        local table_count=$(echo "$tables" | wc -l | tr -d ' ')
        print_success "$db_name has $table_count tables"
    done
}

# Check if OSSPEP database needs migration
check_osspep_needs_migration() {
    local mysql_cmd="mysql -h $DB_HOST -P $DB_PORT -u $DB_USER"
    if [ -n "$DB_PASSWORD" ]; then
        mysql_cmd="$mysql_cmd -p$DB_PASSWORD"
    fi

    # Check if database has any tables
    local table_count=$($mysql_cmd "$OSSPEP_DB_NAME" -e "SHOW TABLES;" 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')

    if [ "$table_count" -eq 0 ]; then
        print_info "OSSPEP database exists but has no tables. Migration needed."
        return 0  # Needs migration
    fi

    print_info "OSSPEP database already has $table_count tables. Skipping migration."
    return 1  # Does not need migration
}

# Load OSSPEP schema (only if tables don't exist)
load_osspep_schema() {
    local osspep_schema_file="$PROJECT_ROOT/schema/osspep.sql"

    print_info "Checking OSSPEP schema migration..."

    if [ ! -f "$osspep_schema_file" ]; then
        print_warning "OSSPEP schema file not found: $osspep_schema_file"
        print_info "Skipping OSSPEP migration"
        return 0
    fi

    # Check if migration is needed
    if ! check_osspep_needs_migration; then
        print_info "OSSPEP migration not needed (tables already exist)"
        return 0
    fi

    print_info "Loading OSSPEP schema into $OSSPEP_DB_NAME..."

    local mysql_cmd="mysql -h $DB_HOST -P $DB_PORT -u $DB_USER"
    if [ -n "$DB_PASSWORD" ]; then
        mysql_cmd="$mysql_cmd -p$DB_PASSWORD"
    fi

    # Load schema (uses CREATE TABLE IF NOT EXISTS, so safe to run)
    $mysql_cmd "$OSSPEP_DB_NAME" < "$osspep_schema_file" || {
        print_error "Failed to load OSSPEP schema into $OSSPEP_DB_NAME"
        print_error "Check the error message above for details"
        exit 1
    }

    # Verify tables were created
    local table_count=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$OSSPEP_DB_NAME" -e "SHOW TABLES;" 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
    print_success "OSSPEP schema loaded into $OSSPEP_DB_NAME ($table_count tables)"
}

setup_rethinkdb() {
    print_info "Setting up RethinkDB..."

    # Use project's virtual environment if available
    PYTHON_CMD="python3"
    if [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
        PYTHON_CMD="$PROJECT_ROOT/venv/bin/python"
        print_info "Using virtual environment Python"
    elif ! command -v python3 &> /dev/null; then
        print_warning "Python3 not found, skipping RethinkDB setup"
        return 0
    fi

    # Run RethinkDB setup using Python
    $PYTHON_CMD << EOF
import sys
try:
    from rethinkdb import RethinkDB
    r = RethinkDB()

    # Connect to RethinkDB
    print("[INFO] Connecting to RethinkDB at ${RETHINKDB_HOST}:${RETHINKDB_PORT}...")
    conn = r.connect(host="${RETHINKDB_HOST}", port=${RETHINKDB_PORT}, timeout=30)

    db_name = "${RETHINKDB_DB}"

    # Drop and recreate the entire database to handle duplicate tables
    # This fixes the "ambiguous table" error that can occur with corrupted state
    db_list = r.db_list().run(conn)
    if db_name in db_list:
        print(f"[INFO] Dropping existing RethinkDB database: {db_name}")
        try:
            r.db_drop(db_name).run(conn)
            print(f"[INFO] Dropped database: {db_name}")
        except Exception as e:
            print(f"[WARNING] Error dropping database: {e}")

    # Create fresh database
    r.db_create(db_name).run(conn)
    print(f"[INFO] Created RethinkDB database: {db_name}")

    # The app will recreate tables on startup:
    # - otp_events (via _setup_rethinkdb_otp_table in creat_table.py)
    # - document_analysis_jobs (via _setup_rethinkdb_jobs_table in creat_table.py)
    print("[INFO] App will recreate tables on startup")

    conn.close()
    print("[SUCCESS] RethinkDB setup completed successfully")

except ImportError:
    print("[WARNING] RethinkDB Python package not installed, skipping RethinkDB setup")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] RethinkDB setup failed: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        print_success "RethinkDB setup completed"
    else
        print_warning "RethinkDB setup had issues (non-fatal)"
    fi
}

show_connection_info() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Database Setup Complete!${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}Connection Details:${NC}"
    echo -e "  Host:     $DB_HOST"
    echo -e "  Port:     $DB_PORT"
    echo -e "  User:     $DB_USER"
    echo -e "  Password: $DB_PASSWORD"
    echo ""
    echo -e "${YELLOW}Main App Databases (tables dropped & recreated):${NC}"
    echo -e "  Instance 1: ${DB_NAMES[0]}"
    echo -e "  Instance 2: ${DB_NAMES[1]}"
    echo -e "  Instance 3: ${DB_NAMES[2]}"
    echo ""
    echo -e "${YELLOW}OSSPEP Database (tables only created if missing):${NC}"
    echo -e "  Database: $OSSPEP_DB_NAME"
    echo -e "  Note: Tables are NEVER dropped or truncated in OSSPEP database"
    echo ""
    echo -e "${YELLOW}Test Connections:${NC}"
    echo -e "  mysql -h $DB_HOST -P $DB_PORT -u $DB_USER -p$DB_PASSWORD ${DB_NAMES[0]}"
    echo -e "  mysql -h $DB_HOST -P $DB_PORT -u $DB_USER -p$DB_PASSWORD ${DB_NAMES[1]}"
    echo -e "  mysql -h $DB_HOST -P $DB_PORT -u $DB_USER -p$DB_PASSWORD ${DB_NAMES[2]}"
    echo -e "  mysql -h $DB_HOST -P $DB_PORT -u $DB_USER -p$DB_PASSWORD $OSSPEP_DB_NAME"
    echo ""
    echo -e "${BLUE}Note:${NC} Each OSINT instance has its own separate database"
    echo ""
    echo -e "${YELLOW}RethinkDB:${NC}"
    echo -e "  Host:     $RETHINKDB_HOST"
    echo -e "  Port:     $RETHINKDB_PORT"
    echo -e "  Database: $RETHINKDB_DB"
    echo -e "  Tables:   All tables dropped (app will recreate on startup)"
    echo ""
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║   OSINT Database Setup                         ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""

    # Check prerequisites
    check_mysql
    check_mysql_server

    # Setup databases (main app dbs: drop tables, OSSPEP: check only)
    create_databases

    # Load main app schemas (drop and recreate tables)
    load_schemas

    # Load OSSPEP schema (only if tables don't exist)
    load_osspep_schema

    # Verify
    verify_tables

    # Setup RethinkDB (drop and recreate table)
    setup_rethinkdb

    # Show connection info
    show_connection_info

    print_success "Database setup completed successfully!"
    print_info "You can now start the OSINT instances with: ./deploy/start-instances.sh"
}

main
