#!/usr/bin/env python
"""
Apply the 009_consolidated_migrations.sql migration.
"""
import re
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Get a direct database connection."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3308")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "rootpassword"),
        database=os.getenv("DB_NAME", "osint_db1"),
        ssl_disabled=True,
        ssl_verify_cert=False
    )

def apply_migration():
    """Apply the migration file."""
    # Read the migration file
    with open('schema/migrations/010_add_missing_indian_banks.sql', 'r') as f:
        sql = f.read()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Execute statements in order, handling multi-line statements
        # Remove comment lines first
        lines = []
        for line in sql.split('\n'):
            stripped = line.strip()
            # Skip comment-only lines
            if stripped.startswith('--'):
                continue
            lines.append(line)

        cleaned_sql = '\n'.join(lines)

        # Find each complete statement (ending with semicolon)
        # Handle statements that span multiple lines
        statements = []
        current_statement = []

        for line in cleaned_sql.split('\n'):
            current_statement.append(line)
            # Check if this line ends with a semicolon (possibly with whitespace)
            if line.strip().endswith(';'):
                statement = '\n'.join(current_statement)
                if statement.strip():
                    statements.append(statement)
                current_statement = []

        # Add any remaining content (shouldn't be any for valid SQL)
        if current_statement:
            remaining = '\n'.join(current_statement)
            if remaining.strip():
                statements.append(remaining)

        print(f"Found {len(statements)} statements to execute")

        for i, statement in enumerate(statements):
            statement = statement.strip()

            # Skip empty statements
            if not statement or len(statement) < 5:
                continue

            # Skip SELECT queries (verification/debugging)
            if statement.upper().startswith('SELECT'):
                print(f"Skipping SELECT query: {statement[:50]}...")
                continue
            if statement.upper().startswith('SHOW'):
                print(f"Skipping SHOW query: {statement[:50]}...")
                continue

            # Remove trailing semicolon for cleaner execution
            statement = statement.rstrip(';')

            try:
                cursor.execute(statement)
                print(f"Executed statement {i+1}/{len(statements)}: {statement[:60]}...")
            except mysql.connector.Error as e:
                error_code = e.errno if hasattr(e, 'errno') else None
                error_msg = str(e)

                # Expected errors for DROP IF EXISTS
                if 'Unknown table' in error_msg or "doesn't exist" in error_msg:
                    print(f"  (Table doesn't exist, skipping DROP: {statement[:40]}...)")
                elif 'Duplicate entry' in error_msg:
                    print(f"  (Duplicate entry, ignoring: {error_msg[:60]}...)")
                else:
                    print(f"WARNING: {e}")
                    print(f"  Statement: {statement[:100]}...")

        conn.commit()
        print("\nMigration applied successfully!")

        # Verification
        cursor.execute("SELECT COUNT(*) FROM banks WHERE country_code='IN'")
        indian_banks = cursor.fetchone()[0]
        print(f"Indian banks in DB: {indian_banks}")

        cursor.execute("SELECT swift_code, legal_name FROM banks WHERE country_code='IN' LIMIT 5")
        print("Sample Indian banks:")
        for row in cursor.fetchall():
            print(f"  {row[0]} - {row[1]}")

    finally:
        conn.close()

if __name__ == '__main__':
    apply_migration()
