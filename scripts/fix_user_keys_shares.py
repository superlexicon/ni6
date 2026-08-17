"""
One-time cleanup for user_keys rows polluted by the replication bug.

Run once per instance database (from the project root, with the project
venv active):

    python scripts/fix_user_keys_shares.py            # backfill + junk removal
    python scripts/fix_user_keys_shares.py --dry-run  # report only, change nothing
    python scripts/fix_user_keys_shares.py --remove-dead  # also drop dead rows

Background: during the broken window, replicated result pushes created
user_keys rows with encrypted_secret_share = NULL (and rows keyed by
ephemeral recovery temp keys). Share-less rows satisfy key-recovery
lookups but cannot produce shares, breaking the 2-of-3 quorum, and they
block the pending->user_keys migration (the selfie service skips the
move when a row already exists).

What this script does:
1. BACKFILL: for user_keys rows with a NULL share that still have a
   matching user_keys_pending row, copy the pending row's share (plus
   mobile/country/api_url/device_id where NULL) into user_keys and delete
   the migrated pending row - exactly what move_pending_to_user_keys
   would have done locally.
2. JUNK REMOVAL (default): delete rows with NULL share AND NULL identity
   AND NULL mobile number (recovery temp-key rows - a real migrated row
   always carries identity + share + mobile).
3. DEAD ROWS (report only; --remove-dead to delete): rows with an
   identity but NULL share and no pending row left (pending was TTL-
   cleaned). Their share is unrecoverable on this instance - the user
   must re-register. Removing the row lets a future re-registration
   migrate cleanly instead of being blocked by the existing-row check.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db.database import get_db_connection_context  # noqa: E402
from app.core.logger import get_logger  # noqa: E402

logger = get_logger()

BACKFILL_SQL = """
    UPDATE user_keys uk
    JOIN user_keys_pending p ON uk.user_public_key = p.user_public_key
    SET uk.encrypted_secret_share = p.encrypted_secret_share,
        uk.mobile_number = COALESCE(uk.mobile_number, p.mobile_number),
        uk.country_code = COALESCE(uk.country_code, p.country_code),
        uk.api_url = COALESCE(uk.api_url, p.api_url),
        uk.device_id = COALESCE(uk.device_id, p.device_id),
        uk.updated_at = CURRENT_TIMESTAMP
    WHERE uk.encrypted_secret_share IS NULL
      AND p.encrypted_secret_share IS NOT NULL
"""

DELETE_MIGRATED_PENDING_SQL = """
    DELETE p FROM user_keys_pending p
    JOIN user_keys uk ON uk.user_public_key = p.user_public_key
    WHERE uk.encrypted_secret_share = p.encrypted_secret_share
      AND uk.encrypted_secret_share IS NOT NULL
"""

JUNK_SELECT_SQL = """
    SELECT user_public_key, user_identity_id, created_at FROM user_keys
    WHERE encrypted_secret_share IS NULL
      AND user_identity_id IS NULL
      AND mobile_number IS NULL
"""

JUNK_DELETE_SQL = """
    DELETE FROM user_keys
    WHERE encrypted_secret_share IS NULL
      AND user_identity_id IS NULL
      AND mobile_number IS NULL
"""

DEAD_SELECT_SQL = """
    SELECT user_public_key, user_identity_id, created_at FROM user_keys
    WHERE encrypted_secret_share IS NULL
"""

DEAD_DELETE_SQL = """
    DELETE FROM user_keys
    WHERE encrypted_secret_share IS NULL
"""


def fetch(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    parser.add_argument("--remove-dead", action="store_true",
                        help="also delete dead share-less rows (users must re-register)")
    args = parser.parse_args()

    with get_db_connection_context() as conn:
        cursor = conn.cursor(dictionary=True)

        # 1. Backfill shares from pending rows
        junk_before = fetch(cursor, JUNK_SELECT_SQL)
        dead_before = fetch(cursor, DEAD_SELECT_SQL)

        if not args.dry_run:
            cursor.execute(BACKFILL_SQL, ())
            backfilled = cursor.rowcount
            cursor.execute(DELETE_MIGRATED_PENDING_SQL, ())
            pending_removed = cursor.rowcount
            conn.commit()
            logger.info(f"Backfilled shares into {backfilled} user_keys row(s); "
                        f"removed {pending_removed} migrated pending row(s)")
        else:
            backfilled = None

        # 2. Junk rows (temp-key recovery rows)
        junk = fetch(cursor, JUNK_SELECT_SQL)
        for row in junk:
            logger.info(f"Junk (temp-key) row: {row['user_public_key'][:16]}... "
                        f"identity={row['user_identity_id']} created={row['created_at']}")
        if junk and not args.dry_run:
            cursor.execute(JUNK_DELETE_SQL, ())
            conn.commit()
            logger.info(f"Deleted {cursor.rowcount} junk user_keys row(s)")

        # 3. Dead rows (share NULL, nothing to backfill from)
        dead = fetch(cursor, DEAD_SELECT_SQL)
        for row in dead:
            logger.warning(
                f"DEAD row (share unrecoverable, user must re-register): "
                f"{row['user_public_key'][:16]}... identity={row['user_identity_id']} "
                f"created={row['created_at']}"
            )
        if dead and args.remove_dead and not args.dry_run:
            cursor.execute(DEAD_DELETE_SQL, ())
            conn.commit()
            logger.info(f"Deleted {cursor.rowcount} dead user_keys row(s)")

        cursor.close()

    summary = f"Done. junk_before={len(junk_before)} dead_before={len(dead_before)}"
    if backfilled is not None:
        summary += f" backfilled={backfilled} junk_now={len(junk)} dead_now={len(dead)}"
    logger.info(summary)
    if dead and not args.remove_dead:
        logger.info("Re-run with --remove-dead to delete dead rows so affected "
                    "users can re-register cleanly.")


if __name__ == "__main__":
    main()
