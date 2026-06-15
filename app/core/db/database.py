import mysql.connector
from mysql.connector import pooling, Error
import os
import threading
from contextlib import contextmanager
from dotenv import load_dotenv
from app.core.logger import get_logger

load_dotenv(override=False)  # Don't override command-line env vars with .env file
logger = get_logger()

def create_database_if_not_exists(host, user, password, database_name, port=3306):
    try:
        # Check if host is a socket path
        if host and host.startswith('/'):
            # Socket connection
            conn = mysql.connector.connect(
                unix_socket=host, user=user, password=password,
                ssl_disabled=True, ssl_verify_cert=False
            )
        else:
            # TCP connection
            conn = mysql.connector.connect(
                host=host, user=user, password=password, port=port,
                ssl_disabled=True, ssl_verify_cert=False
            )
        cursor = conn.cursor()

        # Create database if not exists
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        logger.info(f"Database '{database_name}' is ready")

        cursor.close()
        conn.close()
        return True
    except Error as e:
        logger.error(f"Error creating database: {e}")
        return False


class Database:
    _instance = None
    _lock = threading.Lock()
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Database, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            db_host = os.getenv("DB_HOST")

            # Check if DB_HOST is a socket path (starts with /)
            if db_host and db_host.startswith('/'):
                # Socket connection - remove TCP parameters
                db_config = {
                    "unix_socket": db_host,
                    "user": os.getenv("DB_USER"),
                    "password": os.getenv("DB_PASSWORD"),
                    "database": os.getenv("DB_NAME"),
                }
                logger.info(f"Using MySQL socket connection: {db_host}")
            else:
                # TCP connection
                db_config = {
                    "host": db_host,
                    "user": os.getenv("DB_USER"),
                    "password": os.getenv("DB_PASSWORD"),
                    "database": os.getenv("DB_NAME"),
                    "port": int(os.getenv("DB_PORT", 3306)),
                }
                logger.info(f"Using MySQL TCP connection: {db_host}:{db_config['port']}")

            try:
                # First ensure database exists
                if not create_database_if_not_exists(
                    host=db_host,  # Use original host value to determine connection type
                    user=db_config["user"],
                    password=db_config["password"],
                    database_name=db_config["database"],
                    port=db_config.get("port", 3306),
                ):
                    raise Exception("Failed to ensure database exists")

                # Enhanced db_config for better type handling (SSL disabled for MariaDB)
                # Note: init_command is NOT used with connection pools as it can cause connection failures
                # Session variables are set when needed via cursor.execute()
                enhanced_config = {
                    **db_config,
                    "autocommit": True,
                    "charset": "utf8mb4",
                    "use_unicode": True,
                    "get_warnings": True,  # Enable warnings for proper handling
                    "raise_on_warnings": False,
                    "connect_timeout": 30,
                    "ssl_disabled": True,  # SSL never required for MariaDB
                    "ssl_verify_cert": False,
                }

                # Now create connection pool with enhanced configuration
                # Note: pool_recycle is not supported, use connection timeout instead
                # Pool size increased to handle concurrent API requests + background workers
                self._pool = pooling.MySQLConnectionPool(
                    pool_name="im_osint_pool",
                    pool_size=20,
                    **enhanced_config
                )
                logger.info("Database connection pool created successfully with enhanced configuration.")
                self._initialized = True

            except Exception as err:
                logger.error(f"Error initializing database: {err}")
                Database._instance = None
                self._initialized = False
                raise

    def _handle_warnings(self, connection):
        """Handle and print any warnings from the database connection."""
        try:
            if connection and hasattr(connection, 'warnings'):
                warnings = connection.warnings()
                if warnings:
                    for warning in warnings:
                        logger.warning(f"Database warning: {warning}")
        except Exception as e:
            logger.error(f"Error handling database warnings: {e}")

    def handle_operation_warnings(self, connection):
        """
        Handle warnings after database operations.
        This method should be called after executing queries that might generate warnings.
        """
        self._handle_warnings(connection)

    def get_connection(self):
        """Get a database connection with retry logic for reliability."""
        if self._pool is None:
            raise RuntimeError("Database connection pool is not initialized")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                connection = self._pool.get_connection()
                # Test the connection to ensure it's alive
                connection.ping(reconnect=True, attempts=1)
                # Handle any warnings that might be present
                self._handle_warnings(connection)
                return connection
            except mysql.connector.Error as err:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {err}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get database connection after {max_retries} attempts: {err}")
                    raise RuntimeError(f"Failed to get database connection after {max_retries} attempts: {err}")
                # Brief delay before retry
                import time
                time.sleep(0.5)


def get_db_connection() -> pooling.PooledMySQLConnection:
    db_instance = Database()
    connection = db_instance.get_connection()
    if connection is None:
        raise RuntimeError("Failed to get database connection")
    return connection


def handle_db_warnings(connection):
    """
    Utility function to handle database warnings from any connection.
    This can be used anywhere in the codebase after database operations.
    """
    db_instance = Database()
    db_instance.handle_operation_warnings(connection)


# ==================== OSSPEP Database Connection ====================
# Separate database for sanctions/PEP data (managed by OSSPEP service)
# This application only reads from this database - no writing/syncing

_osspep_pool = None
_osspep_lock = threading.Lock()


def get_osspep_db_connection():
    """
    Get a database connection to the OSSPEP database.

    The OSSPEP database contains sanctions and PEP watchlist data
    that is synced separately by the OSSPEP Quarkus service.
    This application only reads from this database.

    Environment variables:
        OSSPEP_DB_HOST: Database host (default: localhost)
        OSSPEP_DB_PORT: Database port (default: 3306)
        OSSPEP_DB_USER: Database user (default: DB_USER)
        OSSPEP_DB_PASSWORD: Database password (default: DB_PASSWORD)
        OSSPEP_DB_NAME: Database name (default: osspep)

    Returns:
        PooledMySQLConnection: Connection to OSSPEP database
    """
    global _osspep_pool

    with _osspep_lock:
        if _osspep_pool is not None:
            return _osspep_pool.get_connection()

        # Get OSSPEP database config (fallback to main DB config)
        osspep_host = os.getenv("OSSPEP_DB_HOST") or os.getenv("DB_HOST")
        osspep_port = int(os.getenv("OSSPEP_DB_PORT") or os.getenv("DB_PORT", 3306))
        osspep_user = os.getenv("OSSPEP_DB_USER") or os.getenv("DB_USER")
        osspep_password = os.getenv("OSSPEP_DB_PASSWORD") or os.getenv("DB_PASSWORD")
        osspep_database = os.getenv("OSSPEP_DB_NAME", "osspep")

        # Check if host is a socket path
        if osspep_host and osspep_host.startswith('/'):
            db_config = {
                "unix_socket": osspep_host,
                "user": osspep_user,
                "password": osspep_password,
                "database": osspep_database,
            }
            logger.info(f"Using MySQL socket connection for OSSPEP: {osspep_host}")
        else:
            db_config = {
                "host": osspep_host,
                "port": osspep_port,
                "user": osspep_user,
                "password": osspep_password,
                "database": osspep_database,
            }
            logger.info(f"Using MySQL TCP connection for OSSPEP: {osspep_host}:{osspep_port}")

        # Enhanced config for OSSPEP
        # Note: init_command is NOT used with connection pools as it can cause connection failures
        enhanced_config = {
            **db_config,
            "autocommit": True,
            "charset": "utf8mb4",
            "use_unicode": True,
            "get_warnings": True,
            "raise_on_warnings": False,
            "connect_timeout": 30,
            "ssl_disabled": True,
            "ssl_verify_cert": False,
        }

        # Create connection pool for OSSPEP
        _osspep_pool = pooling.MySQLConnectionPool(
            pool_name="osspep_pool",
            pool_size=5,  # Smaller pool since it's read-only
            **enhanced_config
        )
        logger.info(f"OSSPEP database connection pool created: {osspep_database}")

        return _osspep_pool.get_connection()


# ==================== Connection Context Manager ====================
# Safe connection handling with automatic cleanup and error recovery

@contextmanager
def get_db_connection_context():
    """
    Context manager for safe database connection handling.

    Ensures connections are properly cleaned up and not returned to the pool
    in a bad state. Use this for complex operations that might fail.

    Example:
        with get_db_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            results = cursor.fetchall()
            # Connection automatically cleaned up, even if error occurs
    """
    conn = None
    try:
        conn = get_db_connection()  # This raises RuntimeError if conn is None
        if conn is None:
            # This should never happen since get_db_connection() raises RuntimeError,
            # but we keep this as a defensive check
            logger.error("get_db_connection() returned None - this should not happen!")
            raise RuntimeError("Failed to get database connection - pool may not be initialized")
        logger.debug(f"Successfully got database connection: {conn}")
        yield conn
    except Exception as e:
        # If an error occurred, try to reset the connection before returning to pool
        if conn:
            try:
                # Attempt to reset the connection state
                conn.rollback()
                # Ping will reconnect if the connection is dead
                conn.ping(reconnect=True, attempts=1)
            except Exception:
                # Connection is dead, will be discarded by pool
                logger.warning(f"Connection in bad state, will be discarded: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


@contextmanager
def get_osspep_db_connection_context():
    """
    Context manager for safe OSSPEP database connection handling.

    Similar to get_db_connection_context but for the OSSPEP database.
    """
    conn = None
    try:
        conn = get_osspep_db_connection()
        yield conn
    except Exception as e:
        if conn:
            try:
                conn.rollback()
                conn.ping(reconnect=True, attempts=1)
            except Exception:
                logger.warning(f"OSSPEP connection in bad state, will be discarded: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing OSSPEP connection: {e}")
