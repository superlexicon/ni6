import threading
import time
from rethinkdb import RethinkDB
from app.core.logger import get_logger
from app.config.rethinkdb_config import rethinkdb_settings

logger = get_logger()
r = RethinkDB()


class RethinkDBConnection:
    """
    Singleton connection manager for RethinkDB.
    Handles connection establishment, database/table creation, and reconnection.
    """
    _instance = None
    _lock = threading.Lock()
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RethinkDBConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            self.host = rethinkdb_settings.host
            self.port = rethinkdb_settings.port
            self.db_name = rethinkdb_settings.db
            self.table_name = rethinkdb_settings.table
            self.user = rethinkdb_settings.user
            self.password = rethinkdb_settings.password
            self.timeout = rethinkdb_settings.timeout

            try:
                self._connect()
                self._ensure_database_exists()
                self._ensure_table_exists()
                self._initialized = True
                logger.info(f"RethinkDB connection initialized: {self.host}:{self.port}/{self.db_name}")
            except Exception as e:
                logger.error(f"Failed to initialize RethinkDB connection: {e}")
                self._initialized = False
                RethinkDBConnection._instance = None
                raise

    def _connect(self):
        """Establish connection to RethinkDB with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                connect_args = {
                    "host": self.host,
                    "port": self.port,
                    "timeout": self.timeout,
                }
                if self.user and self.password:
                    connect_args["user"] = self.user
                    connect_args["password"] = self.password

                self._connection = r.connect(**connect_args)
                logger.info(f"RethinkDB connected to {self.host}:{self.port}")
                return
            except Exception as e:
                logger.warning(f"RethinkDB connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    def _ensure_database_exists(self):
        """Create database if it doesn't exist."""
        try:
            db_list = r.db_list().run(self._connection)
            if self.db_name not in db_list:
                r.db_create(self.db_name).run(self._connection)
                logger.info(f"Created RethinkDB database: {self.db_name}")
            else:
                logger.info(f"RethinkDB database '{self.db_name}' already exists")
        except Exception as e:
            logger.error(f"Error ensuring database exists: {e}")
            raise

    def _ensure_table_exists(self):
        """Create otp_events table if it doesn't exist."""
        try:
            table_list = r.db(self.db_name).table_list().run(self._connection)
            if self.table_name not in table_list:
                r.db(self.db_name).table_create(self.table_name).run(self._connection)
                # Create index on mobile_number for efficient queries
                r.db(self.db_name).table(self.table_name).index_create("mobile_number").run(self._connection)
                r.db(self.db_name).table(self.table_name).index_create("instance_id").run(self._connection)
                r.db(self.db_name).table(self.table_name).index_wait().run(self._connection)
                logger.info(f"Created RethinkDB table: {self.table_name} with indexes")
            else:
                logger.info(f"RethinkDB table '{self.table_name}' already exists")
        except Exception as e:
            logger.error(f"Error ensuring table exists: {e}")
            raise

    def get_connection(self):
        """
        Get the RethinkDB connection, reconnecting if necessary.
        Returns a new connection object for thread safety.
        """
        try:
            # Test if connection is still alive by running a simple query
            if self._connection and self._connection.is_open():
                return self._connection
        except Exception:
            pass

        # Reconnect
        logger.info("RethinkDB connection lost, attempting reconnect...")
        self._connect()
        return self._connection

    def new_connection(self):
        """
        Create a new independent connection for use in separate threads.
        Each thread should have its own connection for change streams.
        """
        connect_args = {
            "host": self.host,
            "port": self.port,
            "db": self.db_name,
            "timeout": self.timeout,
        }
        if self.user and self.password:
            connect_args["user"] = self.user
            connect_args["password"] = self.password

        return r.connect(**connect_args)

    def close(self):
        """Close the RethinkDB connection."""
        if self._connection:
            try:
                self._connection.close()
                logger.info("RethinkDB connection closed")
            except Exception as e:
                logger.error(f"Error closing RethinkDB connection: {e}")
            finally:
                self._connection = None


def get_rethinkdb_connection() -> RethinkDBConnection:
    """Get the singleton RethinkDB connection manager."""
    return RethinkDBConnection()
