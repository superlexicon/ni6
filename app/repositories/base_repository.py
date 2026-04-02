from typing import  Any, TypeVar, Generic
from mysql.connector.errors import Error as MySQLError
from app.core import logger

T = TypeVar('T')


class BaseRepository(Generic[T]):
    def __init__(self, table_name: str):
        self.logger = logger
        self.table_name = table_name

    def _record_exists(self, field: str, value: Any) -> bool:
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                query = f"SELECT 1 FROM {self.table_name} WHERE {field} = %s"
                with conn.cursor() as cursor:
                    cursor.execute(query, (value,))
                    return bool(cursor.fetchone())
        except MySQLError as e:
            self.logger.error(
                f"Error checking if {field} exists in {self.table_name}: {e}")
            return False

    def _delete_record(self, field: str, value: Any) -> bool:
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                query = f"DELETE FROM {self.table_name} WHERE {field} = %s"
                with conn.cursor() as cursor:
                    cursor.execute(query, (value,))
                    conn.commit()
                    return cursor.rowcount > 0
        except MySQLError as e:
            self.logger.error(
                f"Error deleting record from {self.table_name} where {field} = {value}: {e}")
            return False
