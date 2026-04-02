
from datetime import datetime
MIN_AGE_LIMIT = 21


class DocumentValidator:
    @staticmethod
    def is_valid_age(dob_str: str, date_format: str = "%d%b%Y") -> bool:
        try:
            dob = datetime.strptime(dob_str, date_format)
            today = datetime.now()
            age = today.year - dob.year - \
                ((today.month, today.day) < (dob.month, dob.day))
            return age >= MIN_AGE_LIMIT
        except (ValueError, TypeError):
            return False
