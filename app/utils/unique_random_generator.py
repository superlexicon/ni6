import random
from typing import Optional


class UniqueRandomGenerator:
    def __init__(self):
        self.used_numbers = set()

    def generate_random_number(
        self,
        length: int,
        allowed_digits: Optional[str] = None
    ) -> str:
        """
        Generate a random number string from allowed digits.

        Args:
            length: Length of the random number to generate
            allowed_digits: String of allowed digit characters (default: '0123456789')

        Returns:
            Random number string using only allowed digits

        Raises:
            ValueError: If allowed_digits is empty or length is invalid
        """
        if allowed_digits is None:
            allowed_digits = '0123456789'

        if not allowed_digits:
            raise ValueError("allowed_digits cannot be empty")

        if length <= 0:
            raise ValueError("length must be positive")

        # Calculate maximum unique combinations
        max_combinations = len(allowed_digits) ** length

        while True:
            random_number = ''.join(random.choices(allowed_digits, k=length))
            if random_number not in self.used_numbers:
                self.used_numbers.add(random_number)
                return random_number
            # Clear used numbers if we've exhausted all combinations
            if len(self.used_numbers) >= max_combinations:
                self.used_numbers.clear()
