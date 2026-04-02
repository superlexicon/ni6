
from dataclasses import dataclass
from difflib import get_close_matches

from app.utils.string_matching import jaro_winkler_similarity
from typing import Optional, Pattern, Match


@dataclass
class TextMatchResult:
    matched_text: Optional[str]
    confidence: float
    is_match: bool


MATCH_THRESHOLD = 70.0


class TextMatcher:
    @staticmethod
    def calculate_match_percentage(extracted: str, target: str) -> float:
        if not extracted or not target:
            return 0.0

        extracted_lower = extracted.lower()
        target_lower = target.lower()

        if target_lower in extracted_lower:
            return 100.0

        candidates = extracted_lower.split()
        best_matches = get_close_matches(
            target_lower,
            candidates,
            n=1,
            cutoff=0.3
        )

        if not best_matches:
            return 0.0

        similarity = jaro_winkler_similarity(best_matches[0], target_lower)

        return round(similarity * 100.0, 2)

    @staticmethod
    def extract_field_with_regex(text: str, pattern: Pattern[str]) -> TextMatchResult:
        if not text or not pattern:
            return TextMatchResult(None, 0.0, False)

        match: Optional[Match[str]] = pattern.search(text)
        if not match:
            return TextMatchResult(None, 0.0, False)

        matched_text = match.group(1).replace(
            ' ', '') if match.groups() else match.group(0)
        return TextMatchResult(matched_text, 100.0, True)

    @classmethod
    def check_match(cls, extracted: str, target: str) -> TextMatchResult:
        confidence = cls.calculate_match_percentage(extracted, target)
        return TextMatchResult(
            matched_text=extracted,
            confidence=confidence,
            is_match=confidence >= MATCH_THRESHOLD
        )
