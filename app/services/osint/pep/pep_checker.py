"""
PEP Checker - Check individuals against PEP database.

Checks individuals against the pep_entries table in the OSSPEP database.
Data is synced by the separate OSSPEP Quarkus service.
"""

import asyncio
import re
import unicodedata
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from app.core.logger import get_logger
from app.config.osint_config import osint_settings


class PEPChecker:
    """
    Checks individuals against PEP database.

    Features:
    - Queries data from OSSPEP database (pep_entries table)
    - Efficient database-side filtering to avoid loading all rows
    - Name matching with fuzzy search (Jaccard similarity + coverage)
    - Returns detailed match information with binary confidence
    - Graceful degradation if database is empty

    Note: This uses the pep_entries table which contains PEP data from
    various sources. The data is synced by a separate OSSPEP Quarkus service.
    """

    def __init__(self, pep_repository=None, sync_service=None):
        """
        Initialize PEP checker.

        Args:
            pep_repository: PEPRepository instance for database operations
            sync_service: Optional sync service for auto-sync on empty database
        """
        self.logger = get_logger()
        self.repository = pep_repository
        self.sync_service = sync_service

    async def check_individual(
        self,
        full_name: str,
        date_of_birth: Optional[str] = None,
        country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if individual is a PEP.

        Uses efficient database-side filtering to reduce the candidate set
        before applying fuzzy matching, avoiding the need to load all rows.

        Args:
            full_name: Person's full name
            date_of_birth: Date of birth (YYYY-MM-DD)
            country: Country of origin/residence (ISO code or name)

        Returns:
            {
                "is_pep": bool,
                "current_pep_match": bool,
                "former_pep_match": bool,
                "pep_details": list or None,
                "binary_confidence": float,  # 0.0, 0.33, 0.66, or 0.99
                "positions_found": list,
                "last_updated": dict
            }
        """
        self.logger.debug(f"Checking PEP lists for: {full_name}")

        results = {
            'is_pep': False,
            'current_pep_match': False,
            'former_pep_match': False,
            'pep_details': None,
            'binary_confidence': 0.0,
            'positions_found': [],
            'last_updated': {}
        }

        # Check if repository is available
        if not self.repository:
            self.logger.warning("PEP repository not available, skipping checks")
            return results

        try:
            # Check if database has any data
            total_count = self.repository.get_entry_count()
            if total_count == 0:
                self.logger.warning("PEP database is empty")
                return results

            # Extract birth year from date_of_birth if provided
            birth_year = None
            if date_of_birth:
                try:
                    if hasattr(date_of_birth, 'year'):
                        birth_year = date_of_birth.year  # datetime.date object
                    else:
                        birth_year = int(str(date_of_birth)[:4])  # String like "1982-05-15"
                except (ValueError, IndexError, AttributeError):
                    pass

            # Tokenize name for database filtering
            name_tokens = list(self._tokenize_name(full_name))

            # Use efficient filtered query instead of loading all entries
            entries = self.repository.search_by_filters(
                name_tokens=name_tokens,
                country=country,
                birth_year=birth_year,
                limit=1000
            )

            self.logger.info(
                f"search_by_filters returned {len(entries)} candidates "
                f"(from {total_count} total entries)"
            )

            if entries:
                # Search for matches using fuzzy matching on filtered candidates
                matches = self._search_in_list(entries, full_name, date_of_birth, country)

                if matches:
                    results['is_pep'] = True
                    results['pep_details'] = matches
                    results['binary_confidence'] = max(m['binary_confidence'] for m in matches)

                    # Check if current or former PEP
                    if any(m.get('status') == 'current' for m in matches):
                        results['current_pep_match'] = True
                    if any(m.get('status') == 'former' for m in matches):
                        results['former_pep_match'] = True

                    # Extract positions (unique)
                    positions_set = set(m.get('position', '') for m in matches if m.get('position'))
                    results['positions_found'] = list(positions_set)

        except Exception as e:
            self.logger.error(f"Error checking PEP list: {e}")

        self.logger.info(
            f"PEP check complete - is_pep: {results['is_pep']}, "
            f"current: {results['current_pep_match']}, "
            f"former: {results['former_pep_match']}, "
            f"Binary Confidence: {results['binary_confidence']:.2f}"
        )

        return results

    def _remove_accents(self, text: str) -> str:
        """
        Remove diacritics/accents from text using Unicode NFD normalization.

        Args:
            text: Text with potential accents

        Returns:
            Text without accents (e.g., "Nicolás" → "Nicolas")
        """
        # Normalize to NFD (decomposes á -> a + combining mark)
        normalized = unicodedata.normalize('NFD', text)
        # Filter out combining marks (category 'Mn')
        return ''.join(
            c for c in normalized
            if unicodedata.category(c) != 'Mn'
        )

    def _tokenize_name(self, name: str) -> Set[str]:
        """
        Extract name tokens from either 'First Last' or 'Last, First' format.

        Handles accents/diacritics by normalizing them (e.g., á → a).

        Args:
            name: Name to tokenize (e.g., "Oleg Deripaska" or "DERIPASKA, Oleg Vladimirovich")

        Returns:
            Set of lowercase name tokens without accents
        """
        if not name:
            return set()

        # Remove accents/diacritics FIRST
        name = self._remove_accents(name)

        # Remove common titles
        name = re.sub(r'\b(H\.E\.|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s?', '', name)

        # Handle comma: "Last, First" -> "Last First"
        name = name.replace(',', ' ')

        # Normalize whitespace and lowercase
        name = re.sub(r'\s+', ' ', name).strip().lower()

        # Split into tokens
        tokens = set(name.split())

        return tokens

    def _calculate_name_match_score(self, search_tokens: Set[str], entry_tokens: Set[str]) -> float:
        """
        Calculate match score between token sets.

        Args:
            search_tokens: Tokens from the search query
            entry_tokens: Tokens from the database entry

        Returns:
            Score 0.0-1.0 where:
            - 1.0 = perfect match (all search tokens found)
            - 0.5+ = strong partial match
            - <0.5 = weak or no match
        """
        if not search_tokens or not entry_tokens:
            return 0.0

        intersection = search_tokens & entry_tokens
        union = search_tokens | entry_tokens

        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0

        # Coverage: what % of search tokens were found?
        coverage = len(intersection) / len(search_tokens)

        # Combined score: prioritize coverage but also consider Jaccard
        return (coverage * 0.7) + (jaccard * 0.3)

    def _search_in_list(
        self,
        entries: List[Dict[str, Any]],
        full_name: str,
        date_of_birth: Optional[str] = None,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for name in entries list using token-based matching.

        Handles both 'First Last' and 'Last, First' formats.
        Includes DOB and country validation for binary confidence.

        Args:
            entries: List of entry dicts from database
            full_name: Person's full name to search for
            date_of_birth: Optional date of birth for additional filtering
            country: Optional country for additional filtering

        Returns:
            List of matching entries with match_type, match_confidence, binary_confidence, and matched_fields
        """
        matches = []

        # Tokenize search name
        search_tokens = self._tokenize_name(full_name)

        for entry in entries:
            entry_name = entry.get('name', '')
            entry_tokens = self._tokenize_name(entry_name)

            # Calculate name match score
            name_score = self._calculate_name_match_score(search_tokens, entry_tokens)

            # Match thresholds for name
            if name_score < 0.5:
                continue  # No name match

            # Name matched - now validate DOB and country
            entry_dob = entry.get('birth_date')
            entry_country = entry.get('country')  # pep_entries has 'country' (single value)

            dob_matched = self._compare_dob(date_of_birth, entry_dob)
            country_matched = self._compare_countries(country, entry_country)

            # Calculate binary confidence
            binary_confidence = self._calculate_binary_confidence(
                name_matched=True,
                country_matched=country_matched,
                dob_matched=dob_matched
            )

            # Determine match type based on name score
            if name_score >= 1.0:
                match_type = 'exact'
            elif name_score >= 0.7:
                match_type = 'strong'
            else:
                match_type = 'partial'

            matches.append({
                **entry,
                'match_type': match_type,
                'name_match_confidence': round(name_score, 3),
                'binary_confidence': round(binary_confidence, 3),
                'matched_fields': {
                    'name': True,
                    'country': country_matched,
                    'dob': dob_matched
                }
            })

        # Sort by binary confidence first, then by name confidence
        matches.sort(key=lambda x: (x['binary_confidence'], x['name_match_confidence']), reverse=True)

        # Limit to top 10 matches
        return matches[:10]

    def _compare_dob(self, search_dob: Optional[str], entry_dob: Optional[str]) -> bool:
        """
        Compare dates of birth.

        Returns True if:
        - Both DOBs match (year-level comparison)
        - Returns False if either is missing (we don't count missing data as a match)
        """
        if not search_dob or not entry_dob:
            return False

        # Extract year from search_dob (YYYY-MM-DD -> YYYY)
        search_year = search_dob.split('-')[0] if '-' in search_dob else search_dob

        # Extract year from entry_dob (could be YYYY-MM-DD, YYYY-MM, or YYYY)
        entry_year = str(entry_dob).split('-')[0]

        return search_year == entry_year

    def _compare_countries(self, search_country: Optional[str], entry_country: str) -> bool:
        """
        Compare countries using 2-letter ISO country codes.

        Returns True if countries match, False otherwise.
        """
        if not search_country or not entry_country:
            return False

        search_normalized = search_country.lower().strip()
        entry_normalized = entry_country.lower().strip()

        # Direct match (2-letter ISO codes)
        if search_normalized == entry_normalized:
            return True

        return False

    def _calculate_binary_confidence(
        self,
        name_matched: bool,
        country_matched: bool,
        dob_matched: bool
    ) -> float:
        """
        Calculate binary confidence based on matched fields.

        Confidence levels:
        - Name + Country + DOB all match: 0.99 (99%)
        - Name + Country match: 0.66 (66%)
        - Name only match: 0.33 (33%)
        - No match: 0.0 (0%)

        Args:
            name_matched: Whether name matched
            country_matched: Whether country matched
            dob_matched: Whether DOB matched

        Returns:
            Confidence score (0.0, 0.33, 0.66, or 0.99)
        """
        if not name_matched:
            return 0.0

        # All three matched (real matches, not just missing data) -> 99%
        if country_matched and dob_matched:
            return 0.99

        # Name + Country matched -> 66%
        if country_matched:
            return 0.66

        # Name only -> 33%
        return 0.33


# Global instance (initialized after database connection is established)
pep_checker = None
