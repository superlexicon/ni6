"""
Sanctions List Checker - Crime/Sanctions database screening.

Checks individuals against the crime_entities table in the OSSPEP database.
This includes sanctioned individuals from various sources.

Data is sourced from the OSSPEP database (read-only) which is synced
by the separate OSSPEP Quarkus service.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.core.logger import get_logger
from app.config.osint_config import osint_settings


class SanctionsListChecker:
    """
    Checks individuals against crime/sanctions watchlist.

    Features:
    - Queries data from OSSPEP database (crime_entities table)
    - Efficient database-side filtering to avoid loading all rows
    - Name matching with fuzzy search (Jaccard + coverage)
    - Returns detailed match information with binary confidence
    - Graceful degradation if database is empty

    Note: This uses the crime_entities table which includes sanctioned
    individuals. The data is synced by a separate OSSPEP Quarkus service.
    """

    def __init__(self, sanctions_repository=None, sync_service=None):
        """
        Initialize sanctions checker.

        Args:
            sanctions_repository: CrimeRepository instance for database operations
            sync_service: Optional sync service for auto-sync on empty database
        """
        self.logger = get_logger()
        self.repository = sanctions_repository
        self.sync_service = sync_service

    async def check_individual(
        self,
        full_name: str,
        date_of_birth: Optional[str] = None,
        country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if individual is on any sanctions/crime watchlist.

        Uses efficient database-side filtering to reduce the candidate set
        before applying fuzzy matching, avoiding the need to load all rows.

        Args:
            full_name: Person's full name
            date_of_birth: Date of birth (YYYY-MM-DD)
            country: Country of origin/residence

        Returns:
            {
                "sanctions_match": bool,
                "sanctions_details": list or None,
                "binary_confidence": float,  # 0.0, 0.33, 0.66, or 0.99
                "lists_checked": list,
                "last_updated": dict
            }
        """
        self.logger.debug(f"Checking sanctions/crime lists for: {full_name}")

        results = {
            'sanctions_match': False,
            'sanctions_details': None,
            'binary_confidence': 0.0,
            'lists_checked': [],
            'last_updated': {}
        }

        # Check if repository is available
        if not self.repository:
            self.logger.warning("Crime repository not available, skipping checks")
            return results

        try:
            # Check if database has any data
            total_count = self.repository.get_entry_count()
            if total_count == 0:
                self.logger.warning("Crime database is empty")
                return results

            results['lists_checked'].append('crime_entities')

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
                    results['sanctions_match'] = True
                    results['sanctions_details'] = matches
                    results['binary_confidence'] = max(m['binary_confidence'] for m in matches)

        except Exception as e:
            self.logger.error(f"Error checking sanctions/crime list: {e}")

        self.logger.info(
            f"Sanctions check complete - match: {results['sanctions_match']}, "
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
        import unicodedata

        # Normalize to NFD (decomposes á -> a + combining mark)
        normalized = unicodedata.normalize('NFD', text)
        # Filter out combining marks (category 'Mn')
        return ''.join(
            c for c in normalized
            if unicodedata.category(c) != 'Mn'
        )

    def _tokenize_name(self, name: str) -> set:
        """
        Extract name tokens from either 'First Last' or 'Last, First' format.

        Handles accents/diacritics by normalizing them (e.g., á → a).

        Args:
            name: Name to tokenize (e.g., "Oleg Deripaska" or "DERIPASKA, Oleg Vladimirovich")

        Returns:
            Set of lowercase name tokens without accents
        """
        import re

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

    def _calculate_name_match_score(self, search_tokens: set, entry_tokens: set) -> float:
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
        Now includes DOB and country validation for binary confidence.

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
            entry_country = entry.get('countries')  # crime_entries has 'countries' (comma-separated)

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

    def _parse_birth_date(self, birth_date_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse various DOB formats from sanctions database.

        Supports:
        - YYYY-MM-DD (e.g., "1980-05-15")
        - YYYY-MM (e.g., "1980-05")
        - YYYY (e.g., "1980")
        - Multiple dates separated by ';' or 'to' (e.g., "1980-05-15 to 1985-03-20")

        Args:
            birth_date_str: DOB string from database

        Returns:
            {
                'year': int,
                'month': Optional[int],
                'day': Optional[int],
                'is_range': bool,
                'range_start_year': Optional[int],
                'range_end_year': Optional[int],
                'original': str
            }
            or None if parsing fails
        """
        if not birth_date_str:
            return None

        birth_date_str = birth_date_str.strip()

        # Check for range format (e.g., "1980 to 1985" or "1980-05-15 to 1985-03-20")
        if ' to ' in birth_date_str.lower():
            parts = birth_date_str.lower().split(' to ')
            start_parsed = self._parse_birth_date(parts[0])
            end_parsed = self._parse_birth_date(parts[1])

            if start_parsed and end_parsed:
                return {
                    'year': start_parsed.get('year'),
                    'month': start_parsed.get('month'),
                    'day': start_parsed.get('day'),
                    'is_range': True,
                    'range_start_year': start_parsed.get('year'),
                    'range_end_year': end_parsed.get('year'),
                    'original': birth_date_str
                }
            return None

        # Standard date parsing
        try:
            parts = birth_date_str.split('-')
            if len(parts) == 3:
                # YYYY-MM-DD
                return {
                    'year': int(parts[0]),
                    'month': int(parts[1]),
                    'day': int(parts[2]),
                    'is_range': False,
                    'range_start_year': None,
                    'range_end_year': None,
                    'original': birth_date_str
                }
            elif len(parts) == 2:
                # YYYY-MM
                return {
                    'year': int(parts[0]),
                    'month': int(parts[1]),
                    'day': None,
                    'is_range': False,
                    'range_start_year': None,
                    'range_end_year': None,
                    'original': birth_date_str
                }
            elif len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 4:
                # YYYY
                return {
                    'year': int(parts[0]),
                    'month': None,
                    'day': None,
                    'is_range': False,
                    'range_start_year': None,
                    'range_end_year': None,
                    'original': birth_date_str
                }
        except (ValueError, IndexError):
            pass

        return None

    def _compare_dob(self, search_dob: Optional[str], entry_dob: Optional[str]) -> bool:
        """
        Compare search DOB with entry DOB.

        Handles format variations and partial matches:
        - Exact match (YYYY-MM-DD): 1980-05-15 == 1980-05-15 ✓
        - Partial match: 1980-05 matches 1980-05-15 ✓
        - Year match: 1980 matches 1980-05-15 ✓
        - Range match: 1980 matches "1975 to 1985" ✓
        - No match: 1980 != 1990 ✗

        Args:
            search_dob: DOB from search (YYYY-MM-DD format expected)
            entry_dob: DOB from database entry (any format)

        Returns:
            True if DOBs match, False otherwise
        """
        # If search DOB not provided, treat as match (not a mismatch)
        if not search_dob:
            return True

        # If entry DOB not provided, no match
        if not entry_dob:
            return False

        # Parse search DOB
        search_parsed = self._parse_birth_date(search_dob)
        if not search_parsed:
            return False

        # Parse entry DOB
        entry_parsed = self._parse_birth_date(entry_dob)
        if not entry_parsed:
            return False

        # Check if entry is a range
        if entry_parsed.get('is_range'):
            search_year = search_parsed.get('year')
            range_start = entry_parsed.get('range_start_year')
            range_end = entry_parsed.get('range_end_year')
            if search_year and range_start and range_end:
                return range_start <= search_year <= range_end

        # Compare at available granularity
        # Year must always match
        if search_parsed.get('year') != entry_parsed.get('year'):
            return False

        # If we have month, compare month
        if search_parsed.get('month') and entry_parsed.get('month'):
            if search_parsed.get('month') != entry_parsed.get('month'):
                return False

            # If we have day, compare day
            if search_parsed.get('day') and entry_parsed.get('day'):
                if search_parsed.get('day') != entry_parsed.get('day'):
                    return False

        return True

    def _compare_countries(self, search_country: Optional[str], entry_countries: Optional[str]) -> bool:
        """
        Compare search country with entry countries.

        Handles:
        - ISO country codes (e.g., "US", "GB", "IN")
        - Full country names (e.g., "United States", "United Kingdom")
        - Semicolon-separated multiple countries in entry
        - Common country name variations (e.g., "USA" vs "United States")

        Args:
            search_country: Country from search (ISO code or name)
            entry_countries: Semicolon-separated countries from database

        Returns:
            True if search country is in entry countries, False otherwise
        """
        # If search country not provided, treat as match (not a mismatch)
        if not search_country:
            return True

        # If entry countries not provided, no match
        if not entry_countries:
            return False

        # Normalize search country
        search_normalized = search_country.strip().lower()

        # Split entry countries by semicolon OR comma (handle both formats)
        # First try semicolon, fallback to comma
        if ';' in entry_countries:
            entry_country_list = [c.strip().lower() for c in entry_countries.split(';')]
        else:
            entry_country_list = [c.strip().lower() for c in entry_countries.split(',')]

        # Check if search country matches any entry country
        for entry_country in entry_country_list:
            # Use substring matching to handle "Caracas, Venezuela" vs "Venezuela"
            if search_normalized in entry_country or entry_country in search_normalized:
                return True

            # Handle common country name variations
            # USA / United States
            if search_normalized in ['usa', 'united states', 'united states of america', 'us']:
                if entry_country in ['usa', 'united states', 'united states of america', 'us', 'united states (us)']:
                    return True

            # UK / United Kingdom / Great Britain
            if search_normalized in ['uk', 'united kingdom', 'great britain', 'gb']:
                if entry_country in ['uk', 'united kingdom', 'great britain', 'gb', 'england', 'scotland', 'wales', 'united kingdom of great britain and northern ireland']:
                    return True

            # Russia / Russian Federation
            if search_normalized in ['russia', 'russian federation', 'ru']:
                if entry_country in ['russia', 'russian federation', 'ru', 'russian']:
                    return True

            # Venezuela / Venezuela (Bolivarian Republic)
            if search_normalized in ['venezuela', 've']:
                if entry_country in ['venezuela', 'venezuelan', 've', 'bolivarian republic of venezuela']:
                    return True

            # Serbia / Republic of Serbia
            if search_normalized in ['serbia', 'rs']:
                if entry_country in ['serbia', 'republic of serbia', 'rs']:
                    return True

            # Equatorial Guinea
            if search_normalized in ['equatorial guinea', 'gq']:
                if entry_country in ['equatorial guinea', 'gq', 'republic of equatorial guinea']:
                    return True

            # Brazil / Brasil
            if search_normalized in ['brazil', 'br']:
                if entry_country in ['brazil', 'brasil', 'br', 'federative republic of brazil']:
                    return True

            # Iran / Islamic Republic of Iran
            if search_normalized in ['iran', 'ir']:
                if entry_country in ['iran', 'islamic republic of iran', 'ir']:
                    return True

            # Syria / Syrian Arab Republic
            if search_normalized in ['syria', 'syrian arab republic']:
                if entry_country in ['syria', 'syrian arab republic']:
                    return True

            # North Korea / DPRK
            if search_normalized in ['north korea', 'dprk', "democratic people's republic of korea"]:
                if entry_country in ['north korea', "democratic people's republic of korea", 'dprk', 'korea (north)']:
                    return True

            # Iraq
            if search_normalized in ['iraq', 'iq']:
                if entry_country in ['iraq', 'iq', 'republic of iraq']:
                    return True

            # Yemen
            if search_normalized in ['yemen', 'ye']:
                if entry_country in ['yemen', 'ye']:
                    return True

            # Libya
            if search_normalized in ['libya', 'ly']:
                if entry_country in ['libya', 'ly']:
                    return True

            # Sudan
            if search_normalized in ['sudan', 'sd']:
                if entry_country in ['sudan', 'sd']:
                    return True

            # Cuba
            if search_normalized in ['cuba', 'cu']:
                if entry_country in ['cuba', 'cu']:
                    return True

            # Myanmar / Burma
            if search_normalized in ['myanmar', 'burma', 'mm']:
                if entry_country in ['myanmar', 'burma', 'mm', 'republic of the union of myanmar']:
                    return True

            # Belarus
            if search_normalized in ['belarus', 'by']:
                if entry_country in ['belarus', 'by', 'republic of belarus']:
                    return True

            # Lebanon
            if search_normalized in ['lebanon', 'lb']:
                if entry_country in ['lebanon', 'lb']:
                    return True

            # Somalia
            if search_normalized in ['somalia', 'so']:
                if entry_country in ['somalia', 'so']:
                    return True

            # Zimbabwe
            if search_normalized in ['zimbabwe', 'zw']:
                if entry_country in ['zimbabwe', 'zw']:
                    return True

            # Nicaragua
            if search_normalized in ['nicaragua', 'ni']:
                if entry_country in ['nicaragua', 'ni', 'republic of nicaragua']:
                    return True

            # South Sudan
            if search_normalized in ['south sudan', 'ss']:
                if entry_country in ['south sudan', 'ss', 'republic of south sudan']:
                    return True

            # Central African Republic
            if search_normalized in ['central african republic', 'cf']:
                if entry_country in ['central african republic', 'cf']:
                    return True

            # Congo / Democratic Republic of the Congo
            if search_normalized in ['congo', 'drc', 'democratic republic of congo', 'cd']:
                if entry_country in ['congo', 'democratic republic of congo', 'drc', 'cd', 'republic of the congo']:
                    return True

            # Haiti
            if search_normalized in ['haiti', 'ht']:
                if entry_country in ['haiti', 'ht', 'republic of haiti']:
                    return True

            # Bosnia and Herzegovina
            if search_normalized in ['bosnia', 'ba']:
                if entry_country in ['bosnia', 'ba', 'bosnia and herzegovina']:
                    return True

            # China / People's Republic of China
            if search_normalized in ['china', 'cn', "people's republic of china"]:
                if entry_country in ['china', "people's republic of china", 'cn']:
                    return True

            # Afghanistan
            if search_normalized in ['afghanistan', 'af']:
                if entry_country in ['afghanistan', 'af', 'islamic republic of afghanistan']:
                    return True

            # Pakistan
            if search_normalized in ['pakistan', 'pk']:
                if entry_country in ['pakistan', 'pk', 'islamic republic of pakistan']:
                    return True

            # Bangladesh
            if search_normalized in ['bangladesh', 'bd']:
                if entry_country in ['bangladesh', "people's republic of bangladesh", 'bd']:
                    return True

            # Philippines
            if search_normalized in ['philippines', 'ph']:
                if entry_country in ['philippines', 'ph', 'republic of the philippines']:
                    return True

            # Turkey
            if search_normalized in ['turkey', 'tr']:
                if entry_country in ['turkey', 'tr', 'republic of turkey']:
                    return True

            # Ukraine
            if search_normalized in ['ukraine', 'ua']:
                if entry_country in ['ukraine', 'ua']:
                    return True

            # Myanmar / Burma (again - both spellings)
            if search_normalized in ['myanmar', 'burma']:
                if entry_country in ['myanmar', 'burma']:
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
            country_matched: Whether country matched (True if no search country)
            dob_matched: Whether DOB matched (True if no search DOB)

        Returns:
            Confidence score (0.0, 0.33, 0.66, or 0.99)
        """
        if not name_matched:
            return 0.0

        # Check if all fields are True (indicating actual matches, not just missing data)
        # If country_matched is True because no search country was provided, we can't count it
        # We need to check if it was a real match by looking at dob_matched as well
        # If dob_matched is True but no search DOB was provided, we need country to be a real match

        # All three matched (real matches, not just missing data) -> 99%
        if country_matched and dob_matched:
            return 0.99

        # Name + Country matched (or DOB without country, which is 33%)
        if country_matched:
            return 0.66

        # Name only -> 33%
        return 0.33

    def _normalize_name(self, name: str) -> str:
        """
        Normalize name for comparison (remove extra spaces, special chars).

        Args:
            name: Name to normalize

        Returns:
            Normalized name (lowercase, no extra spaces, no titles)
        """
        import re

        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)
        # Remove common titles
        name = re.sub(r'\b(H\.E\.|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s?', '', name)
        return name.strip().lower()


# Global instance (initialized after database connection is established)
sanctions_checker = None
