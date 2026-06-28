from typing import Dict, Optional
import hashlib


# Country phone prefixes (ISO alpha-2 to international dialing code)
COUNTRY_PHONE_PREFIXES = {
    'US': '+1', 'CA': '+1',
    'GB': '+44', 'UK': '+44',
    'AU': '+61', 'SG': '+65', 'AE': '+971',
    'IN': '+91', 'MY': '+60', 'TH': '+66', 'PH': '+63',
    'ID': '+62', 'VN': '+84', 'CN': '+86', 'HK': '+852',
    'JP': '+81', 'KR': '+82', 'DE': '+49', 'FR': '+33',
    'IT': '+39', 'ES': '+34', 'NL': '+31', 'BE': '+32',
    'CH': '+41', 'AT': '+43', 'SE': '+46', 'NO': '+47',
    'DK': '+45', 'FI': '+358', 'PL': '+48', 'CZ': '+420',
    'HU': '+36', 'GR': '+30', 'PT': '+351', 'IE': '+353',
    'TR': '+90', 'IL': '+972', 'SA': '+966', 'EG': '+20',
    'ZA': '+27', 'NG': '+234', 'KE': '+254', 'GH': '+233',
    'RU': '+7', 'UA': '+380', 'BR': '+55', 'MX': '+52',
    'AR': '+54', 'CO': '+57', 'CL': '+56', 'PE': '+51',
    'NZ': '+64', 'PK': '+92', 'BD': '+880', 'LK': '+94',
    'NP': '+977', 'MM': '+95', 'KH': '+855', 'LA': '+856',
    'BN': '+673', 'TW': '+886', 'MY': '+60', 'QA': '+974',
    'KW': '+965', 'BH': '+973', 'OM': '+968', 'JO': '+962',
    'LB': '+961', 'CY': '+357', 'IS': '+354', 'MT': '+356',
    'LU': '+352', 'RO': '+40', 'BG': '+359', 'HR': '+385',
    'SI': '+386', 'SK': '+421', 'EE': '+372', 'LV': '+371',
    'LT': '+370', 'AL': '+355', 'MK': '+389', 'RS': '+381',
    'BA': '+387', 'ME': '+382', 'AM': '+374', 'GE': '+995',
    'KZ': '+7', 'UZ': '+998', 'KG': '+996', 'TJ': '+992',
    'TM': '+993', 'AZ': '+994', 'AF': '+93', 'NP': '+977',
    'BT': '+975', 'MV': '+960', 'YE': '+967', 'SY': '+963',
    'IQ': '+964', 'IR': '+98', 'AO': '+244', 'CM': '+237',
    'CI': '+225', 'SN': '+221', 'ML': '+223', 'BF': '+226',
    'NE': '+227', 'TD': '+235', 'CF': '+236', 'CG': '+242',
    'CD': '+243', 'GA': '+241', 'GQ': '+240', 'CV': '+238',
    'ST': '+239', 'GW': '+245', 'MR': '+222', 'MA': '+212',
    'DZ': '+213', 'TN': '+216', 'LY': '+218', 'GM': '+220',
    'LR': '+231', 'SL': '+232', 'GH': '+233', 'BJ': '+229',
    'TG': '+228', 'ER': '+291', 'DJ': '+253', 'SO': '+252',
    'ET': '+251', 'SS': '+211', 'SD': '+249', 'RW': '+250',
    'UG': '+256', 'BI': '+257', 'MZ': '+258', 'ZM': '+260',
    'MW': '+265', 'ZW': '+263', 'NA': '+264', 'SZ': '+268',
    'LS': '+266', 'BW': '+267', 'SC': '+248', 'MU': '+230',
    'YT': '+262', 'RE': '+262', 'MG': '+261', 'FM': '+691',
    'PW': '+680', 'KI': '+686', 'MH': '+692', 'NR': '+674',
    'TV': '+688', 'TO': '+676', 'WS': '+685', 'VU': '+678',
    'FJ': '+679', 'PG': '+675', 'SB': '+677', 'CK': '+682',
    'NU': '+683', 'AS': '+684', 'TK': '+690', 'WF': '+681',
    'PN': '+64', 'NF': '+672', 'CX': '+61', 'CC': '+61',
    'HM': '+672', 'AQ': '+672', 'CK': '+682', 'NU': '+683',
    'KI': '+686', 'TV': '+688', 'PW': '+680', 'FM': '+691',
    'MH': '+692', 'MP': '+1', 'GU': '+1', 'VI': '+1',
    'PR': '+1', 'AG': '+1', 'DM': '+1', 'GD': '+1',
    'KN': '+1', 'LC': '+1', 'VC': '+1', 'TT': '+1',
    'BB': '+1', 'BM': '+1', 'BS': '+1', 'BZ': '+501',
    'CR': '+506', 'DO': '+1', 'SV': '+503', 'GT': '+502',
    'HN': '+504', 'NI': '+505', 'PA': '+507', 'GY': '+592',
    'SR': '+597', 'UY': '+598', 'VE': '+58', 'BO': '+591',
    'EC': '+593', 'PE': '+51', 'PY': '+595', 'AR': '+54',
    'CL': '+56', 'CO': '+57', 'CU': '+53', 'JM': '+1',
    'HT': '+509', 'DO': '+1', 'PR': '+1', 'MX': '+52',
}


def get_phone_prefix(country_code: str) -> Optional[str]:
    """
    Convert ISO 3166-1 alpha-2 country code to international phone prefix.

    Args:
        country_code: 2-letter ISO country code (e.g., 'US', 'GB') or phone prefix (e.g., '+1')

    Returns:
        International phone prefix (e.g., '+1', '+44') or None if not found
    """
    if not country_code:
        return None

    country_code = country_code.upper().strip()

    # If already a phone prefix, return as-is
    if country_code.startswith('+'):
        return country_code

    return COUNTRY_PHONE_PREFIXES.get(country_code)


# ISO 3166-1 alpha-2 to alpha-3 conversion map
ISO_COUNTRY_CODES = {
    # Common countries
    'US': 'USA',
    'GB': 'GBR',
    'UK': 'GBR',
    'CA': 'CAN',
    'AU': 'AUS',
    'DE': 'DEU',
    'FR': 'FRA',
    'IT': 'ITA',
    'ES': 'ESP',
    'JP': 'JPN',
    'CN': 'CHN',
    'IN': 'IND',
    'BR': 'BRA',
    'RU': 'RUS',
    'MX': 'MEX',
    'KR': 'KOR',
    'NL': 'NLD',
    'BE': 'BEL',
    'CH': 'CHE',
    'AT': 'AUT',
    'SE': 'SWE',
    'NO': 'NOR',
    'DK': 'DNK',
    'FI': 'FIN',
    'PL': 'POL',
    'CZ': 'CZE',
    'HU': 'HUN',
    'GR': 'GRC',
    'PT': 'PRT',
    'IE': 'IRL',
    'TR': 'TUR',
    'IL': 'ISR',
    'SA': 'SAU',
    'AE': 'ARE',
    'EG': 'EGY',
    'ZA': 'ZAF',
    'NG': 'NGA',
    'KE': 'KEN',
    'GH': 'GHA',
    'TH': 'THA',
    'SG': 'SGP',
    'MY': 'MYS',
    'PH': 'PHL',
    'ID': 'IDN',
    'VN': 'VNM',
    'PK': 'PAK',
    'BD': 'BGD',
    'LK': 'LKA',
    'NP': 'NPL',
    'MM': 'MMR',
    'KH': 'KHM',
    'LA': 'LAO',
    'BN': 'BRN',

    # Additional countries
    'AF': 'AFG',
    'AL': 'ALB',
    'DZ': 'DZA',
    'AD': 'AND',
    'AO': 'AGO',
    'AG': 'ATG',
    'AR': 'ARG',
    'AM': 'ARM',
    'AW': 'ABW',
    'AZ': 'AZE',
    'BS': 'BHS',
    'BH': 'BHR',
    'BB': 'BRB',
    'BY': 'BLR',
    'BZ': 'BLZ',
    'BJ': 'BEN',
    'BT': 'BTN',
    'BO': 'BOL',
    'BA': 'BIH',
    'BW': 'BWA',
    'BN': 'BRN',
    'BG': 'BGR',
    'BF': 'BFA',
    'BI': 'BDI',
    'CV': 'CPV',
    'CM': 'CMR',
    'KY': 'CYM',
    'CF': 'CAF',
    'TD': 'TCD',
    'CL': 'CHL',
    'CO': 'COL',
    'KM': 'COM',
    'CG': 'COG',
    'CR': 'CRI',
    'CI': 'CIV',
    'HR': 'HRV',
    'CU': 'CUB',
    'CY': 'CYP',
    'CZ': 'CZE',
    'DJ': 'DJI',
    'DM': 'DMA',
    'DO': 'DOM',
    'EC': 'ECU',
    'SV': 'SLV',
    'GQ': 'GNQ',
    'ER': 'ERI',
    'EE': 'EST',
    'ET': 'ETH',
    'FJ': 'FJI',
    'GA': 'GAB',
    'GM': 'GMB',
    'GE': 'GEO',
    'GD': 'GRD',
    'GT': 'GTM',
    'GN': 'GIN',
    'GW': 'GNB',
    'GY': 'GUY',
    'HT': 'HTI',
    'HN': 'HND',
    'IS': 'ISL',
    'IQ': 'IRQ',
    'JM': 'JAM',
    'JO': 'JOR',
    'KZ': 'KAZ',
    'KI': 'KIR',
    'KW': 'KWT',
    'KG': 'KGZ',
    'LV': 'LVA',
    'LB': 'LBN',
    'LS': 'LSO',
    'LR': 'LBR',
    'LI': 'LIE',
    'LT': 'LTU',
    'LU': 'LUX',
    'MK': 'MKD',
    'MG': 'MDG',
    'MW': 'MWI',
    'MV': 'MDV',
    'ML': 'MLI',
    'MT': 'MLT',
    'MH': 'MHL',
    'MR': 'MRT',
    'MU': 'MUS',
    'FM': 'FSM',
    'MD': 'MDA',
    'MC': 'MCO',
    'MN': 'MNG',
    'ME': 'MNE',
    'MA': 'MAR',
    'MZ': 'MOZ',
    'NA': 'NAM',
    'NR': 'NRU',
    'NZ': 'NZL',
    'NI': 'NIC',
    'NE': 'NER',
    'NG': 'NGA',
    'KP': 'PRK',
    'OM': 'OMN',
    'PA': 'PAN',
    'PG': 'PNG',
    'PY': 'PRY',
    'PE': 'PER',
    'QA': 'QAT',
    'RO': 'ROU',
    'RW': 'RWA',
    'KN': 'KNA',
    'LC': 'LCA',
    'VC': 'VCT',
    'WS': 'WSM',
    'SM': 'SMR',
    'ST': 'STP',
    'SN': 'SEN',
    'RS': 'SRB',
    'SC': 'SYC',
    'SL': 'SLE',
    'SK': 'SVK',
    'SI': 'SVN',
    'SB': 'SLB',
    'SO': 'SOM',
    'SS': 'SSD',
    'SD': 'SDN',
    'SR': 'SUR',
    'SZ': 'SWZ',
    'SY': 'SYR',
    'TJ': 'TJK',
    'TZ': 'TZA',
    'TG': 'TGO',
    'TO': 'TON',
    'TT': 'TTO',
    'TN': 'TUN',
    'TM': 'TKM',
    'TV': 'TUV',
    'UG': 'UGA',
    'UA': 'UKR',
    'UY': 'URY',
    'UZ': 'UZB',
    'VA': 'VAT',
    'VE': 'VEN',
    'VG': 'VGB',
    'VI': 'VIR',
    'YE': 'YEM',
    'ZM': 'ZMB',
    'ZW': 'ZWE'
}


def convert_to_alpha3(country_code: str) -> str:
    """
    Convert ISO 3166-1 alpha-2 country code to alpha-3 format.

    Args:
        country_code: 2-letter country code (e.g., 'US', 'GB')

    Returns:
        3-letter country code (e.g., 'USA', 'GBR') or original if not found
    """
    if not country_code:
        return country_code

    # Normalize input
    country_code = country_code.upper().strip()

    # Return alpha-3 code if found in mapping
    if country_code in ISO_COUNTRY_CODES:
        return ISO_COUNTRY_CODES[country_code]

    # If it's already 3 characters, return as-is
    if len(country_code) == 3:
        return country_code

    # If not found, return original (could be invalid or already alpha-3)
    return country_code


def generate_passport_hash(passport_country: str, passport_number: str) -> str:
    """
    [DEPRECATED] Generate a SHA-256 hash from country code and passport number.

    DEPRECATED: passport_hash column removed from user_identity_index table.
    Face biometrics is now used as the primary identity uniqueness constraint
    via the trg_face_biometrics_cross_identity_check trigger.

    This function is kept for backward compatibility only and should not be used.

    Args:
        passport_country: 3-letter ISO country code
        passport_number: Passport number

    Returns:
        SHA-256 hash (64 characters)
    """
    # Normalize inputs
    passport_country = convert_to_alpha3(passport_country).upper().strip()
    passport_number = passport_number.upper().strip()

    # Create composite string
    composite = f"{passport_country}_{passport_number}"

    # Generate SHA-256 hash
    hash_obj = hashlib.sha256(composite.encode('utf-8'))
    return hash_obj.hexdigest()


def convert_alpha3_to_alpha2(country_code: str) -> Optional[str]:
    """
    Convert ISO 3166-1 alpha-3 country code to alpha-2 format.

    Args:
        country_code: 3-letter country code (e.g., 'USA', 'GBR')

    Returns:
        2-letter country code (e.g., 'US', 'GB') or None if not found
    """
    if not country_code:
        return None

    country_code = country_code.upper().strip()

    # Build reverse mapping (alpha-3 to alpha-2)
    alpha3_to_alpha2 = {v: k for k, v in ISO_COUNTRY_CODES.items()}

    return alpha3_to_alpha2.get(country_code)


def country_name_to_code(country_name: str) -> Optional[str]:
    """
    Convert country name to ISO 3166-1 alpha-2 country code.

    Args:
        country_name: Full country name (e.g., 'United States', 'Singapore')

    Returns:
        2-letter country code (e.g., 'US', 'SG') or None if not found
    """
    if not country_name:
        return None

    # Common country name to alpha-2 mapping
    COUNTRY_NAME_MAP = {
        # Common names
        'united states': 'US', 'usa': 'US', 'america': 'US',
        'united kingdom': 'GB', 'uk': 'GB', 'britain': 'GB', 'england': 'GB',
        'singapore': 'SG',
        'australia': 'AU',
        'india': 'IN',
        'malaysia': 'MY',
        'thailand': 'TH',
        'philippines': 'PH',
        'indonesia': 'ID',
        'vietnam': 'VN',
        'china': 'CN',
        'hong kong': 'HK',
        'japan': 'JP',
        'south korea': 'KR', 'korea': 'KR',
        'taiwan': 'TW',
        'united arab emirates': 'AE', 'uae': 'AE',
        'saudi arabia': 'SA',
        'germany': 'DE',
        'france': 'FR',
        'italy': 'IT',
        'spain': 'ES',
        'netherlands': 'NL',
        'belgium': 'BE',
        'switzerland': 'CH',
        'canada': 'CA',
        'mexico': 'MX',
        'brazil': 'BR',
        'argentina': 'AR',
        'russia': 'RU',
        'myanmar': 'MM', 'burma': 'MM',
    }

    normalized = country_name.lower().strip()
    return COUNTRY_NAME_MAP.get(normalized)


def validate_iso_country_code(country_code: str) -> bool:
    """
    Validate if a country code is in ISO format (2 or 3 letters).

    Args:
        country_code: Country code to validate

    Returns:
        True if valid ISO format
    """
    if not country_code:
        return False

    country_code = country_code.upper().strip()

    # Check if it's a valid alpha-2 or alpha-3 code
    if len(country_code) == 2:
        return country_code in ISO_COUNTRY_CODES
    elif len(country_code) == 3:
        return country_code in ISO_COUNTRY_CODES.values()

    return False