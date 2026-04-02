"""Country-specific address detection patterns for bank statements."""

import re
from typing import List, Dict

# === NEGATIVE PATTERNS (exclude non-address text) ===
NEGATIVE_KEYWORDS = [
    # Statement metadata
    'summary', 'statement period', 'account statement', 'statement', 'branch',
    'tel.', 'tel ', 'email:', 'email ', 'contact center', 'contact ',
    'page ', 'requested date', 'account name', 'account type',
    'account number', 'account no', 'savings', 'current', 'deposit',
    # Transaction keywords
    'balance', 'withdrawal', 'transaction', 'atm', 'debit', 'credit', 'transfer', 'payment',
    # Footer text
    'www', 'http', 'licensed', 'regulated', 'central bank',
    'signature', 'document generated', 'system',
]

# === COUNTRY-SPECIFIC PATTERNS ===

COUNTRY_PATTERNS: Dict[str, List[str]] = {
    'UAE': [
        r'\bpo\s+box\s+\d{4,6}',           # PO Box
        r'\b(flat|villa|office|shop)\s+\d+',  # Building types
        r'\b(abu dhabi|dubai|sharjah|al\s+ain)\b',  # Cities
        r'\bstreet\b|\broad\b|\bave\b',     # Street types
    ],
    'Thailand': [
        r'\b\d{5}\b',                       # Thai postal code
        r'\b(soi|village|district|province|subdistrict|tambon|amphoe)\b',
        r'\b(ayutthaya|nakhon|wang noi|bangkok|pathum wan)\b',
    ],
    'Singapore': [
        r'\b\d{6}\b',                       # Singapore postal code
        r'\bblock\s+\d+|#\d+-\d+',
        r'\b(street|st|road|rd|lane|ln|avenue|ave|drive|crescent|close|walk)\b',
        r'\b(jalan|lorong)\b',              # Malay words used in Singapore
        r'\b(singapore)\b',                 # Explicit Singapore mention
    ],
    'UK': [
        r'\b[A-Z]{1,2}\d{1,2}[ ]?\d[A-Z]{2}\b',  # UK postcode
    ],
    'Malaysia': [
        r'\b\d{5}\b',                       # Malaysian postal code
        r'\b(jalan|lorong)\b',
    ],
    'India': [
        r'\b\d{6}\b',                       # Indian PIN code (6 digits)
        # Street types
        r'\b(street|st|road|rd|lane|ln|avenue|ave|nagar|marg|marga|colony|sector|gali|mohalla|chowk)\b',
        r'\b(phase|extension|extn|plot|khasra|h\.?\s*no|door\s*no|flat|block)\b',
        r'\b(village|gram|po|post|taluka|tehsil|district|dist|state)\b',
        r'\b(puram|pur|pura|pet|peta|wada|wadi|bazar|bazaar|mandi|market)\b',
        # All 28 States and 8 Union Territories
        r'\b(andaman|nicobar|andhra\s*pradesh|arunachal\s*pradesh|assam|bihar)\b',
        r'\b(chandigarh|chhattisgarh|dadra|nagar\s*haveli|daman|diu|delhi|goa)\b',
        r'\b(gujarat|haryana|himachal\s*pradesh|jammu|kashmir|jharkhand)\b',
        r'\b(karnataka|kerala|ladakh|lakshadweep|madhya\s*pradesh|maharashtra)\b',
        r'\b(manipur|meghalaya|mizoram|nagaland|odisha|puducherry|punjab)\b',
        r'\b(rajasthan|sikkim|tamil\s*nadu|telangana|tripura)\b',
        r'\b(uttar\s*pradesh|uttarakhand|west\s*bengal)\b',
        # Major Cities - Metro Cities
        r'\b(delhi|new\s*delhi|mumbai|kolkata|chennai|bangalore|bengaluru|hyderabad|pune|ahmedabad)\b',
        # Major Cities - State Capitals & Large Cities
        r'\b(jaipur|lucknow|chandigarh|bhopal|patna| Ranchi|guwahati|imphal|shillong|aizawl|kohima|gangtok|itanagar|agar|srinagar)\b',
        r'\b(gandhinagar|shimla|dehradun|raipur|ranchi|bhubaneswar|trivandrum|thiruvananthapuram)\b',
        # Tier-2 Cities - Major Economic Centers
        r'\b(surat|vadodara|baroda|indore|nagpur|nashik|coimbatore|kochi|mysore|mangalore|madurai|trichy|tiruchirappalli)\b',
        r'\b(visakhapatnam|vijayawada|warangal|karimnagar|nizamabad|guntur|kakinada|khammam|tirupati|nellore|kurnool|anantapur|kadapa)\b',
        r'\b(ludhiana|amritsar|jalandhar|patiala|bathinda|mohali|chandigarh)\b',
        r'\b(bareilly|meerut|agra|kanpur|allahabad|prayagraj|varanasi|gorakhpur|mathura|aligarh|moradabad|firozabad|saharanpur|ghaziabad|noida|greater\s*noida)\b',
        r'\b(gurgaon|gurugram|faridabad|rohtak|hisar|sonipat|karnal|panipat|ambala|kurukshetra)\b',
        r'\b(udaipur|jodhpur|bikaner|ajmer|kota|bhilwara|alwar|sikar|pali)\b',
        r'\b(bhilai|durg|bilaspur|raigarh|korba|dhanbad|jamshedpur|rourkela|bokaro|asansol|durgapur|siliguri|malda)\b',
        r'\b(solapur|kolhapur|sangli|satara|aurangabad|nanded|latur|akola|amravati|chandrapur|wardha|yavatmal|jalgaon|dhule|nandurbar|ratnagiri)\b',
        r'\b(hubli|dharwad|belgaum|belagavi|bellary|bijapur|gulbarga|kalaburagi|mangalore|manipal|davanagere|shimoga|tumkur|dakshina|kannada|udupi|chikmagalur|hassan|mandya|kolar|tumakuru)\b',
        r'\b(thrissur|kottayam|kollam|palakkad|kannur|kasaragod|alappuzha|pathanamthitta|idukki|wayanad|malappuram|kozhikode|calicut)\b',
        r'\b(tirunelveli|vellor|thanjavur|salem|erode|tiruppur|dindigul|theni|virudhunagar|ramanathapuram|sivaganga|pudukkottai|karaikudi|nagapattinam|cuddalore|villupuram|kanchipuram|tiruvallur|krishnagiri|dharmapuri|namakkal|perambalur|ariyalur|tiruvarur)\b',
        r'\b(jabalpur|ujjain|sagar|rewa|satna|gwalior|shivpuri|betul|chhindwara|balaghat|seoni|hoshangabad|bhind|bhilwara|morena|damoh|katni|dewas|ratlam|khandwa|khargone|mandsaur|neemuch|sehora|shajapur|vidisha)\b',
        r'\b(gaya|bhagalpur|muzaffarpur|darbhanga|purnia|begusarai|siwan|chapra|hajipur|saharsa|samastipur|munger|buxar|bhabhua|ara|motihari|saran|vaishali|nalanda| Nawada|jamui|lakhisarai|sheikhpura|jatni|barh)\b',
        r'\b(dibrugarh|jorhat|tezipur|silchar|nagaon|tinsukia|guwahati|dhubri|goalpara|barpeta|kokrajhar|sonitpur|lakhimpur|dhemaji|karbi\s*anglong|cachar|karimganj|hailakandi)\b',
        # Additional significant cities
        r'\b(diu|silvassa|daman|kavaratti|port\s*blair|puducherry|karaikal|mahe|yanam)\b',
        # Common local suffixes
        r'\b(puram|pur|pura|palle|pally|pet|peta|wada|wadi|garh|kot|gram|pada|vada)\b',
    ],
    'Myanmar': [
        r'\b\d{5}\b',                       # Myanmar postal code (5 digits)
        r'\b(street|st|road|rd|lane|township|quarter|ward)\b',
        r'\b(yangon|mandalay|naypyidaw)\b',
    ],
}

# Short address patterns that bypass minimum quality checks
# These are typically unit numbers, postal codes, etc. that are valid addresses
# but fail the meets_minimum_quality() check (10+ chars, 3+ words)
SHORT_ADDRESS_PATTERNS: Dict[str, List[str]] = {
    'Singapore': [
        r'^#\d+-\d+$',              # Unit number: #12-34
        r'^\d{6}$',                 # 6-digit postal code: 560105
        r'^singapore\s+\d{6}$',     # Singapore + postal code
    ],
    'Thailand': [
        r'^\d{5}$',                 # 5-digit postal code
    ],
    'Malaysia': [
        r'^\d{5}$',                 # 5-digit postal code
    ],
    'Myanmar': [
        r'^\d{5}$',                 # 5-digit postal code
    ],
    'India': [
        r'^\d{6}$',                 # 6-digit PIN code
    ],
    'UAE': [
        r'^po\s+box\s+\d{4,6}$',    # PO Box only
    ],
    'UK': [
        r'^[A-Z]{1,2}\d{1,2}[ ]?\d[A-Z]{2}$',  # UK postcode only
    ],
}


def is_negative_keyword(text: str) -> bool:
    """Check if text contains negative keywords."""
    text_lower = text.lower()
    for neg in NEGATIVE_KEYWORDS:
        if neg in text_lower:
            return True
    return False


def is_low_quality_ocr(text: str) -> bool:
    """
    Check if text is low-quality OCR (garbled).

    Detects excessive repeated characters.
    """
    text_lower = text.lower()
    consecutive_repeats = re.search(r'(.)\1{2,}', text_lower)
    if consecutive_repeats:
        repeat_count = len(re.findall(r'(.)\1{2,}', text_lower))
        if repeat_count >= 2:
            return True
    return False


def meets_minimum_quality(text: str) -> bool:
    """Check if text meets minimum quality thresholds."""
    if len(text) < 10:
        return False
    if len(text.split()) < 3:
        return False
    return True


def matches_country_patterns(text: str) -> bool:
    """Check if text matches any country-specific address pattern."""
    text_lower = text.lower()

    for country, patterns in COUNTRY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True

    return False


def matches_generic_fallback(text: str) -> bool:
    """
    Generic address pattern fallback.
    Requires: number AND comma AND minimum length
    """
    words = text.split()
    if len(words) < 3:
        return False

    has_number = any(bool(re.search(r'\d', w)) for w in words)
    has_comma = ',' in text
    min_length = len(text) >= 10

    return has_number and has_comma and min_length


def is_valid_address_text(text: str, get_bank_info_func=None, country_hint: str = None) -> bool:
    """
    Main address detection function using country-specific patterns.

    Args:
        text: Text to check
        get_bank_info_func: Function to check if text is a bank name (for dependency injection)
        country_hint: Country to prioritize patterns for (e.g., 'Singapore', 'Thailand')

    Returns:
        True if text looks like an address, False otherwise
    """
    # 1. Filter low-quality OCR
    if is_low_quality_ocr(text):
        return False

    # 2. Check if text IS a bank name (exact matching only)
    # This filters out "DBS Bank" but allows "123 Bank Street"
    if get_bank_info_func:
        bank_info = get_bank_info_func(text)  # Exact match only
        if bank_info:
            return False  # This is a known bank name, not an address

    # 3. Check negative keywords
    if is_negative_keyword(text):
        return False

    text_lower = text.lower()

    # 4. Check if text matches known SHORT address patterns
    # If it does, skip the minimum quality check (these are valid short addresses)
    # IMPORTANT: Only check short patterns when we have a country hint, to avoid
    # false positives from matching unrelated country patterns
    if country_hint:
        short_patterns = SHORT_ADDRESS_PATTERNS.get(country_hint, [])
        for pattern in short_patterns:
            if re.search(pattern, text_lower):
                return True  # Valid short address, skip further checks

    # 5. Check minimum quality (only for non-short patterns)
    if not meets_minimum_quality(text):
        return False

    # 6. Check country-specific patterns (prioritize country_hint if provided)
    if country_hint:
        country_patterns = COUNTRY_PATTERNS.get(country_hint, [])
        for pattern in country_patterns:
            if re.search(pattern, text_lower):
                return True

    # 7. Check all country patterns as fallback
    if matches_country_patterns(text):
        return True

    # 8. Fallback to generic pattern
    if matches_generic_fallback(text):
        return True

    return False
