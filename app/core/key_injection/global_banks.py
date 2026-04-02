"""
Global Bank SWIFT Codes Database
Contains bank names and SWIFT codes organized by country.
SWIFT codes are 8-character base codes (bank + country level).
Also includes currency and country name mappings for bank statement processing.
"""

from typing import Optional, List
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BankInfo:
    """Bank information with SWIFT code."""
    name: str
    swift_code: str
    country: str  # ISO 3166-1 alpha-2


# Currency codes to country codes mapping (ISO 4217 -> ISO 3166-1 alpha-2)
CURRENCY_COUNTRIES = {
    "SGD": "SG",  # Singapore Dollar
    "USD": "US",  # US Dollar
    "GBP": "GB",  # British Pound
    "EUR": "EU",  # Euro (generic)
    "HKD": "HK",  # Hong Kong Dollar
    "AUD": "AU",  # Australian Dollar
    "NZD": "NZ",  # New Zealand Dollar
    "JPY": "JP",  # Japanese Yen
    "CNY": "CN",  # Chinese Yuan
    "RMB": "CN",  # Chinese Yuan (alternate)
    "KRW": "KR",  # Korean Won
    "INR": "IN",  # Indian Rupee
    "MYR": "MY",  # Malaysian Ringgit
    "THB": "TH",  # Thai Baht
    "IDR": "ID",  # Indonesian Rupiah
    "PHP": "PH",  # Philippine Peso
    "VND": "VN",  # Vietnamese Dong
    "CHF": "CH",  # Swiss Franc
    "CAD": "CA",  # Canadian Dollar
    "AED": "AE",  # UAE Dirham
    "SAR": "SA",  # Saudi Riyal
    "BRL": "BR",  # Brazilian Real
    "MXN": "MX",  # Mexican Peso
    "TWD": "TW",  # Taiwan Dollar
    "ZAR": "ZA",  # South African Rand
    "SEK": "SE",  # Swedish Krona
    "NOK": "NO",  # Norwegian Krone
    "DKK": "DK",  # Danish Krone
    "PLN": "PL",  # Polish Zloty
    "CZK": "CZ",  # Czech Koruna
    "HUF": "HU",  # Hungarian Forint
    "TRY": "TR",  # Turkish Lira
    "ILS": "IL",  # Israeli Shekel
    "QAR": "QA",  # Qatari Riyal
    "KWD": "KW",  # Kuwaiti Dinar
    "BHD": "BH",  # Bahraini Dinar
    "OMR": "OM",  # Omani Rial
    "MMK": "MM",  # Myanmar Kyat
}

# Country names to country codes mapping (for address detection)
COUNTRY_NAMES = {
    "singapore": "SG",
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "united kingdom": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "great britain": "GB",
    "hong kong": "HK",
    "australia": "AU",
    "germany": "DE",
    "deutschland": "DE",
    "japan": "JP",
    "china": "CN",
    "india": "IN",
    "malaysia": "MY",
    "thailand": "TH",
    "indonesia": "ID",
    "philippines": "PH",
    "vietnam": "VN",
    "viet nam": "VN",
    "switzerland": "CH",
    "canada": "CA",
    "france": "FR",
    "united arab emirates": "AE",
    "uae": "AE",
    "dubai": "AE",
    "abu dhabi": "AE",
    "south korea": "KR",
    "korea": "KR",
    "netherlands": "NL",
    "holland": "NL",
    "spain": "ES",
    "italy": "IT",
    "brazil": "BR",
    "brasil": "BR",
    "mexico": "MX",
    "new zealand": "NZ",
    "taiwan": "TW",
    "south africa": "ZA",
    "saudi arabia": "SA",
    "qatar": "QA",
    "kuwait": "KW",
    "bahrain": "BH",
    "oman": "OM",
    "myanmar": "MM",
    "burma": "MM",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "poland": "PL",
    "czech republic": "CZ",
    "czechia": "CZ",
    "hungary": "HU",
    "turkey": "TR",
    "israel": "IL",
    "ireland": "IE",
    "belgium": "BE",
    "austria": "AT",
    "portugal": "PT",
    "greece": "GR",
    "finland": "FI",
    "russia": "RU",
    "ukraine": "UA",
}

# Cities, states, provinces to country codes mapping (for address detection)
# This helps detect country from addresses that mention locations but not country names
LOCATION_TO_COUNTRY = {
    # ============================================
    # Singapore - Areas, Districts, Streets
    # ============================================
    "orchard": "SG",
    "raffles": "SG",
    "bugis": "SG",
    "tanjong pagar": "SG",
    "marina bay": "SG",
    "sentosa": "SG",
    "jurong": "SG",
    "tampines": "SG",
    "bedok": "SG",
    "pasir ris": "SG",
    "hougang": "SG",
    "ang mo kio": "SG",
    "toa payoh": "SG",
    "bishan": "SG",
    "clementi": "SG",
    "bukit timah": "SG",
    "serangoon": "SG",
    "kallang": "SG",
    "ghim moh": "SG",
    "commonwealth": "SG",
    "queenstown": "SG",
    "alexandra": "SG",
    "harbourfront": "SG",
    "vivocity": "SG",
    "changi": "SG",
    "paya lebar": "SG",
    "macpherson": "SG",
    "boon lay": "SG",
    "woodlands": "SG",
    "sembawang": "SG",
    "yishun": "SG",
    "choa chu kang": "SG",
    "bukit batok": "SG",
    "jurong east": "SG",
    "jurong west": "SG",
    "pioneer": "SG",
    "clementi": "SG",
    "dover": "SG",
    "buona vista": "SG",
    "holland village": "SG",
    "little india": "SG",
    "farrer park": "SG",
    "lavender": "SG",
    "city hall": "SG",
    "esplanade": "SG",
    "bras basah": "SG",
    "nicoll highway": "SG",
    "mtb": "SG",
    "shenton way": "SG",
    "tanjong pagar": "SG",
    "golden mile": "SG",
    "beach road": "SG",
    "north bridge road": "SG",
    "south bridge road": "SG",
    "stanford road": "SG",
    "collyer quay": "SG",
    "raffles place": "SG",
    "boat quay": "SG",
    "empire dock": "SG",
    " robinson road": "SG",
    "cecil street": "SG",
    "telok ayer": "SG",
    "amoy street": "SG",
    "shenton way": "SG",
    "crosby road": "SG",

    # ============================================
    # Thailand - Provinces, Cities, Districts
    # ============================================
    # Bangkok areas
    "bangkok": "TH",
    "krung thep": "TH",
    "pathum wan": "TH",
    "siam": "TH",
    "sukhumvit": "TH",
    "silom": "TH",
    "sathorn": "TH",
    "rong muang": "TH",
    "wang mai": "TH",
    "yan nawa": "TH",
    "bang rak": "TH",
    "klong toei": "TH",
    "vadhana": "TH",
    "khlong toei": "TH",
    "lat phrao": "TH",
    "huai khwang": "TH",
    "din daeng": "TH",
    "ratchathewi": "TH",
    "phaya thai": "TH",
    "phra nakhon": "TH",
    "samphanthawong": "TH",
    "pom prap sattru phai": "TH",
    "bang khen": "TH",
    "bang kapi": "TH",
    "lat krabang": "TH",
    "nong chok": "TH",
    "min buri": "TH",
    "khan na yao": "TH",
    "sai mai": "TH",
    "bang kho laem": "TH",
    "thon buri": "TH",
    "bangkok noi": "TH",
    "bangkok yai": "TH",
    "phasi charoen": "TH",
    "nong khaem": "TH",
    "chom thong": "TH",
    "thung khru": "TH",
    "bang khun thian": "TH",
    "taling chan": "TH",

    # Central Thailand provinces
    "ayutthaya": "TH",
    "phra nakhon si ayutthaya": "TH",
    "ayuthaya": "TH",
    "wang noi": "TH",
    "bang pa-in": "TH",
    "bang sai": "TH",
    "phachi": "TH",
    "nakhon nayok": "TH",
    "pathum thani": "TH",
    "rangsit": "TH",
    "thanyaburi": "TH",
    "khlong luang": "TH",
    "nakhon pathom": "TH",
    "samut sakhon": "TH",
    "samut songkhram": "TH",
    "suphan buri": "TH",
    "ang thong": "TH",
    "sing buri": "TH",
    "lopburi": "TH",
    "saraburi": "TH",
    "chaophraya surasak": "TH",

    # Northern Thailand
    "chiang mai": "TH",
    "chiang rai": "TH",
    "lamphun": "TH",
    "lampang": "TH",
    "phayao": "TH",
    "nan": "TH",
    "phrae": "TH",
    "mae hong son": "TH",
    "tak": "TH",
    "kamphaeng phet": "TH",
    "phichit": "TH",
    "phitsanulok": "TH",
    "sukhothai": "TH",
    "uttaradit": "TH",

    # Northeastern Thailand (Isan)
    "khon kaen": "TH",
    "udon thani": "TH",
    "nakhon ratchasima": "TH",
    "korat": "TH",
    "ubon ratchathani": "TH",
    "nakhon phanom": "TH",
    " Mukdahan": "TH",
    "yasothon": "TH",
    "roi et": "TH",
    "kalasin": "TH",
    "sakon nakhon": "TH",
    "nong khai": "TH",
    "bueng kan": "TH",
    "loei": "TH",
    "surin": "TH",
    "si saket": "TH",
    "buriram": "TH",
    "chaiyaphum": "TH",
    "mahasarakham": "TH",
    "khon kaen": "TH",

    # Eastern Thailand
    "chonburi": "TH",
    "pattaya": "TH",
    "jomtien": "TH",
    "bang lamung": "TH",
    "si racha": "TH",
    "rayong": "TH",
    "chanthaburi": "TH",
    "trat": "TH",
    "sa kaeo": "TH",
    "chachoengsao": "TH",
    "prachinburi": "TH",
    "nakhon nayok": "TH",

    # Western Thailand
    "kanchanaburi": "TH",
    "ratchaburi": "TH",
    "phetchaburi": "TH",
    "prachuap khiri khan": "TH",
    "hua hin": "TH",
    "cha-am": "TH",
    "tak": "TH",

    # Southern Thailand
    "surat thani": "TH",
    "ko samui": "TH",
    "samui": "TH",
    "phangnga": "TH",
    "phuket": "TH",
    "krabi": "TH",
    "phang-nga": "TH",
    "ranong": "TH",
    "chumphon": "TH",
    "nakhon si thammarat": "TH",
    "trang": "TH",
    "phatthalung": "TH",
    "songkhla": "TH",
    "hat yai": "TH",
    "satun": "TH",
    "yala": "TH",
    "pattani": "TH",
    "narathiwat": "TH",
    "koln": "TH",

    # ============================================
    # Myanmar - States, Regions, Cities
    # ============================================
    "yangon": "MM",
    "rangoon": "MM",
    "mandalay": "MM",
    "naypyidaw": "MM",
    "bago": "MM",
    "taunggyi": "MM",
    "mawlamyine": "MM",
    "monywa": "MM",
    "pathein": "MM",
    "meiktila": "MM",
    "sittwe": "MM",
    "taungoo": "MM",
    "pyay": "MM",
    "myingyan": "MM",
    "kalay": "MM",
    "hlaingbwe": "MM",
    "pakokku": "MM",
    # Myanmar States/Regions
    "kachin": "MM",
    "kayah": "MM",
    "kayin": "MM",
    "chin": "MM",
    "mon": "MM",
    "rakhine": "MM",
    "shan": "MM",
    "Sagaing": "MM",
    "tanintharyi": "MM",
    "ayeyarwady": "MM",
    "bago region": "MM",
    "magway": "MM",
    "mandalay region": "MM",
    "yangon region": "MM",

    # ============================================
    # Malaysia - States, Cities, Areas
    # ============================================
    "kuala lumpur": "MY",
    "kl": "MY",
    "selangor": "MY",
    "petaling jaya": "MY",
    "pj": "MY",
    "subang jaya": "MY",
    "shah alam": "MY",
    "ampang": "MY",
    "gombak": "MY",
    "klang": "MY",
    "bangsar": "MY",
    "mont kiara": "MY",
    "damansara": "MY",
    "hartamas": "MY",
    "cheras": "MY",
    "setapak": "MY",
    "wangsa maju": "MY",
    "sentul": "MY",
    "kepong": "MY",
    "selayang": "MY",
    "batu caves": "MY",
    "puchong": "MY",
    "putrajaya": "MY",
    "cyberjaya": "MY",
    "sepang": "MY",
    "negeri sembilan": "MY",
    "seremban": "MY",
    "port dickson": "MY",
    "melaka": "MY",
    "malacca": "MY",
    "johor": "MY",
    "johor bahru": "MY",
    "jb": "MY",
    "skudai": "MY",
    "tebrau": "MY",
    "kulai": "MY",
    "pasir gudang": "MY",
    "muar": "MY",
    "batu pahat": "MY",
    "kedah": "MY",
    "alor setar": "MY",
    "langkawi": "MY",
    "sungai petani": "MY",
    "kulim": "MY",
    "penang": "MY",
    "pulau pinang": "MY",
    "georgetown": "MY",
    "bayan lepas": "MY",
    "batu ferringhi": "MY",
    "balik pulau": "MY",
    "perak": "MY",
    "ipoh": "MY",
    "taiping": "MY",
    "kuala kangsar": "MY",
    "teluk intan": "MY",
    "perlis": "MY",
    "kangar": "MY",
    "kelantan": "MY",
    "kota bharu": "MY",
    "terengganu": "MY",
    "kuala terengganu": "MY",
    "pahang": "MY",
    "kuantan": "MY",
    "temerloh": "MY",
    "bentong": "MY",
    "raub": "MY",
    "jerantut": "MY",
    "sabah": "MY",
    "kota kinabalu": "MY",
    "sandakan": "MY",
    "tawau": "MY",
    "labuan": "MY",
    "sarawak": "MY",
    "kuching": "MY",
    "miri": "MY",
    "sibu": "MY",
    "bintulu": "MY",

    # ============================================
    # Indonesia - Provinces, Cities, Islands
    # ============================================
    "jakarta": "ID",
    "dki jakarta": "ID",
    "special capital region of jakarta": "ID",
    "pusat": "ID",
    "central jakarta": "ID",
    "jakarta pusat": "ID",
    "utara": "ID",
    "north jakarta": "ID",
    "jakarta utara": "ID",
    "barat": "ID",
    "west jakarta": "ID",
    "jakarta barat": "ID",
    "selatan": "ID",
    "south jakarta": "ID",
    "jakarta selatan": "ID",
    "timur": "ID",
    "east jakarta": "ID",
    "jakarta timur": "ID",
    "bekasi": "ID",
    "depok": "ID",
    "tangerang": "ID",
    "tangerang selatan": "ID",
    "south tangerang": "ID",
    "bogor": "ID",
    "bandung": "ID",
    "west java": "ID",
    "jawa barat": "ID",
    "cirebon": "ID",
    "bekasi": "ID",
    "central java": "ID",
    "jawa tengah": "ID",
    "semarang": "ID",
    "solo": "ID",
    "surakarta": "ID",
    "magelang": "ID",
    "tegal": "ID",
    "pekalongan": "ID",
    "east java": "ID",
    "jawa timur": "ID",
    "surabaya": "ID",
    "malang": "ID",
    "kediri": "ID",
    "mojokerto": "ID",
    "pasuruan": "ID",
    "banyuwangi": "ID",
    "jember": "ID",
    "yogyakarta": "ID",
    "daerah istimewa yogyakarta": "ID",
    "sleman": "ID",
    "bantul": "ID",
    "bali": "ID",
    "denpasar": "ID",
    "badung": "ID",
    "gianyar": "ID",
    "ubud": "ID",
    "kuta": "ID",
    "seminyak": "ID",
    "sanur": "ID",
    "tabanan": "ID",
    "singaraja": "ID",
    "bangli": "ID",
    "karangasem": "ID",
    "klungkung": "ID",
    "sumatra": "ID",
    "sumatera": "ID",
    "medan": "ID",
    "north sumatra": "ID",
    "sumatera utara": "ID",
    "pematang siantar": "ID",
    "binjai": "ID",
    "padang": "ID",
    "west sumatra": "ID",
    "sumatera barat": "ID",
    "palembang": "ID",
    "south sumatra": "ID",
    "sumatera selatan": "ID",
    "riau": "ID",
    "pekanbaru": "ID",
    "dumai": "ID",
    "riau islands": "ID",
    "kepulauan riau": "ID",
    "batam": "ID",
    "bintan": "ID",
    "tanjung pinang": "ID",
    "jambi": "ID",
    "bengkulu": "ID",
    "lampung": "ID",
    "bandar lampung": "ID",
    "south sumatera": "ID",
    "bangka belitung": "ID",
    "pangkalpinang": "ID",
    "borneo": "ID",
    "kalimantan": "ID",
    "west kalimantan": "ID",
    "kalimantan barat": "ID",
    "pontianak": "ID",
    "central kalimantan": "ID",
    "kalimantan tengah": "ID",
    "palangkaraya": "ID",
    "south kalimantan": "ID",
    "kalimantan selatan": "ID",
    "banjarmasin": "ID",
    "east kalimantan": "ID",
    "kalimantan timur": "ID",
    "balikpapan": "ID",
    "samarinda": "ID",
    "north kalimantan": "ID",
    "kalimantan utara": "ID",
    "tarakan": "ID",
    "north sulawesi": "ID",
    "sulawesi utara": "ID",
    "manado": "ID",
    "gorontalo": "ID",
    "central sulawesi": "ID",
    "sulawesi tengah": "ID",
    "palu": "ID",
    "west sulawesi": "ID",
    "sulawesi barat": "ID",
    "mamasa": "ID",
    "south sulawesi": "ID",
    "sulawesi selatan": "ID",
    "makassar": "ID",
    "ujung pandang": "ID",
    "parepare": "ID",
    "southeast sulawesi": "ID",
    "sulawesi tenggara": "ID",
    "kendari": "ID",
    "north maluku": "ID",
    "maluku utara": "ID",
    "ternate": "ID",
    "tidore": "ID",
    "maluku": "ID",
    "ambon": "ID",

    # ============================================
    # Philippines - Regions, Provinces, Cities
    # ============================================
    "metro manila": "PH",
    "ncr": "PH",
    "manila": "PH",
    "makati": "PH",
    "quezon city": "PH",
    "taguig": "PH",
    "bonifacio": "PH",
    "bgc": "PH",
    "pasig": "PH",
    "mandaluyong": "PH",
    "san juan": "PH",
    "pasay": "PH",
    "paranaque": "PH",
    "las pinas": "PH",
    "muntinlupa": "PH",
    "marikina": "PH",
    "valenzuela": "PH",
    "malabon": "PH",
    "navotas": "PH",
    "caloocan": "PH",
    "luzon": "PH",
    "visayas": "PH",
    "mindanao": "PH",
    "bicol": "PH",
    "cagayan": "PH",
    "cagayan de oro": "PH",
    "iloilo": "PH",
    "bacolod": "PH",
    "cebu": "PH",
    "davao": "PH",
    "davao city": "PH",
    "zamboanga": "PH",
    "angeles": "PH",
    "pampanga": "PH",
    "batangas": "PH",
    "cavite": "PH",
    "laguna": "PH",
    "rizal": "PH",
    "bulacan": "PH",
    "pangasinan": "PH",
    "nueva ecija": "PH",
    "tarlac": "PH",
    "zambales": "PH",
    "bataan": "PH",
    "aurora": "PH",
    "quezon": "PH",
    "isabela": "PH",
    "cagayan valley": "PH",
    "ifugao": "PH",
    "kalinga": "PH",
    "mountain province": "PH",
    "benguet": "PH",
    "baguio": "PH",
    "ilocos": "PH",
    "la union": "PH",
    "pangasinan": "PH",
    "ilocos norte": "PH",
    "ilocos sur": "PH",
    "camarines norte": "PH",
    "camarines sur": "PH",
    "albay": "PH",
    "sorsogon": "PH",
    "masbate": "PH",
    "samar": "PH",
    "eastern samar": "PH",
    "northern samar": "PH",
    "western samar": "PH",
    "leyte": "PH",
    "southern leyte": "PH",
    "biliran": "PH",
    "bohol": "PH",
    "siargao": "PH",
    "surigao": "PH",
    "agusan": "PH",
    "bukidnon": "PH",
    "camiguin": "PH",
    "misamis": "PH",
    "lanao": "PH",
    "maguindanao": "PH",
    "sulu": "PH",
    "tawi-tawi": "PH",
    "basilan": "PH",
    "palawan": "PH",
    "puerto princesa": "PH",
    "el nido": "PH",
    "coron": "PH",
    "romblon": "PH",
    "marinduque": "PH",
    "guimaras": "PH",
    "antique": "PH",
    "capiz": "PH",
    "aklan": "PH",
    "negros occidental": "PH",
    "negros oriental": "PH",
    "dumaguete": "PH",
    "siquijor": "PH",

    # ============================================
    # Vietnam - Provinces, Cities
    # ============================================
    "ho chi minh": "VN",
    "saigon": "VN",
    "hanoi": "VN",
    "ha noi": "VN",
    "da nang": "VN",
    "hai phong": "VN",
    "can tho": "VN",
    "hue": "VN",
    "thua thien": "VN",
    "quang nam": "VN",
    "quang ngai": "VN",
    "binh dinh": "VN",
    "phu yen": "VN",
    "khanh hoa": "VN",
    "nha trang": "VN",
    "ninh thuan": "VN",
    "binh thuan": "VN",
    "dong nai": "VN",
    "binh duong": "VN",
    "tay ninh": "VN",
    "ba ria": "VN",
    "vung tau": "VN",
    "baria vungtau": "VN",
    "an giang": "VN",
    "long xuyen": "VN",
    "dong thap": "VN",
    "tien giang": "VN",
    "my tho": "VN",
    "vinh long": "VN",
    "tra vinh": "VN",
    "soc trang": "VN",
    "bac lieu": "VN",
    "ca mau": "VN",
    "kien giang": "VN",
    "phu quoc": "VN",
    "rach gia": "VN",
    "can tho": "VN",
    "ha giang": "VN",
    "cao bang": "VN",
    "bac kan": "VN",
    "tuyen quang": "VN",
    "lang son": "VN",
    "bac giang": "VN",
    "phu tho": "VN",
    "thai nguyen": "VN",
    "bac ninh": "VN",
    "hai duong": "VN",
    "quang ninh": "VN",
    "halong": "VN",
    "ha long": "VN",
    "mong cai": "VN",
    "nam dinh": "VN",
    "ninh binh": "VN",
    "thanh hoa": "VN",
    "nghe an": "VN",
    "vinh": "VN",
    "ha tinh": "VN",
    "quang binh": "VN",
    "dong hoi": "VN",
    "quang tri": "VN",

    # ============================================
    # Hong Kong - Areas, Districts
    # ============================================
    "central": "HK",
    "admiralty": "HK",
    "wan chai": "HK",
    "causeway bay": "HK",
    "tin hau": "HK",
    "fortress hill": "HK",
    "north point": "HK",
    "quarry bay": "HK",
    "shau kei wan": "HK",
    "chai wan": "HK",
    "hong kong island": "HK",
    "hk island": "HK",
    "kowloon": "HK",
    "tsim sha tsui": "HK",
    "tst": "HK",
    "jordan": "HK",
    "yau ma tei": "HK",
    "mong kok": "HK",
    "prince edward": "HK",
    "sham shui po": "HK",
    "lai chi kok": "HK",
    "mei foo": "HK",
    "kwun tong": "HK",
    "lam tin": "HK",
    "yau tong": "HK",
    "lei yue mun": "HK",
    "wong tai sin": "HK",
    "san po kong": "HK",
    "diamond hill": "HK",
    "choi hung": "HK",
    "wong tai sin": "HK",
    "hung hom": "HK",
    "to kwa wan": "HK",
    "ma tau wei": "HK",
    "kowloon bay": "HK",
    "kowloon city": "HK",
    "ho man tin": "HK",
    "shek kip mei": "HK",
    "new territories": "HK",
    "sha tin": "HK",
    "tai wai": "HK",
    "fo tan": "HK",
    "ma on shan": "HK",
    "wu kai sha": "HK",
    "tuen mun": "HK",
    "yuen long": "HK",
    "tin shui wai": "HK",
    "fanling": "HK",
    "sheung shui": "HK",
    "tai po": "HK",
    "north district": "HK",
    "sai kung": "HK",
    "tuk ng": "HK",
    "hang hau": "HK",
    "junk bay": "HK",
    "tseung kwan o": "HK",
    "discovery bay": "HK",
    "lantau island": "HK",
    "tung chung": "HK",
    "disneyland": "HK",
    "hkia": "HK",
    "chek lap kok": "HK",
    "stanley": "HK",
    "repulse bay": "HK",
    "shek o": "HK",
    "aberdeen": "HK",
    "wang chau": "HK",
    "ap lei chau": "HK",
    " Pok Fu Lam": "HK",
    "mid-levels": "HK",
    "soho": "HK",
    "pacific place": "HK",

    # ============================================
    # Taiwan - Cities, Counties
    # ============================================
    "taipei": "TW",
    "new taipei": "TW",
    "taipei city": "TW",
    "da'an": "TW",
    "xinyi": "TW",
    "songshan": "TW",
    "beitou": "TW",
    "shilin": "TW",
    "zhongshan": "TW",
    "datong": "TW",
    "wanhua": "TW",
    "wenshan": "TW",
    "nangang": "TW",
    "neihu": "TW",
    "taichung": "TW",
    "taichung city": "TW",
    "tainan": "TW",
    "tainan city": "TW",
    "kaohsiung": "TW",
    "kaohsiung city": "TW",
    "taoyuan": "TW",
    "taoyuan city": "TW",
    "hsinchu": "TW",
    "hsinchu city": "TW",
    "chiayi": "TW",
    "keelung": "TW",
    "ilan": "TW",
    "yilan": "TW",
    "miaoli": "TW",
    "changhua": "TW",
    "nantou": "TW",
    "yunlin": "TW",
    "pingtung": "TW",
    "taitung": "TW",
    "hua lien": "TW",
    "hualien": "TW",
    "penghu": "TW",
    "kinmen": "TW",
    "lienchiang": "TW",
    "matsu": "TW",

    # ============================================
    # South Korea - Cities, Provinces
    # ============================================
    "seoul": "KR",
    "busan": "KR",
    "incheon": "KR",
    "daegu": "KR",
    "daejeon": "KR",
    "gwangju": "KR",
    "suwon": "KR",
    "ulsan": "KR",
    "seongnam": "KR",
    "goyang": "KR",
    "yongin": "KR",
    "anyang": "KR",
    "cheongju": "KR",
    "anjeong": "KR",
    "pyeongtaek": "KR",
    "jeju": "KR",
    "jeju-do": "KR",
    "gyeonggi": "KR",
    "gyeonggi-do": "KR",
    "gangwon": "KR",
    "gangwon-do": "KR",
    "chungcheong": "KR",
    "chungcheongnam-do": "KR",
    "chungcheongbuk-do": "KR",
    "jeollabuk-do": "KR",
    "jeollanam-do": "KR",
    "gyeongsangbuk-do": "KR",
    "gyeongsangnam-do": "KR",
    "daejeon": "KR",
    "ulsan": "KR",
    "sejong": "KR",

    # ============================================
    # Japan - Prefectures, Cities
    # ============================================
    "tokyo": "JP",
    "osaka": "JP",
    "kyoto": "JP",
    "yokohama": "JP",
    "nagoya": "JP",
    "sapporo": "JP",
    "fukuoka": "JP",
    "kobe": "JP",
    "kawasaki": "JP",
    "saitama": "JP",
    "hiroshima": "JP",
    "sendai": "JP",
    "kitakyushu": "JP",
    "chiba": "JP",
    "sakai": "JP",
    "niigata": "JP",
    "hamamatsu": "JP",
    "kumamoto": "JP",
    "sagamihara": "JP",
    "shizuoka": "JP",
    "okayama": "JP",
    "fukushima": "JP",
    "kanazawa": "JP",
    "nagasaki": "JP",
    "matsuyama": "JP",
    "aichi": "JP",
    "hyogo": "JP",
    "nara": "JP",
    "wakayama": "JP",
    "tottori": "JP",
    "shimane": "JP",
    "yamaguchi": "JP",
    "tokushima": "JP",
    "kagawa": "JP",
    "ehime": "JP",
    "kochi": "JP",
    "fukuoka": "JP",
    "saga": "JP",
    "nagasaki": "JP",
    "kumamoto": "JP",
    "oita": "JP",
    "miyazaki": "JP",
    "kagoshima": "JP",
    "okinawa": "JP",
    "naha": "JP",
    "hokkaido": "JP",
    "tohoku": "JP",
    "kanto": "JP",
    "kansai": "JP",
    "chugoku": "JP",
    "shikoku": "JP",
    "kyushu": "JP",
    "gunma": "JP",
    "tochigi": "JP",
    "ibaraki": "JP",
    "yamanashi": "JP",
    "nagano": "JP",
    "gifu": "JP",
    "shiga": "JP",
    "mie": "JP",
    "ishikawa": "JP",
    "fukui": "JP",
    "akita": "JP",
    "yamagata": "JP",
    "miyagi": "JP",
    "aomori": "JP",
    "iwate": "JP",
    "toyama": "JP",
    "ishikawa": "JP",
    "fukui": "JP",

    # ============================================
    # China - Provinces, Major Cities
    # ============================================
    "beijing": "CN",
    "shanghai": "CN",
    "guangzhou": "CN",
    "shenzhen": "CN",
    "chengdu": "CN",
    "hangzhou": "CN",
    "wuhan": "CN",
    "xian": "CN",
    "chongqing": "CN",
    "nanjing": "CN",
    "tianjin": "CN",
    "shenyang": "CN",
    "dongguan": "CN",
    "foshan": "CN",
    "zhengzhou": "CN",
    "qingdao": "CN",
    "suzhou": "CN",
    "changsha": "CN",
    "ningbo": "CN",
    "wuxi": "CN",
    "xiamen": "CN",
    "dalian": "CN",
    "hefei": "CN",
    "kunming": "CN",
    "fuzhou": "CN",
    "harbin": "CN",
    "jinan": "CN",
    "nanchang": "CN",
    "nanning": "CN",
    "taiyuan": "CN",
    "chanchun": "CN",
    "changchun": "CN",
    "guiyang": "CN",
    "nanning": "CN",
    "lanzhou": "CN",
    "haikou": "CN",
    "shijiazhuang": "CN",
    "guangdong": "CN",
    "jiangsu": "CN",
    "zhejiang": "CN",
    "shandong": "CN",
    "henan": "CN",
    "sichuan": "CN",
    "hubei": "CN",
    "hunan": "CN",
    "anhui": "CN",
    "hebei": "CN",
    "shanxi": "CN",
    "shaanxi": "CN",
    "liaoning": "CN",
    "jilin": "CN",
    "heilongjiang": "CN",
    "jiangxi": "CN",
    "guangxi": "CN",
    "yunnan": "CN",
    "guizhou": "CN",
    "gansu": "CN",
    "qinghai": "CN",
    "nei mongol": "CN",
    "inner mongolia": "CN",
    "xinjiang": "CN",
    "tibet": "CN",
    "ningxia": "CN",
    "hainan": "CN",
    "macau": "CN",
    "hong kong": "CN",

    # ============================================
    # India - States, Major Cities
    # ============================================
    "delhi": "IN",
    "new delhi": "IN",
    "mumbai": "IN",
    "bangalore": "IN",
    "bengaluru": "IN",
    "chennai": "IN",
    "kolkata": "IN",
    "hyderabad": "IN",
    "pune": "IN",
    "ahmedabad": "IN",
    "surat": "IN",
    "jaipur": "IN",
    "lucknow": "IN",
    "kanpur": "IN",
    "nagpur": "IN",
    "indore": "IN",
    "bhopal": "IN",
    "patna": "IN",
    " Vadodara": "IN",
    "ghaziabad": "IN",
    "ludhiana": "IN",
    "agra": "IN",
    "nashik": "IN",
    "faridabad": "IN",
    "meerut": "IN",
    "rajkot": "IN",
    "varanasi": "IN",
    "srinagar": "IN",
    "aurangabad": "IN",
    "dhanbad": "IN",
    "amritsar": "IN",
    "navi mumbai": "IN",
    "allahabad": "IN",
    "prayagraj": "IN",
    "ranchi": "IN",
    "howrah": "IN",
    "jabalpur": "IN",
    "gwalior": "IN",
    "vijayawada": "IN",
    "jodhpur": "IN",
    "madurai": "IN",
    "rajkot": "IN",
    "kota": "IN",
    "guwahati": "IN",
    "chandigarh": "IN",
    "solapur": "IN",
    "hubli": "IN",
    "mysore": "IN",
    "tiruchirappalli": "IN",
    "bareilly": "IN",
    "aligarh": "IN",
    "tamil nadu": "IN",
    "karnataka": "IN",
    "telangana": "IN",
    "andhra pradesh": "IN",
    "kerala": "IN",
    "gujarat": "IN",
    "rajasthan": "IN",
    "maharashtra": "IN",
    "uttar pradesh": "IN",
    "punjab": "IN",
    "haryana": "IN",
    "bihar": "IN",
    "west bengal": "IN",
    "odisha": "IN",
    "madhya pradesh": "IN",
    "jammu and kashmir": "IN",
    "uttarakhand": "IN",
    "himachal pradesh": "IN",
    "goa": "IN",
    "chhattisgarh": "IN",
    "jharkhand": "IN",
    "assam": "IN",
    "meghalaya": "IN",
    "manipur": "IN",
    "tripura": "IN",
    "nagaland": "IN",
    "arunachal pradesh": "IN",
    "mizoram": "IN",
    "sikkim": "IN",

    # ============================================
    # Australia - States, Major Cities
    # ============================================
    "sydney": "AU",
    "melbourne": "AU",
    "brisbane": "AU",
    "perth": "AU",
    "adelaide": "AU",
    "canberra": "AU",
    "gold coast": "AU",
    "newcastle": "AU",
    "wollongong": "AU",
    "logan city": "AU",
    "geelong": "AU",
    "hobart": "AU",
    "darwin": "AU",
    "cairns": "AU",
    "townsville": "AU",
    "toowoomba": "AU",
    "ballarat": "AU",
    "bendigo": "AU",
    "launceston": "AU",
    "albany": "AU",
    "new south wales": "AU",
    "nsw": "AU",
    "victoria": "AU",
    "vic": "AU",
    "queensland": "AU",
    "qld": "AU",
    "western australia": "AU",
    "wa": "AU",
    "south australia": "AU",
    "sa": "AU",
    "tasmania": "AU",
    "tas": "AU",
    "australian capital territory": "AU",
    "act": "AU",
    "northern territory": "AU",
    "nt": "AU",

    # ============================================
    # New Zealand - Cities, Regions
    # ============================================
    "auckland": "NZ",
    "wellington": "NZ",
    "christchurch": "NZ",
    "hamilton": "NZ",
    "dunedin": "NZ",
    "palmerston north": "NZ",
    "nelson": "NZ",
    "rotorua": "NZ",
    "napier": "NZ",
    "hastings": "NZ",
    "new plymouth": "NZ",
    "whangarei": "NZ",
    "invercargill": "NZ",
    "upper hutt": "NZ",
    "lower hutt": "NZ",
    "porirua": "NZ",
    " north shore": "NZ",
    "wanaka": "NZ",
    "queenstown": "NZ",
    "tauranga": "NZ",
    "gisborne": "NZ",
    "west coast": "NZ",
    "canterbury": "NZ",
    "otago": "NZ",
    "southland": "NZ",
    "waikato": "NZ",
    "bay of plenty": "NZ",
    "hawke's bay": "NZ",
    "manawatu": "NZ",
    "taranaki": "NZ",

    # ============================================
    # United Kingdom - England, Scotland, Wales, N.Ireland
    # ============================================
    "london": "GB",
    "manchester": "GB",
    "birmingham": "GB",
    "liverpool": "GB",
    "leeds": "GB",
    "glasgow": "GB",
    "edinburgh": "GB",
    "bristol": "GB",
    "sheffield": "GB",
    "cardiff": "GB",
    "belfast": "GB",
    "newcastle": "GB",
    "nottingham": "GB",
    "leicester": "GB",
    "brighton": "GB",
    "oxford": "GB",
    "cambridge": "GB",
    "york": "GB",
    "plymouth": "GB",
    "southampton": "GB",
    "portsmouth": "GB",
    "bournemouth": "GB",
    "reading": "GB",
    "milton keynes": "GB",
    "derby": "GB",
    "stoke-on-trent": "GB",
    "swansea": "GB",
    "aberdeen": "GB",
    "dundee": "GB",
    "norwich": "GB",
    "ipswich": "GB",
    "exeter": "GB",
    "gloucester": "GB",
    "cheltenham": "GB",
    "lincoln": "GB",
    "chester": "GB",
    "lancaster": "GB",
    "preston": "GB",
    "halifax": "GB",
    "huddersfield": "GB",
    "wakefield": "GB",
    "wolverhampton": "GB",
    "walsall": "GB",
    "coventry": "GB",
    "rugby": "GB",
    "northampton": "GB",
    "milton keynes": "GB",
    "luton": "GB",
    "basildon": "GB",
    "slough": "GB",
    "maidstone": "GB",
    "guildford": "GB",
    "woking": "GB",
    "bracknell": "GB",
    "stevenage": "GB",
    "watford": "GB",
    "harrow": "GB",
    "ealing": "GB",
    "barnet": "GB",
    "camden": "GB",
    "westminster": "GB",
    "kensington": "GB",
    "chelsea": "GB",
    "richmond": "GB",
    "kingston": "GB",
    "croydon": "GB",
    "bromley": "GB",
    "greenwich": "GB",
    "bexley": "GB",
    "dartford": "GB",
    "gravesend": "GB",
    "medway": "GB",
    "canterbury": "GB",
    "dover": "GB",
    "folkestone": "GB",
    "hastings": "GB",
    "eastbourne": "GB",
    "worthing": "GB",
    "chichester": "GB",
    "winchester": "GB",
    "salisbury": "GB",
    "bath": "GB",
    "wells": "GB",
    "truro": "GB",
    "st ives": "GB",
    "penzance": "GB",
    "scilly isles": "GB",
    "isle of wight": "GB",
    "isle of man": "GB",
    "channel islands": "GB",
    "jersey": "GB",
    "guernsey": "GB",
    "shetland": "GB",
    "orkney": "GB",
    "outer hebrides": "GB",
    "inner hebrides": "GB",
    "highlands": "GB",
    "scottish borders": "GB",
    "dumfries": "GB",
    "galloway": "GB",
    "ayrshire": "GB",
    "lanarkshire": "GB",
    "lothian": "GB",
    "fife": "GB",
    "tayside": "GB",
    "angus": "GB",
    "aberdeenshire": "GB",
    "moray": "GB",
    "caithness": "GB",
    "sutherland": "GB",
    "ross-shire": "GB",
    "inverness-shire": "GB",
    "argyll": "GB",
    "buteshire": "GB",
    "renfrewshire": "GB",
    "dunbartonshire": "GB",
    "stirlingshire": "GB",
    "clackmannanshire": "GB",
    "perthshire": "GB",
    "kinross-shire": "GB",
    "angus": "GB",
    "kincardineshire": "GB",
    "banffshire": "GB",
    "elginshire": "GB",
    "nairnshire": "GB",
    "midlothian": "GB",
    "east lothian": "GB",
    "west lothian": "GB",
    "peeblesshire": "GB",
    "selkirkshire": "GB",
    "roxburghshire": "GB",
    "berwickshire": "GB",
    "wigtownshire": "GB",
    "kirkcudbrightshire": "GB",
    " ayrshire": "GB",
    "buteshire": "GB",
    "clwyd": "GB",
    "dyfed": "GB",
    "ged": "GB",
    "powys": "GB",
    " gwent": "GB",
    "monmouthshire": "GB",
    " conwy": "GB",
    "denbighshire": "GB",
    "flintshire": "GB",
    "anglesey": "GB",
    "pembrokeshire": "GB",
    "carmarthenshire": "GB",
    "ceredigion": "GB",
    "brecknockshire": "GB",
    "radnorshire": "GB",
    "montgomeryshire": "GB",
    "antrim": "GB",
    "armagh": "GB",
    "down": "GB",
    "fermanagh": "GB",
    "londonderry": "GB",
    "tyrone": "GB",
    "derry": "GB",
    "belfast": "GB",
    "lisburn": "GB",
    "newry": "GB",
    "bangor": "GB",
    "carryduff": "GB",
    "newtownabbey": "GB",
    " Larne": "GB",
    "ballymena": "GB",
    " coleraine": "GB",
    "craigavon": "GB",
    "dungannon": "GB",
    "eniskillen": "GB",
    "omagh": "GB",
    "strabane": "GB",
    "lisburn": "GB",

    # ============================================
    # United States - States, Major Cities
    # ============================================
    "new york": "US",
    "los angeles": "US",
    "chicago": "US",
    "houston": "US",
    "phoenix": "US",
    "philadelphia": "US",
    "san antonio": "US",
    "san diego": "US",
    "dallas": "US",
    "san jose": "US",
    "austin": "US",
    "jacksonville": "US",
    "fort worth": "US",
    "columbus": "US",
    "charlotte": "US",
    "san francisco": "US",
    "indianapolis": "US",
    "seattle": "US",
    "denver": "US",
    "washington": "US",
    "boston": "US",
    "el paso": "US",
    "nashville": "US",
    "detroit": "US",
    "portland": "US",
    "memphis": "US",
    "oklahoma city": "US",
    "las vegas": "US",
    "louisville": "US",
    "baltimore": "US",
    "milwaukee": "US",
    "albuquerque": "US",
    "tucson": "US",
    "fresno": "US",
    "sacramento": "US",
    "kansas city": "US",
    "mesa": "US",
    "atlanta": "US",
    "omaha": "US",
    "colorado springs": "US",
    "raleigh": "US",
    "miami": "US",
    "long beach": "US",
    "virginia beach": "US",
    "oakland": "US",
    "minneapolis": "US",
    "tulsa": "US",
    "arlington": "US",
    "tampa": "US",
    "new orleans": "US",
    "wichita": "US",
    "cleveland": "US",
    "bakersfield": "US",
    "aurora": "US",
    "anaheim": "US",
    "honolulu": "US",
    "santa ana": "US",
    "riverside": "US",
    "corpus christi": "US",
    "lexington": "US",
    "stockton": "US",
    "st. louis": "US",
    "saint paul": "US",
    "henderson": "US",
    "pittsburgh": "US",
    "cincinnati": "US",
    "anchorage": "US",
    "greensboro": "US",
    "plano": "US",
    "newark": "US",
    "lincoln": "US",
    "orlando": "US",
    "irvine": "US",
    "toledo": "US",
    "jersey city": "US",
    "chula vista": "US",
    "durham": "US",
    "fort wayne": "US",
    "st. petersburg": "US",
    "laredo": "US",
    "buffalo": "US",
    "madison": "US",
    "lubbock": "US",
    "chandler": "US",
    "scottsdale": "US",
    "glendale": "US",
    "reno": "US",
    "norfolk": "US",
    "wins ton": "US",
    "north las vegas": "US",
    "gilbert": "US",
    "chesapeake": "US",
    "irving": "US",
    "hialeah": "US",
    "garland": "US",
    "fremont": "US",
    "richmond": "US",
    "boise": "US",
    "birmingham": "US",
    "baton rouge": "US",
    "des moines": "US",
    "spokane": "US",
    "san bernardino": "US",
    "modesto": "US",
    "tacoma": "US",
    "fontana": "US",
    "santa clarita": "US",
    "baltimore": "US",
    "ogden": "US",
    "kennewick": "US",
    "west valley city": "US",
    "yonkers": "US",

    # US States
    "alabama": "US",
    "alaska": "US",
    "arizona": "US",
    "arkansas": "US",
    "california": "US",
    "colorado": "US",
    "connecticut": "US",
    "delaware": "US",
    "florida": "US",
    "georgia": "US",
    "hawaii": "US",
    "idaho": "US",
    "illinois": "US",
    "indiana": "US",
    "iowa": "US",
    "kansas": "US",
    "kentucky": "US",
    "louisiana": "US",
    "maine": "US",
    "maryland": "US",
    "massachusetts": "US",
    "michigan": "US",
    "minnesota": "US",
    "mississippi": "US",
    "missouri": "US",
    "montana": "US",
    "nebraska": "US",
    "nevada": "US",
    "new hampshire": "US",
    "new jersey": "US",
    "new mexico": "US",
    "new york": "US",
    "north carolina": "US",
    "north dakota": "US",
    "ohio": "US",
    "oklahoma": "US",
    "oregon": "US",
    "pennsylvania": "US",
    "rhode island": "US",
    "south carolina": "US",
    "south dakota": "US",
    "tennessee": "US",
    "texas": "US",
    "utah": "US",
    "vermont": "US",
    "virginia": "US",
    "washington": "US",
    "west virginia": "US",
    "wisconsin": "US",
    "wyoming": "US",
    "district of columbia": "US",
    "dc": "US",
    "guam": "US",
    "puerto rico": "US",
    "american samoa": "US",
    "northern mariana islands": "US",
    "us virgin islands": "US",

    # ============================================
    # Canada - Provinces, Major Cities
    # ============================================
    "toronto": "CA",
    "montreal": "CA",
    "vancouver": "CA",
    "calgary": "CA",
    "edmonton": "CA",
    "ottawa": "CA",
    "winnipeg": "CA",
    "quebec city": "CA",
    "hamilton": "CA",
    "kitchener": "CA",
    "london": "CA",
    "victoria": "CA",
    "halifax": "CA",
    "saskatoon": "CA",
    "regina": "CA",
    "st. john's": "CA",
    "kelowna": "CA",
    "windsor": "CA",
    "barrie": "CA",
    "sudbury": "CA",
    "thunder bay": "CA",
    "sherbrooke": "CA",
    "trois-rivieres": "CA",
    "saint john": "CA",
    "moncton": "CA",
    "fredericton": "CA",
    "charlottetown": "CA",
    "yellowknife": "CA",
    "whitehorse": "CA",
    "iqaluit": "CA",
    "ontario": "CA",
    "quebec": "CA",
    "british columbia": "CA",
    "alberta": "CA",
    "manitoba": "CA",
    "saskatchewan": "CA",
    "nova scotia": "CA",
    "new brunswick": "CA",
    "prince edward island": "CA",
    "newfoundland and labrador": "CA",
    "yukon": "CA",
    "northwest territories": "CA",
    "nunavut": "CA",
}


# Banks organized by common name with default country
# Structure: { "bank_key": {"default": "XX", "XX": BankInfo(...), ...} }
# Loaded from JSON file for easier maintenance
_BANKS_JSON_PATH = Path(__file__).parent / "bank_swift_codes.json"

with open(_BANKS_JSON_PATH, 'r', encoding='utf-8') as f:
    _banks_data = json.load(f)

# Rebuild BANKS_BY_NAME structure from JSON
BANKS_BY_NAME: dict[str, dict[str, BankInfo | str]] = {}
for entry in _banks_data['banks']:
    if entry['key'] not in BANKS_BY_NAME:
        BANKS_BY_NAME[entry['key']] = {}
    BANKS_BY_NAME[entry['key']][entry['country']] = BankInfo(
        name=entry['name'],
        swift_code=entry['swift_code'],
        country=entry['country']
    )

# Add default country for each bank key
for bank_key, default_country in _banks_data.get('defaults', {}).items():
    if bank_key in BANKS_BY_NAME:
        BANKS_BY_NAME[bank_key]['default'] = default_country


# ============================================
# NOTE: The BANKS_BY_NAME dictionary is now loaded from bank_swift_codes.json
# for easier maintenance and LLM verification.
# To add or update bank entries, edit bank_swift_codes.json instead of this file.
# ============================================


def detect_bank_name_in_text(text: str) -> Optional[str]:
    """
    Detect bank common name (key) from text.

    Uses word boundary matching to find common bank names like
    "DBS", "OCBC", "HSBC", "Citibank", etc. Returns the longest match
    to prefer "Standard Chartered" over "Standard".

    Args:
        text: Text to search for bank names

    Returns:
        Bank key (e.g., "dbs", "hsbc") or None if not found
    """
    if not text:
        return None

    import re
    text_lower = text.lower()

    best_match_key = None
    best_match_len = 0

    for bank_key in BANKS_BY_NAME.keys():
        # Use word boundary matching
        if re.search(rf'\b{re.escape(bank_key)}\b', text_lower):
            if len(bank_key) > best_match_len:
                best_match_key = bank_key
                best_match_len = len(bank_key)

    return best_match_key


def get_bank_info(bank_key: str, country: str = None) -> Optional[BankInfo]:
    """
    Get BankInfo for a bank key, with optional country.

    Args:
        bank_key: Bank common name (e.g., "dbs", "hsbc")
        country: ISO country code (e.g., "SG"). If None, uses default.

    Returns:
        BankInfo for the specified or default country
    """
    if not bank_key:
        return None

    bank_key_lower = bank_key.lower()
    if bank_key_lower not in BANKS_BY_NAME:
        return None

    bank_countries = BANKS_BY_NAME[bank_key_lower]

    # Use specified country if available
    if country:
        country_upper = country.upper()
        if country_upper in bank_countries:
            return bank_countries[country_upper]

    # Fall back to default country
    default_country = bank_countries.get("default")
    if default_country and default_country in bank_countries:
        return bank_countries[default_country]

    return None


def lookup_swift(bank_name: str, country: str = None) -> Optional[BankInfo]:
    """
    Look up bank SWIFT code by bank name, optionally filtering by country.

    Args:
        bank_name: Bank name to search for
        country: ISO 3166-1 alpha-2 country code (e.g., "SG", "US")

    Returns:
        BankInfo if found, None otherwise
    """
    if not bank_name:
        return None

    bank_name_lower = bank_name.strip().lower()

    # Try exact match first
    if bank_name_lower in BANKS_BY_NAME:
        return get_bank_info(bank_name_lower, country)

    # Partial match
    for bank_key in BANKS_BY_NAME.keys():
        if bank_key in bank_name_lower or bank_name_lower in bank_key:
            return get_bank_info(bank_key, country)

    return None


def get_all_banks_for_country(country: str) -> List[BankInfo]:
    """Get all banks for a specific country."""
    country_upper = country.upper()

    # Collect unique banks for this country
    seen_swift = set()
    banks = []
    for bank_key, bank_countries in BANKS_BY_NAME.items():
        if country_upper in bank_countries:
            bank_info = bank_countries[country_upper]
            if bank_info.swift_code not in seen_swift:
                seen_swift.add(bank_info.swift_code)
                banks.append(bank_info)
    return banks


def get_country_for_currency(currency: str) -> Optional[str]:
    """
    Get country code for a currency code.

    Args:
        currency: ISO 4217 currency code (e.g., "SGD", "USD")

    Returns:
        ISO 3166-1 alpha-2 country code, or None if not found
    """
    if not currency:
        return None
    return CURRENCY_COUNTRIES.get(currency.upper())


def detect_currency_in_text(text: str) -> Optional[str]:
    """
    Detect currency code from text by finding known currency abbreviations.

    Args:
        text: Text to search for currency codes

    Returns:
        First detected ISO 4217 currency code, or None if not found
    """
    if not text:
        return None

    text_upper = text.upper()
    # Check for currency codes (prioritize longer matches like "SGD" before "S")
    for currency in sorted(CURRENCY_COUNTRIES.keys(), key=len, reverse=True):
        # Use word boundary matching to avoid false positives
        import re
        if re.search(rf'\b{currency}\b', text_upper):
            return currency
    return None


def get_country_code_from_name(country_name: str) -> Optional[str]:
    """
    Get ISO country code from full country name.

    Args:
        country_name: Full country name (e.g., "Singapore", "United States")

    Returns:
        ISO 3166-1 alpha-2 country code, or None if not found
    """
    if not country_name:
        return None
    return COUNTRY_NAMES.get(country_name.lower().strip())


def detect_country_in_text(text: str) -> Optional[str]:
    """
    Detect country code from text by finding known country names, cities, or regions.

    First checks for explicit country names (e.g., "Thailand", "Singapore").
    If no country name found, checks for known cities/states/provinces (e.g., "Ayutthaya" -> TH).

    Args:
        text: Text to search for country names or locations

    Returns:
        ISO 3166-1 alpha-2 country code of first detected country, or None
    """
    if not text:
        return None

    text_lower = text.lower()

    # First, check explicit country names (longer names first)
    for name in sorted(COUNTRY_NAMES.keys(), key=len, reverse=True):
        if name in text_lower:
            return COUNTRY_NAMES[name]

    # If no country name found, check for known cities/states/provinces
    # Check longer names first (e.g., "phra nakhon si ayutthaya" before "wang noi")
    for location in sorted(LOCATION_TO_COUNTRY.keys(), key=len, reverse=True):
        if location in text_lower:
            return LOCATION_TO_COUNTRY[location]

    return None


def detect_bank_in_text(text: str, country: str = None) -> Optional[BankInfo]:
    """
    Detect bank from text by matching known bank names.

    Uses word boundary matching to find common bank names like
    "DBS", "OCBC", "HSBC", "Citibank", etc. Returns the longest match
    to prefer "Standard Chartered" over "Standard".

    Args:
        text: Text to search for bank names
        country: Optional ISO country code to get country-specific SWIFT

    Returns:
        BankInfo if found, None otherwise
    """
    bank_key = detect_bank_name_in_text(text)
    if bank_key:
        return get_bank_info(bank_key, country)
    return None
