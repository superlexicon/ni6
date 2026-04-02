# OSINT Background Screening Service

Free, web-scraping-based background search service as an alternative to World-Check.

## Overview

The OSINT (Open Source Intelligence) Background Screening Service provides comprehensive background checks using freely available sources and web scraping. It's designed as a cost-effective alternative to commercial services like World-Check, particularly for markets in India, Malaysia, Thailand, Indonesia, and the Philippines.

## Features

### Data Sources

| Category | Sources | Method |
|----------|---------|--------|
| **Web Search** | DuckDuckGo, Google (via scraping) | API + Playwright |
| **Sanctions Lists** | OFAC SDN, EU Sanctions, UN Consolidated | Free CSV downloads |
| **Country Records** | MCA (India), SSM (Malaysia), DBD (Thailand), AHU (Indonesia), SEC (Philippines) | Web scraping |
| **Social Media** | LinkedIn, Twitter/X | Playwright scraping (optional) |

### Risk Assessment

The service calculates a comprehensive risk score (0-100) based on:

- **Negative News** (30%): Fraud, scams, criminal activity keywords
- **Identity Consistency** (20%): Name/DOB/address verification
- **Criminal Records** (25%): Sanctions, court records
- **Social Media Risk** (15%): Suspicious activity patterns
- **Digital Footprint** (10%): Web presence analysis

### Risk Categories

| Score Range | Category | Action |
|-------------|----------|--------|
| 0-24 | LOW | Auto-approve |
| 25-49 | MEDIUM | Manual review optional |
| 50-74 | HIGH | Manual review required |
| 75-100 | CRITICAL | Auto-reject |

## Installation

### Dependencies

```bash
# Core dependencies
pip install ddgs
pip install playwright
pip install aiohttp

# Install Playwright browsers
playwright install chromium
```

### Environment Variables

Add to your `.env` file:

```bash
# Screening Provider Selection
SCREENING_PROVIDER=osint  # Options: worldcheck, osint, both

# OSINT Configuration
OSINT_RISK_THRESHOLD=50.0
OSINT_ENABLE_WEB_SEARCH=true
OSINT_ENABLE_SANCTIONS_CHECK=true

# Rate Limiting (respectful scraping)
OSINT_MIN_DELAY_SECONDS=2.0
OSINT_MAX_DELAY_SECONDS=5.0
OSINT_CACHE_RESULTS_HOURS=24

# Browser Automation
OSINT_HEADLESS_BROWSER=true
OSINT_BROWSER_TYPE=playwright
OSINT_ROTATE_USER_AGENTS=true
```

## Usage

### Basic Screening

```python
from app.services.osint_screening_service import osint_screening_service

result = await osint_screening_service.screen_individual(
    full_name="John Doe",
    date_of_birth="1980-01-15",
    country="United States",
    address="123 Main St, New York, NY"
)

print(f"Risk Score: {result['overall_risk_score']}")
print(f"Risk Category: {result['risk_category']}")
print(f"Match Found: {result['is_match']}")
```

### Response Format

```python
{
    # World-Check compatible fields
    "is_match": bool,  # True if risk_score >= threshold
    "match_details": list,
    "screening_id": str,

    # OSINT-specific fields
    "overall_risk_score": float,  # 0-100
    "risk_category": str,  # LOW, MEDIUM, HIGH, CRITICAL
    "component_scores": {
        "negative_news": float,
        "identity_consistency": float,
        "criminal_records": float,
        "social_media_risk": float,
        "digital_footprint": float
    },
    "search_sources": {
        "web_search": {
            "results_count": int,
            "negative_news_count": int,
            "sources": list
        },
        "public_records": {
            "sanctions": dict,
            "country_specific": dict
        }
    }
}
```

## Architecture

### Service Structure

```
app/services/osint_screening_service.py    # Main service orchestrator
app/config/osint_config.py                  # Configuration settings
app/services/osint/
├── search_providers/                       # Web search providers
│   ├── duckduckgo_search_provider.py      # DuckDuckGo (primary)
│   └── google_scraper.py                  # Google scraper (fallback)
├── sanctions/                              # Sanctions list checkers
│   └── sanctions_list_checker.py          # OFAC, EU, UN
├── country_specific/                       # Country-specific scrapers
│   ├── india_mca_scraper.py               # India MCA
│   ├── malaysia_ssm_scraper.py            # Malaysia SSM
│   ├── thailand_dbd_scraper.py            # Thailand DBD
│   ├── indonesia_ahu_scraper.py           # Indonesia AHU
│   └── philippines_sec_scraper.py         # Philippines SEC
└── risk_scorer.py                          # Risk scoring algorithm
```

### Data Flow

```
User Input (Name, DOB, Country, Address)
         ↓
OSINT Screening Service
         ↓
┌─────────────────────────────────┐
│  Parallel Search Execution      │
├─────────────────────────────────┤
│ • Web Search (DuckDuckGo/Google)│
│ • Sanctions Check (OFAC/EU/UN)  │
│ • Country Records (if supported)│
└─────────────────────────────────┘
         ↓
Risk Scoring Engine (0-100 scale)
         ↓
Formatted Response (World-Check compatible)
         ↓
Database Storage
```

## Country-Specific Scrapers

### India (MCA - Ministry of Corporate Affairs)
- **Data**: Company director information, DIN verification
- **Source**: https://www.mca.gov.in/
- **Method**: Web search (MCA requires login for direct access)

### Malaysia (SSM - Suruhanjaya Syarikat Malaysia)
- **Data**: Business registrations, company directorships
- **Source**: https://www.ssm.com.my/
- **Method**: Web search (SSM e-Info requires paid account)

### Thailand (DBD - Department of Business Development)
- **Data**: Company registrations, director information
- **Source**: https://datawarehouse.dbd.go.th/
- **Method**: Web search

### Indonesia (AHU - Administrasi Hukum Umum)
- **Data**: Company registrations (PT, CV, Firma)
- **Source**: https://ahu.go.id/
- **Method**: Web search

### Philippines (SEC - Securities & Exchange Commission)
- **Data**: Corporation registrations
- **Source**: https://www.sec.gov.ph/
- **Method**: Web search

## Testing

### Run Test Script

```bash
# Test with default case (United States)
python scripts/test_osint_screening.py

# Test with specific case (1-6)
python scripts/test_osint_screening.py --test-case 2

# Test configuration only
python scripts/test_osint_screening.py --config-only
```

### Test Cases

1. United States (basic test)
2. India (MCA)
3. Malaysia (SSM)
4. Thailand (DBD)
5. Indonesia (AHU)
6. Philippines (SEC)

## Performance Considerations

### Caching

- **Search Results**: 24-hour cache (configurable via `OSINT_CACHE_RESULTS_HOURS`)
- **Sanctions Lists**: 24-hour cache
- **Storage**: In-memory (production: use Redis)

### Rate Limiting

- **Respectful Scraping**: 2-5 second delays between requests
- **Configurable**: `OSINT_MIN_DELAY_SECONDS`, `OSINT_MAX_DELAY_SECONDS`
- **User Agent Rotation**: Enabled by default to avoid detection

### Browser Resources

- **Headless Mode**: Enabled by default (`OSINT_HEADLESS_BROWSER=true`)
- **Browser Type**: Playwright (Chromium)
- **Cleanup**: Automatic browser closure after requests

## Cost Comparison

| Provider | Monthly Cost | Data Sources | Setup Time |
|----------|--------------|--------------|------------|
| **World-Check** | $1000-5000 | Watchlists, PEPs, Sanctions | Enterprise |
| **OSINT (Free)** | $0 | Web scraping + free sanctions lists | Days |
| **OpenSanctions** | €0.10/call | Sanctions, PEPs, crime | Days |

**Savings**: 100% cost reduction with free OSINT approach (server costs only)

## Trade-offs

### Advantages
- Zero licensing costs
- Comprehensive web presence analysis
- Flexible and extensible
- No API dependencies for core features

### Limitations
- Web scraping can break (requires monitoring)
- No guaranteed uptime/API support
- Higher maintenance (scrapers may need updates)
- Potential legal/ToS considerations
- False positives may be higher

## Legal & Compliance

1. **GDPR/Privacy**: Ensure data processing complies with local regulations
2. **FCRA (US)**: If used for employment, comply with Fair Credit Reporting Act
3. **Data Retention**: Implement automatic deletion after 30-90 days
4. **Consent**: Ensure users consent to background screening in ToS
5. **API Terms**: Respect rate limits and usage policies
6. **Country-Specific Laws**: Some countries restrict public record access

## Troubleshooting

### Playwright Issues

```bash
# Reinstall browsers
playwright install --force chromium

# Check if browsers are installed
playwright install --dry-run chromium
```

### Sanctions List Download Failures

- Check internet connectivity
- Verify official sources are accessible:
  - OFAC: https://www.treasury.gov/ofac/downloads/sdn.csv
  - EU: https://webgate.ec.europa.eu/fsd/fsf/files/files/consolidated_list_xml/csv.zip
  - UN: https://sc.un.org/consolidated/lists/consolidated.csv

### Google Scraper Failures

- Check if headless browser is enabled
- Verify user agent rotation is enabled
- Increase delay times if rate-limited
- Falls back to DuckDuckGo automatically

### Country-Specific Scraper Issues

- Country-specific scrapers use web search as fallback
- Official portals may require login/paid access
- Results may be limited based on public data availability

## Future Enhancements

- [ ] Add more country-specific scrapers
- [ ] Implement social media scrapers (LinkedIn, Twitter/X)
- [ ] Add OpenCorporates free tier integration
- [ ] Implement Redis caching for production
- [ ] Add more sophisticated identity verification
- [ ] Add court record scrapers
- [ ] Implement proxy rotation for scraping
- [ ] Add CAPTCHA handling

## License

This module is part of the IM-OSINT project.
