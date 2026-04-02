"""
OSINT (Open Source Intelligence) Background Search Configuration.

Free sources only - web scraping based approach for identity verification
and risk assessment as an alternative to World-Check.

Target Markets: India, Malaysia, Thailand, Indonesia, Philippines
"""

from pydantic_settings import BaseSettings


class OSINTSettings(BaseSettings):
    """OSINT background search configuration using free sources only."""

    # Risk threshold for rejecting applicants (0-100)
    risk_threshold: float = 50.0

    # Enable/disable specific search sources
    enable_web_search: bool = True
    enable_sanctions_check: bool = True  # Free sanctions lists

    # Rate limiting for scraping (respectful scraping)
    requests_per_minute: int = 10  # Conservative for free sources
    max_concurrent_scrapers: int = 3
    search_timeout_seconds: int = 30
    page_load_timeout_seconds: int = 15

    # Caching
    cache_results_hours: int = 24

    # Browser automation
    enable_browser_automation: bool = True
    headless_browser: bool = True
    browser_type: str = "playwright"  # playwright or selenium

    # Scraping delays (polite scraping)
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 5.0

    # User agent rotation (avoid detection)
    rotate_user_agents: bool = True

    # Sanctions lists cache duration (hours)
    sanctions_cache_hours: int = 24

    # Sanctions sync configuration
    # NOTE: Syncing is now handled by separate OSSPEP Quarkus scraper service
    # This application only reads from database - NO scraping/syncing
    sanctions_sync_hour: int = 2
    sanctions_sync_minute: int = 0
    enable_sanctions_sync: bool = False  # Disabled - handled by OSSPEP
    sanctions_sync_timeout: int = 300

    # PEP (Politically Exposed Persons) configuration
    enable_pep_check: bool = True  # Enable PEP screening (read-only from DB)
    pep_cache_hours: int = 24  # PEP cache duration

    # PEP sync configuration
    # NOTE: Syncing is now handled by separate OSSPEP Quarkus scraper service
    # This application only reads from database - NO scraping/syncing
    enable_pep_sync: bool = False  # Disabled - handled by OSSPEP
    pep_sync_hour: int = 3  # 3 AM (after sanctions sync at 2 AM)
    pep_sync_day_of_week: int = 6  # Sunday (0=Monday, 6=Sunday)
    pep_sync_timeout: int = 600  # 10 minutes (Wikipedia scraping can be slow)

    # Efficient database-side filtering configuration
    # These settings control the probabilistic name matching optimization
    filtered_query_limit: int = 1000  # Max candidates to return from filtered DB query
    enable_efficient_filtering: bool = True  # Enable database-side pre-filtering

    # NLP-enhanced negative news detection
    enable_nlp_enhanced_analysis: bool = True
    nlp_sentiment_threshold: float = -0.3  # VADER compound threshold for negative
    nlp_deep_analysis_limit: int = 5  # Max articles to analyze with FinBERT
    nlp_article_fetch_timeout: int = 10  # Seconds to wait for article fetch

    # Article extraction configuration (trafilatura)
    article_extraction_min_delay: float = 2.0  # Minimum delay between article downloads (seconds)
    article_extraction_max_retries: int = 3  # Maximum retry attempts for failed extractions
    enable_article_extraction_fallback: bool = True  # Use search snippets when extraction fails

    # Face-verified negative news analysis
    # Use image search + face matching to verify articles are about the correct person
    # Set face_verified_max_images=0 to disable this feature
    face_verified_similarity_threshold: float = 0.85  # Same as passport/selfie matching
    face_verified_max_images: int = 10

    model_config = {
        "env_prefix": "OSINT_",
        "env_file": ".env",
        "extra": "ignore"
    }


# Global settings instance
osint_settings = OSINTSettings()
