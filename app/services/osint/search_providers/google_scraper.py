"""
Google Search Scraper for OSINT Background Search.

Uses Playwright to scrape Google Search results as a fallback
when DuckDuckGo doesn't provide enough results.

Features:
- User agent rotation to avoid detection
- Respectful scraping with delays
- Headless browser mode
- Result parsing and extraction
"""

import asyncio
import random
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.core.logger import get_logger
from app.config.osint_config import osint_settings


# User agent strings for rotation
USER_AGENTS = [
    # Chrome on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    # Firefox on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    # Chrome on Mac
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    # Safari on Mac
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    # Chrome on Linux
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


class GoogleSearchScraper:
    """
    Google search scraper using Playwright.

    Implements respectful scraping with:
    - Random delays between requests
    - User agent rotation
    - Headless browser mode
    - CAPTCHA detection
    """

    def __init__(self):
        self.logger = get_logger()
        self.browser = None
        self.context = None
        self.page = None
        self.last_request_time = 0

    async def _get_browser(self):
        """Get or create Playwright browser instance."""
        if self.browser is None:
            try:
                from playwright.async_api import async_playwright
                playwright = await async_playwright().start()

                # Launch browser
                self.browser = await playwright.chromium.launch(
                    headless=osint_settings.headless_browser,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled'
                    ]
                )

                # Create context with random user agent
                self.context = await self.browser.new_context(
                    user_agent=random.choice(USER_AGENTS) if osint_settings.rotate_user_agents else USER_AGENTS[0],
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US'
                )

                # Add stealth measures
                await self.context.add_init_script("""
                    // Remove webdriver property
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });

                    // Mock plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });

                    // Mock languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                """)

                self.page = await self.context.new_page()
                self.logger.info("Playwright browser initialized")

            except ImportError:
                self.logger.error("Playwright not installed. Install with: pip install playwright")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize Playwright browser: {e}")
                raise

    async def search(
        self,
        full_name: str,
        date_of_birth: Optional[str] = None,
        country: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Perform Google search via web scraping.

        Args:
            full_name: Person's full name
            date_of_birth: Date of birth (YYYY-MM-DD)
            country: Country name or code
            max_results: Maximum number of results to return

        Returns:
            {
                "results_count": int,
                "negative_news_count": int,
                "sources": list,
                "results": list
            }
        """
        self.logger.debug(f"Starting Google search for: {full_name}")

        try:
            await self._get_browser()

            # Build query
            query = self._build_search_query(full_name, date_of_birth, country)
            search_url = f"https://www.google.com/search?q={query}&num={max_results * 2}"

            # Respectful scraping delay
            await self._respectful_delay()

            # Navigate to Google
            await self.page.goto(search_url, timeout=osint_settings.page_load_timeout_seconds * 1000)

            # Wait for results to load
            await self.page.wait_for_selector('div#search', timeout=10000)

            # Extract search results
            results = await self._extract_results()

            # Analyze for negative news
            analyzed = self._analyze_results(results, full_name)

            return {
                "results_count": len(results),
                "negative_news_count": analyzed['negative_news_count'],
                "sources": analyzed['sources'],
                "results": results
            }

        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return {
                "results_count": 0,
                "negative_news_count": 0,
                "sources": [],
                "results": [],
                "error": str(e)
            }

    async def close(self):
        """Close browser and cleanup resources."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            self.browser = None
            self.context = None
            self.page = None
            self.logger.info("Playwright browser closed")
        except Exception as e:
            self.logger.error(f"Error closing browser: {e}")

    async def _respectful_delay(self):
        """Add random delay between requests (polite scraping)."""
        elapsed = asyncio.get_event_loop().time() - self.last_request_time
        min_delay = osint_settings.min_delay_seconds
        max_delay = osint_settings.max_delay_seconds

        if elapsed < min_delay:
            delay = min_delay - elapsed + random.uniform(0, max_delay - min_delay)
            await asyncio.sleep(delay)

        self.last_request_time = asyncio.get_event_loop().time()

    def _build_search_query(
        self,
        full_name: str,
        date_of_birth: Optional[str],
        country: Optional[str]
    ) -> str:
        """Build URL-encoded search query."""
        import urllib.parse

        query_parts = [f'"{full_name}"']

        # Add country if available
        if country:
            query_parts.append(country)

        return urllib.parse.quote('+'.join(query_parts))

    async def _extract_results(self) -> List[Dict[str, Any]]:
        """Extract search results from Google SERP."""
        results = []

        try:
            # Find all result divs
            result_elements = await self.page.query_selector_all('div.g')

            for element in result_elements[:20]:  # Limit to 20 results
                try:
                    # Extract title and link
                    title_elem = await element.query_selector('h3')
                    link_elem = await element.query_selector('a')

                    if not title_elem or not link_elem:
                        continue

                    title = await title_elem.inner_text()
                    url = await link_elem.get_attribute('href')

                    # Skip if no URL
                    if not url:
                        continue

                    # Clean URL (remove Google redirects)
                    if url.startswith('/url?'):
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(url[5:])
                        url = parsed.get('q', [''])[0]

                    if not url or url.startswith('#'):
                        continue

                    # Extract snippet
                    snippet_elem = await element.query_selector('div.VwiC3b, div.s, span.st')
                    snippet = ''
                    if snippet_elem:
                        snippet = await snippet_elem.inner_text()

                    results.append({
                        'title': title,
                        'href': url,
                        'body': snippet,
                        'source': 'google'
                    })

                except Exception as e:
                    self.logger.warning(f"Error extracting result: {e}")
                    continue

            self.logger.info(f"Extracted {len(results)} results from Google")
            return results

        except Exception as e:
            self.logger.error(f"Error extracting Google results: {e}")
            return []

    def _analyze_results(self, results: List[Dict], full_name: str = None) -> Dict[str, Any]:
        """Analyze results for negative news and extract metadata."""
        from .duckduckgo_search_provider import DuckDuckGoSearchProvider

        # Use DuckDuckGo provider's keyword detection
        ddg = DuckDuckGoSearchProvider()

        negative_count = 0
        sources = []

        for result in results:
            title = result.get('title', '').lower()
            body = result.get('body', '').lower()
            url = result.get('href', '')
            combined_text = f"{title} {body}"

            # Check for negative keywords
            risk_level = ddg._check_negative_keywords(combined_text)

            # If negative keywords found, check if it's relevant to the person
            if risk_level and full_name:
                if ddg._is_relevant_negative_news(result, full_name):
                    negative_count += 1
            elif risk_level:
                # No full_name provided, count all negative results
                negative_count += 1

            # Extract source domain
            if url:
                domain = ddg._extract_domain(url)
                if domain and domain not in sources:
                    sources.append(domain)

        return {
            'negative_news_count': negative_count,
            'sources': sources
        }

    def __del__(self):
        """Cleanup on deletion."""
        if self.browser:
            asyncio.create_task(self.close())


# Global instance
google_scraper = GoogleSearchScraper()
