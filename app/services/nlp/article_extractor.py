"""
Article content extractor for full-text news article analysis.

Uses trafilatura for reliable article text extraction with anti-bot measures:
- User-Agent rotation
- Rate limiting between requests
- Retry logic with exponential backoff

Trafilatura is preferred over newspaper3k because:
- Better extraction quality
- Handles soft paywalls (loads content before JS paywall activates)
- More modern and actively maintained
- Faster execution

Note: Sites with DataDome/Cloudflare bot protection (NYTimes, etc.) will fail
extraction and fall back to search snippets in the caller.
"""

import asyncio
import random
import time
from typing import Optional, Dict, Any
from app.core.logger import get_logger
from app.config.osint_config import osint_settings


# User agent strings for rotation (same as google_scraper.py)
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


class ArticleExtractor:
    """Extract full article content from news URLs with anti-bot measures."""

    def __init__(self):
        """Initialize article extractor with config settings."""
        self.logger = get_logger()
        # Use config settings with fallback defaults
        self.min_delay = getattr(osint_settings, 'article_extraction_min_delay', 2.0)
        self.max_retries = getattr(osint_settings, 'article_extraction_max_retries', 3)
        self.last_request_time = 0

    async def _rate_limit(self):
        """Enforce minimum delay between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_delay:
            delay = self.min_delay - elapsed
            # Add small random jitter to avoid detection
            jitter = random.uniform(0, 0.5)
            await asyncio.sleep(delay + jitter)

        self.last_request_time = time.time()

    async def extract_article(
        self,
        url: str,
        timeout: int = 10,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Extract article content using trafilatura with anti-bot measures.

        Note: DataDome/Cloudflare-protected sites will fail extraction.
        The caller should fall back to search snippets when None is returned.

        Args:
            url: Article URL
            timeout: Request timeout in seconds
            retry_count: Current retry attempt number

        Returns:
            {
                'title': str,
                'text': str,
                'url': str,
                'extraction_method': 'trafilatura'
            }
            or None if extraction fails
        """
        # Rate limiting before each request
        await self._rate_limit()

        try:
            import trafilatura

            # Select random user agent
            user_agent = random.choice(USER_AGENTS)

            # Download page with custom headers using requests
            self.logger.debug(f"Extracting article from {url} (attempt {retry_count + 1}/{self.max_retries + 1})")

            def fetch_with_headers():
                """Fetch URL with custom headers."""
                import requests
                response = requests.get(
                    url,
                    headers={'User-Agent': user_agent},
                    timeout=timeout
                )
                response.raise_for_status()
                return response.text, response.headers

            downloaded, headers = await asyncio.to_thread(fetch_with_headers)

            if not downloaded:
                if retry_count < self.max_retries:
                    # Exponential backoff
                    backoff_delay = 2 ** retry_count
                    self.logger.warning(f"Failed to download {url}, retrying in {backoff_delay}s...")
                    await asyncio.sleep(backoff_delay)
                    return await self.extract_article(url, timeout, retry_count + 1)
                else:
                    self.logger.warning(f"No content downloaded from {url} after {self.max_retries + 1} attempts")
                    return None

            # Extract article content
            result = await asyncio.to_thread(
                trafilatura.extract,
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )

            if not result:
                if retry_count < self.max_retries:
                    backoff_delay = 2 ** retry_count
                    self.logger.warning(f"No content extracted from {url}, retrying in {backoff_delay}s...")
                    await asyncio.sleep(backoff_delay)
                    return await self.extract_article(url, timeout, retry_count + 1)
                else:
                    self.logger.warning(f"No text extracted from {url} after {self.max_retries + 1} attempts")
                    return None

            # Extract title using BeautifulSoup
            def extract_title_from_html():
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(downloaded, 'html.parser')
                    # Try to get title from various meta tags
                    title_tag = soup.find('title')
                    if title_tag and title_tag.string:
                        return title_tag.string.strip()
                    # Try og:title
                    og_title = soup.find('meta', property='og:title')
                    if og_title and og_title.get('content'):
                        return og_title['content'].strip()
                    # Try twitter:title
                    twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
                    if twitter_title and twitter_title.get('content'):
                        return twitter_title['content'].strip()
                    return None
                except Exception:
                    return None

            title = await asyncio.to_thread(extract_title_from_html)

            self.logger.info(f"Successfully extracted article from {url} (title: {title[:50] if title else 'N/A'})")

            return {
                'title': title or '',
                'text': result,
                'url': url,
                'extraction_method': 'trafilatura'
            }

        except Exception as e:
            # Log the error and retry if we have attempts left
            self.logger.error(f"Error extracting article from {url}: {e}")

            if retry_count < self.max_retries:
                # Exponential backoff on error
                backoff_delay = 2 ** retry_count
                self.logger.info(f"Retrying {url} in {backoff_delay}s due to error: {e}")
                await asyncio.sleep(backoff_delay)
                return await self.extract_article(url, timeout, retry_count + 1)

            return None

    def is_paywalled(self, html: str) -> bool:
        """Detect if article is behind paywall."""
        paywall_indicators = [
            'subscribe to continue',
            'create an account to continue',
            'premium content',
            'subscriber only'
        ]
        html_lower = html.lower()
        return any(indicator in html_lower for indicator in paywall_indicators)


# Global instance
article_extractor = ArticleExtractor()
