"""
Web scraper — dynamic page extraction with Playwright + httpx fallback.

Strategy:
    1. Try Playwright (JS rendering, dynamic wait, scroll)
    2. Fallback to httpx (static HTML) if Playwright unavailable
    3. Extract article list from frontpage via LLM
    4. Crawl each article content via crawl4ai or Playwright
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from services.utils.key_balancer import create_llm, setup_balancer

load_dotenv()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ArticleItem(BaseModel):
    title: str
    url: str
    date: Optional[str] = Field(None, description="YYYY-MM-DD or null")


class ArticleList(BaseModel):
    articles: list[ArticleItem]


@dataclass
class Article:
    title: str
    url: str
    date: Optional[str]
    source: str
    content_md: Optional[str] = None


@dataclass
class ExtractionError:
    url: str
    stage: str
    message: str


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """
You are a structured data extractor for news/blog websites.
Given the HTML of a frontpage, extract ALL listed articles.

Return ONLY valid JSON with this structure (no markdown, no explanation):
{
  "articles": [
    {
      "title": "Article title",
      "url": "https://full-url.com/article",
      "date": "YYYY-MM-DD"
    }
  ]
}

Rules:
- If URL is relative (e.g., /article/123), complete with the provided base domain.
- If date is not explicit, use null.
- Convert relative dates (e.g., "2 hours ago", "yesterday") to YYYY-MM-DD.
- Ignore ads, navigation links, footer — focus on articles/posts.
- Return max 50 articles per frontpage.
""".strip()


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def build_llm(model: str | None = None):
    """Build LLM with SmartKeyBalancer for extraction tasks."""
    try:
        setup_balancer()
        return create_llm(model)
    except Exception:
        # Fallback: direct OpenRouter without balancer
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )


# ---------------------------------------------------------------------------
# Fetch strategies
# ---------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def fetch_html_static(url: str, timeout: int = 20) -> str:
    """Fetch HTML via httpx (static, no JS execution)."""
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout, headers=DEFAULT_HEADERS
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def fetch_html_dynamic(
    url: str,
    wait_selector: str | None = None,
    wait_time_ms: int = 5000,
    scroll: bool = True,
    scroll_pause_ms: int = 1000,
    max_scrolls: int = 5,
    timeout: int = 30,
) -> str:
    """
    Fetch HTML via Playwright with JS rendering and dynamic wait.

    Args:
        url: Target URL.
        wait_selector: CSS selector to wait for before extracting HTML.
        wait_time_ms: Max time to wait for selector (ms).
        scroll: Whether to scroll down to trigger lazy loading.
        scroll_pause_ms: Pause between scrolls (ms).
        max_scrolls: Maximum number of scroll actions.
        timeout: Overall timeout (seconds).

    Returns:
        Rendered HTML string.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError("playwright not installed — run: pip install playwright && playwright install chromium")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            user_agent=DEFAULT_HEADERS["User-Agent"],
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)

            # Wait for specific selector if provided
            if wait_selector:
                try:
                    await page.wait_for_selector(
                        wait_selector, state="attached", timeout=wait_time_ms
                    )
                except Exception:
                    pass  # Continue even if selector not found

            # Additional wait for dynamic content
            await page.wait_for_timeout(min(wait_time_ms, 3000))

            # Scroll to trigger lazy loading / infinite scroll
            if scroll:
                prev_height = 0
                for _ in range(max_scrolls):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(scroll_pause_ms)
                    new_height = await page.evaluate("document.body.scrollHeight")
                    if new_height == prev_height:
                        break
                    prev_height = new_height

            # Wait for network to settle
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            html = await page.content()
            return html

        finally:
            await browser.close()


async def fetch_html_smart(
    url: str,
    wait_selector: str | None = None,
    dynamic: bool = True,
    **kwargs,
) -> str:
    """
    Smart HTML fetch with Playwright (dynamic) → httpx (static) fallback.

    Args:
        url: Target URL.
        wait_selector: CSS selector to wait for (dynamic mode only).
        dynamic: Try Playwright first if True.
        **kwargs: Extra args passed to fetch_html_dynamic.

    Returns:
        HTML string.
    """
    if dynamic:
        try:
            return await fetch_html_dynamic(url, wait_selector=wait_selector, **kwargs)
        except ImportError:
            pass  # Playwright not installed, fallback to static
        except Exception:
            pass  # Playwright failed, fallback to static

    return await fetch_html_static(url)


# ---------------------------------------------------------------------------
# Article extraction
# ---------------------------------------------------------------------------

async def extract_articles_from_html(
    html: str,
    base_url: str,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract article list from frontpage HTML using LLM.

    Args:
        html: Raw HTML of the frontpage.
        base_url: Base URL for resolving relative links.
        model: LLM model name (default from env).

    Returns:
        List of dicts with 'title', 'url', 'date'.
    """
    html_truncated = html[:80_000]
    llm = build_llm(model)

    try:
        structured_llm = llm.with_structured_output(ArticleList)
    except Exception:
        # Fallback: manual JSON parsing if structured output not supported
        structured_llm = llm

    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM),
        HumanMessage(content=f"Base URL: {base_url}\n\nFrontpage HTML:\n\n{html_truncated}"),
    ]

    result = await structured_llm.ainvoke(messages)

    # Handle both structured and unstructured output
    if isinstance(result, ArticleList):
        articles = result.articles
    elif isinstance(result, dict) and "articles" in result:
        articles = [ArticleItem(**a) for a in result["articles"]]
    else:
        # Try to parse from raw text
        raw = str(result)
        try:
            # Find JSON in response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                articles = [ArticleItem(**a) for a in parsed.get("articles", [])]
            else:
                articles = []
        except (json.JSONDecodeError, Exception):
            articles = []

    output: list[dict[str, Any]] = []
    for article in articles:
        item = article.model_dump() if hasattr(article, "model_dump") else dict(article)
        item["url"] = urljoin(base_url, item["url"])
        output.append(item)

    return output


# ---------------------------------------------------------------------------
# Article content crawling
# ---------------------------------------------------------------------------

async def crawl_article_content_crawl4ai(url: str) -> str:
    """Extract article content using crawl4ai (Playwright-based)."""
    try:
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.async_configs import BrowserConfig, CacheMode, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        browser_config = BrowserConfig(verbose=False)
        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            excluded_tags=["form", "header", "footer", "nav"],
            exclude_external_links=True,
            remove_overlay_elements=True,
            process_iframes=True,
            wait_for_images=False,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.6),
                options={"ignore_links": True},
            ),
            cache_mode=CacheMode.ENABLED,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            if not result.success:
                return f"[Extraction error (HTTP {result.status_code}): {result.error_message}]"
            return result.markdown.fit_markdown or result.markdown.raw_markdown or ""

    except ImportError:
        return "[crawl4ai not installed — run: pip install crawl4ai && crawl4ai-setup]"
    except Exception as e:
        return f"[Error: {e}]"


async def crawl_article_content_playwright(url: str, timeout: int = 30) -> str:
    """Extract article content using Playwright directly."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return "[playwright not installed — run: pip install playwright && playwright install chromium]"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            user_agent=DEFAULT_HEADERS["User-Agent"],
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)

            # Wait for article content to load
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass

            # Try to extract main content
            content = await page.evaluate("""
                () => {
                    // Try common article selectors
                    const selectors = [
                        'article',
                        '[role="main"]',
                        '.post-content',
                        '.article-content',
                        '.entry-content',
                        '.blog-post',
                        'main',
                        '.content',
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el && el.innerText.length > 200) {
                            return el.innerText;
                        }
                    }
                    // Fallback: body text
                    return document.body ? document.body.innerText : '';
                }
            """)

            return content or ""

        finally:
            await browser.close()


async def crawl_article_content(url: str, method: str = "crawl4ai") -> str:
    """
    Extract article content with configurable method.

    Args:
        url: Article URL.
        method: 'crawl4ai' (full browser) or 'playwright' (lightweight).

    Returns:
        Article content as markdown or plain text.
    """
    if method == "crawl4ai":
        result = await crawl_article_content_crawl4ai(url)
        if not result.startswith("["):
            return result
        # Fallback to playwright if crawl4ai fails
        return await crawl_article_content_playwright(url)
    else:
        return await crawl_article_content_playwright(url)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string to date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def filter_by_date(
    articles: list[dict[str, Any]],
    date_from: date,
    date_to: date,
) -> list[dict[str, Any]]:
    """Filter articles by date range."""
    result = []
    for a in articles:
        d = parse_date(a.get("date"))
        if d and date_from <= d <= date_to:
            result.append(a)
    return result


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

async def _crawl_with_semaphore(
    article: Article,
    semaphore: asyncio.Semaphore,
    progress_callback=None,
    crawl_method: str = "crawl4ai",
) -> None:
    """Crawl single article with semaphore-controlled concurrency."""
    async with semaphore:
        if progress_callback:
            progress_callback(f"  -> {article.title[:60]}...")
        article.content_md = await crawl_article_content(article.url, method=crawl_method)


async def process_frontpages(
    frontpage_urls: list[str],
    date_from: date,
    date_to: date,
    model: str | None = None,
    crawl_content: bool = True,
    max_articles: int = 20,
    progress_callback=None,
    max_concurrency: int = 5,
    crawl_method: str = "crawl4ai",
    dynamic_fetch: bool = True,
    wait_selector: str | None = None,
) -> list[Article]:
    """
    Process frontpages: extract articles and optionally crawl content.

    Args:
        frontpage_urls: List of frontpage URLs to scrape.
        date_from: Start date filter.
        date_to: End date filter.
        model: LLM model for article extraction.
        crawl_content: Whether to crawl individual article pages.
        max_articles: Max articles per frontpage.
        progress_callback: Optional callback for progress messages.
        max_concurrency: Max concurrent article crawls.
        crawl_method: 'crawl4ai' or 'playwright'.
        dynamic_fetch: Use Playwright for frontpage fetching.
        wait_selector: CSS selector to wait for on frontpage.

    Returns:
        List of Article objects.
    """
    all_articles: list[Article] = []

    for fp_url in frontpage_urls:
        source = urlparse(fp_url).netloc or fp_url
        if progress_callback:
            progress_callback(f"[FETCH] {source}")

        # Fetch frontpage HTML (dynamic or static)
        try:
            html = await fetch_html_smart(
                fp_url,
                wait_selector=wait_selector,
                dynamic=dynamic_fetch,
            )
        except Exception as e:
            if progress_callback:
                progress_callback(f"[ERROR] Failed to fetch {fp_url}: {e}")
            continue

        # Extract articles via LLM
        if progress_callback:
            progress_callback(f"[EXTRACT] Running LLM extraction: {source}")

        try:
            raw_articles = await extract_articles_from_html(html, fp_url, model)
        except Exception as e:
            if progress_callback:
                progress_callback(f"[ERROR] LLM extraction failed ({source}): {e}")
            continue

        # Filter by date
        filtered = filter_by_date(raw_articles, date_from, date_to)
        if progress_callback:
            progress_callback(
                f"[RESULT] {source}: {len(raw_articles)} found, {len(filtered)} in date range"
            )

        for a in filtered[:max_articles]:
            all_articles.append(Article(
                title=a["title"],
                url=a["url"],
                date=a.get("date"),
                source=source,
            ))

    # Crawl individual article content
    if crawl_content and all_articles:
        if progress_callback:
            progress_callback(f"\n[CRAWL] Extracting content from {len(all_articles)} articles...")

        semaphore = asyncio.Semaphore(max_concurrency)
        await asyncio.gather(*[
            _crawl_with_semaphore(article, semaphore, progress_callback, crawl_method)
            for article in all_articles
        ])

    return all_articles


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_articles_json(articles: list[Article], path: str = "articles.json") -> None:
    """Save articles to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)


def save_articles_markdown(articles: list[Article], path: str = "articles.md") -> None:
    """Save articles to Markdown file."""
    lines = [f"# Extracted Articles\n\nTotal: {len(articles)}\n"]
    for a in articles:
        lines.append(f"---\n## {a.title}")
        lines.append(f"- **Source**: {a.source}")
        lines.append(f"- **Date**: {a.date}")
        lines.append(f"- **URL**: {a.url}\n")
        if a.content_md:
            lines.append("### Content\n")
            lines.append(a.content_md)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    url = "https://www.anthropic.com/engineering"
    articles = asyncio.run(process_frontpages(
        [url],
        date(2020, 1, 1),
        date(2100, 1, 1),
        dynamic_fetch=True,
    ))
    save_articles_json(articles, "articles.json")
    save_articles_markdown(articles, "articles.md")
