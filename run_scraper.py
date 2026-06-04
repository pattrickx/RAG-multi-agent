"""
Run web scraper for all sources in data/sources.json.
Saves results to data/articles/
"""

import asyncio
import json
import os
import sys
import re
from datetime import date
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env explicitly
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)
print(f"[ENV] Loaded .env from {env_path}", flush=True)
print(f"[ENV] OPENROUTER_API_KEY1: {os.environ.get('OPENROUTER_API_KEY1', 'NOT FOUND')[:10]}...", flush=True)

# Initialize balancer before scraping
from services.utils.key_balancer import setup_balancer
try:
    balancer = setup_balancer()
    print(f"[BALANCER] Loaded {len(balancer._keys)} keys", flush=True)
except Exception as e:
    print(f"[BALANCER] Error: {e}", flush=True)
    sys.exit(1)

from services.data_loader.web_scraper import (
    process_frontpages, save_articles_json, save_articles_markdown,
    extract_articles_from_html, fetch_html_smart, crawl_article_content
)


def load_sources(path: str = "data/sources.json") -> list[dict]:
    full_path = PROJECT_ROOT / path
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def progress(msg: str):
    print(msg, flush=True)


async def process_single_source(url: str, output_dir: Path, max_articles: int = 30):
    """Process a single source URL."""
    source_name = url.split("//")[1].split("/")[0]
    progress(f"\n[SOURCE] {source_name}")
    
    try:
        # Fetch frontpage
        progress(f"  [FETCH] {source_name}")
        html = await fetch_html_smart(url, dynamic=True)
        progress(f"  [FETCH] Got {len(html)} chars")
        
        # Extract articles
        progress(f"  [EXTRACT] Running LLM extraction")
        articles = await extract_articles_from_html(html, url)
        progress(f"  [EXTRACT] Found {len(articles)} articles")
        
        if not articles:
            progress(f"  [SKIP] No articles found for {source_name}")
            return []
        
        # Crawl content for each article
        progress(f"  [CRAWL] Extracting content from {len(articles)} articles")
        for i, article in enumerate(articles[:max_articles]):
            progress(f"  [CRAWL] {i+1}/{min(len(articles), max_articles)}: {article['title'][:50]}...")
            try:
                content = await crawl_article_content(article["url"], method="playwright")
                article["content"] = content
            except Exception as e:
                progress(f"  [CRAWL ERROR] {e}")
                article["content"] = ""
        
        # Filter articles with content
        articles_with_content = [a for a in articles if a.get("content") and not a["content"].startswith("[")]
        progress(f"  [DONE] {len(articles_with_content)}/{len(articles)} articles with content")
        
        return articles_with_content
        
    except Exception as e:
        progress(f"  [ERROR] {source_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


async def main():
    sources = load_sources()
    output_dir = PROJECT_ROOT / "data" / "articles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group sources by category
    categories: dict[str, list[dict]] = {}
    for src in sources:
        cat = src["category"]
        categories.setdefault(cat, []).append(src)
    
    total_articles = 0
    total_failed = 0
    all_results: dict[str, list[dict]] = {}
    
    for category, src_list in categories.items():
        safe_name = category.lower().replace(" & ", "_").replace(" ", "_")
        progress(f"\n{'='*60}")
        progress(f"[CATEGORY] {category} ({len(src_list)} sources)")
        progress(f"{'='*60}")
        
        category_articles = []
        
        # Process sources concurrently (max 3 at a time)
        semaphore = asyncio.Semaphore(3)
        
        async def process_with_limit(src):
            async with semaphore:
                return await process_single_source(src["url"], output_dir)
        
        results = await asyncio.gather(*[process_with_limit(src) for src in src_list])
        
        for articles in results:
            category_articles.extend(articles)
        
        # Save per category
        if category_articles:
            json_path = output_dir / f"{safe_name}.json"
            md_path = output_dir / f"{safe_name}.md"
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(category_articles, f, ensure_ascii=False, indent=2)
            
            # Generate markdown
            lines = [f"# {category}\n\nTotal: {len(category_articles)}\n"]
            for a in category_articles:
                lines.append(f"---\n## {a['title']}")
                lines.append(f"- **Source**: {a.get('source', '')}")
                lines.append(f"- **Date**: {a.get('date', '')}")
                lines.append(f"- **URL**: {a['url']}\n")
                if a.get("content"):
                    lines.append(a["content"][:5000])  # Limit content length
                lines.append("")
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            
            total_articles += len(category_articles)
            progress(f"\n[SAVED] {len(category_articles)} articles -> {json_path}")
        else:
            total_failed += len(src_list)
    
    # Summary
    progress(f"\n{'='*60}")
    progress(f"[DONE] Total: {total_articles} articles saved")
    progress(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
