"""
Core de extração do projeto.
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
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class ArticleItem(BaseModel):
    title: str
    url: str
    date: Optional[str] = Field(None, description="YYYY-MM-DD ou null")


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


EXTRACTION_SYSTEM = """
Você é um extrator de dados estruturados de HTML de sites de notícias.
Dado o HTML de uma frontpage, extraia TODOS os artigos listados.

Retorne APENAS um JSON válido com esta estrutura (sem markdown, sem explicação):
{
  "articles": [
    {
      "title": "Título do artigo",
      "url": "https://url-completa.com/artigo",
      "date": "YYYY-MM-DD"
    }
  ]
}

Regras:
- Se a URL for relativa (ex: /artigo/123), complete com o domínio base fornecido.
- Se a data não estiver explícita, use null.
- Datas em português (ex: "20 mai 2025", "há 2 horas") devem ser convertidas para YYYY-MM-DD.
- Ignore anúncios, links de navegação e rodapé — foque em artigos/notícias.
- Retorne no máximo 50 artigos por frontpage.
""".strip()


def build_llm(model: str = "openrouter/owl-alpha") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )


async def fetch_html(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


async def extract_articles(html: str, base_url: str, model: str = "gpt-4o-mini") -> list[dict[str, Any]]:
    html_truncated = html[:80_000]
    llm = build_llm(model)
    structured_llm = llm.with_structured_output(ArticleList)
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM),
        HumanMessage(content=f"Base URL: {base_url}\n\nHTML da frontpage:\n\n{html_truncated}"),
    ]
    result: ArticleList = await structured_llm.ainvoke(messages)

    output: list[dict[str, Any]] = []
    for article in result.articles:
        item = article.model_dump()
        item["url"] = urljoin(base_url, item["url"])
        output.append(item)
    return output


def parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def filter_by_date(articles: list[dict[str, Any]], date_from: date, date_to: date) -> list[dict[str, Any]]:
    return [a for a in articles if (d := parse_date(a.get("date"))) and date_from <= d <= date_to]


async def crawl_article_content(url: str) -> str:
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
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.6),
                options={"ignore_links": True},
            ),
            cache_mode=CacheMode.ENABLED,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            if not result.success:
                return f"[Erro ao extrair (HTTP {result.status_code}): {result.error_message}]"
            return result.markdown.fit_markdown or result.markdown.raw_markdown or ""
    except ImportError:
        return "[crawl4ai não instalado — rode: pip install crawl4ai && crawl4ai-setup]"
    except Exception as e:
        return f"[Erro: {e}]"


async def _crawl_with_semaphore(article: Article, semaphore: asyncio.Semaphore, progress_callback=None) -> None:
    async with semaphore:
        if progress_callback:
            progress_callback(f"  ↳ {article.title[:60]}...")
        article.content_md = await crawl_article_content(article.url)


async def process_frontpages(
    frontpage_urls: list[str],
    date_from: date,
    date_to: date,
    model: str = "gpt-4o-mini",
    crawl_content: bool = True,
    max_articles: int = 20,
    progress_callback=None,
    max_concurrency: int = 5,
) -> list[Article]:
    all_articles: list[Article] = []

    for fp_url in frontpage_urls:
        source = urlparse(fp_url).netloc or fp_url
        if progress_callback:
            progress_callback(f"📥 Buscando frontpage: {source}")

        try:
            html = await fetch_html(fp_url)
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Erro ao buscar {fp_url}: {e}")
            continue

        if progress_callback:
            progress_callback(f"🤖 Extraindo artigos com LLM ({model}): {source}")

        try:
            raw_articles = await extract_articles(html, fp_url, model)
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Erro LLM ({source}): {e}")
            continue

        filtered = filter_by_date(raw_articles, date_from, date_to)
        if progress_callback:
            progress_callback(f"✅ {source}: {len(raw_articles)} artigos encontrados, {len(filtered)} no intervalo de datas")

        for a in filtered[:max_articles]:
            all_articles.append(
                Article(
                    title=a["title"],
                    url=a["url"],
                    date=a.get("date"),
                    source=source,
                )
            )

    if crawl_content and all_articles:
        if progress_callback:
            progress_callback(f"\n🕷️ Extraindo conteúdo de {len(all_articles)} artigos...")

        semaphore = asyncio.Semaphore(max_concurrency)
        await asyncio.gather(*[_crawl_with_semaphore(article, semaphore, progress_callback) for article in all_articles])

    return all_articles


def save_articles_json(articles: list[Article], path: str = "articles.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)


def save_articles_markdown(articles: list[Article], path: str = "articles.md") -> None:
    lines = [f"# Artigos Extraídos\n\nTotal: {len(articles)}\n"]
    for a in articles:
        lines.append(f"---\n## {a.title}")
        lines.append(f"- **Fonte**: {a.source}")
        lines.append(f"- **Data**: {a.date}")
        lines.append(f"- **URL**: {a.url}\n")
        if a.content_md:
            lines.append("### Conteúdo\n")
            lines.append(a.content_md)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
if __name__ == "__main__":
    url = "https://www.anthropic.com/engineering"
    articles = asyncio.run(process_frontpages([url], date(2020, 1, 1), date(2100, 1, 1)))
    save_articles_json(articles, "articles.json")
    save_articles_markdown(articles, "articles.md")