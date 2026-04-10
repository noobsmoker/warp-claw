"""
Web Search Tool
DuckDuckGo/Brave search for Warp-Claw.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_tool import BaseTool, ToolResult, ToolRegistry


class WebSearch(BaseTool):
    """
    Search the web using DuckDuckGo instant API.
    """
    
    name = "web_search"
    description = "Search the web for information. Use for fact-checking, finding sources, and current information."
    category = "search"
    tags = ["search", "web", "duckduckgo", "research"]
    default_timeout = 10
    
    # Simple search via DuckDuckGo HTML scraping (no API key needed)
    DDG_URL = "https://html.duckduckgo.com/html/"
    
    async def execute(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> ToolResult:
        """
        Search the web.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            ToolResult with search results
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.DDG_URL,
                    data={"q": query, "b": ""},
                    timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                ) as resp:
                    html = await resp.text()
                    
            # Parse results
            results = self._parse_results(html, max_results)
            
            return ToolResult(
                success=True,
                result={
                    "query": query,
                    "results": results,
                    "count": len(results)
                },
                tokens_estimate=len(query) // 4 + sum(len(r.get("snippet", "")) for r in results) // 4
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                result=None,
                error="Search timeout"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )
    
    def _parse_results(self, html: str, max_results: int) -> List[Dict[str, str]]:
        """Parse DuckDuckGo HTML results."""
        results = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            for result in soup.select('.result')[:max_results]:
                try:
                    title_el = result.select_one('.result__title')
                    snippet_el = result.select_one('.result__snippet')
                    link_el = result.select_one('.result__url')
                    
                    if title_el:
                        title = title_el.get_text(strip=True)
                        link = title_el.get('href', '')
                        # Clean DuckDuckGo redirect
                        if link and 'uddg=' in link:
                            from urllib.parse import parse_qs, urlparse
                            try:
                                parsed = urlparse(link)
                                link = parse_qs(parsed.query).get('uddg', [''])[0]
                            except:
                                pass
                        
                        results.append({
                            "title": title,
                            "url": link or "",
                            "snippet": snippet_el.get_text(strip=True) if snippet_el else ""
                        })
                except:
                    continue
                    
        except ImportError:
            # Fallback: simple regex parsing
            import re
            matches = re.findall(
                r'<a class="result__a" href="([^"]*)"[^>]*>([^<]*)</a>',
                html
            )
            for url, title in matches[:max_results]:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": ""
                })
        
        return results
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


class WebFetch(BaseTool):
    """
    Fetch and extract content from URLs.
    """
    
    name = "web_fetch"
    description = "Fetch a URL and extract readable content."
    category = "search"
    tags = ["fetch", "web", "scrape"]
    default_timeout = 15
    
    async def execute(
        self,
        url: str,
        max_chars: int = 5000,
        **kwargs
    ) -> ToolResult:
        """
        Fetch a URL.
        
        Args:
            url: URL to fetch
            max_chars: Maximum characters to extract
            
        Returns:
            ToolResult with extracted content
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                ) as resp:
                    html = await resp.text()
            
            # Extract text
            content = self._extract_text(html, max_chars)
            
            return ToolResult(
                success=True,
                result={
                    "url": url,
                    "content": content,
                    "length": len(content)
                },
                tokens_estimate=len(content) // 4
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )
    
    def _extract_text(self, html: str, max_chars: int) -> str:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script/style
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:max_chars]
            
        except ImportError:
            # Fallback
            import re
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\s+', ' ', text)
            return text[:max_chars]
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to extract",
                        "default": 5000
                    }
                },
                "required": ["url"]
            }
        }


# Register tools
ToolRegistry.register(WebSearch())
ToolRegistry.register(WebFetch())