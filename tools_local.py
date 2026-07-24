def _wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a summary."""
    import requests

    try:
        # Wikipedia API search
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if not data["query"]["search"]:
            return "No Wikipedia page found for this query."

        # Get the page extract
        page_title = data["query"]["search"][0]["title"]
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json"
        }
        response = requests.get(url, params=params, timeout=10)
        pages = response.json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
        return f"Wikipedia ({page_title}): {extract[:1000]}"
    except Exception as e:
        return f"Wikipedia search failed: {e}"

def _web_search(query: str, max_results: int = 3) -> str:
    """Perform a web search using DuckDuckGo."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No web search results found."

        output = []
        for r in results:
            output.append(f"- {r['title']}: {r['body'][:200]}...")
        return "\n".join(output)
    except Exception as e:
        return f"Web search failed: {e}"
