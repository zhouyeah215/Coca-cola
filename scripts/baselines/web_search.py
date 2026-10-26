import os
import json
import requests
from urllib.parse import urlparse
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize


# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/124.0 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)


def _safe_attr(obj, key, default=None):
    """Support dict-like and object-like SDK return values."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class SearchPaymentRequiredError(RuntimeError):
    """Raised when a search provider reports HTTP 402 / payment required."""


def _is_http_402_error(exc) -> bool:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 402:
        return True
    if getattr(exc, "status_code", None) == 402:
        return True
    if getattr(exc, "status", None) == 402:
        return True
    return "402" in str(exc)



def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def _extract_text_direct(url: str) -> str:
    """Fetch and extract page text directly from the source URL."""
    response = session.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '')
    if 'pdf' in content_type:
        return extract_pdf_text(url)
    try:
        soup = BeautifulSoup(response.text, 'lxml')
    except Exception:
        print("lxml parser not found or failed, falling back to html.parser")
        soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def extract_text_from_url(
    url,
    use_jina=False,
    jina_api_key=None,
    snippet: Optional[str] = None,
    jina_proxies: Optional[dict] = None,
):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                **headers,
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            if jina_api_key:
                jina_headers['Authorization'] = f'Bearer {jina_api_key}'
            try:
                jina_response = session.get(
                    f'https://r.jina.ai/{url}',
                    headers=jina_headers,
                    timeout=30,
                    proxies=jina_proxies,
                )
                jina_response.raise_for_status()
                response_text = jina_response.text
                # Remove URLs
                pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                text = re.sub(pattern, "", response_text).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
            except requests.exceptions.RequestException as e:
                print(f"Jina fetch failed for {url}, falling back to direct fetch: {e}")
                text = _extract_text_direct(url)
        else:
            text = _extract_text_direct(url)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(
    urls,
    max_workers=32,
    use_jina=False,
    jina_api_key=None,
    snippets: Optional[dict] = None,
    jina_proxies: Optional[dict] = None,
):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(
                extract_text_from_url,
                url,
                use_jina,
                jina_api_key,
                snippets.get(url) if snippets else None,
                jina_proxies,
            ): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


def _search_with_tavily(query, subscription_key):
    """Tavily search wrapper returning a normalized dict payload."""
    if not subscription_key:
        print("Web search: missing Tavily API key.")
        return {}
    try:
        from tavily import TavilyClient
    except ImportError:
        print("Web search: install with `pip install tavily-python`")
        return {}

    try:
        client = TavilyClient(api_key=subscription_key)
        result = client.search(
            query=query,
            search_depth="advanced",
            max_results=20,
            include_raw_content=False,
            topic="general",
        )
        return result if isinstance(result, dict) else {}
    except Exception as e:
        if _is_http_402_error(e):
            raise SearchPaymentRequiredError("402ERROR") from e
        print(f"Error occurred during Tavily search request: {e}")
        return {}


def _search_with_exa(query, subscription_key):
    """Exa search wrapper returning a Tavily-like normalized payload."""
    if not subscription_key:
        print("Web search: missing Exa API key.")
        return {}
    try:
        from exa_py import Exa
    except ImportError:
        print("Web search: install with `pip install exa-py`")
        return {}

    try:
        exa = Exa(api_key=subscription_key)
        search_results = exa.search_and_contents(
            query,
            type="auto",
            num_results=20,
            text={"max_characters": 1200},
            highlights=True,
        )
        items = list(_safe_attr(search_results, "results", []) or [])

        normalized = {"query": query, "results": []}
        for item in items:
            url = _safe_attr(item, "url", "") or ""
            title = _safe_attr(item, "title", "") or ""
            score = _safe_attr(item, "score", None)
            text = _safe_attr(item, "text", None)
            highlights = _safe_attr(item, "highlights", None)
            if isinstance(highlights, list):
                highlight_text = "\n".join(
                    str(h).strip() for h in highlights if str(h).strip()
                )
            else:
                highlight_text = str(highlights).strip() if highlights else ""
            content = (str(text)[:1200] if text else "") or highlight_text or ""
            normalized["results"].append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "score": score,
                    "raw_content": None,
                }
            )
        return normalized
    except Exception as e:
        if _is_http_402_error(e):
            raise SearchPaymentRequiredError("402ERROR") from e
        print(f"Error occurred during Exa search request: {e}")
        return {}


def bing_web_search(
    query,
    subscription_key,
    endpoint,
    market='en-US',
    language='en',
    timeout=20,
    provider='tavily',
):
    """
    Web search for agent/RAG pipelines.

    Parameters ``endpoint``, ``market``, ``language`` are kept for call-site compatibility
    but are ignored; choose backend with ``provider``.

    Returns:
        dict: Tavily-style payload (``query``, ``results``, …) for :func:`extract_relevant_info`.
    """
    provider_norm = (provider or "tavily").strip().lower()
    if provider_norm == "tavily":
        return _search_with_tavily(query, subscription_key)
    if provider_norm == "exa":
        return _search_with_exa(query, subscription_key)
    print(f"Web search: unsupported provider {provider!r}. Use 'tavily' or 'exa'.")
    return {}


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, headers=headers, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(search_results):
    """
    Normalize search hits to a list of dicts (title, url, snippet, …).

    Supports **Tavily** responses (``results`` list) and legacy **Bing** JSON (``webPages.value``).
    """
    useful_info = []

    if not search_results:
        return useful_info

    # Tavily (primary)
    if "results" in search_results:
        for idx, result in enumerate(search_results["results"]):
            url = result.get("url", "") or ""
            try:
                site = urlparse(url).netloc
            except Exception:
                site = ""
            snippet = result.get("content") or result.get("raw_content") or ""
            useful_info.append(
                {
                    "id": idx + 1,
                    "title": result.get("title", ""),
                    "url": url,
                    "site_name": site,
                    "date": "",
                    "snippet": snippet,
                    "context": "",
                }
            )
        return useful_info

    # Legacy Bing Web Search API shape
    if "webPages" in search_results and "value" in search_results["webPages"]:
        for id, result in enumerate(search_results["webPages"]["value"]):
            info = {
                "id": id + 1,
                "title": result.get("name", ""),
                "url": result.get("url", ""),
                "site_name": result.get("siteName", ""),
                "date": result.get("datePublished", "").split("T")[0],
                "snippet": result.get("snippet", ""),
                "context": "",
            }
            useful_info.append(info)

    return useful_info


# ------------------------------------------------------------

if __name__ == "__main__":
    # Example usage (Tavily API key)
    query = "Structure of dimethyl fumarate"

    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")
    if not TAVILY_API_KEY or TAVILY_API_KEY.startswith("YOUR_"):
        raise ValueError("Set TAVILY_API_KEY or edit this script with your Tavily API key.")

    print("Performing Tavily search...")
    search_results = bing_web_search(query, TAVILY_API_KEY, "")
    
    print("Extracting relevant information from search results...")
    extracted_info = extract_relevant_info(search_results)

    print("Fetching and extracting context for each snippet...")
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(info['url'], use_jina=True)  # Get full webpage text
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info['snippet'])
            if success:
                info['context'] = context
            else:
                info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info['context'] = f"Failed to fetch full text: {full_text}"

    # print("Your Search Query:", query)
    # print("Final extracted information with context:")
    # print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
