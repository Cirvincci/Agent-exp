"""
Scientific Web Searcher Module for SciDetective_Agent

This module handles searching and retrieving information from scientific databases:
- arXiv integration for preprints
- Semantic Scholar API for academic papers
- CrossRef for DOI resolution
- PubMed integration for biomedical literature
- Web scraping for latest research trends
"""

import asyncio
import aiohttp
import requests
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import logging
import arxiv
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Represents a scientific paper"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    pdf_url: Optional[str]
    publication_date: Optional[str]
    journal: Optional[str]
    doi: Optional[str]
    citation_count: Optional[int]
    keywords: List[str]
    source: str  # 'arxiv', 'semantic_scholar', 'pubmed', etc.

@dataclass
class SearchResult:
    """Represents search results from scientific databases"""
    query: str
    papers: List[Paper]
    total_results: int
    search_time: float
    source: str

class ScientificWebSearcher:
    """
    Handles searching across multiple scientific databases and sources
    """

    def __init__(self):
        """Initialize the web searcher with API configurations"""
        self.session = None

        # API endpoints
        self.endpoints = {
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1',
            'crossref': 'https://api.crossref.org/works',
            'pubmed': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
            'arxiv': 'http://export.arxiv.org/api/query'
        }

        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = {
            'semantic_scholar': 1.0,  # 1 request per second
            'crossref': 0.1,  # 10 requests per second
            'pubmed': 0.34,  # 3 requests per second
            'arxiv': 3.0  # 1 request per 3 seconds
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _rate_limit(self, source: str):
        """Enforce rate limiting for API calls"""
        now = time.time()
        last_time = self.last_request_time.get(source, 0)
        min_interval = self.min_request_interval.get(source, 1.0)

        if now - last_time < min_interval:
            sleep_time = min_interval - (now - last_time)
            time.sleep(sleep_time)

        self.last_request_time[source] = time.time()

    async def search_arxiv(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search arXiv for preprints

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            SearchResult object with papers from arXiv
        """
        start_time = time.time()
        self._rate_limit('arxiv')

        try:
            # Use arxiv library for robust searching
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in client.results(search):
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    publication_date=result.published.strftime('%Y-%m-%d') if result.published else None,
                    journal='arXiv',
                    doi=result.doi,
                    citation_count=None,  # arXiv doesn't provide citation counts
                    keywords=result.categories,
                    source='arxiv'
                )
                papers.append(paper)

            search_time = time.time() - start_time
            logger.info(f"arXiv search completed in {search_time:.2f}s, found {len(papers)} papers")

            return SearchResult(
                query=query,
                papers=papers,
                total_results=len(papers),
                search_time=search_time,
                source='arxiv'
            )

        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            return SearchResult(query, [], 0, time.time() - start_time, 'arxiv')

    async def search_semantic_scholar(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search Semantic Scholar for academic papers

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            SearchResult object with papers from Semantic Scholar
        """
        start_time = time.time()
        self._rate_limit('semantic_scholar')

        try:
            url = f"{self.endpoints['semantic_scholar']}/paper/search"
            params = {
                'query': query,
                'limit': min(max_results, 100),  # API limit
                'fields': 'title,authors,abstract,url,venue,year,citationCount,publicationDate,externalIds'
            }

            headers = {
                'User-Agent': 'SciDetective-Agent/1.0'
            }

            if self.session:
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return SearchResult(query, [], 0, time.time() - start_time, 'semantic_scholar')
            else:
                # Fallback to synchronous request
                response = requests.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                else:
                    logger.error(f"Semantic Scholar API error: {response.status_code}")
                    return SearchResult(query, [], 0, time.time() - start_time, 'semantic_scholar')

            papers = []
            for item in data.get('data', []):
                # Parse authors
                authors = []
                if item.get('authors'):
                    authors = [author.get('name', 'Unknown') for author in item['authors']]

                # Get DOI from external IDs
                doi = None
                if item.get('externalIds'):
                    doi = item['externalIds'].get('DOI')

                paper = Paper(
                    title=item.get('title', 'No title'),
                    authors=authors,
                    abstract=item.get('abstract', 'No abstract available'),
                    url=item.get('url', ''),
                    pdf_url=None,  # Semantic Scholar doesn't directly provide PDF URLs
                    publication_date=item.get('publicationDate'),
                    journal=item.get('venue', 'Unknown'),
                    doi=doi,
                    citation_count=item.get('citationCount'),
                    keywords=[],  # Not provided by Semantic Scholar search
                    source='semantic_scholar'
                )
                papers.append(paper)

            search_time = time.time() - start_time
            logger.info(f"Semantic Scholar search completed in {search_time:.2f}s, found {len(papers)} papers")

            return SearchResult(
                query=query,
                papers=papers,
                total_results=data.get('total', len(papers)),
                search_time=search_time,
                source='semantic_scholar'
            )

        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return SearchResult(query, [], 0, time.time() - start_time, 'semantic_scholar')

    async def search_pubmed(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Search PubMed for biomedical literature

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            SearchResult object with papers from PubMed
        """
        start_time = time.time()
        self._rate_limit('pubmed')

        try:
            # First, search for PMIDs
            search_url = f"{self.endpoints['pubmed']}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json'
            }

            if self.session:
                async with self.session.get(search_url, params=search_params) as response:
                    search_data = await response.json()
            else:
                response = requests.get(search_url, params=search_params)
                search_data = response.json()

            pmids = search_data.get('esearchresult', {}).get('idlist', [])

            if not pmids:
                return SearchResult(query, [], 0, time.time() - start_time, 'pubmed')

            # Fetch detailed information for each PMID
            fetch_url = f"{self.endpoints['pubmed']}/efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }

            if self.session:
                async with self.session.get(fetch_url, params=fetch_params) as response:
                    xml_data = await response.text()
            else:
                response = requests.get(fetch_url, params=fetch_params)
                xml_data = response.text

            # Parse XML response
            papers = self._parse_pubmed_xml(xml_data)

            search_time = time.time() - start_time
            logger.info(f"PubMed search completed in {search_time:.2f}s, found {len(papers)} papers")

            return SearchResult(
                query=query,
                papers=papers,
                total_results=int(search_data.get('esearchresult', {}).get('count', len(papers))),
                search_time=search_time,
                source='pubmed'
            )

        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return SearchResult(query, [], 0, time.time() - start_time, 'pubmed')

    def _parse_pubmed_xml(self, xml_data: str) -> List[Paper]:
        """Parse PubMed XML response to extract paper information"""
        papers = []

        try:
            root = ET.fromstring(xml_data)

            for article in root.findall('.//PubmedArticle'):
                # Extract title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else 'No title'

                # Extract authors
                authors = []
                author_list = article.find('.//AuthorList')
                if author_list is not None:
                    for author in author_list.findall('Author'):
                        last_name = author.find('LastName')
                        first_name = author.find('ForeName')
                        if last_name is not None and first_name is not None:
                            authors.append(f"{first_name.text} {last_name.text}")

                # Extract abstract
                abstract_elem = article.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else 'No abstract available'

                # Extract PMID
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ''

                # Extract journal
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else 'Unknown'

                # Extract publication date
                pub_date = None
                date_elem = article.find('.//PubDate')
                if date_elem is not None:
                    year = date_elem.find('Year')
                    month = date_elem.find('Month')
                    if year is not None:
                        pub_date = year.text
                        if month is not None:
                            pub_date += f"-{month.text}"

                # Extract DOI
                doi = None
                article_ids = article.findall('.//ArticleId')
                for aid in article_ids:
                    if aid.get('IdType') == 'doi':
                        doi = aid.text
                        break

                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else '',
                    pdf_url=None,
                    publication_date=pub_date,
                    journal=journal,
                    doi=doi,
                    citation_count=None,  # PubMed doesn't provide citation counts
                    keywords=[],
                    source='pubmed'
                )
                papers.append(paper)

        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {str(e)}")

        return papers

    async def search_multiple_sources(self, query: str, sources: List[str] = None,
                                    max_results_per_source: int = 5) -> List[SearchResult]:
        """
        Search multiple scientific databases simultaneously

        Args:
            query: Search query
            sources: List of sources to search (default: all available)
            max_results_per_source: Maximum results per source

        Returns:
            List of SearchResult objects from different sources
        """
        if sources is None:
            sources = ['arxiv', 'semantic_scholar', 'pubmed']

        logger.info(f"Searching {len(sources)} sources for query: {query}")

        # Create search tasks
        tasks = []
        if 'arxiv' in sources:
            tasks.append(self.search_arxiv(query, max_results_per_source))
        if 'semantic_scholar' in sources:
            tasks.append(self.search_semantic_scholar(query, max_results_per_source))
        if 'pubmed' in sources:
            tasks.append(self.search_pubmed(query, max_results_per_source))

        # Execute searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, SearchResult):
                valid_results.append(result)
            else:
                logger.error(f"Search task failed: {result}")

        logger.info(f"Completed searches across {len(valid_results)} sources")
        return valid_results

    def get_latest_papers(self, field: str, days_back: int = 30, max_results: int = 20) -> List[Paper]:
        """
        Get the latest papers in a specific field

        Args:
            field: Research field to search
            days_back: Number of days to look back
            max_results: Maximum number of results

        Returns:
            List of latest papers
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Construct query with date filter
        query = f"{field} AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"

        # Search arXiv (best for recent papers)
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            papers = []
            for result in client.results(search):
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    publication_date=result.published.strftime('%Y-%m-%d') if result.published else None,
                    journal='arXiv',
                    doi=result.doi,
                    citation_count=None,
                    keywords=result.categories,
                    source='arxiv'
                )
                papers.append(paper)

            logger.info(f"Found {len(papers)} recent papers in {field}")
            return papers

        except Exception as e:
            logger.error(f"Error getting latest papers: {str(e)}")
            return []

    def extract_related_keywords(self, papers: List[Paper]) -> List[str]:
        """
        Extract related keywords from a list of papers

        Args:
            papers: List of papers to analyze

        Returns:
            List of related keywords
        """
        keywords = []

        # Extract keywords from titles and abstracts
        text_corpus = []
        for paper in papers:
            text_corpus.append(paper.title.lower())
            text_corpus.append(paper.abstract.lower())

        # Simple keyword extraction (could be improved with NLP)
        common_terms = set()
        for text in text_corpus:
            words = text.split()
            for word in words:
                if len(word) > 4 and word.isalpha():  # Filter short words and non-alphabetic
                    common_terms.add(word)

        # Also include explicit keywords from papers
        for paper in papers:
            keywords.extend(paper.keywords)

        # Combine and deduplicate
        all_keywords = list(set(list(common_terms) + keywords))

        # Return top keywords (could implement ranking)
        return all_keywords[:50]  # Limit to top 50

    def search_literature(self, query: str, max_results: int = 20) -> Dict:
        """
        Main literature search method (synchronous wrapper)

        Args:
            query: Search query
            max_results: Maximum total results

        Returns:
            Dictionary with search results and metadata
        """
        async def async_search():
            async with ScientificWebSearcher() as searcher:
                results = await searcher.search_multiple_sources(
                    query,
                    max_results_per_source=max_results // 3
                )
                return results

        # Run async search
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(async_search())
        finally:
            if loop.is_running():
                pass  # Don't close the loop if it's already running
            else:
                loop.close()

        # Combine results from all sources
        all_papers = []
        total_results = 0
        sources_searched = []

        for result in results:
            all_papers.extend(result.papers)
            total_results += result.total_results
            sources_searched.append(result.source)

        # Remove duplicates based on title similarity
        unique_papers = self._remove_duplicate_papers(all_papers)

        # Sort by relevance (citation count if available, else by date)
        unique_papers.sort(key=lambda p: (p.citation_count or 0, p.publication_date or ''), reverse=True)

        # Extract related keywords
        related_keywords = self.extract_related_keywords(unique_papers[:10])

        return {
            'query': query,
            'papers': unique_papers[:max_results],
            'total_found': len(unique_papers),
            'sources_searched': sources_searched,
            'related_keywords': related_keywords,
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_sources': len(results),
                'successful_sources': len([r for r in results if r.papers])
            }
        }

    def _remove_duplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()

        for paper in papers:
            # Simple deduplication based on title
            title_normalized = paper.title.lower().strip()
            if title_normalized not in seen_titles:
                unique_papers.append(paper)
                seen_titles.add(title_normalized)

        return unique_papers

    def get_paper_details(self, doi: str) -> Optional[Paper]:
        """
        Get detailed information about a paper using its DOI

        Args:
            doi: Digital Object Identifier

        Returns:
            Paper object with detailed information
        """
        try:
            url = f"{self.endpoints['crossref']}/{doi}"
            headers = {'User-Agent': 'SciDetective-Agent/1.0'}

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                work = data.get('message', {})

                # Parse CrossRef response
                title = work.get('title', ['Unknown'])[0]
                authors = []
                for author in work.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    authors.append(f"{given} {family}".strip())

                # Get abstract if available
                abstract = work.get('abstract', 'No abstract available')

                # Get journal
                journal = work.get('container-title', ['Unknown'])[0] if work.get('container-title') else 'Unknown'

                # Get publication date
                pub_date = None
                if work.get('published-print'):
                    date_parts = work['published-print'].get('date-parts', [[]])[0]
                    if date_parts:
                        pub_date = '-'.join(map(str, date_parts))

                return Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=work.get('URL', ''),
                    pdf_url=None,
                    publication_date=pub_date,
                    journal=journal,
                    doi=doi,
                    citation_count=work.get('is-referenced-by-count'),
                    keywords=[],
                    source='crossref'
                )

        except Exception as e:
            logger.error(f"Error getting paper details for DOI {doi}: {str(e)}")

        return None