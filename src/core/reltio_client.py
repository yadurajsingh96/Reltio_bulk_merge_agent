"""
High-Performance Reltio API Client

Optimized for bulk operations with:
- Async concurrent requests using httpx
- Connection pooling
- Automatic token refresh
- Retry logic with exponential backoff
- Batch operations where supported
"""

import asyncio
import httpx
import time
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta
import base64

logger = logging.getLogger(__name__)


@dataclass
class ReltioConfig:
    """Reltio connection configuration"""
    client_id: str
    client_secret: str
    tenant_id: str
    environment: str = "dev"  # dev, test, prod, prod-usg
    auth_server: str = "https://auth.reltio.com"
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


class TokenManager:
    """Manages OAuth token lifecycle with automatic refresh"""

    def __init__(self, config: ReltioConfig):
        self.config = config
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def get_token(self, client: httpx.AsyncClient) -> str:
        """Get valid access token, refreshing if needed"""
        async with self._lock:
            if self._is_token_valid():
                return self._token

            await self._refresh_token(client)
            return self._token

    def _is_token_valid(self) -> bool:
        """Check if current token is valid with 5-minute buffer"""
        if not self._token or not self._token_expiry:
            return False
        return datetime.now() < self._token_expiry - timedelta(minutes=5)

    async def _refresh_token(self, client: httpx.AsyncClient) -> None:
        """Refresh OAuth token"""
        credentials = f"{self.config.client_id}:{self.config.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {"grant_type": "client_credentials"}

        response = await client.post(
            f"{self.config.auth_server}/oauth/token",
            headers=headers,
            data=data
        )
        response.raise_for_status()

        token_data = response.json()
        self._token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

        logger.info("OAuth token refreshed successfully")


class ReltioClient:
    """
    High-performance async Reltio API client

    Designed for bulk operations with:
    - Concurrent request execution
    - Automatic token management
    - Connection pooling
    - Retry logic
    """

    def __init__(self, config: ReltioConfig):
        self.config = config
        self.token_manager = TokenManager(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Build base URLs
        self.api_base_url = f"https://{config.environment}.reltio.com/reltio/api/{config.tenant_id}"

    async def __aenter__(self):
        """Async context manager entry"""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.request_timeout),
            limits=httpx.Limits(
                max_connections=self.config.max_concurrent_requests * 2,
                max_keepalive_connections=self.config.max_concurrent_requests
            )
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client:
            await self._client.aclose()

    async def _get_headers(self) -> Dict[str, str]:
        """Get request headers with current token"""
        token = await self.token_manager.get_token(self._client)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute request with retry logic"""
        headers = await self._get_headers()
        kwargs["headers"] = headers

        last_exception = None
        for attempt in range(self.config.retry_attempts):
            try:
                async with self._semaphore:
                    response = await self._client.request(method, url, **kwargs)

                    # Handle 401 - refresh token and retry
                    if response.status_code == 401 and attempt < self.config.retry_attempts - 1:
                        self.token_manager._token = None
                        headers = await self._get_headers()
                        kwargs["headers"] = headers
                        continue

                    response.raise_for_status()
                    return response.json() if response.content else {}

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500:
                    # Retry on server errors
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
            except httpx.RequestError as e:
                last_exception = e
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Request to {url} failed after {self.config.retry_attempts} attempts")

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    async def search_entities(
        self,
        filter_expr: str,
        entity_type: str = "HCP",
        max_results: int = 100,
        offset: int = 0,
        select: str = "uri,label,type,attributes",
        options: str = "ovOnly"
    ) -> Dict[str, Any]:
        """
        Search for entities using Reltio filter expressions

        Args:
            filter_expr: Reltio filter expression
            entity_type: Entity type (HCP, HCO, etc.)
            max_results: Maximum results to return (max 200 per call)
            offset: Pagination offset
            select: Fields to return
            options: Response options

        Returns:
            Search results with entities
        """
        url = f"{self.api_base_url}/entities/_search"

        # Add entity type to filter
        full_filter = f"({filter_expr}) AND equals(type,'configuration/entityTypes/{entity_type}')"

        payload = {
            "filter": full_filter,
            "max": min(max_results, 200),
            "offset": offset,
            "select": select,
            "options": options,
            "activeness": "active"
        }

        return await self._request_with_retry("POST", url, json=payload)

    async def search_by_attributes(
        self,
        attributes: Dict[str, Any],
        entity_type: str = "HCP",
        match_mode: str = "fuzzy",
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by attribute values with intelligent matching

        Args:
            attributes: Dictionary of attribute names and values to search
            entity_type: Entity type
            match_mode: 'exact', 'fuzzy', or 'contains'
            max_results: Maximum results

        Returns:
            List of matching entities
        """
        # Build filter expression based on attributes
        conditions = []

        for attr_name, attr_value in attributes.items():
            if attr_value is None or attr_value == "":
                continue

            # Clean the value
            value = str(attr_value).strip()
            if not value:
                continue

            # Build condition based on match mode
            attr_path = f"attributes.{attr_name}"

            if match_mode == "exact":
                conditions.append(f"equals({attr_path},'{value}')")
            elif match_mode == "fuzzy":
                conditions.append(f"fuzzy({attr_path},'{value}')")
            elif match_mode == "contains":
                conditions.append(f"containsWordStartingWith({attr_path},'{value}')")

        if not conditions:
            return []

        # Combine with OR for broader matching, then rank by relevance
        filter_expr = " OR ".join(conditions)

        result = await self.search_entities(
            filter_expr=filter_expr,
            entity_type=entity_type,
            max_results=max_results
        )

        return result if isinstance(result, list) else []

    # =========================================================================
    # ENTITY OPERATIONS
    # =========================================================================

    async def get_entity(
        self,
        entity_id: str,
        options: str = "ovOnly"
    ) -> Dict[str, Any]:
        """Get entity by ID"""
        # Extract ID if full URI provided
        if entity_id.startswith("entities/"):
            entity_id = entity_id.split("/")[-1]

        url = f"{self.api_base_url}/entities/{entity_id}"
        params = {"options": options}

        return await self._request_with_retry("GET", url, params=params)

    async def get_entities_batch(
        self,
        entity_ids: List[str],
        options: str = "ovOnly"
    ) -> List[Dict[str, Any]]:
        """Get multiple entities concurrently"""
        tasks = [self.get_entity(eid, options) for eid in entity_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        return [r for r in results if isinstance(r, dict)]

    # =========================================================================
    # MATCH OPERATIONS
    # =========================================================================

    async def get_entity_matches(
        self,
        entity_id: str,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Get potential matches for an entity

        Args:
            entity_id: Entity ID to find matches for
            max_results: Maximum matches to return

        Returns:
            Match results with match groups and scores
        """
        if entity_id.startswith("entities/"):
            entity_id = entity_id.split("/")[-1]

        url = f"{self.api_base_url}/entities/{entity_id}/_matches"
        params = {
            "max": max_results,
            "markMatchedValues": "true"
        }

        return await self._request_with_retry("GET", url, params=params)

    async def get_transitive_matches(
        self,
        entity_id: str,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """Get transitive matches with detailed comparison"""
        if entity_id.startswith("entities/"):
            entity_id = entity_id.split("/")[-1]

        url = f"{self.api_base_url}/entities/{entity_id}/_transitiveMatches"
        params = {
            "deep": 1,
            "markMatchedValues": "true",
            "sort": "score",
            "order": "desc",
            "limit": max_results
        }

        return await self._request_with_retry("GET", url, params=params)

    async def find_matches_for_attributes(
        self,
        attributes: Dict[str, Any],
        entity_type: str = "HCP"
    ) -> List[Dict[str, Any]]:
        """
        Find potential matches in Reltio based on input attributes

        This performs a smart search to find entities that could be matches
        for the given attribute values.
        """
        # Strategy 1: Exact match on unique identifiers
        matches = []

        # Check for NPI (most reliable identifier)
        if "NPI" in attributes and attributes["NPI"]:
            npi_matches = await self.search_entities(
                filter_expr=f"equals(attributes.NPI,'{attributes['NPI']}')",
                entity_type=entity_type,
                max_results=10
            )
            if npi_matches:
                matches.extend(npi_matches)

        # Check for name-based matching
        name_parts = []
        for name_field in ["FirstName", "LastName", "Name"]:
            if name_field in attributes and attributes[name_field]:
                name_parts.append(attributes[name_field])

        if name_parts and not matches:
            # Build fuzzy name search
            name_conditions = [f"fuzzy(attributes,'{part}')" for part in name_parts]
            name_filter = " AND ".join(name_conditions)

            name_matches = await self.search_entities(
                filter_expr=name_filter,
                entity_type=entity_type,
                max_results=20
            )
            if name_matches:
                matches.extend(name_matches)

        # Deduplicate by URI
        seen_uris = set()
        unique_matches = []
        for match in matches:
            uri = match.get("uri", "")
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                unique_matches.append(match)

        return unique_matches

    # =========================================================================
    # MERGE OPERATIONS
    # =========================================================================

    async def merge_entities(
        self,
        winner_id: str,
        loser_id: str
    ) -> Dict[str, Any]:
        """
        Merge two entities

        Args:
            winner_id: Entity ID that will survive (winner)
            loser_id: Entity ID that will be merged into winner (loser)

        Returns:
            Merge result with merged entity
        """
        # Normalize IDs
        if not winner_id.startswith("entities/"):
            winner_id = f"entities/{winner_id}"
        if not loser_id.startswith("entities/"):
            loser_id = f"entities/{loser_id}"

        # Extract just the ID for URL
        winner_id_clean = winner_id.split("/")[-1]

        url = f"{self.api_base_url}/entities/{winner_id_clean}/_sameAs"
        params = {"uri": loser_id}

        return await self._request_with_retry("POST", url, params=params)

    async def merge_entities_batch(
        self,
        merge_pairs: List[Dict[str, str]],
        on_progress: callable = None
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple entity pairs concurrently

        Args:
            merge_pairs: List of {"winner_id": ..., "loser_id": ...}
            on_progress: Optional callback for progress updates

        Returns:
            List of merge results
        """
        results = []
        total = len(merge_pairs)

        # Process in batches to avoid overwhelming the API
        batch_size = self.config.max_concurrent_requests

        for i in range(0, total, batch_size):
            batch = merge_pairs[i:i + batch_size]

            tasks = [
                self.merge_entities(pair["winner_id"], pair["loser_id"])
                for pair in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                pair = batch[j]
                if isinstance(result, Exception):
                    results.append({
                        "winner_id": pair["winner_id"],
                        "loser_id": pair["loser_id"],
                        "status": "error",
                        "error": str(result)
                    })
                else:
                    results.append({
                        "winner_id": pair["winner_id"],
                        "loser_id": pair["loser_id"],
                        "status": "success",
                        "result": result
                    })

            # Progress callback
            if on_progress:
                on_progress(min(i + batch_size, total), total)

        return results

    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check API health and connectivity"""
        try:
            # Simple search to verify connectivity
            result = await self.search_entities(
                filter_expr="exists(uri)",
                max_results=1
            )
            return {"status": "healthy", "connected": True}
        except Exception as e:
            return {"status": "unhealthy", "connected": False, "error": str(e)}

    async def get_entity_type_config(self, entity_type: str = "HCP") -> Dict[str, Any]:
        """Get entity type configuration including match rules"""
        url = f"{self.api_base_url}/configuration/entityTypes/{entity_type}"
        return await self._request_with_retry("GET", url)
