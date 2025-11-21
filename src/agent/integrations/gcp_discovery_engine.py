from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1beta as discoveryengine

from src.agent.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    id: str
    score: float
    text: Optional[str]
    metadata: Dict[str, Any]


class GCPDiscoveryEngine:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        data_store_id: Optional[str] = None,
    ) -> None:
        """
        Initialize GCP Discovery Engine Datastore client.

        Args:
            project_id: GCP project ID (defaults to settings.gcp_project_id)
            location: GCP location (defaults to settings.gcp_location or "us")
            data_store_id: Datastore ID (defaults to
                settings.gcp_data_store_id)
        """
        self.project_id = project_id or settings.gcp_project_id
        self.location = location or getattr(settings, "gcp_location", "us")
        self.data_store_id = data_store_id or getattr(
            settings, "gcp_data_store_id", None
        )

        logger.debug(f"Initializing GCP Discovery Engine with project_id={self.project_id}, location={self.location}, data_store_id={self.data_store_id}")

        if not self.project_id:
            logger.error("GCP project ID is missing")
            raise ValueError(
                "GCP project ID is required. Set gcp_project_id in settings "
                "or pass project_id parameter."
            )

        if not self.data_store_id:
            logger.error("GCP data store ID is missing")
            raise ValueError(
                "GCP data store ID is required. Set gcp_data_store_id in "
                "settings or pass data_store_id parameter."
            )

        self._client = None
        logger.debug("GCP Discovery Engine initialization complete")

    def _get_client(self) -> discoveryengine.SearchServiceClient:
        """Get or create Discovery Engine search client."""
        if self._client is None:
            try:
                # Set up credentials if service account JSON is provided
                credentials = None
                if settings.gcp_service_account_json:
                    import os
                    from google.oauth2 import service_account

                    # Set environment variable for ADC
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                        settings.gcp_service_account_json
                    )

                    # Also load credentials explicitly
                    credentials = service_account.Credentials.from_service_account_file(
                        settings.gcp_service_account_json
                    )
                    logger.info(
                        f"Using service account from: "
                        f"{settings.gcp_service_account_json}"
                    )

                client_options = (
                    ClientOptions(
                        api_endpoint=(f"{self.location}-discoveryengine.googleapis.com")
                    )
                    if self.location != "global"
                    else None
                )

                self._client = discoveryengine.SearchServiceClient(
                    credentials=credentials, client_options=client_options
                )

                logger.info(
                    f"Discovery Engine client initialized for project "
                    f"{self.project_id} in location {self.location}"
                )

            except ImportError as e:
                raise RuntimeError(
                    "google-cloud-discoveryengine is required. "
                    "Install it with: pip install google-cloud-discoveryengine"
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize Discovery Engine client: {e}")
                raise RuntimeError(
                    f"Failed to initialize Discovery Engine client: {e}"
                ) from e

        return self._client

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        data_store_id: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Query the GCP Discovery Engine Datastore.

        Args:
            query_text: The text query to search for
            top_k: Number of results to return (default: 7)
            score_threshold: Minimum relevance score threshold (not used)
            data_store_id: Datastore ID to query (defaults to
                self.data_store_id)

        Returns:
            List of VectorSearchResult objects
        """
        # Get client
        client = self._get_client()

        # Use provided datastore or default
        data_store_id = data_store_id or self.data_store_id

        # Set defaults
        page_size = top_k or getattr(settings, "gcp_vector_top_k", 7)

        # Ensure correct type
        try:
            page_size = int(page_size)
        except Exception:
            logger.warning(f"Invalid page_size value: {page_size}, defaulting to 7")
            page_size = 7

        # Build serving config path
        serving_config = (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"collections/default_collection/dataStores/{data_store_id}/"
            f"servingConfigs/default_config"
        )

        logger.info(f"Querying Discovery Engine datastore: {data_store_id}")
        logger.info(f"Query: {query_text}")
        logger.info(f"Page size: {page_size}")

        try:
            # Configure content search spec for chunk-based retrieval
            content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
                snippet_spec=(
                    discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                        return_snippet=False
                    )
                ),
                search_result_mode=(
                    discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS
                ),
            )

            # Create search request
            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query_text,
                page_size=page_size,
                content_search_spec=content_search_spec,
            )

            # Execute search
            response = client.search(request=request)

            # Process results
            results: List[VectorSearchResult] = []

            for idx, result in enumerate(response.results):
                # Extract chunk information
                chunk = result.chunk if hasattr(result, "chunk") else None
                document = result.document if hasattr(result, "document") else None

                # Get text content from chunk
                text = ""
                if chunk and hasattr(chunk, "content"):
                    # Chunk content is raw text, not JSON
                    text = chunk.content
                    logger.debug(
                        f"Chunk content type: {type(chunk.content)}, length: {len(chunk.content) if chunk.content else 0}"
                    )
                    logger.debug(f"Processing chunk content: {chunk.content}")
                elif document and hasattr(document, "derived_struct_data"):
                    logger.debug("Falling back to document content")
                    # Fallback to document content if available
                    struct_data = document.derived_struct_data
                    if struct_data and "snippets" in struct_data:
                        snippets = struct_data["snippets"]
                        if snippets and len(snippets) > 0:
                            text = snippets[0].get("snippet", "")

                # Build metadata
                metadata: Dict[str, Any] = {
                    "data_store_id": data_store_id,
                }

                # Add document metadata if available
                if document:
                    if hasattr(document, "id"):
                        metadata["document_id"] = document.id
                    if hasattr(document, "name"):
                        metadata["document_name"] = document.name
                    if hasattr(document, "derived_struct_data"):
                        struct_data = document.derived_struct_data
                        if struct_data:
                            # Extract common product fields
                            for field in [
                                "name",
                                "title",
                                "price",
                                "model",
                                "sku",
                                "url",
                                "brand",
                                "category",
                            ]:
                                if field in struct_data:
                                    metadata[field] = struct_data[field]

                # Add chunk metadata if available
                if chunk:
                    if hasattr(chunk, "id"):
                        metadata["chunk_id"] = chunk.id
                    if hasattr(chunk, "document_metadata"):
                        chunk_meta = chunk.document_metadata
                        if hasattr(chunk_meta, "uri"):
                            metadata["source_uri"] = chunk_meta.uri

                # Generate result ID
                result_id = (
                    metadata.get("chunk_id")
                    or metadata.get("document_id")
                    or f"result_{idx}"
                )

                # Score is typically not provided in Discovery Engine results
                # Use inverse rank as a proxy
                score = 1.0 - (idx * 0.05)

                results.append(
                    VectorSearchResult(
                        id=result_id, score=score, text=text, metadata=metadata
                    )
                )

            logger.info(f"Returning {len(results)} results")
            return results

        except Exception as e:
            import traceback

            error_msg = str(e)
            detailed_trace = traceback.format_exc()
            logger.error(f"Discovery Engine query failed: {error_msg}")
            logger.debug(f"Stack trace:\n{detailed_trace}")
            raise RuntimeError(
                f"Discovery Engine datastore query failed: {error_msg}"
            ) from e
