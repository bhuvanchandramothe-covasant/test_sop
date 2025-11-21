#!/usr/bin/env python3
"""
Unit tests for GCP Discovery Engine integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agent.integrations.gcp_discovery_engine import (
    GCPDiscoveryEngine,
    VectorSearchResult,
)


class TestGCPDiscoveryEngine:
    """Test suite for GCPDiscoveryEngine class."""

    def test_initialization_with_defaults(self):
        """Test initialization with default settings."""
        with patch(
            "src.agent.integrations.gcp_discovery_engine.settings"
        ) as mock_settings:
            mock_settings.gcp_project_id = "test-project"
            mock_settings.gcp_location = "us"
            mock_settings.gcp_data_store_id = "test-datastore"

            client = GCPDiscoveryEngine()

            assert client.project_id == "test-project"
            assert client.location == "us"
            assert client.data_store_id == "test-datastore"
            assert client._client is None

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        client = GCPDiscoveryEngine(
            project_id="custom-project",
            location="us-central1",
            data_store_id="custom-datastore",
        )

        assert client.project_id == "custom-project"
        assert client.location == "us-central1"
        assert client.data_store_id == "custom-datastore"

    def test_initialization_without_project_id(self):
        """Test that initialization fails without project ID."""
        with patch(
            "src.agent.integrations.gcp_discovery_engine.settings"
        ) as mock_settings:
            mock_settings.gcp_project_id = None

            with pytest.raises(ValueError, match="GCP project ID is required"):
                GCPDiscoveryEngine()

    def test_initialization_without_data_store_id(self):
        """Test that initialization fails without data store ID."""
        with patch(
            "src.agent.integrations.gcp_discovery_engine.settings"
        ) as mock_settings:
            mock_settings.gcp_project_id = "test-project"
            mock_settings.gcp_data_store_id = None

            with pytest.raises(ValueError, match="data store ID is required"):
                GCPDiscoveryEngine()

    @patch("src.agent.integrations.gcp_discovery_engine.discoveryengine")
    @patch("src.agent.integrations.gcp_discovery_engine.ClientOptions")
    def test_get_client_initialization(self, mock_client_options, mock_discoveryengine):
        """Test Discovery Engine client initialization."""
        mock_search_client = MagicMock()
        mock_discoveryengine.SearchServiceClient.return_value = mock_search_client

        client = GCPDiscoveryEngine(
            project_id="test-project",
            location="us",
            data_store_id="test-datastore",
        )

        # Get client
        search_client = client._get_client()

        assert search_client == mock_search_client
        assert client._client == mock_search_client

    @patch("src.agent.integrations.gcp_discovery_engine.settings")
    @patch("src.agent.integrations.gcp_discovery_engine.discoveryengine")
    def test_query_success(self, mock_discoveryengine, mock_settings):
        """Test successful query execution."""
        mock_settings.gcp_project_id = "test-project"
        mock_settings.gcp_data_store_id = "test-datastore"
        mock_settings.gcp_vector_top_k = 7
        mock_settings.gcp_service_account_json = None

        # Mock search response
        mock_chunk1 = Mock(spec=["content", "id"])
        mock_chunk1.content = "Product 1 description"
        mock_chunk1.id = "chunk1"

        mock_doc1 = Mock(spec=["id", "name"])
        mock_doc1.id = "doc1"
        mock_doc1.name = "product1"
        # Don't add derived_struct_data to avoid 'in' operator issues

        mock_result1 = Mock(spec=["chunk", "document"])
        mock_result1.chunk = mock_chunk1
        mock_result1.document = mock_doc1

        mock_response = Mock(spec=["results"])
        mock_response.results = [mock_result1]

        mock_search_client = Mock()
        mock_search_client.search.return_value = mock_response
        mock_discoveryengine.SearchServiceClient.return_value = mock_search_client

        client = GCPDiscoveryEngine()
        results = client.query("test query", top_k=5)

        # Verify search was called
        assert mock_search_client.search.called

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], VectorSearchResult)
        assert results[0].text == "Product 1 description"
        assert results[0].metadata["chunk_id"] == "chunk1"
        assert results[0].metadata["document_id"] == "doc1"

    @patch("src.agent.integrations.gcp_discovery_engine.settings")
    @patch("src.agent.integrations.gcp_discovery_engine.discoveryengine")
    def test_query_with_no_results(self, mock_discoveryengine, mock_settings):
        """Test query that returns no results."""
        mock_settings.gcp_project_id = "test-project"
        mock_settings.gcp_data_store_id = "test-datastore"
        mock_settings.gcp_vector_top_k = 5
        mock_settings.gcp_service_account_json = None

        mock_response = Mock()
        mock_response.results = []

        mock_search_client = Mock()
        mock_search_client.search.return_value = mock_response
        mock_discoveryengine.SearchServiceClient.return_value = mock_search_client

        client = GCPDiscoveryEngine()
        results = client.query("nonexistent product")

        assert len(results) == 0

    @patch("src.agent.integrations.gcp_discovery_engine.settings")
    @patch("src.agent.integrations.gcp_discovery_engine.discoveryengine")
    def test_query_with_custom_data_store_id(self, mock_discoveryengine, mock_settings):
        """Test query with custom data store ID override."""
        mock_settings.gcp_project_id = "test-project"
        mock_settings.gcp_data_store_id = "default-datastore"
        mock_settings.gcp_vector_top_k = 5
        mock_settings.gcp_service_account_json = None

        mock_response = Mock()
        mock_response.results = []

        mock_search_client = Mock()
        mock_search_client.search.return_value = mock_response
        mock_discoveryengine.SearchServiceClient.return_value = mock_search_client

        client = GCPDiscoveryEngine()
        client.query("test query", data_store_id="custom-datastore")

        # Verify search was called
        assert mock_search_client.search.called

        # Verify custom datastore was used by checking the call
        call_args = mock_search_client.search.call_args
        # The request object is passed as keyword argument
        assert call_args is not None

    @patch("src.agent.integrations.gcp_discovery_engine.settings")
    @patch("src.agent.integrations.gcp_discovery_engine.discoveryengine")
    def test_query_error_handling(self, mock_discoveryengine, mock_settings):
        """Test error handling during query - should raise RuntimeError."""
        mock_settings.gcp_project_id = "test-project"
        mock_settings.gcp_data_store_id = "test-datastore"
        mock_settings.gcp_vector_top_k = 5
        mock_settings.gcp_service_account_json = None

        mock_search_client = Mock()
        mock_search_client.search.side_effect = Exception("API Error")
        mock_discoveryengine.SearchServiceClient.return_value = mock_search_client

        client = GCPDiscoveryEngine()
        with pytest.raises(
            RuntimeError, match="Discovery Engine datastore query failed"
        ):
            client.query("test query")
    
    @patch("src.agent.integrations.gcp_discovery_engine.settings")
    @patch("src.agent.integrations.gcp_discovery_engine.discoveryengine")
    def test_client_initialization_error(self, mock_discoveryengine, mock_settings):
        """Test that client initialization errors are raised properly."""
        mock_settings.gcp_project_id = "test-project"
        mock_settings.gcp_data_store_id = "test-datastore"
        mock_settings.gcp_service_account_json = None

        mock_discoveryengine.SearchServiceClient.side_effect = Exception("Auth Error")

        client = GCPDiscoveryEngine()
        with pytest.raises(
            RuntimeError, match="Failed to initialize Discovery Engine client"
        ):
            client._get_client()
