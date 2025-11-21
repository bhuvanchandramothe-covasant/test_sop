"""
Integration tests for SOP Agent Executor.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.agent.executor.sop_agent_executor import SOPAgentExecutor


@pytest.fixture
def mock_config():
    """Fixture for test configuration with multi-tenant structure."""
    return {
        "tenants": {
            "default": {
                "system_prompt": "Test system prompt",
                "retrieval_prompt": "Extract search terms: {conversation}",
                "rag_prompt": "Answer using: {conversation}\n{context}",
                "discovery_engine": {
                    "project_id": None,
                    "location": "us",
                    "data_store_id": "test-datastore",
                },
                "retrieval_config": {
                    "top_k": 7,
                    "score_threshold": 0.7,
                },
                "llm_config": {
                    "retrieval_model": "gpt-4o-mini",
                    "retrieval_temperature": 0.2,
                    "response_model": "gpt-4o",
                    "response_temperature": 0.7,
                },
            },
            "tenant_a": {
                "system_prompt": "Tenant A system prompt",
                "retrieval_prompt": "Extract search terms: {conversation}",
                "rag_prompt": "Answer using: {conversation}\n{context}",
                "discovery_engine": {
                    "project_id": None,
                    "location": "us",
                    "data_store_id": "tenant-a-datastore",
                },
                "retrieval_config": {
                    "top_k": 8,
                    "score_threshold": 0.6,
                },
                "llm_config": {
                    "retrieval_model": "gpt-4o-mini",
                    "retrieval_temperature": 0.2,
                    "response_model": "gpt-4o",
                    "response_temperature": 0.7,
                },
            },
        }
    }


@pytest.fixture
def mock_settings():
    """Fixture for mock settings."""
    with patch("src.agent.executor.sop_agent_executor.settings") as mock:
        mock.gcp_project_id = "test-project"
        mock.gcp_location = "us"
        mock.gcp_data_store_id = "test-datastore"
        mock.postgres_connection_string = None
        yield mock


class TestSOPAgentExecutor:
    """Test suite for SOP Agent Executor."""

    @patch("src.agent.executor.sop_agent_executor.GCPDiscoveryEngine")
    def test_initialization_with_config_dict(self, mock_discovery_engine, mock_config):
        """Test initialization with config dictionary."""
        executor = SOPAgentExecutor(config_dict=mock_config)
        assert executor.tenant_configs == mock_config

    @patch("src.agent.executor.sop_agent_executor.GCPDiscoveryEngine")
    def test_get_tenant_config(self, mock_discovery_engine, mock_config):
        """Test tenant configuration retrieval."""
        executor = SOPAgentExecutor(config_dict=mock_config)

        # Test getting default tenant config
        tenant_config = executor._get_tenant_config("default")
        assert tenant_config["system_prompt"] == "Test system prompt"

        # Test getting tenant_a config
        tenant_config_a = executor._get_tenant_config("tenant_a")
        assert tenant_config_a["system_prompt"] == "Tenant A system prompt"
        assert (
            tenant_config_a["discovery_engine"]["data_store_id"] == "tenant-a-datastore"
        )

        # Test non-existent tenant
        with pytest.raises(ValueError, match="Tenant 'nonexistent' not found"):
            executor._get_tenant_config("nonexistent")

    @patch("src.agent.executor.sop_agent_executor.GCPDiscoveryEngine")
    def test_extract_tenant_id_from_metadata(
        self, mock_discovery_engine, mock_config, mock_settings
    ):
        """Test tenant ID extraction from request metadata."""
        executor = SOPAgentExecutor(config_dict=mock_config)

        # Mock context with tenant_id in metadata
        mock_context = Mock()
        mock_context.task = Mock()
        mock_context.task.metadata = {"tenant_id": "tenant_a"}

        tenant_id = executor._extract_tenant_id(mock_context)
        assert tenant_id == "tenant_a"

    @patch("src.agent.executor.sop_agent_executor.GCPDiscoveryEngine")
    def test_extract_tenant_id_default(
        self, mock_discovery_engine, mock_config, mock_settings
    ):
        """Test tenant ID defaults to 'default' when not provided."""
        executor = SOPAgentExecutor(config_dict=mock_config)

        # Mock context without tenant_id
        mock_context = Mock()
        mock_context.task = None
        mock_context.request_data = {}

        tenant_id = executor._extract_tenant_id(mock_context)
        assert tenant_id == "default"

    @patch("src.agent.executor.sop_agent_executor.GCPDiscoveryEngine")
    @patch("src.agent.executor.sop_agent_executor.get_llm_model")
    @pytest.mark.asyncio
    async def test_execute_with_tenant(
        self, mock_get_llm, mock_discovery_engine_class, mock_config, mock_settings
    ):
        """Test full execution flow with tenant."""
        # Mock LLMs
        mock_retrieval_llm = Mock()
        mock_retrieval_response = Mock()
        mock_retrieval_response.content = "return policy"
        mock_retrieval_llm.invoke.return_value = mock_retrieval_response

        mock_response_llm = Mock()
        mock_response_response = Mock()
        mock_response_response.content = "The return policy is 30 days."
        mock_response_llm.invoke.return_value = mock_response_response

        mock_get_llm.side_effect = [mock_retrieval_llm, mock_response_llm]

        # Mock Discovery Engine
        mock_result = Mock()
        mock_result.id = "doc1"
        mock_result.text = "Return policy: 30 days"
        mock_result.score = 0.95
        mock_result.metadata = {"source_uri": "policy.pdf"}

        mock_discovery_engine = Mock()
        mock_discovery_engine.query.return_value = [mock_result]
        mock_discovery_engine_class.return_value = mock_discovery_engine

        executor = SOPAgentExecutor(config_dict=mock_config)

        # Mock context and event queue
        mock_context = Mock()
        mock_context.get_user_input.return_value = "What is the return policy?"
        mock_context.task = Mock()
        mock_context.task.metadata = {"tenant_id": "default"}
        mock_context.task.thread_id = "test_thread"

        mock_event_queue = AsyncMock()

        await executor.execute(mock_context, mock_event_queue)

        # Verify event was enqueued
        assert mock_event_queue.enqueue_event.called

    @patch("src.agent.executor.sop_agent_executor.GCPDiscoveryEngine")
    def test_update_config(self, mock_discovery_engine, mock_config, mock_settings):
        """Test configuration update."""
        executor = SOPAgentExecutor(config_dict=mock_config)

        new_config = {
            "tenants": {
                "tenant_c": {
                    "system_prompt": "Tenant C prompt",
                    "discovery_engine": {
                        "project_id": None,
                        "location": "us",
                        "data_store_id": "tenant-c-datastore",
                    },
                }
            }
        }

        executor.update_config(new_config)

        # Verify tenant_c was added
        tenant_c_config = executor._get_tenant_config("tenant_c")
        assert tenant_c_config["system_prompt"] == "Tenant C prompt"
