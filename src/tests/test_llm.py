"""
Unit tests for LLM provider.
"""

import pytest
from unittest.mock import patch
from src.agent.utils.llm_provider import get_llm_model


def test_get_llm_model_openai():
    """Test getting OpenAI LLM model."""
    with patch("src.agent.utils.llm_provider.OPENAI_API_KEY", "test-key"):
        model = get_llm_model("openai", "gpt-4o", temperature=0.5)
        assert model is not None
        assert model.model_name == "gpt-4o"
        assert model.temperature == 0.5


def test_get_llm_model_unknown_provider():
    """Test that unknown provider raises error."""
    with pytest.raises(ValueError, match="Unknown LLM model provider"):
        get_llm_model("unknown", "model")
