"""
Unit tests for OpenRouter Event Extractor Adapter
Tests adapter functionality with mocked OpenAI SDK responses
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.openrouter_adapter import OpenRouterEventExtractor
from src.core.config import OpenRouterConfig
from src.core.extractor_factory import ExtractorConfigurationError
from src.core.interfaces import EventRecord
from src.core.constants import DEFAULT_NO_DATE, DEFAULT_NO_CITATION


class TestOpenRouterEventExtractor:
    """Test suite for OpenRouter Event Extractor"""

    def test_initialization_success(self):
        """Test successful initialization with valid config"""
        config = OpenRouterConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3-haiku",
            timeout=30
        )

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            mock_client = Mock()
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)

            assert extractor.config == config
            assert extractor.available is True
            assert extractor._client == mock_client

            # Verify OpenAI client was initialized correctly
            MockOpenAI.assert_called_once_with(
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key",
                timeout=30
            )

    def test_initialization_missing_api_key(self):
        """Test initialization fails with missing API key"""
        config = OpenRouterConfig(
            api_key="",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3-haiku",
            timeout=30
        )

        with pytest.raises(ExtractorConfigurationError) as exc_info:
            OpenRouterEventExtractor(config)

        assert "OpenRouter API key is required" in str(exc_info.value)
        assert "OPENROUTER_API_KEY" in str(exc_info.value)

    def test_initialization_missing_openai_sdk(self):
        """Test initialization with missing OpenAI SDK"""
        config = OpenRouterConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3-haiku",
            timeout=30
        )

        with patch('src.core.openrouter_adapter.OpenAI', side_effect=ImportError("OpenAI SDK not installed")):
            extractor = OpenRouterEventExtractor(config)

            assert extractor.available is False
            assert extractor._client is None

    def test_is_available_true(self):
        """Test is_available returns True when properly configured"""
        config = OpenRouterConfig(api_key="test-key")

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            mock_client = Mock()
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)
            assert extractor.is_available() is True

    def test_is_available_false_no_openai_sdk(self):
        """Test is_available returns False when OpenAI SDK not available"""
        config = OpenRouterConfig(api_key="test-key")

        with patch('src.core.openrouter_adapter.OpenAI', side_effect=ImportError("OpenAI SDK not installed")):
            extractor = OpenRouterEventExtractor(config)
            assert extractor.is_available() is False

    def test_is_available_false_no_api_key(self):
        """Test is_available returns False when API key is missing"""
        config = OpenRouterConfig(api_key="")

        with pytest.raises(ExtractorConfigurationError):
            OpenRouterEventExtractor(config)

    def test_extract_events_success(self):
        """Test successful event extraction"""
        config = OpenRouterConfig(api_key="test-key")

        # Mock successful API response (OpenAI SDK format)
        events_json = json.dumps([
            {
                "event_particulars": "On January 15, 2024, the plaintiff filed a motion for summary judgment.",
                "citation": "Fed. R. Civ. P. 56",
                "date": "2024-01-15",
                "document_reference": ""
            },
            {
                "event_particulars": "Court scheduled hearing for March 10, 2024.",
                "citation": "",
                "date": "2024-03-10",
                "document_reference": ""
            }
        ])

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            # Setup mock OpenAI client
            mock_client = Mock()
            mock_completion = Mock()

            # Mock the response structure
            mock_message = Mock()
            mock_message.content = events_json
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_completion.choices = [mock_choice]

            # Mock usage stats
            mock_usage = Mock()
            mock_usage.prompt_tokens = 100
            mock_usage.completion_tokens = 50
            mock_usage.total_tokens = 150
            mock_completion.usage = mock_usage

            mock_client.chat.completions.create.return_value = mock_completion
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)

            metadata = {"document_name": "test_document.pdf"}
            text = "Legal document text with events..."

            events = extractor.extract_events(text, metadata)

            # Verify results
            assert len(events) == 2
            assert all(isinstance(event, EventRecord) for event in events)

            # Check first event
            assert events[0].number == 1
            assert events[0].date == "2024-01-15"
            assert "summary judgment" in events[0].event_particulars
            assert events[0].citation == "Fed. R. Civ. P. 56"
            assert events[0].document_reference == "test_document.pdf"
            assert events[0].attributes["provider"] == "openrouter"

            # Check second event
            assert events[1].number == 2
            assert events[1].date == "2024-03-10"
            assert "hearing" in events[1].event_particulars
            # Empty citation in JSON is kept as empty string (not normalized to DEFAULT_NO_CITATION)
            assert events[1].citation == ""
            assert events[1].document_reference == "test_document.pdf"

            # Verify API was called correctly
            mock_client.chat.completions.create.assert_called_once()

    def test_extract_events_api_failure(self):
        """Test handling of API failure"""
        config = OpenRouterConfig(api_key="test-key")

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            # Setup mock OpenAI client that raises exception
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)

            metadata = {"document_name": "test_document.pdf"}
            text = "Legal document text..."

            events = extractor.extract_events(text, metadata)

            # Should return fallback record
            # When API call fails, it returns None, then triggers "empty response" message
            assert len(events) == 1
            assert isinstance(events[0], EventRecord)
            assert events[0].attributes["fallback"] is True
            assert "OpenRouter API returned empty response" in events[0].attributes["reason"]

    def test_extract_events_empty_text(self):
        """Test handling of empty text input"""
        config = OpenRouterConfig(api_key="test-key")

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            mock_client = Mock()
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)

            metadata = {"document_name": "test_document.pdf"}
            text = ""

            events = extractor.extract_events(text, metadata)

            # Should return fallback record
            assert len(events) == 1
            assert events[0].attributes["fallback"] is True
            assert "No text content to process" in events[0].attributes["reason"]

    def test_extract_events_not_available(self):
        """Test handling when adapter is not available"""
        config = OpenRouterConfig(api_key="test-key")

        with patch('src.core.openrouter_adapter.OpenAI', side_effect=ImportError("OpenAI SDK not installed")):
            extractor = OpenRouterEventExtractor(config)

            metadata = {"document_name": "test_document.pdf"}
            text = "Legal document text..."

            events = extractor.extract_events(text, metadata)

            # Should return fallback record
            assert len(events) == 1
            assert events[0].attributes["fallback"] is True
            assert "not available" in events[0].attributes["reason"]

    def test_extract_events_invalid_json_response(self):
        """Test handling of invalid JSON response"""
        config = OpenRouterConfig(api_key="test-key")

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            # Setup mock OpenAI client with invalid JSON response
            mock_client = Mock()
            mock_completion = Mock()

            # Mock response with invalid JSON
            mock_message = Mock()
            mock_message.content = "invalid json content"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_completion.choices = [mock_choice]

            # Mock usage stats
            mock_usage = Mock()
            mock_usage.prompt_tokens = 100
            mock_usage.completion_tokens = 50
            mock_usage.total_tokens = 150
            mock_completion.usage = mock_usage

            mock_client.chat.completions.create.return_value = mock_completion
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)

            metadata = {"document_name": "test_document.pdf"}
            text = "Legal document text..."

            events = extractor.extract_events(text, metadata)

            # Should return fallback record for invalid JSON (uses fallback parser)
            assert len(events) == 1
            # The fallback parser creates an event from the text, not necessarily with fallback=True
            assert isinstance(events[0], EventRecord)

    def test_extract_events_empty_response(self):
        """Test handling of empty API response"""
        config = OpenRouterConfig(api_key="test-key")

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            # Setup mock OpenAI client with empty response
            mock_client = Mock()
            mock_completion = Mock()

            # Mock empty choices
            mock_completion.choices = []

            # Mock usage stats
            mock_usage = Mock()
            mock_usage.prompt_tokens = 100
            mock_usage.completion_tokens = 0
            mock_usage.total_tokens = 100
            mock_completion.usage = mock_usage

            mock_client.chat.completions.create.return_value = mock_completion
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)

            metadata = {"document_name": "test_document.pdf"}
            text = "Legal document text..."

            events = extractor.extract_events(text, metadata)

            # Should return fallback record
            assert len(events) == 1
            assert events[0].attributes["fallback"] is True

    def test_call_openrouter_api_request_format(self):
        """Test that API requests are formatted correctly"""
        config = OpenRouterConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3-haiku",
            timeout=30
        )

        with patch('src.core.openrouter_adapter.OpenAI') as MockOpenAI:
            # Setup mock OpenAI client
            mock_client = Mock()
            mock_completion = Mock()

            # Mock successful response
            mock_message = Mock()
            mock_message.content = "[]"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_completion.choices = [mock_choice]

            mock_usage = Mock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 5
            mock_usage.total_tokens = 15
            mock_completion.usage = mock_usage

            mock_client.chat.completions.create.return_value = mock_completion
            MockOpenAI.return_value = mock_client

            extractor = OpenRouterEventExtractor(config)
            extractor._call_openrouter_api("test text")

            # Verify OpenAI client was initialized with correct parameters
            MockOpenAI.assert_called_once_with(
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key",
                timeout=30
            )

            # Verify chat.completions.create was called correctly
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs

            # Check parameters
            assert call_kwargs["model"] == "anthropic/claude-3-haiku"
            assert call_kwargs["temperature"] == 0.0

            # Check that response_format is included for compatible models
            # anthropic/claude-3-haiku is in the compatible list
            assert "response_format" in call_kwargs
            assert call_kwargs["response_format"]["type"] == "json_object"

            # Check messages structure
            messages = call_kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert "test text" in messages[1]["content"]