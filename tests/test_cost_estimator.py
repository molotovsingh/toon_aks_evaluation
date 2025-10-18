"""
Unit Tests for Cost Estimator Module

Tests token estimation heuristics, cost calculations, and model catalog integration.
Includes tests for two-layer cost estimation (document extraction + event extraction).

Run with: uv run python -m pytest tests/test_cost_estimator.py -v
"""

import unittest
import io
from pathlib import Path
from src.ui.cost_estimator import (
    estimate_tokens,
    split_input_output_tokens,
    estimate_cost,
    estimate_all_models,
    get_cost_summary,
    calculate_accuracy,
    suggest_calibration_factor,
    estimate_cost_two_layer,
    estimate_all_models_two_layer,
    CHARS_PER_TOKEN,
    INPUT_TOKEN_RATIO,
    OUTPUT_TOKEN_RATIO
)
from src.ui.document_page_estimator import (
    estimate_document_pages,
    get_confidence_message,
    _estimate_from_file_size,
    SIZE_PER_PAGE_SCANNED_PDF
)
from src.core.document_extractor_catalog import (
    get_doc_extractor_catalog,
    get_doc_extractor,
    get_doc_pricing,
    estimate_doc_cost
)
from src.core.model_catalog import get_model_catalog


class TestTokenEstimation(unittest.TestCase):
    """Test token estimation heuristics"""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation with known inputs"""
        # 4 chars = 1 token rule
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text)
        self.assertEqual(tokens, 2)  # 11/4 = 2.75 → 2 tokens

    def test_estimate_tokens_longer_text(self):
        """Test with longer text sample"""
        # Approximate: 400 chars = 100 tokens
        text = "A" * 400
        tokens = estimate_tokens(text)
        self.assertEqual(tokens, 100)

    def test_estimate_tokens_empty(self):
        """Test empty string edge case"""
        tokens = estimate_tokens("")
        self.assertEqual(tokens, 0)

    def test_estimate_tokens_single_char(self):
        """Test minimum token count"""
        tokens = estimate_tokens("A")
        self.assertEqual(tokens, 1)  # Minimum 1 token

    def test_split_input_output_tokens(self):
        """Test 90/10 input/output split"""
        input_tokens, output_tokens = split_input_output_tokens(1000)

        # Should be 90% input, 10% output
        self.assertEqual(input_tokens, 900)
        self.assertEqual(output_tokens, 100)

    def test_split_zero_tokens(self):
        """Test split with zero tokens"""
        input_tokens, output_tokens = split_input_output_tokens(0)
        self.assertEqual(input_tokens, 0)
        self.assertEqual(output_tokens, 0)


class TestCostEstimation(unittest.TestCase):
    """Test cost calculation with model catalog pricing"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_text = "This is a legal document for testing. " * 25  # ~1000 chars = 250 tokens
        self.catalog = get_model_catalog()

    def test_estimate_cost_gpt4o_mini(self):
        """Test cost estimation for GPT-4o Mini"""
        result = estimate_cost(self.sample_text, "gpt-4o-mini")

        # Validate structure
        self.assertIn("model_id", result)
        self.assertIn("tokens_total", result)
        self.assertIn("cost_usd", result)
        self.assertIn("pricing_available", result)

        # GPT-4o Mini pricing should be available
        self.assertTrue(result["pricing_available"])
        self.assertGreater(result["cost_usd"], 0.0)

        # Cost should be very small for ~250 tokens
        self.assertLess(result["cost_usd"], 0.01)  # Should be well under 1 cent

    def test_estimate_cost_claude_haiku(self):
        """Test cost estimation for Claude 3 Haiku"""
        result = estimate_cost(self.sample_text, "claude-3-haiku-20240307")

        self.assertEqual(result["model_id"], "claude-3-haiku-20240307")
        self.assertTrue(result["pricing_available"])
        self.assertGreater(result["cost_usd"], 0.0)

    def test_estimate_cost_unknown_model(self):
        """Test handling of unknown model"""
        result = estimate_cost(self.sample_text, "unknown-model-xyz")

        self.assertEqual(result["model_id"], "unknown-model-xyz")
        self.assertFalse(result["pricing_available"])
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertIn("not found", result["note"].lower())

    def test_estimate_cost_model_without_pricing(self):
        """Test model that exists but has no pricing (e.g., GPT-5 placeholder)"""
        result = estimate_cost(self.sample_text, "gpt-5")

        # GPT-5 is in catalog but pricing is None
        self.assertEqual(result["model_id"], "gpt-5")
        self.assertFalse(result["pricing_available"])
        self.assertIn("Pricing data not available", result["note"])

    def test_estimate_cost_custom_split(self):
        """Test cost estimation with custom input/output split"""
        result = estimate_cost(
            self.sample_text,
            "gpt-4o-mini",
            input_output_split=(200, 50)  # Custom: 200 input, 50 output
        )

        self.assertEqual(result["tokens_input"], 200)
        self.assertEqual(result["tokens_output"], 50)
        self.assertEqual(result["tokens_total"], 250)

    def test_estimate_cost_free_model(self):
        """Test cost estimation for free model (Gemini 2.0 Flash)"""
        result = estimate_cost(self.sample_text, "gemini-2.0-flash")

        # Gemini Free should have cost = 0
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertTrue(result["pricing_available"])


class TestMultiModelEstimation(unittest.TestCase):
    """Test multi-model cost comparison"""

    def setUp(self):
        self.sample_text = "Legal document text " * 50  # ~1000 chars

    def test_estimate_all_models_no_filter(self):
        """Test estimating across all models"""
        estimates = estimate_all_models(self.sample_text)

        # Should return estimates for multiple models
        self.assertGreater(len(estimates), 0)

        # Each estimate should have expected structure
        for model_id, estimate in estimates.items():
            self.assertIn("cost_usd", estimate)
            self.assertIn("tokens_total", estimate)

    def test_estimate_all_models_by_provider(self):
        """Test filtering by provider"""
        openai_estimates = estimate_all_models(self.sample_text, provider="openai")

        # Should only include OpenAI models
        self.assertGreater(len(openai_estimates), 0)

        # Verify all are OpenAI models
        catalog = get_model_catalog()
        for model_id in openai_estimates.keys():
            model = catalog.get_model(model_id)
            self.assertEqual(model.provider, "openai")

    def test_estimate_all_models_by_category(self):
        """Test filtering by category"""
        budget_estimates = estimate_all_models(self.sample_text, category="Budget")

        self.assertGreater(len(budget_estimates), 0)

        # Verify all are budget models
        catalog = get_model_catalog()
        for model_id in budget_estimates.keys():
            model = catalog.get_model(model_id)
            self.assertEqual(model.category, "Budget")

    def test_estimate_all_models_recommended_only(self):
        """Test filtering for recommended models only"""
        recommended = estimate_all_models(self.sample_text, recommended_only=True)

        # Should have fewer models than total catalog
        all_estimates = estimate_all_models(self.sample_text)
        self.assertLess(len(recommended), len(all_estimates))

    def test_estimate_all_models_invalid_filter(self):
        """Test with filters that match no models"""
        estimates = estimate_all_models(
            self.sample_text,
            provider="nonexistent_provider"
        )

        # Should return empty dict
        self.assertEqual(len(estimates), 0)


class TestCostSummary(unittest.TestCase):
    """Test cost summary statistics"""

    def setUp(self):
        self.sample_text = "Legal document " * 100

    def test_get_cost_summary_basic(self):
        """Test basic cost summary"""
        summary = get_cost_summary(self.sample_text)

        # Validate structure
        self.assertIn("token_estimate", summary)
        self.assertIn("models_analyzed", summary)
        self.assertIn("cost_min", summary)
        self.assertIn("cost_max", summary)
        self.assertIn("cheapest_model", summary)

        # Should have analyzed models
        self.assertGreater(summary["models_analyzed"], 0)

        # Min should be <= max
        self.assertLessEqual(summary["cost_min"], summary["cost_max"])

    def test_get_cost_summary_by_provider(self):
        """Test cost summary filtered by provider"""
        summary = get_cost_summary(self.sample_text, provider="anthropic")

        # Should only analyze Anthropic models
        self.assertGreater(summary["models_analyzed"], 0)

        # Cheapest model should be from Anthropic
        if summary["cheapest_model"]:
            self.assertIn("claude", summary["cheapest_model"].lower())

    def test_get_cost_summary_empty_text(self):
        """Test summary with empty text"""
        summary = get_cost_summary("")

        self.assertEqual(summary["token_estimate"], 0)


class TestCalibrationHelpers(unittest.TestCase):
    """Test calibration and accuracy calculation"""

    def test_calculate_accuracy_perfect(self):
        """Test accuracy calculation with perfect match"""
        accuracy = calculate_accuracy(1000, 1000)
        self.assertEqual(accuracy, 1.0)

    def test_calculate_accuracy_within_tolerance(self):
        """Test accuracy with ~11% error (1000 estimated vs 900 actual)"""
        accuracy = calculate_accuracy(1000, 900)
        # Error = |1000-900|/900 = 0.111, accuracy = 1 - 0.111 = 0.889
        self.assertAlmostEqual(accuracy, 0.889, places=2)

    def test_calculate_accuracy_zero_actual(self):
        """Test edge case with zero actual tokens"""
        accuracy = calculate_accuracy(100, 0)
        self.assertEqual(accuracy, 0.0)

    def test_suggest_calibration_factor_perfect(self):
        """Test calibration with perfect estimates"""
        factor = suggest_calibration_factor(
            estimated_tokens_list=[100, 200, 300],
            actual_tokens_list=[100, 200, 300]
        )
        self.assertAlmostEqual(factor, 1.0, places=2)

    def test_suggest_calibration_factor_overestimate(self):
        """Test calibration when consistently overestimating"""
        factor = suggest_calibration_factor(
            estimated_tokens_list=[100, 200, 300],
            actual_tokens_list=[90, 180, 270]  # Actual is 90% of estimated
        )
        self.assertAlmostEqual(factor, 0.9, places=2)

    def test_suggest_calibration_factor_underestimate(self):
        """Test calibration when consistently underestimating"""
        factor = suggest_calibration_factor(
            estimated_tokens_list=[100, 200, 300],
            actual_tokens_list=[110, 220, 330]  # Actual is 110% of estimated
        )
        self.assertAlmostEqual(factor, 1.1, places=2)

    def test_suggest_calibration_factor_empty_lists(self):
        """Test calibration with empty data"""
        factor = suggest_calibration_factor([], [])
        self.assertEqual(factor, 1.0)

    def test_suggest_calibration_factor_mismatched_lengths(self):
        """Test calibration with mismatched list lengths"""
        factor = suggest_calibration_factor([100, 200], [100])
        self.assertEqual(factor, 1.0)  # Should return default


class TestPricingTableAccuracy(unittest.TestCase):
    """Test that cost estimator matches model catalog pricing"""

    def test_pricing_matches_catalog(self):
        """Verify cost calculations match model catalog pricing"""
        catalog = get_model_catalog()

        # Test with known model (GPT-4o Mini)
        model_id = "gpt-4o-mini"
        pricing = catalog.get_pricing(model_id)

        if pricing:
            # Simulate 1M input tokens, 0 output tokens
            input_tokens = 1_000_000
            output_tokens = 0

            # Manual calculation
            expected_cost = (input_tokens / 1_000_000) * pricing["cost_input_per_1m"]

            # Use estimator (need to create text that estimates to 1M tokens)
            # 1M tokens * 4 chars/token = 4M chars
            large_text = "A" * 4_000_000
            result = estimate_cost(large_text, model_id, input_output_split=(input_tokens, output_tokens))

            # Should match within floating point precision
            self.assertAlmostEqual(result["cost_usd"], expected_cost, places=6)

    def test_all_models_have_valid_pricing_structure(self):
        """Verify all models in catalog return valid cost estimates"""
        catalog = get_model_catalog()
        sample_text = "Test document" * 100

        for model in catalog._registry:
            result = estimate_cost(sample_text, model.model_id)

            # Should always return valid structure
            self.assertIn("cost_usd", result)
            self.assertIn("pricing_available", result)
            self.assertGreaterEqual(result["cost_usd"], 0.0)

            # If model has pricing, estimate should be available
            if model.cost_input_per_1m is not None:
                self.assertTrue(result["pricing_available"])


class TestConstants(unittest.TestCase):
    """Test estimation constants are reasonable"""

    def test_chars_per_token_reasonable(self):
        """Verify CHARS_PER_TOKEN is in expected range"""
        # English text typically 3.5-5 chars/token
        self.assertGreaterEqual(CHARS_PER_TOKEN, 3.0)
        self.assertLessEqual(CHARS_PER_TOKEN, 5.0)

    def test_input_output_ratio_sums_to_one(self):
        """Verify input/output ratios sum to 1.0"""
        total = INPUT_TOKEN_RATIO + OUTPUT_TOKEN_RATIO
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_input_ratio_greater_than_output(self):
        """For extraction tasks, input should dominate"""
        self.assertGreater(INPUT_TOKEN_RATIO, OUTPUT_TOKEN_RATIO)


# ============================================================================
# TWO-LAYER COST ESTIMATION TESTS (Document Extraction + Event Extraction)
# ============================================================================


class TestDocumentPageEstimator(unittest.TestCase):
    """Test document page count estimation without extraction"""

    def test_estimate_single_image(self):
        """Test that images return 1 page with high confidence"""
        # Create mock file object with image extension
        mock_file = io.BytesIO(b"fake_image_data")
        mock_file.name = "test.jpg"

        page_count, confidence = estimate_document_pages(mock_file)

        self.assertEqual(page_count, 1)
        self.assertEqual(confidence, "high")

    def test_estimate_from_file_size_fallback(self):
        """Test file size heuristic fallback"""
        # 3MB file ÷ 300KB per page = 10 pages
        file_size = 3_000_000
        page_count, confidence = _estimate_from_file_size(file_size, SIZE_PER_PAGE_SCANNED_PDF)

        self.assertEqual(page_count, 10)
        self.assertEqual(confidence, "medium")

    def test_estimate_zero_size_file(self):
        """Test edge case with zero-byte file"""
        page_count, confidence = _estimate_from_file_size(0, SIZE_PER_PAGE_SCANNED_PDF)

        # Should return at least 1 page
        self.assertEqual(page_count, 1)
        self.assertEqual(confidence, "medium")

    def test_confidence_message_high(self):
        """Test confidence message for high confidence estimates"""
        message = get_confidence_message("high", 15)
        self.assertIn("metadata", message.lower())
        self.assertIn("15", message)

    def test_confidence_message_medium(self):
        """Test confidence message for medium confidence estimates"""
        message = get_confidence_message("medium", 12)
        self.assertIn("file size", message.lower())
        self.assertIn("±30%", message)

    def test_confidence_message_low(self):
        """Test confidence message for low confidence estimates"""
        message = get_confidence_message("low", 15)
        self.assertIn("default", message.lower())
        self.assertIn("may vary", message.lower())

    def test_estimate_real_pdf_if_exists(self):
        """Test PDF page count estimation on real test file if available"""
        test_pdf = Path("tests/test_documents/sample_legal.pdf")
        if test_pdf.exists():
            page_count, confidence = estimate_document_pages(test_pdf)

            # Should have high confidence for PDF metadata read
            self.assertEqual(confidence, "high")
            self.assertGreater(page_count, 0)


class TestDocumentExtractorCatalog(unittest.TestCase):
    """Test document extractor catalog and pricing queries"""

    def setUp(self):
        """Set up test fixtures"""
        self.catalog = get_doc_extractor_catalog()

    def test_get_docling_extractor(self):
        """Test retrieving Docling (free local OCR) extractor"""
        extractor = get_doc_extractor("docling")

        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.extractor_id, "docling")
        self.assertEqual(extractor.cost_per_page, 0.0)
        self.assertEqual(extractor.provider, "local")
        self.assertTrue(extractor.recommended)

    def test_get_qwen_vl_extractor(self):
        """Test retrieving Qwen-VL (budget vision) extractor"""
        extractor = get_doc_extractor("qwen_vl")

        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.extractor_id, "qwen_vl")
        self.assertEqual(extractor.cost_per_page, 0.00512)  # $0.077 per 15 pages
        self.assertEqual(extractor.provider, "openrouter")
        self.assertTrue(extractor.supports_vision)

    def test_get_gemini_extractor(self):
        """Test retrieving Gemini 2.5 (premium vision) extractor"""
        extractor = get_doc_extractor("gemini")

        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.extractor_id, "gemini")
        self.assertAlmostEqual(extractor.cost_per_page, 0.0583, places=4)  # Midpoint of $0.50-1.25 range
        self.assertEqual(extractor.provider, "google")
        self.assertTrue(extractor.supports_vision)

    def test_get_unknown_extractor(self):
        """Test handling of unknown extractor ID"""
        extractor = get_doc_extractor("unknown_extractor_xyz")
        self.assertIsNone(extractor)

    def test_list_local_extractors(self):
        """Test filtering by local provider"""
        local_extractors = self.catalog.list_extractors(provider="local")

        self.assertGreater(len(local_extractors), 0)
        for extractor in local_extractors:
            self.assertEqual(extractor.provider, "local")

    def test_list_vision_extractors(self):
        """Test filtering by vision capability"""
        vision_extractors = self.catalog.list_extractors(supports_vision=True)

        self.assertGreater(len(vision_extractors), 0)
        for extractor in vision_extractors:
            self.assertTrue(extractor.supports_vision)

    def test_list_free_extractors(self):
        """Test filtering for zero-cost extractors"""
        free_extractors = self.catalog.list_extractors(free_only=True)

        self.assertGreater(len(free_extractors), 0)
        for extractor in free_extractors:
            self.assertEqual(extractor.cost_per_page, 0.0)

    def test_list_recommended_extractors(self):
        """Test filtering for recommended extractors"""
        recommended = self.catalog.list_extractors(recommended_only=True)

        self.assertGreater(len(recommended), 0)
        for extractor in recommended:
            self.assertTrue(extractor.recommended)

        # Should have Docling in recommended
        recommended_ids = [e.extractor_id for e in recommended]
        self.assertIn("docling", recommended_ids)

    def test_get_pricing_docling(self):
        """Test pricing retrieval for Docling (free)"""
        pricing = get_doc_pricing("docling")

        self.assertIsNotNone(pricing)
        self.assertEqual(pricing["cost_per_page"], 0.0)
        self.assertEqual(pricing["cost_display"], "FREE")

    def test_get_pricing_qwen_vl(self):
        """Test pricing retrieval for Qwen-VL"""
        pricing = get_doc_pricing("qwen_vl")

        self.assertIsNotNone(pricing)
        self.assertEqual(pricing["cost_per_page"], 0.00512)
        self.assertIn("$0.077", pricing["cost_display"])

    def test_get_pricing_unknown(self):
        """Test pricing retrieval for unknown extractor"""
        pricing = get_doc_pricing("unknown_extractor")
        self.assertIsNone(pricing)

    def test_estimate_doc_cost_docling(self):
        """Test cost estimation for Docling (free)"""
        result = estimate_doc_cost("docling", page_count=15)

        self.assertEqual(result["extractor_id"], "docling")
        self.assertEqual(result["page_count"], 15)
        self.assertEqual(result["cost_per_page"], 0.0)
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertEqual(result["cost_display"], "FREE")
        self.assertTrue(result["pricing_available"])

    def test_estimate_doc_cost_qwen_vl(self):
        """Test cost estimation for Qwen-VL"""
        result = estimate_doc_cost("qwen_vl", page_count=15)

        self.assertEqual(result["extractor_id"], "qwen_vl")
        self.assertEqual(result["page_count"], 15)
        self.assertEqual(result["cost_per_page"], 0.00512)
        self.assertAlmostEqual(result["cost_usd"], 0.0768, places=4)  # 15 * 0.00512
        self.assertIn("$0.0768", result["cost_display"])
        self.assertTrue(result["pricing_available"])

    def test_estimate_doc_cost_gemini(self):
        """Test cost estimation for Gemini 2.5"""
        result = estimate_doc_cost("gemini", page_count=15)

        self.assertEqual(result["extractor_id"], "gemini")
        self.assertEqual(result["page_count"], 15)
        self.assertAlmostEqual(result["cost_per_page"], 0.0583, places=4)
        expected_cost = 15 * 0.0583  # ~$0.8745
        self.assertAlmostEqual(result["cost_usd"], expected_cost, places=4)
        self.assertTrue(result["pricing_available"])

    def test_estimate_doc_cost_unknown_extractor(self):
        """Test cost estimation for unknown extractor"""
        result = estimate_doc_cost("unknown_extractor", page_count=15)

        self.assertEqual(result["extractor_id"], "unknown_extractor")
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertFalse(result["pricing_available"])
        self.assertIn("Unknown extractor", result["cost_display"])

    def test_validate_extractor_id(self):
        """Test extractor ID validation"""
        self.assertTrue(self.catalog.validate_extractor_id("docling"))
        self.assertTrue(self.catalog.validate_extractor_id("qwen_vl"))
        self.assertTrue(self.catalog.validate_extractor_id("gemini"))
        self.assertFalse(self.catalog.validate_extractor_id("unknown"))

    def test_get_all_extractor_ids(self):
        """Test retrieving all extractor IDs"""
        extractor_ids = self.catalog.get_all_extractor_ids()

        self.assertGreater(len(extractor_ids), 0)
        self.assertIn("docling", extractor_ids)
        self.assertIn("qwen_vl", extractor_ids)
        self.assertIn("gemini", extractor_ids)


class TestTwoLayerCostEstimation(unittest.TestCase):
    """Test two-layer cost estimation (document extraction + event extraction)"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_text = "Legal document text " * 100  # ~2000 chars

        # Create mock uploaded file (BytesIO with file-like interface)
        self.mock_file = io.BytesIO(b"x" * 300_000)  # 300KB = ~1 page scanned PDF
        self.mock_file.name = "test_document.pdf"

    def test_estimate_cost_two_layer_docling_gpt4o_mini(self):
        """Test two-layer estimation with Docling (free) + GPT-4o Mini"""
        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="docling",
            event_model="gpt-4o-mini",
            extracted_texts=[self.sample_text]
        )

        # Validate structure
        self.assertIn("document_cost", result)
        self.assertIn("event_cost", result)
        self.assertIn("total_cost", result)
        self.assertIn("page_count", result)
        self.assertIn("page_confidence", result)
        self.assertIn("tokens_total", result)
        self.assertIn("pricing_available", result)

        # Layer 1: Docling is free
        self.assertEqual(result["document_cost"], 0.0)
        self.assertEqual(result["document_cost_display"], "FREE")

        # Layer 2: GPT-4o Mini should have cost > 0
        self.assertGreater(result["event_cost"], 0.0)

        # Total should equal event cost (since doc cost is free)
        self.assertAlmostEqual(result["total_cost"], result["event_cost"], places=6)

        # Page count should be estimated
        self.assertGreater(result["page_count"], 0)

        # Should have pricing available
        self.assertTrue(result["pricing_available"])

    def test_estimate_cost_two_layer_qwen_vl_claude_haiku(self):
        """Test two-layer estimation with Qwen-VL + Claude Haiku"""
        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="qwen_vl",
            event_model="claude-3-haiku-20240307",
            extracted_texts=[self.sample_text]
        )

        # Both layers should have cost > 0
        self.assertGreater(result["document_cost"], 0.0)
        self.assertGreater(result["event_cost"], 0.0)

        # Total should be sum of both layers
        expected_total = result["document_cost"] + result["event_cost"]
        self.assertAlmostEqual(result["total_cost"], expected_total, places=6)

        # Should have pricing available
        self.assertTrue(result["pricing_available"])

    def test_estimate_cost_two_layer_gemini_gemini_flash(self):
        """Test two-layer estimation with Gemini 2.5 doc + Gemini 2.0 Flash event"""
        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="gemini",
            event_model="gemini-2.0-flash",
            extracted_texts=[self.sample_text]
        )

        # Layer 1: Gemini doc extraction should have cost > 0
        self.assertGreater(result["document_cost"], 0.0)

        # Layer 2: Gemini 2.0 Flash is free
        self.assertEqual(result["event_cost"], 0.0)

        # Total should equal document cost
        self.assertAlmostEqual(result["total_cost"], result["document_cost"], places=6)

    def test_estimate_cost_two_layer_multiple_files(self):
        """Test two-layer estimation with multiple uploaded files"""
        mock_file2 = io.BytesIO(b"y" * 600_000)  # 600KB = ~2 pages
        mock_file2.name = "test_document2.pdf"

        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file, mock_file2],
            doc_extractor="qwen_vl",
            event_model="gpt-4o-mini",
            extracted_texts=[self.sample_text, self.sample_text]
        )

        # Page count should be sum of both files (~3 pages)
        self.assertGreaterEqual(result["page_count"], 2)

        # Document cost should reflect multiple pages
        expected_doc_cost = result["page_count"] * 0.00512  # Qwen-VL rate
        self.assertAlmostEqual(result["document_cost"], expected_doc_cost, places=4)

    def test_estimate_cost_two_layer_unknown_extractor(self):
        """Test two-layer estimation with unknown document extractor"""
        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="unknown_extractor",
            event_model="gpt-4o-mini",
            extracted_texts=[self.sample_text]
        )

        # Document cost should be 0 for unknown extractor
        self.assertEqual(result["document_cost"], 0.0)

        # Event cost should still be calculated
        self.assertGreater(result["event_cost"], 0.0)

        # Should indicate pricing not fully available
        self.assertFalse(result["pricing_available"])

    def test_estimate_cost_two_layer_unknown_model(self):
        """Test two-layer estimation with unknown event extraction model"""
        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="docling",
            event_model="unknown_model_xyz",
            extracted_texts=[self.sample_text]
        )

        # Document cost should be calculated correctly
        self.assertEqual(result["document_cost"], 0.0)  # Docling is free

        # Event cost should be 0 for unknown model
        self.assertEqual(result["event_cost"], 0.0)

        # Should indicate pricing not available
        self.assertFalse(result["pricing_available"])

    def test_estimate_cost_two_layer_no_extracted_text(self):
        """Test two-layer estimation when extraction hasn't happened yet"""
        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="docling",
            event_model="gpt-4o-mini",
            extracted_texts=None  # Not extracted yet
        )

        # Document cost should still be estimated
        self.assertEqual(result["document_cost"], 0.0)  # Docling is free

        # Event cost should be estimated from page count (non-zero for non-empty files)
        # Page count ~1, estimated chars 2000, so small but non-zero cost
        self.assertGreater(result["event_cost"], 0.0)
        self.assertLess(result["event_cost"], 0.01)  # Should be very small

        # Should have note explaining this is an estimate
        self.assertIn("note", result)

    def test_estimate_all_models_two_layer(self):
        """Test two-layer estimation across all models"""
        estimates = estimate_all_models_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="docling",
            extracted_texts=[self.sample_text]
        )

        # Should return estimates for multiple models
        self.assertGreater(len(estimates), 0)

        # Each estimate should have two-layer structure
        for model_id, estimate in estimates.items():
            self.assertIn("document_cost", estimate)
            self.assertIn("event_cost", estimate)
            self.assertIn("total_cost", estimate)

            # All should have same document cost (using Docling)
            self.assertEqual(estimate["document_cost"], 0.0)

    def test_estimate_all_models_two_layer_with_filter(self):
        """Test two-layer estimation filtered by provider"""
        estimates = estimate_all_models_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="qwen_vl",
            extracted_texts=[self.sample_text],
            provider="openai"
        )

        # Should only include OpenAI models
        self.assertGreater(len(estimates), 0)

        # Verify all are OpenAI models
        model_catalog = get_model_catalog()
        for model_id in estimates.keys():
            model = model_catalog.get_model(model_id)
            self.assertEqual(model.provider, "openai")

        # All should have same non-zero document cost (Qwen-VL)
        for estimate in estimates.values():
            self.assertGreater(estimate["document_cost"], 0.0)

    def test_confidence_propagation(self):
        """Test that page count confidence is properly propagated"""
        # Create image file (high confidence for page count)
        image_file = io.BytesIO(b"image_data")
        image_file.name = "test.jpg"

        result = estimate_cost_two_layer(
            uploaded_files=[image_file],
            doc_extractor="docling",
            event_model="gpt-4o-mini",
            extracted_texts=[self.sample_text]
        )

        # Should have high confidence for images
        self.assertEqual(result["page_confidence"], "high")

    def test_cost_display_formatting(self):
        """Test that cost display strings are properly formatted"""
        result = estimate_cost_two_layer(
            uploaded_files=[self.mock_file],
            doc_extractor="qwen_vl",
            event_model="gpt-4o-mini",
            extracted_texts=[self.sample_text]
        )

        # All cost display strings should be present
        self.assertIsNotNone(result["document_cost_display"])
        self.assertIsNotNone(result["event_cost_display"])
        self.assertIsNotNone(result["total_cost_display"])

        # Should contain dollar signs (or "FREE")
        total_display = result["total_cost_display"]
        self.assertTrue("$" in total_display or "FREE" in total_display)


class TestTwoLayerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for two-layer cost estimation"""

    def test_empty_file_list(self):
        """Test two-layer estimation with empty file list"""
        result = estimate_cost_two_layer(
            uploaded_files=[],
            doc_extractor="docling",
            event_model="gpt-4o-mini",
            extracted_texts=["sample text"]
        )

        # Should handle gracefully
        self.assertEqual(result["page_count"], 0)
        self.assertEqual(result["document_cost"], 0.0)

    def test_mismatched_files_and_texts(self):
        """Test when number of files doesn't match number of texts"""
        mock_file = io.BytesIO(b"x" * 100_000)
        mock_file.name = "test.pdf"

        # 1 file, 2 texts
        result = estimate_cost_two_layer(
            uploaded_files=[mock_file],
            doc_extractor="docling",
            event_model="gpt-4o-mini",
            extracted_texts=["text1", "text2"]
        )

        # Should still work - uses max of file count vs text count
        self.assertGreater(result["page_count"], 0)
        self.assertGreater(result["event_cost"], 0.0)

    def test_very_large_document(self):
        """Test cost estimation for very large document"""
        # Simulate 100-page document
        large_file = io.BytesIO(b"x" * 30_000_000)  # 30MB
        large_file.name = "large_doc.pdf"

        result = estimate_cost_two_layer(
            uploaded_files=[large_file],
            doc_extractor="qwen_vl",
            event_model="gpt-4o-mini",
            extracted_texts=["text" * 10000]  # Large text
        )

        # Page count should be substantial
        self.assertGreater(result["page_count"], 50)

        # Costs should be non-trivial
        self.assertGreater(result["document_cost"], 0.5)  # Qwen-VL on 100 pages
        self.assertGreater(result["event_cost"], 0.0)

    def test_zero_cost_both_layers(self):
        """Test when both layers are free (Docling + Gemini 2.0 Flash)"""
        mock_file = io.BytesIO(b"x" * 100_000)
        mock_file.name = "test.pdf"

        result = estimate_cost_two_layer(
            uploaded_files=[mock_file],
            doc_extractor="docling",
            event_model="gemini-2.0-flash",
            extracted_texts=["sample text"]
        )

        # Both layers free
        self.assertEqual(result["document_cost"], 0.0)
        self.assertEqual(result["event_cost"], 0.0)
        self.assertEqual(result["total_cost"], 0.0)

        # Display should show FREE
        self.assertEqual(result["total_cost_display"], "FREE")


class TestNoPaidExtractionDuringEstimation(unittest.TestCase):
    """
    Regression tests to ensure cost estimation does not trigger paid document extractors.

    This is a CRITICAL constraint from order cost-estimator-003:
    "Cost previews do not make outbound calls to paid document extractors"
    """

    def test_estimate_cost_two_layer_does_not_call_extractor(self):
        """
        Test that estimate_cost_two_layer works WITHOUT extracted_texts (no extraction run).

        This is the key regression test: when extracted_texts=None, the function should
        use page count heuristics (2000 chars/page) instead of calling the paid extractor.
        """
        mock_file = io.BytesIO(b"x" * 300_000)  # 300KB file
        mock_file.name = "test.pdf"

        # Call with extracted_texts=None - should use heuristics, not extraction
        result = estimate_cost_two_layer(
            uploaded_files=[mock_file],
            doc_extractor="qwen_vl",  # Paid extractor
            event_model="gpt-4o-mini",
            extracted_texts=None  # No extraction yet
        )

        # Should successfully estimate costs using heuristics
        self.assertGreater(result["document_cost"], 0.0)
        self.assertGreater(result["event_cost"], 0.0)
        self.assertIn("note", result)  # Should have note explaining estimate method

    def test_estimate_all_models_two_layer_does_not_call_extractor(self):
        """Test that multi-model estimation works without extraction"""
        mock_file = io.BytesIO(b"x" * 300_000)
        mock_file.name = "test.pdf"

        # Call with extracted_texts=None
        estimates = estimate_all_models_two_layer(
            uploaded_files=[mock_file],
            doc_extractor="gemini",  # Paid extractor
            extracted_texts=None,  # No extraction yet
            recommended_only=True
        )

        # Should return estimates for multiple models
        self.assertGreater(len(estimates), 0)

        # Each should have document cost without calling extractor
        for model_id, estimate in estimates.items():
            self.assertIn("document_cost", estimate)
            self.assertIn("event_cost", estimate)
            self.assertGreaterEqual(estimate["document_cost"], 0.0)

    def test_page_count_estimation_without_extraction(self):
        """Test that page count can be estimated from file metadata/size"""
        from src.ui.document_page_estimator import estimate_document_pages

        # Test PDF metadata path (no extraction needed)
        mock_file = io.BytesIO(b"x" * 300_000)
        mock_file.name = "test.pdf"

        page_count, confidence = estimate_document_pages(mock_file)

        # Should return a page count without extraction
        self.assertGreater(page_count, 0)
        self.assertIn(confidence, ["high", "medium", "low"])

    def test_docling_free_extraction_allowed(self):
        """Test that Docling (free) extraction is allowed during estimation"""
        mock_file = io.BytesIO(b"x" * 300_000)
        mock_file.name = "test.pdf"

        # Docling is free, so calling it for page count is OK
        result = estimate_cost_two_layer(
            uploaded_files=[mock_file],
            doc_extractor="docling",  # Free extractor
            event_model="gpt-4o-mini",
            extracted_texts=None
        )

        # Should work - Docling is free
        self.assertEqual(result["document_cost"], 0.0)
        self.assertGreater(result["event_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
