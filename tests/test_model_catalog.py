"""
Unit tests for model catalog registry.

Tests cover:
- Model lookup by ID
- Query filtering (provider, tier, category, status)
- Capability extraction
- Pricing data retrieval
- Model validation
- Runtime model resolution
- Helper API functions
"""

import pytest
from src.core.model_catalog import (
    ModelCatalog,
    ModelEntry,
    ModelTier,
    ModelStatus,
    get_model_catalog,
    get_model,
    list_models,
    get_capabilities,
    get_pricing,
    validate_model_id,
    get_ui_model_config_list,
)


class TestModelCatalogBasics:
    """Test basic catalog operations"""

    def test_get_model_by_id_exists(self):
        """Should retrieve model by exact ID match"""
        catalog = get_model_catalog()

        # Ground truth model
        claude = catalog.get_model("claude-sonnet-4-5")
        assert claude is not None
        assert claude.display_name == "Claude Sonnet 4.5"
        assert claude.provider == "anthropic"
        assert claude.tier == ModelTier.TIER_1

        # Production model with OpenRouter prefix
        gpt4o_mini = catalog.get_model("gpt-4o-mini")
        assert gpt4o_mini is not None
        assert gpt4o_mini.display_name == "GPT-4o Mini"
        assert gpt4o_mini.provider == "openai"

    def test_get_model_by_id_not_exists(self):
        """Should return None for non-existent model ID"""
        catalog = get_model_catalog()

        result = catalog.get_model("nonexistent-model-id")
        assert result is None

        result = catalog.get_model("gpt-6")  # Not released yet
        assert result is None

    def test_validate_model_id_exists(self):
        """Should validate existing model IDs"""
        catalog = get_model_catalog()

        assert catalog.validate_model_id("claude-sonnet-4-5") is True
        assert catalog.validate_model_id("gpt-4o-mini") is True
        assert catalog.validate_model_id("deepseek/deepseek-r1-distill-llama-70b") is True

    def test_validate_model_id_not_exists(self):
        """Should reject non-existent model IDs"""
        catalog = get_model_catalog()

        assert catalog.validate_model_id("fake-model") is False
        assert catalog.validate_model_id("") is False

    def test_get_all_model_ids(self):
        """Should return complete list of model IDs"""
        catalog = get_model_catalog()

        model_ids = catalog.get_all_model_ids()

        # Should include all major models from design doc
        assert "claude-sonnet-4-5" in model_ids
        assert "gpt-5" in model_ids
        assert "gpt-4o-mini" in model_ids
        assert "deepseek-chat" in model_ids
        assert "gemini-2.0-flash" in model_ids

        # Should have 15 models total (from design)
        assert len(model_ids) >= 15


class TestModelQueryFiltering:
    """Test catalog query methods with various filters"""

    def test_list_models_no_filters(self):
        """Should return all models with no filters"""
        catalog = get_model_catalog()

        all_models = catalog.list_models()
        assert len(all_models) >= 15  # At least 15 models in catalog

    def test_list_models_by_provider(self):
        """Should filter by provider name"""
        catalog = get_model_catalog()

        anthropic_models = catalog.list_models(provider="anthropic")
        assert len(anthropic_models) >= 2  # Claude Sonnet 4.5, Haiku, Opus 4
        assert all(m.provider == "anthropic" for m in anthropic_models)

        openai_models = catalog.list_models(provider="openai")
        assert len(openai_models) >= 3  # GPT-5, GPT-4o, GPT-4o-mini
        assert all(m.provider == "openai" for m in openai_models)

        google_models = catalog.list_models(provider="google")
        assert len(google_models) >= 2  # Gemini 2.0 Flash, 2.5 Pro
        assert all(m.provider == "google" for m in google_models)

    def test_list_models_by_tier(self):
        """Should filter by tier classification"""
        catalog = get_model_catalog()

        tier1 = catalog.list_models(tier=ModelTier.TIER_1)
        assert len(tier1) == 1  # Only Claude Sonnet 4.5
        assert tier1[0].model_id == "claude-sonnet-4-5"

        tier2 = catalog.list_models(tier=ModelTier.TIER_2)
        assert len(tier2) >= 2  # GPT-5, Gemini 2.5 Pro

        production = catalog.list_models(tier=ModelTier.PRODUCTION)
        assert len(production) >= 5  # Multiple production models

    def test_list_models_by_category(self):
        """Should filter by category string"""
        catalog = get_model_catalog()

        ground_truth = catalog.list_models(category="Ground Truth")
        assert len(ground_truth) >= 4  # Sonnet 4.5, GPT-5, Gemini 2.5 Pro, Opus 4

        production = catalog.list_models(category="Production")
        assert len(production) >= 5

        budget = catalog.list_models(category="Budget")
        assert len(budget) >= 3  # DeepSeek, Llama, Qwen, etc.

    def test_list_models_by_status(self):
        """Should filter by model status"""
        catalog = get_model_catalog()

        stable = catalog.list_models(status=ModelStatus.STABLE)
        assert len(stable) >= 10

        placeholders = catalog.list_models(status=ModelStatus.PLACEHOLDER)
        assert len(placeholders) >= 2  # GPT-5, Gemini 2.5 Pro

        experimental = catalog.list_models(status=ModelStatus.EXPERIMENTAL)
        # Qwen QwQ 32B should be experimental
        assert any(m.model_id == "qwen/qwq-32b" for m in experimental)

    def test_list_models_by_min_context(self):
        """Should filter by minimum context window size"""
        catalog = get_model_catalog()

        # Models with 200K+ context
        large_context = catalog.list_models(min_context=200000)
        assert all(m.context_window >= 200000 for m in large_context)
        assert any(m.model_id == "claude-sonnet-4-5" for m in large_context)

        # Models with 1M+ context
        huge_context = catalog.list_models(min_context=1000000)
        assert all(m.context_window >= 1000000 for m in huge_context)
        assert any(m.model_id == "gemini-2.0-flash" for m in huge_context)

    def test_list_models_recommended_only(self):
        """Should filter to recommended models"""
        catalog = get_model_catalog()

        recommended = catalog.list_models(recommended_only=True)
        assert len(recommended) >= 5  # Multiple recommended models
        assert all(m.recommended for m in recommended)

        # Claude Sonnet 4.5 should be recommended
        assert any(m.model_id == "claude-sonnet-4-5" for m in recommended)

    def test_list_models_combined_filters(self):
        """Should apply multiple filters simultaneously"""
        catalog = get_model_catalog()

        # Production Anthropic models
        result = catalog.list_models(
            provider="anthropic",
            tier=ModelTier.PRODUCTION
        )
        assert all(m.provider == "anthropic" and m.tier == ModelTier.PRODUCTION for m in result)

        # Stable ground truth models
        result = catalog.list_models(
            category="Ground Truth",
            status=ModelStatus.STABLE
        )
        assert all(m.category == "Ground Truth" and m.status == ModelStatus.STABLE for m in result)


class TestCapabilitiesAndPricing:
    """Test capability flags and pricing retrieval"""

    def test_get_capabilities_json_mode(self):
        """Should retrieve JSON mode support flag"""
        catalog = get_model_catalog()

        # Models with native JSON mode
        claude_caps = catalog.get_capabilities("claude-sonnet-4-5")
        assert claude_caps["supports_json_mode"] is True

        gpt4o_caps = catalog.get_capabilities("gpt-4o-mini")
        assert gpt4o_caps["supports_json_mode"] is True

        # Models without native JSON mode (use prompt-based)
        deepseek_caps = catalog.get_capabilities("deepseek/deepseek-r1-distill-llama-70b")
        assert deepseek_caps["supports_json_mode"] is False

    def test_get_capabilities_responses_api(self):
        """Should identify GPT-5 Responses API requirement"""
        catalog = get_model_catalog()

        gpt5_caps = catalog.get_capabilities("gpt-5")
        assert gpt5_caps["requires_responses_api"] is True

        # Other models should not require it
        claude_caps = catalog.get_capabilities("claude-sonnet-4-5")
        assert claude_caps["requires_responses_api"] is False

    def test_get_capabilities_vision(self):
        """Should identify vision support"""
        catalog = get_model_catalog()

        gpt4o_caps = catalog.get_capabilities("gpt-4o")
        assert gpt4o_caps["supports_vision"] is True

        gpt4o_mini_caps = catalog.get_capabilities("gpt-4o-mini")
        assert gpt4o_mini_caps["supports_vision"] is True

    def test_get_capabilities_nonexistent_model(self):
        """Should return empty dict for non-existent model"""
        catalog = get_model_catalog()

        caps = catalog.get_capabilities("fake-model-id")
        assert caps == {}

    def test_get_pricing_stable_model(self):
        """Should retrieve pricing for stable models"""
        catalog = get_model_catalog()

        claude_pricing = catalog.get_pricing("claude-sonnet-4-5")
        assert claude_pricing is not None
        assert claude_pricing["cost_input_per_1m"] == 3.0
        assert claude_pricing["cost_output_per_1m"] == 15.0

        gpt4o_mini_pricing = catalog.get_pricing("gpt-4o-mini")
        assert gpt4o_mini_pricing is not None
        assert gpt4o_mini_pricing["cost_input_per_1m"] == 0.15
        assert gpt4o_mini_pricing["cost_output_per_1m"] == 0.60

    def test_get_pricing_free_model(self):
        """Should handle free models correctly"""
        catalog = get_model_catalog()

        gemini_pricing = catalog.get_pricing("gemini-2.0-flash")
        assert gemini_pricing is not None
        assert gemini_pricing["cost_input_per_1m"] == 0.0
        assert gemini_pricing["cost_output_per_1m"] == 0.0

    def test_get_pricing_placeholder_model(self):
        """Should return None for models without pricing (GPT-5, Gemini 2.5 Pro)"""
        catalog = get_model_catalog()

        gpt5_pricing = catalog.get_pricing("gpt-5")
        assert gpt5_pricing is None  # Pricing TBD

        gemini25_pricing = catalog.get_pricing("gemini-2.5-pro")
        assert gemini25_pricing is None  # Pricing TBD

    def test_get_pricing_nonexistent_model(self):
        """Should return None for non-existent model"""
        catalog = get_model_catalog()

        pricing = catalog.get_pricing("fake-model-id")
        assert pricing is None


class TestRuntimeModelResolution:
    """Test runtime model override resolution logic"""

    def test_resolve_with_runtime_override(self):
        """Should prioritize runtime_model parameter (UI selection)"""
        catalog = get_model_catalog()

        result = catalog.resolve_runtime_model(
            provider="anthropic",
            runtime_model="claude-opus-4",  # UI override
            env_defaults={"anthropic": "claude-3-haiku-20240307"}
        )

        assert result == "claude-opus-4"  # UI wins

    def test_resolve_with_env_default(self):
        """Should use env_defaults when no runtime override"""
        catalog = get_model_catalog()

        result = catalog.resolve_runtime_model(
            provider="openai",
            runtime_model=None,
            env_defaults={"openai": "gpt-4o"}
        )

        assert result == "gpt-4o"

    def test_resolve_fallback_to_catalog(self):
        """Should use first catalog model when no overrides"""
        catalog = get_model_catalog()

        result = catalog.resolve_runtime_model(
            provider="anthropic",
            runtime_model=None,
            env_defaults={}
        )

        # Should return first Anthropic model from catalog
        anthropic_models = catalog.list_models(provider="anthropic")
        assert result == anthropic_models[0].model_id

    def test_resolve_invalid_provider(self):
        """Should raise ValueError for unknown provider"""
        catalog = get_model_catalog()

        with pytest.raises(ValueError, match="No models found for provider"):
            catalog.resolve_runtime_model(
                provider="nonexistent-provider",
                runtime_model=None,
                env_defaults={}
            )


class TestGroundTruthAndRecommendedHelpers:
    """Test specialized query helpers"""

    def test_get_ground_truth_models(self):
        """Should return all Tier 1-3 models"""
        catalog = get_model_catalog()

        gt_models = catalog.get_ground_truth_models()

        # Should include Claude Sonnet 4.5 (T1), GPT-5 (T2), Gemini 2.5 Pro (T2), Opus 4 (T3)
        assert len(gt_models) >= 4

        # All should be ground truth tiers
        valid_tiers = {ModelTier.TIER_1, ModelTier.TIER_2, ModelTier.TIER_3}
        assert all(m.tier in valid_tiers for m in gt_models)

        # Should include specific models
        model_ids = {m.model_id for m in gt_models}
        assert "claude-sonnet-4-5" in model_ids
        assert "gpt-5" in model_ids

    def test_get_recommended_models(self):
        """Should return all models marked as recommended"""
        catalog = get_model_catalog()

        recommended = catalog.get_recommended_models()

        # Should have at least 5 recommended models
        assert len(recommended) >= 5

        # All should be marked recommended
        assert all(m.recommended for m in recommended)

        # Should include key models
        model_ids = {m.model_id for m in recommended}
        assert "claude-sonnet-4-5" in model_ids  # Tier 1
        assert "claude-3-haiku-20240307" in model_ids  # Speed champion
        assert "gpt-4o-mini" in model_ids  # Recommended starting point


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_convenience_get_model(self):
        """Convenience function should match catalog method"""
        result = get_model("claude-sonnet-4-5")
        assert result is not None
        assert result.display_name == "Claude Sonnet 4.5"

    def test_convenience_list_models(self):
        """Convenience function should support filters"""
        result = list_models(provider="openai", tier=ModelTier.PRODUCTION)
        assert all(m.provider == "openai" and m.tier == ModelTier.PRODUCTION for m in result)

    def test_convenience_get_capabilities(self):
        """Convenience function should return capabilities dict"""
        caps = get_capabilities("gpt-4o-mini")
        assert "supports_json_mode" in caps
        assert caps["supports_json_mode"] is True

    def test_convenience_get_pricing(self):
        """Convenience function should return pricing dict"""
        pricing = get_pricing("gpt-4o-mini")
        assert pricing is not None
        assert "cost_input_per_1m" in pricing

    def test_convenience_validate_model_id(self):
        """Convenience function should validate IDs"""
        assert validate_model_id("claude-sonnet-4-5") is True
        assert validate_model_id("fake-model") is False


class TestUICompatibilityLayer:
    """Test app.py migration helper"""

    def test_get_ui_model_config_list_structure(self):
        """Should return list of dicts matching old ModelConfig format"""
        ui_models = get_ui_model_config_list()

        # Should have multiple models
        assert len(ui_models) >= 15

        # Each should have required fields
        for model in ui_models:
            assert "provider" in model
            assert "model_id" in model
            assert "display_name" in model
            assert "category" in model
            assert "cost_per_1m" in model
            assert "context_window" in model
            assert "quality_score" in model
            assert "badges" in model

    def test_get_ui_model_config_list_values(self):
        """Should map ModelEntry fields correctly"""
        ui_models = get_ui_model_config_list()

        # Find Claude Sonnet 4.5
        claude = next((m for m in ui_models if m["model_id"] == "claude-sonnet-4-5"), None)
        assert claude is not None

        assert claude["provider"] == "anthropic"
        assert claude["display_name"] == "Claude Sonnet 4.5"
        assert claude["category"] == "Ground Truth"
        assert claude["cost_per_1m"] == "$3/M"
        assert claude["context_window"] == "200K"
        assert claude["quality_score"] == "10/10"
        assert "Tier 1" in claude["badges"]
        assert "Recommended" in claude["badges"]

    def test_format_inline_method(self):
        """Should provide format_inline() method matching app.py usage"""
        ui_models = get_ui_model_config_list()

        # Find Claude Sonnet 4.5 to test format_inline()
        claude = next((m for m in ui_models if m.model_id == "claude-sonnet-4-5"), None)
        assert claude is not None

        # Method should exist
        assert hasattr(claude, "format_inline"), "UILegacyModelConfig missing format_inline() method"
        assert callable(claude.format_inline), "format_inline should be callable"

        # Should return correct format: quality • cost • context • badges
        inline = claude.format_inline()
        assert isinstance(inline, str)
        assert "10/10" in inline  # quality_score
        assert "$3/M" in inline   # cost_per_1m
        assert "200K" in inline   # context_window
        assert "Tier 1" in inline # badge
        assert "Recommended" in inline # badge

        # Should use bullet separator
        assert " • " in inline

    def test_attribute_access(self):
        """Should support attribute access (not just dictionary access)"""
        ui_models = get_ui_model_config_list()

        # Find Claude Sonnet 4.5
        claude = next((m for m in ui_models if m.model_id == "claude-sonnet-4-5"), None)
        assert claude is not None

        # Attribute access should work (this is what app.py uses)
        assert claude.provider == "anthropic"
        assert claude.model_id == "claude-sonnet-4-5"
        assert claude.display_name == "Claude Sonnet 4.5"
        assert claude.category == "Ground Truth"
        assert claude.cost_per_1m == "$3/M"
        assert claude.context_window == "200K"
        assert claude.quality_score == "10/10"
        assert isinstance(claude.badges, list)
        assert "Tier 1" in claude.badges

    def test_badge_ordering(self):
        """Should maintain correct ordering in format_inline() output"""
        ui_models = get_ui_model_config_list()

        # Find Claude Sonnet 4.5
        claude = next((m for m in ui_models if m.model_id == "claude-sonnet-4-5"), None)
        assert claude is not None

        inline = claude.format_inline()
        parts = inline.split(" • ")

        # Should have at least 5 parts (quality, cost, context, 2+ badges)
        assert len(parts) >= 5

        # Order should be: quality_score, cost_per_1m, context_window, badges...
        assert parts[0] == "10/10"  # quality first
        assert parts[1] == "$3/M"   # cost second
        assert parts[2] == "200K"   # context third
        # Remaining parts are badges
        assert "Tier 1" in parts[3:]
        assert "Recommended" in parts[3:]


class TestCatalogSingleton:
    """Test singleton pattern for global catalog instance"""

    def test_get_model_catalog_returns_same_instance(self):
        """Should return same catalog instance on repeated calls"""
        catalog1 = get_model_catalog()
        catalog2 = get_model_catalog()

        # Should be same object (singleton pattern)
        assert catalog1 is catalog2

    def test_catalog_has_consistent_data(self):
        """Catalog data should be consistent across calls"""
        catalog1 = get_model_catalog()
        models1 = catalog1.get_all_model_ids()

        catalog2 = get_model_catalog()
        models2 = catalog2.get_all_model_ids()

        assert models1 == models2


class TestDataQuality:
    """Test data quality and consistency in catalog"""

    def test_all_models_have_required_fields(self):
        """All models should have essential metadata"""
        catalog = get_model_catalog()

        for model in catalog._registry:
            # Identity fields
            assert model.provider, f"Missing provider for {model.model_id}"
            assert model.model_id, "Missing model_id"
            assert model.display_name, f"Missing display_name for {model.model_id}"

            # Classification
            assert model.tier is not None, f"Missing tier for {model.model_id}"
            assert model.category, f"Missing category for {model.model_id}"
            assert model.status is not None, f"Missing status for {model.model_id}"

            # Context and cost display
            assert model.context_window > 0, f"Invalid context_window for {model.model_id}"
            assert model.context_display, f"Missing context_display for {model.model_id}"
            assert model.cost_display, f"Missing cost_display for {model.model_id}"

    def test_stable_models_have_pricing(self):
        """Stable production models should have pricing data (except free tier)"""
        catalog = get_model_catalog()

        stable_production = catalog.list_models(
            tier=ModelTier.PRODUCTION,
            status=ModelStatus.STABLE
        )

        for model in stable_production:
            # Free models can have None pricing
            if model.cost_display != "Free":
                assert model.cost_input_per_1m is not None, \
                    f"Stable production model {model.model_id} missing input pricing"
                assert model.cost_output_per_1m is not None, \
                    f"Stable production model {model.model_id} missing output pricing"

    def test_placeholder_models_have_tbd_pricing(self):
        """Placeholder models should have $TBD pricing display"""
        catalog = get_model_catalog()

        placeholders = catalog.list_models(status=ModelStatus.PLACEHOLDER)

        for model in placeholders:
            assert model.cost_display == "$TBD", \
                f"Placeholder model {model.model_id} should have '$TBD' cost display"
            assert model.cost_input_per_1m is None, \
                f"Placeholder model {model.model_id} should not have pricing yet"

    def test_recommended_models_have_quality_scores(self):
        """Recommended models should have quality scores"""
        catalog = get_model_catalog()

        recommended = catalog.get_recommended_models()

        # Most recommended models should have quality scores
        # (Allow some flexibility for newly added models)
        with_scores = [m for m in recommended if m.quality_score]
        assert len(with_scores) >= len(recommended) * 0.8, \
            "At least 80% of recommended models should have quality scores"

    def test_no_duplicate_model_ids(self):
        """Catalog should not contain duplicate model IDs"""
        catalog = get_model_catalog()

        model_ids = [m.model_id for m in catalog._registry]
        unique_ids = set(model_ids)

        assert len(model_ids) == len(unique_ids), \
            f"Found duplicate model IDs: {[id for id in model_ids if model_ids.count(id) > 1]}"

    def test_context_window_display_matches_value(self):
        """Context display should match actual context window"""
        catalog = get_model_catalog()

        for model in catalog._registry:
            # Extract numeric value from display (e.g., "128K" → 128000)
            if model.context_display.endswith("M"):
                expected = int(model.context_display[:-1]) * 1_000_000
            elif model.context_display.endswith("K"):
                expected = int(model.context_display[:-1]) * 1_000
            else:
                pytest.fail(f"Invalid context_display format: {model.context_display}")

            assert model.context_window == expected, \
                f"Model {model.model_id} context mismatch: {model.context_window} != {expected}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
