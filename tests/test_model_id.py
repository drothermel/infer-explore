"""Tests for the model_id extraction system."""

import pytest

from infer_explore.sources.model_id import extract_identity, ModelIdentity


# ---------------------------------------------------------------------------
# Claude family parser tests
# ---------------------------------------------------------------------------


class TestClaudeParser:
    """Test Claude model name parsing."""

    def test_sonnet_version_first(self):
        """AA-style: 'Claude 4.5 Sonnet (Non-reasoning)'"""
        r = extract_identity("Claude 4.5 Sonnet (Non-reasoning)")
        assert r.model_id == "anthropic/claude-sonnet-4.5"
        assert r.variant == "sonnet"
        assert r.version == "4.5"
        assert r.reasoning_mode == "non-reasoning"

    def test_sonnet_variant_first(self):
        """Vantage-style: 'Claude Sonnet 4.6'"""
        r = extract_identity("Claude Sonnet 4.6 (Non-reasoning, High Effort)")
        assert r.model_id == "anthropic/claude-sonnet-4.6"
        assert r.variant == "sonnet"
        assert r.version == "4.6"
        assert r.reasoning_mode == "non-reasoning"
        assert r.effort_level == "high"

    def test_opus_adaptive_max(self):
        r = extract_identity("Claude Opus 4.6 (Adaptive Reasoning, Max Effort)")
        assert r.model_id == "anthropic/claude-opus-4.6"
        assert r.reasoning_mode == "adaptive"
        assert r.effort_level == "max"

    def test_slug_form(self):
        """Bifrost-style: 'claude-sonnet-4-5'"""
        r = extract_identity("claude-sonnet-4-5")
        assert r.model_id == "anthropic/claude-sonnet-4.5"
        assert r.variant == "sonnet"
        assert r.version == "4.5"

    def test_claude_haiku_slug(self):
        r = extract_identity("claude-haiku-4-5")
        assert r.model_id == "anthropic/claude-haiku-4.5"

    def test_version_first_slug(self):
        """Bifrost-style: 'claude-3-7-sonnet'"""
        r = extract_identity("claude-3-7-sonnet")
        assert r.model_id == "anthropic/claude-sonnet-3.7"

    def test_old_claude_version(self):
        r = extract_identity("Claude 2.0")
        assert r.model_id == "anthropic/claude-2.0"
        assert r.variant == ""
        assert r.version == "2.0"

    def test_claude_instant(self):
        r = extract_identity("Claude Instant")
        assert r.model_id == "anthropic/claude-instant-1"
        assert r.variant == "instant"

    def test_date_tag_extraction(self):
        r = extract_identity("Claude 3.5 Sonnet (June '24)")
        assert r.model_id == "anthropic/claude-sonnet-3.5"
        assert r.date_tag is not None

    def test_configs_share_model_id(self):
        """Multiple reasoning configs of the same model should share model_id."""
        ids = set()
        for name in [
            "Claude Sonnet 4.6 (Non-reasoning, High Effort)",
            "Claude Sonnet 4.6 (Adaptive Reasoning, Max Effort)",
            "Claude Sonnet 4.6 (Non-reasoning, Low Effort)",
            "claude-sonnet-4-6",
        ]:
            ids.add(extract_identity(name).model_id)
        assert len(ids) == 1
        assert "anthropic/claude-sonnet-4.6" in ids

    def test_config_ids_differ(self):
        """Different configs should have different config_ids."""
        configs = set()
        for name in [
            "Claude Sonnet 4.6 (Non-reasoning, High Effort)",
            "Claude Sonnet 4.6 (Adaptive Reasoning, Max Effort)",
            "Claude Sonnet 4.6 (Non-reasoning, Low Effort)",
        ]:
            configs.add(extract_identity(name).config_id)
        assert len(configs) == 3


# ---------------------------------------------------------------------------
# GPT family parser tests
# ---------------------------------------------------------------------------


class TestGPTParser:
    """Test GPT model name parsing."""

    def test_gpt_base(self):
        r = extract_identity("GPT-5.4 (xhigh)")
        assert r.model_id == "openai/gpt-5.4"
        assert r.effort_level == "xhigh"

    def test_gpt_mini(self):
        """Mini is a DIFFERENT model_id."""
        r = extract_identity("GPT-5.4 mini (xhigh)")
        assert r.model_id == "openai/gpt-5.4-mini"
        assert r.effort_level == "xhigh"

    def test_gpt_nano(self):
        r = extract_identity("GPT-5.4 nano (medium)")
        assert r.model_id == "openai/gpt-5.4-nano"
        assert r.effort_level == "medium"

    def test_gpt_pro(self):
        r = extract_identity("GPT-5.4 Pro (xhigh)")
        assert r.model_id == "openai/gpt-5.4-pro"

    def test_gpt_codex(self):
        r = extract_identity("GPT-5 Codex (high)")
        assert r.model_id == "openai/gpt-5-codex"
        assert r.effort_level == "high"

    def test_gpt_4o(self):
        r = extract_identity("GPT-4o (Nov '24)")
        assert r.model_id == "openai/gpt-4o"

    def test_gpt_4o_mini(self):
        r = extract_identity("GPT-4o mini")
        assert r.model_id == "openai/gpt-4o-mini"

    def test_gpt_4o_slug(self):
        r = extract_identity("gpt-4o")
        assert r.model_id == "openai/gpt-4o"

    def test_gpt_4o_mini_slug(self):
        r = extract_identity("gpt-4o-mini")
        assert r.model_id == "openai/gpt-4o-mini"

    def test_gpt_35_turbo(self):
        r = extract_identity("GPT-3.5 Turbo")
        assert r.model_id == "openai/gpt-3.5-turbo"

    def test_gpt_non_reasoning(self):
        r = extract_identity("GPT-5.4 (Non-reasoning)")
        assert r.model_id == "openai/gpt-5.4"
        assert r.reasoning_mode == "non-reasoning"

    def test_gpt_slug_form(self):
        r = extract_identity("gpt-5.4")
        assert r.model_id == "openai/gpt-5.4"

    def test_gpt_configs_share_model_id(self):
        ids = set()
        for name in [
            "GPT-5.4 (xhigh)",
            "GPT-5.4 (Non-reasoning)",
            "gpt-5.4",
        ]:
            ids.add(extract_identity(name).model_id)
        assert len(ids) == 1

    def test_mini_and_base_different(self):
        base = extract_identity("GPT-5.4 (xhigh)").model_id
        mini = extract_identity("GPT-5.4 mini (xhigh)").model_id
        assert base != mini


# ---------------------------------------------------------------------------
# Gemini family parser tests
# ---------------------------------------------------------------------------


class TestGeminiParser:
    """Test Gemini model name parsing."""

    def test_gemini_flash(self):
        r = extract_identity("Gemini 2.5 Flash (Non-reasoning)")
        assert r.model_id == "google/gemini-flash-2.5"
        assert r.reasoning_mode == "non-reasoning"

    def test_gemini_pro(self):
        r = extract_identity("Gemini 2.5 Pro")
        assert r.model_id == "google/gemini-pro-2.5"

    def test_gemini_flash_lite(self):
        r = extract_identity("Gemini 2.0 Flash-Lite (Feb '25)")
        assert r.model_id == "google/gemini-flash-2.0-lite"

    def test_gemini_flash_8b(self):
        r = extract_identity("Gemini 1.5 Flash-8B")
        assert r.model_id == "google/gemini-flash-1.5-8b"

    def test_gemini_preview_same_model(self):
        """Preview vs non-preview of same version = same model_id."""
        non_preview = extract_identity("Gemini 2.5 Flash (Non-reasoning)").model_id
        preview = extract_identity("Gemini 2.5 Flash Preview (Non-reasoning)").model_id
        assert non_preview == preview

    def test_gemini_slug(self):
        r = extract_identity("gemini-2.5-flash")
        assert r.model_id == "google/gemini-flash-2.5"

    def test_gemini_flash_reasoning_configs(self):
        ids = set()
        for name in [
            "Gemini 2.5 Flash (Non-reasoning)",
            "Gemini 2.5 Flash (Reasoning)",
            "gemini-2.5-flash",
        ]:
            ids.add(extract_identity(name).model_id)
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# Llama family parser tests
# ---------------------------------------------------------------------------


class TestLlamaParser:
    """Test Llama model name parsing."""

    def test_llama_maverick(self):
        r = extract_identity("Llama 4 Maverick")
        assert r.model_id == "meta/llama-maverick-4"
        assert r.variant == "maverick"

    def test_llama_scout(self):
        r = extract_identity("Llama 4 Scout")
        assert r.model_id == "meta/llama-scout-4"
        assert r.variant == "scout"

    def test_llama_param_count(self):
        """Parameter counts create different model_ids."""
        r = extract_identity("Llama 3.3 Instruct 70B")
        assert r.model_id == "meta/llama-70b-3.3"
        assert r.variant == "70b"

    def test_llama_405b(self):
        r = extract_identity("Llama 3.1 Instruct 405B")
        assert r.model_id == "meta/llama-405b-3.1"

    def test_llama_slug(self):
        r = extract_identity("llama-4-maverick")
        assert r.model_id == "meta/llama-maverick-4"

    def test_different_sizes_different_ids(self):
        id_70b = extract_identity("Llama 3.1 Instruct 70B").model_id
        id_405b = extract_identity("Llama 3.1 Instruct 405B").model_id
        assert id_70b != id_405b


# ---------------------------------------------------------------------------
# Qwen family parser tests
# ---------------------------------------------------------------------------


class TestQwenParser:
    """Test Qwen model name parsing."""

    def test_qwen3_params(self):
        r = extract_identity("Qwen3 235B A22B (Reasoning)")
        assert r.model_id == "alibaba/qwen-235b-a22b-3"
        assert r.reasoning_mode == "reasoning"

    def test_qwen3_coder(self):
        r = extract_identity("Qwen3 Coder 480B A35B Instruct")
        assert r.model_id == "alibaba/qwen-480b-a35b-3-coder"

    def test_qwen35_params(self):
        r = extract_identity("Qwen3.5 27B (Non-reasoning)")
        assert r.model_id == "alibaba/qwen-27b-3.5"
        assert r.reasoning_mode == "non-reasoning"

    def test_qwen3_max(self):
        r = extract_identity("Qwen2.5 Max")
        assert r.model_id == "alibaba/qwen-2.5-max"

    def test_qwen3_thinking(self):
        """Thinking suffix = reasoning config."""
        r = extract_identity("Qwen3 Max Thinking")
        assert r.model_id == "alibaba/qwen-3-max"
        assert r.reasoning_mode == "reasoning"

    def test_qwen_slug(self):
        r = extract_identity("qwen3-235b-a22b")
        assert r.model_id == "alibaba/qwen-235b-a22b-3"

    def test_qwen_reasoning_configs_share_id(self):
        ids = set()
        for name in [
            "Qwen3 235B A22B (Non-reasoning)",
            "Qwen3 235B A22B (Reasoning)",
            "qwen3-235b-a22b",
        ]:
            ids.add(extract_identity(name).model_id)
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# DeepSeek family parser tests
# ---------------------------------------------------------------------------


class TestDeepSeekParser:
    """Test DeepSeek model name parsing."""

    def test_deepseek_v3(self):
        r = extract_identity("DeepSeek-V3")
        assert r.model_id == "deepseek/deepseek-v3"
        assert r.version == "v3"

    def test_deepseek_r1(self):
        r = extract_identity("DeepSeek R1 (Jan '25)")
        assert r.model_id == "deepseek/deepseek-r1"
        assert r.variant == "r1"

    def test_deepseek_v31_non_reasoning(self):
        r = extract_identity("DeepSeek V3.1 (Non-reasoning)")
        assert r.model_id == "deepseek/deepseek-v3.1"
        assert r.reasoning_mode == "non-reasoning"

    def test_deepseek_v31_terminus(self):
        r = extract_identity("DeepSeek V3.1 Terminus (Reasoning)")
        assert r.model_id == "deepseek/deepseek-v3.1-terminus"
        assert r.reasoning_mode == "reasoning"

    def test_deepseek_r1_v3_different(self):
        """R1 and V3 are different model families."""
        r1 = extract_identity("DeepSeek R1").model_id
        v3 = extract_identity("DeepSeek-V3").model_id
        assert r1 != v3

    def test_deepseek_reasoning_configs(self):
        ids = set()
        for name in [
            "DeepSeek V3.1 (Non-reasoning)",
            "DeepSeek V3.1 (Reasoning)",
        ]:
            ids.add(extract_identity(name).model_id)
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# Miscellaneous family tests
# ---------------------------------------------------------------------------


class TestMiscFamilies:
    """Test other family parsers."""

    def test_mistral_large(self):
        r = extract_identity("Mistral Large 3")
        assert r.model_id == "mistral/mistral-large-3"

    def test_mistral_small(self):
        r = extract_identity("Mistral Small 4 (Reasoning)")
        assert r.model_id == "mistral/mistral-small-4"
        assert r.reasoning_mode == "reasoning"

    def test_glm(self):
        r = extract_identity("GLM-4.6 (Non-reasoning)")
        assert r.model_id == "zai/glm-4.6"
        assert r.reasoning_mode == "non-reasoning"

    def test_grok(self):
        r = extract_identity("Grok 4 Fast (Reasoning)")
        assert r.model_id == "xai/grok-4-fast"
        assert r.reasoning_mode == "reasoning"

    def test_command(self):
        r = extract_identity("Command A", creator="Cohere")
        assert r.model_id == "cohere/command-a"


# ---------------------------------------------------------------------------
# Golden set test: many real names → fewer model_ids
# ---------------------------------------------------------------------------


# These are real name strings from the actual data that should collapse
# to a smaller set of model_ids
GOLDEN_SET = [
    # Claude Sonnet 4.6: 4 names → 1 model_id
    ("Claude Sonnet 4.6 (Non-reasoning, High Effort)", "anthropic/claude-sonnet-4.6"),
    ("Claude Sonnet 4.6 (Adaptive Reasoning, Max Effort)", "anthropic/claude-sonnet-4.6"),
    ("Claude Sonnet 4.6 (Non-reasoning, Low Effort)", "anthropic/claude-sonnet-4.6"),
    ("claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"),

    # Claude Opus 4.5: 3 names → 1 model_id
    ("Claude Opus 4.5 (Non-reasoning)", "anthropic/claude-opus-4.5"),
    ("Claude Opus 4.5 (Reasoning)", "anthropic/claude-opus-4.5"),
    ("claude-opus-4-5", "anthropic/claude-opus-4.5"),

    # Claude 4.5 Sonnet (AA word order): same as Bifrost
    ("Claude 4.5 Sonnet (Non-reasoning)", "anthropic/claude-sonnet-4.5"),
    ("Claude 4.5 Sonnet (Reasoning)", "anthropic/claude-sonnet-4.5"),
    ("claude-sonnet-4-5", "anthropic/claude-sonnet-4.5"),

    # GPT-5.4 family: separate model_ids for base, mini, nano, pro
    ("GPT-5.4 (xhigh)", "openai/gpt-5.4"),
    ("GPT-5.4 (Non-reasoning)", "openai/gpt-5.4"),
    ("gpt-5.4", "openai/gpt-5.4"),
    ("GPT-5.4 mini (xhigh)", "openai/gpt-5.4-mini"),
    ("GPT-5.4 mini (Non-Reasoning)", "openai/gpt-5.4-mini"),
    ("GPT-5.4 nano (medium)", "openai/gpt-5.4-nano"),
    ("GPT-5.4 Pro (xhigh)", "openai/gpt-5.4-pro"),

    # Gemini 2.5 Flash: preview and non-preview same model_id
    ("Gemini 2.5 Flash (Non-reasoning)", "google/gemini-flash-2.5"),
    ("Gemini 2.5 Flash (Reasoning)", "google/gemini-flash-2.5"),
    ("Gemini 2.5 Flash Preview (Non-reasoning)", "google/gemini-flash-2.5"),
    ("gemini-2.5-flash", "google/gemini-flash-2.5"),

    # Gemini Flash-Lite: different model
    ("Gemini 2.5 Flash-Lite (Non-reasoning)", "google/gemini-flash-2.5-lite"),

    # Llama 4: named variants
    ("Llama 4 Maverick", "meta/llama-maverick-4"),
    ("llama-4-maverick", "meta/llama-maverick-4"),
    ("Llama 4 Scout", "meta/llama-scout-4"),

    # Qwen3: param counts
    ("Qwen3 235B A22B (Reasoning)", "alibaba/qwen-235b-a22b-3"),
    ("Qwen3 235B A22B (Non-reasoning)", "alibaba/qwen-235b-a22b-3"),
    ("qwen3-235b-a22b", "alibaba/qwen-235b-a22b-3"),

    # DeepSeek: V3 configs
    ("DeepSeek V3.1 (Non-reasoning)", "deepseek/deepseek-v3.1"),
    ("DeepSeek V3.1 (Reasoning)", "deepseek/deepseek-v3.1"),

    # DeepSeek R1 distinct from V3
    ("DeepSeek R1", "deepseek/deepseek-r1"),
    ("deepseek-r1", "deepseek/deepseek-r1"),

    # GLM reasoning configs
    ("GLM-4.6 (Non-reasoning)", "zai/glm-4.6"),
    ("GLM-4.6 (Reasoning)", "zai/glm-4.6"),
    ("glm-4-6", "zai/glm-4.6"),
]


class TestGoldenSet:
    """Test the golden set: real names that should map to expected model_ids."""

    @pytest.mark.parametrize("name,expected_id", GOLDEN_SET)
    def test_golden_set(self, name, expected_id):
        result = extract_identity(name)
        assert result.model_id == expected_id, (
            f"Name '{name}' → model_id='{result.model_id}' "
            f"(expected '{expected_id}')"
        )

    def test_golden_set_dedup_count(self):
        """The golden set should collapse to ~15 unique model_ids."""
        unique_ids = set()
        for name, expected_id in GOLDEN_SET:
            result = extract_identity(name)
            unique_ids.add(result.model_id)
        # 37 name strings should collapse to ~15 unique model_ids
        assert len(unique_ids) <= 18
        assert len(unique_ids) >= 12


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Test that the full merge pipeline produces reasonable output."""

    def test_merge_produces_fewer_models(self):
        """The merge should produce fewer models than the old pipeline (was 1201)."""
        from pathlib import Path
        from infer_explore.helpers import get_data_dir

        data_dir = get_data_dir()
        merged_path = data_dir / "merged_models.json"

        if not merged_path.exists():
            pytest.skip("merged_models.json not found (run merge_all first)")

        import json
        data = json.load(open(merged_path))
        count = data.get("count", 0)

        # Should be significantly fewer than the old 1201
        assert count < 1000, f"Expected < 1000 models, got {count}"
        # But not too few — should still have most models
        assert count > 500, f"Expected > 500 models, got {count}"

    def test_merged_records_have_model_id(self):
        """All merged records should have model_id, family, and configurations."""
        from pathlib import Path
        from infer_explore.helpers import get_data_dir

        data_dir = get_data_dir()
        merged_path = data_dir / "merged_models.json"

        if not merged_path.exists():
            pytest.skip("merged_models.json not found")

        import json
        data = json.load(open(merged_path))
        models = data.get("models", [])

        for m in models:
            assert "model_id" in m, f"Missing model_id in {m.get('name')}"
            assert "family" in m, f"Missing family in {m.get('name')}"
            assert "configurations" in m, f"Missing configurations in {m.get('name')}"
            assert isinstance(m["configurations"], list)
