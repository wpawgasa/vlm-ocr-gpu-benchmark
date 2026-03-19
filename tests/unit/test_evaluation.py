"""Unit tests for the quality evaluation module (Phase C, no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    GPUType,
    ModelConfig,
    PrecisionMode,
    QualityConfig,
)
from vlm_ocr_bench.evaluation.benchmarks.olmocr_bench import (
    OlmOCRBenchResult,
    evaluate_olmocr_bench,
)
from vlm_ocr_bench.evaluation.benchmarks.omnidocbench import (
    DocTypeScore,
    ElementScore,
    OmniDocBenchResult,
    evaluate_omnidocbench,
    evaluate_sample,
)
from vlm_ocr_bench.evaluation.benchmarks.real5 import (
    Real5Result,
    ScenarioScore,
    evaluate_real5,
)
from vlm_ocr_bench.evaluation.metrics.bleu import compute_bleu
from vlm_ocr_bench.evaluation.metrics.cdm import compute_cdm
from vlm_ocr_bench.evaluation.metrics.edit_distance import (
    block_level_edit_distance,
    normalized_edit_distance,
)
from vlm_ocr_bench.evaluation.metrics.meteor import compute_meteor
from vlm_ocr_bench.evaluation.metrics.structural import (
    formula_structural_accuracy,
    table_structural_accuracy,
)
from vlm_ocr_bench.evaluation.parsers import (
    BlockType,
    extract_formulas,
    extract_tables,
    parse_markdown,
)
from vlm_ocr_bench.evaluation.runner import (
    BenchmarkQuality,
    PrecisionComparison,
    PrecisionDelta,
    QualityEvalResult,
    QualityEvalRunner,
)

# ─── Fixtures ───


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        name="dots_ocr_1.5_3b",
        hf_model_id="rednote-hilab/dots.ocr-1.5",
        params_billion=3.0,
        tier="compact",
        adapter="DotsOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=8192,
    )


@pytest.fixture
def h100_gpu_config() -> GPUConfig:
    return GPUConfig(
        gpu_type=GPUType.H100_SXM,
        precision_modes=[PrecisionMode.BF16, PrecisionMode.FP8],
    )


@pytest.fixture
def quality_config() -> QualityConfig:
    return QualityConfig(
        benchmarks=["omnidocbench_v1.5"],
        metrics=["edit_distance", "bleu", "meteor"],
        precision_modes=[PrecisionMode.BF16],
        max_samples=10,
    )


# Helper to create DocSample-like objects
class _FakeSample:
    def __init__(
        self,
        sample_id: str,
        doc_type: str = "academic",
        language: str = "en",
        attributes: dict[str, object] | None = None,
    ):
        self.sample_id = sample_id
        self.doc_type = doc_type
        self.language = language
        self.image_path = Path("/tmp/fake.png")
        self.attributes = attributes or {}


# ─── Edit Distance Tests ───


class TestNormalizedEditDistance:
    def test_identical(self) -> None:
        assert normalized_edit_distance("hello", "hello") == 1.0

    def test_completely_different(self) -> None:
        score = normalized_edit_distance("abc", "xyz")
        assert score < 0.5

    def test_empty_strings(self) -> None:
        assert normalized_edit_distance("", "") == 1.0

    def test_one_empty(self) -> None:
        assert normalized_edit_distance("hello", "") == 0.0

    def test_unicode_normalization(self) -> None:
        # NFKC normalizes ﬁ → fi
        score = normalized_edit_distance("ﬁle", "file")
        assert score == 1.0

    def test_partial_match(self) -> None:
        score = normalized_edit_distance("hello world", "hello")
        assert 0.0 < score < 1.0

    def test_score_range(self) -> None:
        score = normalized_edit_distance("abc", "abd")
        assert 0.0 <= score <= 1.0


class TestBlockLevelEditDistance:
    def test_identical_blocks(self) -> None:
        blocks = ["paragraph one", "paragraph two"]
        assert block_level_edit_distance(blocks, blocks) == 1.0

    def test_empty_blocks(self) -> None:
        assert block_level_edit_distance([], []) == 1.0

    def test_one_empty(self) -> None:
        assert block_level_edit_distance(["hello"], []) == 0.0
        assert block_level_edit_distance([], ["hello"]) == 0.0

    def test_reordered_blocks(self) -> None:
        pred = ["block B", "block A"]
        ref = ["block A", "block B"]
        score = block_level_edit_distance(pred, ref)
        # Hungarian matching should find optimal assignment
        assert score == 1.0

    def test_partial_match(self) -> None:
        pred = ["hello world", "foo bar"]
        ref = ["hello world", "baz qux"]
        score = block_level_edit_distance(pred, ref)
        assert 0.0 < score < 1.0

    def test_unmatched_blocks_penalized(self) -> None:
        pred = ["hello"]
        ref = ["hello", "world", "extra"]
        score = block_level_edit_distance(pred, ref)
        assert score < 1.0

    def test_greedy_strategy(self) -> None:
        pred = ["a", "b"]
        ref = ["a", "b"]
        score = block_level_edit_distance(pred, ref, match_strategy="greedy")
        assert score == 1.0

    def test_unknown_strategy_falls_back_to_greedy(self) -> None:
        # Unknown strategy falls back to greedy (match in order)
        pred = ["a", "b"]
        ref = ["a", "b"]
        score = block_level_edit_distance(pred, ref, match_strategy="unknown_strategy")
        assert score == 1.0


# ─── BLEU Tests ───


class TestComputeBleu:
    def test_identical(self) -> None:
        score = compute_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_completely_different(self) -> None:
        # Use longer texts with clearly zero n-gram overlap
        pred = "the quick brown fox jumps over the lazy dog near the river bank"
        ref = "colorful butterflies migrate southward during autumn seasonal change"
        score = compute_bleu(pred, ref)
        assert score < 0.1

    def test_empty_prediction(self) -> None:
        assert compute_bleu("", "reference text") == 0.0

    def test_empty_reference(self) -> None:
        assert compute_bleu("prediction text", "") == 0.0

    def test_score_range(self) -> None:
        score = compute_bleu("the quick brown fox", "the slow brown fox")
        assert 0.0 <= score <= 1.0

    def test_custom_ngram_order(self) -> None:
        # Unigram BLEU should be > 4-gram BLEU for short texts with partial overlap
        text_a = "the cat"
        text_b = "the dog"
        score_1gram = compute_bleu(text_a, text_b, n_gram=1)
        score_4gram = compute_bleu(text_a, text_b, n_gram=4)
        # 1-gram BLEU counts "the" as a match; 4-gram BLEU finds fewer matches
        assert score_1gram >= score_4gram


# ─── METEOR Tests ───


class TestComputeMeteor:
    def test_identical(self) -> None:
        score = compute_meteor("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_completely_different(self) -> None:
        score = compute_meteor("alpha beta gamma", "one two three")
        assert score < 0.3

    def test_empty_prediction(self) -> None:
        assert compute_meteor("", "reference text") == 0.0

    def test_empty_reference(self) -> None:
        assert compute_meteor("prediction text", "") == 0.0

    def test_score_range(self) -> None:
        score = compute_meteor("the quick brown fox", "the slow brown fox")
        assert 0.0 <= score <= 1.0


# ─── CDM Tests ───


class TestComputeCdm:
    """Tests for CDM (token-level F1 fallback for LaTeX formula comparison).

    Note: compute_cdm is a token-level F1 approximation, NOT the actual CDM
    metric from the OmniDocBench paper. These tests verify internal correctness
    of the tokenizer and F1 computation via the public API.
    """

    def test_identical(self) -> None:
        assert compute_cdm("x^2 + y^2", "x^2 + y^2") == 1.0

    def test_different(self) -> None:
        score = compute_cdm("x^2", "\\frac{a}{b}")
        assert score < 1.0

    def test_empty(self) -> None:
        assert compute_cdm("", "") == 1.0

    def test_one_empty(self) -> None:
        assert compute_cdm("x^2", "") == 0.0
        assert compute_cdm("", "x^2") == 0.0

    def test_latex_commands_tokenized(self) -> None:
        # \\frac{a}{b} should tokenize to multiple tokens including \\frac
        score_full = compute_cdm("\\frac{a}{b}", "\\frac{a}{b}")
        assert score_full == 1.0

    def test_numbers_tokenized(self) -> None:
        score = compute_cdm("3.14", "3.14")
        assert score == 1.0

    def test_partial_overlap(self) -> None:
        # x^2 + y^2 vs x^2 — partial token overlap
        score = compute_cdm("x^2 + y^2", "x^2")
        assert 0.0 < score < 1.0

    def test_score_range(self) -> None:
        score = compute_cdm("a + b = c", "x - y = z")
        assert 0.0 <= score <= 1.0


# ─── Structural Tests ───


class TestTableStructuralAccuracy:
    """Tests for table_structural_accuracy.

    Also validates _parse_markdown_table behavior indirectly:
    - basic table parsing (header + data rows, separator skipped)
    - empty and non-table inputs
    """

    def test_identical_tables(self) -> None:
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = table_structural_accuracy(table, table)
        assert result["row_count_match"] == 1.0
        assert result["col_count_match"] == 1.0
        assert result["cell_content_accuracy"] == 1.0

    def test_different_tables(self) -> None:
        pred = "| A | B |\n|---|---|\n| 1 | 2 |"
        ref = "| X | Y | Z |\n|---|---|---|\n| 7 | 8 | 9 |"
        result = table_structural_accuracy(pred, ref)
        assert result["col_count_match"] == 0.0
        assert result["cell_content_accuracy"] < 1.0

    def test_empty_tables(self) -> None:
        result = table_structural_accuracy("", "")
        assert result["row_count_match"] == 1.0

    def test_one_empty(self) -> None:
        result = table_structural_accuracy("| A |", "")
        assert result["structure_f1"] == 0.0

    def test_table_with_header_separator_and_data(self) -> None:
        # Verify separator row is skipped (3 meaningful rows: header + 2 data)
        table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        result = table_structural_accuracy(table, table)
        assert result["row_count_match"] == 1.0
        assert result["cell_content_accuracy"] == 1.0

    def test_table_format_param_accepted(self) -> None:
        # table_format param is accepted (not shadowing builtin 'format')
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = table_structural_accuracy(table, table, table_format="markdown")
        assert result["row_count_match"] == 1.0


class TestFormulaStructuralAccuracy:
    def test_identical(self) -> None:
        result = formula_structural_accuracy("x^2 + y^2 = z^2", "x^2 + y^2 = z^2")
        assert result["exact_match"] == 1.0
        assert result["normalized_edit_distance"] == 1.0
        assert result["cdm_score"] == 1.0

    def test_different(self) -> None:
        result = formula_structural_accuracy("x^2", "\\frac{a}{b}")
        assert result["exact_match"] == 0.0
        assert result["normalized_edit_distance"] < 1.0

    def test_token_level_f1_equals_cdm_score(self) -> None:
        result = formula_structural_accuracy("x + y", "x - y")
        assert result["token_level_f1"] == result["cdm_score"]


# ─── Parser Tests ───


class TestParseMarkdown:
    def test_heading(self) -> None:
        blocks = parse_markdown("# Title")
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.HEADING
        assert blocks[0].content == "Title"
        assert blocks[0].level == 1

    def test_h2(self) -> None:
        blocks = parse_markdown("## Subtitle")
        assert blocks[0].level == 2

    def test_paragraph(self) -> None:
        blocks = parse_markdown("This is a paragraph.")
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.PARAGRAPH

    def test_table(self) -> None:
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        blocks = parse_markdown(md)
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.TABLE

    def test_formula(self) -> None:
        md = "$$\nx^2 + y^2 = z^2\n$$"
        blocks = parse_markdown(md)
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.FORMULA
        assert "x^2" in blocks[0].content

    def test_inline_formula(self) -> None:
        md = "$$x^2 + y^2$$"
        blocks = parse_markdown(md)
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.FORMULA

    def test_code_block(self) -> None:
        md = "```python\nprint('hello')\n```"
        blocks = parse_markdown(md)
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.CODE

    def test_list(self) -> None:
        md = "- item 1\n- item 2\n- item 3"
        blocks = parse_markdown(md)
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.LIST

    def test_mixed_content(self) -> None:
        md = "# Title\n\nSome paragraph text.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n$$E=mc^2$$"
        blocks = parse_markdown(md)
        types = [b.type for b in blocks]
        assert BlockType.HEADING in types
        assert BlockType.PARAGRAPH in types
        assert BlockType.TABLE in types
        assert BlockType.FORMULA in types

    def test_empty(self) -> None:
        assert parse_markdown("") == []
        assert parse_markdown("\n\n\n") == []


class TestExtractTables:
    def test_extracts_tables(self) -> None:
        md = "# Title\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nSome text."
        tables = extract_tables(md)
        assert len(tables) == 1
        assert "| A | B |" in tables[0]

    def test_no_tables(self) -> None:
        assert extract_tables("Just text.") == []


class TestExtractFormulas:
    def test_extracts_formulas(self) -> None:
        md = "Text before.\n\n$$\nE = mc^2\n$$\n\nText after."
        formulas = extract_formulas(md)
        assert len(formulas) == 1
        assert "E = mc^2" in formulas[0]

    def test_no_formulas(self) -> None:
        assert extract_formulas("Just text.") == []


# ─── Benchmark Pipeline Tests ───


class TestEvaluateSample:
    def test_identical(self) -> None:
        text = "# Title\n\nSome paragraph with text content here."
        scores = evaluate_sample(text, text)
        assert scores["edit_distance"] == 1.0
        assert scores["bleu"] > 0.9
        assert scores["meteor"] > 0.9

    def test_empty_prediction(self) -> None:
        scores = evaluate_sample("", "Some reference text.")
        assert scores["edit_distance"] == 0.0
        assert scores["bleu"] == 0.0

    def test_hallucinated_table_penalized(self) -> None:
        # Prediction has table, reference does not — should penalize table_accuracy
        pred = "| A | B |\n|---|---|\n| 1 | 2 |"
        ref = "Some plain text without any tables."
        scores = evaluate_sample(pred, ref)
        assert scores["table_accuracy"] == 0.0

    def test_missing_table_penalized(self) -> None:
        # Reference has table, prediction does not — should penalize table_accuracy
        pred = "Some plain text without any tables."
        ref = "| A | B |\n|---|---|\n| 1 | 2 |"
        scores = evaluate_sample(pred, ref)
        assert scores["table_accuracy"] == 0.0

    def test_no_tables_in_either(self) -> None:
        # Neither has tables — table_accuracy should be 1.0
        pred = "Just some plain text."
        ref = "Just some plain text."
        scores = evaluate_sample(pred, ref)
        assert scores["table_accuracy"] == 1.0


class TestEvaluateOmniDocBench:
    def test_basic(self) -> None:
        predictions = {"s1": "hello world", "s2": "foo bar"}
        references = {"s1": "hello world", "s2": "foo bar"}
        samples = [_FakeSample("s1"), _FakeSample("s2")]

        result = evaluate_omnidocbench(predictions, references, samples)
        assert isinstance(result, OmniDocBenchResult)
        assert result.num_samples == 2
        assert result.num_failed == 0
        assert result.overall_edit_distance == 1.0

    def test_with_failed(self) -> None:
        predictions = {"s1": "", "s2": "foo bar"}
        references = {"s1": "hello world", "s2": "foo bar"}
        samples = [_FakeSample("s1"), _FakeSample("s2")]

        result = evaluate_omnidocbench(predictions, references, samples)
        assert result.num_failed == 1

    def test_per_doc_type(self) -> None:
        predictions = {"s1": "hello", "s2": "world"}
        references = {"s1": "hello", "s2": "world"}
        samples = [
            _FakeSample("s1", doc_type="academic"),
            _FakeSample("s2", doc_type="financial"),
        ]

        result = evaluate_omnidocbench(predictions, references, samples, per_document_type=True)
        assert "academic" in result.doc_type_scores
        assert "financial" in result.doc_type_scores

    def test_empty_predictions(self) -> None:
        result = evaluate_omnidocbench({}, {"s1": "hello"}, [_FakeSample("s1")])
        assert result.num_failed == 1


class TestEvaluateOlmOCRBench:
    def test_basic(self) -> None:
        predictions = {"s1": "hello world", "s2": "foo bar"}
        references = {"s1": "hello world", "s2": "foo bar"}

        result = evaluate_olmocr_bench(predictions, references)
        assert isinstance(result, OlmOCRBenchResult)
        assert result.num_samples == 2
        assert result.overall_edit_distance == 1.0

    def test_with_failed(self) -> None:
        result = evaluate_olmocr_bench({"s1": ""}, {"s1": "hello"})
        assert result.num_failed == 1

    def test_empty(self) -> None:
        result = evaluate_olmocr_bench({}, {})
        assert result.num_samples == 0

    def test_per_sample_scores_use_stable_id(self) -> None:
        # sample_id should be stored as a plain string, not a hash
        predictions = {"doc_001": "hello world"}
        references = {"doc_001": "hello world"}
        result = evaluate_olmocr_bench(predictions, references)
        assert len(result.per_sample_scores) == 1
        assert result.per_sample_scores[0]["sample_id"] == "doc_001"

    def test_failed_sample_has_stable_id(self) -> None:
        predictions = {"doc_001": ""}
        references = {"doc_001": "hello"}
        result = evaluate_olmocr_bench(predictions, references)
        assert result.per_sample_scores[0]["sample_id"] == "doc_001"


class TestEvaluateReal5:
    def test_basic(self) -> None:
        predictions = {"s1": "hello world"}
        references = {"s1": "hello world"}
        samples = [_FakeSample("s1", attributes={"scenario": "scanning"})]

        result = evaluate_real5(predictions, references, samples)
        assert isinstance(result, Real5Result)
        assert result.num_samples == 1
        assert result.overall_edit_distance == 1.0
        assert "scanning" in result.scenario_scores

    def test_multiple_scenarios(self) -> None:
        predictions = {"s1": "hello", "s2": "world"}
        references = {"s1": "hello", "s2": "world"}
        samples = [
            _FakeSample("s1", attributes={"scenario": "scanning"}),
            _FakeSample("s2", attributes={"scenario": "warping"}),
        ]

        result = evaluate_real5(predictions, references, samples)
        assert len(result.scenario_scores) == 2

    def test_with_failed(self) -> None:
        predictions = {"s1": ""}
        references = {"s1": "hello"}
        samples = [_FakeSample("s1", attributes={"scenario": "skew"})]

        result = evaluate_real5(predictions, references, samples)
        assert result.num_failed == 1
        assert result.scenario_scores["skew"].num_failed == 1


# ─── Runner Dataclass Tests ───


class TestPrecisionDelta:
    def test_defaults(self) -> None:
        d = PrecisionDelta(precision="fp8")
        assert d.baseline_precision == "bf16"
        assert d.edit_distance_delta == 0.0

    def test_with_values(self) -> None:
        d = PrecisionDelta(
            precision="fp4",
            edit_distance_delta=-0.02,
            bleu_delta=-0.03,
            meteor_delta=-0.01,
        )
        assert d.edit_distance_delta == -0.02


class TestPrecisionComparison:
    def test_defaults(self) -> None:
        c = PrecisionComparison(benchmark="omnidocbench_v1.5")
        assert c.baseline_precision == "bf16"
        assert c.deltas == []


class TestBenchmarkQuality:
    def test_defaults(self) -> None:
        bq = BenchmarkQuality(benchmark="omnidocbench_v1.5", precision="bf16")
        assert bq.edit_distance == 0.0
        assert bq.num_samples == 0
        assert bq.details is None

    def test_with_values(self) -> None:
        bq = BenchmarkQuality(
            benchmark="omnidocbench_v1.5",
            precision="bf16",
            edit_distance=0.85,
            bleu=0.72,
            meteor=0.78,
            num_samples=100,
            num_failed=5,
        )
        assert bq.edit_distance == 0.85
        assert bq.num_samples == 100


class TestQualityEvalResult:
    def test_defaults(self) -> None:
        r = QualityEvalResult(model_name="test", gpu_type="h100_sxm")
        assert r.benchmarks == []
        assert r.precision_comparisons == []
        assert r.total_wall_time_s == 0.0

    def test_with_benchmarks(self) -> None:
        r = QualityEvalResult(
            model_name="test",
            gpu_type="h100_sxm",
            benchmarks=[
                BenchmarkQuality(
                    benchmark="omnidocbench_v1.5",
                    precision="bf16",
                    edit_distance=0.85,
                ),
            ],
        )
        assert len(r.benchmarks) == 1


class TestQualityEvalRunner:
    def test_init(
        self,
        model_config: ModelConfig,
        h100_gpu_config: GPUConfig,
        quality_config: QualityConfig,
    ) -> None:
        runner = QualityEvalRunner(model_config, h100_gpu_config, quality_config)
        assert runner._model_config == model_config
        assert runner._gpu_config == h100_gpu_config
        assert runner._quality_config == quality_config

    def test_compute_precision_comparisons(
        self,
        model_config: ModelConfig,
        h100_gpu_config: GPUConfig,
        quality_config: QualityConfig,
    ) -> None:
        runner = QualityEvalRunner(model_config, h100_gpu_config, quality_config)

        benchmarks = [
            BenchmarkQuality(
                benchmark="omnidocbench_v1.5",
                precision="bf16",
                edit_distance=0.90,
                bleu=0.80,
                meteor=0.85,
            ),
            BenchmarkQuality(
                benchmark="omnidocbench_v1.5",
                precision="fp8",
                edit_distance=0.88,
                bleu=0.78,
                meteor=0.83,
            ),
        ]

        comparisons = runner._compute_precision_comparisons(benchmarks)
        assert len(comparisons) == 1
        assert comparisons[0].benchmark == "omnidocbench_v1.5"
        assert len(comparisons[0].deltas) == 1
        delta = comparisons[0].deltas[0]
        assert delta.precision == "fp8"
        assert abs(delta.edit_distance_delta - (-0.02)) < 1e-9
        assert abs(delta.bleu_delta - (-0.02)) < 1e-9

    def test_compute_precision_comparisons_no_baseline(
        self,
        model_config: ModelConfig,
        h100_gpu_config: GPUConfig,
        quality_config: QualityConfig,
    ) -> None:
        runner = QualityEvalRunner(model_config, h100_gpu_config, quality_config)

        benchmarks = [
            BenchmarkQuality(
                benchmark="omnidocbench_v1.5",
                precision="fp8",
                edit_distance=0.88,
            ),
        ]

        comparisons = runner._compute_precision_comparisons(benchmarks)
        # No bf16 baseline → no comparisons
        assert len(comparisons) == 0


# ─── Dataclass Result Tests ───


class TestDocTypeScore:
    def test_defaults(self) -> None:
        s = DocTypeScore(doc_type="academic")
        assert s.num_samples == 0
        assert s.num_failed == 0
        assert s.element_scores == {}


class TestElementScore:
    def test_defaults(self) -> None:
        s = ElementScore(element_type="text")
        assert s.count == 0
        assert s.edit_distance == 0.0


class TestScenarioScore:
    def test_defaults(self) -> None:
        s = ScenarioScore(scenario="scanning")
        assert s.num_samples == 0
        assert s.edit_distance == 0.0


class TestOlmOCRBenchResult:
    def test_defaults(self) -> None:
        r = OlmOCRBenchResult()
        assert r.num_samples == 0
        assert r.per_sample_scores == []


class TestReal5Result:
    def test_defaults(self) -> None:
        r = Real5Result()
        assert r.scenario_scores == {}


# ─── Module Export Tests ───


class TestModuleExports:
    def test_evaluation_module_imports(self) -> None:
        from vlm_ocr_bench.evaluation import (
            BenchmarkQuality,
            Block,
            BlockType,
            PrecisionComparison,
            PrecisionDelta,
            QualityEvalResult,
            QualityEvalRunner,
            block_level_edit_distance,
            compute_bleu,
            compute_cdm,
            compute_meteor,
            formula_structural_accuracy,
            normalized_edit_distance,
            parse_markdown,
            table_structural_accuracy,
        )

        assert QualityEvalRunner is not None
        assert QualityEvalResult is not None
        assert BenchmarkQuality is not None
        assert PrecisionComparison is not None
        assert PrecisionDelta is not None
        assert normalized_edit_distance is not None
        assert block_level_edit_distance is not None
        assert compute_bleu is not None
        assert compute_meteor is not None
        assert compute_cdm is not None
        assert table_structural_accuracy is not None
        assert formula_structural_accuracy is not None
        assert parse_markdown is not None
        assert Block is not None
        assert BlockType is not None

    def test_metrics_module_imports(self) -> None:
        from vlm_ocr_bench.evaluation.metrics import (
            compute_bleu,
            compute_meteor,
            normalized_edit_distance,
        )

        assert normalized_edit_distance is not None
        assert compute_bleu is not None
        assert compute_meteor is not None

    def test_benchmarks_module_imports(self) -> None:
        from vlm_ocr_bench.evaluation.benchmarks import (
            evaluate_olmocr_bench,
            evaluate_omnidocbench,
            evaluate_real5,
        )

        assert evaluate_omnidocbench is not None
        assert evaluate_olmocr_bench is not None
        assert evaluate_real5 is not None
