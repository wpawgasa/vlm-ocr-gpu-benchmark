"""Markdown report generator — produces structured benchmark reports."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from vlm_ocr_bench.analysis.aggregator import AggregatedConfig, AggregatedMetric

logger = structlog.get_logger()


@dataclass
class ReportSection:
    """A single section of the report."""

    title: str
    content: str
    level: int = 2  # heading level


@dataclass
class ReportData:
    """All data needed to generate a report."""

    experiment_name: str = ""
    description: str = ""
    gpu_types: list[str] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)
    inference_results: list[AggregatedConfig] = field(default_factory=list)
    training_results: list[AggregatedConfig] = field(default_factory=list)
    quality_results: list[dict[str, Any]] = field(default_factory=list)
    tco_results: dict[str, Any] = field(default_factory=dict)
    plot_paths: dict[str, str] = field(default_factory=dict)


class ReportGenerator:
    """Generates a Markdown benchmark report from aggregated results."""

    def __init__(self, data: ReportData, output_dir: Path | str = "results/reports") -> None:
        self._data = data
        self._output_dir = Path(output_dir)

    def generate(self, filename: str | None = None) -> Path:
        """Generate the full Markdown report.

        Args:
            filename: output filename (e.g. "report.md"). If None, a timestamped
                filename is used to avoid overwriting previous reports.

        Returns:
            Path to the generated report file.
        """
        sections = [
            self._executive_summary(),
            self._inference_results(),
            self._training_results(),
            self._quality_analysis(),
            self._power_efficiency(),
            self._tco_analysis(),
            self._recommendations(),
            self._appendix(),
        ]

        # Build full document
        lines: list[str] = []
        lines.append(f"# Benchmark Report: {self._data.experiment_name}")
        lines.append("")
        lines.append(f"*Generated: {datetime.datetime.now(tz=datetime.UTC):%Y-%m-%d %H:%M UTC}*")
        lines.append("")
        if self._data.description:
            lines.append(f"> {self._data.description}")
            lines.append("")

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for i, section in enumerate(sections, 1):
            anchor = section.title.lower().replace(" ", "-").replace("&", "").replace("  ", "-")
            lines.append(f"{i}. [{section.title}](#{anchor})")
        lines.append("")

        for section in sections:
            heading = "#" * section.level
            lines.append(f"{heading} {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        report_text = "\n".join(lines)

        # Write file
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"
        report_path = self._output_dir / filename
        report_path.write_text(report_text, encoding="utf-8")

        logger.info("report_generated", path=str(report_path), sections=len(sections))
        return report_path

    def _executive_summary(self) -> ReportSection:
        """Generate executive summary with headline metrics."""
        lines: list[str] = []

        gpus = ", ".join(self._data.gpu_types) if self._data.gpu_types else "N/A"
        models = ", ".join(self._data.model_names) if self._data.model_names else "N/A"
        lines.append(f"**GPUs tested:** {gpus}")
        lines.append(f"**Models evaluated:** {models}")
        lines.append("")

        # Headline throughput if available
        if self._data.inference_results:
            best = max(
                self._data.inference_results,
                key=lambda c: c.metrics.get("pages_per_second", _empty_metric()).mean,
            )
            pps = best.metrics.get("pages_per_second")
            if pps:
                lines.append(
                    f"**Best throughput:** {pps.mean:.1f} pages/s "
                    f"({best.model_name}, {best.gpu_type}, {best.precision})"
                )

        return ReportSection(title="Executive Summary", content="\n".join(lines))

    def _inference_results(self) -> ReportSection:
        """Generate inference results section with tables."""
        lines: list[str] = []

        if not self._data.inference_results:
            lines.append("No inference results available.")
            return ReportSection(title="Inference Results", content="\n".join(lines))

        # Throughput table
        lines.append("### Throughput Summary")
        lines.append("")
        lines.append(
            "| Model | GPU | Precision | Batch Size | Pages/s | Tokens/s | P50 Latency (ms) |"
        )
        lines.append(
            "|-------|-----|-----------|------------|---------|----------|-------------------|"
        )

        for cfg in sorted(
            self._data.inference_results,
            key=lambda c: c.metrics.get("pages_per_second", _empty_metric()).mean,
            reverse=True,
        ):
            pps = cfg.metrics.get("pages_per_second", _empty_metric())
            tps = cfg.metrics.get("tokens_per_second", _empty_metric())
            lat = cfg.metrics.get("e2e_latency_p50", _empty_metric())
            lines.append(
                f"| {cfg.model_name} | {cfg.gpu_type} | {cfg.precision} | "
                f"{cfg.batch_size} | {pps.mean:.1f} | {tps.mean:.0f} | {lat.mean:.1f} |"
            )

        # Include plot reference if available
        if "throughput_comparison" in self._data.plot_paths:
            lines.append("")
            plot_ref = self._data.plot_paths["throughput_comparison"]
            lines.append(f"![Throughput Comparison]({plot_ref})")

        return ReportSection(title="Inference Results", content="\n".join(lines))

    def _training_results(self) -> ReportSection:
        """Generate training results section."""
        lines: list[str] = []

        if not self._data.training_results:
            lines.append("No training results available.")
            return ReportSection(title="Training Results", content="\n".join(lines))

        lines.append("### Training Throughput Summary")
        lines.append("")
        lines.append("| Model | GPU | Precision | Batch Size | Samples/s | Peak Memory (GB) |")
        lines.append("|-------|-----|-----------|------------|-----------|-------------------|")

        for cfg in sorted(
            self._data.training_results,
            key=lambda c: c.metrics.get("samples_per_second", _empty_metric()).mean,
            reverse=True,
        ):
            sps = cfg.metrics.get("samples_per_second", _empty_metric())
            mem = cfg.metrics.get("peak_gpu_memory_gb", _empty_metric())
            lines.append(
                f"| {cfg.model_name} | {cfg.gpu_type} | {cfg.precision} | "
                f"{cfg.batch_size} | {sps.mean:.1f} | {mem.mean:.1f} |"
            )

        return ReportSection(title="Training Results", content="\n".join(lines))

    def _quality_analysis(self) -> ReportSection:
        """Generate quality analysis section."""
        lines: list[str] = []

        if not self._data.quality_results:
            lines.append("No quality evaluation results available.")
            return ReportSection(title="Quality Analysis", content="\n".join(lines))

        lines.append("### Quality Metrics by Precision")
        lines.append("")
        lines.append("| Model | GPU | Precision | Benchmark | Edit Distance | BLEU | METEOR |")
        lines.append("|-------|-----|-----------|-----------|---------------|------|--------|")

        for qr in self._data.quality_results:
            lines.append(
                f"| {qr.get('model_name', '')} | {qr.get('gpu_type', '')} | "
                f"{qr.get('precision', '')} | {qr.get('benchmark', '')} | "
                f"{qr.get('edit_distance', 0):.3f} | {qr.get('bleu', 0):.3f} | "
                f"{qr.get('meteor', 0):.3f} |"
            )

        return ReportSection(title="Quality Analysis", content="\n".join(lines))

    def _power_efficiency(self) -> ReportSection:
        """Generate power efficiency section."""
        lines: list[str] = []

        power_configs = [
            c
            for c in self._data.inference_results
            if c.metrics.get("pages_per_watt_hour", _empty_metric()).mean > 0
        ]

        if not power_configs:
            lines.append("No power efficiency data available.")
            return ReportSection(title="Power Efficiency", content="\n".join(lines))

        lines.append("### Power Efficiency Comparison")
        lines.append("")
        lines.append("| Model | GPU | Precision | Pages/Wh | Energy/Page (mWh) | Mean Power (W) |")
        lines.append("|-------|-----|-----------|----------|-------------------|----------------|")

        for cfg in power_configs:
            ppwh = cfg.metrics.get("pages_per_watt_hour", _empty_metric())
            epw = cfg.metrics.get("energy_per_page_wh", _empty_metric())
            power = cfg.metrics.get("mean_power_watts", _empty_metric())
            lines.append(
                f"| {cfg.model_name} | {cfg.gpu_type} | {cfg.precision} | "
                f"{ppwh.mean:.1f} | {epw.mean * 1000:.1f} | {power.mean:.0f} |"
            )

        return ReportSection(title="Power Efficiency", content="\n".join(lines))

    def _tco_analysis(self) -> ReportSection:
        """Generate TCO analysis section."""
        lines: list[str] = []

        if not self._data.tco_results:
            lines.append("No TCO analysis data available.")
            return ReportSection(title="TCO Analysis", content="\n".join(lines))

        lines.append("### Cost Comparison")
        lines.append("")
        lines.append("| Config | GPUs Required | Monthly Cost | Projected Total | Cost/1K Pages |")
        lines.append("|--------|---------------|--------------|-----------------|---------------|")

        for label, tco in self._data.tco_results.items():
            lines.append(
                f"| {label} | {tco.get('gpus_required', 0)} | "
                f"${tco.get('monthly_total', 0):,.0f} | "
                f"${tco.get('projected_total', tco.get('annual_total', 0)):,.0f} | "
                f"${tco.get('cost_per_1k_pages', 0):.3f} |"
            )

        return ReportSection(title="TCO Analysis", content="\n".join(lines))

    def _recommendations(self) -> ReportSection:
        """Generate recommendations section."""
        lines: list[str] = []
        lines.append("### Model x GPU Pairing Guide")
        lines.append("")
        lines.append("*Recommendations based on benchmark results:*")
        lines.append("")

        if self._data.inference_results:
            # Find best throughput config
            best_throughput = max(
                self._data.inference_results,
                key=lambda c: c.metrics.get("pages_per_second", _empty_metric()).mean,
            )
            pps = best_throughput.metrics.get("pages_per_second", _empty_metric())
            lines.append(
                f"- **Best throughput:** {best_throughput.model_name} on "
                f"{best_throughput.gpu_type} ({best_throughput.precision}) — "
                f"{pps.mean:.1f} pages/s"
            )

            # Find most memory-efficient
            valid_mem = [
                c
                for c in self._data.inference_results
                if c.metrics.get("peak_gpu_memory_gb", _empty_metric()).mean > 0
            ]
            if valid_mem:
                best_mem = min(
                    valid_mem,
                    key=lambda c: c.metrics.get("peak_gpu_memory_gb", _empty_metric()).mean,
                )
                mem = best_mem.metrics.get("peak_gpu_memory_gb", _empty_metric())
                lines.append(
                    f"- **Most memory-efficient:** {best_mem.model_name} — {mem.mean:.1f} GB peak"
                )
        else:
            lines.append("- No benchmark data available for recommendations.")

        return ReportSection(title="Recommendations", content="\n".join(lines))

    def _appendix(self) -> ReportSection:
        """Generate appendix with config and metadata."""
        lines: list[str] = []
        lines.append("### Experiment Configuration")
        lines.append("")
        lines.append(f"- **Experiment:** {self._data.experiment_name}")
        lines.append(f"- **GPUs:** {', '.join(self._data.gpu_types)}")
        lines.append(f"- **Models:** {', '.join(self._data.model_names)}")
        lines.append(f"- **Inference configs:** {len(self._data.inference_results)}")
        lines.append(f"- **Training configs:** {len(self._data.training_results)}")
        lines.append(f"- **Quality benchmarks:** {len(self._data.quality_results)}")

        return ReportSection(title="Appendix", content="\n".join(lines))


def _empty_metric() -> AggregatedMetric:
    """Return a stub metric with mean=0 for safe access."""
    return AggregatedMetric(name="empty")
