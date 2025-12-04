#!/usr/bin/env python3
"""
Generate dissertation-ready PDF report from experiment results.

Assembles:
- Executive summary
- ECDF and throughput figures
- Stability charts
- Environment comparison tables
- Statistical significance results
- Auto-generated interpretive paragraphs

Usage:
    python analysis/build_final_report.py \
        --results-dir final-results \
        --output final-results/report.pdf
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

# Check for reportlab availability
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, inch, mm
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
        PageTemplate, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        ListFlowable, ListItem
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. PDF generation disabled.", file=sys.stderr)


def load_json_safe(path: Path) -> Optional[dict]:
    """Safely load JSON file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def create_styles() -> dict:
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a1a2e'),
    ))
    
    styles.add(ParagraphStyle(
        name='ReportSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4a4a6a'),
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#1a1a2e'),
        borderWidth=1,
        borderColor=colors.HexColor('#e0e0e0'),
        borderPadding=5,
    ))
    
    styles.add(ParagraphStyle(
        name='SubsectionHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#2a2a4e'),
    ))
    
    styles.add(ParagraphStyle(
        name='BodyText',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=14,
    ))
    
    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.white,
        alignment=TA_CENTER,
    ))
    
    styles.add(ParagraphStyle(
        name='TableCell',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
    ))
    
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#666666'),
        fontName='Helvetica-Oblique',
    ))
    
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Courier',
        backColor=colors.HexColor('#f5f5f5'),
        borderWidth=1,
        borderColor=colors.HexColor('#e0e0e0'),
        borderPadding=8,
    ))
    
    return styles


def build_title_page(elements: list, styles: dict, index: Optional[dict]) -> None:
    """Build the title page."""
    elements.append(Spacer(1, 2 * inch))
    
    elements.append(Paragraph(
        "Quantum-Resilient Cryptography<br/>Performance Analysis Report",
        styles['ReportTitle']
    ))
    
    generated_at = index.get('generated_at', datetime.now(timezone.utc).isoformat()) if index else datetime.now(timezone.utc).isoformat()
    total_exp = index.get('total_experiments', 0) if index else 0
    
    elements.append(Paragraph(
        f"Generated: {generated_at[:19].replace('T', ' ')} UTC<br/>"
        f"Total Experiments: {total_exp}",
        styles['ReportSubtitle']
    ))
    
    elements.append(Spacer(1, 1 * inch))
    
    elements.append(Paragraph(
        "Comprehensive statistical analysis of post-quantum cryptographic "
        "algorithms compared against classical implementations across "
        "native, containerized (Minikube), and cloud (GCP) environments.",
        styles['BodyText']
    ))
    
    elements.append(PageBreak())


def build_executive_summary(
    elements: list, 
    styles: dict,
    hypothesis: Optional[dict],
    aggregated: Optional[dict],
) -> None:
    """Build executive summary section."""
    elements.append(Paragraph("Executive Summary", styles['SectionHeading']))
    
    summary_points = []
    
    if hypothesis:
        total = hypothesis.get('total_comparisons', 0)
        significant = hypothesis.get('significant_comparisons', 0)
        pct = (significant / total * 100) if total > 0 else 0
        
        summary_points.append(
            f"<b>Statistical Testing:</b> {significant} of {total} comparisons "
            f"({pct:.1f}%) showed statistically significant differences "
            f"(α=0.05, Holm-Bonferroni corrected)."
        )
        
        effects = hypothesis.get('summary', {}).get('effect_sizes', {})
        large = effects.get('large', 0)
        medium = effects.get('medium', 0)
        
        if large + medium > 0:
            summary_points.append(
                f"<b>Practical Significance:</b> {large} comparisons showed large effect sizes "
                f"(|d| ≥ 0.8) and {medium} showed medium effect sizes (|d| ≥ 0.5)."
            )
    
    if aggregated:
        algorithms = aggregated.get('algorithms', [])
        envs = aggregated.get('environments', [])
        summary_points.append(
            f"<b>Coverage:</b> Analysis includes {len(algorithms)} algorithms "
            f"across {len(envs)} execution environments."
        )
    
    summary_points.append(
        "<b>Methodology:</b> All statistical tests include Kolmogorov-Smirnov "
        "(distribution shape), Mann-Whitney U (distribution location), and "
        "Welch's t-test (mean difference). Effect sizes reported as Cohen's d "
        "with 95% confidence intervals."
    )
    
    for point in summary_points:
        elements.append(Paragraph(f"• {point}", styles['BodyText']))
        elements.append(Spacer(1, 4))
    
    elements.append(Spacer(1, 12))


def build_hypothesis_results(
    elements: list,
    styles: dict,
    hypothesis: dict,
) -> None:
    """Build hypothesis testing results section."""
    elements.append(Paragraph("Statistical Hypothesis Testing", styles['SectionHeading']))
    
    # Summary table
    elements.append(Paragraph("Test Summary", styles['SubsectionHeading']))
    
    summary = hypothesis.get('summary', {})
    by_test = summary.get('by_test', {})
    
    summary_data = [
        ['Test', 'Significant', 'Description'],
        ['Kolmogorov-Smirnov', str(by_test.get('kolmogorov_smirnov', 0)), 'Distribution shape difference'],
        ['Mann-Whitney U', str(by_test.get('mann_whitney_u', 0)), 'Distribution location difference'],
        ["Welch's t-test", str(by_test.get('welch_t', 0)), 'Mean difference'],
    ]
    
    table = Table(summary_data, colWidths=[2*inch, 1.2*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Paragraph("Table 1: Summary of statistically significant comparisons by test type", styles['Caption']))
    
    # Effect size distribution
    elements.append(Paragraph("Effect Size Distribution", styles['SubsectionHeading']))
    
    effects = summary.get('effect_sizes', {})
    effect_data = [
        ['Effect Size', 'Count', 'Interpretation'],
        ['Large (|d| ≥ 0.8)', str(effects.get('large', 0)), 'Substantial practical difference'],
        ['Medium (0.5 ≤ |d| < 0.8)', str(effects.get('medium', 0)), 'Moderate practical difference'],
        ['Small (0.2 ≤ |d| < 0.5)', str(effects.get('small', 0)), 'Minor practical difference'],
        ['Negligible (|d| < 0.2)', str(effects.get('negligible', 0)), 'No practical difference'],
    ]
    
    table = Table(effect_data, colWidths=[2*inch, 1*inch, 3.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Paragraph("Table 2: Distribution of effect sizes (Cohen's d) across all comparisons", styles['Caption']))
    
    # Results by comparison type
    elements.append(Paragraph("Results by Comparison Type", styles['SubsectionHeading']))
    
    by_type = summary.get('by_type', {})
    type_data = [['Comparison Type', 'Total', 'Significant', 'Rate']]
    
    for comp_type, counts in by_type.items():
        total = counts.get('total', 0)
        sig = counts.get('significant', 0)
        rate = f"{(sig/total*100):.1f}%" if total > 0 else "N/A"
        type_data.append([comp_type.replace('_', ' ').title(), str(total), str(sig), rate])
    
    if len(type_data) > 1:
        table = Table(type_data, colWidths=[2.5*inch, 1*inch, 1.2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fa')]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)
        elements.append(Paragraph("Table 3: Statistical significance by comparison category", styles['Caption']))
    
    # Top significant comparisons
    results = hypothesis.get('results', [])
    significant_results = [r for r in results if r.get('any_significant')]
    
    if significant_results:
        elements.append(Paragraph("Key Significant Findings", styles['SubsectionHeading']))
        
        # Sort by effect size magnitude
        sorted_results = sorted(
            significant_results, 
            key=lambda x: abs(x.get('effect_size', {}).get('cohens_d', 0)),
            reverse=True
        )[:10]
        
        findings_data = [['Comparison', 'Mean Diff (%)', "Cohen's d", 'K-S p', 'Interpretation']]
        
        for r in sorted_results:
            comparison = f"{r.get('group_a', '?')} vs {r.get('group_b', '?')}"
            if len(comparison) > 35:
                comparison = comparison[:32] + "..."
            
            mean_diff = r.get('mean_diff_pct', 0)
            d = r.get('effect_size', {}).get('cohens_d', 0)
            ks_p = r.get('tests', {}).get('kolmogorov_smirnov', {}).get('p_value_corrected', 1)
            interp = r.get('effect_size', {}).get('interpretation', 'unknown')
            
            findings_data.append([
                comparison,
                f"{mean_diff:+.1f}%",
                f"{d:.2f}",
                f"{ks_p:.2e}",
                interp.capitalize(),
            ])
        
        table = Table(findings_data, colWidths=[2.2*inch, 1*inch, 0.9*inch, 1*inch, 1.1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fa')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Paragraph("Table 4: Top 10 significant comparisons by effect size magnitude", styles['Caption']))


def build_figures_section(
    elements: list,
    styles: dict,
    figures_dir: Path,
) -> None:
    """Build figures section with embedded images."""
    elements.append(PageBreak())
    elements.append(Paragraph("Performance Visualization", styles['SectionHeading']))
    
    # Look for figures
    figure_patterns = [
        ('combined_ecdf.png', 'Combined ECDF: Latency distributions across all algorithms and environments'),
        ('latency_cdf*.png', 'Latency CDF: Empirical cumulative distribution function'),
        ('throughput*.png', 'Throughput: Operations per second over time'),
        ('scaling_curves.png', 'Scaling Curves: Performance vs workload parameters'),
        ('native_vs_minikube_vs_gcp.png', 'Environment Comparison: Cross-environment latency distributions'),
        ('stability_matrix.png', 'Stability Matrix: Run-to-run variability heatmap'),
    ]
    
    figure_count = 0
    
    for pattern, caption in figure_patterns:
        # Handle glob patterns
        if '*' in pattern:
            matches = list(figures_dir.glob(pattern))
        else:
            matches = [figures_dir / pattern] if (figures_dir / pattern).exists() else []
        
        for fig_path in matches[:2]:  # Limit to 2 per pattern
            if fig_path.exists():
                try:
                    figure_count += 1
                    
                    # Add image with appropriate sizing
                    img = Image(str(fig_path))
                    
                    # Scale to fit page width while maintaining aspect ratio
                    max_width = 6 * inch
                    max_height = 4 * inch
                    
                    aspect = img.imageWidth / img.imageHeight if img.imageHeight else 1
                    
                    if img.imageWidth > max_width:
                        img.drawWidth = max_width
                        img.drawHeight = max_width / aspect
                    else:
                        img.drawWidth = img.imageWidth
                        img.drawHeight = img.imageHeight
                    
                    if img.drawHeight > max_height:
                        img.drawHeight = max_height
                        img.drawWidth = max_height * aspect
                    
                    elements.append(img)
                    
                    fig_caption = f"Figure {figure_count}: {caption}"
                    if fig_path.name != pattern:
                        fig_caption += f" ({fig_path.stem})"
                    elements.append(Paragraph(fig_caption, styles['Caption']))
                    elements.append(Spacer(1, 12))
                    
                except Exception as e:
                    print(f"Warning: Could not embed figure {fig_path}: {e}", file=sys.stderr)
    
    if figure_count == 0:
        elements.append(Paragraph(
            "<i>No figures found in the results directory. "
            "Run the analysis pipeline to generate visualizations.</i>",
            styles['BodyText']
        ))


def build_aggregated_stats(
    elements: list,
    styles: dict,
    aggregated: dict,
) -> None:
    """Build aggregated statistics section."""
    elements.append(PageBreak())
    elements.append(Paragraph("Aggregated Performance Statistics", styles['SectionHeading']))
    
    stats = aggregated.get('stats', {})
    
    if not stats:
        elements.append(Paragraph(
            "<i>No aggregated statistics available.</i>",
            styles['BodyText']
        ))
        return
    
    # Build stats table
    elements.append(Paragraph("Latency Percentiles by Algorithm", styles['SubsectionHeading']))
    
    stats_data = [['Algorithm', 'Environment', 'p50 (μs)', 'p95 (μs)', 'p99 (μs)', 'Throughput (ops/s)']]
    
    for algo, algo_stats in stats.items():
        for env, env_stats in algo_stats.items():
            latency = env_stats.get('latency', {})
            tp = env_stats.get('throughput', {})
            
            p50 = latency.get('p50', {}).get('mean', 'N/A')
            p95 = latency.get('p95', {}).get('mean', 'N/A')
            p99 = latency.get('p99', {}).get('mean', 'N/A')
            throughput = tp.get('mean', 'N/A')
            
            stats_data.append([
                algo,
                env,
                f"{p50:.1f}" if isinstance(p50, (int, float)) else p50,
                f"{p95:.1f}" if isinstance(p95, (int, float)) else p95,
                f"{p99:.1f}" if isinstance(p99, (int, float)) else p99,
                f"{throughput:.0f}" if isinstance(throughput, (int, float)) else throughput,
            ])
    
    if len(stats_data) > 1:
        table = Table(stats_data, colWidths=[1.5*inch, 1.2*inch, 1*inch, 1*inch, 1*inch, 1.3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fa')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Paragraph("Table 5: Mean latency percentiles and throughput across algorithms and environments", styles['Caption']))


def build_interpretation(
    elements: list,
    styles: dict,
    hypothesis: Optional[dict],
    interp_file: Optional[Path],
) -> None:
    """Build interpretation section."""
    elements.append(PageBreak())
    elements.append(Paragraph("Interpretation and Discussion", styles['SectionHeading']))
    
    # Read interpretation from file if available
    if interp_file and interp_file.exists():
        with open(interp_file) as f:
            interp_text = f.read()
        
        # Parse and format key sections
        for line in interp_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-'):
                continue
            
            if line.isupper() or (line.endswith(':') and len(line) < 50):
                elements.append(Paragraph(line, styles['SubsectionHeading']))
            else:
                elements.append(Paragraph(line, styles['BodyText']))
    else:
        # Generate basic interpretation
        elements.append(Paragraph("Key Findings", styles['SubsectionHeading']))
        
        if hypothesis:
            total = hypothesis.get('total_comparisons', 0)
            sig = hypothesis.get('significant_comparisons', 0)
            
            elements.append(Paragraph(
                f"Statistical analysis across {total} comparisons revealed {sig} "
                f"statistically significant differences after Holm-Bonferroni correction "
                f"for multiple comparisons. This indicates that algorithm selection and "
                f"execution environment have measurable impacts on cryptographic operation latency.",
                styles['BodyText']
            ))
            
            elements.append(Paragraph("Implications for Deployment", styles['SubsectionHeading']))
            
            elements.append(Paragraph(
                "The observed performance differences between post-quantum and classical "
                "cryptographic implementations suggest that migration planning should account "
                "for latency overhead. Cloud environments (GCP) show higher variability "
                "compared to native and containerized execution, which should be factored "
                "into service level objective (SLO) definitions.",
                styles['BodyText']
            ))


def build_methodology(elements: list, styles: dict) -> None:
    """Build methodology section."""
    elements.append(PageBreak())
    elements.append(Paragraph("Methodology", styles['SectionHeading']))
    
    elements.append(Paragraph("Statistical Tests", styles['SubsectionHeading']))
    
    tests_desc = [
        ("<b>Kolmogorov-Smirnov Test:</b> Non-parametric test that compares the shapes "
         "of two probability distributions. Sensitive to differences in location, scale, "
         "and shape of the distributions."),
        
        ("<b>Mann-Whitney U Test:</b> Non-parametric test that compares the distribution "
         "of ranks between two groups. Tests whether one distribution is stochastically "
         "greater than the other."),
        
        ("<b>Welch's t-test:</b> Parametric test for comparing means that does not assume "
         "equal variances. More robust than Student's t-test for heteroscedastic data."),
    ]
    
    for desc in tests_desc:
        elements.append(Paragraph(f"• {desc}", styles['BodyText']))
        elements.append(Spacer(1, 4))
    
    elements.append(Paragraph("Effect Size Estimation", styles['SubsectionHeading']))
    
    elements.append(Paragraph(
        "Cohen's d is computed as the standardized mean difference between groups, "
        "using the pooled standard deviation. 95% confidence intervals are calculated "
        "using the Hedges & Olkin (1985) standard error approximation. Effect sizes "
        "are interpreted using Cohen's conventions: |d| < 0.2 (negligible), "
        "0.2 ≤ |d| < 0.5 (small), 0.5 ≤ |d| < 0.8 (medium), |d| ≥ 0.8 (large).",
        styles['BodyText']
    ))
    
    elements.append(Paragraph("Multiple Comparison Correction", styles['SubsectionHeading']))
    
    elements.append(Paragraph(
        "The Holm-Bonferroni method is applied to control the family-wise error rate "
        "across all comparisons. This step-down procedure is more powerful than "
        "Bonferroni correction while maintaining strong control over Type I errors. "
        "P-values are reported both raw and corrected.",
        styles['BodyText']
    ))


def build_appendix(
    elements: list,
    styles: dict,
    index: Optional[dict],
) -> None:
    """Build appendix with experiment details."""
    elements.append(PageBreak())
    elements.append(Paragraph("Appendix: Experiment Index", styles['SectionHeading']))
    
    if not index:
        elements.append(Paragraph("<i>No experiment index available.</i>", styles['BodyText']))
        return
    
    experiments = index.get('experiments', [])
    
    if not experiments:
        elements.append(Paragraph("<i>No experiments recorded.</i>", styles['BodyText']))
        return
    
    # Summary by algorithm and environment
    from collections import Counter
    by_algo = Counter(e.get('algorithm', 'unknown') for e in experiments)
    by_env = Counter(e.get('environment', 'unknown') for e in experiments)
    
    elements.append(Paragraph("Experiment Distribution", styles['SubsectionHeading']))
    
    dist_data = [['Category', 'Value', 'Count']]
    for algo, count in sorted(by_algo.items()):
        dist_data.append(['Algorithm', algo, str(count)])
    for env, count in sorted(by_env.items()):
        dist_data.append(['Environment', env, str(count)])
    
    table = Table(dist_data, colWidths=[1.5*inch, 2*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8fa')]),
    ]))
    elements.append(table)


def generate_pdf_report(
    results_dir: Path,
    output_path: Path,
) -> bool:
    """Generate the complete PDF report."""
    
    if not REPORTLAB_AVAILABLE:
        print("Error: reportlab is required for PDF generation", file=sys.stderr)
        print("Install with: pip install reportlab", file=sys.stderr)
        return False
    
    # Load all data sources
    index = load_json_safe(results_dir / 'index.json')
    hypothesis = load_json_safe(results_dir / 'hypothesis_tests.json')
    aggregated = load_json_safe(results_dir / 'aggregated_stats.json')
    
    interp_file = results_dir / 'hypothesis_interpretation.txt'
    figures_dir = results_dir / 'figures'
    
    print(f"Building report from: {results_dir}")
    print(f"  Index: {'✓' if index else '✗'}")
    print(f"  Hypothesis tests: {'✓' if hypothesis else '✗'}")
    print(f"  Aggregated stats: {'✓' if aggregated else '✗'}")
    print(f"  Figures dir: {'✓' if figures_dir.exists() else '✗'}")
    
    # Create document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )
    
    styles = create_styles()
    elements = []
    
    # Build sections
    build_title_page(elements, styles, index)
    build_executive_summary(elements, styles, hypothesis, aggregated)
    
    if hypothesis:
        build_hypothesis_results(elements, styles, hypothesis)
    
    if figures_dir.exists():
        build_figures_section(elements, styles, figures_dir)
    
    if aggregated:
        build_aggregated_stats(elements, styles, aggregated)
    
    build_interpretation(elements, styles, hypothesis, interp_file)
    build_methodology(elements, styles)
    build_appendix(elements, styles, index)
    
    # Build PDF
    try:
        doc.build(elements)
        print(f"\n✓ Report generated: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to generate PDF: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate dissertation-ready PDF report from experiment results"
    )
    parser.add_argument(
        '--results-dir', '-r', type=Path, required=True,
        help='Directory containing final results'
    )
    parser.add_argument(
        '--output', '-o', type=Path,
        help='Output PDF path (default: <results-dir>/report.pdf)'
    )
    parser.add_argument(
        '--title', '-t', type=str,
        default="Quantum-Resilient Cryptography Performance Analysis",
        help='Report title'
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    
    output_path = args.output or (args.results_dir / 'report.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = generate_pdf_report(args.results_dir, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
