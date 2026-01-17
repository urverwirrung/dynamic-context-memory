#!/usr/bin/env python3
"""
Phase 1a Results Analysis

Load experiment results from JSON, generate summary statistics and visualizations.

Usage:
    # Analyze most recent results
    python analyze_results.py

    # Analyze specific results file
    python analyze_results.py results/phase1a_results_20260117_143022.json

    # Compare multiple runs
    python analyze_results.py results/*.json --compare
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: Path) -> Dict:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def find_latest_results(results_dir: Path) -> Optional[Path]:
    """Find the most recent results file."""
    files = list(results_dir.glob("phase1a_results_*.json"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert per-example results to DataFrame."""
    if not results.get("stage2_per_example"):
        return pd.DataFrame()

    df = pd.DataFrame(results["stage2_per_example"])
    df["example_idx"] = range(len(df))
    return df


def print_summary(results: Dict) -> None:
    """Print summary statistics to console."""
    config = results.get("config", {})
    baselines = results.get("baselines", {})
    stage1 = results.get("stage1")
    stage2 = results.get("stage2")

    print("=" * 70)
    print("PHASE 1a RESULTS SUMMARY")
    print("=" * 70)

    # Config
    print(f"\nConfiguration:")
    print(f"  Model: {config.get('model', 'N/A')}")
    print(f"  Examples: {config.get('num_examples', 'N/A')}")
    print(f"  K (content slots): {config.get('K', 'N/A')}")
    print(f"  Hidden dim (d): {config.get('d', 'N/A')}")
    print(f"  Steps: {config.get('num_steps', 'N/A')}")
    print(f"  Restarts: {config.get('num_restarts', 'N/A')}")
    print(f"  Learning rate: {config.get('lr', 'N/A')}")
    print(f"  Loss threshold: {config.get('loss_threshold', 'N/A'):.4f}")
    print(f"  Stages run: {config.get('stages_run', 'N/A')}")

    # Baselines
    if baselines:
        print(f"\nBaselines:")
        print(f"  Random embeddings (Q): {baselines.get('random_q_mean', 0):.4f}")
        print(f"  Random embeddings (A): {baselines.get('random_a_mean', 0):.4f}")
        print(f"  Random combined: {baselines.get('random_combined_mean', 0):.4f}")
        print(f"  Full context L(A|Q): {baselines.get('full_context_mean', 0):.4f}")

    # Stage 1
    if stage1:
        print(f"\nStage 1 (Single-Target Diagnostic):")
        print(f"  Q convergence: {stage1.get('q_convergence_rate', 0):.1%}")
        print(f"  A convergence: {stage1.get('a_convergence_rate', 0):.1%}")
        print(f"  Combined: {stage1.get('combined_convergence_rate', 0):.1%}")

        threshold = 0.7
        status = "PASSED" if stage1.get('combined_convergence_rate', 0) >= threshold else "FAILED"
        print(f"  Status: {status} (threshold: {threshold:.0%})")

    # Stage 2
    if stage2:
        print(f"\nStage 2 (Joint Q-A Optimization):")
        print(f"  Convergence: {stage2.get('convergence_rate', 0):.1%} ({stage2.get('converged', 0)}/{stage2.get('total', 0)})")
        print(f"  Mean loss: {stage2.get('mean_loss', 0):.4f}")
        print(f"  Mean loss Q: {stage2.get('mean_loss_q', 0):.4f}")
        print(f"  Mean loss A: {stage2.get('mean_loss_a', 0):.4f}")
        print(f"  Loss ratio vs random: {stage2.get('mean_loss_ratio_vs_random', 0):.2f}x")
        print(f"  Mean restarts: {stage2.get('mean_restarts', 0):.1f}")

        conv_rate = stage2.get('convergence_rate', 0)
        if conv_rate >= 0.5:
            status = "PASSED (>50%)"
        elif conv_rate >= 0.3:
            status = "MARGINAL (30-50%)"
        else:
            status = "FAILED (<30%)"
        print(f"  Status: {status}")

    # Spot checks
    spot_checks = results.get("spot_checks")
    if spot_checks:
        print(f"\nStage 3 (Spot Checks): {len(spot_checks)} examples checked")
        for i, sc in enumerate(spot_checks[:3]):  # Show first 3
            print(f"\n  Example {sc.get('index', i)}:")
            print(f"    Target Q: {sc.get('target_q', '')[:60]}...")
            print(f"    Target A: {sc.get('target_a', '')}")
            print(f"    Gen A: {sc.get('generated_a', '')[:60]}")

    print("\n" + "=" * 70)


def plot_loss_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot distribution of final losses."""
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Combined loss histogram
    ax = axes[0]
    ax.hist(df["loss"], bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(df["loss"].median(), color="red", linestyle="--", label=f"Median: {df['loss'].median():.2f}")
    ax.set_xlabel("Combined Loss (Q + A)")
    ax.set_ylabel("Count")
    ax.set_title("Loss Distribution")
    ax.legend()

    # Q vs A loss scatter
    ax = axes[1]
    colors = ["green" if c else "red" for c in df["converged"]]
    ax.scatter(df["loss_q"], df["loss_a"], c=colors, alpha=0.6, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Q Loss")
    ax.set_ylabel("A Loss")
    ax.set_title("Q vs A Loss (green=converged)")
    ax.plot([0, df["loss_q"].max()], [0, df["loss_a"].max()], "k--", alpha=0.3)  # diagonal

    # Convergence by example
    ax = axes[2]
    ax.bar(df["example_idx"], df["converged"].astype(int), color="steelblue", alpha=0.7)
    ax.set_xlabel("Example Index")
    ax.set_ylabel("Converged (1=yes)")
    ax.set_title(f"Convergence by Example ({df['converged'].sum()}/{len(df)} = {df['converged'].mean():.1%})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_vs_restarts(df: pd.DataFrame, output_path: Path) -> None:
    """Plot relationship between loss and restarts needed."""
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss vs restarts
    ax = axes[0]
    ax.scatter(df["num_restarts_tried"], df["loss"], alpha=0.6, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Restarts Tried")
    ax.set_ylabel("Final Loss")
    ax.set_title("Loss vs Restarts")

    # Restarts distribution
    ax = axes[1]
    restart_counts = df["num_restarts_tried"].value_counts().sort_index()
    ax.bar(restart_counts.index, restart_counts.values, color="steelblue", alpha=0.7)
    ax.set_xlabel("Restarts Tried")
    ax.set_ylabel("Count")
    ax.set_title("Restarts Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(results_list: List[Dict], labels: List[str], output_path: Path) -> None:
    """Compare multiple experiment runs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Convergence rates
    ax = axes[0]
    conv_rates = []
    for r in results_list:
        s2 = r.get("stage2", {})
        conv_rates.append(s2.get("convergence_rate", 0) * 100)
    ax.bar(labels, conv_rates, color="steelblue", alpha=0.7)
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Stage 2 Convergence")
    ax.axhline(50, color="green", linestyle="--", alpha=0.5, label="50% threshold")
    ax.axhline(30, color="orange", linestyle="--", alpha=0.5, label="30% threshold")
    ax.legend()

    # Mean losses
    ax = axes[1]
    x = range(len(labels))
    width = 0.35
    q_losses = [r.get("stage2", {}).get("mean_loss_q", 0) for r in results_list]
    a_losses = [r.get("stage2", {}).get("mean_loss_a", 0) for r in results_list]
    ax.bar([i - width/2 for i in x], q_losses, width, label="Q Loss", alpha=0.7)
    ax.bar([i + width/2 for i in x], a_losses, width, label="A Loss", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Loss")
    ax.set_title("Q vs A Loss")
    ax.legend()

    # Loss ratio vs random
    ax = axes[2]
    ratios = [r.get("stage2", {}).get("mean_loss_ratio_vs_random", 1) for r in results_list]
    colors = ["green" if ratio < 0.3 else "orange" if ratio < 0.5 else "red" for ratio in ratios]
    ax.bar(labels, ratios, color=colors, alpha=0.7)
    ax.axhline(0.3, color="green", linestyle="--", alpha=0.5, label="0.3x (excellent)")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1.0x (no better than random)")
    ax.set_ylabel("Loss Ratio vs Random")
    ax.set_title("Compression Quality")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def analyze_single(results_path: Path, output_dir: Path) -> None:
    """Analyze a single results file."""
    results = load_results(results_path)

    # Print summary
    print_summary(results)

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    df = results_to_dataframe(results)
    if not df.empty:
        timestamp = results.get("timestamp", "unknown")
        plot_loss_distribution(df, plots_dir / f"loss_distribution_{timestamp}.png")
        plot_loss_vs_restarts(df, plots_dir / f"loss_vs_restarts_{timestamp}.png")

    # Save summary as CSV
    if not df.empty:
        csv_path = output_dir / f"phase1a_summary_{results.get('timestamp', 'unknown')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


def analyze_comparison(results_paths: List[Path], output_dir: Path) -> None:
    """Compare multiple results files."""
    results_list = [load_results(p) for p in results_paths]
    labels = [p.stem.replace("phase1a_results_", "") for p in results_paths]

    # Print comparison table
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    rows = []
    for path, results in zip(results_paths, results_list):
        config = results.get("config", {})
        stage2 = results.get("stage2", {})
        rows.append({
            "file": path.name,
            "model": config.get("model", "N/A").split("/")[-1],
            "K": config.get("K", "N/A"),
            "examples": config.get("num_examples", "N/A"),
            "conv_rate": f"{stage2.get('convergence_rate', 0):.1%}",
            "mean_loss": f"{stage2.get('mean_loss', 0):.3f}",
            "loss_ratio": f"{stage2.get('mean_loss_ratio_vs_random', 0):.2f}x",
        })

    df_compare = pd.DataFrame(rows)
    print(df_compare.to_string(index=False))

    # Generate comparison plot
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(results_list, labels, plots_dir / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1a experiment results")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Results file(s) to analyze. If empty, uses most recent.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple results files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for plots and summaries (default: results)",
    )
    args = parser.parse_args()

    # Find results files
    if not args.paths:
        latest = find_latest_results(args.output_dir)
        if latest is None:
            print("No results files found in", args.output_dir)
            sys.exit(1)
        args.paths = [latest]
        print(f"Using latest results: {latest}\n")

    # Validate paths
    for path in args.paths:
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)

    # Run analysis
    if args.compare and len(args.paths) > 1:
        analyze_comparison(args.paths, args.output_dir)
    else:
        for path in args.paths:
            analyze_single(path, args.output_dir)


if __name__ == "__main__":
    main()
