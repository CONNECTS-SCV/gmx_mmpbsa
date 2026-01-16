#!/usr/bin/env python3
"""
Aggregate Alanine Scanning Results
Combines results from multiple alanine scanning calculations
"""

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def parse_final_results(result_file):
    """Parse FINAL_RESULTS_MMPBSA.dat file for ΔΔG value"""
    if not result_file.exists():
        return None

    with open(result_file, 'r') as f:
        content = f.read()

    # Look for alanine scanning result
    # Pattern: "ΔΔH binding = value +/- error" or "ΔΔG binding = value +/- error"
    ddg_match = re.search(r'ΔΔ[HG]\s+binding\s*=\s*([-\d.]+)\s*\+/-\s*([-\d.]+)', content)

    if ddg_match:
        ddg = float(ddg_match.group(1))
        error = float(ddg_match.group(2))
        return {'ddg': ddg, 'error': error}

    # Alternative: Look for DELTA DELTA lines
    delta_match = re.search(r'DELTA DELTA.*?([-\d.]+)\s+\+/-\s+([-\d.]+)', content, re.DOTALL)
    if delta_match:
        ddg = float(delta_match.group(1))
        error = float(delta_match.group(2))
        return {'ddg': ddg, 'error': error}

    return None


def aggregate_results(residues, output_dir, working_dir=None):
    """Aggregate results from all residue directories"""
    results = []

    # Set base directory
    base_dir = Path(working_dir) if working_dir else Path.cwd()

    for residue in residues:
        # Convert residue name to directory name (A/50 -> A_50)
        residue_clean = residue.replace('/', '_')
        result_dir = base_dir / f"MMPBSA_{residue_clean}"

        if not result_dir.exists():
            print(f"[WARNING] Directory not found: {result_dir}")
            continue

        # Parse results
        result_file = result_dir / "FINAL_RESULTS_MMPBSA.dat"
        result_data = parse_final_results(result_file)

        if result_data:
            results.append({
                'Residue': residue,
                'Chain': residue.split('/')[0] if '/' in residue else '',
                'ResNum': residue.split('/')[1] if '/' in residue else residue,
                'ΔΔG (kcal/mol)': result_data['ddg'],
                'Error (kcal/mol)': result_data['error'],
                'Importance': classify_importance(result_data['ddg'])
            })
            print(f"[INFO] Parsed {residue}: ΔΔG = {result_data['ddg']:.2f} ± {result_data['error']:.2f}")
        else:
            print(f"[WARNING] Could not parse results for {residue}")

    return pd.DataFrame(results)


def classify_importance(ddg):
    """Classify residue importance based on ΔΔG"""
    if ddg > 2:
        return "Critical"
    elif ddg > 0.5:
        return "Important"
    elif ddg > -0.5:
        return "Neutral"
    else:
        return "Favorable"


def create_summary_plot(df, output_file):
    """Create summary visualization"""
    # Sort by ΔΔG (descending - most critical first)
    df_sorted = df.sort_values('ΔΔG (kcal/mol)', ascending=False)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(df) * 0.3)))
    fig.suptitle('Alanine Scanning Summary - All Residues', fontsize=16, fontweight='bold')

    # Color mapping
    color_map = {
        'Critical': '#e74c3c',
        'Important': '#f39c12',
        'Neutral': '#95a5a6',
        'Favorable': '#2ecc71'
    }
    colors = [color_map[imp] for imp in df_sorted['Importance']]

    # Subplot 1: Vertical bar chart
    x_pos = np.arange(len(df_sorted))
    bars = ax1.bar(x_pos, df_sorted['ΔΔG (kcal/mol)'],
                   yerr=df_sorted['Error (kcal/mol)'],
                   capsize=5, color=colors, edgecolor='black',
                   alpha=0.8, linewidth=1.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_sorted['Residue'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax1.set_title('Mutant Effect on Binding Free Energy', fontsize=13, fontweight='bold')

    # Set Y-axis limits based on data range with some padding
    ddg_values = df_sorted['ΔΔG (kcal/mol)'].values
    errors = df_sorted['Error (kcal/mol)'].values
    y_min = min(ddg_values - errors) - 0.5
    y_max = max(ddg_values + errors) + 0.5
    ax1.set_ylim(y_min, y_max)

    ax1.axhline(0, color='black', linewidth=2, linestyle='-')
    # Only show reference lines if they're within the data range
    if y_max > 2:
        ax1.axhline(2, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Critical')
    if y_max > 0.5:
        ax1.axhline(0.5, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Important')
    if y_min < -0.5:
        ax1.axhline(-0.5, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Favorable')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Horizontal bar chart (easier to read)
    y_pos = np.arange(len(df_sorted))
    ax2.barh(y_pos, df_sorted['ΔΔG (kcal/mol)'],
             xerr=df_sorted['Error (kcal/mol)'],
             capsize=3, color=colors, edgecolor='black',
             alpha=0.8, linewidth=1.5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_sorted['Residue'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax2.set_title('Ranked by Importance (Top = Most Critical)', fontsize=13, fontweight='bold')

    # Set X-axis limits based on data range with some padding
    x_min = min(ddg_values - errors) - 0.5
    x_max = max(ddg_values + errors) + 1.0  # Extra padding for labels
    ax2.set_xlim(x_min, x_max)

    ax2.axvline(0, color='black', linewidth=2, linestyle='-')
    # Only show reference lines if they're within the data range
    if x_max > 2:
        ax2.axvline(2, color='red', linewidth=1, linestyle='--', alpha=0.3)
    if x_max > 0.5:
        ax2.axvline(0.5, color='orange', linewidth=1, linestyle='--', alpha=0.3)
    if x_min < -0.5:
        ax2.axvline(-0.5, color='green', linewidth=1, linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (y, val, err) in enumerate(zip(y_pos, df_sorted['ΔΔG (kcal/mol)'], df_sorted['Error (kcal/mol)'])):
        label_x = val + err + 0.3 if val > 0 else val - err - 0.3
        ha = 'left' if val > 0 else 'right'
        ax2.text(label_x, y, f'{val:.1f}±{err:.1f}', va='center', ha=ha, fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Summary plot saved: {output_file}")


def create_text_report(df, output_file):
    """Create text summary report"""
    # Sort by ΔΔG (descending)
    df_sorted = df.sort_values('ΔΔG (kcal/mol)', ascending=False)

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Alanine Scanning - Combined Results Summary\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total residues analyzed: {len(df)}\n\n")

        # Importance summary
        importance_counts = df['Importance'].value_counts()
        f.write("Classification Summary:\n")
        f.write("-"*80 + "\n")
        for importance in ['Critical', 'Important', 'Neutral', 'Favorable']:
            count = importance_counts.get(importance, 0)
            f.write(f"  {importance:12s}: {count:3d} residues\n")

        f.write("\n" + "="*80 + "\n")
        f.write("All Residues (Ranked by ΔΔG):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6} {'Residue':<12} {'ΔΔG (kcal/mol)':<18} {'Error':<10} {'Importance':<12}\n")
        f.write("-"*80 + "\n")

        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            residue = row['Residue']
            ddg = row['ΔΔG (kcal/mol)']
            error = row['Error (kcal/mol)']
            importance = row['Importance']

            # Add emoji/symbol
            if importance == 'Critical':
                symbol = '⚠️ '
            elif importance == 'Important':
                symbol = '⚠️ '
            elif importance == 'Favorable':
                symbol = '✅'
            else:
                symbol = 'ℹ️ '

            f.write(f"{idx:<6} {residue:<12} {ddg:>8.2f} ± {error:<6.2f} {error:<10.2f} {symbol} {importance:<12}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Interpretation Guide:\n")
        f.write("-"*80 + "\n")
        f.write("ΔΔG = ΔG_mutant - ΔG_wildtype\n\n")
        f.write("⚠️  Critical (ΔΔG > 2):       Essential for binding, mutation severely weakens affinity\n")
        f.write("⚠️  Important (0.5 < ΔΔG ≤ 2): Contributes to binding, mutation moderately reduces affinity\n")
        f.write("ℹ️  Neutral (-0.5 ≤ ΔΔG ≤ 0.5): Minimal effect on binding\n")
        f.write("✅ Favorable (ΔΔG < -0.5):   Mutation improves binding (unusual)\n\n")

        f.write("Note: Positive ΔΔG = mutation weakens binding (loss of affinity)\n")
        f.write("      Negative ΔΔG = mutation strengthens binding (gain of affinity)\n")

    print(f"[INFO] Text report saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate alanine scanning results')
    parser.add_argument('--residues', nargs='+', required=True,
                       help='List of residues that were scanned')
    parser.add_argument('--output', default='MMPBSA_aggregated',
                       help='Output directory')
    parser.add_argument('--working-dir', default=None,
                       help='Working directory containing MMPBSA_* subdirectories')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Aggregating alanine scanning results...")
    print(f"[INFO] Residues: {', '.join(args.residues)}")
    if args.working_dir:
        print(f"[INFO] Working directory: {args.working_dir}")

    # Aggregate results
    df = aggregate_results(args.residues, output_dir, args.working_dir)

    if df.empty:
        print("[ERROR] No results found!")
        sys.exit(1)

    # Save combined CSV
    csv_file = output_dir / "alanine_scan_combined.csv"
    df.to_csv(csv_file, index=False)
    print(f"[INFO] Combined results saved: {csv_file}")

    # Create visualizations
    plot_file = output_dir / "alanine_scan_summary.png"
    create_summary_plot(df, plot_file)

    # Create text report
    report_file = output_dir / "alanine_scan_report.txt"
    create_text_report(df, report_file)

    print("\n[SUCCESS] Aggregation complete!")
    print(f"Results directory: {output_dir}")


if __name__ == '__main__':
    main()
