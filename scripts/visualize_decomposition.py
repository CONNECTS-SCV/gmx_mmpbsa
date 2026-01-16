#!/usr/bin/env python3
"""
Energy Decomposition Visualization Tool
Creates gmx_MMPBSA_ana style plots from decomposition results
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re


def detect_decomp_type(dat_file):
    """
    Detect decomposition type: per_residue or pairwise
    Returns: 'per_residue' or 'pairwise'
    """
    with open(dat_file, 'r') as f:
        lines = f.readlines()

    # Find header line
    for line in lines:
        # Check for pairwise format (two residue columns)
        if 'Resid 1,Resid 2' in line or 'Residue 1' in line or 'Residue1' in line:
            return 'pairwise'
        # Check for per-residue format (single residue column)
        elif 'Residue,Internal' in line or ('Residue' in line and 'Resid 1' not in line):
            return 'per_residue'

    # Default to per_residue if can't detect
    return 'per_residue'


def parse_decomp_dat(dat_file, decomp_type='per_residue'):
    """
    Parse FINAL_DECOMP_MMPBSA.dat file

    Args:
        dat_file: Path to decomposition file
        decomp_type: 'per_residue' or 'pairwise'

    Returns:
        DataFrame with decomposition data
    """
    print(f"[INFO] Parsing {dat_file} (type: {decomp_type})...")

    with open(dat_file, 'r') as f:
        lines = f.readlines()

    # Find data section
    data_start = None
    for i, line in enumerate(lines):
        # Look for header lines
        if 'Resid 1,Resid 2' in line or 'Residue,Internal' in line or ('Internal' in line and ('Residue' in line or 'Resid' in line)):
            data_start = i + 2  # Skip header and subheader
            break

    if data_start is None:
        raise ValueError("Could not find data section in file")

    if decomp_type == 'per_residue':
        # Parse per-residue format
        residues = []
        total_energies = []
        vdw_energies = []
        eel_energies = []

        for line in lines[data_start:]:
            if line.strip() == '' or line.startswith('-'):
                break

            parts = line.strip().split(',')
            if len(parts) < 17:
                continue

            try:
                residue = parts[0]
                vdw = float(parts[4])  # van der Waals Avg
                eel = float(parts[7])  # Electrostatic Avg
                total = float(parts[16])  # TOTAL Avg

                residues.append(residue)
                vdw_energies.append(vdw)
                eel_energies.append(eel)
                total_energies.append(total)
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame({
            'Residue': residues,
            'VDW': vdw_energies,
            'EEL': eel_energies,
            'Total': total_energies
        })

    else:  # pairwise
        # Parse pairwise format
        residue1_list = []
        residue2_list = []
        total_energies = []
        vdw_energies = []
        eel_energies = []

        for line in lines[data_start:]:
            if line.strip() == '' or line.startswith('-'):
                break

            parts = line.strip().split(',')
            if len(parts) < 18:  # Pairwise has one more column
                continue

            try:
                residue1 = parts[0]
                residue2 = parts[1]
                vdw = float(parts[5])  # van der Waals Avg (shifted by 1)
                eel = float(parts[8])  # Electrostatic Avg (shifted by 1)
                total = float(parts[17])  # TOTAL Avg (shifted by 1)

                residue1_list.append(residue1)
                residue2_list.append(residue2)
                vdw_energies.append(vdw)
                eel_energies.append(eel)
                total_energies.append(total)
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame({
            'Residue1': residue1_list,
            'Residue2': residue2_list,
            'VDW': vdw_energies,
            'EEL': eel_energies,
            'Total': total_energies
        })

    print(f"[INFO] Parsed {len(df)} {'residue pairs' if decomp_type == 'pairwise' else 'residues'}")
    return df


def plot_per_residue_decomp(df, output_file, top_n=20):
    """
    Plot per-residue energy decomposition (gmx_MMPBSA_ana style)
    Bar plot showing most significant residues
    """
    # Sort by absolute total energy
    df_sorted = df.reindex(df['Total'].abs().sort_values(ascending=False).index)
    df_top = df_sorted.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(df_top))
    width = 0.25

    # Plot bars
    bars1 = ax.bar(x - width, df_top['VDW'], width, label='van der Waals',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, df_top['EEL'], width, label='Electrostatic',
                   color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, df_top['Total'], width, label='Total',
                   color='#2ecc71', alpha=0.8)

    # Formatting
    ax.set_xlabel('Residue', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Contribution (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Residue Energy Contributions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_top['Residue'], rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Per-residue plot saved: {output_file}")


def plot_energy_heatmap(df, output_file):
    """
    Create heatmap of VDW and EEL contributions
    """
    # Select top 30 residues by absolute total energy
    df_sorted = df.reindex(df['Total'].abs().sort_values(ascending=False).index)
    df_top = df_sorted.head(30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Prepare data for heatmap
    vdw_data = df_top['VDW'].values.reshape(-1, 1)
    eel_data = df_top['EEL'].values.reshape(-1, 1)

    # VDW heatmap
    im1 = ax1.imshow(vdw_data, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
    ax1.set_yticks(range(len(df_top)))
    ax1.set_yticklabels(df_top['Residue'], fontsize=8)
    ax1.set_xticks([])
    ax1.set_title('van der Waals Contribution', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Energy (kcal/mol)')

    # EEL heatmap
    im2 = ax2.imshow(eel_data, cmap='RdBu_r', aspect='auto',
                     vmin=df_top['EEL'].min(), vmax=df_top['EEL'].max())
    ax2.set_yticks(range(len(df_top)))
    ax2.set_yticklabels(df_top['Residue'], fontsize=8)
    ax2.set_xticks([])
    ax2.set_title('Electrostatic Contribution', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Energy (kcal/mol)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Heatmap saved: {output_file}")


def plot_favorable_unfavorable(df, output_file, threshold=1.0):
    """
    Separate favorable (negative) and unfavorable (positive) contributions
    """
    df_favorable = df[df['Total'] < -threshold].sort_values('Total')
    df_unfavorable = df[df['Total'] > threshold].sort_values('Total', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Favorable (stabilizing) residues
    if len(df_favorable) > 0:
        top_favorable = df_favorable.head(15)
        ax1.barh(range(len(top_favorable)), top_favorable['Total'],
                 color='#2ecc71', alpha=0.8)
        ax1.set_yticks(range(len(top_favorable)))
        ax1.set_yticklabels(top_favorable['Residue'], fontsize=9)
        ax1.set_xlabel('Energy (kcal/mol)', fontsize=11, fontweight='bold')
        ax1.set_title('Most Favorable Residues (Stabilizing)', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)

    # Unfavorable (destabilizing) residues
    if len(df_unfavorable) > 0:
        top_unfavorable = df_unfavorable.head(15)
        ax2.barh(range(len(top_unfavorable)), top_unfavorable['Total'],
                 color='#e74c3c', alpha=0.8)
        ax2.set_yticks(range(len(top_unfavorable)))
        ax2.set_yticklabels(top_unfavorable['Residue'], fontsize=9)
        ax2.set_xlabel('Energy (kcal/mol)', fontsize=11, fontweight='bold')
        ax2.set_title('Most Unfavorable Residues (Destabilizing)', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Favorable/unfavorable plot saved: {output_file}")


def create_summary_table(df, output_file, top_n=20):
    """Create text summary table"""
    df_sorted = df.reindex(df['Total'].abs().sort_values(ascending=False).index)
    df_top = df_sorted.head(top_n)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Top {top_n} Residues by Energy Contribution\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Rank':<6}{'Residue':<20}{'VDW':>12}{'EEL':>12}{'Total':>12}\n")
        f.write("-" * 80 + "\n")

        for i, (idx, row) in enumerate(df_top.iterrows(), 1):
            f.write(f"{i:<6}{row['Residue']:<20}"
                   f"{row['VDW']:>12.2f}{row['EEL']:>12.2f}{row['Total']:>12.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total residues analyzed: {len(df)}\n")
        f.write(f"Most favorable (stabilizing): {df['Total'].min():.2f} kcal/mol ({df.loc[df['Total'].idxmin(), 'Residue']})\n")
        f.write(f"Most unfavorable (destabilizing): {df['Total'].max():.2f} kcal/mol ({df.loc[df['Total'].idxmax(), 'Residue']})\n")
        f.write(f"Mean total contribution: {df['Total'].mean():.2f} kcal/mol\n")
        f.write(f"Std dev: {df['Total'].std():.2f} kcal/mol\n")

    print(f"[SUCCESS] Summary table saved: {output_file}")


# ============================================================================
# Pairwise Decomposition Visualization Functions
# ============================================================================

def plot_pairwise_interaction_matrix(df, output_file, top_n=30):
    """
    Plot pairwise interaction matrix as heatmap (gmx_MMPBSA_ana style)
    """
    # Get unique residues
    all_residues = list(set(df['Residue1'].tolist() + df['Residue2'].tolist()))

    # Sort by total contribution
    residue_totals = {}
    for res in all_residues:
        total = df[df['Residue1'] == res]['Total'].sum() + df[df['Residue2'] == res]['Total'].sum()
        residue_totals[res] = abs(total)

    top_residues = sorted(residue_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_residue_names = [x[0] for x in top_residues]

    # Create matrix
    matrix = np.zeros((len(top_residue_names), len(top_residue_names)))
    for idx, row in df.iterrows():
        if row['Residue1'] in top_residue_names and row['Residue2'] in top_residue_names:
            i = top_residue_names.index(row['Residue1'])
            j = top_residue_names.index(row['Residue2'])
            matrix[i, j] = row['Total']
            matrix[j, i] = row['Total']  # Symmetric

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10)

    # Set ticks
    ax.set_xticks(range(len(top_residue_names)))
    ax.set_yticks(range(len(top_residue_names)))
    ax.set_xticklabels(top_residue_names, rotation=90, fontsize=8)
    ax.set_yticklabels(top_residue_names, fontsize=8)

    # Title
    ax.set_title(f'Pairwise Interaction Matrix (Top {top_n} Residues)',
                 fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Interaction Energy (kcal/mol)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Pairwise matrix plot saved: {output_file}")


def plot_top_pairwise_interactions(df, output_file, top_n=20):
    """
    Plot top pairwise interactions (strongest attractive and repulsive)
    """
    df_sorted = df.reindex(df['Total'].abs().sort_values(ascending=False).index)
    df_top = df_sorted.head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top attractive (negative)
    df_attractive = df[df['Total'] < 0].sort_values('Total').head(15)
    if len(df_attractive) > 0:
        labels = [f"{row['Residue1']}\n↔\n{row['Residue2']}"
                  for _, row in df_attractive.iterrows()]
        ax1.barh(range(len(df_attractive)), df_attractive['Total'],
                 color='#2ecc71', alpha=0.8)
        ax1.set_yticks(range(len(df_attractive)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel('Interaction Energy (kcal/mol)', fontsize=11, fontweight='bold')
        ax1.set_title('Strongest Attractive Interactions', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)

    # Top repulsive (positive)
    df_repulsive = df[df['Total'] > 0].sort_values('Total', ascending=False).head(15)
    if len(df_repulsive) > 0:
        labels = [f"{row['Residue1']}\n↔\n{row['Residue2']}"
                  for _, row in df_repulsive.iterrows()]
        ax2.barh(range(len(df_repulsive)), df_repulsive['Total'],
                 color='#e74c3c', alpha=0.8)
        ax2.set_yticks(range(len(df_repulsive)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel('Interaction Energy (kcal/mol)', fontsize=11, fontweight='bold')
        ax2.set_title('Strongest Repulsive Interactions', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Top pairwise interactions plot saved: {output_file}")


def plot_pairwise_network_graph(df, output_file, threshold=2.0, top_n=50):
    """
    Plot interaction network showing strong residue-residue connections
    """
    # Filter strong interactions
    df_strong = df[df['Total'].abs() > threshold].copy()
    df_strong = df_strong.reindex(df_strong['Total'].abs().sort_values(ascending=False).index).head(top_n)

    if len(df_strong) == 0:
        print(f"[WARNING] No interactions above threshold {threshold} kcal/mol")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique residues
    all_residues = list(set(df_strong['Residue1'].tolist() + df_strong['Residue2'].tolist()))
    n_residues = len(all_residues)

    # Create circular layout
    angles = np.linspace(0, 2 * np.pi, n_residues, endpoint=False)
    positions = {res: (np.cos(angle), np.sin(angle))
                 for res, angle in zip(all_residues, angles)}

    # Draw edges (interactions)
    for _, row in df_strong.iterrows():
        res1, res2 = row['Residue1'], row['Residue2']
        if res1 in positions and res2 in positions:
            x1, y1 = positions[res1]
            x2, y2 = positions[res2]

            # Color by interaction type
            color = '#2ecc71' if row['Total'] < 0 else '#e74c3c'
            alpha = min(abs(row['Total']) / 10, 1.0)
            linewidth = min(abs(row['Total']) / 2, 3)

            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                   linewidth=linewidth, zorder=1)

    # Draw nodes (residues)
    for res, (x, y) in positions.items():
        ax.scatter(x, y, s=300, c='#3498db', alpha=0.8,
                  edgecolors='white', linewidths=2, zorder=2)
        ax.text(x * 1.15, y * 1.15, res, fontsize=7, ha='center', va='center',
               fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=2, label='Attractive (< 0)'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label='Repulsive (> 0)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Interaction Network (|E| > {threshold} kcal/mol, Top {top_n})',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Network graph saved: {output_file}")


def create_pairwise_summary_table(df, output_file, top_n=30):
    """Create text summary for pairwise interactions"""
    df_sorted = df.reindex(df['Total'].abs().sort_values(ascending=False).index)
    df_top = df_sorted.head(top_n)

    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"Top {top_n} Pairwise Interactions by Energy Magnitude\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Rank':<6}{'Residue 1':<20}{'Residue 2':<20}{'VDW':>12}{'EEL':>12}{'Total':>12}\n")
        f.write("-" * 100 + "\n")

        for i, (idx, row) in enumerate(df_top.iterrows(), 1):
            f.write(f"{i:<6}{row['Residue1']:<20}{row['Residue2']:<20}"
                   f"{row['VDW']:>12.2f}{row['EEL']:>12.2f}{row['Total']:>12.2f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("Statistics:\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total residue pairs analyzed: {len(df)}\n")

        most_attractive = df.loc[df['Total'].idxmin()]
        most_repulsive = df.loc[df['Total'].idxmax()]

        f.write(f"Most attractive interaction: {most_attractive['Total']:.2f} kcal/mol "
               f"({most_attractive['Residue1']} ↔ {most_attractive['Residue2']})\n")
        f.write(f"Most repulsive interaction: {most_repulsive['Total']:.2f} kcal/mol "
               f"({most_repulsive['Residue1']} ↔ {most_repulsive['Residue2']})\n")
        f.write(f"Mean pairwise energy: {df['Total'].mean():.2f} kcal/mol\n")
        f.write(f"Std dev: {df['Total'].std():.2f} kcal/mol\n")

    print(f"[SUCCESS] Pairwise summary table saved: {output_file}")


def visualize_decomposition(input_dir, output_dir=None):
    """
    Main function to create all decomposition visualizations
    Automatically detects per-residue vs pairwise format
    """
    input_path = Path(input_dir)

    # Find decomposition file
    decomp_file = input_path / "FINAL_DECOMP_MMPBSA.dat"
    if not decomp_file.exists():
        raise FileNotFoundError(f"Could not find {decomp_file}")

    # Set output directory
    if output_dir is None:
        output_dir = input_path / "decomp_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("Energy Decomposition Visualization")
    print("=" * 80 + "\n")

    # Detect decomposition type
    decomp_type = detect_decomp_type(decomp_file)
    print(f"[INFO] Detected decomposition type: {decomp_type}")

    # Parse data
    df = parse_decomp_dat(decomp_file, decomp_type)

    # Create visualizations based on type
    if decomp_type == 'per_residue':
        print("\n[INFO] Generating per-residue visualizations (gmx_MMPBSA_ana style)...\n")

        plot_per_residue_decomp(df, output_dir / "01_per_residue_decomp.png", top_n=20)
        plot_energy_heatmap(df, output_dir / "02_energy_heatmap.png")
        plot_favorable_unfavorable(df, output_dir / "03_favorable_unfavorable.png")
        create_summary_table(df, output_dir / "04_summary_table.txt", top_n=30)

        # Save processed data as CSV
        df_sorted = df.sort_values('Total', key=abs, ascending=False)
        df_sorted.to_csv(output_dir / "decomp_data.csv", index=False)
        print(f"[SUCCESS] Processed data saved: {output_dir}/decomp_data.csv")

    else:  # pairwise
        print("\n[INFO] Generating pairwise visualizations (gmx_MMPBSA_ana style)...\n")

        plot_pairwise_interaction_matrix(df, output_dir / "01_pairwise_matrix.png", top_n=30)
        plot_top_pairwise_interactions(df, output_dir / "02_top_interactions.png", top_n=20)
        plot_pairwise_network_graph(df, output_dir / "03_interaction_network.png", threshold=2.0)
        create_pairwise_summary_table(df, output_dir / "04_pairwise_summary.txt", top_n=30)

        # Save processed data as CSV
        df_sorted = df.reindex(df['Total'].abs().sort_values(ascending=False).index)
        df_sorted.to_csv(output_dir / "pairwise_data.csv", index=False)
        print(f"[SUCCESS] Processed data saved: {output_dir}/pairwise_data.csv")

    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print(f"Decomposition type: {decomp_type}")
    print("=" * 80 + "\n")

    return output_dir


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_decomposition.py <input_directory> [output_directory]")
        print("\nExample:")
        print("  python visualize_decomposition.py example/6")
        print("  python visualize_decomposition.py example/6 my_analysis")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        visualize_decomposition(input_dir, output_dir)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
