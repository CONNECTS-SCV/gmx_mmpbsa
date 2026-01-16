#!/usr/bin/env python3
"""
Energy Components Visualization Tool
Standalone script to create energy component plots from parsed results
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_energy_summary(results, output_file):
    """Plot summary: ΔG gas, ΔG solv, ΔG total"""
    if 'delta_g_gas' not in results or 'delta_g_solv' not in results:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))

    components = ['ΔG gas', 'ΔG solv', 'ΔG total']
    values = [
        results['delta_g_gas'],
        results['delta_g_solv'],
        results['delta_total']
    ]
    errors = [
        results['delta_g_gas_std'],
        results['delta_g_solv_std'],
        results['delta_total_std']
    ]

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    bars = ax.bar(components, values, yerr=errors, capsize=5, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
    ax.set_title('Binding Free Energy Summary', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Summary plot saved: {output_file}")
    return True


def plot_main_components(results, output_file):
    """Plot main energy components: EEL, EVDW, EGB, ESURF"""
    if 'delta_eel' not in results or 'delta_evdw' not in results:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    components = []
    values = []
    errors = []
    colors = []

    if 'delta_eel' in results:
        components.append('ΔEEL\n(Electrostatic)')
        values.append(results['delta_eel'])
        errors.append(results['delta_eel_std'])
        colors.append('#e74c3c')

    if 'delta_evdw' in results:
        components.append('ΔEVDW\n(van der Waals)')
        values.append(results['delta_evdw'])
        errors.append(results['delta_evdw_std'])
        colors.append('#e67e22')

    if 'delta_egb' in results:
        components.append('ΔEGB\n(GB Solvation)')
        values.append(results['delta_egb'])
        errors.append(results['delta_egb_std'])
        colors.append('#3498db')

    if 'delta_esurf' in results:
        components.append('ΔESURF\n(Non-polar)')
        values.append(results['delta_esurf'])
        errors.append(results['delta_esurf_std'])
        colors.append('#1abc9c')

    bars = ax.bar(components, values, yerr=errors, capsize=5, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
    ax.set_title('Main Energy Components', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Main components plot saved: {output_file}")
    return True


def plot_all_components(results, output_file):
    """Plot all energy components including internal terms"""
    
    fig, ax = plt.subplots(figsize=(14, 6))

    components = []
    values = []
    errors = []
    colors = []

    # Define all energy terms with labels and colors
    energy_terms = [
        ('delta_eel', 'ΔEEL', '#e74c3c'),
        ('delta_evdw', 'ΔVDW', '#e67e22'),
        ('delta_egb', 'ΔEGB', '#3498db'),
        ('delta_esurf', 'ΔSURF', '#1abc9c'),
        ('delta_bond', 'ΔBOND', '#95a5a6'),
        ('delta_angle', 'ΔANGLE', '#7f8c8d'),
        ('delta_dihed', 'ΔDIHED', '#bdc3c7'),
        ('delta_ub', 'ΔUB', '#95a5a6'),
        ('delta_imp', 'ΔIMP', '#7f8c8d'),
        ('delta_cmap', 'ΔCMAP', '#bdc3c7'),
        ('delta_vdw14', 'Δ1-4VDW', '#95a5a6'),
        ('delta_eel14', 'Δ1-4EEL', '#7f8c8d'),
    ]

    for key, label, color in energy_terms:
        if key in results:
            components.append(label)
            values.append(results[key])
            errors.append(results[key + '_std'])
            colors.append(color)

    if not components:
        return False

    bars = ax.bar(components, values, yerr=errors, capsize=4, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
    ax.set_title('All Energy Terms', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] All components plot saved: {output_file}")
    return True


def plot_combined_3panel(results, output_file):
    """Create combined 3-panel plot: Summary + Main + All"""
    
    if 'delta_g_gas' not in results or 'delta_g_solv' not in results:
        return False

    # Create three subplots
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel 1: Summary
    components1 = ['ΔG gas', 'ΔG solv', 'ΔG total']
    values1 = [results['delta_g_gas'], results['delta_g_solv'], results['delta_total']]
    errors1 = [results['delta_g_gas_std'], results['delta_g_solv_std'], results['delta_total_std']]
    colors1 = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    ax1.bar(components1, values1, yerr=errors1, capsize=5, color=colors1, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Energy (kcal/mol)', fontsize=11)
    ax1.set_title('Summary', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', labelsize=9)

    # Panel 2: Main components
    if 'delta_eel' in results and 'delta_evdw' in results:
        components2 = []
        values2 = []
        errors2 = []
        colors2 = []

        main_terms = [
            ('delta_eel', 'ΔEEL\n(elec)', '#e74c3c'),
            ('delta_evdw', 'ΔEVDW\n(vdW)', '#e67e22'),
            ('delta_egb', 'ΔEGB\n(GB)', '#3498db'),
            ('delta_esurf', 'ΔESURF\n(surf)', '#1abc9c'),
        ]

        for key, label, color in main_terms:
            if key in results:
                components2.append(label)
                values2.append(results[key])
                errors2.append(results[key + '_std'])
                colors2.append(color)

        ax2.bar(components2, values2, yerr=errors2, capsize=5, color=colors2, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_ylabel('Energy (kcal/mol)', fontsize=11)
        ax2.set_title('Main Components', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', labelsize=9)

    # Panel 3: All components
    components3 = []
    values3 = []
    errors3 = []
    colors3 = []

    all_terms = [
        ('delta_eel', 'ΔEEL', '#e74c3c'),
        ('delta_evdw', 'ΔVDW', '#e67e22'),
        ('delta_egb', 'ΔEGB', '#3498db'),
        ('delta_esurf', 'ΔSURF', '#1abc9c'),
        ('delta_bond', 'ΔBOND', '#95a5a6'),
        ('delta_angle', 'ΔANGLE', '#7f8c8d'),
        ('delta_dihed', 'ΔDIHED', '#bdc3c7'),
        ('delta_ub', 'ΔUB', '#95a5a6'),
        ('delta_imp', 'ΔIMP', '#7f8c8d'),
        ('delta_cmap', 'ΔCMAP', '#bdc3c7'),
        ('delta_vdw14', 'Δ1-4VDW', '#95a5a6'),
        ('delta_eel14', 'Δ1-4EEL', '#7f8c8d'),
    ]

    for key, label, color in all_terms:
        if key in results:
            components3.append(label)
            values3.append(results[key])
            errors3.append(results[key + '_std'])
            colors3.append(color)

    if components3:
        ax3.bar(components3, values3, yerr=errors3, capsize=4, color=colors3, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax3.set_ylabel('Energy (kcal/mol)', fontsize=11)
        ax3.set_title('All Energy Terms', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Combined 3-panel plot saved: {output_file}")
    return True


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Visualize energy components')
    parser.add_argument('-r', '--results', required=True, help='Results JSON file')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    parser.add_argument('-m', '--mode', default='combined', 
                       choices=['summary', 'main', 'all', 'combined'],
                       help='Plot mode')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots based on mode
    if args.mode == 'summary':
        plot_energy_summary(results, output_dir / 'energy_summary.png')
    elif args.mode == 'main':
        plot_main_components(results, output_dir / 'energy_main.png')
    elif args.mode == 'all':
        plot_all_components(results, output_dir / 'energy_all.png')
    elif args.mode == 'combined':
        plot_combined_3panel(results, output_dir / 'energy_components.png')
