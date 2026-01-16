#!/usr/bin/env python3
"""
gmx_MMPBSA Results Analyzer
Automatically analyze and visualize gmx_MMPBSA results based on analysis type
"""

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path


class MMPBSAAnalyzer:
    """Base analyzer for gmx_MMPBSA results"""

    def __init__(self, result_dir, analysis_type, folder_name=None):
        self.result_dir = Path(result_dir)
        self.analysis_type = analysis_type
        # Create analysis-specific folder
        if folder_name:
            self.output_dir = self.result_dir / "analysis_output" / folder_name
        else:
            self.output_dir = self.result_dir / "analysis_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_final_results(self):
        """Parse FINAL_RESULTS_MMPBSA.dat file"""
        dat_file = self.result_dir / "FINAL_RESULTS_MMPBSA.dat"
        if not dat_file.exists():
            print(f"[WARNING] {dat_file} not found!")
            return None

        with open(dat_file, 'r') as f:
            content = f.read()

        results = {}

        # Parse DELTA section (Complex - Receptor - Ligand)
        # ΔTOTAL pattern
        delta_total_match = re.search(r'ΔTOTAL\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if delta_total_match:
            results['delta_total'] = float(delta_total_match.group(1))  # Average
            results['delta_total_std'] = float(delta_total_match.group(3))  # SD
            results['delta_total_sem'] = float(delta_total_match.group(5))  # SEM

        # ΔGGAS pattern
        ggas_match = re.search(r'ΔGGAS\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if ggas_match:
            results['delta_g_gas'] = float(ggas_match.group(1))
            results['delta_g_gas_std'] = float(ggas_match.group(3))
            results['delta_g_gas_sem'] = float(ggas_match.group(5))

        # ΔGSOLV pattern
        gsolv_match = re.search(r'ΔGSOLV\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if gsolv_match:
            results['delta_g_solv'] = float(gsolv_match.group(1))
            results['delta_g_solv_std'] = float(gsolv_match.group(3))
            results['delta_g_solv_sem'] = float(gsolv_match.group(5))

        # Parse detailed energy components
        # ΔEEL (Electrostatic)
        eel_match = re.search(r'ΔEEL\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if eel_match:
            results['delta_eel'] = float(eel_match.group(1))
            results['delta_eel_std'] = float(eel_match.group(3))

        # ΔEVDW or ΔVDWAALS (van der Waals)
        evdw_match = re.search(r'Δ(?:EVDW|VDWAALS)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if evdw_match:
            results['delta_evdw'] = float(evdw_match.group(1))
            results['delta_evdw_std'] = float(evdw_match.group(3))

        # ΔEGB (GB solvation)
        egb_match = re.search(r'ΔEGB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if egb_match:
            results['delta_egb'] = float(egb_match.group(1))
            results['delta_egb_std'] = float(egb_match.group(3))

        # ΔESURF (Non-polar solvation)
        esurf_match = re.search(r'ΔESURF\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if esurf_match:
            results['delta_esurf'] = float(esurf_match.group(1))
            results['delta_esurf_std'] = float(esurf_match.group(3))

        # Parse additional internal energy terms (mostly ~0 for single trajectory)
        # ΔBOND
        bond_match = re.search(r'ΔBOND\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if bond_match:
            results['delta_bond'] = float(bond_match.group(1))
            results['delta_bond_std'] = float(bond_match.group(3))

        # ΔANGLE
        angle_match = re.search(r'ΔANGLE\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if angle_match:
            results['delta_angle'] = float(angle_match.group(1))
            results['delta_angle_std'] = float(angle_match.group(3))

        # ΔDIHED
        dihed_match = re.search(r'ΔDIHED\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if dihed_match:
            results['delta_dihed'] = float(dihed_match.group(1))
            results['delta_dihed_std'] = float(dihed_match.group(3))

        # ΔUB (Urey-Bradley)
        ub_match = re.search(r'ΔUB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if ub_match:
            results['delta_ub'] = float(ub_match.group(1))
            results['delta_ub_std'] = float(ub_match.group(3))

        # ΔIMP (Improper)
        imp_match = re.search(r'ΔIMP\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if imp_match:
            results['delta_imp'] = float(imp_match.group(1))
            results['delta_imp_std'] = float(imp_match.group(3))

        # ΔCMAP
        cmap_match = re.search(r'ΔCMAP\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if cmap_match:
            results['delta_cmap'] = float(cmap_match.group(1))
            results['delta_cmap_std'] = float(cmap_match.group(3))

        # Δ1-4 VDW
        vdw14_match = re.search(r'Δ1-4 VDW\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if vdw14_match:
            results['delta_vdw14'] = float(vdw14_match.group(1))
            results['delta_vdw14_std'] = float(vdw14_match.group(3))

        # Δ1-4 EEL
        eel14_match = re.search(r'Δ1-4 EEL\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if eel14_match:
            results['delta_eel14'] = float(eel14_match.group(1))
            results['delta_eel14_std'] = float(eel14_match.group(3))

        # Parse final ΔG binding with entropy correction
        # Pattern: "ΔG binding = value +/- error"
        final_dg_match = re.search(r'ΔG binding\s*=\s*([-\d.]+)\s*\+/-\s*([-\d.]+)', content)
        if final_dg_match:
            results['dg_binding'] = float(final_dg_match.group(1))
            results['dg_binding_std'] = float(final_dg_match.group(2))
            results['has_entropy'] = True

        # Parse Interaction Entropy if present
        # Format: "GB    IE    σ(Int.Energy)   Average   SD   SEM"
        ie_match = re.search(r'GB\s+IE\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if ie_match:
            results['entropy_ie_sigma'] = float(ie_match.group(1))  # σ(Int. Energy)
            results['entropy_tds'] = float(ie_match.group(2))  # -TΔS (Average)
            results['entropy_sd'] = float(ie_match.group(3))  # SD
            results['entropy_sem'] = float(ie_match.group(4))  # SEM
            results['has_entropy'] = True

        # Check for other entropy methods
        if 'NMODE' in content or 'Quasi-Harmonic' in content or 'C2 Entropy' in content:
            results['has_entropy'] = True

        return results

    def generate_summary_report(self, results):
        """Generate summary text report"""
        report_file = self.output_dir / "summary_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"gmx_MMPBSA Analysis Report\n")
            f.write(f"Analysis Type: {self.analysis_type}\n")
            f.write("="*80 + "\n\n")

            if results:
                f.write("Energy Decomposition (MM/GBSA):\n")
                f.write("-"*80 + "\n")

                # Detailed energy components
                if 'delta_eel' in results or 'delta_evdw' in results:
                    f.write("Gas Phase Energies:\n")
                    if 'delta_eel' in results:
                        f.write(f"  ΔEEL (elec):  {results['delta_eel']:>10.2f} ± {results['delta_eel_std']:>6.2f} kcal/mol\n")
                    if 'delta_evdw' in results:
                        f.write(f"  ΔEVDW (vdW):  {results['delta_evdw']:>10.2f} ± {results['delta_evdw_std']:>6.2f} kcal/mol\n")
                    if 'delta_g_gas' in results:
                        f.write(f"  ΔG gas:       {results['delta_g_gas']:>10.2f} ± {results['delta_g_gas_std']:>6.2f} kcal/mol\n")
                    f.write("\n")

                if 'delta_egb' in results or 'delta_esurf' in results:
                    f.write("Solvation Energies:\n")
                    if 'delta_egb' in results:
                        f.write(f"  ΔEGB (polar): {results['delta_egb']:>10.2f} ± {results['delta_egb_std']:>6.2f} kcal/mol\n")
                    if 'delta_esurf' in results:
                        f.write(f"  ΔESURF (np):  {results['delta_esurf']:>10.2f} ± {results['delta_esurf_std']:>6.2f} kcal/mol\n")
                    if 'delta_g_solv' in results:
                        f.write(f"  ΔG solvation: {results['delta_g_solv']:>10.2f} ± {results['delta_g_solv_std']:>6.2f} kcal/mol\n")
                    f.write("\n")

                if 'delta_total' in results:
                    f.write("Total Binding Free Energy:\n")
                    f.write(f"  ΔG (MM/GBSA): {results['delta_total']:>10.2f} ± {results['delta_total_std']:>6.2f} kcal/mol\n")

                f.write("\n")

                # Entropy correction if present
                if 'dg_binding' in results:
                    f.write("Entropy-Corrected Binding Free Energy:\n")
                    f.write("-"*80 + "\n")
                    if 'entropy_tds' in results:
                        f.write(f"ΔH (enthalpy):  {results['delta_total']:>10.2f} ± {results['delta_total_std']:>6.2f} kcal/mol\n")
                        f.write(f"-TΔS (entropy): {results['entropy_tds']:>10.2f} ± {results['entropy_sd']:>6.2f} kcal/mol\n")
                    f.write(f"ΔG binding:     {results['dg_binding']:>10.2f} ± {results['dg_binding_std']:>6.2f} kcal/mol\n")
                    f.write("\n")
                    if 'entropy_ie_sigma' in results:
                        f.write(f"(Using Interaction Entropy method, σ(Int. Energy) = {results['entropy_ie_sigma']:.2f} kcal/mol)\n")
                        if results['entropy_ie_sigma'] > 3.6:
                            f.write("⚠ WARNING: σ(Int. Energy) > 3.6 kcal/mol - entropy values may not be reliable\n")
                    f.write("\n")

                # Interpretation
                dg_value = results.get('dg_binding', results.get('delta_total'))
                if dg_value is not None:
                    f.write("Interpretation:\n")
                    f.write("-"*80 + "\n")
                    if dg_value < -10:
                        f.write(f"Very strong binding (ΔG = {dg_value:.2f} kcal/mol)\n")
                        f.write("Expected Kd: < 1 nM (nanomolar range)\n")
                    elif dg_value < -7:
                        f.write(f"Strong binding (ΔG = {dg_value:.2f} kcal/mol)\n")
                        f.write("Expected Kd: 1-100 nM (nanomolar range)\n")
                    elif dg_value < -5:
                        f.write(f"Good binding (ΔG = {dg_value:.2f} kcal/mol)\n")
                        f.write("Expected Kd: 100 nM - 1 μM (sub-micromolar range)\n")
                    elif dg_value < -2:
                        f.write(f"Moderate binding (ΔG = {dg_value:.2f} kcal/mol)\n")
                        f.write("Expected Kd: 1-100 μM (micromolar range)\n")
                    elif dg_value < 0:
                        f.write(f"Weak binding (ΔG = {dg_value:.2f} kcal/mol)\n")
                        f.write("Expected Kd: > 100 μM (weak affinity)\n")
                    else:
                        f.write(f"Unfavorable binding (ΔG = {dg_value:.2f} kcal/mol)\n")
                        f.write("No binding expected (thermodynamically unfavorable)\n")

        print(f"[INFO] Summary report saved: {report_file}")
        return report_file


class BindingEnergyAnalyzer(MMPBSAAnalyzer):
    """Analyzer for binding free energy (Type 1)"""

    def __init__(self, result_dir):
        super().__init__(result_dir, "Binding Free Energy", "1_binding_energy")

    def analyze(self):
        """Perform analysis"""
        print(f"[INFO] Analyzing binding free energy results...")

        results = self.parse_final_results()
        if results:
            self.generate_summary_report(results)
            self.plot_energy_components(results)

        # Check for decomposition
        if (self.result_dir / "FINAL_DECOMP_MMPBSA.dat").exists():
            print("[INFO] Decomposition data found, analyzing...")
            self.analyze_decomposition()

    def plot_energy_components(self, results):
        """Plot energy components - check for CSV data first"""
        if 'delta_g_gas' not in results or 'delta_g_solv' not in results:
            return

        # Check if per-frame CSV data exists
        csv_file = self.result_dir / "energy_per_frame.csv"

        if csv_file.exists():
            # Plot time series from CSV data
            print(f"[INFO] Found per-frame energy data: {csv_file.name}")
            self._plot_binding_timeseries(csv_file, results)
        else:
            # Plot summary from averaged results only
            print(f"[INFO] No per-frame data, plotting summary only")

        # Always plot the summary bar chart
        self._plot_inline(results)

    def _plot_binding_timeseries(self, csv_file, results):
        """Plot binding energy time series from CSV file (gmx_MMPBSA_ana style)

        Note: CSV file contains Complex energies only. We show Complex energy
        time series to evaluate simulation stability and convergence.
        Delta values (Complex - Receptor - Ligand) are shown in summary plots.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        try:
            # Read CSV data - Find header row dynamically
            # gmx_MMPBSA may output different number of header rows
            # Look for the row that starts with "Frame" or "Frame #"
            with open(csv_file, 'r') as f:
                skip_rows = 0
                for i, line in enumerate(f):
                    if line.startswith('Frame'):
                        skip_rows = i
                        break

            df = pd.read_csv(csv_file, skiprows=skip_rows)

            # Convert all numeric columns to float (they may be read as strings due to line endings)
            for col in df.columns:
                if col not in ['Frame #', 'Frame']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass

            # Create figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 11))
            fig.suptitle('Complex Energy Analysis - Time Series\n(Simulation Stability Check)',
                        fontsize=16, fontweight='bold')

            # Get frame numbers - check multiple possible column names
            if 'Frame #' in df.columns:
                frames = pd.to_numeric(df['Frame #'], errors='coerce').values
            elif 'Frame' in df.columns:
                frames = pd.to_numeric(df['Frame'], errors='coerce').values
            else:
                frames = np.arange(1, len(df) + 1)

            # Subplot 1: Main energy components (GGAS, GSOLV, TOTAL)
            ax1 = axes[0]
            if 'GGAS' in df.columns:
                ax1.plot(frames, df['GGAS'].values, 'o-', label='G gas',
                        linewidth=2, markersize=4, alpha=0.7, color='#3498db')
            if 'GSOLV' in df.columns:
                ax1.plot(frames, df['GSOLV'].values, 's-', label='G solv',
                        linewidth=2, markersize=4, alpha=0.7, color='#2ecc71')
            if 'TOTAL' in df.columns:
                ax1.plot(frames, df['TOTAL'].values, '^-', label='G total',
                        linewidth=2.5, markersize=5, alpha=0.8, color='#e74c3c')

                # Add mean line
                mean_total = df['TOTAL'].mean()
                ax1.axhline(mean_total, color='darkred', linestyle='--',
                           linewidth=2, label=f'Mean: {mean_total:.2f}')

            ax1.set_xlabel('Frame', fontsize=12)
            ax1.set_ylabel('Energy (kcal/mol)', fontsize=12)
            ax1.set_title('Complex Energy Components vs Frame', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Subplot 2: TOTAL energy distribution
            ax2 = axes[1]
            if 'TOTAL' in df.columns:
                ax2.hist(df['TOTAL'].values, bins=30, color='coral',
                        edgecolor='black', alpha=0.7)
                mean_val = df['TOTAL'].mean()
                std_val = df['TOTAL'].std()
                ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_val:.2f}')
                ax2.axvline(mean_val + std_val, color='orange', linestyle=':',
                           linewidth=2, label=f'±1σ: {std_val:.2f}')
                ax2.axvline(mean_val - std_val, color='orange', linestyle=':',
                           linewidth=2)
                ax2.set_xlabel('G total (kcal/mol)', fontsize=12)
                ax2.set_ylabel('Frequency', fontsize=12)
                ax2.set_title('Complex Total Energy Distribution', fontsize=13, fontweight='bold')
                ax2.legend(loc='best', frameon=True, shadow=True, fontsize=10)
                ax2.grid(True, alpha=0.3, axis='y')

            # Subplot 3: Detailed energy components with rolling average
            ax3 = axes[2]
            window = max(5, len(df) // 20)  # 5% window for smoothing

            components = {
                'EEL': ('EEL (elec)', '#e74c3c', 'o-'),
                'VDWAALS': ('VDW (vdW)', '#3498db', 's-'),
                'EGB': ('EGB (solv)', '#2ecc71', '^-'),
                'ESURF': ('ESURF (np)', '#f39c12', 'd-')
            }

            for col, (label, color, marker) in components.items():
                if col in df.columns:
                    values = df[col].values
                    # Plot original with transparency
                    ax3.plot(frames, values, marker, label=label,
                            linewidth=1, markersize=2, alpha=0.3, color=color)
                    # Plot rolling average
                    rolling_avg = pd.Series(values).rolling(window=window, center=True).mean()
                    ax3.plot(frames, rolling_avg, '-', linewidth=2.5,
                            color=color, label=f'{label} (avg)')

            ax3.axhline(0, color='black', linewidth=0.8, linestyle='-')
            ax3.set_xlabel('Frame', fontsize=12)
            ax3.set_ylabel('Energy (kcal/mol)', fontsize=12)
            ax3.set_title(f'Detailed Energy Components (rolling avg, window={window})',
                         fontsize=13, fontweight='bold')
            ax3.legend(loc='best', frameon=True, shadow=True, fontsize=9, ncol=2)
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            output_file = self.output_dir / "binding_energy_timeseries.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Binding energy time series plot saved: {output_file}")

        except Exception as e:
            print(f"[WARNING] Could not plot binding time series: {e}")
            import traceback
            traceback.print_exc()

    def _plot_inline(self, results):
        """Inline plotting fallback"""
        # Create three subplots: summary, main components, all components
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.5])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        # Plot 1: Summary (ΔG gas, ΔG solv, ΔG total)
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
        bars = ax1.bar(components, values, yerr=errors, capsize=5, color=colors, alpha=0.7)

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('Energy (kcal/mol)', fontsize=11)
        ax1.set_title('Summary', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', labelsize=9)

        # Plot 2: Main components (EEL, EVDW, EGB, ESURF)
        if 'delta_eel' in results and 'delta_evdw' in results:
            main_components = []
            main_values = []
            main_errors = []
            main_colors = []

            if 'delta_eel' in results:
                main_components.append('ΔEEL\n(elec)')
                main_values.append(results['delta_eel'])
                main_errors.append(results['delta_eel_std'])
                main_colors.append('#e74c3c')

            if 'delta_evdw' in results:
                main_components.append('ΔEVDW\n(vdW)')
                main_values.append(results['delta_evdw'])
                main_errors.append(results['delta_evdw_std'])
                main_colors.append('#e67e22')

            if 'delta_egb' in results:
                main_components.append('ΔEGB\n(GB)')
                main_values.append(results['delta_egb'])
                main_errors.append(results['delta_egb_std'])
                main_colors.append('#3498db')

            if 'delta_esurf' in results:
                main_components.append('ΔESURF\n(surf)')
                main_values.append(results['delta_esurf'])
                main_errors.append(results['delta_esurf_std'])
                main_colors.append('#1abc9c')

            bars2 = ax2.bar(main_components, main_values, yerr=main_errors,
                           capsize=5, color=main_colors, alpha=0.7)

            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax2.set_ylabel('Energy (kcal/mol)', fontsize=11)
            ax2.set_title('Main Components', fontsize=13, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            ax2.tick_params(axis='x', labelsize=9)

        # Plot 3: All components (including internal terms)
        all_components = []
        all_values = []
        all_errors = []
        all_colors = []

        # Order: EEL, EVDW, EGB, ESURF, then internal terms
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
                all_components.append(label)
                all_values.append(results[key])
                all_errors.append(results[key + '_std'])
                all_colors.append(color)

        if all_components:
            bars3 = ax3.bar(all_components, all_values, yerr=all_errors,
                           capsize=4, color=all_colors, alpha=0.7)

            ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax3.set_ylabel('Energy (kcal/mol)', fontsize=11)
            ax3.set_title('All Energy Terms', fontsize=13, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            ax3.tick_params(axis='x', labelsize=8, labelrotation=45)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        output_file = self.output_dir / "energy_components.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Energy components plot saved: {output_file}")

    def analyze_decomposition(self):
        """Analyze per-residue decomposition if available"""
        dat_file = self.result_dir / "FINAL_DECOMP_MMPBSA.dat"
        if not dat_file.exists():
            print("[WARNING] FINAL_DECOMP_MMPBSA.dat not found")
            return

        # Check if this is pairwise decomposition
        with open(dat_file, 'r') as f:
            content = f.read()
            is_pairwise = 'Resid 1,Resid 2' in content or ',R:' in content

        if is_pairwise:
            self.analyze_pairwise_decomposition()
            return

        # Parse per-residue decomposition file
        residues = []
        with open(dat_file, 'r') as f:
            for line in f:
                if line.startswith('R:'):
                    parts = line.strip().split(',')
                    if len(parts) >= 17:
                        residue_full = parts[0]  # R:A:TYR:15
                        total_avg = float(parts[16])  # TOTAL average (0-indexed)
                        total_sd = float(parts[17]) if len(parts) > 17 else 0.0

                        # Parse residue info
                        res_parts = residue_full.split(':')
                        if len(res_parts) == 4:
                            chain = res_parts[1]
                            resname = res_parts[2]
                            resnum = res_parts[3]
                            residues.append({
                                'full': residue_full,
                                'chain': chain,
                                'resname': resname,
                                'resnum': resnum,
                                'total': total_avg,
                                'total_sd': total_sd
                            })

        if not residues:
            print("[WARNING] No residue decomposition data found")
            return

        df = pd.DataFrame(residues)

        # Sort by total energy (most negative = strongest contribution)
        df_sorted = df.sort_values('total', ascending=True).head(20)

        # Generate text report
        report_file = self.output_dir / "decomposition_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Per-Residue Energy Decomposition\n")
            f.write("="*80 + "\n\n")
            f.write("Top 20 Contributing Residues (most favorable to binding):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6}{'Residue':<15}{'Chain':<7}{'Energy (kcal/mol)':<20}{'SD':<10}\n")
            f.write("-"*80 + "\n")

            for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
                res_label = f"{row['resname']}{row['resnum']}"
                f.write(f"{idx:<6}{res_label:<15}{row['chain']:<7}{row['total']:<20.2f}{row['total_sd']:<10.2f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Interpretation:\n")
            f.write("-"*80 + "\n")
            f.write("- Negative values indicate favorable contributions to binding\n")
            f.write("- Positive values indicate unfavorable contributions\n")
            f.write("- Residues with large absolute values are key interaction sites\n")

        print(f"[INFO] Decomposition report saved: {report_file}")

        # Plot top contributing residues
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(df_sorted))
        colors = ['red' if x > 0 else 'steelblue' for x in df_sorted['total']]
        ax.barh(y_pos, df_sorted['total'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['chain']}:{row['resname']}{row['resnum']}" for _, row in df_sorted.iterrows()])
        ax.invert_yaxis()
        ax.set_xlabel('Energy Contribution (kcal/mol)', fontsize=12)
        ax.set_title('Top 20 Contributing Residues (Per-Residue Decomposition)', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "top_residues_decomp.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Top residues plot saved: {output_file}")

        # Try to generate per-residue/per-frame heatmap from CSV
        self.plot_residue_frame_heatmap()

    def analyze_pairwise_decomposition(self):
        """Analyze pairwise decomposition and convert to per-residue contributions"""
        dat_file = self.result_dir / "FINAL_DECOMP_MMPBSA.dat"

        # Parse pairwise data
        residue_contributions = {}

        with open(dat_file, 'r') as f:
            for line in f:
                if line.startswith('R:') and ',R:' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 18:
                        try:
                            res1 = parts[0]  # R:A:TYR:15
                            res2 = parts[1]  # R:A:CYS:16
                            total_avg = float(parts[16])
                            total_sd = float(parts[17])

                            # Add half of the pairwise energy to each residue
                            # (to avoid double counting)
                            for res in [res1, res2]:
                                if res not in residue_contributions:
                                    residue_contributions[res] = {
                                        'total': 0.0,
                                        'total_sd': 0.0,
                                        'count': 0
                                    }
                                residue_contributions[res]['total'] += total_avg / 2
                                residue_contributions[res]['total_sd'] += (total_sd / 2) ** 2  # Sum of variances
                                residue_contributions[res]['count'] += 1
                        except (ValueError, IndexError):
                            continue

        if not residue_contributions:
            print("[WARNING] No valid pairwise decomposition data found")
            return

        # Convert to DataFrame and calculate final SD
        residues = []
        for res, data in residue_contributions.items():
            res_parts = res.split(':')
            if len(res_parts) == 4:
                residues.append({
                    'residue': res,
                    'chain': res_parts[1],
                    'resname': res_parts[2],
                    'resnum': int(res_parts[3]),
                    'total': data['total'],
                    'total_sd': np.sqrt(data['total_sd'])  # SD from variance
                })

        df = pd.DataFrame(residues)

        # Sort by absolute energy contribution
        df['abs_total'] = df['total'].abs()
        df_sorted = df.sort_values('abs_total', ascending=False).head(20)

        # Generate report
        report_file = self.output_dir / "decomposition_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Per-Residue Energy Decomposition\n")
            f.write("(Computed from Pairwise Decomposition Data)\n")
            f.write("="*80 + "\n\n")

            f.write("Top 20 Contributing Residues (most favorable to binding):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6}{'Residue':<15}{'Chain':<7}{'Energy (kcal/mol)':<20}{'SD':<10}\n")
            f.write("-"*80 + "\n")

            for idx, row in enumerate(df_sorted.itertuples(), 1):
                res_label = f"{row.resname}{row.resnum}"
                f.write(f"{idx:<6}{res_label:<15}{row.chain:<7}{row.total:<20.2f}{row.total_sd:<10.2f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Interpretation:\n")
            f.write("-"*80 + "\n")
            f.write("- Negative values indicate favorable contributions to binding\n")
            f.write("- Positive values indicate unfavorable contributions\n")
            f.write("- Values computed by summing half of each pairwise interaction\n")
            f.write("- Residues with large absolute values are key interaction sites\n")

        print(f"[INFO] Decomposition report saved: {report_file}")

        # Plot top contributing residues
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(df_sorted))
        colors = ['red' if x > 0 else 'steelblue' for x in df_sorted['total']]
        ax.barh(y_pos, df_sorted['total'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row.chain}:{row.resname}{row.resnum}" for row in df_sorted.itertuples()])
        ax.invert_yaxis()
        ax.set_xlabel('Energy Contribution (kcal/mol)', fontsize=12)
        ax.set_title('Top 20 Contributing Residues (from Pairwise Decomposition)', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "top_residues_decomp.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Top residues plot saved: {output_file}")

        # Try to generate per-residue/per-frame heatmap from CSV
        self.plot_residue_frame_heatmap()

        # Try to generate pairwise interaction heatmap
        self.plot_pairwise_interaction_heatmap()

    def plot_residue_frame_heatmap(self):
        """Plot per-residue energy contribution over frames (heatmap)"""
        csv_file = self.result_dir / "decomp_per_frame.csv"

        if not csv_file.exists():
            print("[INFO] No per-frame decomposition CSV found, skipping heatmap")
            return

        try:
            # Read CSV data - skip header lines if needed
            # Try reading with different skip options
            df = None
            for skip_rows in [0, 1, 2, 3, 4]:
                try:
                    temp_df = pd.read_csv(csv_file, skiprows=skip_rows)
                    # Check if this looks like valid frame-residue data
                    if 'Frame' in temp_df.columns or 'Frame #' in temp_df.columns:
                        df = temp_df
                        break
                except:
                    continue

            if df is None:
                # File format not suitable for heatmap (e.g., pairwise decomposition)
                print("[INFO] Decomposition CSV format not suitable for residue-frame heatmap (may be pairwise data)")
                return

            # Check if we have Frame column
            if 'Frame' not in df.columns and 'Frame #' not in df.columns:
                print("[WARNING] No Frame column in decomp CSV, skipping heatmap")
                return

            # Identify residue columns (exclude Frame and metadata columns)
            frame_col = 'Frame' if 'Frame' in df.columns else 'Frame #'
            residue_cols = [col for col in df.columns if col not in [frame_col, 'Time', 'Time(ns)']]

            if len(residue_cols) == 0:
                print("[WARNING] No residue data in CSV, skipping heatmap")
                return

            # Limit to top N residues by average absolute energy
            n_top_residues = min(30, len(residue_cols))
            residue_avg = df[residue_cols].abs().mean()
            top_residues = residue_avg.nlargest(n_top_residues).index.tolist()

            # Create heatmap data (residues x frames)
            heatmap_data = df[top_residues].T
            frames = df[frame_col].values

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
            fig.suptitle('Per-Residue Energy Contribution Heatmap (gmx_MMPBSA_ana style)',
                        fontsize=16, fontweight='bold')

            # Subplot 1: Full heatmap
            im1 = ax1.imshow(heatmap_data, cmap='RdBu_r', aspect='auto',
                           vmin=-5, vmax=5, interpolation='nearest')
            ax1.set_xlabel('Frame', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Residue', fontsize=12, fontweight='bold')
            ax1.set_title(f'Top {n_top_residues} Residues Energy Evolution',
                         fontsize=13, fontweight='bold')

            # Set y-axis labels (residue names)
            ax1.set_yticks(range(len(top_residues)))
            ax1.set_yticklabels(top_residues, fontsize=8)

            # Set x-axis to show frame numbers
            n_frames = len(frames)
            tick_spacing = max(1, n_frames // 10)
            tick_positions = range(0, n_frames, tick_spacing)
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels([int(frames[i]) for i in tick_positions], fontsize=9)

            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Energy (kcal/mol)', fontsize=11, fontweight='bold')

            # Subplot 2: Average energy per residue (bar chart)
            avg_energies = df[top_residues].mean()
            colors = ['#e74c3c' if x > 0 else '#3498db' for x in avg_energies]

            y_pos = np.arange(len(top_residues))
            ax2.barh(y_pos, avg_energies, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_residues, fontsize=8)
            ax2.invert_yaxis()
            ax2.set_xlabel('Average Energy (kcal/mol)', fontsize=12, fontweight='bold')
            ax2.set_title('Time-Averaged Contribution', fontsize=13, fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax2.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (y, val) in enumerate(zip(y_pos, avg_energies)):
                label_x = val + 0.3 if val > 0 else val - 0.3
                ha = 'left' if val > 0 else 'right'
                ax2.text(label_x, y, f'{val:.1f}', va='center', ha=ha, fontsize=8)

            plt.tight_layout()

            # Save figure
            output_file = self.output_dir / "residue_frame_heatmap.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Per-residue/frame heatmap saved: {output_file}")

        except Exception as e:
            print(f"[WARNING] Could not generate heatmap: {e}")
            import traceback
            traceback.print_exc()

    def plot_pairwise_interaction_heatmap(self):
        """Plot pairwise residue-residue interaction heatmap"""
        csv_file = self.result_dir / "decomp_per_frame.csv"

        if not csv_file.exists():
            return

        try:
            # Read the file to check format
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # Check if this is pairwise decomposition format
            if 'Resid 1' not in ''.join(lines[:10]):
                return

            # Find the header line with "Frame #,Resid 1,Resid 2,..."
            header_idx = None
            for i, line in enumerate(lines):
                if 'Frame #' in line and 'Resid 1' in line and 'Resid 2' in line:
                    header_idx = i
                    break

            if header_idx is None:
                print("[INFO] No pairwise decomposition data found in CSV")
                return

            # Read pairwise data - only the first section (Complex)
            # Read until the next header line
            data_lines = []
            reading_data = False
            for i in range(header_idx, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                if 'Frame #' in line and 'Resid 1' in line:
                    if reading_data:  # Hit next section
                        break
                    reading_data = True
                    continue
                if reading_data:
                    data_lines.append(line)

            if not data_lines:
                print("[INFO] No pairwise data found")
                return

            # Parse data
            import io
            header = "Frame #,Resid 1,Resid 2,Internal,van der Waals,Electrostatic,Polar Solvation,Non-Polar Solv.,TOTAL"
            csv_str = header + '\n' + '\n'.join(data_lines)
            df = pd.read_csv(io.StringIO(csv_str))

            # Average across frames for each residue pair
            if 'TOTAL' not in df.columns:
                print("[INFO] No TOTAL column in pairwise data")
                return

            # Get average interaction energy for each pair
            pair_energies = df.groupby(['Resid 1', 'Resid 2'])['TOTAL'].mean().reset_index()

            # Get unique residues
            all_residues = sorted(set(pair_energies['Resid 1'].unique()) |
                                set(pair_energies['Resid 2'].unique()))

            # Filter to most important residues (top contributors)
            residue_importance = {}
            for res in all_residues:
                res_energy = pair_energies[
                    (pair_energies['Resid 1'] == res) |
                    (pair_energies['Resid 2'] == res)
                ]['TOTAL'].abs().sum()
                residue_importance[res] = res_energy

            # Select top N residues
            n_top = min(30, len(all_residues))
            top_residues = sorted(residue_importance.items(),
                                key=lambda x: x[1], reverse=True)[:n_top]
            top_residue_names = [r[0] for r in top_residues]

            # Create interaction matrix
            n = len(top_residue_names)
            interaction_matrix = np.zeros((n, n))

            for _, row in pair_energies.iterrows():
                res1, res2, energy = row['Resid 1'], row['Resid 2'], row['TOTAL']
                if res1 in top_residue_names and res2 in top_residue_names:
                    i = top_residue_names.index(res1)
                    j = top_residue_names.index(res2)
                    interaction_matrix[i, j] = energy
                    if i != j:  # Make symmetric
                        interaction_matrix[j, i] = energy

            # Create heatmap
            fig, ax = plt.subplots(figsize=(16, 14))

            # Mask diagonal for better visibility
            mask = np.eye(n, dtype=bool)
            masked_matrix = np.ma.masked_array(interaction_matrix, mask=mask)

            im = ax.imshow(masked_matrix, cmap='RdBu_r', aspect='auto',
                          vmin=-10, vmax=10, interpolation='nearest')

            # Set ticks and labels
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(top_residue_names, rotation=90, ha='right', fontsize=8)
            ax.set_yticklabels(top_residue_names, fontsize=8)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Interaction Energy (kcal/mol)',
                          fontsize=12, fontweight='bold')

            # Title
            ax.set_title('Pairwise Residue-Residue Interaction Energy\n(Time-Averaged)',
                        fontsize=14, fontweight='bold', pad=20)

            # Grid
            ax.set_xticks(np.arange(n) - 0.5, minor=True)
            ax.set_yticks(np.arange(n) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

            plt.tight_layout()

            output_file = self.output_dir / "pairwise_interaction_heatmap.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Pairwise interaction heatmap saved: {output_file}")

        except Exception as e:
            print(f"[INFO] Could not generate pairwise heatmap: {e}")


class StabilityAnalyzer(MMPBSAAnalyzer):
    """Analyzer for protein stability (Type 2)"""

    def __init__(self, result_dir):
        super().__init__(result_dir, "Protein Stability", "2_stability")

    def analyze(self):
        """Perform analysis"""
        print(f"[INFO] Analyzing protein stability results...")

        results = self.parse_stability_results()
        if results:
            self.generate_stability_report(results)
            self.plot_stability_energy(results)

    def parse_stability_results(self):
        """Parse stability calculation results (Complex only, no Delta)"""
        dat_file = self.result_dir / "FINAL_RESULTS_MMPBSA.dat"
        if not dat_file.exists():
            print(f"[WARNING] {dat_file} not found!")
            return None

        with open(dat_file, 'r') as f:
            content = f.read()

        results = {}

        # Extract Complex section
        complex_match = re.search(r'Complex:(.*?)(?=Receptor:|Ligand:|Delta|$)', content, re.DOTALL)
        if not complex_match:
            print("[WARNING] Could not find Complex section in results")
            return None

        complex_section = complex_match.group(1)

        # Parse all energy components with Average and SD
        # Pattern: COMPONENT    Average    SD(Prop.)    SD    SEM(Prop.)    SEM
        energy_terms = {
            'BOND': 'bond', 'ANGLE': 'angle', 'DIHED': 'dihed',
            'UB': 'ub', 'IMP': 'imp', 'CMAP': 'cmap',
            'VDWAALS': 'vdw', 'EEL': 'eel',
            '1-4 VDW': 'vdw14', '1-4 EEL': 'eel14',
            'EGB': 'egb', 'ESURF': 'esurf',
            'GGAS': 'ggas', 'GSOLV': 'gsolv', 'TOTAL': 'total'
        }

        for term_name, term_key in energy_terms.items():
            # Match: TERM_NAME   value1   value2   value3   value4   value5
            pattern = rf'{re.escape(term_name)}\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
            match = re.search(pattern, complex_section)
            if match:
                results[term_key] = float(match.group(1))  # Average
                results[f'{term_key}_std'] = float(match.group(3))  # SD
                results[f'{term_key}_sem'] = float(match.group(5))  # SEM

        return results

    def generate_stability_report(self, results):
        """Generate stability-specific report"""
        report_file = self.output_dir / "summary_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"gmx_MMPBSA Analysis Report\n")
            f.write(f"Analysis Type: Protein Stability\n")
            f.write("="*80 + "\n\n")

            if results:
                f.write("Protein Stability (Complex Energy):\n")
                f.write("-"*80 + "\n")
                if 'ggas' in results:
                    f.write(f"G gas:          {results['ggas']:>10.2f} ± {results['ggas_std']:>6.2f} kcal/mol\n")
                if 'gsolv' in results:
                    f.write(f"G solvation:    {results['gsolv']:>10.2f} ± {results['gsolv_std']:>6.2f} kcal/mol\n")
                if 'total' in results:
                    f.write(f"G total:        {results['total']:>10.2f} ± {results['total_std']:>6.2f} kcal/mol\n")

                f.write("\n")
                f.write("Interpretation:\n")
                f.write("-"*80 + "\n")
                g_total = results.get('total', 0)
                if g_total < -10000:
                    f.write(f"Very stable protein (G = {g_total:.2f} kcal/mol)\n")
                elif g_total < -5000:
                    f.write(f"Stable protein (G = {g_total:.2f} kcal/mol)\n")
                elif g_total < 0:
                    f.write(f"Moderately stable protein (G = {g_total:.2f} kcal/mol)\n")
                else:
                    f.write(f"Unstable protein (G = {g_total:.2f} kcal/mol)\n")

                f.write("\n" + "="*80 + "\n")
                f.write("Stability Analysis Notes:\n")
                f.write("-"*80 + "\n")
                f.write("- More negative G = More stable protein fold\n")
                f.write("- This is the absolute free energy of the complex\n")
                f.write("- Compare with mutant structures to assess stability changes (ΔΔG)\n")
                f.write("- This calculation does NOT separate receptor and ligand\n")
                f.write("- Ligand group is present but not used in stability calculations\n")

        print(f"[INFO] Stability report saved: {report_file}")

    def plot_stability_energy(self, results):
        """Plot stability energy components"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Check if per-frame CSV data exists
        csv_file = self.result_dir / "energy_per_frame.csv"

        if csv_file.exists():
            # Plot time series from CSV data
            self._plot_energy_timeseries(csv_file)
        else:
            # Plot summary bar chart from averaged results
            self._plot_energy_summary(results)

    def _plot_energy_timeseries(self, csv_file):
        """Plot energy time series from CSV file"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        print(f"[INFO] Plotting energy time series from {csv_file.name}")

        try:
            # Read CSV data - Find header row dynamically
            # gmx_MMPBSA may output different number of header rows
            # Look for the row that starts with "Frame" or "Frame #"
            with open(csv_file, 'r') as f:
                skip_rows = 0
                for i, line in enumerate(f):
                    if line.startswith('Frame'):
                        skip_rows = i
                        break

            df = pd.read_csv(csv_file, skiprows=skip_rows)

            # Convert all numeric columns to float (they may be read as strings due to line endings)
            for col in df.columns:
                if col not in ['Frame #', 'Frame']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass

            # Create figure with 4 subplots (2x2 grid for better layout)
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Top row, full width
            ax2 = fig.add_subplot(gs[1, 0])   # Bottom left
            ax3 = fig.add_subplot(gs[1, 1])   # Bottom right

            fig.suptitle('Stability Analysis - Energy Time Series', fontsize=16, fontweight='bold')

            # Subplot 1: Main energy components (GGAS, GSOLV, TOTAL) - Top row full width
            if 'GGAS' in df.columns and 'GSOLV' in df.columns and 'TOTAL' in df.columns:
                # Get frame numbers
                if 'Frame #' in df.columns:
                    frames = pd.to_numeric(df['Frame #'], errors='coerce').values
                elif 'Frame' in df.columns:
                    frames = pd.to_numeric(df['Frame'], errors='coerce').values
                else:
                    frames = np.arange(1, len(df) + 1)
                ax1.plot(frames, df['GGAS'], 'o-', label='G gas', linewidth=2, markersize=4)
                ax1.plot(frames, df['GSOLV'], 's-', label='G solv', linewidth=2, markersize=4)
                ax1.plot(frames, df['TOTAL'], '^-', label='G total', linewidth=2.5, markersize=5, color='red')
                ax1.set_xlabel('Frame', fontsize=12)
                ax1.set_ylabel('Energy (kcal/mol)', fontsize=12)
                ax1.set_title('Main Energy Components vs Frame', fontsize=13, fontweight='bold')
                ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
                ax1.grid(True, alpha=0.3)

            # Subplot 2: Energy distribution histogram - Bottom left
            if 'TOTAL' in df.columns:
                ax2.hist(df['TOTAL'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                mean_val = df['TOTAL'].mean()
                std_val = df['TOTAL'].std()
                ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax2.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, label=f'±1σ: {std_val:.2f}')
                ax2.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2)
                ax2.set_xlabel('Energy (kcal/mol)', fontsize=12)
                ax2.set_ylabel('Frequency', fontsize=12)
                ax2.set_title('Total Energy Distribution', fontsize=13, fontweight='bold')
                ax2.legend(loc='best', frameon=True, shadow=True, fontsize=10)
                ax2.grid(True, alpha=0.3, axis='y')

            # Subplot 3: Detailed energy components - Rolling average for clarity (gmx_MMPBSA_ana style)
            energy_components = ['EEL', 'VDWAALS', 'EGB', 'ESURF', 'BOND', 'ANGLE', 'DIHED']
            component_labels = ['EEL (elec)', 'VDW', 'EGB (solv)', 'ESURF (np)', 'BOND', 'ANGLE', 'DIHED']
            available_components = [(comp, label) for comp, label in zip(energy_components, component_labels) if comp in df.columns]

            if available_components:
                frames = df['Frame'] if 'Frame' in df.columns else np.arange(len(df))

                # Use rolling average for smoother visualization (window=5)
                window = min(5, len(df) // 10) if len(df) > 10 else 1

                # Plot top 4-5 most variable components for clarity
                colors_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
                for idx, (comp, label) in enumerate(available_components[:7]):  # Limit to 7 for readability
                    if window > 1:
                        smoothed = df[comp].rolling(window=window, center=True).mean()
                        ax3.plot(frames, smoothed, linestyle='-',
                                label=label, linewidth=2, alpha=0.85, color=colors_palette[idx % len(colors_palette)])
                    else:
                        ax3.plot(frames, df[comp], linestyle='-',
                                label=label, linewidth=1.5, alpha=0.85, color=colors_palette[idx % len(colors_palette)])

                ax3.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax3.set_xlabel('Frame', fontsize=12)
                ax3.set_ylabel('Energy (kcal/mol)', fontsize=12)
                ax3.set_title('Key Energy Components (Smoothed)', fontsize=13, fontweight='bold')
                ax3.legend(loc='best', frameon=True, shadow=True, fontsize=9, ncol=2)
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            output_file = self.output_dir / "energy_timeseries.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Time series plot saved: {output_file}")

        except Exception as e:
            print(f"[WARNING] Could not plot time series: {e}")

    def _plot_energy_summary(self, results):
        """Plot energy summary bar chart (when no CSV data available)"""
        import matplotlib.pyplot as plt
        import numpy as np

        print(f"[INFO] Plotting energy summary (averaged data)")

        # Prepare data for plotting
        components = []
        values = []
        errors = []

        # Main components
        for key in ['ggas', 'gsolv', 'total']:
            if key in results:
                label = key.upper().replace('GGAS', 'G gas').replace('GSOLV', 'G solv').replace('TOTAL', 'G total')
                components.append(label)
                values.append(results[key])
                errors.append(results.get(f'{key}_std', 0))

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Stability Analysis - Energy Summary', fontsize=16, fontweight='bold')

        # Subplot 1: Main components
        ax1 = axes[0]
        x_pos = np.arange(len(components))
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        ax1.bar(x_pos, values, yerr=errors, capsize=10, color=colors[:len(components)],
               edgecolor='black', alpha=0.7, error_kw={'linewidth': 2, 'elinewidth': 2})
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(components)
        ax1.set_ylabel('Energy (kcal/mol)', fontsize=12)
        ax1.set_title('Main Energy Components', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='black', linewidth=0.8)

        # Subplot 2: Detailed components
        ax2 = axes[1]
        detail_components = []
        detail_values = []
        detail_errors = []

        for key in ['bond', 'angle', 'dihed', 'vdw', 'eel', 'egb', 'esurf']:
            if key in results:
                detail_components.append(key.upper())
                detail_values.append(results[key])
                detail_errors.append(results.get(f'{key}_std', 0))

        if detail_components:
            x_pos2 = np.arange(len(detail_components))
            ax2.bar(x_pos2, detail_values, yerr=detail_errors, capsize=5,
                   color='lightcoral', edgecolor='black', alpha=0.7,
                   error_kw={'linewidth': 2})
            ax2.set_xticks(x_pos2)
            ax2.set_xticklabels(detail_components, rotation=45, ha='right')
            ax2.set_ylabel('Energy (kcal/mol)', fontsize=12)
            ax2.set_title('Detailed Energy Components', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(0, color='black', linewidth=0.8)

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / "energy_summary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Summary plot saved: {output_file}")


class AlanineScanAnalyzer(MMPBSAAnalyzer):
    """Analyzer for alanine scanning (Type 3)"""

    def __init__(self, result_dir):
        super().__init__(result_dir, "Alanine Scanning", "3_alanine_scanning")

    def analyze(self):
        """Perform analysis"""
        print(f"[INFO] Analyzing alanine scanning results...")

        results = self.parse_alanine_results()
        if results:
            self.generate_alanine_report(results)
            self.plot_alanine_results(results)

    def parse_alanine_results(self):
        """Parse alanine scanning specific results - can be multiple mutants"""
        dat_file = self.result_dir / "FINAL_RESULTS_MMPBSA.dat"
        if not dat_file.exists():
            return None

        with open(dat_file, 'r') as f:
            content = f.read()

        # Alanine scanning can produce multiple mutant results
        # Look for all residue-specific ΔΔG values
        results = {
            'mutants': [],      # List of mutant names
            'ddg_values': [],   # List of ΔΔG values
            'ddg_errors': []    # List of errors
        }

        # Pattern 1: Standard alanine scanning result format
        # "RESULT OF ALANINE SCANNING (A/50 - SERxALA):"
        # "ΔΔH binding =     -0.02 +/-    0.00"
        alanine_results = re.findall(
            r'RESULT OF ALANINE SCANNING \(([^)]+)\):\s*\n\s*ΔΔ[HG]\s+binding\s*=\s*([-\d.]+)\s*\+/-\s*([\d.]+)',
            content, re.MULTILINE
        )

        if alanine_results:
            for mutant_name, ddg, error in alanine_results:
                results['mutants'].append(mutant_name.strip())
                results['ddg_values'].append(float(ddg))
                results['ddg_errors'].append(float(error))
        else:
            # Pattern 2: Alternative format - multiple mutants with individual sections
            # e.g., "Results for mutant ALA_15" followed by DELTA values
            mutant_sections = re.findall(r'Results for mutant\s+(\S+).*?DELTA DELTA.*?([-\d.]+)\s+\+/-\s+([\d.]+)',
                                         content, re.DOTALL)

            if mutant_sections:
                for mutant, ddg, error in mutant_sections:
                    results['mutants'].append(mutant)
                    results['ddg_values'].append(float(ddg))
                    results['ddg_errors'].append(float(error))
            else:
                # Pattern 3: Generic single mutant result
                mutant_match = re.search(r'Mutant:\s+(\S+)', content)
                ddg_match = re.search(r'DELTA DELTA.*?([-\d.]+)\s+\+/-\s+([\d.]+)', content)

                if mutant_match and ddg_match:
                    results['mutants'].append(mutant_match.group(1))
                    results['ddg_values'].append(float(ddg_match.group(1)))
                    results['ddg_errors'].append(float(ddg_match.group(2)))

        # If no results found, return None
        if not results['mutants']:
            return None

        return results

    def generate_alanine_report(self, results):
        """Generate alanine scanning report"""
        report_file = self.output_dir / "alanine_scan_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Alanine Scanning Analysis Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total mutants analyzed: {len(results['mutants'])}\n\n")

            f.write("ΔΔG values (positive = weaker binding, negative = stronger binding):\n")
            f.write("-"*80 + "\n")

            # Sort by ΔΔG value (most critical first)
            sorted_indices = sorted(range(len(results['ddg_values'])),
                                  key=lambda i: results['ddg_values'][i], reverse=True)

            for idx in sorted_indices:
                mutant = results['mutants'][idx]
                ddg = results['ddg_values'][idx]
                error = results['ddg_errors'][idx]

                f.write(f"{mutant:20s}  ΔΔG = {ddg:>7.2f} ± {error:>5.2f} kcal/mol")

                # Add interpretation
                if ddg > 2:
                    f.write("  ⚠️  CRITICAL")
                elif ddg > 0.5:
                    f.write("  ⚠️  Important")
                elif ddg > -0.5:
                    f.write("  ℹ️  Neutral")
                else:
                    f.write("  ✅ Favorable")
                f.write("\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Interpretation Guide:\n")
            f.write("-"*80 + "\n")
            f.write("⚠️  CRITICAL (ΔΔG > 2):  Essential for binding\n")
            f.write("⚠️  Important (0.5-2):   Contributes to binding\n")
            f.write("ℹ️  Neutral (-0.5-0.5):  Minimal effect\n")
            f.write("✅ Favorable (< -0.5):  Mutation improves binding\n")

        print(f"[INFO] Alanine scan report saved: {report_file}")

    def plot_alanine_results(self, results):
        """Plot alanine scanning results (gmx_MMPBSA_ana style)"""
        import matplotlib.pyplot as plt
        import numpy as np

        print(f"[INFO] Plotting alanine scanning results...")

        try:
            mutants = results['mutants']
            ddg_values = results['ddg_values']
            ddg_errors = results['ddg_errors']

            # Sort by ΔΔG value
            sorted_indices = sorted(range(len(ddg_values)),
                                  key=lambda i: ddg_values[i], reverse=True)

            sorted_mutants = [mutants[i] for i in sorted_indices]
            sorted_ddg = [ddg_values[i] for i in sorted_indices]
            sorted_errors = [ddg_errors[i] for i in sorted_indices]

            # Color code by importance
            colors = []
            for ddg in sorted_ddg:
                if ddg > 2:
                    colors.append('#e74c3c')  # Red - Critical
                elif ddg > 0.5:
                    colors.append('#f39c12')  # Orange - Important
                elif ddg > -0.5:
                    colors.append('#95a5a6')  # Gray - Neutral
                else:
                    colors.append('#2ecc71')  # Green - Favorable

            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Alanine Scanning Results (ΔΔG = WT - Mutant)', fontsize=16, fontweight='bold')

            # Subplot 1: Bar chart with error bars
            ax1 = axes[0]
            x_pos = np.arange(len(sorted_mutants))
            bars = ax1.bar(x_pos, sorted_ddg, yerr=sorted_errors, capsize=5,
                          color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

            # Add threshold lines
            ax1.axhline(0, color='black', linewidth=2, linestyle='-')
            ax1.axhline(2, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Critical threshold')
            ax1.axhline(0.5, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Important threshold')
            ax1.axhline(-0.5, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Favorable threshold')

            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(sorted_mutants, rotation=45, ha='right', fontsize=9)
            ax1.set_ylabel('ΔΔG (kcal/mol)', fontsize=12)
            ax1.set_title('Mutant Effect on Binding Free Energy', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3, axis='y')

            # Subplot 2: Horizontal bar chart (easier to read many mutants)
            ax2 = axes[1]
            y_pos = np.arange(len(sorted_mutants))
            ax2.barh(y_pos, sorted_ddg, xerr=sorted_errors, capsize=3,
                    color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

            ax2.axvline(0, color='black', linewidth=2, linestyle='-')
            ax2.axvline(2, color='red', linewidth=1, linestyle='--', alpha=0.3)
            ax2.axvline(0.5, color='orange', linewidth=1, linestyle='--', alpha=0.3)
            ax2.axvline(-0.5, color='green', linewidth=1, linestyle='--', alpha=0.3)

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(sorted_mutants, fontsize=9)
            ax2.set_xlabel('ΔΔG (kcal/mol)', fontsize=12)
            ax2.set_title('Ranked by Importance (Top = Most Critical)', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.invert_yaxis()  # Highest ΔΔG at top

            plt.tight_layout()

            # Save figure
            output_file = self.output_dir / "alanine_scan_results.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Alanine scan plot saved: {output_file}")

        except Exception as e:
            print(f"[WARNING] Could not plot alanine scan results: {e}")
            import traceback
            traceback.print_exc()


class QMMMAnalyzer(MMPBSAAnalyzer):
    """Analyzer for QM/MMGBSA (Type 4)"""

    def __init__(self, result_dir):
        super().__init__(result_dir, "QM/MMGBSA", "4_qm_mmgbsa")

    def analyze(self):
        """Perform analysis"""
        print(f"[INFO] Analyzing QM/MMGBSA results...")

        results = self.parse_final_results()
        if results:
            report_file = self.generate_summary_report(results)

            # Add QM-specific notes
            with open(report_file, 'a') as f:
                f.write("\n" + "="*80 + "\n")
                f.write("QM/MM Analysis Notes:\n")
                f.write("-"*80 + "\n")
                f.write("- QM region energies are more accurate for:\n")
                f.write("  • Metal coordination\n")
                f.write("  • Covalent binding\n")
                f.write("  • Charge transfer\n")
                f.write("  • Polarization effects\n")
                f.write("- Compare with standard MM results (Type 1) to assess QM effects\n")

            # Generate same visualizations as Type 1 (QM/MM has same structure)
            self.plot_energy_components(results)

            # Check for decomposition
            decomp_file = self.result_dir / "FINAL_DECOMP_MMPBSA.dat"
            if decomp_file.exists():
                self.analyze_decomposition()


class EntropyAnalyzer(MMPBSAAnalyzer):
    """Analyzer for entropy correction (Type 5)"""

    def __init__(self, result_dir):
        super().__init__(result_dir, "Entropy Correction", "5_entropy_correction")

    def analyze(self):
        """Perform analysis"""
        print(f"[INFO] Analyzing entropy-corrected results...")

        results = self.parse_entropy_results()
        if results:
            self.generate_entropy_report(results)
            self.plot_entropy_components(results)

    def parse_entropy_results(self):
        """Parse entropy-specific results"""
        dat_file = self.result_dir / "FINAL_RESULTS_MMPBSA.dat"
        if not dat_file.exists():
            return None

        with open(dat_file, 'r') as f:
            content = f.read()

        results = self.parse_final_results()

        # Parse Interaction Entropy (IE) results
        # Format: "GB    IE    σ(Int. Energy)    Average           SD        SEM"
        # Example: "GB                          IE         80.00     229.70        60.57      12.11"
        ie_match = re.search(r'GB\s+IE\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
        if ie_match:
            results['ie_sigma'] = float(ie_match.group(1))  # σ(Int. Energy)
            results['ie_tds'] = float(ie_match.group(2))    # -TΔS (Average)
            results['ie_sd'] = float(ie_match.group(3))     # SD
            results['ie_sem'] = float(ie_match.group(4))    # SEM

            # Check for warning about σ(Int. Energy)
            if results['ie_sigma'] > 3.6:
                results['ie_warning'] = True

        # Parse ΔG binding with entropy correction
        # Format: "ΔG binding =    321.36 +/-   61.26"
        dg_binding_match = re.search(r'ΔG binding\s*=\s*([-\d.]+)\s*\+/-\s*([-\d.]+)', content)
        if dg_binding_match:
            results['dg_binding'] = float(dg_binding_match.group(1))
            results['dg_binding_std'] = float(dg_binding_match.group(2))

        # Calculate enthalpy if we have entropy data
        # ΔG = ΔH - TΔS  =>  ΔH = ΔG + TΔS
        if 'ie_tds' in results and 'delta_total' in results:
            # ΔTOTAL is ΔH (enthalpy without entropy)
            # ΔG binding includes entropy correction
            results['delta_h'] = results['delta_total']  # This is the enthalpy
            results['tds'] = results['ie_tds']           # -TΔS from IE

            # Verify: ΔG = ΔH - TΔS
            if 'dg_binding' in results:
                results['delta_g'] = results['dg_binding']
            else:
                results['delta_g'] = results['delta_h'] - results['tds']

        return results

    def generate_entropy_report(self, results):
        """Generate entropy analysis report"""
        report_file = self.output_dir / "entropy_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Entropy-Corrected Free Energy Analysis (Interaction Entropy)\n")
            f.write("="*80 + "\n\n")

            # Interaction Entropy details
            if 'ie_sigma' in results:
                f.write("Interaction Entropy (IE) Method:\n")
                f.write("-"*80 + "\n")
                f.write(f"σ(Int. Energy):       {results['ie_sigma']:>10.2f} kcal/mol\n")
                f.write(f"-TΔS (Average):       {results['ie_tds']:>10.2f} kcal/mol\n")
                f.write(f"Standard Deviation:   {results['ie_sd']:>10.2f} kcal/mol\n")
                f.write(f"SEM:                  {results['ie_sem']:>10.2f} kcal/mol\n\n")

                if results.get('ie_warning', False):
                    f.write("⚠️  WARNING: σ(Int. Energy) > 3.6 kcal/mol\n")
                    f.write("   Entropy values may not be reliable.\n")
                    f.write("   Consider using more frames or checking system equilibration.\n\n")

            # Thermodynamic components
            if 'delta_h' in results:
                f.write("Thermodynamic Components:\n")
                f.write("-"*80 + "\n")
                f.write(f"ΔH (enthalpy):        {results['delta_h']:>10.2f} ± {results.get('delta_total_std', 0):>6.2f} kcal/mol\n")
                f.write(f"-TΔS (entropy):       {results['tds']:>10.2f} ± {results.get('ie_sd', 0):>6.2f} kcal/mol\n")

                if 'dg_binding' in results:
                    f.write(f"ΔG binding:           {results['dg_binding']:>10.2f} ± {results.get('dg_binding_std', 0):>6.2f} kcal/mol\n\n")
                else:
                    f.write(f"ΔG binding:           {results['delta_g']:>10.2f} kcal/mol\n\n")

                # Energy breakdown
                f.write("Energy Breakdown (without entropy):\n")
                f.write("-"*80 + "\n")
                if 'delta_g_gas' in results:
                    f.write(f"ΔG gas:               {results['delta_g_gas']:>10.2f} ± {results.get('delta_g_gas_std', 0):>6.2f} kcal/mol\n")
                if 'delta_g_solv' in results:
                    f.write(f"ΔG solv:              {results['delta_g_solv']:>10.2f} ± {results.get('delta_g_solv_std', 0):>6.2f} kcal/mol\n")
                f.write(f"ΔH (ΔG gas + solv):   {results['delta_h']:>10.2f} ± {results.get('delta_total_std', 0):>6.2f} kcal/mol\n\n")

                # Interpretation
                f.write("Interpretation:\n")
                f.write("-"*80 + "\n")
                if results['tds'] > 0:
                    f.write("⚠️  Entropy penalty: Binding reduces conformational freedom\n")
                    f.write("   (Unfavorable entropy contribution)\n")
                else:
                    f.write("✅ Entropy favorable: Binding increases disorder\n")
                    f.write("   (Favorable entropy contribution)\n")

                # Enthalpy vs Entropy driven
                if abs(results['tds']) > 0.01:  # Avoid division by zero
                    h_contribution = abs(results['delta_h']) / (abs(results['delta_h']) + abs(results['tds']))
                    f.write(f"\nEnthalpy contribution: {h_contribution*100:.1f}%\n")
                    f.write(f"Entropy contribution:  {(1-h_contribution)*100:.1f}%\n\n")

                    if h_contribution > 0.7:
                        f.write("Binding is ENTHALPY-DRIVEN (favorable interactions dominate)\n")
                    elif h_contribution < 0.3:
                        f.write("Binding is ENTROPY-DRIVEN (entropy gain dominates)\n")
                    else:
                        f.write("Binding has BALANCED enthalpy and entropy contributions\n")

        print(f"[INFO] Entropy report saved: {report_file}")

    def plot_entropy_components(self, results):
        """Plot entropy components - gmx_MMPBSA_ana style"""
        if 'delta_h' not in results:
            return

        # Create 2-panel plot: Bar chart + Breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Thermodynamic Components Bar Chart
        components = ['ΔH\n(Enthalpy)', '-TΔS\n(Entropy)', 'ΔG binding\n(Total)']

        # Use proper values
        dg_value = results.get('dg_binding', results.get('delta_g', results['delta_h'] - results['tds']))
        values = [results['delta_h'], results['tds'], dg_value]

        errors = [
            results.get('delta_total_std', 0),
            results.get('ie_sd', 0),
            results.get('dg_binding_std', 0)
        ]

        colors = ['#e74c3c', '#3498db', '#2ecc71']

        bars = ax1.bar(components, values, yerr=errors, capsize=8,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.set_ylabel('Energy (kcal/mol)', fontsize=13, fontweight='bold')
        ax1.set_title('Thermodynamic Components (ΔG = ΔH - TΔS)',
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            label_y = height + err + 5 if height > 0 else height - err - 5
            ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{val:.1f} ± {err:.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10, fontweight='bold')

        # Panel 2: Energy Component Breakdown
        breakdown_components = []
        breakdown_values = []
        breakdown_errors = []
        breakdown_colors = []

        # Add all available energy components
        if 'delta_eel' in results:
            breakdown_components.append('ΔEEL')
            breakdown_values.append(results['delta_eel'])
            breakdown_errors.append(results.get('delta_eel_std', 0))
            breakdown_colors.append('#e74c3c')

        if 'delta_evdw' in results:
            breakdown_components.append('ΔVDW')
            breakdown_values.append(results['delta_evdw'])
            breakdown_errors.append(results.get('delta_evdw_std', 0))
            breakdown_colors.append('#e67e22')

        if 'delta_egb' in results:
            breakdown_components.append('ΔEGB')
            breakdown_values.append(results['delta_egb'])
            breakdown_errors.append(results.get('delta_egb_std', 0))
            breakdown_colors.append('#3498db')

        if 'delta_esurf' in results:
            breakdown_components.append('ΔESURF')
            breakdown_values.append(results['delta_esurf'])
            breakdown_errors.append(results.get('delta_esurf_std', 0))
            breakdown_colors.append('#1abc9c')

        # Add separator and entropy
        if breakdown_components:
            breakdown_components.append('')  # Separator
            breakdown_values.append(0)
            breakdown_errors.append(0)
            breakdown_colors.append('white')

        breakdown_components.append('-TΔS\n(IE)')
        breakdown_values.append(results['tds'])
        breakdown_errors.append(results.get('ie_sd', 0))
        breakdown_colors.append('#9b59b6')

        bars2 = ax2.bar(breakdown_components, breakdown_values, yerr=breakdown_errors,
                       capsize=5, color=breakdown_colors, alpha=0.7,
                       edgecolor='black', linewidth=1.2)

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Energy (kcal/mol)', fontsize=13, fontweight='bold')
        ax2.set_title('Energy Component Breakdown (including Entropy)',
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, val, err) in enumerate(zip(bars2, breakdown_values, breakdown_errors)):
            if breakdown_components[i] == '':  # Skip separator
                continue
            height = bar.get_height()
            if abs(val) > 0.1:  # Only show non-zero values
                label_y = height + (err + 5 if height > 0 else -(err + 5))
                ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                       f'{val:.1f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=9, fontweight='bold')

        # Add warning if sigma is high
        if results.get('ie_warning', False):
            fig.text(0.5, 0.02,
                    '⚠ WARNING: σ(Int. Energy) > 3.6 kcal/mol - Entropy values may not be reliable',
                    ha='center', fontsize=11, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        output_file = self.output_dir / "entropy_components.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Entropy components plot saved: {output_file}")


class DecompositionAnalyzer(MMPBSAAnalyzer):
    """Analyzer for energy decomposition (Type 6)"""

    def __init__(self, result_dir):
        super().__init__(result_dir, "Energy Decomposition", "6_decomposition")

    def analyze(self):
        """Perform analysis - use visualize_decomposition.py for rich visualization"""
        print(f"[INFO] Analyzing energy decomposition results...")

        dat_file = self.result_dir / "FINAL_DECOMP_MMPBSA.dat"

        if not dat_file.exists():
            print(f"[ERROR] {dat_file} not found!")
            return

        # Use the dedicated visualize_decomposition.py script
        import subprocess
        import sys

        script_dir = Path(__file__).parent
        visualize_script = script_dir / "visualize_decomposition.py"

        if visualize_script.exists():
            print(f"[INFO] Using visualize_decomposition.py for detailed visualization...")
            try:
                # visualize_decomposition.py expects: <input_directory> [output_directory]
                result = subprocess.run(
                    [sys.executable, str(visualize_script), str(self.result_dir), str(self.output_dir)],
                    capture_output=True,
                    text=True,
                    check=False
                )

                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"[WARNING] visualize_decomposition.py error: {result.stderr}")
                    self.basic_analysis(dat_file)
            except Exception as e:
                print(f"[WARNING] Could not run visualize_decomposition.py: {e}")
                self.basic_analysis(dat_file)
        else:
            print(f"[INFO] visualize_decomposition.py not found, using basic visualization...")
            self.basic_analysis(dat_file)

    def basic_analysis(self, dat_file):
        """Fallback basic analysis"""
        df = self.parse_decomp_dat_file(dat_file)
        if df is None or len(df) == 0:
            print(f"[ERROR] Failed to parse {dat_file}")
            return

        self.plot_top_residues(df)
        self.plot_energy_breakdown(df)
        self.generate_decomp_report(df)

    def plot_top_residues(self, df, top_n=20):
        """Plot top contributing residues"""
        df_sorted = df.sort_values('TDC', ascending=True).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 10))

        y_pos = np.arange(len(df_sorted))
        colors = ['#d32f2f' if x < 0 else '#388e3c' for x in df_sorted['TDC']]

        bars = ax.barh(y_pos, df_sorted['TDC'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        labels = [f"{row['Residue']}{row['Residue Number']}" for _, row in df_sorted.iterrows()]
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()

        ax.set_xlabel('Energy Contribution (kcal/mol)', fontsize=12)
        ax.set_title(f'Top {top_n} Contributing Residues', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df_sorted['TDC'])):
            ax.text(val, i, f' {val:.1f}', va='center', fontsize=9)

        plt.tight_layout()
        output_file = self.output_dir / "top_residues_decomp.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Top residues plot saved: {output_file}")

    def plot_energy_breakdown(self, df):
        """Plot energy term breakdown for top residues"""
        df_top = df.sort_values('TDC', ascending=True).head(10)

        # Energy terms: van der Waals, Electrostatic, Polar solv, Nonpolar solv
        terms = ['vdW', 'Elec', 'Pol', 'Apol']
        term_labels = ['van der Waals', 'Electrostatic', 'Polar Solvation', 'Nonpolar Solvation']

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(df_top))
        width = 0.2

        for i, (term, label) in enumerate(zip(terms, term_labels)):
            if term in df_top.columns:
                offset = width * (i - 1.5)
                ax.bar(x + offset, df_top[term], width, label=label, alpha=0.8)

        ax.set_xlabel('Residue', fontsize=12)
        ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
        ax.set_title('Energy Term Breakdown (Top 10 Residues)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Residue']}{row['Residue Number']}" for _, row in df_top.iterrows()],
                          rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "energy_breakdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Energy breakdown plot saved: {output_file}")

    def generate_decomp_report(self, df):
        """Generate decomposition report"""
        report_file = self.output_dir / "decomposition_report.txt"

        df_sorted = df.sort_values('TDC', ascending=True)

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Energy Decomposition Analysis Report\n")
            f.write("="*80 + "\n\n")

            f.write("Top 20 Contributing Residues (Most Favorable):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6} {'Residue':<10} {'Total':<12} {'vdW':<12} {'Elec':<12}\n")
            f.write("-"*80 + "\n")

            for i, (_, row) in enumerate(df_sorted.head(20).iterrows(), 1):
                res_name = f"{row['Residue']}{row['Residue Number']}"
                total = row['TDC']
                vdw = row.get('vdW', 0)
                elec = row.get('Elec', 0)
                f.write(f"{i:<6} {res_name:<10} {total:>10.2f}  {vdw:>10.2f}  {elec:>10.2f}\n")

            f.write("\n\n")
            f.write("Hot Spot Residues (|TDC| > 2 kcal/mol):\n")
            f.write("-"*80 + "\n")

            hotspots = df[abs(df['TDC']) > 2].sort_values('TDC', ascending=True)
            if len(hotspots) > 0:
                for _, row in hotspots.iterrows():
                    res_name = f"{row['Residue']}{row['Residue Number']}"
                    total = row['TDC']
                    f.write(f"  {res_name:<10} {total:>10.2f} kcal/mol\n")
            else:
                f.write("  No hot spot residues identified.\n")

        print(f"[INFO] Decomposition report saved: {report_file}")

    def parse_decomp_dat_file(self, dat_file):
        """Parse decomposition DAT file (both pairwise and per-residue formats)"""

        # First, detect format
        with open(dat_file, 'r') as f:
            for line in f:
                if line.startswith('R:'):
                    is_pairwise = ',R:' in line
                    break
            else:
                return pd.DataFrame()  # No data found

        rows = []

        if is_pairwise:
            # Parse pairwise decomposition
            residue_data = {}
            with open(dat_file, 'r') as f:
                for line in f:
                    if line.startswith('R:') and ',R:' in line:
                        parts = line.strip().split(',')
                        if len(parts) >= 18:
                            try:
                                res1 = parts[0]
                                res2 = parts[1]
                                internal = float(parts[2])
                                vdw = float(parts[5])
                                elec = float(parts[8])
                                pol_solv = float(parts[11])
                                np_solv = float(parts[14])
                                total = float(parts[16])

                                for res in [res1, res2]:
                                    if res not in residue_data:
                                        residue_data[res] = {
                                            'Internal': 0.0, 'vdW': 0.0, 'Elec': 0.0,
                                            'Pol': 0.0, 'Apol': 0.0, 'TDC': 0.0
                                        }
                                    residue_data[res]['Internal'] += internal / 2
                                    residue_data[res]['vdW'] += vdw / 2
                                    residue_data[res]['Elec'] += elec / 2
                                    residue_data[res]['Pol'] += pol_solv / 2
                                    residue_data[res]['Apol'] += np_solv / 2
                                    residue_data[res]['TDC'] += total / 2
                            except (ValueError, IndexError):
                                continue

            for res, energies in residue_data.items():
                res_parts = res.split(':')
                if len(res_parts) == 4:
                    rows.append({
                        'Residue': res_parts[2],
                        'Residue Number': int(res_parts[3]),
                        'Chain': res_parts[1],
                        'Internal': energies['Internal'],
                        'vdW': energies['vdW'],
                        'Elec': energies['Elec'],
                        'Pol': energies['Pol'],
                        'Apol': energies['Apol'],
                        'TDC': energies['TDC']
                    })
        else:
            # Parse per-residue decomposition
            with open(dat_file, 'r') as f:
                for line in f:
                    if line.startswith('R:'):
                        parts = line.strip().split(',')
                        if len(parts) >= 17:
                            try:
                                res = parts[0]  # R:A:TYR:15
                                res_parts = res.split(':')
                                if len(res_parts) != 4:
                                    continue

                                # Parse energy components (Avg values at specific indices)
                                internal = float(parts[1])
                                vdw = float(parts[4])
                                elec = float(parts[7])
                                pol_solv = float(parts[10])
                                np_solv = float(parts[13])
                                total = float(parts[16])

                                rows.append({
                                    'Residue': res_parts[2],
                                    'Residue Number': int(res_parts[3]),
                                    'Chain': res_parts[1],
                                    'Internal': internal,
                                    'vdW': vdw,
                                    'Elec': elec,
                                    'Pol': pol_solv,
                                    'Apol': np_solv,
                                    'TDC': total
                                })
                            except (ValueError, IndexError):
                                continue

        return pd.DataFrame(rows)


def auto_detect_analysis_type(result_dir):
    """Auto-detect analysis type from result files"""
    result_dir = Path(result_dir)

    # Check for decomposition
    if (result_dir / "FINAL_DECOMP_MMPBSA.csv").exists():
        return "decomposition"

    # Check input file for clues
    input_file = result_dir / "_MMPBSA_info"
    if input_file.exists():
        # Could parse pickle to determine type
        # For now, check filenames
        pass

    # Default to binding energy
    return "binding"


def main():
    parser = argparse.ArgumentParser(description='Analyze gmx_MMPBSA results')
    parser.add_argument('-d', '--dir', default='.',
                       help='Result directory (default: current directory)')
    parser.add_argument('-t', '--type',
                       choices=['binding', 'stability', 'alanine', 'qmmm', 'entropy', 'decomposition', 'auto'],
                       default='auto',
                       help='Analysis type (default: auto-detect)')

    args = parser.parse_args()

    result_dir = Path(args.dir)

    if not result_dir.exists():
        print(f"[ERROR] Directory not found: {result_dir}")
        sys.exit(1)

    # Auto-detect if needed
    if args.type == 'auto':
        args.type = auto_detect_analysis_type(result_dir)
        print(f"[INFO] Auto-detected analysis type: {args.type}")

    # Select analyzer
    analyzers = {
        'binding': BindingEnergyAnalyzer,
        'stability': StabilityAnalyzer,
        'alanine': AlanineScanAnalyzer,
        'qmmm': QMMMAnalyzer,
        'entropy': EntropyAnalyzer,
        'decomposition': DecompositionAnalyzer
    }

    analyzer_class = analyzers.get(args.type)
    if not analyzer_class:
        print(f"[ERROR] Unknown analysis type: {args.type}")
        sys.exit(1)

    # Run analysis
    analyzer = analyzer_class(result_dir)
    analyzer.analyze()

    print(f"\n[SUCCESS] Analysis complete! Results saved to: {analyzer.output_dir}")


if __name__ == '__main__':
    main()
