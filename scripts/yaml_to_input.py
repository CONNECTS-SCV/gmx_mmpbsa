#!/usr/bin/env python3
"""
YAML to gmx_MMPBSA Input File Converter
Converts YAML configuration to .in file for gmx_MMPBSA
"""

import yaml
import sys
import argparse
from pathlib import Path


def normalize_calc_type(calc_type):
    """Normalize calculation type to lowercase with underscores"""
    if calc_type:
        return calc_type.lower().replace('/', '_').replace('-', '_')
    return 'binding_free_energy'


def generate_general_namelist(config):
    """Generate &general namelist"""
    general = config.get('general', {})
    calc_type = normalize_calc_type(config.get('calculation_type', 'binding_free_energy'))
    entropy = config.get('entropy', {})

    lines = ["&general"]

    # System name
    lines.append("  sys_name='MMPBSA',")

    # Frame selection
    lines.append(f"  startframe={general.get('startframe', 1)},")
    lines.append(f"  endframe={general.get('endframe', -1)},")
    lines.append(f"  interval={general.get('interval', 1)},")

    # Forcefield
    if general.get('forcefields'):
        lines.append(f"  forcefields='{general['forcefields']}',")

    # Temperature
    lines.append(f"  temperature={general.get('temperature', 298.15)},")

    # Entropy
    if entropy.get('enabled'):
        method = entropy.get('method', 'nmode')
        if method == 'nmode':
            # NMODE entropy is activated by presence of &nmode section
            pass
        elif method == 'qh':
            lines.append("  qh_entropy=1,")
        elif method == 'ie':
            lines.append("  interaction_entropy=1,")
            lines.append(f"  ie_segment={entropy.get('ie_segment', 25)},")
        elif method == 'c2':
            lines.append("  c2_entropy=1,")

    # Keep files
    lines.append(f"  keep_files={general.get('keep_files', 0)},")

    # Verbose
    lines.append(f"  verbose={general.get('verbose', 1)},")

    lines.append("/")
    return lines


def generate_gb_namelist(config):
    """Generate &gb namelist"""
    solvent = config.get('solvent_model', {})
    advanced = config.get('advanced', {})
    qm_mm = config.get('qm_mm', {})
    calc_type = normalize_calc_type(config.get('calculation_type', 'binding_free_energy'))

    if solvent.get('method') != 'gb' and calc_type != 'qm_mmgbsa':
        return []

    lines = ["&gb"]

    # GB model
    gb_model = solvent.get('gb_model', 8)
    lines.append(f"  igb={gb_model},")

    # Salt concentration
    salt = config.get('general', {}).get('salt_concentration', 0.15)
    lines.append(f"  saltcon={salt},")

    # Dielectric constants
    lines.append(f"  intdiel={advanced.get('intdiel', 1.0)},")
    lines.append(f"  extdiel={advanced.get('extdiel', 80.0)},")

    # Surface tension
    lines.append(f"  surften={advanced.get('surften', 0.0072)},")
    lines.append(f"  surfoff={advanced.get('surfoff', 0.0)},")

    # QM/MM options
    if calc_type == 'qm_mmgbsa':
        lines.append("  ifqnt=1,")
        lines.append(f"  qm_theory='{qm_mm.get('qm_theory', 'PM3')}',")
        lines.append(f"  qm_residues='{qm_mm.get('qm_residues', '')}',")
        lines.append(f"  qmcharge_com={qm_mm.get('qm_charge_com', 0)},")
        lines.append(f"  qmcharge_rec={qm_mm.get('qm_charge_rec', 0)},")
        lines.append(f"  qmcharge_lig={qm_mm.get('qm_charge_lig', 0)},")
        lines.append(f"  qmcut={qm_mm.get('qmcut', 9999.0)},")

    lines.append("/")
    return lines


def generate_pb_namelist(config):
    """Generate &pb namelist"""
    solvent = config.get('solvent_model', {})
    advanced = config.get('advanced', {})
    membrane = config.get('membrane', {})

    if solvent.get('method') != 'pb':
        return []

    lines = ["&pb"]

    # PB solver
    pb_solver = solvent.get('pb_solver', 1)
    lines.append(f"  ipb={pb_solver},")

    # Dielectric constants
    lines.append(f"  indi={advanced.get('intdiel', 1.0)},")
    lines.append(f"  exdi={advanced.get('extdiel', 80.0)},")

    # Salt concentration
    salt = config.get('general', {}).get('salt_concentration', 0.15)
    lines.append(f"  istrng={salt},")

    # Grid parameters
    lines.append(f"  fillratio={advanced.get('fillratio', 4.0)},")
    lines.append(f"  scale={advanced.get('scale', 2.0)},")

    # Surface tension
    lines.append(f"  cavity_surften={advanced.get('surften', 0.0072)},")
    lines.append(f"  cavity_offset={advanced.get('surfoff', 0.0)},")

    # Membrane options
    if membrane.get('enabled'):
        lines.append(f"  memopt={membrane.get('model', 1)},")
        lines.append(f"  mthick={membrane.get('thickness', 20.0)},")
        lines.append(f"  mctrdz={membrane.get('center_z', 0.0)},")

    lines.append("/")
    return lines


def generate_rism_namelist(config):
    """Generate &rism namelist"""
    solvent = config.get('solvent_model', {})

    if solvent.get('method') != '3drism':
        return []

    lines = ["&rism"]

    # Closure
    closure = solvent.get('rism_closure', 'kh')
    lines.append(f"  closure='{closure}',")

    # Default parameters
    lines.append("  grdspc=0.5, 0.5, 0.5,")
    lines.append("  tolerance=0.00001,")

    lines.append("/")
    return lines


def generate_alanine_scanning_namelist(config):
    """Generate &alanine_scanning namelist"""
    calc_type = normalize_calc_type(config.get('calculation_type'))

    if calc_type != 'alanine_scanning':
        return []

    ala_scan = config.get('alanine_scanning', {})

    lines = ["&alanine_scanning"]

    lines.append(f"  mutant='{ala_scan.get('mutant', 'ALA')}',")
    lines.append(f"  mutant_res='{ala_scan.get('residues', '')}',")

    mutant_only = 1 if ala_scan.get('mutant_only', False) else 0
    lines.append(f"  mutant_only={mutant_only},")

    lines.append("/")
    return lines


def generate_nmode_namelist(config):
    """Generate &nmode namelist"""
    entropy = config.get('entropy', {})

    if not entropy.get('enabled') or entropy.get('method') != 'nmode':
        return []

    lines = ["&nmode"]

    lines.append(f"  maxcyc={entropy.get('nmode_maxcyc', 10000)},")
    lines.append(f"  drms={entropy.get('nmode_drms', 0.001)},")

    lines.append("/")
    return lines


def generate_decomp_namelist(config):
    """Generate &decomp namelist"""
    decomp = config.get('decomposition', {})

    if not decomp.get('enabled'):
        return []

    lines = ["&decomp"]

    # Determine idecomp value
    decomp_type = decomp.get('type', 'per_residue')
    idecomp_explicit = decomp.get('idecomp')

    if idecomp_explicit is not None:
        # User explicitly set idecomp, use it as-is
        idecomp = idecomp_explicit
    else:
        # Auto-determine based on type if not specified
        if decomp_type == 'pairwise':
            idecomp = 4  # pairwise with 1-4 interactions (default for pairwise)
        else:
            idecomp = 2  # per_residue with 1-4 interactions (default for per_residue)

    # Just warn if there's inconsistency, but don't change user's choice
    if decomp_type == 'per_residue' and idecomp in [3, 4]:
        print(f"WARNING: type='per_residue' but idecomp={idecomp} (pairwise mode). "
              f"This may not work as expected.", file=sys.stderr)
    elif decomp_type == 'pairwise' and idecomp in [1, 2]:
        print(f"WARNING: type='pairwise' but idecomp={idecomp} (per_residue mode). "
              f"This may not work as expected.", file=sys.stderr)

    lines.append(f"  idecomp={idecomp},")
    lines.append(f"  dec_verbose={decomp.get('dec_verbose', 0)},")

    if decomp.get('print_res'):
        lines.append(f"  print_res='{decomp['print_res']}',")

    csv = 1 if decomp.get('csv_format', True) else 0
    lines.append(f"  csv_format={csv},")

    lines.append("/")
    return lines


def yaml_to_input(yaml_file, output_file=None):
    """Convert YAML config to gmx_MMPBSA input file"""

    # Load YAML
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    # Generate namelists
    all_lines = []

    # &general (always required)
    all_lines.extend(generate_general_namelist(config))
    all_lines.append("")

    # &gb (if GB or QM/MM)
    gb_lines = generate_gb_namelist(config)
    if gb_lines:
        all_lines.extend(gb_lines)
        all_lines.append("")

    # &pb (if PB)
    pb_lines = generate_pb_namelist(config)
    if pb_lines:
        all_lines.extend(pb_lines)
        all_lines.append("")

    # &rism (if 3D-RISM)
    rism_lines = generate_rism_namelist(config)
    if rism_lines:
        all_lines.extend(rism_lines)
        all_lines.append("")

    # &alanine_scanning (if alanine scanning)
    ala_lines = generate_alanine_scanning_namelist(config)
    if ala_lines:
        all_lines.extend(ala_lines)
        all_lines.append("")

    # &nmode (if NMODE entropy)
    nmode_lines = generate_nmode_namelist(config)
    if nmode_lines:
        all_lines.extend(nmode_lines)
        all_lines.append("")

    # &decomp (if decomposition)
    decomp_lines = generate_decomp_namelist(config)
    if decomp_lines:
        all_lines.extend(decomp_lines)
        all_lines.append("")

    # Write to file
    if output_file is None:
        output_file = yaml_file.replace('.yaml', '.in').replace('.yml', '.in')

    with open(output_file, 'w') as f:
        f.write('\n'.join(all_lines))

    print(f"Generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Convert YAML config to gmx_MMPBSA input file'
    )
    parser.add_argument('yaml_file', help='YAML configuration file')
    parser.add_argument('-o', '--output', help='Output .in file (default: auto)')

    args = parser.parse_args()

    yaml_to_input(args.yaml_file, args.output)


if __name__ == '__main__':
    main()
