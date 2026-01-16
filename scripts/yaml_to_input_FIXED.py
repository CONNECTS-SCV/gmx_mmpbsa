#!/usr/bin/env python3
"""
YAML to gmx_MMPBSA Command Builder
Converts YAML configuration to gmx_MMPBSA command with all options
"""

import yaml
import sys
import argparse
import json
from pathlib import Path


def generate_general_namelist(config):
    """Generate &general namelist"""
    general = config.get('general', {})
    calc_type = config.get('calculation_type', 'binding_free_energy')
    entropy = config.get('entropy', {})

    lines = ["&general"]
    lines.append("  sys_name='MMPBSA',")
    lines.append(f"  startframe={general.get('startframe', 1)},")
    lines.append(f"  endframe={general.get('endframe', -1)},")
    lines.append(f"  interval={general.get('interval', 1)},")

    if general.get('forcefields'):
        lines.append(f"  forcefields='{general['forcefields']}',")

    lines.append(f"  temperature={general.get('temperature', 298.15)},")

    # Entropy
    if entropy.get('enabled'):
        method = entropy.get('method', 'nmode')
        if method == 'nmode':
            lines.append("  entropy=1,")
        elif method == 'qh':
            lines.append("  qh_entropy=1,")
        elif method == 'ie':
            lines.append("  interaction_entropy=1,")
            lines.append(f"  ie_segment={entropy.get('ie_segment', 25)},")
        elif method == 'c2':
            lines.append("  c2_entropy=1,")

    lines.append(f"  keep_files={general.get('keep_files', 0)},")
    lines.append(f"  verbose={general.get('verbose', 1)},")
    lines.append("/")
    return lines


def generate_gb_namelist(config):
    """Generate &gb namelist"""
    solvent = config.get('solvent_model', {})
    advanced = config.get('advanced', {})
    qm_mm = config.get('qm_mm', {})

    if solvent.get('method') != 'gb':
        return []

    lines = ["&gb"]
    lines.append(f"  igb={solvent.get('gb_model', 8)},")
    salt = config.get('general', {}).get('salt_concentration', 0.15)
    lines.append(f"  saltcon={salt},")
    lines.append(f"  intdiel={advanced.get('intdiel', 1.0)},")
    lines.append(f"  extdiel={advanced.get('extdiel', 80.0)},")
    lines.append(f"  surften={advanced.get('surften', 0.0072)},")
    lines.append(f"  surfoff={advanced.get('surfoff', 0.0)},")

    # QM/MM options
    if qm_mm.get('enabled'):
        lines.append("  ifqnt=1,")
        lines.append(f"  qm_theory='{qm_mm.get('qm_theory', 'PM6')}',")
        lines.append(f"  qm_residues='{qm_mm.get('qm_residues', '')}',")
        lines.append(f"  qmcharge_com={qm_mm.get('qm_charge_com', 0)},")
        lines.append(f"  qmcharge_rec={qm_mm.get('qm_charge_rec', 0)},")
        lines.append(f"  qmcharge_lig={qm_mm.get('qm_charge_lig', 0)},")

    lines.append("/")
    return lines


def generate_pb_namelist(config):
    """Generate &pb namelist"""
    solvent = config.get('solvent_model', {})
    advanced = config.get('advanced', {})

    if solvent.get('method') != 'pb':
        return []

    lines = ["&pb"]
    lines.append(f"  ipb={solvent.get('pb_solver', 1)},")
    lines.append(f"  indi={advanced.get('intdiel', 1.0)},")
    lines.append(f"  exdi={advanced.get('extdiel', 80.0)},")
    salt = config.get('general', {}).get('salt_concentration', 0.15)
    lines.append(f"  istrng={salt},")
    lines.append(f"  fillratio={advanced.get('fillratio', 4.0)},")
    lines.append(f"  scale={advanced.get('scale', 2.0)},")
    lines.append(f"  cavity_surften={advanced.get('surften', 0.0072)},")
    lines.append(f"  cavity_offset={advanced.get('surfoff', 0.0)},")
    lines.append("/")
    return lines


def generate_alanine_scanning_namelist(config):
    """Generate &alanine_scanning namelist"""
    if config.get('calculation_type') != 'alanine_scanning':
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
    lines.append(f"  idecomp={decomp.get('idecomp', 2)},")
    lines.append(f"  dec_verbose={decomp.get('dec_verbose', 0)},")
    if decomp.get('print_res'):
        lines.append(f"  print_res='{decomp['print_res']}',")
    csv = 1 if decomp.get('csv_format', True) else 0
    lines.append(f"  csv_format={csv},")
    lines.append("/")
    return lines


def yaml_to_command(yaml_file):
    """Convert YAML config to gmx_MMPBSA command and input file"""
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate .in file
    all_lines = []
    all_lines.extend(generate_general_namelist(config))
    all_lines.append("")
    
    gb_lines = generate_gb_namelist(config)
    if gb_lines:
        all_lines.extend(gb_lines)
        all_lines.append("")
    
    pb_lines = generate_pb_namelist(config)
    if pb_lines:
        all_lines.extend(pb_lines)
        all_lines.append("")
    
    ala_lines = generate_alanine_scanning_namelist(config)
    if ala_lines:
        all_lines.extend(ala_lines)
        all_lines.append("")
    
    nmode_lines = generate_nmode_namelist(config)
    if nmode_lines:
        all_lines.extend(nmode_lines)
        all_lines.append("")
    
    decomp_lines = generate_decomp_namelist(config)
    if decomp_lines:
        all_lines.extend(decomp_lines)
        all_lines.append("")
    
    # Extract file paths
    input_files = config.get('input_files', {})
    execution = config.get('execution', {})
    
    result = {
        'input_content': '\n'.join(all_lines),
        'files': {
            'complex_structure': input_files.get('complex_structure', ''),
            'complex_trajectory': input_files.get('complex_trajectory', ''),
            'complex_index': input_files.get('complex_index', ''),
            'receptor_group': input_files.get('receptor_group', 1),
            'ligand_group': input_files.get('ligand_group', 13),
            'receptor_structure': input_files.get('receptor_structure', ''),
            'receptor_trajectory': input_files.get('receptor_trajectory', ''),
            'ligand_mol2': input_files.get('ligand_mol2', ''),
        },
        'execution': {
            'mpi': execution.get('mpi', False),
            'cores': execution.get('cores', 4),
        }
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Convert YAML config to gmx_MMPBSA command'
    )
    parser.add_argument('yaml_file', help='YAML configuration file')
    parser.add_argument('-o', '--output', help='Output .in file')
    parser.add_argument('--json', action='store_true', help='Output JSON with all info')
    
    args = parser.parse_args()
    
    result = yaml_to_command(args.yaml_file)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Write .in file
        output_file = args.output if args.output else '/tmp/mmpbsa_generated.in'
        with open(output_file, 'w') as f:
            f.write(result['input_content'])
        print(f"Generated: {output_file}")
        
        # Print file info
        print(f"\nFiles from YAML:")
        for key, val in result['files'].items():
            if val:
                print(f"  {key}: {val}")


if __name__ == '__main__':
    main()
