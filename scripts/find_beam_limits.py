#!/usr/bin/env python
"""
Script to systematically find and print dx limits for beams in grism configuration files.

This script processes aXe-compatible configuration files and determines the dx limits
where dispersed beams produce wavelengths within specified bounds (default: 9000-20000 Å).

The script samples detector positions at corners, edges, and center to find the most
extreme wavelength values across the entire detector (default: 0-4088 in x and y).

Usage:
    python find_beam_limits.py --pattern "path/to/config/files/*.conf" [--output output.txt]
"""


# ADD A FILTER FOR DY
# OR, GROW dx UNTIL WAVELENGTH RANGE IS CAPTURED

import numpy as np
import sys
import os
import argparse
from glob import glob

# Add the observing-program/py directory to path
github_dir = os.getenv('github_dir')
if github_dir:
    sys.path.append(os.path.join(github_dir, 'observing-program/py'))
else:
    # Fallback: assume relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(script_dir, '../../observing-program/py'))

import grism_dispersion


def calculate_beam_limits(conf_file, det_min=0, det_max=4088, test_dx_range=(-10000, 10000),
                          lam_min=9000, lam_max=20000, beams=None, n_samples=25):
    """
    Calculate dx limits for beams in a configuration file by sampling across detector.

    Parameters
    ----------
    conf_file : str
        Path to the aXe configuration file
    det_min, det_max : int
        Detector bounds in x and y (default: 0 to 4088)
    test_dx_range : tuple
        Range of dx values to test (min, max)
    lam_min, lam_max : int
        Wavelength bounds in Angstroms to consider valid
    beams : list or None
        List of beam names to process. If None, uses ['A', 'B', 'C']
    n_samples : int
        Number of sample points along each detector edge (default: 25)

    Returns
    -------
    results : dict
        Dictionary with beam names as keys and limit info as values
    """
    if beams is None:
        beams = ['A', 'B', 'C']

    # Load configuration
    grism_conf = grism_dispersion.aXeConf(conf_file)

    # Create test dx array
    test_dx = np.arange(test_dx_range[0], test_dx_range[1], 1)

    # Sample positions across detector: corners, edges, and center
    sample_positions = []

    # Corners
    sample_positions.extend([
        (det_min, det_min),
        (det_min, det_max),
        (det_max, det_min),
        (det_max, det_max),
    ])

    # Center
    det_center = (det_min + det_max) // 2
    sample_positions.append((det_center, det_center))

    # Sample along edges
    edge_samples = np.linspace(det_min, det_max, n_samples)

    # Top and bottom edges
    for x in edge_samples:
        sample_positions.append((int(x), det_min))
        sample_positions.append((int(x), det_max))

    # Left and right edges
    for y in edge_samples:
        sample_positions.append((det_min, int(y)))
        sample_positions.append((det_max, int(y)))

    results = {}

    for beam_name in beams:
        try:
            all_lam_min = []
            all_lam_max = []
            all_dx_min = []
            all_dx_max = []

            # Sample beam trace at different positions on detector
            for xc, yc in sample_positions:
                try:
                    # Get beam trace at this position
                    beam = grism_conf.get_beam_trace(beam=beam_name, x=xc, y=yc, dx=test_dx)

                    # beam returns (dy, lam) where:
                    # - beam[0] is dy offset
                    # - beam[1] is effective wavelength in Angstroms

                    # Filter for valid wavelengths
                    valid = np.isfinite(beam[1])
                    if valid.sum() > 0:
                        lam_vals = beam[1][valid]
                        dy_vals = beam[0][valid]
                        dx_vals = test_dx[valid]

                        # Select points within specified wavelength bounds
                        sel = (lam_vals >= lam_min) & (lam_vals <= lam_max)
                        sel &= (dy_vals  < 250) & (dy_vals > -250)

                        if sel.sum() > 0:
                            all_lam_min.append(lam_vals[sel].min())
                            all_lam_max.append(lam_vals[sel].max())
                            all_dx_min.append(dx_vals[sel].min())
                            all_dx_max.append(dx_vals[sel].max())

                except Exception:
                    # Skip positions that fail (e.g., out of bounds)
                    continue

            if len(all_lam_min) == 0:
                results[beam_name] = {
                    'has_dispersion': False,
                    'lam_min': None,
                    'lam_max': None,
                    'dx_min': None,
                    'dx_max': None,
                    'n_positions': 0
                }
            else:
                results[beam_name] = {
                    'has_dispersion': True,
                    'lam_min': float(np.min(all_lam_min)),
                    'lam_max': float(np.max(all_lam_max)),
                    'dx_min': float(np.min(all_dx_min)),
                    'dx_max': float(np.max(all_dx_max)),
                    'n_positions': len(all_lam_min)
                }
        except Exception as e:
            results[beam_name] = {
                'has_dispersion': False,
                'error': str(e),
                'lam_min': None,
                'lam_max': None,
                'dx_min': None,
                'dx_max': None,
                'n_positions': 0
            }

    return results


def format_results(conf_file, results):
    """
    Format results as a text string.

    Parameters
    ----------
    conf_file : str
        Path to configuration file
    results : dict
        Results dictionary from calculate_beam_limits

    Returns
    -------
    output : str
        Formatted text output
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Configuration File: {conf_file}")
    lines.append("=" * 80)

    for beam_name in sorted(results.keys()):
        beam_info = results[beam_name]
        lines.append(f"\nBeam {beam_name}:")

        if not beam_info['has_dispersion']:
            if 'error' in beam_info:
                lines.append(f"  Status: ERROR - {beam_info['error']}")
            else:
                lines.append(f"  Status: No dispersion within specified wavelength bounds")
        else:
            lines.append(f"  Status: OK")
            lines.append(f"  Wavelength range: {beam_info['lam_min']:.1f} - {beam_info['lam_max']:.1f} Å")
            lines.append(f"  dx range: {beam_info['dx_min']:.1f} - {beam_info['dx_max']:.1f}")
            lines.append(f"  dx range: {beam_info['dx_max'] - beam_info['dx_min']}")
            lines.append(f"  Sampled positions: {beam_info['n_positions']}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Find dx limits for beams in grism configuration files by sampling detector.'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        required=True,
        help='Glob pattern for configuration files to process (e.g., "data/*.conf")'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output text file to save results'
    )
    parser.add_argument(
        '--det-min',
        type=int,
        default=0,
        help='Minimum detector coordinate (default: 0)'
    )
    parser.add_argument(
        '--det-max',
        type=int,
        default=4088,
        help='Maximum detector coordinate (default: 4088)'
    )
    parser.add_argument(
        '--dx-min',
        type=int,
        default=-10000,
        help='Minimum dx to test (default: -10000)'
    )
    parser.add_argument(
        '--dx-max',
        type=int,
        default=10000,
        help='Maximum dx to test (default: 10000)'
    )
    parser.add_argument(
        '--lam-min',
        type=int,
        default=9000,
        help='Minimum wavelength [Å] (default: 9000)'
    )
    parser.add_argument(
        '--lam-max',
        type=int,
        default=20000,
        help='Maximum wavelength [Å] (default: 20000)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=25,
        help='Number of sample points along each detector edge (default: 25)'
    )
    parser.add_argument(
        '--beams',
        type=str,
        nargs='+',
        default=['A', 'B', 'C'],
        help='Beam names to process (default: A B C)'
    )

    args = parser.parse_args()

    # Find matching files
    config_files = glob(args.pattern)

    if not config_files:
        print(f"ERROR: No files found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(config_files)} configuration file(s)")
    print(f"Sampling detector from ({args.det_min}, {args.det_min}) to ({args.det_max}, {args.det_max})")
    print(f"Using {args.n_samples} samples along each edge, plus corners and center")
    print("")

    # Prepare output file if specified
    output_file = None
    if args.output:
        output_file = open(args.output, 'w')
        output_file.write(f"Beam Limit Analysis\n")
        output_file.write(f"Generated by: {os.path.basename(__file__)}\n")
        output_file.write(f"Pattern: {args.pattern}\n")
        output_file.write(f"Detector bounds: ({args.det_min}, {args.det_min}) to ({args.det_max}, {args.det_max})\n")
        output_file.write(f"Wavelength range: {args.lam_min} - {args.lam_max} Å\n")
        output_file.write(f"Files processed: {len(config_files)}\n")
        output_file.write("\n")

    # Process each configuration file
    for conf_file in sorted(config_files):
        print(f"Processing: {conf_file}")

        try:
            results = calculate_beam_limits(
                conf_file,
                det_min=args.det_min,
                det_max=args.det_max,
                test_dx_range=(args.dx_min, args.dx_max),
                lam_min=args.lam_min,
                lam_max=args.lam_max,
                n_samples=args.n_samples,
                beams=args.beams
            )

            output_text = format_results(conf_file, results)
            print(output_text)

            if output_file:
                output_file.write(output_text + "\n")

        except Exception as e:
            error_msg = f"ERROR processing {conf_file}: {e}"
            print(error_msg)
            print("")

            if output_file:
                output_file.write(error_msg + "\n\n")

    if output_file:
        output_file.close()
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
