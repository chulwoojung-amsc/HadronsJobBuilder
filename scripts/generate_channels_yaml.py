#!/usr/bin/env python3
"""
generate_channels_yaml.py — Generate observable_channels.yaml from Hadrons/Grid sources.

Reads:
  - Grid/qcd/spin/Gamma.h           (all gamma algebra names + groupings)
  - Hadrons/Modules/MContraction/   (available 2pt modules + baryon shorthands)
  - scripts/observable_channels.yaml (existing file, if any — new entries are appended)

Writes:
  - scripts/observable_channels.yaml

Idempotent: channels whose 'name' already exists in the YAML are not re-added.

Usage:
  python3 scripts/generate_channels_yaml.py [--grid-dir PATH] [--hadrons-dir PATH] [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = REPO_ROOT / "scripts" / "observable_channels.yaml"

# ---------------------------------------------------------------------------
# Physics channel templates — the mapping from gamma group → channel definition.
# Gamma names inside are placeholders; they are replaced with names discovered
# from Gamma.h so the script stays correct when Grid renames or adds matrices.
#
# Each entry has:
#   channel_name  - type literal for the observable
#   description   - human-readable description
#   hadrons_module - which Hadrons module to use
#   n_propagators  - number of quark propagator inputs
#   gamma_group    - key into the discovered gamma groups dict (see classify_gammas)
#   gamma_mode     - "single" (one literal pair) or "axes" (N×N product)
# ---------------------------------------------------------------------------

MESON_CHANNEL_TEMPLATES = [
    dict(channel_name="pion2pt",
         description="Pseudoscalar meson 2pt function (pi, K, eta). J^PC = 0^-+.",
         gamma_group="pseudoscalar", gamma_mode="single"),
    dict(channel_name="scalar2pt",
         description="Scalar meson 2pt function (sigma, f0, a0). J^PC = 0^++.",
         gamma_group="scalar", gamma_mode="single"),
    dict(channel_name="vector2pt",
         description="Vector meson 2pt function (rho, K*, phi). J^PC = 1^--.",
         gamma_group="spatial_vector", gamma_mode="axes"),
    dict(channel_name="temporalvector2pt",
         description="Temporal vector meson 2pt function. J^PC = 1^--.",
         gamma_group="temporal_vector", gamma_mode="single"),
    dict(channel_name="axialvector2pt",
         description="Axial vector meson 2pt function (a1, f1, b1). J^PC = 1^++.",
         gamma_group="spatial_axial", gamma_mode="axes"),
    dict(channel_name="axialtemporalvector2pt",
         description="Axial temporal vector meson 2pt function. J^PC = 1^+-.",
         gamma_group="temporal_axial", gamma_mode="single"),
    dict(channel_name="tensor2pt",
         description="Tensor meson 2pt function (h1, b1). Antisymmetric tensor gamma structures.",
         gamma_group="tensor", gamma_mode="axes"),
]

BARYON_CHANNEL_TEMPLATES = [
    dict(channel_name="nucleon2pt",
         description="Nucleon baryon 2pt function (spin-1/2). Standard j12 interpolator.",
         shorthand="j12"),
    dict(channel_name="delta2pt_x",
         description="Delta baryon 2pt function (spin-3/2, x polarisation). j32X interpolator.",
         shorthand="j32X"),
    dict(channel_name="delta2pt_y",
         description="Delta baryon 2pt function (spin-3/2, y polarisation). j32Y interpolator.",
         shorthand="j32Y"),
    dict(channel_name="delta2pt_z",
         description="Delta baryon 2pt function (spin-3/2, z polarisation). j32Z interpolator.",
         shorthand="j32Z"),
]

# ---------------------------------------------------------------------------
# Step 1: Parse Grid Gamma.h — extract positive algebra names and group them
# ---------------------------------------------------------------------------

def parse_gamma_names(grid_dir: Path) -> list[str]:
    """Return all positive (non-Minus) gamma algebra names from Gamma.h."""
    gamma_h = grid_dir / "Grid" / "qcd" / "spin" / "Gamma.h"
    if not gamma_h.exists():
        sys.exit(f"Gamma.h not found: {gamma_h}")

    text = gamma_h.read_text()
    # Find the GRID_SERIALIZABLE_ENUM(Algebra, ...) block
    m = re.search(r'GRID_SERIALIZABLE_ENUM\s*\(Algebra.*?\);', text, re.DOTALL)
    if not m:
        sys.exit("Could not find Algebra enum in Gamma.h")

    # Extract all identifiers followed by a comma and integer
    entries = re.findall(r'\b([A-Za-z][A-Za-z0-9]*)\s*,\s*\d+', m.group(0))
    # Skip the enum name itself and drop Minus* variants
    positives = [e for e in entries if e != 'undef' and not e.startswith('Minus')]
    return positives


def classify_gammas(names: list[str]) -> dict[str, list[str]]:
    """
    Group gamma names into physics channel families.
    Returns a dict: group_key -> [gamma_name, ...]
    """
    groups = {
        "pseudoscalar":   [],
        "scalar":         [],
        "spatial_vector": [],
        "temporal_vector":[],
        "spatial_axial":  [],
        "temporal_axial": [],
        "tensor":         [],
        "other":          [],
    }
    for name in names:
        if name == "Gamma5":
            groups["pseudoscalar"].append(name)
        elif name == "Identity":
            groups["scalar"].append(name)
        elif name == "GammaT":
            groups["temporal_vector"].append(name)
        elif name == "GammaTGamma5":
            groups["temporal_axial"].append(name)
        elif re.match(r'^Gamma[XYZ]$', name):
            groups["spatial_vector"].append(name)
        elif re.match(r'^Gamma[XYZ]Gamma5$', name):
            groups["spatial_axial"].append(name)
        elif name.startswith("Sigma"):
            groups["tensor"].append(name)
        else:
            groups["other"].append(name)
    return groups


# ---------------------------------------------------------------------------
# Step 2: Parse Baryon.hpp — extract named shorthands and their expansions
# ---------------------------------------------------------------------------

def parse_baryon_shorthands(hadrons_dir: Path) -> dict[str, str]:
    """
    Return dict of shorthand -> expanded gamma pair string.
    e.g. {"j12": "(Identity SigmaXZ)", "j32X": "(Identity MinusGammaZGamma5)", ...}
    """
    baryon_hpp = hadrons_dir / "Hadrons" / "Modules" / "MContraction" / "Baryon.hpp"
    if not baryon_hpp.exists():
        sys.exit(f"Baryon.hpp not found: {baryon_hpp}")

    text = baryon_hpp.read_text()
    # Match: regex_replace(gammaString, std::regex("NAME"), "EXPANSION")
    pattern = re.compile(
        r'regex_replace\s*\(\s*gammaString\s*,\s*std::regex\s*\(\s*"([^"]+)"\s*\)\s*,\s*"([^"]+)"\s*\)'
    )
    shorthands = {}
    for m in pattern.finditer(text):
        name, expansion = m.group(1), m.group(2)
        shorthands[name] = expansion
    return shorthands


# ---------------------------------------------------------------------------
# Step 3: Build YAML entries
# ---------------------------------------------------------------------------

def build_meson_entry(tmpl: dict, groups: dict[str, list[str]],
                       module: str) -> dict | None:
    """Build a YAML channel dict for a meson template, or None if group is empty."""
    group_key = tmpl["gamma_group"]
    gammas = groups.get(group_key, [])
    if not gammas:
        print(f"  WARNING: gamma group '{group_key}' is empty — skipping '{tmpl['channel_name']}'")
        return None

    entry = {
        "name": tmpl["channel_name"],
        "hadrons_module": module,
        "n_propagators": 2,
        "description": tmpl["description"],
    }

    if tmpl["gamma_mode"] == "single":
        g = gammas[0]
        entry["gammas"] = f"({g} {g})"
    else:  # axes
        entry["gammas_axes"] = gammas

    return entry


def build_baryon_entry(tmpl: dict, shorthands: dict[str, str],
                        module: str) -> dict | None:
    """Build a YAML channel dict for a baryon template."""
    sh = tmpl["shorthand"]
    if sh not in shorthands:
        print(f"  WARNING: shorthand '{sh}' not found in Baryon.hpp — skipping '{tmpl['channel_name']}'")
        return None

    expansion = shorthands[sh]
    entry = {
        "name": tmpl["channel_name"],
        "hadrons_module": module,
        "n_propagators": 3,
        "description": tmpl["description"],
        # Baryon format: ((src_A src_B)(snk_A snk_B))
        # Using the shorthand; the Baryon module expands it at runtime.
        "gammas": f"({sh} {sh})",
        "_gammas_expanded": f"(({expansion[1:-1]})({expansion[1:-1]}))",
    }
    return entry


# ---------------------------------------------------------------------------
# Step 4: Discover which meson/baryon modules exist
# ---------------------------------------------------------------------------

def find_2pt_modules(hadrons_dir: Path) -> tuple[list[str], list[str]]:
    """
    Return (meson2pt_modules, baryon2pt_modules) found in MContraction headers.
    Uses same Par-field classification heuristic as generate_observables.py.
    """
    mcontraction = hadrons_dir / "Hadrons" / "Modules" / "MContraction"
    meson_mods, baryon_mods = [], []
    for hpp in sorted(mcontraction.glob("*.hpp")):
        text = hpp.read_text()
        fields = set(re.findall(r'std::string,\s*(\w+)', text))
        if {'q1', 'q2', 'gammas', 'sink'}.issubset(fields) and 'q3' not in fields:
            meson_mods.append(f"MContraction::{hpp.stem}")
        elif {'q1', 'q2', 'q3', 'gammas', 'sinkq1'}.issubset(fields):
            baryon_mods.append(f"MContraction::{hpp.stem}")
    return meson_mods, baryon_mods


# ---------------------------------------------------------------------------
# Step 5: Load existing YAML (for idempotency) and write output
# ---------------------------------------------------------------------------

def load_existing_names(yaml_path: Path) -> set[str]:
    if not yaml_path.exists():
        return set()
    with open(yaml_path) as f:
        existing = yaml.safe_load(f) or []
    return {ch["name"] for ch in existing if isinstance(ch, dict) and "name" in ch}


def write_yaml(yaml_path: Path, entries: list[dict], dry_run: bool):
    """Append new entries to the YAML file (or print them if dry_run)."""
    existing_names = load_existing_names(yaml_path)
    new_entries = [e for e in entries if e["name"] not in existing_names]
    skipped = [e["name"] for e in entries if e["name"] in existing_names]

    if skipped:
        print(f"\n  Skipping already-present channels: {', '.join(skipped)}")

    if not new_entries:
        print("\nNothing new to write — all channels already in YAML.")
        return

    print(f"\n  Adding {len(new_entries)} new channels: "
          f"{', '.join(e['name'] for e in new_entries)}")

    # Strip internal _gammas_expanded annotation before writing
    clean = []
    for e in new_entries:
        row = {k: v for k, v in e.items() if not k.startswith("_")}
        clean.append(row)

    if dry_run:
        print("\n--- observable_channels.yaml additions (dry run) ---")
        print(yaml.dump(clean, default_flow_style=False, sort_keys=False))
        return

    # Append to existing file (or create new)
    mode = "a" if yaml_path.exists() else "w"
    with open(yaml_path, mode) as f:
        if mode == "a":
            f.write("\n")
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote: {yaml_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-dir", default=str(REPO_ROOT.parent / "Grid" / "develop"),
                        help="Path to Grid source root (default: ../../Grid/develop)")
    parser.add_argument("--hadrons-dir", default=str(REPO_ROOT.parent / "Hadrons"),
                        help="Path to Hadrons source root (default: ../Hadrons)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying the file")
    args = parser.parse_args()

    grid_dir    = Path(args.grid_dir)
    hadrons_dir = Path(args.hadrons_dir)

    # --- Parse Grid gamma algebra ---
    print(f"Parsing gamma algebra from: {grid_dir}")
    gamma_names = parse_gamma_names(grid_dir)
    groups = classify_gammas(gamma_names)
    print(f"  Found {len(gamma_names)} positive gamma matrices")
    for grp, names in groups.items():
        if names:
            print(f"    {grp:<20} {names}")

    # --- Parse Baryon shorthands ---
    print(f"\nParsing baryon shorthands from: {hadrons_dir}")
    shorthands = parse_baryon_shorthands(hadrons_dir)
    for sh, exp in shorthands.items():
        print(f"  {sh:<12} -> {exp}")

    # --- Discover available 2pt modules ---
    print(f"\nDiscovering 2pt modules in: {hadrons_dir}")
    meson_mods, baryon_mods = find_2pt_modules(hadrons_dir)
    print(f"  Meson 2pt:  {meson_mods}")
    print(f"  Baryon 2pt: {baryon_mods}")

    if not meson_mods:
        print("  WARNING: no meson 2pt modules found")
    if not baryon_mods:
        print("  WARNING: no baryon 2pt modules found")

    # --- Build entries ---
    entries = []

    # Meson channels — use first discovered meson module (typically MContraction::Meson)
    if meson_mods:
        meson_mod = meson_mods[0]
        for tmpl in MESON_CHANNEL_TEMPLATES:
            entry = build_meson_entry(tmpl, groups, meson_mod)
            if entry:
                entries.append(entry)

    # Baryon channels — use first discovered baryon module
    if baryon_mods:
        baryon_mod = baryon_mods[0]
        for tmpl in BARYON_CHANNEL_TEMPLATES:
            entry = build_baryon_entry(tmpl, shorthands, baryon_mod)
            if entry:
                entries.append(entry)

    # --- Write YAML ---
    write_yaml(YAML_PATH, entries, args.dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
