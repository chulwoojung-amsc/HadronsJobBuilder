#!/usr/bin/env python3
"""
generate_observables.py — Generate Pydantic observable classes from Hadrons MContraction headers.

Reads:
  - Hadrons/Hadrons/Modules/MContraction/*.hpp  (auto-discovered)
  - scripts/observable_channels.yaml            (human-maintained physics channel definitions)

Writes (in-place, idempotent):
  - src/femtomeas/meas_config_agent/observable_info.py
  - src/femtomeas/meas_config_agent/observable_config.py

Generated sections are delimited by marker comments so the script can be re-run
safely without touching hand-written code.

Usage:
  python3 scripts/generate_observables.py [--hadrons-dir PATH] [--dry-run]
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

# ---------------------------------------------------------------------------
# Paths (relative to repo root, resolved at runtime)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = REPO_ROOT / "scripts" / "observable_channels.yaml"
INFO_PATH = REPO_ROOT / "src" / "femtomeas" / "meas_config_agent" / "observable_info.py"
CONFIG_PATH = REPO_ROOT / "src" / "femtomeas" / "meas_config_agent" / "observable_config.py"

MARKER_BEGIN = "# AUTO-GENERATED BEGIN"
MARKER_END   = "# AUTO-GENERATED END"

# ---------------------------------------------------------------------------
# Step 1: Parse Hadrons MContraction headers
# ---------------------------------------------------------------------------

def parse_par_fields(hpp_text: str) -> list[str]:
    """Extract field names from GRID_SERIALIZABLE_CLASS_MEMBERS(…) in a *Par class."""
    # Find all GRID_SERIALIZABLE_CLASS_MEMBERS blocks
    pattern = re.compile(
        r'GRID_SERIALIZABLE_CLASS_MEMBERS\s*\(.*?\)',
        re.DOTALL
    )
    fields = []
    for match in pattern.finditer(hpp_text):
        content = match.group(0)
        # Strip the macro name and outer parens, split on commas
        inner = re.sub(r'^GRID_SERIALIZABLE_CLASS_MEMBERS\s*\(', '', content)
        inner = re.sub(r'\)\s*$', '', inner)
        tokens = [t.strip() for t in inner.split(',')]
        # Tokens alternate: type, name, type, name, ...  (skip first token = struct name)
        # First token is the struct name, then pairs of (type, fieldname)
        for i in range(2, len(tokens), 2):
            if i < len(tokens):
                fields.append(tokens[i])
    return fields


def classify_module(fields: list[str]) -> str:
    """Classify a module family from its Par field names."""
    fset = set(fields)
    if {'q1', 'q2', 'gammas', 'sink'}.issubset(fset) and 'q3' not in fset:
        return 'meson2pt'
    if {'q1', 'q2', 'q3', 'gammas', 'sinkq1'}.issubset(fset):
        return 'baryon2pt'
    if {'left', 'right', 'gammas', 'mom'}.issubset(fset):
        return 'a2a_meson'
    if {'q_loop', 'gammas'}.issubset(fset):
        return 'disc_loop'
    return 'other'


def discover_modules(hadrons_dir: Path) -> dict[str, dict]:
    """
    Scan MContraction headers and return a dict:
      { "MContraction::Meson": {"family": "meson2pt", "fields": [...]} }
    """
    mcontraction_dir = hadrons_dir / "Hadrons" / "Modules" / "MContraction"
    if not mcontraction_dir.exists():
        sys.exit(f"MContraction directory not found: {mcontraction_dir}")

    modules = {}
    for hpp in sorted(mcontraction_dir.glob("*.hpp")):
        text = hpp.read_text()
        fields = parse_par_fields(text)
        family = classify_module(fields)
        module_type = f"MContraction::{hpp.stem}"
        modules[module_type] = {"family": family, "fields": fields, "file": hpp.name}

    return modules


# ---------------------------------------------------------------------------
# Step 2: Load and validate channel config
# ---------------------------------------------------------------------------

def load_channels(yaml_path: Path) -> list[dict]:
    with open(yaml_path) as f:
        channels = yaml.safe_load(f)
    if not isinstance(channels, list):
        sys.exit(f"{yaml_path}: expected a YAML list of channel entries")
    return channels


def validate_channels(channels: list[dict], discovered: dict[str, dict]) -> list[dict]:
    """Abort if any channel references a module not found in the headers."""
    errors = []
    for ch in channels:
        mod = ch.get('hadrons_module', '')
        if mod not in discovered:
            errors.append(
                f"  Channel '{ch.get('name')}' references '{mod}' "
                f"which was not found in MContraction headers"
            )
    if errors:
        sys.exit("Validation errors:\n" + "\n".join(errors))
    return channels


# ---------------------------------------------------------------------------
# Step 3: Build gamma strings
# ---------------------------------------------------------------------------

def build_gamma_string(channel: dict) -> str:
    if 'gammas' in channel:
        return channel['gammas']
    axes = channel['gammas_axes']
    return "".join(f"({a} {b})" for a in axes for b in axes)


# ---------------------------------------------------------------------------
# Step 4: Generate Python source blocks
# ---------------------------------------------------------------------------

def class_name(name: str) -> str:
    """'scalar2pt' -> 'Scalar2pt'"""
    return ''.join(part.capitalize() for part in re.split(r'[_\-]', name))


def generate_obs_class(ch: dict) -> str:
    cname = class_name(ch['name']) + "Obs"
    n = ch['n_propagators']
    desc = ch['description']
    type_lit = ch['name']
    return textwrap.dedent(f"""\
        class {cname}(BaseModel):
           \"\"\"{desc}\"\"\"
           type: Literal["{type_lit}"] = "{type_lit}"
           n_propagator: Literal[{n}] = Field({n}, description="The required number of propagators")
           obs_info: Literal[""] = Field("", description="General information about this observable")
        """)


def generate_config_class(ch: dict) -> str:
    cname = class_name(ch['name']) + "Config"
    type_lit = ch['name']
    n = ch['n_propagators']
    desc = ch['description']
    gammas = build_gamma_string(ch)
    mod = ch['hadrons_module']

    if mod == 'MContraction::Meson':
        prop_type = f"tuple[str, str]"
        xml_call = (
            f"mesonModuleXML(name, xml, \"{gammas}\", "
            f"self.propagators[0], self.propagators[1])"
        )
    else:  # MContraction::Baryon
        prop_type = f"tuple[str, str, str]"
        xml_call = (
            f"baryonModuleXML(name, xml, \"{gammas}\", "
            f"self.propagators[0], self.propagators[1], self.propagators[2])"
        )

    return textwrap.dedent(f"""\
        class {cname}(BaseModel):
            \"\"\"{desc}\"\"\"
            type: Literal["{type_lit}"] = "{type_lit}"
            propagators: {prop_type} = Field(..., description="The tags of the propagators used to compute the observable")

            def setXML(self, name, xml):
                {xml_call}

            def validate(self, state):
                return validateProps(self.propagators, state)
        """)


# ---------------------------------------------------------------------------
# Step 5: Patch target files
# ---------------------------------------------------------------------------

def existing_type_literals(path: Path) -> set[str]:
    """Return all type literal strings already defined in the file."""
    text = path.read_text()
    return set(re.findall(r'type:\s*Literal\["([^"]+)"\]', text))


def replace_or_insert_generated_block(path: Path, new_block_lines: str) -> str:
    """
    Replace the AUTO-GENERATED block if present, otherwise insert it
    just before the class definition that contains the first Union[ field.
    Returns the new file text (does not write).
    """
    text = path.read_text()
    block = f"{MARKER_BEGIN}\n{new_block_lines}\n{MARKER_END}\n"

    if MARKER_BEGIN in text:
        # Replace existing block
        pattern = re.compile(
            re.escape(MARKER_BEGIN) + r'.*?' + re.escape(MARKER_END) + r'\n?',
            re.DOTALL
        )
        return pattern.sub(block, text)
    else:
        # Find the first Union[ occurrence
        union_match = re.search(r'^.*Union\[', text, re.MULTILINE)
        if not union_match:
            sys.exit(f"Could not find insertion point (Union[) in {path}")
        union_pos = union_match.start()

        # Walk backwards to find the start of the enclosing class definition
        class_match = None
        for m in re.finditer(r'^class \w+', text, re.MULTILINE):
            if m.start() < union_pos:
                class_match = m
            else:
                break
        if class_match is None:
            sys.exit(f"Could not find enclosing class before Union[ in {path}")

        insert_pos = class_match.start()
        return text[:insert_pos] + block + "\n" + text[insert_pos:]


def update_union(text: str, union_var: str, all_class_names: list[str]) -> str:
    """
    Rewrite the Union[...] annotation for `union_var` to include all_class_names.
    Matches: `union_var`: Union[...] = Field(  or  obs_type: Union[...]
    """
    # Match the field annotation line containing Union[...]
    pattern = re.compile(
        r'(' + re.escape(union_var) + r'\s*:\s*)Union\[([^\]]+)\]'
    )
    match = pattern.search(text)
    if not match:
        print(f"  WARNING: could not find '{union_var}: Union[...]' — Union not updated")
        return text

    # Parse existing class names from the Union
    existing = [n.strip() for n in match.group(2).split(',')]
    # Merge: keep existing order, append new ones not already present
    merged = list(existing)
    for name in all_class_names:
        if name not in merged:
            merged.append(name)

    new_union = match.group(1) + "Union[" + ", ".join(merged) + "]"
    return text[:match.start()] + new_union + text[match.end():]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hadrons-dir', default=str(REPO_ROOT.parent / "Hadrons"),
                        help="Path to the Hadrons source root (default: ../Hadrons)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print generated code without writing files")
    args = parser.parse_args()

    hadrons_dir = Path(args.hadrons_dir)

    # --- Step 1: Discover modules ---
    print(f"Scanning MContraction headers in: {hadrons_dir}")
    discovered = discover_modules(hadrons_dir)
    print(f"  Found {len(discovered)} modules")

    # Report discovered families
    for mod, info in sorted(discovered.items()):
        print(f"    {mod:<45} [{info['family']}]")

    # Report any modules that are 2pt-relevant but have no channel config
    relevant_families = {'meson2pt', 'baryon2pt'}
    relevant_modules = {m for m, i in discovered.items() if i['family'] in relevant_families}

    # --- Step 2: Load and validate channels ---
    print(f"\nLoading channel config: {YAML_PATH}")
    channels = load_channels(YAML_PATH)
    validate_channels(channels, discovered)
    print(f"  {len(channels)} channels defined")

    # Check for relevant modules not covered by any channel
    covered_modules = {ch['hadrons_module'] for ch in channels}
    uncovered = relevant_modules - covered_modules
    if uncovered:
        print(f"\n  NOTE: these 2pt-relevant modules have no channel entry in the YAML:")
        for m in sorted(uncovered):
            print(f"    {m}")

    # --- Steps 3 & 4: Generate code, skipping already-implemented channels ---
    existing_info   = existing_type_literals(INFO_PATH)
    existing_config = existing_type_literals(CONFIG_PATH)

    new_obs_classes    = []
    new_config_classes = []
    new_obs_names      = []
    new_config_names   = []
    skipped = []

    for ch in channels:
        name = ch['name']
        obs_cname    = class_name(name) + "Obs"
        config_cname = class_name(name) + "Config"

        in_info   = name in existing_info
        in_config = name in existing_config

        if in_info and in_config:
            skipped.append(name)
            continue

        if in_info:
            print(f"  WARNING: '{name}' already in observable_info.py but not observable_config.py — generating config only")
        if in_config:
            print(f"  WARNING: '{name}' already in observable_config.py but not observable_info.py — generating info only")

        if not in_info:
            new_obs_classes.append(generate_obs_class(ch))
            new_obs_names.append(obs_cname)
        if not in_config:
            new_config_classes.append(generate_config_class(ch))
            new_config_names.append(config_cname)

    if skipped:
        print(f"\n  Skipping already-implemented channels: {', '.join(skipped)}")

    if not new_obs_classes and not new_config_classes:
        print("\nNothing to generate — all channels already implemented.")
        return

    print(f"\n  Generating: {', '.join(ch['name'] for ch in channels if ch['name'] not in skipped)}")

    # --- Step 5: Patch files ---

    # observable_info.py
    obs_block = "\n".join(new_obs_classes)
    info_text = replace_or_insert_generated_block(INFO_PATH, obs_block)

    # Collect ALL Obs class names (hand-written + newly generated) for Union update
    all_obs_names = list(re.findall(r'class (\w+Obs)\b', info_text))
    info_text = update_union(info_text, "obs_type", all_obs_names)

    # observable_config.py
    config_block = "\n".join(new_config_classes)
    config_text = replace_or_insert_generated_block(CONFIG_PATH, config_block)

    all_config_names = list(re.findall(r'class (\w+Config)\b', config_text))
    # Filter to only *observable* Config classes (exclude ObservablesConfig, ObservableConfig)
    obs_config_names = [n for n in all_config_names
                        if n not in ('ObservablesConfig', 'ObservableConfig')]
    config_text = update_union(config_text, "obs", obs_config_names)

    if args.dry_run:
        print("\n--- observable_info.py (dry run) ---")
        print(info_text)
        print("\n--- observable_config.py (dry run) ---")
        print(config_text)
    else:
        INFO_PATH.write_text(info_text)
        CONFIG_PATH.write_text(config_text)
        print(f"\nWrote: {INFO_PATH}")
        print(f"Wrote: {CONFIG_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
