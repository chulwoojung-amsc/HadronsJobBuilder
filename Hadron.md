# HadronsJobBuilder Development Plans

## Plan: Enable All 2pt Functions

### Current State

Three observables are implemented, all routed through two Hadrons contraction modules:

| Observable | Module | Gamma structure |
|---|---|---|
| `pion2pt` | `MContraction::Meson` | `(Gamma5 Gamma5)` |
| `vector2pt` | `MContraction::Meson` | 9× `(GammaX/Y/Z GammaX/Y/Z)` |
| `nucleon2pt` | `MContraction::Baryon` | `(CG5 CG5)` |

### Grid Gamma Algebra (complete set of 16 positive matrices)

From `Grid/qcd/spin/Gamma.h`:

| Name | Physics channel |
|---|---|
| `Gamma5` | pseudoscalar (π, K, η) — **done** |
| `GammaX`, `GammaY`, `GammaZ` | vector (ρ, K*, φ) — **done** |
| `GammaT` | temporal vector |
| `Identity` | scalar (σ, f0, a0) |
| `GammaXGamma5`, `GammaYGamma5`, `GammaZGamma5` | axial vector (a1, f1, b1) |
| `GammaTGamma5` | axial temporal / pseudovector |
| `SigmaXT`, `SigmaXY`, `SigmaXZ`, `SigmaYT`, `SigmaYZ`, `SigmaZT` | tensor (h1, b1) |

### What to Add

**5 new meson channels** (all via `MContraction::Meson`, only gamma strings change):

| New observable | Gamma pairs emitted | Hadrons module |
|---|---|---|
| `scalar2pt` | `(Identity Identity)` | `MContraction::Meson` |
| `axialvector2pt` | 9× `(GammaXGamma5/YGamma5/ZGamma5 × same)` | `MContraction::Meson` |
| `temporalvector2pt` | `(GammaT GammaT)` | `MContraction::Meson` |
| `axialtemporalvector2pt` | `(GammaTGamma5 GammaTGamma5)` | `MContraction::Meson` |
| `tensor2pt` | 36× all `(SigmaXT/XY/XZ/YT/YZ/ZT × same)` | `MContraction::Meson` |

**1 new baryon channel:**

| New observable | Gamma structure | Hadrons module |
|---|---|---|
| `delta2pt` | `(CGX CGX)` (and Y, Z permutations) | `MContraction::Baryon` |

### Files to Modify

Only two files change:

**`src/femtomeas/meas_config_agent/observable_info.py`**
- Add one `*Obs` Pydantic class per new observable (copy of existing pattern — only the `type` literal and docstring differ; `n_propagator` stays 2 for mesons, 3 for `delta2pt`)
- Add each new class to the `Union` in `ObservableInfo.obs_type`

**`src/femtomeas/meas_config_agent/observable_config.py`**
- Add one `*Config` Pydantic class per new observable, implementing `setXML(name, xml)` with the appropriate gamma string
- Add each to the `Union` in `ObservableConfig.obs`

No changes needed to `state.py`, `hadrons_xml.py`, `common.py`, or any workflow manager code — `configureObservables` is already fully generic.

### Implementation Steps

1. **`scalar2pt`** — trivial: one gamma pair `(Identity Identity)`, same shape as `pion2pt`

2. **`axialvector2pt`** — same double-loop pattern as `vector2pt` but over `GammaXGamma5`, `GammaYGamma5`, `GammaZGamma5`; emits 9 pairs

3. **`temporalvector2pt`** — trivial: one gamma pair `(GammaT GammaT)`

4. **`axialtemporalvector2pt`** — trivial: one gamma pair `(GammaTGamma5 GammaTGamma5)`

5. **`tensor2pt`** — double loop over the 6 antisymmetric tensor gammas (`SigmaXT`, `SigmaXY`, `SigmaXZ`, `SigmaYT`, `SigmaYZ`, `SigmaZT`); emits 36 gamma pairs

6. **`delta2pt`** — calls `baryonModuleXML` with Delta spin-3/2 gamma structure. The standard interpolator uses spatial gammas (`CGX`/`CGY`/`CGZ`) instead of `CG5`. Verify the exact gamma string format accepted by `BaryonUtils.h` before implementing.

7. **Update `Union` discriminators** in both files — append the 6 new `*Obs`/`*Config` classes

### Notes

- Steps 1–5 are completely mechanical; the gamma strings are literals with no logic change.
- Step 6 (`delta2pt`) requires verifying the gamma string format for spin-3/2 in `BaryonUtils.h` before writing `setXML`.
- The `observables → observable_configs` LLM pipeline requires no changes; the new types become visible to the LLM automatically once added to the Union (the JSON schema is embedded in the system prompt).
- `identifyObservables` uses `create_agent` (currently broken — non-standard LangChain extension). `configureObservables` uses `callModelWithStructuredOutput` and is working. New config work only touches the latter.

---

## Plan: Script to Generate 2pt Observables from Hadrons Source

Rather than editing `observable_info.py` and `observable_config.py` by hand, a script reads the Hadrons MContraction headers and a physics-channel config file to generate and insert the Pydantic + XML code. Re-running the script when Hadrons adds new modules keeps HadronsJobBuilder up to date.

### Deliverables

```
scripts/
  generate_observables.py       # the generator script
  observable_channels.yaml      # physics channel definitions (human-maintained)
```

### Background: MContraction Module Taxonomy

From scanning `Hadrons/Hadrons/Modules/MContraction/*.hpp`, modules fall into these families based on their `*Par` fields:

| Family | Signature (key fields) | Modules |
|---|---|---|
| **meson 2pt** | `q1`, `q2`, `gammas`, `sink` | `Meson.hpp` |
| **baryon 2pt** | `q1`, `q2`, `q3`, `gammas`, `sinkq1/2/3` | `Baryon.hpp` |
| **A2A meson field** | `left`, `right`, `gammas`, `mom` | `A2AMesonField.hpp`, `A2ASmearedMesonField.hpp` |
| **disc loop** | `q_loop`, `gammas`, `mom` | `DiscLoop.hpp` |
| **3pt** | 3+ quarks, current insertion | `Gamma3pt.hpp`, `BaryonGamma3pt.hpp`, `WeakEye3pt.hpp`, `WeakNonEye3pt.hpp`, `WeakMesonDecayKl2.hpp` |
| **transition** | spectator + initial/final quarks | `SigmaToNucleonEye.hpp`, `SigmaToNucleonNonEye.hpp`, `XiToSigmaEye.hpp`, `RareKaonNeutralDisc.hpp` |
| **diagnostics** | single prop + action | `WardIdentity.hpp` |
| **QED** | loop + photon | `QEDBurgerLong.hpp`, `QEDBurgerShort.hpp`, etc. |

Only the **meson 2pt** and **baryon 2pt** families are targets for the initial run. A2A and disc-loop families can be added later by extending `observable_channels.yaml`.

### `observable_channels.yaml` — Physics Channel Config

Each entry describes one named observable. The script validates that the referenced `hadrons_module` exists in the Hadrons headers before generating code.

```yaml
# Meson channels — all routed through MContraction::Meson

- name: scalar2pt
  hadrons_module: MContraction::Meson
  n_propagators: 2
  description: "Scalar meson 2pt function (sigma, f0, a0)"
  gammas: "(Identity Identity)"

- name: axialvector2pt
  hadrons_module: MContraction::Meson
  n_propagators: 2
  description: "Axial vector meson 2pt function (a1, f1, b1)"
  gammas_axes: [GammaXGamma5, GammaYGamma5, GammaZGamma5]  # script emits all 9 pairs

- name: temporalvector2pt
  hadrons_module: MContraction::Meson
  n_propagators: 2
  description: "Temporal vector meson 2pt function"
  gammas: "(GammaT GammaT)"

- name: axialtemporalvector2pt
  hadrons_module: MContraction::Meson
  n_propagators: 2
  description: "Axial temporal vector meson 2pt function"
  gammas: "(GammaTGamma5 GammaTGamma5)"

- name: tensor2pt
  hadrons_module: MContraction::Meson
  n_propagators: 2
  description: "Tensor meson 2pt function (h1, b1)"
  gammas_axes: [SigmaXT, SigmaXY, SigmaXZ, SigmaYT, SigmaYZ, SigmaZT]  # script emits all 36 pairs

# Baryon channels — routed through MContraction::Baryon

- name: delta2pt
  hadrons_module: MContraction::Baryon
  n_propagators: 3
  description: "Delta baryon 2pt function (spin-3/2)"
  gammas: "(CGX CGX)"  # verify against BaryonUtils.h before use
```

**`gammas` vs `gammas_axes`:**
- `gammas`: literal string passed directly into the Hadrons XML `<gammas>` field
- `gammas_axes`: list; the script generates all N×N pairs `(A B)` for A, B in the list and concatenates them

### `generate_observables.py` — Script Design

#### Step 1 — Parse Hadrons Headers

Scan every `.hpp` in `Hadrons/Hadrons/Modules/MContraction/`. For each file:
- Extract the `GRID_SERIALIZABLE_CLASS_MEMBERS(…)` block from the `*Par` class
- Parse out field names and types
- Derive the Hadrons module type string from the filename (`Meson.hpp` → `MContraction::Meson`)

Classification heuristic based on Par field names:
- `q1` + `q2` + `gammas` + `sink` (no `q3`) → **meson 2pt**
- `q1` + `q2` + `q3` + `gammas` + `sinkq1` → **baryon 2pt**
- `left` + `right` + `gammas` + `mom` → **A2A meson field**
- `q_loop` + `gammas` → **disc loop**
- anything else → unclassified (reported to user, not generated)

#### Step 2 — Load and Validate Channel Config

Read `observable_channels.yaml`. For each entry:
- Confirm `hadrons_module` appears in the discovered headers; abort with a clear error if not
- Check whether the `name` type literal already exists in `observable_info.py`; skip with a warning if so (makes re-runs idempotent)

#### Step 3 — Generate Gamma String

- `gammas` field present: use as-is
- `gammas_axes` field present: build `"(A0 A0)(A0 A1)...(AN AN)"` by iterating all N×N pairs

#### Step 4 — Generate Python Code

For each new channel, produce two code blocks:

**`observable_info.py` block** (`*Obs` class):
```python
class Scalar2ptObs(BaseModel):
    """Scalar meson 2pt function (sigma, f0, a0)."""
    type: Literal["scalar2pt"] = "scalar2pt"
    n_propagator: Literal[2] = Field(2, description="The required number of propagators")
    obs_info: Literal[""] = Field("", description="General information about this observable")
```

**`observable_config.py` block** (`*Config` class):
```python
class Scalar2ptConfig(BaseModel):
    """An instance of the scalar meson two-point function calculation."""
    type: Literal["scalar2pt"] = "scalar2pt"
    propagators: tuple[str, str] = Field(..., description="The tags of the propagators used to compute the observable")

    def setXML(self, name, xml):
        mesonModuleXML(name, xml, "(Identity Identity)", self.propagators[0], self.propagators[1])

    def validate(self, state):
        return validateProps(self.propagators, state)
```

For baryon channels: `n_propagators: 3`, `tuple[str, str, str]`, calling `baryonModuleXML`.

#### Step 5 — Patch Target Files

The script edits two files in-place using marker comments:

```python
# AUTO-GENERATED BEGIN
...
# AUTO-GENERATED END
```

On each run:
- Locate the marker block (or insert one before the `Union` line if absent on first run)
- Replace the block contents with freshly generated classes
- Rewrite the `Union[...]` type annotation in `ObservableInfo.obs_type` / `ObservableConfig.obs` to include all known types (manually-written originals + generated ones)

This makes re-runs safe: only the generated section is replaced; hand-written code is untouched.

### What the Script Does NOT Handle

- **A2A observables** (`A2AMesonField`, `A2ASmearedMesonField`): take A2A vectors, not standard propagators; require a different propagator model. Add later via a new YAML section.
- **Disconnected loops** (`DiscLoop`): requires a noise propagator input. Separate observable type needed.
- **3pt and transition modules**: out of scope for 2pt generation.
- **Gamma string correctness for baryons**: `delta2pt`'s gamma string must be manually verified against `BaryonUtils.h` before adding to the YAML. The script trusts whatever is in the YAML.

### Files Modified by the Script

| File | Change |
|---|---|
| `src/femtomeas/meas_config_agent/observable_info.py` | Insert `*Obs` classes + update `Union` in `ObservableInfo.obs_type` |
| `src/femtomeas/meas_config_agent/observable_config.py` | Insert `*Config` classes + update `Union` in `ObservableConfig.obs` |

No other files change. `configureObservables` is already generic; the LLM sees new types automatically because the JSON schema is embedded in the system prompt at runtime.

### Re-running When Hadrons Adds a New 2pt Module

1. Re-run `generate_observables.py` — it scans headers fresh and reports newly discovered modules not yet in `observable_channels.yaml`
2. Add a YAML entry for the new module (name, gammas, n_propagators, description)
3. Re-run — the new `*Obs` / `*Config` classes are generated and inserted; existing entries are unchanged

---

## Session Transcript — 2026-06-03

### 1. Workflow Error Diagnosis

Analyzed `main/workflow.log` and diagnosed two bugs preventing local execution:

**Bug 1 — Wrong API backend selected.**
`globals.api_impl` defaults to `"IRI"` via the env var `FEMTOMEAS_API_IMPL` (unset). The IRI `setupWorkflowAgent` unconditionally calls `setupSFapi(sfapi_key_path)`, which crashes with `TypeError: expected str, bytes or os.PathLike object, not NoneType` when `sfapi_key_path` is `None` (as it is in `workflow_local.json`).
Fix: set `FEMTOMEAS_API_IMPL=LOCAL` before running locally.

**Bug 2 — Placeholder bin path in `workflow_local.json`.**
`"bin": "/path/to/hadrons/bin"` was never updated.
Fix: updated to `/home/chulwoo/Claude/install/Hadrons/bin`.

Files changed:
- `main/workflow_local.json` — updated `bin` path
- `setup.sh` — added commented-out `export FEMTOMEAS_API_IMPL=LOCAL` with a note

To run locally:
```bash
export FEMTOMEAS_API_IMPL=LOCAL
source setup.sh
python3 main/workflow.py --execute-workflow main/workflow_local.json
```

### 2. Plan: Enable All 2pt Functions (manual approach)

Surveyed the full set of Grid gamma matrices (`Grid/qcd/spin/Gamma.h`) and all `MContraction` headers in the Hadrons source. Identified 6 missing observable types relative to the 3 already implemented (`pion2pt`, `vector2pt`, `nucleon2pt`). Wrote a manual implementation plan to `Hadron.md` (see "Plan: Enable All 2pt Functions" section above).

### 3. Plan: Script-Based Generation (revised approach)

User requested a script-based approach so the generator can be re-run when Hadrons adds new 2pt modules. Replaced the manual plan with a generator design and appended it to `Hadron.md` (see "Plan: Script to Generate 2pt Observables from Hadrons Source" section above).

### 4. Implementation

Implemented the script plan in full:

**`scripts/observable_channels.yaml`** — defines 6 new physics channels:

| Name | Module | Gammas |
|---|---|---|
| `scalar2pt` | `MContraction::Meson` | `(Identity Identity)` |
| `axialvector2pt` | `MContraction::Meson` | 9 pairs from `GammaXGamma5/YGamma5/ZGamma5` |
| `temporalvector2pt` | `MContraction::Meson` | `(GammaT GammaT)` |
| `axialtemporalvector2pt` | `MContraction::Meson` | `(GammaTGamma5 GammaTGamma5)` |
| `tensor2pt` | `MContraction::Meson` | 36 pairs from 6 `Sigma*` tensor gammas |
| `delta2pt` | `MContraction::Baryon` | `(CGX CGX)` |

**`scripts/generate_observables.py`** — the generator script:
- Scans all 25 `MContraction/*.hpp` headers and classifies each by Par field signature
- Validates every YAML channel references a discovered module
- Builds gamma strings (literal or N×N axis product)
- Generates `*Obs` and `*Config` Pydantic classes
- Patches `observable_info.py` and `observable_config.py` in-place between `# AUTO-GENERATED BEGIN/END` markers
- Rewrites both `Union[...]` discriminator annotations
- Idempotent: skips channels whose type literal already exists in the target files

Script was run and verified:
- All 6 new `*Obs` classes inserted before `ObservableInfo` in `observable_info.py`
- All 6 new `*Config` classes inserted before `ObservableConfig` in `observable_config.py`
- Both `Union[...]` annotations updated to include all 9 types
- All new Pydantic models validated successfully via `model_validate`
- Re-run confirmed idempotent ("Nothing to generate — all channels already implemented")

---

## Session Transcript — 2026-06-03 (continued)

### 5. YAML Generator Script

User requested a second script to auto-generate `observable_channels.yaml` itself from the Hadrons and Grid sources, so that the entire pipeline — YAML and Pydantic classes — can be regenerated when Hadrons adds new 2pt modules.

**`scripts/generate_channels_yaml.py`** was implemented with the following design:

**Inputs parsed from source:**
- `Grid/qcd/spin/Gamma.h` — extracts all 16 positive gamma algebra names from the `GRID_SERIALIZABLE_ENUM(Algebra, ...)` block and classifies them into groups:

  | Group | Names |
  |---|---|
  | `pseudoscalar` | `Gamma5` |
  | `scalar` | `Identity` |
  | `spatial_vector` | `GammaX`, `GammaY`, `GammaZ` |
  | `temporal_vector` | `GammaT` |
  | `spatial_axial` | `GammaXGamma5`, `GammaYGamma5`, `GammaZGamma5` |
  | `temporal_axial` | `GammaTGamma5` |
  | `tensor` | `SigmaXT`, `SigmaXY`, `SigmaXZ`, `SigmaYT`, `SigmaYZ`, `SigmaZT` |

- `Hadrons/Modules/MContraction/Baryon.hpp` — extracts named baryon interpolator shorthands from `parseGammaString`:

  | Shorthand | Expansion |
  |---|---|
  | `j12` | `(Identity SigmaXZ)` |
  | `j32X` | `(Identity MinusGammaZGamma5)` |
  | `j32Y` | `(Identity GammaT)` |
  | `j32Z` | `(Identity GammaXGamma5)` |
  | `j12_alt1` | `(Gamma5 MinusSigmaYT)` |
  | `j12_alt2` | `(Identity GammaYGamma5)` |

- `Hadrons/Modules/MContraction/*.hpp` — re-uses the same Par field heuristic as `generate_observables.py` to discover which modules are meson2pt vs baryon2pt.

**Channel templates** are hardcoded in the script (physics knowledge: which gamma group maps to which channel name and J^PC). When the script runs it fills in the actual gamma names discovered from the headers.

**Baryon gamma format:** The `MContraction::Baryon` module's `parseGammaString` requires the format `((A B)(A B))` — pairs of pairs. The shorthands are used as `(j12 j12)`, which the module expands to `((Identity SigmaXZ)(Identity SigmaXZ))` at runtime. This corrects a pre-existing bug in the hand-written `Nucleon2ptConfig.setXML` which used `(CG5 CG5)` — `CG5` is not a valid Grid gamma algebra name and the format is wrong. The YAML generator produces the correct format; `generate_observables.py` will skip `nucleon2pt` since it already exists (the hand-written class would need manual correction separately).

**Idempotency:** The script skips any channel whose `name` already exists in the YAML and only appends new entries.

**Dry-run output confirmed:**
- 5 channels already in YAML skipped (`scalar2pt`, `axialvector2pt`, `temporalvector2pt`, `axialtemporalvector2pt`, `tensor2pt`)
- 6 new entries generated: `pion2pt`, `vector2pt`, `nucleon2pt`, `delta2pt_x`, `delta2pt_y`, `delta2pt_z`
- Baryon entries use correct shorthand format: `(j12 j12)`, `(j32X j32X)`, etc.

**Full two-script pipeline:**
```bash
# Step 1: update YAML from Hadrons/Grid sources
python3 scripts/generate_channels_yaml.py

# Step 2: update Pydantic classes from YAML
python3 scripts/generate_observables.py
```

### 6. YAML Generator Run and Idempotency Confirmation

`generate_channels_yaml.py` was run for real, appending 6 new entries to `observable_channels.yaml`:
`pion2pt`, `vector2pt`, `nucleon2pt`, `delta2pt_x`, `delta2pt_y`, `delta2pt_z`.
The 5 channels already in the YAML were skipped. Total entries: 12 (including the pre-existing `delta2pt` with the incorrect `(CGX CGX)` gamma string, which remains for manual review).

`generate_observables.py` was then run in dry-run mode to confirm it would not create duplicate entries for observables already implemented in the source files. Result:

- **Skipped** (type literal already present in source): `scalar2pt`, `axialvector2pt`, `temporalvector2pt`, `axialtemporalvector2pt`, `tensor2pt`, `delta2pt`, `pion2pt`, `vector2pt`, `nucleon2pt`
- **Would generate** (new): `delta2pt_x`, `delta2pt_y`, `delta2pt_z`

The idempotency check scans target files for `type: Literal["name"]` declarations and skips any channel whose literal is already present, regardless of whether it was hand-written or previously generated. No duplicates will be created.
