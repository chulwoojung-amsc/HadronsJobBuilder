from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class _AgentFriendly(BaseModel):
    """Base for config models filled by the conversational agent. The agent marks
    fields it has not been given a value for with the literal string "<UNKNOWN>"
    (an agent_common convention); drop those so the field default applies. A
    required field left "<UNKNOWN>" still errors correctly, since dropping the key
    leaves it missing."""
    @model_validator(mode="before")
    @classmethod
    def _drop_unknowns(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v != "<UNKNOWN>"}
        return data


class DatasetConfig(_AgentFriendly):
    """Where the correlator data lives and how to interpret it."""
    data_path: str = Field(..., description="Directory containing the data file(s)")
    format: Literal["flat_text", "hadrons_xml", "auto"] = Field(
        "auto", description="Input format: flat_text (96I .dat), hadrons_xml "
        "(MContraction::Meson output), or auto-detect")
    name: str = Field("", description="Flat text: data file name, (Nsample*Nbin*Nt rows) x (Nop cols). "
                      "Hadrons XML: base observable name, or '' to auto-discover a single one under data_path")
    Nt: int = Field(..., description="Temporal extent of the lattice")
    Nbin: int = Field(1, description="Number of consecutive measurements averaged into one sample (binning)")
    i_fit: List[int] = Field(..., description="Operator (column) indices included in the fit")
    op_names: List[str] = Field(default_factory=list, description="Operator labels for plots. Either one per data column (length Nop) or one per fitted operator in i_fit order (length len(i_fit)); auto-generated if empty")
    tmin_default: int = Field(3, description="Default fit-range lower bound for all operators")
    tmax_default: int = Field(25, description="Default fit-range upper bound for all operators")
    tmin_overrides: dict[int, int] = Field(default_factory=dict, description="Per-operator tmin, keyed by operator index")
    tmax_overrides: dict[int, int] = Field(default_factory=dict, description="Per-operator tmax, keyed by operator index")


class PreprocessConfig(_AgentFriendly):
    """Optional GEVP preprocessing (3x3 smeared matrix from ops 0-8)."""
    use_gevp: bool = Field(False, description="Replace raw ops with GEVP principal correlators")
    gevp_t0: int = Field(4, description="GEVP reference timeslice t0")
    gevp_neig: int = Field(3, description="Number of principal correlators to keep")


class ModelConfig(_AgentFriendly):
    """Fit model: sum of exponentials plus optional alternating-sign states."""
    Nmass: int = Field(2, description="Number of normal exponential states")
    NmassAlt: int = Field(1, description="Number of alternating-sign exponential states")
    mass_param: Literal["direct", "log_delta"] = Field("log_delta", description="Mass parametrization")
    model_backend: Literal["analytic", "sympy"] = Field("sympy", description="Model implementation backend")
    fac: float = Field(1e6, description="Overall scale factor applied to the model")
    m1_grid: List[float] = Field(default_factory=lambda: [0.40, 0.50, 0.60, 0.70, 0.80],
                                 description="Multi-start grid for the first excited mass")
    m2_grid: List[float] = Field(default_factory=lambda: [0.80, 1.00, 1.20, 1.40, 1.60],
                                 description="Multi-start grid for the second excited mass (Nmass >= 3)")
    malt0_grid: List[float] = Field(default_factory=lambda: [0.35, 0.45, 0.55, 0.65, 0.75],
                                    description="Multi-start grid for the alt-sector ground mass")
    m0_grid_1exp: List[float] = Field(default_factory=lambda: [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00],
                                      description="Multi-start grid for the 1-exp seed fit")
    n_late: int = Field(10, description="1-exp seed fit window: t in [tmax-n_late, tmax]")


class StatConfig(_AgentFriendly):
    """Covariance construction and resampling."""
    cov_on: Literal["meff", "corr"] = Field("corr", description="Fit data vector: log-ratio ('meff') or raw correlator ('corr')")
    inner_resample: Literal["jackknife", "bootstrap"] = Field("jackknife", description="Covariance estimation method")
    outer_resample: Literal["jackknife", "bootstrap"] = Field("bootstrap", description="Error estimation method")
    Nboots_inner: int = Field(200, description="Bootstrap samples for covariance (inner_resample='bootstrap')")
    Nboots_outer: int = Field(100, description="Outer bootstrap samples for errors")
    Nboots_pval: int = Field(200, description="Bootstrap samples for the p-value computation")
    LW: float = Field(1.0, description="Ledoit-Wolf shrinkage: 0 = full covariance, 1 = diagonal")
    block_size: int = Field(1, description="Bootstrap block length in bin units; 1 = IID bootstrap")
    block_circular: bool = Field(False, description="Circular (wrap-around) vs moving-block bootstrap")
    rng_seed: int = Field(42, description="Base RNG seed")


class RunConfig(_AgentFriendly):
    """Optimizer, refinement, tuning and output options."""
    engine: Literal["spectrumfit", "pysarlac"] = Field("spectrumfit",
        description="Fit engine: 'spectrumfit' (in-house multi-exponential optimizer) or "
        "'pysarlac' (PySARLaC distribution-native single periodic-cosh fit)")
    optimizer: Literal["scipy", "custom"] = Field("scipy", description="BFGS implementation")
    use_de_refinement: bool = Field(True, description="Refine the multi-start result with differential evolution on profiled chi^2")
    de_popsize: int = Field(12, description="DE individuals per dimension")
    de_maxiter: int = Field(400, description="DE max generations")
    de_tol: float = Field(1e-7, description="DE convergence tolerance")
    de_m0_bounds: Tuple[float, float] = Field((0.10, 2.00), description="DE search range for m_0")
    de_malt0_bounds: Tuple[float, float] = Field((0.10, 1.50), description="DE search range for mAlt_0")
    de_gap_bounds: Tuple[float, float] = Field((-3.00, 2.00), description="DE search range for log mass gaps")
    tmin_tuning: Literal["none", "optuna", "sa"] = Field("none", description="Optional tmin tuning after the main fit")
    make_plots: bool = Field(True, description="Save diagnostic plots (pdf)")
    plot_dir: Optional[str] = Field(None, description="Output directory; default {data_path}Nbin{Nbin}_spec/")
    suffix: Optional[str] = Field(None, description="Output file suffix; default _Nmass{N}[Alt{M}]LW{LW}")
    op_offset: float = Field(0.02, description="t-axis offset between operators in plots")


class FitConfig(BaseModel):
    dataset: DatasetConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    stats: StatConfig = Field(default_factory=StatConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    def resolved_suffix(self) -> str:
        if self.run.suffix is not None:
            return self.run.suffix
        m = self.model
        return ('_Nmass' + str(m.Nmass)
                + ('Alt' + str(m.NmassAlt) if m.NmassAlt else '')
                + 'LW' + str(self.stats.LW))

    def resolved_plot_dir(self) -> str:
        d = self.run.plot_dir
        if d is None:
            d = self.dataset.data_path + 'Nbin' + str(self.dataset.Nbin) + '_spec/'
        #Output paths are built as `dir + name + suffix`, so the directory part
        #must end in a separator or files spill out as siblings of the dir.
        if not d.endswith('/'):
            d += '/'
        return d
