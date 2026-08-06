"""Staged conversational agent for building a FitConfig, mirroring
femtomeas.meas_config_agent.agent: argparse-free orchestrator that fills one
FitConfig sub-model per stage via parameterAgent, checkpointing after each so an
interrupted session resumes mid-conversation.

Reuses femtomeas.agent_common directly (SpectrumFit_workflow_plan.md decision 2).
"""
import os
import json

from pydantic import BaseModel, Field

from femtomeas.agent_common.agent_base import parameterAgent
from femtomeas.agent_common.common import Print
from langchain.messages import HumanMessage
from langchain.tools import tool

from .config import FitConfig, DatasetConfig, PreprocessConfig, ModelConfig, StatConfig, RunConfig
from .data import resolve_format, _discover_xml, _parse_meson_xml


class FitState(BaseModel):
    """Per-stage checkpoint. Each sub-model is None until its stage completes,
    exactly like meas_config_agent.state.State."""
    query: str | None = Field(None, description="The original query")
    dataset: DatasetConfig | None = Field(None, description="Dataset location, format, dimensions, fit ranges")
    model: ModelConfig | None = Field(None, description="Fit model: exponential states and options")
    stats: StatConfig | None = Field(None, description="Covariance construction and resampling")
    run: RunConfig | None = Field(None, description="Optimizer, refinement, tuning and output")
    preprocess: PreprocessConfig | None = Field(None, description="Optional GEVP preprocessing")

    def toFitConfig(self) -> FitConfig:
        return FitConfig(dataset=self.dataset, model=self.model, stats=self.stats,
                         run=self.run, preprocess=self.preprocess or PreprocessConfig())


def checkpointState(state: FitState, filename: str):
    with open(filename, 'w') as wr:
        wr.write(json.dumps(json.loads(state.model_dump_json()), indent=2))


def reloadStateCheckpoint(filename: str) -> FitState:
    Print("Reloading fit state checkpoint from", filename)
    with open(filename, 'r') as rd:
        return FitState.model_validate_json(rd.read())


@tool
def peekDataset(data_path: str, name: str = "", Nbin: int = 1) -> str:
    """Report the shape of the correlator dataset under data_path so parameters
    can be checked against the real files, without needing Nt up front. For
    Hadrons XML returns the observable base name, trajectory range, Nt, number of
    channels and their gamma labels. For flat text returns row/column counts and
    the Nt values consistent with the row count."""
    import numpy as np
    try:
        fmt = resolve_format("auto", data_path, name)
        if fmt == "hadrons_xml":
            base, files = _discover_xml(data_path, name)
            channels, arr = _parse_meson_xml(files[0][1])
            Nt, Nop, Nfiles = arr.shape[0], arr.shape[1], len(files)
            trajs = [t for t, _ in files]
            labels = [f"{snk}-{src}" for snk, src in channels]
            return (f"format=hadrons_xml; base={base!r}; {Nfiles} trajectories "
                    f"({trajs[0]}..{trajs[-1]}); Nt={Nt}; {Nop} channel(s): {labels}; "
                    f"Nsample after Nbin={Nbin} = {Nfiles // Nbin}")
        data = np.genfromtxt(os.path.join(data_path, name))
        rows, cols = (data.shape if data.ndim == 2 else (data.shape[0], 1))
        cand = [nt for nt in (8, 16, 24, 32, 48, 64, 96, 128) if rows % (nt * Nbin) == 0]
        return (f"format=flat_text; file={name!r}; rows={rows}; {cols} operator column(s). "
                f"Nt*Nbin must divide rows; with Nbin={Nbin}, consistent Nt: {cand}")
    except Exception as e:
        return f"peekDataset error: {e}"


def _dataset_stage(model, messages):
    role = ("identifying the correlator dataset to fit: where the data lives, its "
            "format and dimensions, which operator columns to fit and the fit "
            "time-ranges, based solely on user input.")
    parameter_rules = [
        """data_path: Ask the user for the directory containing the data. As soon as you have it, call the peekDataset tool with that path and report what it found (format, dimensions, channels) so subsequent answers can be checked against the real files.""",
        """format: Never ask the user for the format. Use the format peekDataset reports (flat_text or hadrons_xml) and tell the user which was detected. You may record it as "auto".""",
        """name: For hadrons_xml leave name as the empty string to auto-discover a single observable (peekDataset reports the discovered base name); only set it to a base name if peekDataset reports more than one observable. For flat_text, ask for the data file name.""",
        """Nt: Use the Nt reported by peekDataset. For hadrons_xml it is unambiguous; for flat_text confirm the intended value against the consistent-Nt list peekDataset prints.""",
        """Nbin: The number of consecutive measurements averaged into one sample. Default 1; suggest that.""",
        """i_fit: The operator column indices to include in the fit. peekDataset reports how many are available; for a single-channel dataset this is [0].""",
        """op_names: Optional plot labels. Suggest leaving this empty, in which case labels are auto-generated (for hadrons_xml from the gamma channels). If the user wants to name operators, accept either one label per fitted operator in the same order as i_fit (the usual choice), or one label per data column. Do not press for them; a single question offering the empty default is enough.""",
        """tmin_default, tmax_default: Ask for the fit time-range once (not one bound at a time). Defaults tmin=3, tmax=25.""",
        """tmin_overrides, tmax_overrides: Per-operator exceptions keyed by operator index. Default empty; only fill if the user asks.""",
    ]
    return parameterAgent(model, DatasetConfig, role, tools=[peekDataset],
                          parameter_rules=parameter_rules, input_messages=messages)


def _model_stage(model, messages):
    role = ("choosing the fit model: the number of exponential states and how the "
            "model is parametrized, based solely on user input.")
    parameter_rules = [
        """Nmass: number of normal exponential states. Default 2.""",
        """NmassAlt: number of alternating-sign exponential states (staggered-style). Default 1; suggest 0 for a plain sum of exponentials.""",
        """model_backend: 'analytic' (faster) or 'sympy'. Suggest 'analytic'.""",
        """mass_param, fac, and the *_grid multi-start lists: these have sensible defaults; suggest the defaults and only change on request.""",
    ]
    return parameterAgent(model, ModelConfig, role, tools=[],
                          parameter_rules=parameter_rules, input_messages=messages)


def _stats_stage(model, messages):
    role = ("choosing the covariance construction and resampling scheme, based "
            "solely on user input.")
    parameter_rules = [
        """cov_on: 'corr' (raw correlator) or 'meff' (log-ratio). Default 'corr'.""",
        """inner_resample, outer_resample: 'jackknife' or 'bootstrap'. Suggest the defaults (jackknife inner, bootstrap outer).""",
        """Nboots_outer: number of outer bootstrap samples for errors. Default 100; note that fewer (e.g. 20) is much faster for exploration.""",
        """LW: Ledoit-Wolf shrinkage, 0=full covariance, 1=diagonal. Default 1.0.""",
        """The remaining fields (Nboots_inner, Nboots_pval, block_size, block_circular, rng_seed) have defaults; suggest them.""",
    ]
    return parameterAgent(model, StatConfig, role, tools=[],
                          parameter_rules=parameter_rules, input_messages=messages)


def _run_stage(model, messages):
    role = ("choosing optimizer, refinement, tuning and output options, based "
            "solely on user input.")
    parameter_rules = [
        """use_de_refinement: differential-evolution refinement of the multi-start fit. Default True; note it is the slow part and False is fine for exploration.""",
        """tmin_tuning: 'none', 'optuna' or 'sa'. Default 'none'.""",
        """make_plots: whether to save diagnostic PDFs. Default True.""",
        """plot_dir, suffix: output directory and file suffix. Leave unset to use the package default ({data_path}Nbin{Nbin}_spec/); the manager config's output_dir may fill plot_dir.""",
        """The DE bound fields and optimizer have defaults; suggest them.""",
    ]
    return parameterAgent(model, RunConfig, role, tools=[],
                          parameter_rules=parameter_rules, input_messages=messages)


def fitAgent(query, model, ckpoint_file="fit_ckpoint_state.json", reload_state=False) -> FitConfig:
    """Run the staged conversation and return an assembled FitConfig. Resumes from
    ckpoint_file when reload_state and the file exists."""
    if reload_state and os.path.exists(ckpoint_file):
        state = reloadStateCheckpoint(ckpoint_file)
        query = state.query
        Print("Reloaded query from state file:", query)
    else:
        state = FitState()

    state.query = query
    messages = [HumanMessage(query)] if query else [HumanMessage("Start your workflow")]

    if state.dataset is None:
        Print("\n---\n## DATASET\n---")
        state.dataset = _dataset_stage(model, messages.copy())
        checkpointState(state, ckpoint_file)
    messages.append(HumanMessage("The chosen dataset configuration is:\n" + state.dataset.model_dump_json(indent=2)))

    if state.model is None:
        Print("\n---\n## FIT MODEL\n---")
        state.model = _model_stage(model, messages.copy())
        checkpointState(state, ckpoint_file)
    messages.append(HumanMessage("The chosen fit model is:\n" + state.model.model_dump_json(indent=2)))

    if state.stats is None:
        Print("\n---\n## COVARIANCE & RESAMPLING\n---")
        state.stats = _stats_stage(model, messages.copy())
        checkpointState(state, ckpoint_file)
    messages.append(HumanMessage("The chosen covariance/resampling is:\n" + state.stats.model_dump_json(indent=2)))

    if state.run is None:
        Print("\n---\n## RUN & OUTPUT\n---")
        state.run = _run_stage(model, messages.copy())
        checkpointState(state, ckpoint_file)

    Print("FIT AGENT COMPLETE")
    return state.toFitConfig()
