#!/usr/bin/env python3
"""Conversational + batch CLI for the spectrumfit correlator fitter.

Mirrors main/workflow.py: argparse -> manager config -> (staged agent, Step 3)
-> runFit. This is Step 2 of SpectrumFit_dualformat_plan.md: the batch path is
implemented, the LLM conversation is stubbed until Step 3.

Batch use (no LLM), works for both flat_text and hadrons_xml datasets:

    source setup.sh
    python3 main/fit_workflow.py main/fit_workflow_config.json \
            --reload-checkpoint main/fit_pion_16c_sdcc.json --skip-agent --execute-fit

The positional argument is the *manager* config (paths / LLM), analogous to
workflow_local.json. The --reload-checkpoint file is a serialized FitConfig - the
fit specification itself - analogous to ckpoint_state.json.
"""
import os
import argparse
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from spectrumfit import FitConfig, runFit


def buildLLM(mgr):
    """Construct the AmSC LLM with a fail-fast probe. timeout/max_retries guard
    against the silent hang seen with workflow.py."""
    from langchain_openai import ChatOpenAI
    from femtomeas.agent_common.agent_config import readI2APIkey
    if not mgr.i2api_key_path:
        raise SystemExit("The agent needs an LLM: set i2api_key_path in the manager config, "
                         "or use --skip-agent for a batch fit.")
    llm = ChatOpenAI(model=mgr.llm_model, base_url=mgr.llm_base_url, temperature=0,
                     api_key=readI2APIkey(mgr.i2api_key_path), timeout=120, max_retries=2)
    try:
        llm.invoke("ping")
    except Exception as e:
        raise SystemExit(f"LLM probe failed (endpoint/key unreachable): {e}")
    return llm


class FitManagerConfig(BaseModel):
    """Environment for the fit CLI (not the fit itself - that is FitConfig)."""
    i2api_key_path: str = Field("", description="AmSC I2 API key path; needed only for the agent conversation")
    llm_model: str = Field("gpt-oss-120b", description="LLM model name for the agent (Step 3)")
    llm_base_url: str = Field("https://api.i2-core.american-science-cloud.org/",
                              description="LLM API base URL")
    default_data_dir: str = Field("", description="Default directory offered to the dataset agent stage")
    output_dir: str = Field("", description="If set, fills FitConfig.run.plot_dir when the FitConfig leaves it unset")


def readManagerConfig(path):
    try:
        return FitManagerConfig.model_validate_json(Path(path).read_text())
    except ValidationError as e:
        raise SystemExit(f"Could not parse fit manager config {path}: {e}")


def loadFitConfig(path):
    try:
        return FitConfig.model_validate_json(Path(path).read_text())
    except ValidationError as e:
        raise SystemExit(f"Could not parse FitConfig {path}: {e}")


def arg_filename_true_or_none(value, default_val):
    if value is None:
        return default_val
    if value.lower() in ("true", "1"):
        return default_val
    return value


def checkpoint_arg(value):
    return arg_filename_true_or_none(value, "fit_ckpoint_state.json")


def config_out_arg(value):
    return arg_filename_true_or_none(value, "fit_config.json")


def parse_args():
    parser = argparse.ArgumentParser(description="spectrumfit workflow controller")
    parser.add_argument('config_file', help='Path to the fit manager config json')
    parser.add_argument(
        "--reload-checkpoint", nargs="?", const="fit_ckpoint_state.json",
        type=checkpoint_arg, metavar="FILENAME|true",
        help="Load a FitConfig (or agent checkpoint) json to resume from or run in batch")
    parser.add_argument(
        "--write-config", nargs="?", const="fit_config.json",
        type=config_out_arg, metavar="FILENAME|true",
        help="Write the assembled FitConfig as json (optionally specify filename)")
    parser.add_argument(
        "--execute-fit", action="store_true",
        help="Run runFit on the assembled FitConfig and print the summary")
    parser.add_argument(
        "--skip-agent", action="store_true",
        help="Skip the LLM conversation; take the FitConfig from --reload-checkpoint (batch mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    mgr = readManagerConfig(args.config_file)

    reload_file = args.reload_checkpoint if args.reload_checkpoint is not None else "fit_ckpoint_state.json"
    have_reload = args.reload_checkpoint is not None and os.path.exists(reload_file)

    if args.skip_agent:
        #Batch mode: --reload-checkpoint is a serialized FitConfig (the fit spec).
        if not have_reload:
            raise SystemExit("--skip-agent requires --reload-checkpoint <FitConfig.json> "
                             f"(no such file: {reload_file})")
        fit_config = loadFitConfig(reload_file)
    else:
        #Conversational mode: --reload-checkpoint is a FitState checkpoint to resume.
        from spectrumfit.fit_agent import fitAgent, reloadStateCheckpoint
        #Only stand up the LLM if a stage actually remains - a complete checkpoint
        #assembles straight to a FitConfig with no conversation.
        need_llm = True
        if have_reload:
            st = reloadStateCheckpoint(reload_file)
            need_llm = any(getattr(st, f) is None for f in ("dataset", "model", "stats", "run"))
        query = "" if have_reload else input("Describe the correlator fit you want to run: ")
        llm = buildLLM(mgr) if need_llm else None
        fit_config = fitAgent(query, llm, ckpoint_file=reload_file, reload_state=have_reload)

    #Manager output_dir is a convenience default: it fills plot_dir only when the
    #FitConfig itself does not pin one, so an explicit FitConfig always wins.
    if mgr.output_dir and fit_config.run.plot_dir is None:
        fit_config.run.plot_dir = mgr.output_dir

    if args.write_config:
        Path(args.write_config).write_text(fit_config.model_dump_json(indent=2))
        print(f"Wrote assembled FitConfig to {args.write_config}")

    if args.execute_fit:
        result = runFit(fit_config)
        print("\n==== fit summary ====")
        print(result.summary())
    elif not args.write_config:
        print("FitConfig parsed OK. Pass --execute-fit to run or --write-config to save.")


if __name__ == "__main__":
    main()
