import os
from langchain_openai import ChatOpenAI
from femtomeas.meas_config_agent.agent import measConfigAgent
from femtomeas.meas_config_agent.state import reloadStateCheckpoint, State
from femtomeas.workflow_manager.manager_config import readManagerConfigFile, setupManager
from femtomeas.workflow_manager.manager import JobManager
from femtomeas.workflow_manager.hadrons_workflow import hadronsSubmissionAgent
from femtomeas.agent_common.agent_config import readI2APIkey
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import argparse

def arg_filename_true_or_none(value, default_val):
    if value is None:
        # Happens when flag is present with no value
        return default_val

    v = value.lower()
    if v in ("true", "1"):
        return default_val
    return value  # assume it's a filename
    

def checkpoint_arg(value):
    return arg_filename_true_or_none(value, "ckpoint_state.json")
def xml_arg(value):
    return arg_filename_true_or_none(value, "hadrons_run.xml")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hadrons workflow controller"
    )

    parser.add_argument('config_file', help='The path to the configuration file')
    
    parser.add_argument(
        "--reload-checkpoint",
        nargs="?",                 # 0 or 1 values
        const="ckpoint_state.json",  # used if no value provided
        type=checkpoint_arg,
        metavar="FILENAME|true",
        help="Reload a partial agent state checkpoint, resuming with the agent where previous activity left off (optionally specify filename)"
    )

    parser.add_argument(
        "--write-xml",
        nargs="?",                 # 0 or 1 values
        const="hadrons_run.xml",  # used if no value provided
        type=xml_arg,
        metavar="FILENAME|true",
        help="Write the measurement configuration in Hadrons XML format (optionally specify filename)"
    )

    parser.add_argument(
        "--execute-workflow",
        action="store_true",
        help="Activate the job manager and enqueue the workflow. Job manager will remain active until killed (safe).",
    )

    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Activate the job manager without running the configuration agent, resuming its control over existing workflows (requires --execute-workflow <config>)"
    )

    parser.add_argument(
        "--write-xml-generator",
        nargs=1,
        metavar="FILENAME",
        help="Write Python code that generates the measurement config in Hadrons XML format"
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Show diagnostic output (per-turn struct dumps, manifests, registry chatter)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress all but the essential conversation I/O"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from femtomeas.agent_common.common import setVerbosity
    setVerbosity(2 if args.verbose else 0 if args.quiet else 1)

    config = readManagerConfigFile(args.config_file)

    #LLM endpoint is environment-switchable so setup.sh / setup-BNL.sh can point the
    #agent at different backends (AmSC, BNL, a local server, ...). Defaults reproduce
    #the AmSC I2 endpoint, so unset vars keep the previous behaviour.
    #  FEMTOMEAS_LLM_MODEL     model name
    #  FEMTOMEAS_LLM_BASE_URL  OpenAI-compatible base URL (the root the client appends
    #                          to; the AmSC value has no trailing /v1)
    #  FEMTOMEAS_LLM_API_KEY   raw API key; falls back to the manager config's i2 key
    llm_model    = os.environ.get("FEMTOMEAS_LLM_MODEL", "gpt-oss-120b")
    llm_base_url = os.environ.get("FEMTOMEAS_LLM_BASE_URL", "https://api.i2-core.american-science-cloud.org/")
    llm_api_key  = os.environ.get("FEMTOMEAS_LLM_API_KEY") or readI2APIkey(config.agent.i2api_key_path)

    llm = ChatOpenAI(
        model=llm_model,
        base_url=llm_base_url,
        temperature=0,
        api_key=llm_api_key,
    )
    print(f"Agent LLM: model={llm_model} base_url={llm_base_url}")

    #--- Superseded hardcoded LLM definitions, replaced by the env-driven block
    #    above. Left commented for revival if a portion is needed. ---
    # nemotron = ChatNVIDIA(
    #     model="nemotron-super-3",
    #     base_url="https://api.i2-core.american-science-cloud.org/v1",
    #     temperature=0,
    #     api_key = readI2APIkey(config.agent.i2api_key_path)
    # )
    # oss_20b = ChatOpenAI(
    #     model="gpt-oss-20b",
    #     base_url="https://api.i2-core.american-science-cloud.org/",
    #     temperature=0,
    #     api_key = readI2APIkey(config.agent.i2api_key_path)
    # )
    # llm = amsc_llm_0t
    # #llm = nemotron
    # #llm = oss_20b

    #Always checkpoint, but overwrite input checkpoint file if reloading and continuing
    checkpoint_file = args.reload_checkpoint if args.reload_checkpoint is not None else "ckpoint_state.json" #NB: argparse default argument (const) is only used if the arg is specified but a value not provided, not when the arg is not specified
    reload_checkpoint = args.reload_checkpoint is not None and os.path.exists(checkpoint_file)

    write_xml_file = args.write_xml
    write_xml = args.write_xml is not None
   
    if not args.skip_agent:
        state = measConfigAgent(llm, 
                                input_state=reloadStateCheckpoint(checkpoint_file) if reload_checkpoint else None,
                                checkpoint_state=(True, checkpoint_file)  )
       
        if write_xml:
            state.toHadronsXML().write(write_xml_file)  #note, if the XML uses non-local files it cannot be used directly

        if args.write_xml_generator is not None:
            print(args.write_xml_generator)
            state.toXMLgeneratorCode(args.write_xml_generator[0])

    #Start the job manager
    jman = None
    if args.execute_workflow:
        setupManager(config)
        jman = JobManager("jobs.db")
        jman.start()
        
        if not args.skip_agent:
            hadronsSubmissionAgent(state, jman, llm)
        
        jman.stop() #will wait until the job queue is complete
