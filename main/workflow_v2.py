import os
from langchain_openai import ChatOpenAI
from femtomeas.meas_config_agent_v2.agent import measConfigAgent
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = readManagerConfigFile(args.config_file)

    local_llm = ChatOpenAI(
        model="gpt-oss-120b-GGUF",
        openai_api_key="sk-local",
        openai_api_base="http://localhost:8000/v1",
        temperature=0
    )

    amsc_llm_0t = ChatOpenAI(
        model="gpt-oss-120b",
        base_url="https://api.i2-core.american-science-cloud.org/",
        temperature=0,
        api_key = readI2APIkey(config.agent.i2api_key_path)
    )

    # nemotron = ChatNVIDIA(
    #     model="nemotron-super-3",
    #     base_url="https://api.i2-core.american-science-cloud.org/v1",
    #     temperature=0,
    #     api_key = readI2APIkey(config.agent.i2api_key_path)
    # )

    oss_20b = ChatOpenAI(
        model="gpt-oss-20b",
        base_url="https://api.i2-core.american-science-cloud.org/",
        temperature=0,
        api_key = readI2APIkey(config.agent.i2api_key_path)
    )

    
    llm = amsc_llm_0t
    #llm = nemotron
    #llm = oss_20b
    
    reload_checkpoint_file = args.reload_checkpoint if args.reload_checkpoint is not None else "ckpoint_state.json" #NB: argparse default argument (const) is only used if the arg is specified but a value not provided, not when the arg is not specified
    reload_checkpoint = args.reload_checkpoint is not None and os.path.exists(reload_checkpoint_file)

    write_xml_file = args.write_xml
    write_xml = args.write_xml is not None
   
    if not args.skip_agent:
        query = "" if reload_checkpoint else input("Describe the observables you wish to compute: ")
        state = measConfigAgent(query, llm, reload_state=reload_checkpoint, ckpoint_file=reload_checkpoint_file)
       
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
