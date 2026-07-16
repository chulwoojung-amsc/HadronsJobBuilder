import os
from langchain_openai import ChatOpenAI
from femtomeas.workflow_manager.manager_config import readManagerConfigFile, setupManager
from femtomeas.workflow_manager.manager import JobManager
from femtomeas.workflow_manager.hadrons_workflow import hadronsSubmissionAgent
from femtomeas.agent_common.agent_config import readI2APIkey
from femtomeas.agent_common.python_update_agent import parameterAgent
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import argparse

from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from femtomeas.agent_common.common import *
from femtomeas.meas_config_agent.meas_agent_common import Gammas
from femtomeas.meas_config_agent_v2.state import State
from femtomeas.meas_config_agent_v2.observable_info_models import ObservablesInfo, ObservableInfo, Meson2ptObs
from femtomeas.meas_config_agent_v2.action_config import identifyActions

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
        description="Test"
    )
    parser.add_argument('config_file', help='The path to the configuration file')

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
   
    state = State()
    state.observables = ObservablesInfo(observables=[ ObservableInfo(obs_type=Meson2ptObs(meson_type="pion"), obs_tag="pion2pt")  ])


    while(1):
        identifyActions(llm, "pion2pt", "", state)
        print("ACTIONS", state.actions)
        print("OBS ACTIONS", state.observable_actions["pion2pt"])
       
        
