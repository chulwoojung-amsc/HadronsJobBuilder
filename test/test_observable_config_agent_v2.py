import os
from langchain_openai import ChatOpenAI
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
from femtomeas.agent_common.agent_config import readI2APIkey
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
from femtomeas.meas_config_agent_v2.action_config_models import ActionConfig, DWFaction
from femtomeas.meas_config_agent_v2.source_config_models import SourceConfig, WallSource
from femtomeas.meas_config_agent_v2.solver_config_models import SolverConfig, RBPrecCGsolver
from femtomeas.meas_config_agent_v2.propagator_config_models import PropagatorConfig
from femtomeas.meas_config_agent_v2.smeared_prop_config_models import SmearedPropagatorConfig, WallSmear
from femtomeas.meas_config_agent_v2.observable_config import configureMeson2pt

from femtomeas.agent_common.python_update_agent import InstanceInfo

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

    #actions
    action_insts = [ ActionConfig(name="action_1", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ) ]
    state.actions = f"""
actions = [ { ", ".join([ str(a.model_dump()) for a in action_insts ]) } ]
"""
    print("ACTIONS",state.actions)

    use_action_insts = [ InstanceInfo(instance_tag="action_1", user_info="") ]
    state.observable_actions = { "pion2pt" :  f"pion2pt_actions = [{ ", ".join([ str(a.model_dump()) for a in use_action_insts ] )}]" }

    print("OBSERVABLE_ACTIONS", state.observable_actions)

    #sources
    source_insts = [ SourceConfig(name="wall_1", source=WallSource(timeslice=0) ) ]
    state.sources = f"""
sources = [ { ", ".join([ str(a.model_dump()) for a in source_insts ]) } ]
"""
    print("SOURCES",state.sources)

    use_source_insts = [ InstanceInfo(instance_tag="wall_1", user_info="") ]
    state.observable_sources = { "pion2pt" :  f"pion2pt_sources = [{ ", ".join([ str(a.model_dump()) for a in use_source_insts ] )}]" }

    #solvers
    solver_insts = [ SolverConfig(name="solv_1", action="action_1", solver_args=RBPrecCGsolver(residual=1e-8, maxIteration=10000, guesser="") ) ]
    state.solvers = f"""
solvers = [ { ", ".join([ str(a.model_dump()) for a in solver_insts ]) } ]
"""
    print("SOLVERS",state.solvers)

    use_solver_insts = [ InstanceInfo(instance_tag="solv_1", user_info="") ]
    state.observable_solvers = { "pion2pt" :  f"pion2pt_solvers = [{ ", ".join([ str(a.model_dump()) for a in use_solver_insts ] )}]" }

    if 1:
        print("""#################
        ### TEST 1 BASIC, TWO PROPS THE SAME
        #################
        """)

        #props
        prop_insts = [ PropagatorConfig(name="prop_1", source="wall_1", solver="solv_1") ]
        state.propagators = f"""
propagators = [ { ", ".join([ str(a.model_dump()) for a in prop_insts ]) } ]
    """
        print("PROPS",state.propagators)

        use_prop_insts = [ InstanceInfo(instance_tag="prop_1", user_info="use for both input/output props of the two-point function") ]
        state.observable_propagators = { "pion2pt" :  f"pion2pt_propagators = [{ ", ".join([ str(a.model_dump()) for a in use_prop_insts ] )}]" }

        state.smeared_propagators = None
        state.observable_smeared_propagators = None

        configureMeson2pt(llm, "pion2pt", "", state)
        print("OBS", state.observables)
        print("OBS CONFIG", state.observable_observable_configs["pion2pt"])


    if 1:
        print("""#################
        ### TEST 2 TWO PROPS, TWO SMEARED PROPS
        #################
        """)

        #props
        prop_insts = [ PropagatorConfig(name="prop_1", source="wall_1", solver="solv_1") ]
        state.propagators = f"""
propagators = [ { ", ".join([ str(a.model_dump()) for a in prop_insts ]) } ]
    """
        print("PROPS",state.propagators)
        use_prop_insts = [ InstanceInfo(instance_tag="prop_1", user_info="use for both input/output props of the two-point function") ]
        state.observable_propagators = { "pion2pt" :  f"pion2pt_propagators = [{ ", ".join([ str(a.model_dump()) for a in use_prop_insts ] )}]" }

        #smeared prpos
        sprop_insts = [ SmearedPropagatorConfig(name="sprop_1", input_prop="prop_1", smearing=WallSmear(momentum=None) ) ]
        state.smeared_propagators = f"""
smeared_propagators = [ { ", ".join([ str(a.model_dump()) for a in sprop_insts ]) } ]
    """
        use_sprop_insts = [ InstanceInfo(instance_tag="sprop_1", user_info="use for both input/output props of the two-point function") ]
        state.observable_smeared_propagators = { "pion2pt" :  f"pion2pt_smeared_propagators = [{ ", ".join([ str(a.model_dump()) for a in use_sprop_insts ] )}]" }

        configureMeson2pt(llm, "pion2pt", "", state)
        print("OBS", state.observables)
        print("OBS CONFIG", state.observable_observable_configs["pion2pt"])


 
