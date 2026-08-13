from femtomeas.meas_config_agent.agent_workflows import *
from femtomeas.meas_config_agent.agent import *
from femtomeas.meas_config_agent.action_config_models import DWFaction, ActionConfig
from femtomeas.agent_common.agent_base import parameterModelCall
from femtomeas.agent_common.python_update_agent import InstanceInfo
from femtomeas.agent_common.python_output_agent import executeCodeAndParse
from femtomeas.meas_config_agent.source_config_models import SourceConfig, PointSource, WallSource
from femtomeas.meas_config_agent.solver_config_models import RBPrecCGsolver, SolverConfig
from femtomeas.meas_config_agent.propagator_config_models import PropagatorConfig
from femtomeas.meas_config_agent.smeared_prop_config_models import SmearedPropagatorConfig, WallSmear
from femtomeas.meas_config_agent.eigenvectors_models import EigenSolverConfig, LanczosEigenSolver, ChebyParams

from femtomeas.meas_config_agent.action_config import createActionGroup, ActionGroup
from femtomeas.meas_config_agent.solver_config import createSolverGroup, SolverGroup
from femtomeas.meas_config_agent.source_config import createSourceGroup, SourceGroup
from femtomeas.meas_config_agent.propagator_config import createPropagatorGroup, PropagatorGroup
from femtomeas.meas_config_agent.observable_config import createMeson2ptGroup
from femtomeas.meas_config_agent.smeared_prop_config import createSmearedPropagatorGroup, SmearedPropagatorGroup
from femtomeas.meas_config_agent.eigenvectors import createEigenSolverGroup, EigenSolverGroup

from femtomeas.meas_config_agent.state import State, reloadStateCheckpoint, checkpointState
import os
from femtomeas.meas_config_agent.gauge import GaugeFieldConfig, UnitGauge
import io
from langchain_openai import ChatOpenAI
import sys


def encodeInstances(list_name, inst):
    if not isinstance(inst, list):
        return encodeInstances(list_name, [inst])
    return f"{list_name} = [" + ", ".join([ str(a.model_dump()) for a in inst ] ) + "]"    


class SubsetCode(BaseModel):
    code : str = Field(...,description="Python code for the subset of instances")



    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Need param file")

    config = readManagerConfigFile(sys.argv[1])

    amsc_llm_0t = ChatOpenAI(
        model="gpt-oss-120b",
        base_url="https://api.i2-core.american-science-cloud.org/",
        temperature=0,
        api_key = readI2APIkey(config.agent.i2api_key_path)
    )

    source_code = """sources = [ {"name" : f"src_t{t}", "source" : { "type":"wall", "timeslice" : t, "momentum" : (0,0,0,0) } } for t in range(32)  ]"""
    sources,e = executeCodeAndParse(source_code, SourceConfig, "sources")    
    assert len(e) == 0
    source_group_code = """source_group = [ {"instance_tag": "src_t" + str(t), "user_info": "" } for t in [1,3,5,7,9,21]  ]"""
    source_group,e = executeCodeAndParse(source_group_code, InstanceInfo, "source_group")    
    assert len(e) == 0

    sources_map = { s.name : s for s in sources }
    source_group_map = { s.instance_tag for s in source_group }

    role=f""" writing Python code for defining a subset of dictionary instances.

The following code defines a set of dictionary instances:
{source_code}

You must take the input code and output new code in a similar format that defines *only* the instances with "name" fields corresponding to the "instance_tag" fields in the following:
{source_group_code}

Put the code in the "code" field of your output.

------------------------
Rules for output code:
------------------------
- The output code must only contain definitions for those instances and no others.
- Prefer shorter code output by using loops and list comprehension where reasonable.
"""

def validate(obj):
    subset_sources,e = executeCodeAndParse(obj.code, SourceConfig, "sources")
    if len(e) > 0:
        return (False, "Executing the output code failed due to: ", e)
    subset_sources_map = { s.name : s for s in subset_sources }

    #Check for extra unwanted instances
    for s in subset_sources_map.keys():
        if s not in source_group_map:
            return (False, f"Instance {s} should not be in the output code")

    #Check wanted instances and content
    for s in source_group_map:
        if s not in subset_sources_map.keys():
            return (False, f"Instance {s} is missing from the output")
        if subset_sources_map[s] != sources_map[s]:
            return (False, f"Parameters differ between input and output for instance {s}")
    
    return (True,"")

obj = parameterModelCall(amsc_llm_0t, SubsetCode, role, validator=validate, do_human_validation=False)

print(obj.code)

# def parameterModelCall(llm_model, structured_output_model : BaseModel,
#                        role: str,                    
#                        parameter_rules : List[str] = [],
#                        input_messages = [ HumanMessage("Start your workflow") ],
#                        validator : Callable | None = None                   
#                    ):

# executeCodeAndParse(code, model, output_list_name):
    #sources = [ SourceConfig(name=f"src_t{t}", source=WallSource(timeslice=t, momentum=(0,0,0,0))) for t in range(32)  ]
    #@code = encodeInstances("sources", sources)
    #print(code)    

    #