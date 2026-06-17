import os
from langchain_openai import ChatOpenAI
from femtomeas.workflow_manager.manager_config import readManagerConfigFile, setupManager
from femtomeas.workflow_manager.manager import JobManager
from femtomeas.workflow_manager.hadrons_workflow import hadronsSubmissionAgent
from femtomeas.agent_common.agent_config import readI2APIkey
from femtomeas.agent_common.python_output_agent import parameterAgent
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

    return parser.parse_args()




class PointSource(BaseModel):
    """A point or single-location source"""
    type: Literal["point"] = "point"
    location: Tuple[NonNegativeInt,NonNegativeInt,NonNegativeInt,NonNegativeInt] = Field(..., description="The point source 4D location")

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Point")
        HadronsXML.setValue(opt, "position", spaceSeparateSeq(self.location))

    def check(self, state, src_names):
        return (True, "")
       
    
class WallSource(BaseModel):
    """A wall or wall-momentum (aka just "momentum") source. A wall source requires just a timeslice, whereas a wall-momentum source needs a momentum also. When describing these sources, list them as two separate types: wall and wall-momentum/momentum"""
    type: Literal["wall"] = "wall"
    timeslice: int = Field(..., description="time slice of the wall")
    momentum: Optional[Tuple[float,float,float,float]] = Field(
        None, description="Optional four-momentum"
    )

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Wall")
        HadronsXML.setValues(opt, [ ("tW",self.timeslice), ("mom", "0. 0. 0. 0." if self.momentum == None else spaceSeparateSeq(self.momentum) ) ])

    def check(self, state, src_names):
        return (True, "")
        
class SeqGammaSource(BaseModel):
    """A sequential propagator source where a source is constructed from the slice of a propagator between two timeslices with a specific gamma matrix structure and momentum,
       src_x = q_x * theta(x_3 - tA) * theta(tB - x_3) * gamma * exp(i x.mom)
    """
    type: Literal["seq_gamma"] = "seq_gamma"
    t_a : int = Field(..., description="Start timeslice of sequential source")
    t_b : int = Field(..., description="End timeslice of sequential source")
    gamma: Gammas = Field(...,description="Gamma-matrix structure of the sequential source")
    momentum: Optional[Tuple[float,float,float,float]] = Field(
        None, description="Optional four-momentum"
    )
    q: str = Field(..., description="The name of the propagator to construct the sequential source from")
    q_source_name : str = Field(..., description="The name of the source that will be used for this input propagator")
    q_prop_info: str= Field(...,description="Other information associated with the input propagator provided by the user")
    
    def setXML(self,name,xml):
        pass
    
    def check(self, state, src_names):
        if self.q_source_name not in src_names:
            return (False, f"Source {self.q_source_name} for input propagator {self.q} does not exist in the list of sources")
        return (True,"")
        
    
class SourceConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the source")
    source: Union[PointSource, WallSource,SeqGammaSource] = Field(
        ..., description="Information about the source.", discriminator='type'  # Each item must have a 'type' field. Valid values are: 'point', 'wall'  
    )
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables/propagators this source will be used for")
    
    def setXML(self,xml):
        self.source.setXML(self.name,xml)

    def check(self, state, all_sources):
        return self.source.check(state, all_sources)

def agent(model):
    role = """creating instances of SourceConfig for every propagator source required by the user.

    Perform the following workflow:
    - Ask the user what sources they want to use.
    - Create code to instantiate SourceConfig instances for each of those sources, and work with the user to obtain the parameters
    """

    parameter_rules = [
        """sources:
          
  Perform the following workflow:
    1) Check the message history to see if the source types of the required propagators has been specified.

       If the source types have not yet been specified:
         - ask the user to specify what source *types* they wish to use for which propagators. This question should not be specific to one observable or propagator; rather you should allow the user the freedom to specify information that could apply to multiple or even all propagators. In your question, list the sources that you support but do not list their associated parameters.
           For example, "Specify the sources required for the calculation (supported options: <OPTIONS>)."
         - Do not ask the user to provide parameters at this stage.
          

       If the source types have been specified:
         - indicate to the user what source types you have identified and ask them if this is correct.
         - take note of any additional information about the parameters of those sources that is contained in the message history, e.g. source timeslices or locations 
    2) Identify the set of source instances required for the calculation according to the rules below.
    3) Instantiate a SourceConfig instance for each
        
  The rules for identifying the required source instances are as follows:
  - Create a separate entry for each unique collection of source parameters, for example if the user specified propagators with point sources at [0,0,0,0] and [12,24,12,24], create two separate source instances with different source locations.
  - Create a separate entry for each source instance, even if the same source type appears multiple times with different parameters.
  - Your list must include every source instance explicitly mentioned, and only those. Do not invent instances. Do not combine instances unless the user explicitly describes them as the same.
  - Only create separate entries for sources whose source parameters differ, even if those sources will be associated with different actions in their associated propagators. For instance, if there are two action instances, 'action_a' and 'action_b' which both need wall sources with t=0, create only one wall source instance.""",


       """SourceConfig.name:
  - You must assign a unique tag/name to the instance via the SourceConfig.name field. Do not ask the user to specify a tag.
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other source instances, prefering shorter tags if possible.""",


      """SourceConfig.user_info:
  - For the 'user_info' field, you must summarize any information relevant to what observables/propagators this source will be used for provided by the user. Never ask the user to provide this parameter.
  - It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon.
  - If the user does now specify any details, use an empty string. For example, if the user specifies that this source will be used for light quark propagators, enter "use for all light quark propagators" in user_info."""
      
      """SourceConfig.source:
  - Insert the appropriate schema for the source type""",

      """SeqGammaSource.q, SeqGammaSource.q_prop_info, SeqGammaSource.q_source_name:
  For these parameters you MUST perform the following workflow:
    1) Explain to the user that they can provide a specific propagator name that will be used as a handle, or else they can let the agent decide on the name. Tell them that they can provide other relevant information that will allow them to distinguish between different input propagators.
    2) If the user specified a propagator name, record it as the "q" parameter. If they let you decide, you must choose a new, unique name for that propagator and record it instead. The name you choose must differ from the name of the sequential source or any other source.
    3) - If the user has previously specified which source to use for the input propagator "q", you MUST confirm this with the user. If accepted, use the name of this source as the "q_source_name" 
       - If the user has NOT previously specified the source or rejected your suggestion you MUST ask the user to specify what source will be used for the input propagator "q".
        - Based on their response, determine if this source already exists,
           - if so use the name of that source for q_source_name
           - if not, instantiate a new SourceConfig instance in the "sources" field of your output for this extra source and obtain its parameters from the user
    4) In the "q_prop_info" field, record any other information provided by the user about the propagator such as its mass, action, etc. If no further information is given by the user, do not prompt them to do so again; instead use an empty string for "q_prop_info".""",

      """SeqGammaSource.t_a, SeqGammaSource.t_b: Do not ask for these parameters separately. Instead, explain that t_a and t_b denote the start and end timeslices of the sequential source, and that they can be equal. Ask for both parameters to be specified together."""
        
    ]
    
    additional_user_query_rules = [    
        """"Do not require a specific format for the user's input. For example, never ask "provide each value separated by a comma" """,
        "When asking for parameters, phrase your questions to refer to groups of propagators that share the same partial set of parameters rather than specific propagators.",
        "Do not assume that the sources associated with propagators in these groups will all have the same parameters.",
        "Do not include options in your questions that accept a single parameter value for all propagators in the group. If the user wants to specify a parameter that applies to more than one propagator in the group they will do so explicitly.",
        """Use plurals for parameter names associated with groups containing more than one propagator
  Examples:
    "Provide the timeslices for the wall sources." (plural)
    "What source locations are associated with the three baryon point source propagators?" (plural)
    "What is the momentum for the wall-momentum source?" (singular)
    "What timeslices should be used for the wall sources used for the pion and vector two-point functions?" (plural)
  Use the following plurals:
    timeslice -> time slices
    location -> locations
    momentum -> momenta""",
    
        "When asking a question referring to a group, ensure your question clearly identifies the group."
        ]
    
    state = None
    def checkAll(sources):
        print("SOURCE CHECK",type(sources),len(sources),type(sources[0]) if len(sources) > 0 else None)
        src_names = [s.name for s in sources]
        val = True
        reason = ""
        
        for i in range(len(sources)):
            p = sources[i].check(state, src_names)
            if not p[0]:
                val=False
                reason += f"\nsources[{i}] ({sources[i].name}): {p[1]}"
        return (val,reason)

    return parameterAgent(model, SourceConfig, role, tools=[], parameter_rules=parameter_rules, group_validator=checkAll )



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
   
    code = agent(llm)
    print("FINAL CODE",code)
       
        
