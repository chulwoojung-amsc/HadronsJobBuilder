from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from .hadrons_xml import HadronsXML
import json
from femtomeas.agent_common.common import *
from femtomeas.agent_common.agent_base import parameterAgent
from .meas_agent_common import Gammas

class PointSource(BaseModel):
    """A point or single-location source"""
    type: Literal["point"] = "point"
    location: Tuple[NonNegativeInt,NonNegativeInt,NonNegativeInt,NonNegativeInt] = Field(..., description="The point source 4D location")

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MSource::Point")
        HadronsXML.setValue(opt, "position", spaceSeparateSeq(self.location))
        
    
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
    prop_info: str= Field(...,description="Other information associated with the input propagator provided by the user")
    
    def setXML(self,name,xml):
        pass
    
    
class SourceConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the source")
    source: Union[PointSource, WallSource,SeqGammaSource] = Field(
        ..., description="Information about the source.", discriminator='type'  # Each item must have a 'type' field. Valid values are: 'point', 'wall'  
    )
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables/propagators this source will be used for")
    
    def setXML(self,xml):
        self.source.setXML(self.name,xml)
    

class SourcesConfig(BaseModel):
    sources: List[SourceConfig] = Field(...,description="The list of source instances")



def identifySources(model, state, user_interactions: list[BaseMessage]) -> SourcesConfig:
    """
    Parse the list of messages to identify a list of propagator sources and their associated parameters
    """

    role = """identifying all lattice QCD propagator sources required to compute the propagators required for the calculation, based solely on user input.

Previous agent interactions have identified a set of observables and their required number of propagators. Sources are inputs to constructing those propagators. A source instance has a source type (e.g. point, wall) along with a set of parameters that depend on the source type. Each propagator requires a source, but can share the same source instance.    
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
    4) If the user wants to use sequential propagators, you must work with the user to identify the source for the input propagator.
        
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

      """SeqGammaSource.q, SeqGammaSource.prop_info:
  Perform the following workflow:
    1) Explain to the user that they can provide a specific propagator name that will be used as a handle, or else they can let the agent decide on the name. Tell them that they can provide other relevant information that will allow them to distinguish between different input propagators.
    2) If the user specified a propagator name, record it as the "q" parameter. If they let you decide, you must choose a new, unique name for that propagator and record it instead. The name you choose must differ from the name of the sequential source or any other source.
    3) In the "prop_info" field, record any other information provided by the user about the propagator that can be used to uniquely identify it, such as its underlying source type, mass, action, etc. If no further information is given by the user, do not prompt them to do so again; instead use an empty string for "prop_info".
    4) You MUST write to your scratchpad using scratchPadWrite tool a note to yourself that that you need to determine what the source for the input propagator is. Prefix this note with "TODO". In this note, include the name (the value of "q") as well as the source type and other details if available. 
      
  Do not confuse sources with propagators. Propagators are constructed from sources, and sequential propagators use the sink of a propagator to construct a source, thus extending the propagator's quark flow to another location.""",

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

    return parameterAgent(model, SourcesConfig, role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=user_interactions, additional_user_query_rules=additional_user_query_rules)
