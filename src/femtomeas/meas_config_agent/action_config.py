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
import xml.etree.ElementTree as ET
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML

from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_output_agent import parameterAgent

class DWFaction(BaseModel):
    """A Domain Wall Fermion (DWF) action instance"""
    type: Literal["DWF"] = "DWF"
    Ls: int = Field(..., description="The length/size of the fifth dimension")
    mass: float = Field(..., description="the mass parameter of the action and its associated propagators")
    M5: float = Field(..., description="the M5 parameter of the action")

    def setXML(self,name,xml):
        opt = xml.addModule(name,"MAction::DWF")
        HadronsXML.setValues(opt, [ ("gauge", "gauge"), ("Ls", self.Ls), ("mass", self.mass), ("M5",self.M5), ("boundary", "1 1 1 -1"), ("twist", "0. 0. 0. 0.") ] )

       
class WilsonCloverAction(BaseModel):
    """A Wilson-Clover (aka Clover) action instance"""
    type: Literal["WilsonClover"] = "WilsonClover"
    mass: float = Field(..., description="the mass parameter of the action and its associated propagators")
    csw_r: float = Field(..., description="Clover-term coefficient c_SW^r")
    csw_t: float = Field(..., description="Clover-term coefficient c_SW^t")
                        
    def setXML(self,name,xml):
        opt = xml.addModule(name,"MAction::DWF")
        HadronsXML.setValues(opt, [ ("gauge", "gauge"), ("mass", self.mass), ("csw_r",self.csw_r), ("csw_t",self.csw_t) ] )

        ca = ET.SubElement(opt, "clover_anisotropy")
        HadronsXML.setValues(ca, [ ("isAnisotropic", "false"), ("t_direction",3), ("xi_0", "1.0"), ("nu", "1.0") ] )
                        
        HadronsXML.setValues(opt, [ ("boundary", "1 1 1 -1"), ("twist", "0. 0. 0. 0.") ] )                               
                      

    
class ActionConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the action instance")
    action: Union[DWFaction,WilsonCloverAction] = Field(..., description="Parameters of the action. Each item must have a 'type' field. Valid values are: DWF, WilsonClover",discriminator='type')
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables/propagators this action will be used for")
    
    def setXML(self,xml):
        self.action.setXML(self.name, xml)


def identifyActions(model, user_interactions: list[BaseMessage]) -> str:
    """
    Parse the list of messages to identify a list of actions and their associated parameters
    """

    role = "identifying all lattice QCD action instances required to compute the propagators required for the calculation, based solely on user input."

    parameter_rules = [
        """actions:
  Perform the following workflow:
    1) If the user has not already done so in their previous responses, ask the user to specify which action type or types they want to use. Typically a single action type is used for all propagators so you should phrase your question as if they will select just one of the options, but explain in parentheses that they are able to choose different actions if desired. In this question, *do not* list the parameters associated with those action types.
    2) Identify the set of action instances required for the calculation according to the rules below.
    3) Instantiate an ActionConfig instance for each

  The rules for identifying the required action instances are:
  - Create a separate entry for each unique collection of parameters, for example, if the user specifies DWF propagators with Ls=12, M5=1.8 and masses of 0.03 and 0.05, create two separate action instances with different mass values.    
  - Create a separate entry for each action instance, even if the action appears multiple times with different parameters.
  - Your list must include every action instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.""",
        
        "ActionConfig.action: Insert the action type (e.g. DWF, WilsonClover) associated with the instance",
        
        """ActionConfig.name:
  - You must assign a unique tag/name to the instance via the ActionConfig.name field. Do not ask the user to specify a tag.     
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other action instances, prefering shorter tags if possible.""",

        """ActionConfig.user_info:
  - Summarize any information relevant to what observables/propagators this action will be used for provided by the user. Do not ask the user to provide this summary.
  - It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon.
  - If the user does not specify any details, use an empty string. For example, if the user specifies that this action will be used for light quark propagators, enter "use for all light quark propagators" in user_info.""" ]


    return parameterAgent(model, ActionConfig, "actions", role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=user_interactions)
