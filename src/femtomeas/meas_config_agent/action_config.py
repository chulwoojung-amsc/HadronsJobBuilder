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
from .hadrons_xml import HadronsXML

from .common import *


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
    
class ActionsConfig(BaseModel):
    actions: List[ActionConfig] = Field(...,description="The list of action instances")


def identifyActions(model, user_interactions: list[BaseMessage]) -> ActionsConfig:
    """
    Parse the list of messages to identify a list of actions and their associated parameters
    """

    sys = """
You are an assistant responsible for identifying all lattice QCD action instances required to compute the propagators required for the calculation, based solely on user input.

Action instances are parameterized by ActionConfig structures. Your workflow is:
1) If the user has not already done so in their previous responses, ask the user to specify which action type or types they want to use. Typically a single action type is used for all propagators so you should phrase your question as if they will select just one of the options, but explain in parentheses that they are able to choose different actions if desired. In this question, *do not* list the parameters associated with those action types.
2) Identify the set of action instances required for the calculation according to the rules below.
3) Instantiate an ActionConfig instance for each
4) populate the parameters in conjunction with the user
5) insert it into the ActionsConfig.actions list.

Action instance rules:
- Each action instance has an action type (e.g. DWF, WilsonClover) along with a set of parameters including the quark mass. When you instantiate the ActionConfig instance you will need to identify the corresponding dataclass to put in the ActionConfig.action field.
- Create a separate entry for each unique collection of parameters, for example, if the user specifies DWF propagators with Ls=12, M5=1.8 and masses of 0.03 and 0.05, create two separate action instances with different mass values.    
- Create a separate entry for each action instance, even if the action appears multiple times with different parameters.
- Your list must include every action instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.
- For the 'user_info' field, summarize any information relevant to what observables/propagators this action will be used for provided by the user. It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon. If the user does now specify any details, use an empty string. For example, if the user specifies that this action will be used for light quark propagators, enter "use for all light quark propagators" in user_info.
- You must also assign a unique tag/name to the instance via the ActionConfig.name field. Never use the same tag for different instances. The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other action instances, prefering shorter tags if possible.

User Query rules:
- Use the getUserInput tool to ask questions of the user
- If the user responds to a query with an invalid response, repeat the query until a valid response is provided. Never accept an invalid response.
- Instead of answering your question, the user might respond to your query with a question. If this occurs, answer the user's question using provideInformationToUser tool and ensure the user is satisfied with a follow-up call to getUserInput. Once satisfied, repeat the original question.


Your output must be in JSON format and adhere to the following schema:    
""" + json.dumps(ActionsConfig.model_json_schema())  ##appears necessary to also include the schema in the prompt else it hallucinates valid actions and does not validate
    
    accepted = False
    obj = None
    while(accepted == False):
        agent = create_agent(model=model, tools=[getUserInput,provideInformationToUser], system_prompt=sys, response_format=ActionsConfig)
        
        resp = agent.invoke({ "messages": user_interactions },     {"configurable": {"thread_id": "1"}})
        obj = resp["structured_response"]
        user_interactions = resp['messages']
        
        output = f"Obtained {len(obj.actions)} action instances\n" + prettyPrintPydantic(obj.actions)
        Print(output)
        
        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))            
    return obj









################## Manual implementation of tools wrapping structures. This is needed when the LLM provider structured output strategy is broken (deprecated)
    
@tool
def addDWFaction(name: str, user_info: str, Ls: int, mass: float, M5: float, runtime: ToolRuntime) -> None:
    """Add an instance of the Domain-Wall fermion (DWF) action to the list of action instances
    Args:
       name: The name/tag for the action instance
       user_info: Additional information (if any) provided by the user on what observables/propagators this action will be used for
       Ls: The length/size of the fifth dimension
       mass: the mass parameter of the action and its associated propagators
       M5: the M5 parameter of the action
    """
    storeListAppend("actions", ActionConfig(name=name, action=DWFaction(Ls=Ls, mass=mass, M5=M5), user_info=user_info), runtime.store)

@tool
def addWilsonCloverAction(name: str, user_info: str, mass: float, csw_r: float, csw_t : float, runtime: ToolRuntime) -> None:
    """Add an instance of the Domain-Wall fermion (DWF) action to the list of action instances
    Args:
       name: The name/tag for the action instance
       user_info: Additional information (if any) provided by the user on what observables/propagators this action will be used for
       mass: the mass parameter of the action and its associated propagators
       csw_r: Clover-term coefficient c_SW^r
       csw_t: Clover-term coefficient c_SW^t
    """
    storeListAppend("actions", ActionConfig(name=name, action=WilsonCloverAction(mass=mass,csw_r=csw_r, csw_t=csw_t), user_info=user_info), runtime.store)
    
       
    
def identifyActionsUsingTools(model, user_interactions: list[BaseMessage]) -> ActionsConfig:
    """
    Parse the list of messages to identify a list of actions and their associated parameters
    """

    sys = """
You are an assistant responsible for identifying all lattice QCD action instances required to compute the propagators required for the calculation, based solely on user input.

You add action instances using tool calls. An action instance has an action type (e.g. DWF, WilsonClover) along with a set of parameters including the quark mass. Create a separate entry for each unique collection of parameters, for example, if the user specifies DWF propagators with Ls=12, M5=1.8 and masses of 0.03 and 0.05, create two separate action instances with different mass values.

For each required action:
1. Identify the appropriate tool based on the action type. If the user does not specify an action type you must ask the user. Never guess an action type.
2. Call the tool using the action parameters specified by the user. If a parameter value is unknown you must ask the user; never guess parameters.

   For the 'user_info' field, summarize any information relevant to what observables/propagators this action will be used for provided by the user. It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon. If the user does now specify any details, use an empty string. For example, if the user specifies that this action will be used for light quark propagators, enter "use for all light quark propagators" in user_info.

   When calling the tool you must also assign a unique tag/name to the instance. Never use the same tag for different instances. The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other action instances, prefering shorter tags if possible.

   
Action instance rules:    
- Create a separate entry for each action instance, even if the action appears multiple times with different parameters.
- Your list must include every action instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.

User Query rules:
- Use the getUserInput tool
- If the user responds to a query with an invalid response, repeat the query until a valid response is provided. Never accept an invalid response.
- Instead of answering your question, the user might respond to your query with a question. If this occurs, answer the user's question using provideInformationToUser tool and ensure the user is satisfied with a follow-up call to getUserInput. Once satisfied, repeat the original question.
"""
    accepted = False
    obj = None
    while(accepted == False):
        store = InMemoryStore()
        agent = create_agent(model=model, tools=[getUserInput,provideInformationToUser,addDWFaction,addWilsonCloverAction], system_prompt=sys, store=store)
        
        resp = agent.invoke({ "messages": user_interactions },     {"configurable": {"thread_id": "1"}})

        obj = ActionsConfig(actions = storeGetList("actions",store))

        Print("Obtained", len(obj.actions), "action instances")
        for r in obj.actions:
            Print(r)

        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))            
    return obj
