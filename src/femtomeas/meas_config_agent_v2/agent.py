import os
from femtomeas.agent_common.common import *
from .state import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.print_pydantic_meta import llm_json_text
import femtomeas.workflow_manager as wfman
from .action_config import identifyActions
from .observable_info_models import observableSkills
from .observable_info import identifyObservables
from .source_config import identifySources
from .eigenvectors import setupEigenSolvers
from .solver_config import identifySolvers
from .propagator_config import identifyPropagators
from .smeared_prop_config import identifySmearedPropagators
from .observable_config import configureObservables
from femtomeas.meas_config_agent.gauge import identifyGaugeConfigs


from typing import Tuple

from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter, PositiveFloat, PositiveInt, create_model, model_validator
from typing import Literal, Union, List, Optional, Tuple, Any
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState, dynamic_prompt, ModelRequest
import json
from femtomeas.agent_common.common import getUserInput, provideInformationToUser, queryYesNo, prettyPrintPydantic, getStructuredResponse, Print as AgentPrint, Input as AgentInput
from femtomeas.workflow_manager.api_general import getKnownMachines, getUserAccountProjects, getMachineQueues
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
import re
import traceback

from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState,
    ExtendedModelResponse
)
from langgraph.types import Command
from typing_extensions import NotRequired
from femtomeas.agent_common.agent_base import listEnumerateStr, indentExceptFirst, promptStringList

class AgentOutput(BaseModel):
    """Structured output for the agent"""    
    question_to_user: str = Field("", description="A question posed to the user")
    answer_to_user: str = Field("", description="An answer to a question posed by the user")
    scratchpad_note: str = Field("", description="Notes kept by the agent, appended to the current scratchpad")
    done: bool = Field(..., description="The agent's workflow is complete")


class AgentState:
    def reset(self, config_state: State, llm_model, ckpoint_file: str):                
        self.scratch = []
        self.config_state = config_state
        self.llm_model = llm_model
        self.ckpoint_file = ckpoint_file

    def __init__(self):
        pass        

agent_state = AgentState()                

@tool
def observable_info(user_text: str)->str:
    """Pass the user's input 'user_text' to the observables subagent. Its response will contain structures describing any new observables identified along with associated background."""
    AgentPrint("Invoking observable identification agent...")
    new_obs = identifyObservables(agent_state.llm_model, user_text, agent_state.config_state)

    print("UPDATED OBSERVABLES", agent_state.config_state.observables.model_dump_json(), "END UPDATED OBSERVABLES")

    return f"""----------------
New observables:
----------------
{new_obs.model_dump_json()}

{observableSkills(new_obs)}
"""
          
@tool
def observable_actions(obs_tag: str, obs_action_info: str)->bool:
    """For a given observable, identify the required action module instances and store them internally.
    Parameters:
        obs_tag: The 'obs_tag' string of the observable. This must correspond to a tag in one of the JSON observable descriptions
        obs_action_info: This parameter MUST contain any information you know about the *action* types and their parameters that the user requires for this observable. Only include information about actions.
    Return:
        True if the new information gathered by the subagent invalidates the later workflow stages, requiring them to be re-run
    """
    AgentPrint(f"Invoking action agent for observable {obs_tag} with known info '{obs_action_info}'...")
    try:
        return identifyActions(agent_state.llm_model, obs_tag, obs_action_info, agent_state.config_state) 
    except Exception as e:
        print("CAUGHT EXCEPTION",e)
        raise e

@tool
def observable_sources(obs_tag: str, obs_source_info: str)->bool:
    """For a given observable, identify the required source module instances and store them internally.
    Parameters:
        obs_tag: The 'obs_tag' string of the observable. This must correspond to a tag in one of the JSON observable descriptions
        obs_source_info: This parameter MUST contain any information you know about the *source* types and their parameters that the user requires for this observable. Only include information about sources.
    Return:
        True if the new information gathered by the subagent invalidates the later workflow stages, requiring them to be re-run        
    """
    AgentPrint(f"Invoking source agent for observable {obs_tag} with known info '{obs_source_info}'...")
    try:
        return identifySources(agent_state.llm_model, obs_tag, obs_source_info, agent_state.config_state) 
    except Exception as e:
        print("CAUGHT EXCEPTION",e)
        traceback.print_exc()
        raise e

@tool
def observable_solvers(obs_tag: str, obs_solver_info: str)->bool:
    """For a given observable, identify the required solver module instances and store them internally.
    Parameters:
        obs_tag: The 'obs_tag' string of the observable. This must correspond to a tag in one of the JSON observable descriptions
        obs_solver_info: This parameter MUST contain any information you know about the *solver* types and their parameters that the user requires for this observable. Only include information about solvers.
    Return:
        True if the new information gathered by the subagent invalidates the later workflow stages, requiring them to be re-run        
    """
    AgentPrint(f"Invoking solver agent for observable {obs_tag} with known info '{obs_solver_info}'...")
    try:
        return identifySolvers(agent_state.llm_model, obs_tag, obs_solver_info, agent_state.config_state) 
    except Exception as e:
        print("CAUGHT EXCEPTION",e)
        traceback.print_exc()
        raise e


@tool
def observable_eigensolvers(obs_tag: str, obs_eigensolver_info: str)->bool:
    """For a given observable, identify the required eigensolver module instances and store them internally.
    Parameters:
        obs_tag: The 'obs_tag' string of the observable. This must correspond to a tag in one of the JSON observable descriptions
        obs_eigensolver_info: This parameter MUST contain any information you know about the *eigensolver* types and their parameters that the user requires for this observable. Only include information about eigensolvers.
    Return:
        True if the new information gathered by the subagent invalidates the later workflow stages, requiring them to be re-run        
    """
    AgentPrint(f"Invoking eigensolver agent for observable {obs_tag} with known info '{obs_eigensolver_info}'...")
    try:
        return setupEigenSolvers(agent_state.llm_model, obs_tag, obs_eigensolver_info, agent_state.config_state) 
    except Exception as e:
        print("CAUGHT EXCEPTION",e)
        traceback.print_exc()
        raise e
    
@tool
def observable_propagators(obs_tag: str, obs_prop_info: str)->bool:
    """For a given observable, identify the required propagator module instances and store them internally.
    Parameters:
        obs_tag: The 'obs_tag' string of the observable. This must correspond to a tag in one of the JSON observable descriptions
        obs_prop_info: This parameter MUST contain any information you know about the *propagators* and their parameters that the user requires for this observable. Only include information about propagators.
    Return:
        True if the new information gathered by the subagent invalidates the later workflow stages, requiring them to be re-run        
    """
    AgentPrint(f"Invoking propagator agent for observable {obs_tag} with known info '{obs_prop_info}'...")
    try:
        return identifyPropagators(agent_state.llm_model, obs_tag, obs_prop_info, agent_state.config_state) 
    except Exception as e:
        print("CAUGHT EXCEPTION",e)
        traceback.print_exc()
        raise e


@tool
def observable_smeared_propagators(obs_tag: str, obs_sprop_info: str)->bool:
    """For a given observable, identify the required smeared-propagator module instances and store them internally.
    Parameters:
        obs_tag: The 'obs_tag' string of the observable. This must correspond to a tag in one of the JSON observable descriptions
        obs_sprop_info: This parameter MUST contain any information you know about the *smeared-propagators* and their parameters that the user requires for this observable. Only include information about propagators.
    Return:
        True if the new information gathered by the subagent invalidates the later workflow stages, requiring them to be re-run        
    """
    AgentPrint(f"Invoking smeared-propagator agent for observable {obs_tag} with known info '{obs_sprop_info}'...")
    try:
        return identifySmearedPropagators(agent_state.llm_model, obs_tag, obs_sprop_info, agent_state.config_state) 
    except Exception as e:
        print("CAUGHT EXCEPTION",e)
        traceback.print_exc()
        raise e


@tool
def observable_calculation(obs_tag: str, obs_corr_info: str)->bool:
    """For a given observable, identify the required observable calculation module instances (i.e. those that compute the observable itself from the propagators/other inputs) and store them internally.
    Parameters:
        obs_tag: The 'obs_tag' string of the observable. This must correspond to a tag in one of the JSON observable descriptions
        obs_corr_info: This parameter MUST contain any information you know about the *observable calculations* and their parameters that the user requires for this observable. Only include information about observable calculations.
    Return:
        True if the new information gathered by the subagent invalidates the later workflow stages, requiring them to be re-run        
    """
    AgentPrint(f"Invoking observable calculation agent for observable {obs_tag} with known info '{obs_corr_info}'...")
    try:
        return configureObservables(agent_state.llm_model, obs_tag, obs_corr_info, agent_state.config_state) 
    except Exception as e:
        print("CAUGHT EXCEPTION",e)
        traceback.print_exc()
        raise e



def measConfigAgent(query, llm_model, ckpoint_file="state.json", reload_state=False)-> State :
    if reload_state and os.path.exists(ckpoint_file):
        state = reloadStateCheckpoint(ckpoint_file)
        query = state.query
        print("Reloaded query from state file:", query)
    else:
        state = State()

    state.query = query    
    checkpointState(state,ckpoint_file)

    print(state)

    agent_state.reset(state, llm_model, ckpoint_file)
    
    #for optional steps, call the subagent anyway as the subagent will determine whether it needs to instantiate anything
    #subagents return code snippet by handle for generating the *additional* instances
    #subagents also return routing info, like "user "

    #observable_info agent return handles for the observable types, must be passed to tools in chain

    sys = f"""
You are a router agent responsible for deciding on and executing an workflow to aid the user in constructing a lattice QCD measurement job specification. Measurement jobs are composed of module instances described alongside their parameters in an XML document that is passed to the LQCD software.

Module instances fall into classes: action modules, solver modules (for inverting the Dirac operator), eigensolver modules, source modules (for propagator sources), propagator modules and sink-smeared propagator modules.

Module dependencies form a directed graph. Most common observables such as two-point functions are built from modules belonging to the above classes with the following dependencies:
    action -> solver
    action -> eigensolver
    source, solver, (optional eigensolver) -> propagator
    propagator -> sink-smeared propagator
    propagators / sink-smeared propagators -> observables

A workflow is composed of a sequence of tool calls, each of which calls a subagent responsible for a particular class of module instances. You must call the appropriate tool to pass the task of identify the required modules in a class to the appropriate subagent.
    - Where a corresponding parameter exists, you must pass any known information about the module class instances or their parameters to the tool
    - The tools may return an indicator that later workflow stages were invalidated. This can occur if the subagent changed the existing calculation setup. If this happens, you must rerun the later workflow stages in order from where the invalidation occurred.

Your task is to identify and execute the appropriate workflow / call chain to construct the observables. This chain must be followed in the direction of dependency flow.

On each turn you must respond with structured output in the AgentOutput schema:
{AgentOutput.model_json_schema() }

Aside from making tool calls to perform these workflows, you can also ask the user questions and respond to the user's questions. 
    - Use the output field "answer_to_user" to answer a question that the user posed, if any. If the user asks a question, your response must contain an answer.
    - Use the output field "question_to_user" to ask a question.
    - You can only ask the user questions about the overall workflow. The actual module instances within each class are determined by the subagent.
    - Do not ask the user to provide module instance types or their parameters; this will be done by the subagent.
These outputs will be sent to the user and their response will be contained in the next message you receive. 

Use the "scratchpad_note" field of your output to record notes to yourself (these are not visible to the user). Refer to the scratchpad rules below for appropriate content.

Before calling a tool sequence you must first describe the plan to the user via the "answer_to_user" output field. Use the scratchpad to take note of the call sequence you constructed.

Once all workflows have been completed, signal completion using the "done" parameter in your output. 
            
-------------------------------------------
Scratchpad rules:
-------------------------------------------
    - Use the scratchpad to record TODO notes for yourself to help you plan.
"""
    
    config = {"configurable": {"thread_id": "1", "stream" : False}}

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        prompt = sys + f"""
---------------------------
Current scratchpad contents
---------------------------        
{listEnumerateStr(agent_state.scratch)}
"""
        return prompt
    
    messages = [HumanMessage(query)]
    agent = create_agent(model=llm_model, tools=[observable_info, observable_actions, observable_sources, observable_solvers, observable_eigensolvers, observable_propagators, observable_smeared_propagators, observable_calculation], middleware=[system_prompt], response_format=AgentOutput)

    accepted = False
    obj = None

    while(accepted == False):
        #Invoke the agent
        try:
            resp = agent.invoke({ "messages": messages }, config=config)
            resp_struct = getStructuredResponse(resp, AgentOutput)
        except Exception as e:
            messages.append(HumanMessage(f"Encountered an error: {e}"))
            continue

        if resp_struct.done:            
            accepted = queryYesNo("Would you like any more assistance?")
            
            if(accepted == False):
                reason = AgentInput("What would you like me to do?: ")
                agent_state.done = False
                messages.append(HumanMessage(f"The user requires you to continue your workflow for the following reason:{reason}"))
                continue
            else:
                break
        
        else:
            #Append any notes to the scratchpad
            if len(resp_struct.scratchpad_note) > 0:
                print("SCRATCHPAD NOTE", resp_struct.scratchpad_note,"END SCRATCHPAD NOTE")
                agent_state.scratch.append(resp_struct.scratchpad_note)            

            #Compose the AI message to the user
            ai_msg = resp_struct.answer_to_user + ("\n\n" if len(resp_struct.answer_to_user) > 0 else "") + resp_struct.question_to_user

            #Add it to the message history
            messages.append(AIMessage(ai_msg))

            #Obtain the user response
            user_resp = AgentInput(ai_msg)
            messages.append(HumanMessage(user_resp))

    return state

