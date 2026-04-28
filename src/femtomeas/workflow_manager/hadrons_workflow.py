from typing import Tuple
from femtomeas.meas_config_agent.state import State
from .manager import JobManager, TransferToAction, TransferFromAction, HadronsComputeAction, HadronsJobSpec
from . import globals
from .logging import wfmanLog


from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter, PositiveFloat, PositiveInt
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
import json
from femtomeas.meas_config_agent.common import getUserInput, provideInformationToUser, queryYesNo, prettyPrintPydantic, Print as AgentPrint, Input as AgentInput
from femtomeas.workflow_manager.api_general import getKnownMachines, getUserAccountProjects, getMachineQueues
from langchain.tools import tool
from .hadrons import defaultRankGeom
from langgraph.checkpoint.memory import MemorySaver

def enqueueStandardHadronsWorkflow(state : State, jman : JobManager,
                            mpi : Tuple[int,int,int,int],
                            machine : str, group_name : str,
                            account: str, queue : str, time : str):
    configs, source_uuid = state.gauge.getJobConfigurationsAndSource()
    grid = state.gauge.getGrid()
    
    if machine not in globals.remote_workdir:
        raise Exception(f"Unknown machine {machine}")
    
    job_dir = globals.remote_workdir[machine] + f"/{group_name}/<JOBID>"
    cfg_staging_dir = globals.remote_workdir[machine] + f"/{group_name}/configurations"

    wfmanLog("enqueueStandardHadronsWorkflow is queueing",len(configs),"configurations:", configs)
    for i in range(len(configs)):
        workflow = []

        #If the configs are remote they will need to staged in
        override_cfgpath = None
        if source_uuid != None and configs[i] != None:
            action = TransferToAction(source_endpoint=source_uuid, source_path=configs[i], machine=machine, dest_path=cfg_staging_dir)
            workflow.append(action)
            override_cfgpath = cfg_staging_dir

        xml = state.toHadronsXMLsingleConf(i, override_path = override_cfgpath)
        spec = HadronsJobSpec(job_rundir=job_dir, xml=xml, grid=grid)

        workflow.append(
            HadronsComputeAction(machine=machine, account=account, queue=queue, time=time, spec=spec, mpi=mpi)
            )

        #Todo: stage out

        with jman as jd:
            jd.enqueueJob(workflow, group_name)

@tool
def agentGetKnownMachines() -> List[str]:
    """
    Get a list of valid machine names.

    Note that this list of known machines is not exhaustive; rather FemtoMeas has an inbuilt mapping of known machines to their IRI API addresses. Functionality is not yet available to extend this.

    Return:
    A list of known machines
    """
    return getKnownMachines()

@tool
def agentGetUserAccounts(machine : str)-> List[str]:
    """
    Get a list of accounts on a particular machine

    Args:
    machine: The name of the machine

    Return:
    A list of user accounts
    
    """  
    return getUserAccountProjects(machine)

@tool
def agentGetMachineQueues(machine: str)->List[ Tuple[str,str] ]:
    """
    Provide a list of queues and associated information for a given machine

    Args:
    machine: The name of the machine
    
    Return: a list of string tuples, with the first tuple entry being the queue name and the second relevant information about the queue
    """
    return getMachineQueues(machine)

state_ = None

@tool
def getLatticeSize()->Tuple[int,int,int,int]:
    """
    Get the lattice size of the job as a tuple of four integers (Lx,Ly,Lz,Lt)
    """
    return state_.gauge.getGrid()

@tool
def getDefaultRankGeometry(ranks : int)->Tuple[int,int,int,int]:
    """
    Compute a suitable rank geometry for the job

    Args:
       ranks: The number of ranks to use
    """
    geom, _ = defaultRankGeom(ranks, state_.gauge.getGrid())
    return geom


class JobSubmissionParameters(BaseModel):
    """Parameters for Job submission"""
    machine: str =  Field(..., description="The name of the machine on which the jobs will be executed")
    account: str = Field(..., description="The user account on the machine")
    queue: str = Field(..., description="The queue on the machine")
    duration: int = Field(...,description="The job time in seconds")
    rank_geom: Tuple[int,int,int,int] = Field(..., description="The MPI rank decomposition of the lattice. The four integers indicate the number of ranks in the x,y,z,t directions, respectively. The total number of ranks is the product of these four numbers.")
    job_group: str = Field(...,description="A name to assign this collection of jobs.")
    
def hadronsSubmissionAgent(state : State, jman : JobManager, model):
    AgentPrint("""
---        
## JOB SUBMISSION PARAMETERS
---
        """)       
    
    
    global state_
    state_ = state
    
    sys = """
    You are an agent responsible for gathering information from the user for submitting a collection of batch jobs to American Science Cloud (AmSC) compute resources via the IRI API (a REST API for controlling the compute resources)

    Your task is to aid the user in selecting the parameters and using those to fill in a complete JobSubmissionParameters structure.

    Workflow:
    - For each parameter in JobSubmissionParameters, obtain the value by querying the user with an appropriate question.
    - You MUST ensure that the user has specified values for ALL parameters before finishing your workflow.
    - Continue asking questions until all fields are supplied or confirmed by the user
    - Before finishing your workflow, go back over the user interactions and check that all parameters have been specified explicitly by the user.
    
    Parameter rules:
    - **Never** guess a parameter. The values should always be obtained from the user. Follow the User Query rules below for questions to the user.
    - If you use a tool to obtain a value, always use that value as a suggestion to the user; never set a parameter value that is not specified or confirmed by the user.
    - Never insert a question as the value of a particular parameter.
    - Do not place a value into the final JSON unless the user explicitly provided or confirmed it

    Rules for specific parameters:
    - rank_geom:
         - First, always ask the user to provide the MPI rank decomposition
         - if the user needs reminding of the lattice size, use the tool getLatticeSize to obtain it
         - if the user wants help in choosing an appropriate decomposition, perform the following workflow. Do not skip steps.
               1) ask the user for the total number of ranks then use the getDefaultRankGeometry tool to obtain a suggested decomposition. This is only a suggestion.
               2) output this suggestion using the provideInformationToUser tool
               3) ask the user again to either accept this suggestion or provide the rank decomposition
         - once the user has specified a value, do not ask them again to confirm
    
    - duration
         - the value you put in the JSON must be in *seconds*.
         - if the user provides a value in other unit (minutes, hours, etc) do the following. Do not skip steps.
             1) convert these numbers to seconds
             2) output the value in seconds to the user the provideInformationToUser tool and explain that you converted to seconds

    - job_group         
         - Explain that this parameter distinguishes between different job collections. It is used as the name of a parent directory within the sandbox to keep different collections separate.
    
    Tool rules:
    - If a tool provides a list of valid responses, only accept values from among that list as valid choices by the user. If you list the values, ensure you only list those returned by the tool; never make up entries.
    
    User Query rules:
    - Use the getUserInput tool to query the user
    - Be brief and to the point with your questions. Prefer asking multiple consecutive questions rather than one question that requires specifying many choices.
    - If you ask a question where the user is asked to choose between a set of known options, first obtain the list of options (calling any appropriate tools) then list those options. If there are more than 6 choices, list only the first 6 and indicate that there are more options.
    - If the user responds to a query with an invalid response, always explain that the response is invalid and ask again, repeating this process until a valid response is provided. Never accept an invalid response.
    - Instead of answering your question about a parameter, the user might respond to your query with a question. If this occurs, perform the following workflow sequentially. Do not skip steps.
       1) answer the user's question using provideInformationToUser tool
       2) ask whether the user has any follow-up questions using getUserInput. If so, repeat steps 1 and 2 until the user is satisfied.
       3) repeat the original question about the parameter. Never move onto another parameter until the user has specified a value.

    Your final output must be in JSON format and adhere to the following schema:    
    """ + json.dumps(JobSubmissionParameters.model_json_schema())

    tools = [getUserInput,provideInformationToUser,
             agentGetKnownMachines,agentGetUserAccounts,
             getLatticeSize,getDefaultRankGeometry,
             agentGetMachineQueues]
    config = {"configurable": {"thread_id": "1"}}
    agent = create_agent(model=model, tools=tools, system_prompt=sys, response_format=JobSubmissionParameters, checkpointer=MemorySaver())

    user_interactions = [ HumanMessage("Start your workflow") ]
    accepted = False
    obj = None
    while(accepted == False):
        try:
            resp = agent.invoke({ "messages": user_interactions }, config=config)
            obj = resp["structured_response"]        
            user_interactions = resp['messages']
        except Exception as e:
            user_interactions = agent.get_state(config).values["messages"]
            user_interactions.append(HumanMessage(f"Encountered an error: {e}"))
            continue
            
        #Human validation
        output = f"Obtained:\n" + prettyPrintPydantic(obj)
        AgentPrint(output)

        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = AgentInput("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))

    AgentPrint("Submitting job to workflow manager...")            
    enqueueStandardHadronsWorkflow(state, jman, obj.rank_geom, obj.machine, obj.job_group, obj.account, obj.queue, str(obj.duration))
