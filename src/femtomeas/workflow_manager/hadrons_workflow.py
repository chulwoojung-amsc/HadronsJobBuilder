from typing import Tuple
from femtomeas.meas_config_agent.state import State
from .manager import JobManager
from .globus_transfer_action import GlobusTransfer
from .hadrons_compute_action import HadronsComputeAction, HadronsJobSpec
from . import globals
from .logging import wfmanLog

from pydantic import BaseModel, Field, PositiveInt
from typing import List, Tuple, Any
from langchain.agents.middleware import before_model, after_model, AgentState
from femtomeas.agent_common.common import Print as AgentPrint, Input as AgentInput
from femtomeas.agent_common.agent_base import parameterAgent
from femtomeas.workflow_manager.api_general import getKnownMachines, getUserAccountProjects, getMachineQueues, listSpecialGlobusEndpoints

from langchain.tools import tool
from .hadrons import defaultRankGeom
from langgraph.runtime import Runtime
from langchain.agents.middleware import AgentState
    


def enqueueStandardHadronsWorkflow(state : State, jman : JobManager,
                            mpi : Tuple[int,int,int,int],
                            machine : str, group_name : str,
                            account: str, queue : str, time : str,
                            stage_out: Tuple[str,str] | None = None
                            ):
    """
    Insert a "standard" Hadrons workflow into the job manager: For each configuration in state.gauge 
    1) (If configs are not stored locally to the compute resource) Stage in the configuration
    2) Execute the measurement job
    3) (Optional) Copy the results to another machine (use stage_out)

    stage_out : If not None, provide a tuple containing the destination Globus endpoint and a path. Files will be placed in subdirectories of that path labeled by the job index
    """
    
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
            action = GlobusTransfer(dest_endpoint=machine, dest_path=cfg_staging_dir, source_endpoint=source_uuid, source_path=configs[i])
            workflow.append(action)
            override_cfgpath = cfg_staging_dir

        xml = state.toHadronsXMLsingleConf(i, override_path = override_cfgpath)
        spec = HadronsJobSpec(job_rundir=job_dir, xml=xml, grid=grid)

        workflow.append(
            HadronsComputeAction(machine=machine, account=account, queue=queue, time=time, spec=spec, mpi=mpi)
            )

        if stage_out:
            workflow.append(GlobusTransfer(dest_endpoint=stage_out[0], dest_path=stage_out[1], source_endpoint=machine, source_path=job_dir)) 
            
            
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
    Compute a suitable MPI decomposition given a specific total number of MPI ranks.

    Args:
       ranks: The total number of MPI ranks to use
    """
    geom, _ = defaultRankGeom(ranks, state_.gauge.getGrid())
    return geom
  

class JobSubmissionParameters(BaseModel):
    """Parameters for Job submission"""
    machine: str =  Field(..., description="The name of the machine on which the jobs will be executed")
    account: str = Field(..., description="The user account on the machine")
    queue: str = Field(..., description="The queue on the machine")
    duration: PositiveInt = Field(...,description="The job time in seconds")
    rank_geom: Tuple[int,int,int,int] = Field(..., description="The MPI rank decomposition of the lattice. The four integers indicate the number of ranks in the x,y,z,t directions, respectively. The total number of ranks is the product of these four numbers.")
    job_group: str = Field(...,description="A name to assign this collection of jobs.")
    copy_out: Tuple[str,str] | None = Field(...,description="A tuple containing 1) the Globus endpoint UUID and 2) the base path, for copying out results to a remote machine. Use None if and only if the user specifies that they don't want to copy out the results.")

    def check(self):       
        if self.machine not in getKnownMachines():
            return (False, f"Machine {self.machine} not in list of known machines: {getKnownMachines()}")
        elif self.account not in getUserAccountProjects(self.machine):
            return (False, f"Account {self.account} not in list of available accounts: {getUserAccountProjects(self.machine)}")
        elif self.queue not in (queues := [ q[0] for q in getMachineQueues(self.machine) ] ):
            return (False, f"Queue {self.queue} not in list of available queues: {queues}")
        return (True, "")
        
        
class ParameterCheck(BaseModel):
    missing_parameters: List[str] =  Field(..., description="The list of parameters for which the users has not specified a value.")
    


@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Hook to debug model state after calls (sadly does not trigger before processing structured output)
    """
    print(f"Model state: {state}")
    return None


def hadronsSubmissionAgent(state : State, jman : JobManager, model):
    AgentPrint("""
---        
## JOB SUBMISSION PARAMETERS
---
        """)       
       
    global state_
    state_ = state

    role = "gathering information from the user for submitting a collection of batch jobs to American Science Cloud (AmSC) compute resources via the IRI API (a REST API for controlling the compute resources)"
    
    tool_rules = [
        "For getDefaultRankGeometry, the number of MPI ranks input to this tool must be that explicitly specified by the user. Never assume or guess a number of ranks"
        ]
    parameter_rules = [
"""rank_geom:
  - First, ALWAYS ask the user to provide the MPI rank decomposition. Always include the lattice size, which you can obtain from the getLatticeSize tool. Mention that you can help them choose a decomposition. Never ask for the total number of MPI ranks unless you are helping choose the geometry as part of the workflow below.
  - rank_geom is a blocking parameter. While rank_geom is unresolved, you MUST NOT ask about or advance to any other JobSubmissionParameters field.

  - If the user asks for help choosing a decomposition, follow this workflow exactly and do not deviate:

    1) You MUST obtain the number of MPI ranks explicitly from the user.
       - NEVER infer, derive, or guess the number of ranks from any other information.
       - Check the last user response. If it contains the number of ranks, use that value. If the user has not explicitly provided the number of ranks, ask only:
         "How many MPI ranks will this job use?"

    2) After the user provides a valid integer number of ranks, you MUST immediately call getDefaultRankGeometry with that value.
    3) Your next response MUST present the suggested decomposition and ask for confirmation or an alternative decomposition.
       - Do not ask about any other parameter in this response.

    4) Stay in the rank_geom flow until the user either confirms the suggested decomposition or supplies a different valid decomposition. Repeat steps 2-4 if the user asks for a geometry with a different number of ranks.

    Never start this workflow unless the user has explicitly asked for help choosing the decomposition.""",

"""duration:
  - if the user provides a value in any unit other than seconds (e.g. minutes, hours, etc), convert the user's response to seconds then output the value in seconds and explain that you converted to seconds on a separate line before your next question
    For example
      "You:  What is the duration of each run? You can answer in any unit.
       Human: 5 mins
       You: I've converted this to 300 seconds
            <NEXT QUESTION>
      "
""",

"""job_group:         
  - In your response, on a separate line before your question, explain that this parameter distinguishes between different job collections. It is used as the name of a parent directory within the sandbox to keep different collections separate.
""",

f"""copy_out:
  For this parameter, perform the following:
   1) For your first question, you MUST ask:
      "Do you want to copy results to a remote machine for postprocessing?"  

   2) If the user says yes:
      - Ask the user to provide the globus endpoint
      - Ask the user to obtain the base path

      If the user says no:
      - set copy_out to None
    
  Note that we also accept special UUIDs {listSpecialGlobusEndpoints()} in place of regular ID strings."""
  ]

    tools = [agentGetKnownMachines,agentGetUserAccounts,
             getLatticeSize,getDefaultRankGeometry,
             agentGetMachineQueues]

    obj = parameterAgent(model, JobSubmissionParameters, role, tools, tool_rules, parameter_rules)  
    
    AgentPrint("Submitting job to workflow manager...")
    enqueueStandardHadronsWorkflow(state, jman, obj.rank_geom, obj.machine, obj.job_group, obj.account, obj.queue, str(obj.duration), obj.copy_out)
