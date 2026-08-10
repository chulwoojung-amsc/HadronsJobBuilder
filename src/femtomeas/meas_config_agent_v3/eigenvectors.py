from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent, InstanceInfo, executeCodeAndParse
from femtomeas.meas_config_agent_v2.eigenvectors_models import EigenSolverConfig
from .state import State, registerInstanceClass
from .agent_workflows import BaseGroup, BaseGroupHandle, registerWorkflowOperation, checkValidNewGroupName, getUniqueIdx, addReservedName, getCurrentState
from femtomeas.agent_common.callgraph import Node
from .action_config import ActionGroup, ActionGroupHandle
from typing import ClassVar

registerInstanceClass("eigensolvers", EigenSolverConfig)

def identifyEigenSolvers(model, group_name: str, action_group_name: str,  state: State):     
    action_group_code = state.groups[action_group_name].code
   
    role = f"""identifying the lattice QCD Dirac operator eigenvector solvers (aka, "eigensolvers") required by the user. 
    
    Eigenvectors can be used to optionally accelerate the calculation of quark propagators (particularly for light quarks) or as an independent target for the calculation.

    You are restricted to creating eigensolver instances with the following actions:
    {action_group_code}

    where the complete set of actions is defined with the following Python code:
    {state.instances["actions"]}

    Refer to the following information about the solvers. Disregard any information you think you know that contradicts this:
        - LanczosEigenSolver
          This is the implicitly restarted Lanczos algorithm applied to the even-odd preconditioned fermion fields.
          For each iteration the solver estimates Nm = Nk + Nextra eigenvectors by running the Lanczos algorithm for a fixed number of steps then applying a transformation that compresses the interesting information into an Nk-dimension subspace, discarding the remaining Nextra spurious vectors. This process is repeated until Nstop (where Nstop <= Nk) eigenvectors have converged to a required tolerance.
          A Chebyshev polynomial filter is used to make the wanted eigenvectors large and well separated which suppressing the contribution of eigenvectors above the desired range. This filter *suppresses* eigenvalues in the range [alpha, beta] and *enhances* eigenvalues in the range [0,alpha] and [beta, infinity]. We set beta to a value above the largest eigenvalue of the (Hermitian) Dirac operator and alpha to a value just larger than the largest desired eigenvector. This suppresses eigenvectors outside of the desired, low-eigenvalue range [0,alpha]. The polynomial order is typically chosen to be sufficiently large that the in-range eigenvectors become large and well separated but not so large that the added computational cost of applying the Dirac operator more times outweights the benefit. Typical values are in the range 100-300.
          
    --------------------------
    Eigensolve instance rules
    --------------------------
    - A separate instance is required if any of the parameters differ; for instance, if eigenvectors are required for two different action instances or if solvers for the same action are required with different residual tolerances, create two separate entries.
    - Create a separate entry for each eigensolver instance, even if the same eigensolver type appears multiple times with different parameters.
    - Your list must include every eigensolver instance explicitly mentioned, and only those. Do not invent instances. Do not combine instances unless the user explicitly describes them as the same.
    - Only create separate entries for eigensolvers whose parameters differ, even if those eigensolver will be associated with different propagators.
    """
          
    parameter_rules = [
        """EigenSolverConfig.name:
  - You must assign a unique tag/name to the instance. Do not ask the user for this parameter
  - Never use the same tag for different instances.
  - The tag should include the solver type and action name and enough of the parameter values to uniquely distinguish it among the other eigensolver instances, prefering shorter tags if possible.""",

        "LanczosEigenSolver.fileStem: It is only necessary to specify a fileStem if the user wants the eigenvectors to be saved (storeEvecs == True). If they do not, you must use an empty string.",        

        "ChebyParams.Npoly: The polynomial order Npoly must be an odd integer. If the user specifies an even integer, explain the issue to the user and request an odd value."
    ]
    
    additional_user_query_rules = [
        """If the user asks for advice or help regarding which solver to use or for what parameters to use, refer to the information provided above regarding each solver.""" ]
    
    user_info_rules = """- You must use an empty string for this parameter."""

    act, _ = executeCodeAndParse(action_group_code, InstanceInfo, action_group_name) 
    used_actions = [ a.instance_tag for a in act ]

    #Validators for EigenSolverConfig
    def check(instance):
        if instance.action not in used_actions:
            return (False, "You are restricted to using actions within the provided subset")

        return instance.check(state)

    def groupCheck(eig):
        for i in range(len(eig)):
            for j in range(i+1, len(eig)):
                if eig[i].solver_args == eig[j].solver_args:
                    return (False, f"Eigensolver {eig[i].name} is the same as {eig[j].name}. Eigensolvers must be unique")
        return (True,"")    

    inst = state.getInstanceCode("eigensolvers")

    updated_eig_code, group_eig_code, _ = parameterAgent(model, EigenSolverConfig, "eigensolvers", inst.value, group_name, None, \
                                                        role, tools=[], tool_rules=[], parameter_rules=parameter_rules, additional_user_query_rules=additional_user_query_rules, instance_validator=check, group_validator=groupCheck, user_info_rules=user_info_rules)
    inst.value = updated_eig_code
    return group_eig_code

class EigenSolverGroupHandle(BaseGroupHandle):
    pass

class EigenSolverGroup(BaseGroup):
    handle_type : ClassVar[type] = EigenSolverGroupHandle
    code: str = Field(..., description="Code for generating the list of eigensolver instances in the group")
   
def createEigenSolverGroup(group_name: str, actions: ActionGroupHandle)->EigenSolverGroupHandle:
    checkValidNewGroupName(group_name)
    def doit(group_name, gactions_group_name: str):
        state, llm_model = getCurrentState()
        assert gactions_group_name in state.groups and isinstance(state.groups[gactions_group_name], ActionGroup)

        state.groups[group_name] = EigenSolverGroup(code = identifyEigenSolvers(llm_model, group_name, gactions_group_name, state ))   
        print("createEigenSolverGroup: ", state.groups[group_name].code,  "\nEigensolvers is now: ", state.instances["eigensolvers"])   
        return group_name
   
    return EigenSolverGroupHandle(group_name, Node(f"createEigenSolverGroup_{getUniqueIdx()}", lambda gactions: doit(group_name, gactions), input_deps=[actions] ) )

registerWorkflowOperation(createEigenSolverGroup)
addReservedName("eigensolvers")
