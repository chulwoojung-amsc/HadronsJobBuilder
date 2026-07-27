from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent, InstanceInfo
from .eigenvectors_models import EigenSolverConfig
from .state import State

def setupEigenSolvers(model, observable_tag:str, observable_info:str, state: State): 
    """
    Setup (optional) eigensolvers
    """
    
    role = """identifying all lattice QCD Dirac operator eigenvector solvers (aka, "eigensolvers") required for the calculation of the observable, based solely on user input. Eigenvectors can be used to optionally accelerate the calculation of quark propagators (particularly for light quarks) or as an independent target for the calculation.

  Refer to the followiung information about the solvers. Disregard any information you think you know that contradicts this:
   - LanczosEigenSolver
          This is the implicitly restarted Lanczos algorithm applied to the even-odd preconditioned fermion fields.
          For each iteration the solver estimates Nm = Nk + Nextra eigenvectors by running the Lanczos algorithm for a fixed number of steps then applying a transformation that compresses the interesting information into an Nk-dimension subspace, discarding the remaining Nextra spurious vectors. This process is repeated until Nstop (where Nstop <= Nk) eigenvectors have converged to a required tolerance.
          A Chebyshev polynomial filter is used to make the wanted eigenvectors large and well separated which suppressing the contribution of eigenvectors above the desired range. This filter *suppresses* eigenvalues in the range [alpha, beta] and *enhances* eigenvalues in the range [0,alpha] and [beta, infinity]. We set beta to a value above the largest eigenvalue of the (Hermitian) Dirac operator and alpha to a value just larger than the largest desired eigenvector. This suppresses eigenvectors outside of the desired, low-eigenvalue range [0,alpha]. The polynomial order is typically chosen to be sufficiently large that the in-range eigenvectors become large and well separated but not so large that the added computational cost of applying the Dirac operator more times outweights the benefit. Typical values are in the range 100-300."""
    

    parameter_rules = [
        """solvers:
  Perform the following workflow:
           
    1. Based upon the message history determine whether the user has requested the use of eigenvectors either for any of the propagators or as an explicit calculation target
    2. If the user did not ask for eigenvectors, you must ask the user whether they wish to compute them to accelerate their calculation. An example output is as follows:
       "I have determined that you did not request the use of eigenvectors to accelerate the propagator calculation. This can provide benefits if either the eigenvectors already exist on disk or if you are computing a large number of propagators for the same action. Do you wish to use eigenvectors for the calculation?"
   
       Note that eigenvectors are commonly used when applying the all-mode averaging (AMA) method as this typically involves solving propagators on every timeslice.    
    3. If the user answers in the negative, leave the "solvers" field empty and perform no further actions.
    4. If eigenvectors are desired, ask the user to describe either which action instances or which propagators/observables to use the eigenvectors for. 
       - Ask this question only once and use the answer to infer your following actions.
       - If there is only one action instance, do not ask this question
        
    5. Using this information and the message history, determine how many eigensolver instances are required.
       - The action instances can be inferred from the propagators

   Use the following rules to determine which eigensolver instances are required:
    - A separate instance is required if any of the parameters differ; for instance, if eigenvectors are required for two different action instances or if solvers for the same action are required with different residual tolerances, create two separate entries.
    - Create a separate entry for each eigensolver instance, even if the same eigensolver type appears multiple times with different parameters.
    - Your list must include every eigensolver instance explicitly mentioned, and only those. Do not invent instances. Do not combine instances unless the user explicitly describes them as the same.
    - Only create separate entries for eigensolvers whose parameters differ, even if those eigensolver will be associated with different propagators.""",

        """EigenSolverConfig.name:
  - You must assign a unique tag/name to the instance. Do not ask the user for this parameter
  - Never use the same tag for different instances.
  - The tag should include the solver type and action name and enough of the parameter values to uniquely distinguish it among the other eigensolver instances, prefering shorter tags if possible.""",

        "LanczosEigenSolver.fileStem: It is only necessary to specify a fileStem if the user wants the eigenvectors to be saved (storeEvecs == True). If they do not, you must use an empty string.",

        "action: infer the action name based upon the information the user provided regarding the use of the eigenvectors by correlating that information with the user_info fields of the action instances. However you must ask the user to confirm your result.",

        "ChebyParams.Npoly: The polynomial order Npoly must be an odd integer. If the user specifies an even integer, explain the issue to the user and request an odd value."
    ]
    
    additional_user_query_rules = [
        """If the user asks for advice or help regarding which solver to use or for what parameters to use, refer to the information provided above regarding each solver.""" ]
    
    user_info_rules = """- You must summarize any information relevant to what observables/propagators this eigensolver will be used for provided by the user. Do not ask the user to provide this summary.
- It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon.
- If the user does not specify any details, use an empty string. For example, if the user specifies that this source will be used for light quark propagators, enter "use for all light quark propagators" in user_info."""

    #Generate instructions
    ########################
    obs_type = state.getObservableType(observable_tag)
    info = ""
    if observable_info != "":
        info = f"The following information is known about the eigensolvers for this observable:\n{observable_info}"

    instructions = f"""Perform your workflow for the observable {observable_tag} with type:
{obs_type.model_dump_json()}
{info}

The following knowledge applies to this observable:
{obs_type.skill()}

The complete set of action instances are generated using the following code:
{state.actions}

For this observable you must use the action instances described by the following code:
{state.observable_actions[observable_tag]}
"""

    ##############################################
    #Keep an expanded copy of the used-actions list
    act, _ = executeCodeAndParse(state.observable_actions[observable_tag], InstanceInfo, f"{observable_tag}_actions") 
    used_actions = [ a.instance_tag for a in act ]

    #Define the check for entries in the used-instance list that verifies the corresponding actions are still in the used-actions list
    def validateUsedInstance(m : EigenSolverConfig):
        return m.action in used_actions

    ###############################################
    #Validators for EigenSolverConfig
    def check(instance):
        return instance.check(state)

    def groupCheck(eig):
        for i in range(len(eig)):
            for j in range(i+1, len(eig)):
                if eig[i].solver_args == eig[j].solver_args:
                    return (False, f"Eigensolver {eig[i].name} is the same as {eig[j].name}. Eigensolvers must be unique")
        return (True,"")

    #################################################
    input_obs_eig_code = state.observable_eigensolvers[observable_tag] if state.observable_eigensolvers is not None and observable_tag in state.observable_eigensolvers else None

    updated_eig_code, obs_eig_code, invalidate_later_workflow_stages = parameterAgent(model, EigenSolverConfig, "eigensolvers", state.eigensolvers, f"{observable_tag}_eigensolvers", input_obs_eig_code, \
                                                                                      role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=[ HumanMessage(instructions) ], additional_user_query_rules=additional_user_query_rules, instance_validator=check, group_validator=groupCheck, user_info_rules=user_info_rules)
    state.eigensolvers = updated_eig_code

    if state.observable_eigensolvers is None:
        state.observable_eigensolvers = dict()
    state.observable_eigensolvers[observable_tag] = obs_eig_code

    return invalidate_later_workflow_stages