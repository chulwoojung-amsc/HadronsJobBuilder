from femtomeas.meas_config_agent_v3.agent import *
from femtomeas.meas_config_agent_v2.action_config_models import DWFaction, ActionConfig
from femtomeas.agent_common.python_update_agent import InstanceInfo
from femtomeas.meas_config_agent_v2.source_config_models import SourceConfig, PointSource, WallSource
from femtomeas.meas_config_agent_v2.solver_config_models import RBPrecCGsolver, SolverConfig
from femtomeas.meas_config_agent_v2.propagator_config_models import PropagatorConfig

def subWorkflowAgent(llm_model):
    role = f"""creating a code snippet that performs the instructions provided by the user during your conversation.

When you begin your workflow, ask the user for instructions.

Your code snippet must employ the functions in the "Registry" below to construct these module instances for this particular observable. These functions act upon a hidden internal state and query the user internally. Your focus should only be on calling these tools in the appropriate order. Follow the "Code rules" below. 

Do not ask the user to confirm or accept your code.
Never ask the user to provide the code.

If the user asks you questions you must answer them.

-------------
Registry
-------------
{function_manifest()}

-------------
Code rules
-------------
- Do not import any modules or functions; assume that the functions in the registry have already been imported.
- Functions in the registry act on a hidden internal state. The inputs and outputs are merely handles for chaining the logic. Handles must all be consumed within the code snippet; do not store them in any output structures (lists, dictionaries, etc)
- Your snippet should only perform the user's instructions and nothing else. Do not add code to write outputs.
- The output handles from your code snippet should be stored in an array named 'results'. Only store the handles for the outputs (i.e. the last calls in any given chain), not the intermediaries.

-----------
Tool usage
-----------
- The workflow you define may be one of many. You have been provided tools (listGroupHandles, listWorkflows, retrieveWorkflowCode) that you can use to obtain information about previously-created groups and workflows.
- If a tool for obtaining a list returns an empty list, interpret this as meaning no elements currently exist. NEVER repeatedly call the same tool over and over if it returns an empty list.
"""

    user_query_rules = [
    "You can only ask the user questions about the sequence of registry function calls", 
    "NEVER ask the user to provide details on groups or their parameters.",
    "If there are optional steps, you MUST ask the user if they want to perform those steps; NEVER make assumptions.",
    "You are allowed to choose group names if they have not been specified by the user, unless otherwise directed. Never tell the user that you cannot choose a group name for them."
    ]

    def printCode(obj):
        return prettyPrintPydantic(obj.code)

    graphs = None
    
    def validator(obj):
        symtable_in = asteval.make_symbol_table(use_numpy=False, **registry)
        aeval = asteval.Interpreter(symtable=symtable_in)
        aeval(obj.code)

        errors = ""
        if len(aeval.error)>0:
            for err in aeval.error:
                e = err.get_error()
                errors = errors + f"{e[0]}:{e[1]}\n"
        if len(errors) > 0:
            print("USED INSTANCE CODE ERRORS", errors)
            return False, HumanMessage(f"Running your use_instance_code code produced error(s): {errors}")    

        if "results" not in aeval.symtable:
            print("RESULTS NOT IN CODE")
            return False, HumanMessage("Your code must produce an array of handles named 'results'")
        if not isinstance(aeval.symtable['results'], list):
            print("RESULTS NOT LIST")
            return False, HumanMessage("Your 'results' output must be an array")
        for h in aeval.symtable['results']:
            if not isinstance(h, BaseGroupHandle):
                print("RESULTS ELEMENT NOT GROUPHANDLE")
                return False, HumanMessage("Your 'results' output array must contain only handles")

        nonlocal graphs
        graphs = [ h.parent_node for h in aeval.symtable['results'] ]
        return True, ""

    obj = parameterAgent(llm_model, AgentOutput, role, tools=[listGroupHandles,listWorkflows, retrieveWorkflowCode], additional_user_query_rules=user_query_rules, human_validation_output_formatter=printCode, validator=validator)

    assert graphs is not None
    cache={}
    #Evaluate all output handles with caching in case they are branches from the same chain
    for g in graphs:
        g.evalWithCache(cache)

    workflow_name = AgentInput("Provide a name for this workflow")
    state.workflows[workflow_name] = (graphs, obj.code)


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

    def testEnactor(func, args):
        print("ENACTING ",func.__name__, args)
        return func(*args)

    if 0:
        #Test solver agent
        state = State()
        action_inst = ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) )
        state.actions = f"""
actions = [ {action_inst.model_dump()} ]
"""         
        state.groups["agroup1"] = ActionGroup(code = f"agroup1 = [{InstanceInfo(instance_tag="action_inst", user_info="").model_dump() } ]")

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        sg1_h = createSolverGroup("sgroup1", ag1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)

    if 0:
        #Test source agent
        state = State()
        initializeState(amsc_llm_0t, state)

        sg1_h = createSourceGroup("sgroup1")
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test propagator agent
        state = State()
        source_insts = [ SourceConfig(name="wall_0", source=WallSource(timeslice=0, momentum=(0,0,0,0)) ),  SourceConfig(name="point_0", source=PointSource(location=(0,0,0,0)) )   ]
        state.sources = f"""
sources = [ {source_insts[0].model_dump()},  {source_insts[1].model_dump()}  ]
"""         
        state.groups["source_group"] = SourceGroup(code = f"source_group = [ {InstanceInfo(instance_tag="wall_0", user_info="").model_dump() },  {InstanceInfo(instance_tag="point_0", user_info="").model_dump() }  ]")

        solv_insts = [ SolverConfig(name="solv", solver_args=RBPrecCGsolver(residual=1e-8,maxIteration=10000,guesser=""), action="action_1")]
        state.solvers = f"""
solvers = [ {solv_insts[0].model_dump()} ]
"""         
        state.groups["solver_group"] = SolverGroup(code = f"solver_group = [ {InstanceInfo(instance_tag="solv", user_info="").model_dump() } ]")


        initializeState(amsc_llm_0t, state)

        src_h = retrieveGroupHandle("source_group")
        slv_h = retrieveGroupHandle("solver_group")
        prop_h = createPropagatorGroup("prop1", src_h, slv_h)
            
        graph = prop_h.parent_node
        graph.eval(enactor=testEnactor)

    if 1:
        #Test meson2pt agent
        state = State()
        prop_insts = [ PropagatorConfig(name="prop_wall_t32", source="wall_src_t32", solver="solver"),   PropagatorConfig(name="prop_wall_t0", source="wall_src_t0", solver="solver")     ]
        state.propagators = f"""
propagators = [ {prop_insts[0].model_dump()},  {prop_insts[1].model_dump()}  ]
"""         
        state.groups["prop_group"] = PropagatorGroup(code = f"prop_group = [ {InstanceInfo(instance_tag="prop_wall_t32", user_info="").model_dump() },  {InstanceInfo(instance_tag="prop_wall_t0", user_info="").model_dump() }  ]")

        initializeState(amsc_llm_0t, state)

        prop_h = retrieveGroupHandle("prop_group")
        meson_h = createMeson2ptGroup("meson2pt_group",prop_h)

        graph = meson_h.parent_node
        graph.eval(enactor=testEnactor)


