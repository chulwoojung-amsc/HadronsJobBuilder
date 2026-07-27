from femtomeas.agent_common.callgraph import Node
from femtomeas.agent_common.common import queryYesNo, Input
import copy

class State:
    def __init__(self):
        self.actions = []
        self.use_actions = []
        self.sources = []
        self.use_sources = []
        self.eigensolvers = []
        self.use_eigensolvers = []
        self.solvers = []
        self.use_solvers = []
        self.propagators = []
        self.use_propagators = []

def identifyActionsDummy(state):
    aname = f"action_{len(state.actions)}"
    print("Appending action ",aname)
    state.actions.append(aname)   
    state.use_actions = copy.deepcopy(state.actions)
    print("Using actions ", state.use_actions)

def identifySourcesDummy(state):
    sname = f"source_{len(state.sources)}"
    print("Appending source ",sname)
    state.sources.append(sname)
    state.use_sources = copy.deepcopy(state.sources)
    print("Using sources ", state.use_sources)

def setupEigenSolversDummy(state):
    touse = []
    for a in state.use_actions:
        ename = f"esolver_{a}"
        if ename not in state.eigensolvers:
            print("Appending eigensolver ", ename)
            state.eigensolvers.append(ename)
        touse.append(ename)

    state.use_eigensolvers = touse #only eigensolvers for actions we are using, regardless of what is in the state
    print("Using eigensolvers ", state.use_eigensolvers)

def identifySolversDummy(state):
    touse = []
    for a in state.use_actions:
        ename = f"esolver_{a}"
        if ename in state.eigensolvers:
            sname = f"solver_{a}_{ename}"
        else:
            sname = f"solver_{a}"

        if sname not in state.solvers:
            print("Appending solver ", sname)
            state.solvers.append(sname)
        touse.append(sname)

    state.use_solvers = touse
    print("Using solvers ", state.use_solvers)

def identifyPropagatorsDummy(state):
    touse = []
    for src in state.use_sources:
        for sol in state.use_solvers:
            pname = f"propagator_{src}_{sol}"
            if pname not in state.propagators:
                print("Appending propagator ", pname)
                state.propagators.append(pname)
            touse.append(pname)

    state.use_propagators = touse
    print("Using propagators ", state.use_propagators)



counter = 0 #for unique indexing of graph nodes
def getUniqueIdx():
    global counter
    counter += 1
    return counter - 1

class ActionsBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifyActions(*,observable_tag: str)->ActionsBlob:
    """Obtain the set of actions required for a specific observable
    inputs:
        - observable_tag: An observable_tag for a specific observable
    outputs:
        - An object describing the actions required for that observable
    """    
    return ActionsBlob(Node(f"TaskIdentifyActions_{getUniqueIdx()}", identifyActionsDummy) )

class SourcesBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifySources(*,observable_tag: str)->SourcesBlob:
    """Obtain the set of sources required for a specific observable
    inputs:
        - observable_tag: An observable_tag for a specific observable
    outputs:
        - An object describing the sources required for that observable
    """        
    return SourcesBlob(Node(f"TaskIdentifySources_{getUniqueIdx()}", identifySourcesDummy) )

class EigenSolversBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifyEigenSolvers(*,observable_tag: str, actions: ActionsBlob)->EigenSolversBlob:
    """A task to obtain the set of eigensolvers required for a specific observable
    inputs:
        - An observable_tag for a specific observable
        - The actions required for that observable
    outputs:
        - The eigensolvers required for that observable
    """    
    return EigenSolversBlob(Node(f"TaskIdentifyEigenSolvers_{getUniqueIdx()}", setupEigenSolversDummy, [actions]) )

class SolversBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifySolvers(*,observable_tag: str, actions: ActionsBlob, eigensolvers: EigenSolversBlob | None)-> SolversBlob:
    """A task to obtain the set of solvers required for a specific observable
    inputs:
      - An observable_tag for a specific observable 
      - The actions required for that observable
      - (Optional) The eigenvectors required for that observable
    outputs:
      - The solvers required for that observable    
    """
    return SolversBlob(Node(f"TaskIdentifySolvers_{getUniqueIdx()}", identifySolversDummy, [actions, eigensolvers]) )

class PropagatorsBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifyPropagators(*,observable_tag: str, sources: SourcesBlob, solvers: SolversBlob)-> PropagatorsBlob:
    """Obtain the set of propagators required for a specific observable
    inputs:
        - An observable_tag for a specific observable
        - The solvers required to compute the propagators for this observable
        - The sources required to compute the propagators for this observable
    outputs:
        - The propagators required to compute that observable
        """
    return PropagatorsBlob(Node(f"TaskIdentifyPropagators_{getUniqueIdx()}", identifyPropagatorsDummy, [sources,solvers]))

otag = "obs"
act = TaskIdentifyActions(observable_tag=otag)
esol = TaskIdentifyEigenSolvers(observable_tag=otag, actions=act)
sol = TaskIdentifySolvers(observable_tag=otag, actions=act, eigensolvers=esol)
src = TaskIdentifySources(observable_tag=otag)
props = TaskIdentifyPropagators(observable_tag=otag, sources=src, solvers=sol)

graph = props.parent_node

print(graph.gatherInfo())

state = State()
enactor = lambda f, args: f(state)

cache={}
graph.evalWithCache(cache,enactor)
print(cache)


print([i.info() for i in graph.depChain(esol.parent_node)])

print("Calling from eigensolver task")
graph.evalDepChain(esol.parent_node, cache, enactor)

print("Calling from sources task")
graph.evalDepChain(src.parent_node, cache, enactor)

print("Calling from actions task")
graph.evalDepChain(act.parent_node, cache, enactor)

assert graph.findNode(act.parent_node.name) is act.parent_node
assert graph.findNode(sol.parent_node.name) is sol.parent_node

#Test using exceptions to pause the graph
state = State()
cache = {}

class PauseGraph(Exception):
    """Pause the graph execution"""
    pass

def enactor(f, _):
    print("ENACTOR ", f.__name__)
    if f.__name__ == "identifySolversDummy":
        raise PauseGraph("Intentionally pausing at solvers")
    else:
        return f(state)

try:
    graph.evalWithCache(cache, enactor)
except PauseGraph as e:
    print("Paused graph")

print("CACHE ",cache)

all_nodes = graph.gatherTree()
for n in all_nodes:
    if n.name not in cache:
        print("Node ", n.name, " was not performed due to the pause")

#Show that when we restart the graph while holding cache it only executes the unperformed nodes
print("RESTARTING GRAPH")
enactor = lambda f, args: f(state)
graph.evalWithCache(cache, enactor) 



#Test jumping back
def decider(jumpback_points):
    jumpback = queryYesNo("Do you want to jump back?")
    if jumpback:
        print("Jumpback points: ", jumpback_points)
        pt = ""
        while(pt not in jumpback_points):
            pt = Input("Provide the jumpback node name")
        return pt
    return None

state = State()
cache = {}
graph.evalWithCacheOOO(cache, decider, enactor) 

# state = State()
# cache = {}

# #Build a map between function pointers and nodes
# fnodes = {}
# for n in all_nodes:
#    fnodes[n.func.__name__] = n

# jumpback_to = None

# def enactor(func, _):
#     ret = func(state)    

#     if len(cache) > 0:
#         jumpback = queryYesNo("Do you want to jump back?")
#         if jumpback:
#             print("Jumpback points: ", cache.keys())
#             pt = ""
#             while(pt not in cache.keys()):
#                 pt = Input("Provide the jumpback node name")
#             global jumpback_to
#             jumpback_to = pt

#             #As this node was executed, we should cache its return value to prevent recall when iterating
#             cache[ fnodes[func.__name__].name ] = ret

#             raise PauseGraph("Paused graph due to jump back")
#     return ret

# done = False
# while not done:
#     try:
#         graph.evalWithCache(cache, enactor) 
#     except PauseGraph as e:
#         ##Invalidate cache entries below and including jumpback point
#         jb = graph.findNode(jumpback_to)
#         assert jb != None
#         chain = graph.depChain(jb)
#         for c in chain:
#             cache.pop(c.name, None)

#     done=True
#     for n in all_nodes:
#         if n.name not in cache:
#             done = False
    