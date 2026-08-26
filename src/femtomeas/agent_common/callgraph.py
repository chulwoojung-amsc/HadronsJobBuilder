from inspect import signature, getdoc
from typing import Literal, Union, List, Optional, Tuple, Any, Callable, Self

class PauseGraph(Exception):
    pass

class Node:
    """A callgraph node"""

    @staticmethod
    def getNode(dep_obj):
        """Dependency input objects are either nodes or objects that have a member 'parent_node'"""
        return dep_obj if isinstance(dep_obj,Node) else dep_obj.parent_node

    
    def __init__(self, name, func, input_deps = None, node_info=""):
        """node_info is extra, optional metadata"""
        self.name = name
        self.func = func
        self.node_info = node_info
        self.input_deps = None
        if input_deps is not None:
            self.input_deps = [self.getNode(i) if i is not None else None for i in input_deps ]            

    def info(self):
        return (self.name, self.node_info, self.func.__name__,signature(self.func), getdoc(self.func) )

    def _gatherInfo(self, functions):
        functions.append(self.info())
        if self.input_deps is not None:
            for n in self.input_deps:
                if n is not None:
                    n._gatherInfo(functions)

    def gatherInfo(self):
        functions = []
        self._gatherInfo(functions)
        return functions

    def _gatherTree(self, tree):
        tree.append(self)
        if self.input_deps is not None:
            for n in self.input_deps:
                if n is not None:
                    n._gatherTree(tree)

    def gatherTree(self):
        """Return all nodes in the tree"""
        tree = []
        self._gatherTree(tree)
        return tree

    def evalWithCache(self, cache, enactor : Callable = lambda f, a: f(*a) ):
        if self.name in cache:
            return cache[self.name]

        args = []
        if self.input_deps is not None:
            for n in self.input_deps:                
                args.append(n.evalWithCache(cache, enactor) if n is not None else None)
        val = enactor(self.func, args)
        cache[self.name] = val
        return val

    def eval(self, enactor : Callable = lambda f, a: f(*a)):
        cache = {}
        return self.evalWithCache(cache, enactor)
       
    def evalWithCacheOOO(self, cache, jumpback_decider: Callable,  enactor : Callable = lambda f, a: f(*a)):
        """Evaluate the graph but allow out-of-order execution, jumping back to nodes that have already been performed. A jumpback will invalidate the dependency chain below it.
        jumpback_decider: Given the node names as strings, return the node name to jump back to or None if not jumping back
        """
        #Build a map between function pointers and nodes
        tree = self.gatherTree()
        fnodes = {}
        for n in tree:
            fnodes[n.func.__name__] = n

        jumpback_to = None

        def enactorj(func, args):
            ret = enactor(func, args)

            if len(cache) > 0:
                jumpback = jumpback_decider(set(cache.keys()))
                if jumpback is not None:
                    assert jumpback in cache.keys()

                    nonlocal jumpback_to
                    jumpback_to = jumpback

                    #As this node was executed, we should cache its return value to prevent recall when iterating
                    cache[ fnodes[func.__name__].name ] = ret

                    raise PauseGraph("Paused graph due to jump back")
            return ret

        done = False
        while not done:
            try:
                self.evalWithCache(cache, enactorj) 
            except PauseGraph as e:
                ##Invalidate cache entries below and including jumpback point
                jb = self.findNode(jumpback_to)
                assert jb != None
                chain = self.depChain(jb)
                for c in chain:
                    cache.pop(c.name, None)

            done=True
            for n in tree:
                if n.name not in cache:
                    done = False

        return cache[ tree[0].name ] #graph handle always first in tree


    def isNodeInLineage(self, which: Self):
        """Is the node 'which' in the lineage of this node?"""
        if which is self:
            return True
        if self.input_deps is not None:
            for n in self.input_deps:
                if n is not None and n.isNodeInLineage(which):
                    return True
        return False

    def _depChain(self, chain: List[Self], to: Self):
        if self.isNodeInLineage(to):
            chain.append(self)
            if self.input_deps is not None:
                for n in self.input_deps:
                    if n is not None:
                        n._depChain(chain,to)

    def depChain(self, to: Self):
        """Find the dependency chain spanning from 'to' to 'this'"""
        chain = []
        self._depChain(chain, to)
        chain.reverse()
        return chain

    def evalDepChain(self, from_node: Self, cache: dict, enactor : Callable = lambda f, a: f(*a)):
        """Evaluate the dependency chain including and below "from_node" 
        The cache should contain cached return values of all previous calls to the nodes (e.g. obtain using evalWithCache)
        After calling, the cache will be updated to the new state

        This will also perform any other nodes that were not in the cache
        """
        
        #get the nodes we need to invalidate
        chain = self.depChain(from_node)
        for c in chain:
            cache.pop(c.name, None)
        return self.evalWithCache(cache, enactor)

    def findNode(self, node_name:str)->Self:
        if node_name == self.name:
            return self
        if self.input_deps is not None:
            for n in self.input_deps:
                np = n.findNode(node_name)
                if np is not None:
                    return np
        return None


class Graph:
    """A collection of nodes with connected edges, possibly disjoint"""

    def __init__(self, leaves: list[Node]):
        """leaves: The leaf nodes of the graphs"""
        self.leaves = [ Node.getNode(l) for l in leaves ]

    def evalWithCache(self, cache, enactor : Callable = lambda f, a: f(*a) ):
        ret = []
        for n in self.leaves:
            ret.append( n.evalWithCache(cache, enactor=enactor) )
        return ret

    def eval(self, enactor : Callable = lambda f, a: f(*a)):
        cache = {}
        return self.evalWithCache(cache, enactor)