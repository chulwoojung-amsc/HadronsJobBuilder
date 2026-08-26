from femtomeas.agent_common.callgraph import Node, Graph

if 1:
    #Test correct execution for a node that has multiple children    
    parent = Node("parent", lambda: print("Parent"))
    children = [
        Node("child1", lambda x: print("Child1"), [parent]),
        Node("child2", lambda x: print("Child2"), [parent]),
    ]
    g = Graph(children)
    g.eval()
    


if 0:
    #Check ordered arguments are all retained even if some are None
    a = Node("a", lambda: "a")
    b = Node("b", lambda: "b")
    d = Node("d", lambda: "d")
    n = Node("result", lambda aa,bb,cc,dd: print(aa,bb,cc,dd), [a,b,None,d] )  #expect "a b None d"
    n.eval()