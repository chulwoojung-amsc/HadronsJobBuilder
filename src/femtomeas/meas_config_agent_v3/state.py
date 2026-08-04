from femtomeas.agent_common.python_output_agent import executeCode, executeCodeAndParse

class State:
    def __init__(self):
        self.groups = {}
        self.workflows = {}

        self.actions = None #Code for generating action instances
        self.solvers = None
        self.sources = None
        self.propagators = None
        self.smeared_propagators = None
        self.observable_configs = None

    def isValidSource(self, source_name):
        r, e = executeCode(self.sources)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "sources" in r.keys()
        sources = r["sources"]

        for p in sources:
            if p["name"] == source_name:
                return True
        return False

    def isValidSolver(self, solver_name):
        r, e = executeCode(self.solvers)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "solvers" in r.keys()
        solvers = r["solvers"]

        for p in solvers:
            if p["name"] == solver_name:
                return True
        return False        


    def isValidPropagator(self, propagator_name):
        if self.propagators is None:
            return False

        r, e = executeCode(self.propagators)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "propagators" in r.keys()
        propagators = r["propagators"]

        for p in propagators:
            if p["name"] == propagator_name:
                return True
        return False


    def isValidSmearedPropagator(self, sprop_name):
        if self.smeared_propagators is None:
            return False

        r, e = executeCode(self.smeared_propagators)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "smeared_propagators" in r.keys()
        smeared_props = r["smeared_propagators"]

        for p in smeared_props:
            if p["name"] == sprop_name:
                return True
        return False