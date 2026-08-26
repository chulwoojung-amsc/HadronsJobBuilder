from femtomeas.agent_common.routing_agent import routingAgent, BaseRoutingHandle
from femtomeas.agent_common.routing_agent_registries import registerWorkflowOperation, getWorkflowCallables
from femtomeas.workflow_manager.globus_transfer_action import TransferToAction, TransferFromAction
from femtomeas.workflow_manager.hadrons_compute_action import HadronsComputeAction, HadronsJobSpec, HadronsXML
from femtomeas.agent_common.common import queryYesNo, prettyPrintPydantic, getStructuredResponse, callModelWithStructuredOutput, Print as AgentPrint, Input as AgentInput, InputMulti as AgentInputMulti

from pydantic import Field, BaseModel
import io
from langchain_openai import ChatOpenAI
import sys
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
from femtomeas.agent_common.agent_config import readI2APIkey
from femtomeas.agent_common.callgraph import Node,Graph

from enum import Enum
from bidict import bidict

def getUniqueIdx():
    return getWorkflowCallables("job_workflow").getUniqueIdx()

class FileType(Enum):
    CONFIGURATION = 1
    EIGENVECTORS = 2
    PROPAGATORS = 3
    CORRELATORS = 4

#We will use the Globus UUID as a means of identifying machines for now
class FileHandle(BaseRoutingHandle):
    def __init__(self, node, file_type : FileType, on_uuid: str):
        super().__init__(node)
        self.file_type = file_type
        self.on_uuid = on_uuid

tags_to_uuid = bidict({ 'dtn' : "9d6d994a-6d04-11e5-ba46-22000b92c6ec", "perlmutter" : "6bdc7956-fc0f-4ad2-989c-7aa5ee643a79"  })
compute_machines = ["perlmutter"]

@registerWorkflowOperation(registry_name="job_workflow")
def globusTransferFile(source_UUID_or_tag : str,
                        source_path_spec : str,
                        destination_UUID_or_tag : str,
                        destination_dir : str,
                        file_type: FileType)->FileHandle:
    """Transfer a file using Globus
    Inputs:
        source_UUID_or_tag: The Globus UUID or the tag of the source machine
        source_path_spec: The source path but with substring <IDX> which will be replaced by a configuration index later
        destination_UUID_or_tag : The Globus UUID or the tag of the destination machine        
        destination_dir: The directory to which the file will be copied
        file_type: The type of the file
    """
    if '<IDX>' not in source_path_spec:
        raise Exception("source_path_spec must contain <IDX> substring")
    
    def filePath(idx : int, workflow, plan_cache):
        fn = source_path_spec.replace('<IDX>', str(idx))   
        #Requires a workflow action
        # if destination_UUID_or_tag in compute_machines:
        #     workflow.append( TransferToAction(source_endpoint=source_UUID_or_tag, source_path=fn, machine=destination_UUID_or_tag, dest_path=destination_dir)   )
        # else:
        #     workflow.append( TransferFromAction(

        #         source_endpoint=source_UUID_or_tag, source_path=fn, machine=destination_UUID_or_tag, dest_path=destination_dir)   )

    # machine: str
    # source_path: str
    # dest_endpoint: str
    # dest_path: str
        
        return fn        
        
    return FileHandle(Node(f"transferFileToMachine_{getUniqueIdx()}", filePath), file_type, tags_to_uuid[destination_UUID_or_tag] if destination_UUID_or_tag in tags_to_uuid else destination_UUID_or_tag  )




@registerWorkflowOperation(registry_name="job_workflow")
def globusTransferFileFromHandle(source_file_handle: FileHandle,
                        destination_UUID_or_tag : str,
                        destination_dir : str)->FileHandle:
    """Transfer a file using Globus, where the details of the source file are contained within an input FileHandle
    Inputs:
        source_file_handle: The FileHandle for the source file        
        destination_UUID_or_tag : The Globus UUID or the tag of the destination machine        
        destination_dir: The directory to which the file will be copied        
    """    
    
    def filePath(idx : int, workflow, plan_cache, filename):
        #TODO: Copy filename via transfer action
        #Modify filename in output to destination_dir        
        return filename
        
    return FileHandle(Node(f"globusTransferFileFromHandle_{getUniqueIdx()}", filePath, [source_file_handle]), source_file_handle.file_type, tags_to_uuid[destination_UUID_or_tag] if destination_UUID_or_tag in tags_to_uuid else destination_UUID_or_tag  )




@registerWorkflowOperation(registry_name="job_workflow")
def useLocalFile(source_path_spec : str,
                 machine_name : str,
                 file_type: FileType)->FileHandle:
    """Use a file local to the compute machine.
    Inputs:        
        source_path_spec: The source path but with substring <IDX> which will be replaced by a configuration index later
        machine_name: A machine name from the list of known compute machines        
        file_type: The type of the file
    """
    if '<IDX>' not in source_path_spec:
        raise Exception("source_path_spec must contain <IDX> substring")
    if machine_name not in compute_machines:
        raise Exception("Unknown machine name")

    def filePath(idx : int, workflow, plan_cache):
        fn = source_path_spec.replace('<IDX>', str(idx))
        return fn   
        
    return FileHandle(Node(f"useLocalFile_{getUniqueIdx()}", filePath), file_type, tags_to_uuid[machine_name] )


class ComputeResultsHandle(BaseRoutingHandle):
    def __init__(self, node, machine: str, output_types: list[FileType]):
        super().__init__(node)        
        self.machine = machine
        self.output_types = output_types


@registerWorkflowOperation(registry_name="job_workflow")
def measurementComputation(machine_name : str,
                           configurations: FileHandle,
                           output_types : list[FileType] = [],
                           input_propagators: FileHandle | None = None,
                           input_eigenvectors: FileHandle | None = None
                           )->ComputeResultsHandle:
    """
    Perform measurement jobs using Hadrons on the provided machine. 
    Inputs:
        machine_name: The compute machine where the calculation will be performed
        configurations: A handle for the gauge configuration files. Must have file_type==FileType.CONFIGURATION.
        input_propagators: A handle for pre-computed propagators. Must have file_type==FileType.PROPAGATORS. (optional)
        input_eigenvectors: A handle for pre-computed eigenvectors.  Must have file_type==FileType.EIGENVECTORS. (optional)
        output_types: Indicate that the measurement must compute the given file types. Access the associated FileHandle objects using getComputeResultsFileHandle. If an empty array, the measurement will still be able to compute and write these data, only you won't be able to access the FileHandles for them.
    """
    def checkFileHandle(handle, expect_type, descr):
        if handle.file_type != expect_type:
            raise Exception(f"{descr} handle must have file_type=={expect_type}")
        if tags_to_uuid.inverse[handle.on_uuid] != machine_name:
            raise Exception(f"{descr} must be on machine {machine_name} (Globus UUID {tags_to_uuid[machine_name]})")

    if machine_name not in compute_machines:
        raise Exception("Unknown machine name")
        
    checkFileHandle(configurations, FileType.CONFIGURATION , "configurations")
    if input_propagators is not None:
        checkFileHandle(input_propagators, FileType.PROPAGATORS, "input_propagators")
    if input_eigenvectors is not None:
        checkFileHandle(input_eigenvectors, FileType.EIGENVECTORS, "input_eigenvectors")
        
    def compute(idx : int, workflow, plan_cache, config, props, evecs):
        #TODO: Run an agent to produce a state object here that is used in 'compute', put in cache. Should have means of optionally setting props and evecs file             
        #Should be able to reload previously defined workflows        

        meas_workflow = None        
        xml = HadronsXML() #has configuration index included
        spec = HadronsJobSpec("/path/to/job", xml, [4,4,4,4])            
        
        #Requires a workflow action
        workflow.append( HadronsComputeAction(spec=spec, mpi=[1,1,1,1], machine=machine_name, account="amsc013_g", queue="debug", time=300) )
        return meas_workflow   

    return ComputeResultsHandle(Node(f"measurementComputation_{getUniqueIdx()}", compute, [configurations, input_propagators, input_eigenvectors]  ), 
                                machine_name, output_types )


@registerWorkflowOperation(registry_name="job_workflow")
def getComputeResultsFileHandle(compute_results_handle: ComputeResultsHandle,
                                file_type: FileType):
    if file_type not in compute_results_handle.output_types:
        raise Exception("Provided FileType is not in the list of output_types provided to the measurementComputation call")

    def fileName(idx: int, workflow, plan_cache, meas_workflow):
        #TODO: Obtain appropriate path from meas_workflow State object
        return f"/path/to/output_file.{idx}"

    return FileHandle( Node(f"getComputeResultsFileHandle_{getUniqueIdx()}", fileName, [compute_results_handle]),
                      file_type, tags_to_uuid[compute_results_handle.machine]   )



#Allow for lists of configurations as well as single configurations


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

    if 1:
        extra_sys = f"""
    The following map exists between special machine tags and Globus UUIDs:
    {tags_to_uuid}

    Compute actions can only be performed on the following machines:
    {compute_machines}
    """

        graph = routingAgent(amsc_llm_0t, "job_workflow", extra_symbols={ "FileType":FileType }, additional_sys_prompt_content=extra_sys)

    if 0:
        t = globusTransferFile("dtn", "/path/to/config.<IDX>", "perlmutter", "/path/to/compute", FileType.CONFIGURATION)
        c = measurementComputation("perlmutter", t)
        graph = Graph([c])


    plan_cache={} #avoid running internal agents more than once
    for idx in range(3):
        workflow = []
        def enactor(func, args):
            return func(idx, workflow, plan_cache, *args)
        graph.eval(enactor=enactor)

    

    # def routingAgent(llm_model, registry_name :  str, role_header : str | None = None,
    #              user_query_rules : list[str] = [], code_rules : list[str] = [],
    #              additional_sys_prompt_content = "",
    #              tools=[],
    #              node_enactor=lambda func, args: func(*args) ):