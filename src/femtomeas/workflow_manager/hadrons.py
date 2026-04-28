import os
import tempfile
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from .api_general import *
from typing import Literal, Union, List, Optional, Tuple
from . import globals
from .utils import checkSafePath

hadrons_info = None

def setHadronsInfo(hadrons_info_ : dict):
    """
    Provide the information necessary to run Hadrons on a remote machine
    Args:
       hadrons_info_ : dict    machine_name -> {
                                                 "bin" : "/path/to/hadrons/bin/dir",
                                                 "env" (optional) : "Command line instructions to set up environment, e.g.  module load hadrons" }
                                               }
    """
    for m in hadrons_info_.keys():
        if m not in globals.remote_workdir:
            raise Exception("Invalid machine name")
        if "bin" not in hadrons_info_[m]:
            raise Exception("Bin dir must be provided")
        if "env" not in hadrons_info_[m]:
            hadrons_info_[m]["env"] = ""
    global hadrons_info
    hadrons_info = hadrons_info_
        

def validateHadronsXML(machine: str, hadrons_xml_file : str) -> bool:
    if hadrons_info == None:
        raise Exception("Must run setHadronsInfo")
    if machine not in hadrons_info.keys():
        raise Exception("Invalid machine")

    #create scratch dir in sandbox if not there already
    scratch_dir = globals.remote_workdir[machine] + "/scratch"
    remoteMkdirUnsafe(machine, scratch_dir)
    
    tmp_file = scratch_dir + "/hadrons_validate.xml"
    uploadSmallFile(machine, tmp_file, hadrons_xml_file)
    ret = remoteRun(machine,  f'{ hadrons_info[machine]["env"] }; { hadrons_info[machine]["bin"] }/HadronsXmlValidate { tmp_file }').strip().split('\n')

    if "Application valid" in ret[-1]:
        return True
    else:
        print(ret)
        return False

def defaultRankGeom(ranks : int, grid : Tuple[int,int,int,int]):
    #rem = ranks
    # mu = 0
    # 
    # while(rem % 2 != 1):
    #     grid[mu] *= 2
    #     rem = rem // 2
    #     mu = (mu + 1) % 4
    # grid[mu] *= rem

    rem = ranks
    geom = [1,1,1,1]
    grid_rem = list(grid)
    while(rem % 2 != 1):
        #Find a grid direction divisible by 2 and that has the smallest number of ranks in that direction
        smu=None
        for mu in range(4):
            if grid_rem[mu] % 2 == 0 and ( True if smu == None else geom[mu] < geom[smu] ):
                smu = mu
        if smu == None:
            raise Exception(f"Remaining ranks is {rem} but remaining grid {grid_rem} has no directions divisible by 2")
        
        geom[smu] *= 2
        grid_rem[smu] = grid_rem[smu] // 2        
        rem = rem // 2

    smu = None
    for mu in range(4):
        if grid_rem[mu] % rem == 0 and ( True if smu == None else geom[mu] < geom[smu] ):
            smu = mu
    if smu == None:
        raise Exception(f"Remaining ranks is {rem} but remaining grid {grid_rem} has no directions divisible by it")
    geom[smu] *= rem
    grid_rem[smu] = grid_rem[smu] // rem
   
    prod = 1    
    for mu in range(4):
        prod *= geom[mu]
        assert geom[mu] * grid_rem[mu] == grid[mu]
        
    assert(prod == ranks)
    
    return geom, grid_rem

def sizesToGridArgList(sizes : List[int]):
    if(len(sizes) == 0):
        return ""
    
    out = str(sizes[0])
    for i in range(1,len(sizes)):
        out = out + f".{sizes[i]}"
    return out

def submitHadronsJob(machine: str,
                     hadrons_xml_file : str,
                     job_run_dir : str,
                     account : str,
                     queue : str,
                     time : str,
                     grid : Tuple[int, int, int, int], 
                     mpi : Tuple[int, int, int, int] | None = None, 
                     ranks = None,
                     delete_xml_after_upload = False
                     ):
    if hadrons_info == None:
        raise Exception("Must run setHadronsInfo")
    if machine not in hadrons_info.keys():
        raise Exception("Invalid machine")
    
    if ranks == None and mpi == None:
        raise Exception("Must specify either the rank geometry or the total number of ranks")
    elif ranks == None:
        ranks = 1
        for mu in range(4):
            ranks *= mpi[mu]
            if grid[mu] % mpi[mu] != 0:
                raise Exception(f"Global lattice size {grid[mu]} in direction {mu} does not divide evenly over {mpi[mu]} ranks")
    elif mpi == None:
        mpi = defaultRankGeom(ranks, grid)

    if not checkSafePath(machine, job_run_dir):
        raise Exception(f"Provided job path {job_run_dir} is not in the sandbox")

    remoteMkdir(machine, job_run_dir)
    uploadSmallFile(machine, f"{job_run_dir}/run.xml", hadrons_xml_file)

    if delete_xml_after_upload:
        os.remove(hadrons_xml_file)        
    
    grid_str = sizesToGridArgList(grid)
    mpi_str = sizesToGridArgList(mpi)

    ###################################
    if machine == "Perlmutter":
        if ranks < 4:
            bind="" #Entire node must be allocated for verbose,map_ldom
        else:
            bind = "--cpu-bind=verbose,map_ldom:3,2,1,0"
            
        nodes = (ranks + 3) // 4
        script=f"""#!/bin/bash
#SBATCH -q {queue}
#SBATCH -C gpu
#SBATCH -A {account}
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --gpus-per-task=1
#SBATCH -N {nodes}
#SBATCH -t {time}
#SBATCH -o {job_run_dir}/run.log

now=$(date)        
echo "Hadrons job started at ${{now}}"
        
BIND="{bind}"
export MPICH_OFI_NIC_POLICY=GPU

#Hadrons uses an SQLite database that has I/O failures on Lustre. We need to actually run in a temporary directory then move everything back        
SCRATCH_DIR=${{SCRATCH}}/${{SLURM_JOB_ID}}
mkdir -p ${{SCRATCH_DIR}}        
cd ${{SCRATCH_DIR}}
        
cat <<EOF > wrap.sh
#!/bin/bash
export CUDA_VISIBLE_DEVICES=\\${{SLURM_LOCALID}}  
echo "Rank \\${{SLURM_PROCID}}, local rank \\${{SLURM_LOCALID}} : visible devices \\${{CUDA_VISIBLE_DEVICES}}"
cd ${{SCRATCH_DIR}}
\\$*
EOF
        
chmod u+x wrap.sh
{ hadrons_info[machine]["env"] }
        
srun -c 32 --gpu-bind=none ${{BIND}} -n {ranks} ./wrap.sh { hadrons_info[machine]["bin"] }/HadronsXmlRun {job_run_dir}/run.xml --mpi {mpi_str} --grid {grid_str} --accelerator-threads 8 --shm 3072 --device-mem 15360 --threads 8 --log Iterative,Message,Error,Warning,Performance --comms-overlap --comms-concurrent --shm-mpi 1
mv ${{SCRATCH_DIR}}/* {job_run_dir}/
cd {job_run_dir}        
rmdir ${{SCRATCH_DIR}}
now=$(date)
echo "Hadrons job completed at ${{now}}"
"""
    #######################################
        
    remote_script_path = f"{job_run_dir}/batch_script.sh"
    
    fd, tmp_path = tempfile.mkstemp(prefix="batch_script_", text=True, dir="/tmp", suffix=".sh")
    print("submitHadronsJob staging batch script through temporary file",tmp_path)
    with os.fdopen(fd, 'w') as f:    
        f.write(script)

    uploadSmallFile(machine, remote_script_path, tmp_path)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    #return executeBatchJob(machine, remote_script_path)
    return executeBatchJobCompat(machine, f"source {remote_script_path}", nodes=nodes, ranks_per_node=4, gpus_per_rank=1,
                                 time=time, queue=queue, account=account, job_run_dir=job_run_dir, exclusive=True, allow_unsafe=False)
