from femtomeas.workflow_manager.api_general import *
from femtomeas.workflow_manager.api_tools import *
from femtomeas.workflow_manager.hadrons import *
import time
import stat
import os
import json
from femtomeas.workflow_manager.iri_api import executeBatchJobTest

key_path = os.getenv("NERSC_SFAPI_KEY_PATH")
if key_path == None:
    raise Exception("Expect environment variable NERSC_SFAPI_KEY_PATH")

machine = "Perlmutter"
safe_dir = "/global/cfs/cdirs/mp13/ckelly/agent_safe_dir" #home is mounted read-only on PM
setupWorkflowAgent(key_path, { machine : safe_dir }  )

print("Machine",machine, "is up?:", queryMachineStatus(machine))

if 0:
    jobid = executeBatchJobCompat(machine, '''echo -e '#!/bin/bash\necho "Hello from ${SLURM_PROCID}"' > script.sh
    chmod u+x script.sh
    srun -n 4 ./script.sh
    ''',
    nodes=1, ranks_per_node=4, gpus_per_rank=1, time="5", queue="debug", account="amsc013_g", job_run_dir=safe_dir, exclusive=False, allow_unsafe=True)

    watchJobStatus(machine, jobid)

    
if 0:
    jobid = executeBatchJobCompat(machine, '''echo -e '#!/bin/bash\necho "Hello from ${SLURM_PROCID}"' > script.sh
    chmod u+x script.sh
    srun -n 4 ./script.sh
    ''',
    nodes=1, ranks_per_node=4, gpus_per_rank=1, time="1800", queue="debug", account="amsc013_g", job_run_dir=safe_dir, exclusive=False, allow_unsafe=True)
    getJobState(machine,jobid)
    cancelJob(machine, jobid)

    
if 0:
    test_file = "test_upload.txt"
    with open(test_file,"w") as f:
       f.write("HELLO")

    assert uploadSmallFile(machine, safe_dir + "/test1.txt", test_file ) == True

if 0: #test if it preserves executable privilege  NO
    test_script = "test_upload_script.sh"
    with open(test_script,"w") as f:
       f.write('''#!/bin/bash
       echo HELLO''')
    os.chmod(test_script, os.stat(test_script).st_mode | stat.S_IXUSR)

    assert uploadSmallFile(machine, safe_dir + "/test_upload_script.sh", test_script ) == True

if 0:
    test_script = "test_upload_script.sh"
    with open(test_script,"w") as f:
       f.write('''#!/bin/bash
       echo HELLO''')
    assert uploadSmallFile(machine, safe_dir + "/test_upload_script.sh", test_script ) == True
    remoteChmod(machine, safe_dir + "/test_upload_script.sh", "744")
    
    
    
if 0:
    caught_exception=False
    try:
       remoteMkdir(machine, "pooh", allow_unsafe=True)
    except Exception as e:
       print("Caught expected exception '",e,"'")
       caught_exception = True
    assert caught_exception

if 0:    
    ret = remoteMkdirUnsafe(machine, safe_dir + "/1/2")
    assert ret == 1 or ret == 2

setHadronsInfo({ "Perlmutter" : { "bin" : "/global/u2/c/ckelly/CPS/install_mpi_pm_new/Hadrons_pm_new/bin",    "env" : "source /global/u2/c/ckelly/CPS/bld/grid_pm_develop/sourceme.sh" } } )
#validateHadronsXML(machine, "hadrons_run.xml")

if 0:
    remoteMkdir(machine, safe_dir)
    uploadSmallFile(machine, safe_dir + "/run.xml", "hadrons_run.xml")

    script = f"""#!/bin/bash
    #SBATCH -C gpu
    #SBATCH -A mp13_g
    #SBATCH -q debug
    #SBATCH -N 1
    #SBATCH -G 1
    #SBATCH -t 5
    #SBATCH -o {safe_dir}/test_run.log

    source /global/u2/c/ckelly/CPS/bld/grid_pm_develop/sourceme.sh
    cd ${{SCRATCH}} #SQLite DB is apparently not writeable on compute nodes!?
    srun -n 1 /global/u2/c/ckelly/CPS/install_mpi_pm_new/Hadrons_pm_new/bin/HadronsXmlRun {safe_dir}/run.xml --mpi 1.1.1.1 --grid 8.8.8.8
    """

    print(script)
    jobid = executeBatchJob(machine, script)

    while(1):
        state = getJobState(machine, jobid)
        print(state)
        if state not in ("new", "queued", "active"):
            print("Detected job completion")
            break    
        time.sleep(10)

if 0:        
    jobid = submitHadronsJob(machine, "hadrons_run.xml", f"{safe_dir}/test_job", "mp13_g", "debug", "5", (16,16,8,8), (2,2,1,1))
    while(1):
        state = getJobState(machine, jobid)
        print(state)
        if state not in ("new", "queued", "active"):
            print("Detected job completion")
            break    
        time.sleep(10)

if 0:
    print(globusTransferStatus(machine, "1234"))
        
if 1:
    tid = globusCopyToMachine(machine, safe_dir, "dtn", "/global/cfs/cdirs/mp13/ckelly/globus_source_test_dir", block_until_complete=True)
    for i in range(10):
        print(globusTransferStatus(machine, tid))
        time.sleep(2)

if 0:
    globusCopyFromMachine("dtn", "/global/cfs/cdirs/mp13/ckelly/globus_source_test_dir/copyback",  machine, safe_dir + "/test.dat", block_until_complete=True)
