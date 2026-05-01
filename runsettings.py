import math
import random
import time
import datetime
from pathlib import Path
import shutil
import numpy as np
import os

from tornado.process import task_id

try:
    import cupy as cp
#except ImportError:
except Exception as e:
    print("Import error", e)
    cp = np
    print("cupy not installed. Using numpy.")

def init(serialcounter1=None,serialcounter2=None,serialcounter3=None,serialcounter4=None,serialcounter5=None):
    global output_path
    global slurm_id
    global task_id
    global start_phase_ini
    global scalepad
    global phases
    global saveoutputimages
    global plot_images_global

    global curvature
    global fb_global
    global fbmix_global
    global exp_amp_global_wu
    global exp_phase_global_wu
    global exp_amp_global_diff_wu
    global wu_gamma_out
    global wu_beta_in



    global final_measures_dst
    global method
    global learning_rate
    global efficiency_limit
    global efficiency_limit_scale
    global pattern_num
    #Added
    global Final_efficiency
    global Final_box_fidelity
    global Final_ion_fidelity
    global amps_current_std

    global start_phase_curve
    global Timestampstart

    global waist_scale
    global scalepad_global

    Timestampstart = datetime.datetime.now()



    saveoutputimages=False
    random.seed(42)

    #get SLURM ID
    slurm_id=2026#os.environ.get('SLURM_ARRAY_JOB_ID','noSLURM')
    task_id = 4  # int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    if serialcounter1 is not None:
        task_id = serialcounter1


    #now_folder = (datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))
    now_folder = str(slurm_id)+"_"+str(task_id)#+"_0"
    output_path = os.path.join(os.getcwd(), "sim_output",str(slurm_id), now_folder)
    # Create the directory
    os.makedirs(output_path, exist_ok=True)

    #code_dst = output_path +"/code"
    #os.makedirs(code_dst, exist_ok=True)
    #code_src = Path("/home/md76/SLM_comp/SLM_code")
    #for file in code_src.glob("*.py"):
    #    shutil.copy(file, code_dst)


    code_dst = os.path.join(os.getcwd(), "sim_output",str(slurm_id), str(slurm_id)+"_code")

    if not os.path.exists(code_dst):
        os.makedirs(code_dst)
        code_src = Path("/home/md76/SLM_comp/SLM_code")
        for file in code_src.glob("*.py"):
            shutil.copy(file, code_dst)
    else:
        print(f"Folder '{code_dst}' already exists. Skipping copy.")

    pattern_num=np.mod(task_id,10)
    print("pattern_num = ", pattern_num)

    #scalepad=4
    #start_phase definition
    #Phase of beams
    #phases=tuple(float(x) for x in np.random.rand(5)) #(np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0])
    phases=(0.0,0.0,0.0,0.0,0.0)
    curvature=0.75
    start_phase_curve=0.5#0.001 for quadratic curve
    #phases=(0.585,0.974,0.576,0.153,0.974)
    if task_id==0:
        phases=(0.0,0.0,0.0,0.0,0.0)
    if not os.path.exists(output_path+'/'+'runconfig.txt'):
        open(output_path+'/'+'runconfig.txt', 'w').close()  # Create empty file
    with open(output_path+'/'+'runconfig.txt', 'a') as f:
        f.write(f"phases={phases}\n")


    method="grad"
    exp_amp_global_wu=0.4#0.1*serialcounter1
    exp_phase_global_wu=0.4#0.1*serialcounter2
    fbmix_global=0.0
    fb_global=0.0#serialcounter2*0.1#np.random.rand()
    wu_gamma_out=0.11#0.1*serialcounter3
    wu_beta_in=0.9#0.1*serialcounter4
    exp_amp_global_diff_wu = 0.3#0.1 * serialcounter5


    learning_rate=0.008
    efficiency_limit=1#*serialcounter1
    efficiency_limit_scale =4# * serialcounter2

    waist_scale=0.55/serialcounter5 #(0.62)#0.62#1#0.55/0.45#1
    scalepad_global=2

    final_measures_dst = os.path.join(os.getcwd(), "sim_output",str(slurm_id))
    if not os.path.exists(final_measures_dst+'/'+'final_measures_grad_pattern_'+str(pattern_num)+'.txt'):
        open(final_measures_dst+'/'+'final_measures_grad_pattern_'+str(pattern_num)+'.txt', 'w').close()  # Create empty file

    plot_images_global=True


    print(f"[Init] Taskid: {task_id}")
    print(f"[Init] SLURM Job ID: {slurm_id}")
    print(f"[Init] Output path: {output_path}")

def storedata():
    import pandas as pd
    import os

    file_name = 'data_storage.csv'

    # 1. Load existing data or create a new file
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col=0)
    else:
        # Initialize with at least one column/row or empty
        df = pd.DataFrame()

    # 2. Choose row and column names for data insertion
    rownum=slurm_id
    row_name = str(rownum)

    # 3. Insert/Update data at the intersection
    #df.loc[row_name, col_name] = data_value
    df.loc[row_name, 'slurm_id'] = slurm_id
    df.loc[row_name, 'task_id'] = task_id
    df.loc[row_name, 'Timestampstart'] = Timestampstart
    df.loc[row_name, 'Fidelity_box'] = Final_box_fidelity
    df.loc[row_name, 'Fidelity_ion'] = Final_ion_fidelity
    df.loc[row_name, 'Efficiency'] = Final_efficiency
    df.loc[row_name, 'amps_current_std'] = amps_current_std
    df.loc[row_name, 'Method'] = method
    df.loc[row_name, 'Curvature'] = curvature
    df.loc[row_name, 'start_phase_curve'] = start_phase_curve
    df.loc[row_name, 'Learning_rate'] = learning_rate
    df.loc[row_name, 'Exp_amp_wu'] = exp_amp_global_wu
    df.loc[row_name, 'Exp_phase_wu'] = exp_phase_global_wu
    df.loc[row_name, 'fb_global'] = fb_global
    df.loc[row_name, 'fbmix_global'] = fbmix_global
    df.loc[row_name, 'wu_gamma_out'] = wu_gamma_out
    df.loc[row_name, 'wu_beta_in'] = wu_beta_in
    df.loc[row_name, 'exp_amp_diff_wu'] = exp_amp_global_diff_wu
    df.loc[row_name, 'comment'] = ""
    df.loc[row_name, 'waist_scale'] = waist_scale

    # 4. Save back to the file
    df.to_csv(file_name)