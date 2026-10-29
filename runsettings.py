import math
import random
import time
import datetime
from pathlib import Path
import shutil
import numpy as np
import os


try:
    import cupy as cp
#except ImportError:
except Exception as e:
    print("Import error", e)
    cp = np
    print("cupy not installed. Using numpy.")

# scalepad_global = 2 # 26/05/13 gt: added due to attribute error in ArrayModulator_v1.py
# curvature = 0.75
# plot_images_global=True
# serialcounter5=None
# waist_scale=0.55/serialcounter5


def init(serialcounter1=None,serialcounter2=None,serialcounter3=None,serialcounter4=None,serialcounter5=None):
    global output_path
    global slurm_id
    global task_id
    global start_phase_ini
    global scalepad
    global phases
    global saveoutputimages
    global plot_images_global

    global n
    global m
    global amps
    global phases
    global x_pitch0
    global tem01_0
    global xarblist0
    global yarblist0
    global anglearblist0



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

    global loss_beta
    global loss_gamma

    #Added
    global Final_efficiency
    global Final_box_fidelity
    global Final_ion_fidelity
    global amps_current_std

    global loss_spots
    global loss_spots_phase
    global loss_local_phase_std

    global fidelity_ion_list
    global efficiency_ion_list
    global Maximum_possible_efficiency_complex

    global start_phase_curve
    global Timestampstart

    global waist_scale
    global scalepad_global
    global directory_SLM_computation
    global directory_SLM_Pytorch
    global directory_FlirCamera

    global first_round

    directory_SLM_computation=r'C:\Users\TrappedIonRiceDell1\Documents\SLM_computation'
    directory_SLM_Pytorch=r'C:\Users\TrappedIonRiceDell1\Documents\SLM_Pytorch'
    directory_FlirCamera=r'C:\Users\TrappedIonRiceDell1\Documents\FlirCamera'

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
    curvature=serialcounter5#0*0.575#5#0.575#75#0.75#0.575#0.75
    start_phase_curve=serialcounter4#0.0050#0.001 for quadratic curve
    #phases=(0.585,0.974,0.576,0.153,0.974)
    if task_id==0:
        phases=(0.0,0.0,0.0,0.0,0.0)
    if not os.path.exists(output_path+'/'+'runconfig.txt'):
        open(output_path+'/'+'runconfig.txt', 'w').close()  # Create empty file
    with open(output_path+'/'+'runconfig.txt', 'a') as f:
        f.write(f"phases={phases}\n")

    first_round=True


    method="grad" #grad
    exp_amp_global_wu=0.1
    exp_phase_global_wu=0.1
    fbmix_global=0.1
    fb_global=0.1#serialcounter2*0.1#np.random.rand()
    wu_gamma_out=0.1#serialcounter3
    wu_beta_in=0.1#serialcounter4
    exp_amp_global_diff_wu = 0.1#serialcounter5

    loss_beta = serialcounter3
    loss_gamma=0.0


    learning_rate=0.002#0.002 used for old datas and for 33 by 33
    efficiency_limit=0.1#*serialcounter1
    efficiency_limit_scale =10#* serialcounter2

    waist_scale=0.55/0.62*0.97466906054#0.62#1#0.55/0.45#1
    scalepad_global=2

    final_measures_dst = os.path.join(os.getcwd(), "sim_output",str(slurm_id))
    if not os.path.exists(final_measures_dst+'/'+'final_measures_grad_pattern_'+str(pattern_num)+'.txt'):
        open(final_measures_dst+'/'+'final_measures_grad_pattern_'+str(pattern_num)+'.txt', 'w').close()  # Create empty file

    plot_images_global=True


    print(f"[Init] Taskid: {task_id}")
    print(f"[Init] SLURM Job ID: {slurm_id}")
    print(f"[Init] Output path: {output_path}")

def storedata_csv():
    import pandas as pd
    import os

    file_name = 'data_storage_temp.csv'

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
    df.loc[row_name, 'efficiency_limit'] = efficiency_limit
    print("fidelity_ion_list",fidelity_ion_list)

    # Force the column to support arbitrary objects (like lists)
    df['fidelity_ion_list'] = df['fidelity_ion_list'].astype(object)

    # Assign the list to the row (remove the extra brackets if fidelity_ion_list is already a list)
    df.at[row_name, 'fidelity_ion_list'] = fidelity_ion_list[0]
    # Force the column to support arbitrary objects
    df['efficiency_ion_list'] = df['efficiency_ion_list'].astype(object)

    # Assign the list using .at for a single cell assignment
    df.at[row_name, 'efficiency_ion_list'] = efficiency_ion_list[0]



    # 4. Save back to the file
    df.to_csv(file_name)

import pandas as pd

import os
import json
import numpy as np  # Ensure numpy is imported for the helper function

import os
import json
import numpy as np  # Ensure numpy is imported for the helper function


def storedata():
    target_dir = 'JSONoutput'
    log_file = os.path.join(target_dir, 'data_storage_zero_curve_temp.jsonl') # _11by11_curvature_change.jsonl')

    # 1. Create the directory if it doesn't exist to prevent FileNotFoundError
    os.makedirs(target_dir, exist_ok=True)

    # Helper function to convert arrays, tensors, complex numbers, and objects to JSON-serializable types
    def to_serializable(val):
        # Handle PyTorch Tensors (checks for CUDA/CPU and detaches)
        if hasattr(val, 'detach'):
            val = val.detach().cpu()

        if hasattr(val, 'item'):  # Safely unpacks NumPy scalars and PyTorch Tensors
            try:
                return to_serializable(val.item())
            except ValueError:  # If it's a multi-element tensor/array, convert to a list
                val = val.tolist()
        elif isinstance(val, np.ndarray):
            val = val.tolist()

        # CRITICAL: Handle complex numbers (native python 'complex' or numpy complex types)
        if isinstance(val, (complex, np.complex64, np.complex128)):
            return {"real": float(val.real), "imag": float(val.imag)}

        # Automatically flatten nested lists like [[0.012, 0.024, ...]] down to [0.012, 0.024, ...]
        if isinstance(val, list):
            # Recursively apply serialization to every item inside the list (handles lists of complex numbers)
            val = [to_serializable(x) for x in val]
            if len(val) == 1 and isinstance(val[0], list):
                return val[0]
            return val

        return val  # Return as-is if it's already a regular float, int, or string

    # Build your run metrics dictionary, wrapping numbers with the helper
    run_metrics = {
        'slurm_id': str(slurm_id),
        'task_id': task_id,
        'Timestampstart': str(Timestampstart),
        'n': to_serializable(n),
        'm': to_serializable(m),
        'tem01_0': to_serializable(tem01_0),
        'Fidelity_box': to_serializable(Final_box_fidelity),
        'Fidelity_ion': to_serializable(Final_ion_fidelity),
        'Efficiency': to_serializable(Final_efficiency),
        'amps_current_std': to_serializable(amps_current_std),
        'Method': to_serializable(method),
        'Curvature': to_serializable(curvature),
        'start_phase_curve': to_serializable(start_phase_curve),
        'Learning_rate': to_serializable(learning_rate),
        'Exp_amp_wu': to_serializable(exp_amp_global_wu),
        'Exp_phase_wu': to_serializable(exp_phase_global_wu),
        'fb_global': to_serializable(fb_global),
        'fbmix_global': to_serializable(fbmix_global),
        'wu_gamma_out': to_serializable(wu_gamma_out),
        'wu_beta_in': to_serializable(wu_beta_in),
        'loss_beta': to_serializable(loss_beta),
        'loss_gamma': to_serializable(loss_gamma),
        'exp_amp_diff_wu': to_serializable(exp_amp_global_diff_wu),
        'comment': "",
        'waist_scale': to_serializable(waist_scale),
        'Maximum_possible_efficiency_complex': to_serializable(Maximum_possible_efficiency_complex),
        'loss_spots': to_serializable(loss_spots),
        'loss_local_phase_std': to_serializable(loss_local_phase_std),
        'loss_spots_phase': to_serializable(loss_spots_phase),
        'x_pitch0': to_serializable(x_pitch0),
        'amps': to_serializable(amps),
        'phases': to_serializable(phases),
        'xarblist0': to_serializable(xarblist0),
        'yarblist0': to_serializable(yarblist0),
        'fidelity_ion_list': to_serializable(fidelity_ion_list),
        'efficiency_ion_list': to_serializable(efficiency_ion_list)
    }

    # 2. Append the row safely on the NOTS cluster VAST file system
    with open(log_file, 'a') as f:
        f.write(json.dumps(run_metrics) + '\n')

