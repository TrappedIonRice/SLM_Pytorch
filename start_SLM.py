import runsettings
import ArrayModulator_v1
import numpy as np

# import os
# import torch
#temnm_Offcenter2Darbython
# # Explicitly add the library paths for the environment
# env_base = r"C:\Users\TrappedIonRiceDell1\anaconda3\envs\SLMPytorchenvironment"
# os.add_dll_directory(os.path.join(env_base, "Lib", "site-packages", "torch", "lib"))
# os.add_dll_directory(os.path.join(env_base, "Library", "bin"))
#
# # Force initialize CUDA to verify it's working
# try:
#     print(f"CUDA Available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"Using Device: {torch.cuda.get_device_name(0)}")
# except Exception as e:
#     print(f"Verification error: {e}")

#serial counters can be used in runsettings to change values parametrically

for serialcounter1 in np.arange(0.1,6.6,6.9):
    for serialcounter2 in np.arange(0,0.20,1):
        for serialcounter3 in np.arange(2000, 3000, 2000): #Used for loss_beta in runsettings
            for serialcounter4 in [0.001]:#np.arange(0.00, 0.1, 0.01): #start_phase curve
                for serialcounter5 in [0*0.575]:#np.arange(0.6, 1.0, 0.05):#np.append(np.arange(0.0, 0.006, 0.05),0.575): #For curvature
                    for serialcounter_not_used in np.arange(0, 1, 1):
                        runsettings.init(serialcounter1,serialcounter2,serialcounter3,serialcounter4,serialcounter5)
                        ArrayModulator_v1.create_phase()
                        runsettings.storedata()