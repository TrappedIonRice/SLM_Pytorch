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
    for serialcounter2 in np.arange(0.1,0.82,0.8):
        for serialcounter3 in np.arange(1, 1.2, 2):
            for serialcounter4 in np.arange(0.7, 1.6, 11.2):
                for serialcounter5 in np.arange(0.8, 2.7, 10):
                    for serialcounter_not_used in np.arange(0, 20, 1):
                        runsettings.init(serialcounter1,serialcounter2,serialcounter3,serialcounter4,serialcounter5)
                        ArrayModulator_v1.create_phase()
                        runsettings.storedata()