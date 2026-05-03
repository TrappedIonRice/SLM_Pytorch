import sys
import sys

# 1. Define the Tee logic
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

# 2. Redirect BEFORE importing other files
sys.stdout = Tee("all_output.txt")

import runsettings
import ArrayModulator_v1
import numpy as np






for serialcounter1 in np.arange(0.4,0.6,100 ):
    for serialcounter2 in np.arange(45,50,5):
        for serialcounter3 in np.arange(9.5, 10.1, 10):
            for serialcounter4 in np.arange(10.5, 19.9, 10):
                for serialcounter5 in np.linspace(0.62, 0.72, 1):
                    runsettings.init(serialcounter1,serialcounter2,serialcounter3,serialcounter4,serialcounter5)
                    ArrayModulator_v1.create_phase()
                    runsettings.storedata()