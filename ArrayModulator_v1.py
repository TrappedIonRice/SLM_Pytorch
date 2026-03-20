# import slmsuite.holography.algorithms as algorithms
# import slmsuite.holography.toolbox.phase
import sys
# import struct
# import platform
# import subprocess
# import global_variables
#
# def run_cmd(cmd):
#     try:
#         return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
#     except:
#         return "Not Found"
#
# print("--- SYSTEM DIAGNOSTIC ---")
# # 1. Check Python Architecture (The #1 reason for 'No matching distribution')
# bits = struct.calcsize("P") * 8
# print(f"Python Version: {sys.version.split()[0]}")
# print(f"Python Architecture: {bits}-bit {'(CORRECT)' if bits == 64 else '(ERROR: MUST BE 64-BIT)'}")
#
# # 2. Check OS
# print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
#
# # 3. Check NVIDIA Driver & CUDA Support
# gpu_info = run_cmd("nvidia-smi")
# if "nvidia-smi" in gpu_info.lower():
#     driver = gpu_info.split("Driver Version: ")[1].split()[0]
#     cuda_v = gpu_info.split("CUDA Version: ")[1].split()[0]
#     print(f"NVIDIA Driver: {driver}")
#     print(f"Max CUDA Supported: {cuda_v}")
# else:
#     print("NVIDIA Driver: NOT FOUND (Install from nvidia.com/drivers)")
#
# # 4. Check for Visual C++ Runtime (Required for CuPy/Torch)
# import os
# vcredist = os.path.exists("C:\\Windows\\System32\\vcruntime140.dll")
# print(f"Visual C++ Redist: {'Installed' if vcredist else 'NOT FOUND (Install VC++ 2015-2022)'}")
#
# # 5. Check Pip Version
# pip_v = run_cmd("pip --version")
# print(f"Pip Version: {pip_v}")
#
# print("-------------------------")
#

import math
import random
import time
import cupy as cp
#print(cp.show_config())

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import slmsuite.holography.toolbox as toolbox
from networkx.algorithms.swap import double_edge_swap

#from slmsuite.holography.toolbox.phase import hermite_gaussian
import global_variables

import cv2
import os
# import WGS
import slm_v1 as slm
import profile_v1 as profile
import PhaseExtraction_v1 as PhaseExtraction
from PIL import Image
from PIL import Image as im
from slm_v1 import SLM,pad_border
from skimage.restoration import unwrap_phase
from sklearn.linear_model import  LinearRegression
from profile_v1 import Profile, temnm, laserbeamsizefromimage, propagate

# from InversePhase import inverse_phase
from IFTA_v1 import wu, plot_gradient, Wu, OuterLoop,WGS, IFTA, OuterLoop_MDS
import tkinter.filedialog
from skimage.transform import resize
from slm_v1 import pad_border
from scipy.ndimage import  rotate

try:
    import cupy as cp
except ImportError:
    cp = np
    print("cupy not installed. Using numpy.")

# matplotlib.use('QtAgg')


# Use elementary functions of the SLM to for instance scatter/deflect incident beams and arrays of beams
class ArrayModulator(slm.SLM):

    def __init__(self, size=np.array( (1024,1272)), correction_path="images/413corrwithLUT.bmp", beams=1, beam_positions=None, beam_sizes=None):
        super(ArrayModulator, self).__init__(size, correction_path)
        """
            size: SLM size in pixels
            correction_path: file location of the correction pattern
            beams: number of beams
            beam_positions: position of each beam
            beam_sizes: size of each beam
        """
        # Note: all coordinates are normalized to [-1, 1], and phases are on a 2pi scale

        # By default, space beams equally along the x axis
        if beam_positions is None:
            beam_positions = [[-1 + (i + 0.5) * 2 / beams, 0] for i in range(beams)]

        # By default, each beam gets an equal portion of the SLM in the x direction, and the entire SLM in y
        if beam_sizes is None:
            beam_sizes = [[2 / beams, 2] for _ in range(beams)]

        # if len(beams.shape) == 1:
        # Array of beams, where each element is [[xpos, ypos], [xsize, ysize]]
        self.beams = np.array([[beam_positions[i], beam_sizes[i]] for i in range(beams)])

        # Initial SLM Phase map
        self.phase = np.zeros(shape=size)

    # Define a region of size shape around pos, returning the normalized coordinates of [xmin, ymin] and [xmax, ymax]
    def region(self, pos, shape):
        corners = [pos - shape / 2, pos + shape / 2]
        corners = [np.clip(corner, -1, 1) for corner in corners]
        # print(pos)
        # print(shape)
        # print(corners)
        return corners

    # Add a phase pattern to a region of size shape centered around pos
    def region_add(self, pos, shape, phase):
        self_corners = [self.coord(corner) for corner in self.region(pos, shape)]
        # addition_corners = [self.coord(corner) for corner in self.region(pos, shape)]
        # print(corners)
        self.phase[self_corners[0][1]:self_corners[1][1],
         self_corners[0][0]:self_corners[1][0]] = self.add(self.phase[self_corners[0][1]:
            self_corners[1][1], self_corners[0][0]:self_corners[1][0]], phase)

    # Convert pixel size region to normalized coordinates
    def region_px_to_coords(self, size, r_px=np.array((0, 0))):
        shape = np.array([size[1], size[0]])
        r = np.array(2 * (r_px / shape - 0.5))
        return r

    # Create a meshgrid from the normalized shape for use in methods that use functions from slmsuite
    def create_meshgrid(self, shape):
        phase = np.transpose(np.zeros(shape=self.shape_px(shape)))

        x_list = np.arange(-phase.shape[1] // 2, phase.shape[1] // 2)
        y_list = np.arange(-phase.shape[0] // 2, phase.shape[0] // 2)
        x_grid, y_grid = np.meshgrid(x_list, y_list)
        grid = (x_grid, y_grid)

        return grid

    # Apply the correction pattern to the entire SLM
    def apply_correction(self, active_beams=None):
        if active_beams is None:
            active_beams = [i for i in range(len(self.beams))]

        self.phase = self.add(self.phase, self.correction)

    # Generate a TEM01 phase pattern of specified shape along specified axis
    def tem01(self, shape, axis=0):
        phase = np.transpose(np.zeros(shape=self.shape_px(shape)))
        if axis <= 0:
            phase[:, phase.shape[1] // 2:] = np.pi
        else:
            phase[phase.shape[0] // 2:, :] = np.pi
        return phase

    # Generate a flat phase shift
    def flat(self, shape, shift):
        phase = np.transpose(np.ones(shape=self.shape_px(shape))) * shift

        return phase

    # Generate a phase gradient to deflect incident light at angle (in radians)
    def gradient(self, shape, angle, axis=0, wavelength=413):

        phase = np.transpose(np.zeros(shape=self.shape_px(shape)))

        delta = 2 * np.pi * self.pitch / wavelength * np.tan(angle)
        if axis <= 0:
            for i in range(len(phase)):
                phase[i, :] += i * delta
        else:
            for i in range(len(phase[0])):
                phase[:, i] += i * delta

        phase=np.mod(phase + np.pi * ((np.indices(phase.shape).sum(axis=0)) % 2),2*np.pi)  #MDS added 03/03/2026 to try to remove zeroth order

        return phase 

    def array2D(self, n=4, m=4, x_pitch=0.02, y_pitch=0.02, size=(0.05, 0.05)):
        wgs = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)
                                               ),
                  target=Profile.spot_array(m, n, x_pitch=y_pitch, y_pitch=x_pitch))
        wgs.iterate(20)
        #wgs.save_pattern('%dx%d' % (n, m), self, target=False)

        return wgs

    # Make a simple thin lens (normalized units)
    def lens(self, shape, f=(np.inf, np.inf)):
        phase = np.transpose(np.zeros(shape=self.shape_px(shape)))
        for i in range(len(phase)):
            for j in range(len(phase[i])):
                phase[i, j] = np.pi * np.sum(self.region_px_to_coords([len(phase[i]), len(phase)], np.array([i, j]))**2 / np.array(f)**2)
        return phase

    # Make a simple thin lens, where f is in meters
    def lens_realunits(self, shape, f=(np.inf, np.inf), wavelength=413e-9, eff_area=(12.8e-3, 15.9e-3)):
        print(shape)
        phase = np.transpose(np.zeros(shape=self.shape_px(shape)))
        for i in range(len(phase)):
            for j in range(len(phase[i])):
                phase[i, j] = np.pi * np.sum((self.region_px_to_coords([len(phase[i]), len(phase)], np.array([i, j])) * np.array(eff_area) / 2)**2 / np.array(f)**2) / wavelength
        # print(np.max(phase))
        return phase

    def lens_correction(self, shape, f=200):
        x=np.linspace(-1,1,shape[1])
        y = np.linspace(-1, 1, shape[0])
        xx,yy=np.meshgrid(x,y)
        print(shape)
        phase=-2*np.pi*(f/200.0)*( xx**2+yy**2)

        plt.imshow(phase)
        plt.show()
        return phase

    # Wrapper for zernike_sum_phase in slmsuite
    def zernike_sum(self, shape, weights=(((2, 0), 1),((2, 1), -1),((3, 1), 1)), aperture="circular", waist=(1, 1)):
        grid = self.create_meshgrid(shape * np.array(waist))
        phase = toolbox.phase.zernike_sum(grid=grid, weights=weights, aperture=aperture)
        return phase

    # Wrapper for the hermite_gaussian function in slmsuite
    def temnm(self, shape, n=3, m=1):
        grid = self.create_meshgrid(shape)
        phase = toolbox.phase.hermite_gaussian(grid=grid, n=n, m=m)
        return phase

    # Wrapper for the laguerre_gaussian function in slmsuite
    def laguerre_gaussian(self, shape, l=9, p=3):
        grid = self.create_meshgrid(shape)
        phase = toolbox.phase.laguerre_gaussian(grid=grid, l=l, p=p)
        return phase

    # Wrapper to add a beam array phase mask to the SLM, using OuterLoop from IFTA.py
    def wu_algorithm(self, N=40, M=20, n=5, wavelength=411, name='wu_1x4', amps=(1., 1., 1., 1.), amps_guess=(1., 1., 1., 1.), phases=(0, 0, 0, 0), x_pitch=0.004, input_profile=None, plots=True, res_factor=1, phase_memory=False, figs=(None, None, None), tem01=False):
        size = np.array(np.array( (1024,1272)) * res_factor, dtype=np.uint)
        outer = OuterLoop(slm=self, input_profile=input_profile, n=n, wavelength=wavelength, name=name, amps=amps, amps_guess=amps_guess, phases=phases, x_pitch=x_pitch, phase_memory=phase_memory, size=size, tem01=tem01)
        phase = outer.iterate(N=N, M=M)
        if plots:
            # print(figs[2])
            outer.plot(show=False, figs=figs)
        # phase = wu(self, N=N, M=M, n=n, size=size, wavelength=wavelength, name=name, show=False, amps=amps, amps_guess=amps_guess, phases=phases, x_pitch=x_pitch, input_profile=input_profile, plots=plots, phase_memory=phase_memory).get()
        return np.angle(phase.get())

    def wu_algorithm2D_MDS(self,  amps_tem_compensation=None,phase_tem_compensation=None ,N=40, M=20, m=4,n=5, wavelength=411, name='wu_1x4', amps=(1., 1., 1., 1.), amps_guess=(1., 1., 1., 1.), phases=(0, 0, 0, 0), x_pitch=0.004,y_pitch=0.004, input_profile=None, plots=True, res_factor=1, phase_memory=False, figs=(None, None, None), tem01=False):
        res_factor=1
        size = np.array(np.array( (1024,1272)) * res_factor, dtype=np.uint)
        count = 0

        # phases = tuple(0. if i%2==0 else 0.5 for i in range(n*m))
        # print('phase',phases)
        # print('phase', phase_tem_compensation)
        outer = OuterLoop_MDS(slm=self,amps_tem_compensation=amps_tem_compensation,phase_tem_compensation=phase_tem_compensation,
                          input_profile=input_profile,
                          m=m, n=n, wavelength=wavelength,
                          name=name, amps=amps, amps_guess=amps_guess, phases=phases,
                          x_pitch=x_pitch,y_pitch=y_pitch,
                          phase_memory=phase_memory, size=size, tem01=tem01)
        phase = outer.iterate_MDS(N=4, M=M)
        if plots:
            # print(figs[2])
            outer.plot(show=False, figs=figs)
        # phase = wu(self, N=N, M=M, n=n, size=size, wavelength=wavelength, name=name, show=False, amps=amps, amps_guess=amps_guess, phases=phases, x_pitch=x_pitch, input_profile=input_profile, plots=plots, phase_memory=phase_memory).get()
        return np.angle(phase.get())


    def wu_algorithm2D(self,  amps_tem_compensation=None,phase_tem_compensation=None ,N=40, M=20, m=4,n=5, wavelength=411, name='wu_1x4', amps=(1., 1., 1., 1.), amps_guess=(1., 1., 1., 1.), phases=(0, 0, 0, 0), x_pitch=0.004, input_profile=None, plots=True, res_factor=1, phase_memory=False, figs=(None, None, None), tem01=False, uni_spacing=True ,xarblist0=None, yarblist0=None, anglearblist0=None,double_amps_in=None):
        res_factor=res_factor
        size = np.array(np.array((1024,1272)) * res_factor, dtype=np.uint)
        count = 0

        # phases = tuple(0. if i%2==0 else 0.5 for i in range(n*m))
        # print('phase',phases)
        # print('phase', phase_tem_compensation)
        outer = OuterLoop(slm=self,amps_tem_compensation=amps_tem_compensation,phase_tem_compensation=phase_tem_compensation,
                          input_profile=input_profile,
                          m=m, n=n, wavelength=wavelength,
                          name=name, amps=amps, amps_guess=amps_guess, phases=phases,
                          x_pitch=x_pitch,
                          phase_memory=phase_memory, size=size, tem01=tem01, uni_spacing=uni_spacing, xarblist0=xarblist0, yarblist0=yarblist0, anglearblist0=anglearblist0,double_amps_in=double_amps_in)
        phase = outer.iterate_updatedloop_outer(N=20, M=M) #n=30
        #phase = outer.iterate_Gradient(N=5, M=M) #n=30
        if plots:
            # print(figs[2])
            outer.plot(show=True, figs=figs)
        # phase = wu(self, N=N, M=M, n=n, size=size, wavelength=wavelength, name=name, show=False, amps=amps, amps_guess=amps_guess, phases=phases, x_pitch=x_pitch, input_profile=input_profile, plots=plots, phase_memory=phase_memory).get()
        return np.angle(phase.get())


    # Generate a phase pattern to create an arbitrary 2d light field in the image plane using the Wu algorithm only
    def wu_image(self, N=40, wavelength=411, input_profile=None, target=None, size= (1024,1272)):
        if target is None:
            target = self.BMPToAmp(path=tkinter.filedialog.askopenfilename(title='Select Image'), norm=True)
            # target = cp.array(resize(target, size))
            target = cp.array(pad_border(target, size))
        if input_profile is None:
            input_profile = cp.array(profile.Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
        wu = Wu(size=size, input=input_profile, target=[target, [(0, 0)]], wavelength=wavelength, array=False)
        wu.iterate(N)
        return cp.angle(wu.slm_field).get()

    # Generate a random phase pattern to scatter all the incident light
    def rand(self, shape):
        phase = np.transpose(np.random.rand(self.shape_px(shape)[0], self.shape_px(shape)[1])) * 2 * np.pi
        return phase

    # Apply a phase pattern to each beam
    def add_phase(self, phase, active_beams=None):
        if active_beams is None:
            active_beams = [i for i in range(len(self.beams))]

        for beam in active_beams:
            self.region_add(self.beams[beam][0], self.beams[beam][1], phase)

    def backpropagate(self, phase):
        return cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(phase), norm="ortho"))

    def propagate(self, phase):
        return cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(phase), norm="ortho"))

    def set_phase(self, phase):
        self.phase = phase


# Create a phase pattern
def create_phase(wavelength=411, phases = (0.,0.,0.,0.),amps=(1.,1.,1.,1.)):

    # mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
    # target = mod.BMPToAmp(path=r'C:/Users/RiceT/Documents/FlirCamera/pictures/images.bmp', norm=True)
    # #target = cp.array(resize(target, (40,20)))
    #
    # target =cp.array(slm.pad_border(target, (1024,1272)))
    # arb2d = mod.wu_image(N=100, wavelength=411, input_profile=cp.array(profile.Profile.input_gaussian(beam_size=(0.4, 0.4), size=np.array((1024,1272)))), target=target)
    # grady = mod.gradient(mod.beams[0][1], angle=0.12 * np.pi / 180, axis=0, wavelength=wavelength)
    #
    # # temnm = mod.temnm(mod.beams[0][1], n=1, m=0)
    # mod.add_phase(grady)
    # mod.add_phase(arb2d)

    # matrix_correction = np.zeros((400, 400))
    # for i in range(16):
    #     for j in range(16):
    #         if 1:
    #             for m in range(25):
    #                 for n in range(25):
    #                     matrix_correction[25 * i + m][25 * j + n] = correction[i][j]
    #
    # mod.phase = mod.phase + slm.pad_border(matrix_correction, (1024,1272))
    #
    # mod.phaseToBMP(mod.phase, name='test_halloween', color=False, correction=False, wavelength=411)
    # mod.add_phase(-mod.phase)
    mod = ArrayModulator(beams=1 ,correction_path='images/%dcorrwithLUT.bmp' % wavelength)



    phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase.txt'
    correction = np.loadtxt(phase_path)
    grady = mod.gradient(mod.beams[0][1], angle=0.12* np.pi / 180, axis=0, wavelength=wavelength) #1315#yangle=0.12 #yangle=0.1310 for expt
    gradx = mod.gradient(mod.beams[0][1], angle=-0.12* np.pi / 180, axis=1, wavelength=wavelength)
    #temnm = mod.temnm(mod.beams[0][1], n=1, m=1)
    #temnm[25:,25:]=temnm[:-25,:-25]
    # gradx=rotate(g)
    mod.add_phase(grady)
    mod.add_phase(gradx)
    #mod.add_phase(temnm)
    #lens_phase=mod.lens_correction((1024,1272),f=200)
    #mod.add_phase(lens_phase)
    randmodphase=np.zeros(grady.shape)
    for ii in range(len(mod.phase)):
        for jj in range(len(mod.phase[0])):
            if (ii-len(mod.phase)*0.5)**2+(jj-len(mod.phase[0])*0.5)**2>(len(mod.phase)*0.25)**2:
                randmodphase[ii,jj]=(np.random.rand(1)*2*np.pi)[0]

    randmodphase[:,int(len(mod.phase[0])*0.5)-2:int(len(mod.phase[0])*0.5)+2]=np.random.rand(len(grady),4)*2*np.pi
    #mod.add_phase(randmodphase)
    #tem01phase=mod.phase
    # mod.add_phase(temnm)#/(0.7624614246165774+2.9579199832069927)*np.pi)

    #UNCOMMENT THIS FOR TEM01...?
    #phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01.txt'
    phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01_nocorr.txt'
    phase_mask_tem01 = np.loadtxt(phase_path)
    phase_mask = np.loadtxt(phase_path)
    phase_mask_flip=np.flip(phase_mask, axis=1)
    #mod.add_phase(np.array(phase_mask_flip))
    #plt.imshow(phase_mask_tem01)
    #plt.show()
    #plt.imshow(phase_mask_flip)
    #plt.show()
    wu_1x4_intensity=(profile.Profile.input_gaussian(beam_size=(0.3, 0.3),size=(1024, 1272)))*255
    #fig, axs = plt.subplots(2, 1)
    #axs[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wu_1x4_intensity*np.exp(np.multiply(1j,phase_mask_tem01))), norm="ortho"))))
    #axs[0].set_title("mod fft abs")
    #plt.show()
    #axs[1].imshow((1/(2*np.pi))*np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wu_1x4_intensity*np.exp(np.multiply(1j,phase_mask_tem01))), norm="ortho"))))
    #axs[1].set_title("mod fft phase")
    #plt.show()



    # plt.figure(figsize=(8, 6))
    # plt.imshow(np.array(phase_mask_tem01), cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title('phase of slm')
    # plt.show()

    scalepad=2
    n =11
    m =1
    amps = tuple(1. for i in range(n * m))#(1.0,1.0,1.0,1.0)#
    double_amps_in = tuple(1. for i in range(2*n * m))
    # measuredlist=(59,54.5,58.89,58.46,59.52,56.71,61.26,61.26,62.88,60.89,62.51,69.24,63.2,63.32,68.56,56.90,63.26,60.64,64.75,53.10,57.77,54.91)
    # measuredlist=[0.05398330942652553, 0.0396954990228096, 0.05511213627639267, 0.049480514926320114, 0.04781828321827654, 0.04997678537957286, 0.0612697978879915, 0.05174388585520492, 0.06673473379117498, 0.062241233348226954, 0.05541752957793415, 0.07505884830333326, 0.0588718327578685, 0.05117197394185436, 0.07309332464206454, 0.05205140472528165, 0.060815377194856725, 0.04537543649955126, 0.06630963039629556, 0.044450518314419506, 0.05218498173991211, 0.04650294510858862, 0.05795210346086446, 0.0314641670181334, 0.06449610848446843, 0.05159982989961452, 0.04841624584552598, 0.05376089416433478, 0.06275862009707912, 0.06836873553098222, 0.07181910639085509, 0.05865442689495316, 0.056734318281365194, 0.07343465256875494, 0.06846976412353963, 0.06335345171007599, 0.0602284913009532, 0.06475767331328726, 0.06444151788100558, 0.04692588847384371, 0.06604675388218495, 0.04277473289333789, 0.05726921237685519, 0.05267169267307754, 0.0607293001672049, 0.05265528440707133, 0.05610585949854914, 0.0586101300044448, 0.05550631949701244, 0.04869635457382515, 0.059109109413590204, 0.04761001509456172, 0.0679575214397785, 0.06467291709980849, 0.05645138456035185, 0.08634725957108809, 0.07601051524432097, 0.07053892983339297, 0.07713666201464427, 0.04849210560310542, 0.0825997827549384, 0.05841759868190757, 0.072568958291567, 0.054283435546649585, 0.06133546415285358, 0.055244012094115176, 0.05321071063749584, 0.04541746597325644, 0.06180993681143275, 0.05118743412740429, 0.06348429177348087, 0.06363279490665757, 0.060783266457214866, 0.0608223116340539, 0.0826774929423158, 0.07365090773080878, 0.06024646844717145, 0.07702444929789237, 0.07826201602831602, 0.05964778816760386, 0.08496223973832918, 0.054408302994304575, 0.07169999369618534, 0.05361495315438125, 0.07209865388687048, 0.042037274112969446, 0.06604211635804066, 0.049593184622848105, 0.061847192095190455, 0.042481058664529225, 0.06898547297731107, 0.0546570705739509, 0.06688819493063639, 0.054205459168369316, 0.07932002013047466, 0.07224186024096377, 0.08031333529618187, 0.08211827146871324, 0.06874241099736447, 0.07589498178167296, 0.07283130715455477, 0.07291797522912605, 0.07829922990436929, 0.053264838647478864, 0.08031819607730996, 0.04828847083109554, 0.06430822041291089, 0.043376857686021, 0.06947904250639972, 0.05005408308264247, 0.054312127025543576, 0.04442216303888325, 0.05901534522542543, 0.054778614060452153, 0.07074373923432696, 0.046797752276482786, 0.058990438304617754, 0.06096388070642089, 0.06736948583691804, 0.061287794390651865, 0.07221279890401522, 0.07754853891565751, 0.07214854806574776, 0.07111320859963481, 0.07736766888614752, 0.04965545910706833, 0.07158766213239875, 0.04032623952021174, 0.05845348517172539, 0.04081896861201355, 0.06478296653372612, 0.04957393266123324, 0.05706159999388386, 0.04865393887357903, 0.0669794093721949, 0.05168571751249264, 0.05761388787994593, 0.059863763768533966, 0.058504797449220436, 0.06214841632623152, 0.07140413994872828, 0.06796097365263726, 0.07399288261461784, 0.0825165692004787, 0.07524153346961346, 0.06350846374473566, 0.07691352165078282, 0.056890803366279975, 0.07680771543805164, 0.06196294629092937, 0.06289210104371752, 0.032451545347999704, 0.055721157387466164, 0.05475638047149201, 0.054153371081897224, 0.057366816587295234, 0.053480507837493065, 0.053778475896162026, 0.06065409663303889, 0.06163065594883199, 0.07668410431512668, 0.07972176174203285, 0.06657074971066403, 0.05661696507316772, 0.0677600346962255, 0.07348276073843513, 0.07301286027759733, 0.07530041753630543, 0.07646179947668083, 0.06177876125668194, 0.08361482331437975, 0.0581786551011664, 0.07122457244724156, 0.034739789794463585, 0.058790509543023145, 0.041734142308090166, 0.05592970205885973, 0.05343137123408515, 0.05042014766905258, 0.06256422535834842, 0.06150076685440195, 0.05749650438398673, 0.06277120051918855, 0.07372485661907778, 0.0686314022991628, 0.06727209891463064, 0.06334090051066098, 0.06840570468059404, 0.06584300129880055, 0.06276579034933626, 0.07066713565981651, 0.06092761183824183, 0.0807592588908831, 0.050337345363089626, 0.07263406641791786, 0.045692760924050696, 0.06346297936548476, 0.04814571216437536, 0.0686909014352028, 0.04331225005200005, 0.06664704767352236, 0.06113292015041113, 0.051292783142833094, 0.05649751661798163, 0.059605873564188526, 0.07524774113521343, 0.06575615600742443, 0.058704017453668295, 0.04343173143582351, 0.0658357623190167, 0.06893673139231636, 0.06708891151085274, 0.06901148235854836, 0.06750661021477157, 0.07074664519320491, 0.04789925242364705, 0.06849190008662112, 0.044043528568420434, 0.05191626115167067, 0.05033614221253326, 0.05652139114772611, 0.04882896370713447, 0.0749634137074898, 0.06541253533871207, 0.055955301700749194, 0.055168526813492845, 0.05, 0.06199373216382463, 0.055182246163603184, 0.06572992254667202, 0.055555932494147704, 0.06847705247326383, 0.07262537035381258, 0.06864599074960961, 0.05865342052046889, 0.059477677887396344, 0.07832838061394405, 0.05634901958666412, 0.07321041982690099, 0.048269078029428626, 0.05957633922343583, 0.05245315511582282]
    # Use this below: for tem11
    #     measuredlist=[0.7112479643062029, 0.5584936174243781, 0.6495180909966237, 0.6840123791570076, 0.6712518777573904, 0.584581135920065, 0.6663965183250093, 0.6938436530432069, 0.810735742503992, 0.7831742091794136, 0.7316087484091442, 0.7134404831838036, 0.7299404806114957, 0.6238276280861504, 0.8096249674148528, 0.6386940284740465, 0.8248483830550905, 0.6948712701493621, 0.724915051629277, 0.6521868921449657, 0.7107393835911553, 0.7007812949737895, 0.6945826604533404, 0.5657692497517232, 0.7386871782075585, 0.6992546390196998, 0.7046293916561486, 0.6360133118501606, 0.7314624539943868, 0.7914748308785805, 0.7680935067760672, 0.7596860981880713, 0.6268167470428307, 0.7988587624619707, 0.7393042033987668, 0.7667177456773618, 0.7234012896937636, 0.6621262384307466, 0.7615095220540894, 0.5962872590420263, 0.8182774930625861, 0.6161886240573379, 0.7137288611448153, 0.6673854327929777, 0.819487362958361, 0.7488685226427622, 0.7258319293459977, 0.729967796276516, 0.6081888304545295, 0.6858379603241613, 0.7015072249471787, 0.7083778310204113, 0.8702033505080512, 0.830950607349572, 0.761355614101251, 0.7676425204918258, 0.8741397713240081, 0.7300429839660658, 0.9132625953165248, 0.716874179657513, 0.8799691993004711, 0.7036933371141706, 0.8485731343295803, 0.7088455114782365, 0.7748829450057619, 0.6515841472497762, 0.7707712857849599, 0.6012637123158094, 0.7176436088143436, 0.7201972984321342, 0.7531713741794538, 0.8105528425845073, 0.7878068991501518, 0.8420876603050244, 0.8977239060699498, 0.8159912397702412, 0.7915727679136776, 0.874982173709869, 0.8459250549816334, 0.8063378548469098, 1.0, 0.7200401541375444, 0.7267437480445631, 0.6750102168773758, 0.8067357397806401, 0.6844938264250509, 0.8408010246609748, 0.6280888830836395, 0.7145087527514552, 0.6625000921122356, 0.8102095030636917, 0.7785005090261261, 0.6473608449190588, 0.7252172555176721, 0.8388117762462214, 0.8750604818285368, 0.9105435521926907, 0.8853081904535244, 0.8690500285143793, 0.9271812081238533, 0.8658816526793612, 0.8458922805671786, 0.9124378708196925, 0.731167516891093, 0.9163586502124903, 0.7284783790780868, 0.8982335852345164, 0.6735504583859182, 0.8746875894817223, 0.6698203081495493, 0.6887755468656398, 0.6467893592968782, 0.7782179923454222, 0.755562569216539, 0.8317794950556483, 0.6908410204156885, 0.7311497551571507, 0.758525585729411, 0.8175102775785585, 0.8640380469705442, 0.905838137646754, 0.8336492555659517, 0.851356967260156, 0.8550680695653586, 0.9476340348895617, 0.7612898403063775, 0.8990203383633655, 0.7589377875516026, 0.8831038109002506, 0.6993645885466718, 0.8167996652346954, 0.674700115485295, 0.724416435301632, 0.7076556125917951, 0.7209153000828268, 0.6770610435194618, 0.7131844795109141, 0.7656322346146562, 0.7205449335464743, 0.777849727801034, 0.8312346892838699, 0.7432962757676371, 0.7684344453999683, 0.8692171581673331, 0.8922985804269836, 0.7562183407134101, 0.9119061068961078, 0.7897942900302212, 0.848447782630904, 0.7599602345042379, 0.774490511662453, 0.6248884419698736, 0.8129345670268733, 0.6112422600482804, 0.7460809406820424, 0.7979001493218687, 0.7393922809746949, 0.7493509023981266, 0.8115803576873729, 0.7402630156185348, 0.7167811770165939, 0.8403258034852084, 0.6765947323673915, 0.8200965015765893, 0.7973048100237256, 0.8467643341680633, 0.9052180170901861, 0.9140565407974728, 0.9040110939960901, 0.7840844985023289, 0.9619323142537453, 0.8098574864479762, 0.8482720690809343, 0.6616527818817338, 0.7655972125288399, 0.634706915886047, 0.7714863713226758, 0.6842993407611865, 0.6992984730856332, 0.8529734667915061, 0.726408711429442, 0.7488184490281723, 0.8038800849322717, 0.8622282851739435, 0.7695266568264292, 0.7661768096834017, 0.8073787058269519, 0.8231095570805778, 0.6067042481214593, 0.7862998409446579, 0.841045470784491, 0.7855539682661243, 0.9863304284816253, 0.7371604387649875, 0.8190424473713117, 0.7385228986491956, 0.8197860761456941, 0.6353159954178443, 0.7969856079277313, 0.6932767467471863, 0.7558309484335483, 0.8040481561002084, 0.6973496002222023, 0.7065507503712573, 0.7634746738391145, 0.8115022792079015, 0.8907961527697462, 0.6815909317746891, 0.7132398444513899, 0.9015074620247152, 0.7212765233483774, 0.8340075339909012, 0.7893914670257041, 0.8124851985970053, 0.8799615829211291, 0.7577145897077383, 0.819675302647894, 0.7238771009757352, 0.7153406742349846, 0.6961854040094232, 0.8224868089410345, 0.7180726124779432, 0.7946150677311107, 0.7681170885739412, 0.7304267339678158, 0.7321246968954263, 0.7074373376123471, 0.7846742487074879, 0.8812952090182388, 0.8237586330463611, 0.7456732730302857, 0.9444543539366507, 0.8620130752220921, 0.9134275084930806, 0.8050605561336741, 0.6818523698317311, 0.8500600967121628, 0.7723221521355297, 0.8600924670348685, 0.7779016754685789, 0.8032763250807106, 0.7612154695428989]
    #     measuredlist2=[0.8138027251621458, 0.8219185721355745, 0.8527536609024321, 0.774077321923712, 0.78234284163144, 0.8944214423406537, 0.8577692691643156, 0.7429652914903893, 0.7163242807597678, 0.8691395656485672, 0.8199058845632168, 0.873187239292632, 0.848729546021269, 0.9363048409258106, 0.7968909854927619, 0.8122390318185535, 0.8392879315519899, 0.7422227460416012, 0.9531700139628755, 0.7422597411540496, 0.8646564488888379, 0.7685790571241246, 0.8442854759576688, 0.7447858490812616, 0.8095430297160877, 0.7969976586265574, 0.7880997936835069, 0.8468387957558821, 0.7897613724799962, 0.8011273345007047, 0.8594993218999761, 0.8785996555074198, 0.8542195166438504, 0.8242830767238006, 0.7991251561240095, 0.7912137864737343, 0.8820137946496952, 0.785311517571643, 0.8783863507630428, 0.8407214009387228, 0.8127216122898363, 0.8259787598496364, 0.8454404878422423, 0.747700307569167, 0.7165900225634917, 0.8065977225986501, 0.7976885758794211, 0.792446072820569, 0.7955388987139074, 0.7974723648124052, 0.7468010666801086, 0.896862076258894, 0.7224062639138863, 0.8405139866321081, 0.834426867096211, 0.9585586763425892, 0.8370775808709205, 0.9321774348919492, 0.8156162438353619, 0.7625573658585062, 0.833077440718585, 0.7983756568859723, 0.8876556805098494, 0.7831608935120976, 0.8776960376097206, 0.82106035545298, 0.7901045743815667, 0.9161879521755784, 0.820221603984125, 0.8450525993642064, 0.7515439524619824, 0.8733862335146196, 0.7626663570126367, 0.8034061997565743, 0.8092455543615532, 0.8093760242433877, 0.7645693106533873, 0.7374064086485507, 0.8473877759700141, 0.8255661995659828, 0.7958035441701282, 0.7743443307649286, 0.9587187583658483, 0.8462207920393071, 0.9246811865584421, 0.7689692422975722, 0.8489221917133725, 0.8088877370457274, 0.8228878608998211, 0.7983428455401789, 0.7702219340349195, 0.7917291987376821, 0.8922506891461347, 0.8256404074515188, 0.766214190207208, 0.8524344379980646, 0.8086042188962077, 0.8773008677159587, 0.8023580798056139, 0.774675513543589, 0.8222452022372796, 0.8740512049256003, 0.8091624956942846, 0.7947191929377065, 0.8053265797299103, 0.7018917778254008, 0.7665364151923273, 0.8028027473544257, 0.7623617708351632, 0.7334388733717043, 0.8242827514161409, 0.8120767842403935, 0.7882820137014204, 0.7503108997940834, 0.8017693709627142, 0.8752428352800943, 0.8266239232269571, 0.8338695540434031, 0.8134958810511617, 0.8533959322155839, 0.7715695411676086, 0.7698951321995561, 0.7941834015439723, 0.8842650820790918, 0.8058229586005939, 0.7537472397589203, 0.8531885341528066, 0.7900597808048399, 0.7671288696878641, 0.80147492937902, 0.7648488502898777, 0.7709033055344683, 0.7470503140125029, 0.8233642858904917, 0.957979436667884, 0.8736217689157852, 0.8323788882540126, 0.8643652893857878, 0.8465154701980325, 0.8583999381339037, 0.7940214371990441, 0.8707432440213708, 0.8165022426578288, 0.8748705116747687, 0.8525707077688911, 0.8489562325448635, 0.7993768613084501, 0.8153881415987998, 0.8495501218496453, 0.8869488830585898, 0.881632956596254, 0.8176338856707582, 0.814270217989063, 0.9216629821127328, 0.8146461632273525, 0.8092007000911847, 0.8791950935104679, 0.8904282690904163, 0.8412610581255285, 0.8877037600681663, 0.969583346635214, 0.8472998625403175, 1.0, 0.881661822027034, 0.8419996993937805, 0.934688799701966, 0.9075228915470431, 0.8008775538901872, 0.8082023133637353, 0.862324274945327, 0.9214710664126686, 0.9247350609280914, 0.9674477651429884, 0.9324553009706231, 0.9370244149079969, 0.8760150822199463, 0.8254474629141166, 0.8045144463333135, 0.8508482708834838, 0.8033300245515109, 0.887387460661289, 0.8216950377863408, 0.8963126048754249, 0.8273893017543725, 0.7787738070720364, 0.9265656793294982, 0.9420397116901741, 0.9094860612040374, 0.9861580900045861, 0.8450231408832332, 0.8477848442648541, 0.8050075231564585, 0.8552225098187725, 0.8522638198475007, 0.9489107854973634, 0.8391756059399672, 0.9123746933595768, 0.8204924792023298, 0.780883972035389, 0.891809488333904, 0.8520271243852167, 0.7884848166267474, 0.8427888204636726, 0.875865958050041, 0.8222364541931159, 0.867241503172105, 0.8540606612247016, 0.861195479938578, 0.8670954083377042, 0.9181086098947464, 0.9738303844677559, 0.7378172442913801, 0.8346294673391288, 0.834205454935495, 0.8054025290124663, 0.8069445969940513, 0.8600045871273959, 0.7699550253437563, 0.8960911760517242, 0.7755956356017044, 0.8075009561122609, 0.882647596546186, 0.8390403810818182, 0.8495272000353602, 0.7726896657760306, 0.9039864218943254, 0.8496977368728107, 0.8211839621582095, 0.841123797523864, 0.7649360342354866, 0.8003196609845222, 0.8067969832473099, 0.8553314069088017, 0.7675118224405434, 0.8614261404297469, 0.8785714111109433, 0.9104972231916104, 0.7479988458100522, 0.8434355285065425, 0.8003098424972818, 0.8177704583473794, 0.8025852121846729]

    #Alternate 11*11 tem01
    # measuredlist=[0.7266995672241572, 0.777180331370064, 0.7220068804941505, 0.8847329587613973, 0.7661439150857582, 0.7412893131149472, 0.7192611059735112, 0.7982212181007626, 0.6727067837207711, 0.7548360197135107, 0.8630253160746997, 0.7438114820249132, 0.9072493580713928, 0.7478354293553152, 0.7274563525867113, 0.8107139037542138, 0.8489564934789534, 0.7505490500222597, 0.8385833481839717, 0.8764471243644202, 0.8469607837255475, 0.6691063841311486, 0.6952959920875926, 0.8829228879895713, 0.7770361240257406, 0.8282469329493163, 0.7994991266130235, 0.672373663876673, 0.7438950267219875, 0.7706056217540594, 0.7599144700482255, 0.7270761492417834, 0.7667025328476302, 0.7222930440051416, 0.9114298252704017, 0.7608056829030392, 0.7049939172148035, 0.7119607919620566, 0.74273312019507, 0.7004285768301178, 0.7338107302828356, 0.7785264316134405, 0.7804366072098262, 0.6930142626526471, 0.6227601617003847, 0.6130656539823651, 0.711265597613907, 0.7068830720785075, 0.8216286220538395, 0.6972609380328697, 0.7285578360276409, 0.848078823980775, 0.8929465136189805, 0.7472983800702128, 0.8582831725443764, 0.7350114823446413, 0.8097976313492563, 0.7409086571313692, 0.7078657143019341, 0.7718261911030528, 0.8300610175098655, 0.7743122592004853, 0.7381232867717771, 0.7562774435110986, 0.7499602144623396, 0.6634750690419251, 0.5937735506240518, 0.6626037319082498, 0.6645909083213569, 0.8181115032953601, 0.7632207967695297, 0.621859174051138, 0.7705112503175293, 0.7339248221110153, 0.8010023549130446, 0.8823480672212818, 0.8050613291627349, 0.7530836380018358, 0.772641120377682, 0.6310674930619932, 0.838221008550263, 0.687248146073323, 0.7867096177214481, 0.7678742808607707, 0.7428162098611382, 0.7071963213882358, 0.7696262497009587, 0.6384488338092181, 0.7058919805432988, 0.6802467253741313, 0.6718565142203081, 0.7510664282504625, 0.6706719864590168, 0.7998089508982955, 0.7887871752291077, 0.8067254043181362, 0.7285778307037637, 0.7566484402605121, 0.7180571802044057, 0.7905401460730886, 0.8026271326978567, 0.7555224818977017, 0.7400804702584691, 0.7134349690369853, 0.8156991159063559, 0.74029118321347, 0.7814248546089029, 0.8376075519615661, 0.7373110666251684, 0.6823940517465089, 0.7030048921445442, 0.713786051116188, 0.7203800950879462, 0.8826186944886006, 0.6754879119601275, 0.7405787191791091, 0.7137394215751753, 0.8131105583466154, 0.8503597217922383, 0.763757815579087, 0.773992629811455, 0.6826464377367808, 0.8140332873190218, 0.750466466773144, 0.7634870848399476, 0.6883930229278267, 0.8000891056879587, 0.6782198050550389, 0.801723689245734, 0.7860922797507738, 0.7354607706931471, 0.6996055206547969, 0.7715580325766896, 0.7368031048675552, 0.8014813682865489, 0.8771067256685708, 0.836747997810879, 0.8151227252815432, 0.7975213755390976, 0.8474163141684459, 0.7991760926640473, 0.738100642939896, 0.8386226040359197, 0.7961044077288206, 0.8741892301661237, 0.6708941396516719, 0.7123943171736062, 0.821876158586234, 0.7663095511060344, 0.7487204100328843, 0.721954065362709, 0.7248133299033259, 0.7262357306326139, 0.6468666481333248, 0.7660953793101664, 0.723943388989403, 0.7664197867526572, 0.8976960546544078, 0.9055926725069926, 0.8391081763597281, 0.9136040753642471, 0.8932606108152827, 0.972124556470795, 0.9861424289142696, 0.9605646569402561, 0.9077938217242625, 0.9986499878752851, 0.8307920338538272, 0.8174945484941685, 0.9441205053332672, 0.830425307358282, 0.8074513506271458, 0.8928841001378277, 0.9114124697665827, 0.7701330077155031, 0.8646081840366702, 0.7991608256643231, 0.8377211296024697, 0.78309093131283, 0.9137869294865189, 0.9193979111093382, 0.8856895968942236, 0.8042041057097143, 0.7897215995274012, 0.8359741685036206, 0.7899147400395894, 0.8267778804947874, 0.7206121748550276, 0.8909445963614108, 0.7344007111581148, 0.7868949187429266, 0.8515318825555852, 0.9406415928493795, 0.7657895589749552, 0.9596562247035326, 0.809188169906683, 0.8823269556936938, 0.6538138631273726, 0.7925498400132883, 0.8382999452433444, 0.7883193077815132, 0.9434317445007844, 0.8137902136633851, 0.7805897333073731, 0.8956072306767228, 0.8477562543327185, 0.7884989039225683, 0.7923977621105516, 0.9376872288987306, 0.828464719708636, 0.9602953030958088, 0.7167152149518418, 0.7371390790642102, 0.8890572822993472, 0.8632256680660841, 0.8058392679660037, 0.9189593702134808, 0.7389529643673419, 0.8436284475392835, 0.7407623935998059, 0.8354549326967529, 0.84708045903068, 0.8283693715733336, 0.9145776139751093, 0.7922917047441853, 0.7120499186038334, 0.8273866931253535, 0.9594903874035272, 0.9972077510299704, 0.8597705686510262, 1.0, 0.8273866931253535, 0.9349615441660726, 0.8183487093494872, 0.807238219413181, 0.8341689685518326, 0.9249536027258811, 0.7756118680055588, 0.882684531822857, 0.7817678536862503, 0.8054506303334179, 0.8221032568286847]
    # measuredlist2=[0.7968768933487336, 0.7227505972747098, 0.8225109788847978, 0.7809304918512372, 0.7728566409441961, 0.7454730108942853, 0.7798765178017153, 0.8070432006301944, 0.8550947022325212, 0.809756174970495, 0.665186207568039, 0.8145642791469196, 0.8306319928435874, 0.7362480378623119, 0.8310602483828409, 0.8223789254313014, 0.8688140597334537, 0.7591474215463725, 0.7840856831796446, 0.755547325791152, 0.7006502273186663, 0.7878984742480238, 0.8288050257962659, 0.7840326013729556, 0.7905967778489918, 0.8162657888584369, 0.7894017865733396, 0.8522236725057969, 0.8329707891844584, 0.8081077243439133, 0.8100748659238395, 0.754813759355127, 0.7958766296175181, 0.7270807907931092, 0.7855674341578235, 0.7932561813905514, 0.8067289639312357, 0.785189088217709, 0.8003125482039808, 0.7858813925281277, 0.7880726419444494, 0.8012425863680326, 0.8429746103592419, 0.7540182345804238, 0.8931758648103915, 1.0, 0.8814452745649297, 0.8621727741516183, 0.7940283763054412, 0.881632397914897, 0.8817280910619242, 0.8190110864471564, 0.7938850345988856, 0.7815421635848198, 0.7538325619518977, 0.7907779560483509, 0.9451137075903022, 0.8385544286848037, 0.8352447584361401, 0.795400653187529, 0.7629508445573401, 0.7938756442162127, 0.9027231447581678, 0.8096891903258258, 0.8515890213091568, 0.7982986236937109, 0.8437072651760543, 0.7839743428256263, 0.8195617666687011, 0.8678972866151178, 0.8217721373345469, 0.873616182456615, 0.8354382153622928, 0.8806325453949928, 0.8632464447183611, 0.8695579027043824, 0.8690935498294579, 0.8782348919542612, 0.8894524361722689, 0.9176758177087417, 0.8823031375677952, 0.8715831729574625, 0.810276141525064, 0.873979714242001, 0.8548091517374609, 0.8999125275504709, 0.8276645241364403, 0.9068672133707472, 0.8466256368382212, 0.7738940153024862, 0.8775247696071865, 0.8086219663575108, 0.8884723627601644, 0.7820732065289564, 0.7425938447308965, 0.8701661706593964, 0.836636890872773, 0.8638174539791257, 0.8562058504352357, 0.857582176644033, 0.8655931838449359, 0.8152538254780746, 0.8201534650640018, 0.8109651809243467, 0.8887900566351447, 0.8973701900342147, 0.8570426683836028, 0.8680708255309247, 0.8505406402698983, 0.7953526878689743, 0.8545863522914233, 0.7891455192002931, 0.7707913025837868, 0.7771749075253717, 0.8524337312789834, 0.8277717287916075, 0.8245871181766556, 0.8604031328693547, 0.803126990259923, 0.8397192451387748, 0.815177676423785, 0.79477368675052, 0.7526959324008875, 0.8417035623278055, 0.7948635378569096, 0.8847577220459497, 0.808605423297624, 0.8504175725696123, 0.8363262004085642, 0.7747391295913206, 0.8452085365926537, 0.7832605806397415, 0.7487355988143163, 0.7865874852717285, 0.7583079375263929, 0.7986359735977793, 0.6785647881083338, 0.7005274611650684, 0.8163175863241663, 0.8014261095044684, 0.8256098204642534, 0.789716657553071, 0.795982279189712, 0.7673713828331127, 0.7533861483530454, 0.7604513969017402, 0.729933808444535, 0.8329206013332486, 0.8349066389544004, 0.784415912693882, 0.8322164494753896, 0.7812331450710711, 0.780894036151498, 0.8754426721803452, 0.7493346497231885, 0.667759594818286, 0.7378319079380701, 0.8203580179579796, 0.7549943265939876, 0.7239517436604546, 0.7498689912895217, 0.6816785616612978, 0.8415004340413793, 0.7650645969085361, 0.7213051370223915, 0.6731630923456184, 0.7973672872448017, 0.7663051197856564, 0.7406591475663963, 0.8166045087690345, 0.8766948078960989, 0.7785860345880751, 0.7509145974432738, 0.8072797585152608, 0.759990472494021, 0.7310883930837978, 0.7510992490999063, 0.8109224206634544, 0.8423465822008678, 0.7860561364054734, 0.8209377294728035, 0.7367101296684907, 0.8003605578734065, 0.8475069139527052, 0.8397859719989945, 0.8667085886621215, 0.8709132151503701, 0.8647809724049449, 0.8250485576236929, 0.8294518367335603, 0.8413074964125634, 0.7669983227944742, 0.7859288648057355, 0.8048650584880538, 0.7874743989034648, 0.8372126400112491, 0.7644510440541329, 0.7291693664097031, 0.8133736811839789, 0.8226640018526393, 0.8410368291172762, 0.7761202393946752, 0.8251031866923167, 0.7957211513676121, 0.8389767238315665, 0.7468810321291339, 0.8876023887213904, 0.8375440795228944, 0.8966506199135523, 0.7914608147117445, 0.8084114645223899, 0.7621159493467172, 0.7701421204886127, 0.8185724205450141, 0.8559419023574835, 0.8552376177943034, 0.7986135889076309, 0.9028271829347294, 0.8187110269682235, 0.7661511874409027, 0.8465796809290645, 0.8775104966324322, 0.8536260830181643, 0.8528202615456354, 0.9235334700107762, 0.9453011079718007, 0.8599734844246506, 0.8596418623388664, 0.8371907939479846, 0.8478053664972508, 0.8044157316545337, 0.7352056370512885, 0.8608322927467199, 0.8540284077088858, 0.8289902048385507, 0.8624152229941019, 0.8392327596841619, 0.7841289800699448, 0.8434684271814635, 0.920606701439069, 0.8599734844246506, 0.8633251704976141]

    #11*1 tem01
    # measuredlist=[0.8250557423552831, 0.7367869477540264, 0.8702613467677986, 0.7717284754898386, 0.82430526168514, 0.789914596615007, 0.829782589539004, 0.867448670635903, 0.9340572384770985, 0.8578627683775045, 0.8807864893186127, 1.0, 0.9068762311641432, 0.8494274328117886, 0.9265600962792654, 0.795033901855943, 0.9847145066126559, 0.791393515532803, 0.9517424780315795, 0.7394707461409011, 0.868370189998113, 0.7069394852929907]
    # measuredlist2=[0.9606576178606444, 0.8959477396922586, 0.858518613067637, 0.9642195912788575, 0.9478599514314308, 0.9616777001063902, 0.9603517850174241, 0.8809813738234382, 0.9414953949128965, 0.9661514389936849, 0.9169109501375723, 0.8637025400651956, 0.9327728428855525, 0.9788883041236855, 0.9922323125306233, 0.9422377390105106, 0.9450696519623029, 0.8917227227854803, 0.9183216487778515, 0.9757499164462918, 1.0, 0.9911303146306695]
    # measuredlist=[0.7517477160296847, 0.8064450715802828, 0.8214322137850009, 0.8097536325087087, 0.9155143987308968, 0.8386239144238745, 0.8457382719261097, 0.8819582430317416, 0.7892236384323558, 0.8519641313839206, 0.841792737439132, 0.7427625631756984, 0.7681705541593814, 0.830527298863882, 0.806021866479161, 0.870894165664351, 0.7688751364358785, 0.8425109480384408, 0.8349711663220457, 0.7899769372564366, 0.8014276165114992, 0.738712553128494, 0.8366939753803344, 0.7930626985963891, 0.6943578192151237, 0.8080475012401884, 0.824033550071347, 0.8182514168675508, 0.82085885610704, 0.8602770787301287, 0.8125102164588308, 0.8531548659047632, 0.8109034014130772, 0.6744117406291029, 0.8011266303848497, 0.8230196426957159, 0.8154646859202077, 0.8860851228958536, 0.774137277023141, 0.8623134387433329, 0.8471102199159745, 0.8026440946432714, 0.8238603780527548, 0.8004034649257435, 0.7437067108530627, 0.7724797332924664, 0.7650126357353189, 0.85791448172081, 0.9270504793131719, 0.8894130046359774, 0.8952228117677477, 0.9266891297453178, 0.7707901062138948, 0.7957232821936198, 0.8006840427590234, 0.8092704618740565, 0.836087666780449, 0.8421424662237286, 0.8762052841767664, 0.9218826679870074, 0.9201457273513737, 0.8794242846938771, 0.909168278284228, 0.8068970545210522, 0.7983125055377841, 0.8467785282138203, 0.7977330068752363, 0.8568458600380531, 0.8840564370029498, 0.9078740841200962, 0.9601118789998379, 0.8784448071167792, 0.9701406814982696, 0.9049591849553198, 0.8581255722993394, 0.8099170263570207, 0.8467392964762666, 0.8688732658986986, 0.8716865068490528, 0.9229169804105352, 0.9135152338286467, 0.887554340487714, 0.8927948367447069, 1.0, 0.9345942502894746, 0.8798509893532317, 0.9322841741932935, 0.8846569738565729, 0.7441314375091378, 0.804486749318681, 0.8270717085655619, 0.8801964753953561, 0.9145610406567197, 0.8127334507320321, 0.9105304354594972, 0.8504951645073245, 0.8213976828861834, 0.8905847140242948, 0.7927415025538715, 0.8507078277240979, 0.8166122845128738, 0.8656020704505238, 0.8713117959783754, 0.9542669161863183, 0.8404295067238312, 0.9520389728981623, 0.8959455700439161, 0.7469580982946161, 0.8838931551218074, 0.775507486839911, 0.8823884630782183, 0.933976144400124, 0.9133496094132784, 0.9038394482799549, 0.9677885515846245, 0.9408959909712553, 0.9623401010113354, 0.9964071016088846, 0.854418517206639, 0.8861235180888145, 0.9285472127524874]
    # measuredlist2=[0.8357840482371337, 0.8098423136637405, 0.7881295125854817, 0.7784536670066787, 0.8573192059832138, 0.8614263719481335, 0.8080557219993185, 0.8286366627365085, 0.8168951362334037, 0.8484550724488737, 0.8984855075031284, 0.8030016825088819, 0.8186148726737293, 0.7994956910034818, 0.8250602288763951, 0.7445975878560076, 0.8014587430983048, 0.809289322358554, 0.7628410759937766, 0.7932192478425698, 0.8154475417103284, 0.7453127521334137, 0.9824957253917884, 0.829045888859754, 0.7177346014984701, 0.8487563308276213, 0.7686795395287143, 0.8868317368896358, 0.8497704619819726, 0.8639445328082529, 0.8807027191946203, 0.889530773031478, 0.8683531924032207, 0.6836085087915221, 0.8323778532288519, 1.0, 0.8421817850549868, 0.882201736233937, 0.8332473700699045, 0.8881858957068278, 0.8412834078999282, 0.8600939489868975, 0.8215793617451994, 0.7862337586528424, 0.7893566265109806, 0.7191436245828629, 0.7468405836484665, 0.7964004198073724, 0.8164374440140952, 0.7921790283660497, 0.8103630760174994, 0.7826560884784408, 0.7757429358729871, 0.7936753457672011, 0.7933041367056114, 0.8510870382469892, 0.8745877466088567, 0.8060251564169284, 0.773391155337353, 0.7869723357720211, 0.7898686911044144, 0.7667742892957402, 0.7614740237960176, 0.8243091406211163, 0.8260083476103204, 0.8248895626606118, 0.6939832369309197, 0.786639378132486, 0.7587286121764275, 0.7232140461695608, 0.8388015484299005, 0.7444904853486416, 0.7229113281604725, 0.7253957166672591, 0.7689929500871849, 0.695690702690928, 0.7568756310400806, 0.875380728966613, 0.8672246814446866, 0.9332896601044024, 0.8223782167055339, 0.7571395321324222, 0.8176976236396054, 0.8441432731181733, 0.8266426279401816, 0.8307540730371007, 0.931104548154352, 0.8967910766012851, 0.7767492165160682, 0.7953131867072023, 0.8102767367584183, 0.8526895753087843, 0.8065246950433105, 0.7913678325216128, 0.815651740616128, 0.8233022575658083, 0.923875240213269, 0.8317795996349309, 0.8689999579077642, 0.9651604128754253, 0.9159751578818253, 0.912064945238924, 0.8365297257319082, 0.9232482557708257, 0.8916915722732011, 0.9250115811206304, 0.8732595906333368, 0.8055194478258084, 0.8238813914655431, 0.8551124320605028, 0.8364953233830722, 0.8124631048441249, 0.8651622910563763, 0.8612687415835466, 0.8640720367542546, 0.8654732466551459, 0.8759263127983005, 0.8242330414205703, 0.8549932597864701, 0.861957013533507, 0.8589296019100988]
    # measuredlist3=[0.8679493695924059, 0.8325405298220532, 0.851570507308853, 0.8264398052553323, 0.9056980335669443, 0.9260554395325342, 0.8585027191845201, 0.8622062481150035, 0.8640372089540799, 0.8937686981268784, 0.9367519848045819, 0.8289294816109893, 0.8447360830967681, 0.8700420926121094, 0.8581948666755842, 0.780808787779585, 0.844287746833775, 0.871934885736627, 0.8072487043276612, 0.8357745679592125, 0.8475218748842566, 0.7802531443962892, 0.9860627282942432, 0.8529430263072787, 0.6923016734534423, 0.8561425275096121, 0.7789572017132533, 0.8987806764097983, 0.8503506577646324, 0.8703461955721852, 0.8937928470665848, 0.9059223562455597, 0.8724726296544382, 0.6836391203223052, 0.8322797564449638, 1.0, 0.8560341086176245, 0.8787417824946387, 0.8589511371556616, 0.8781171592250879, 0.8462938297658643, 0.8750025424313131, 0.8254075443528239, 0.8083524980131698, 0.7873295586963323, 0.7488552611775595, 0.7408655255766963, 0.8116736067346065, 0.8356754265119575, 0.7820685374670625, 0.8121339115554456, 0.8079225277662144, 0.7792362674913161, 0.8142140833339211, 0.815642385570427, 0.8615700605383092, 0.909097987145763, 0.8477792629329335, 0.8227079926940181, 0.8227559611533115, 0.8163917999859095, 0.8031912563839017, 0.7850801525192747, 0.8642020930548487, 0.8430333083409415, 0.8833059510168515, 0.7226891889345717, 0.8345161264007906, 0.846321495688279, 0.7875731522279884, 0.9018341674099903, 0.8118939756267011, 0.7563927207823371, 0.7527869842861025, 0.8189284576899796, 0.7440916554830044, 0.7976073035755203, 0.941107955310527, 0.8794557189658706, 0.9813185942899119, 0.8804954614972704, 0.7906664989037178, 0.8976438441636857, 0.8988918154415851, 0.8762236778216647, 0.8918652793250859, 0.9850270775580298, 0.9369812535963113, 0.8135000035094256, 0.8589354845722279, 0.8515228246117355, 0.9005407429729739, 0.8467884550748888, 0.8357346460008801, 0.8611427833545662, 0.8571789707367671, 0.9770873959615453, 0.868475768943985, 0.8950085077456164, 0.9735072548671492, 0.9064953943075083, 0.8968919571284016, 0.849558623512362, 0.9178686919609347, 0.9170821693617512, 0.917953379032719, 0.857552952580327, 0.8185833970153614, 0.8202689015055744, 0.8521152543079862, 0.8510457752707861, 0.8167846278276483, 0.874938858818774, 0.8735686219654233, 0.8708623005289745, 0.8692811116993137, 0.8874835045552603, 0.813240881986258, 0.8720861821213729, 0.8793152360492686, 0.871832463915997]
    #
    # measuredlist=[0.7356598366456737, 0.7709394281918857, 0.7748247354457821, 0.790243628656148, 0.8975066947164021, 0.8452187469613592, 0.8176303958736874, 0.8020313168543801, 0.796384627339719, 0.8341319037550213, 0.7708808284950701, 0.6760148465652459, 0.7592864718321054, 0.758932935728734, 0.7926879907052016, 0.800538253912695, 0.7726673784687611, 0.8233687332637277, 0.796372869444743, 0.7322217634104659, 0.8024206378616806, 0.7395576826019163, 0.7275786516231149, 0.7861154688574712, 0.65046256218779, 0.7429752630682165, 0.8032755207194504, 0.7667324811361845, 0.8063598406283438, 0.794796060505313, 0.769875500622927, 0.8362439004447285, 0.7511413546251587, 0.6538682367780114, 0.7180804302457625, 0.7703112096294327, 0.7891104391186909, 0.7789459454600108, 0.8185092169704613, 0.8032773095773924, 0.845055372663845, 0.7342575217123202, 0.792993790963154, 0.7496070387488352, 0.7129272470641207, 0.7763255555787566, 0.6722809492593916, 0.7963088065853019, 0.9447935572790395, 0.8515917997154777, 0.8405227594676975, 0.8505563540227731, 0.717169916441396, 0.7625826241001511, 0.7712854542032623, 0.7245403147018308, 0.791244028397615, 0.8250475353986235, 0.8107986360157963, 0.9536048215355513, 0.8403059825863811, 0.9166820769276466, 0.7939859264041853, 0.8251101696258133, 0.8196750706245547, 0.8034413776996803, 0.7903703940461151, 0.8213629806505441, 0.8769413104604495, 0.8360167711261136, 0.8832386619342787, 0.9080294587723549, 0.905552263415131, 0.8269658136772442, 0.8086900656277333, 0.8057589585087946, 0.7825045535023432, 0.9027970950765163, 0.8471777913302807, 0.8834036265786898, 0.9390896636507832, 0.8838146298100769, 0.896144984371311, 1.0, 0.875382747251624, 0.9056445592776765, 0.9404188172353961, 0.8030538230661094, 0.7797728765002961, 0.7749448005373957, 0.7675542704997549, 0.8767580555498804, 0.8570145663017359, 0.8507156097103606, 0.805446919425181, 0.8220963971807915, 0.8020282037522093, 0.8510397549578843, 0.7480665825042876, 0.8440044461088005, 0.798428289308369, 0.8337412241296622, 0.8711406198461638, 0.8911531321182139, 0.8815636014403636, 0.9198450976430858, 0.8442221181931032, 0.7545203985469309, 0.787730851062836, 0.7509763916583787, 0.8628003121937159, 0.9024467301346917, 0.8528156792083605, 0.8815232011218539, 0.9489769138414794, 0.9176970148506319, 0.9903848463331646, 0.8822382408641111, 0.8734874072961905, 0.9209977686269992, 0.904931722152924]
    # measuredlist2=[0.8824918397366254, 0.8579305320609845, 0.862804085376876, 0.8125422062571802, 0.9265770681098525, 0.9032117430720644, 0.8488226562667438, 0.8908457120190493, 0.8832178151641313, 0.8794056269482311, 0.9022585008468934, 0.8009356388105047, 0.8460263647689901, 0.8581016893385387, 0.852095036350773, 0.7533235083952454, 0.7635567869359681, 0.7976103237453949, 0.8442877754774395, 0.7679205428409092, 0.8416543283844562, 0.8369840388238574, 0.9811201815476347, 0.9117982641856561, 0.7387613885087267, 0.8047532996420779, 0.8593551964017685, 0.7899927199132412, 0.9073102154282697, 0.7751314577282714, 0.9101674071430313, 0.9113873775920596, 0.8621198183065965, 0.7614998081110973, 0.7463418733908864, 0.9760760334949451, 0.8708249494501309, 0.8112381091907991, 0.8558767915777739, 0.8611866531448803, 0.895409801079872, 0.797937849419097, 0.7966197025026767, 0.8215228269113227, 0.8841942369774819, 0.7935117774469475, 0.6763963080113511, 0.8291774320898796, 0.8402636124487892, 0.8463811750710958, 0.7391896967369581, 0.8859063589558538, 0.699871522418679, 0.7909522238513743, 0.8172796941615637, 0.8690772889704496, 0.8366292413347051, 1.0, 0.842977668252538, 0.8395227829294657, 0.8258305207795237, 0.8989266884254556, 0.7750621895032528, 0.9322800989102416, 0.8904835861310075, 0.8107918036947405, 0.7447013478669707, 0.7938073333597965, 0.8286530893864972, 0.764446384818795, 0.8225386089086764, 0.8326818029837729, 0.7500163032719669, 0.8036263740484916, 0.7646182964207897, 0.6896381499505596, 0.7950544489029289, 0.9353458315348506, 0.871106198771975, 0.8788356951859481, 0.9931959083969963, 0.8389534861807671, 0.8273494895773159, 0.942102821104872, 0.8969679439638735, 0.9628621477221484, 0.9809688900568109, 0.866513149575762, 0.789371106843395, 0.7980280182102969, 0.7762343650757848, 0.85346801737036, 0.8058873045896904, 0.7964763067061316, 0.73122718252687, 0.8071190387884402, 0.9058924423250477, 0.889054389561248, 0.8219321754828275, 0.8914028303713748, 0.8724024589803506, 0.8595107190775754, 0.8006760154254137, 0.8234586367528488, 0.8275789774595351, 0.92990635398848, 0.867146875682932, 0.7918579687030477, 0.8158674402743207, 0.8567536214897267, 0.8437131510095041, 0.7945956003361226, 0.8489844424223657, 0.8248754078040157, 0.8123226375630268, 0.7987287297823711, 0.8250550455053512, 0.791652702125427, 0.8019523924179545, 0.8627522170040379, 0.7897667719210661]
    #
    # measuredlist=[0.8103387713652084, 0.9396305650510194, 0.8455262837644664, 0.8517603099428861, 0.7787434982332848, 0.788389913581711, 0.7960793940889901, 0.8334969586573509, 0.8824143243512894, 0.9702874503165865, 0.7998849317297407, 0.8476599719609124, 0.8682945072631483, 0.7768268334531193, 0.8866962441895697, 0.7762790575159952, 0.7952134855648342, 0.8195979170614336, 0.8369765507561888, 0.7929562708559498, 0.9037800467695287, 0.7990546254636943, 0.7247580885719214, 0.8266527219036852, 0.8718545795099737, 0.8169524664763698, 0.8069058028489587, 0.8131463174227485, 0.853665966276833, 0.8029039285271549, 0.7631713633487076, 0.8337708231582713, 0.7813095079795923, 0.712414826696922, 0.7475247638815307, 0.7500691748244546, 0.8668639960637616, 0.7862313352276102, 0.834474521134378, 0.6678113891087027, 0.7763412398126596, 0.7281705560500215, 0.7460506881183988, 0.7612599694729155, 0.7596355801722765, 0.8477633562955631, 0.7765672929022809, 0.72541187010566, 0.756610126395376, 0.7245229258377305, 0.7795679914586875, 0.8064248415696016, 0.8115072300254225, 0.8931316695240833, 0.817078491151164, 0.7183490146387393, 0.7956382069525755, 0.7817056856156578, 0.704872614984256, 0.7584844572038303, 0.7342657569424281, 0.7881884304457895, 0.7373139933745424, 0.7379024706124916, 0.8536566845612619, 0.7670800133091686, 0.7745698119479083, 0.8010728002515443, 0.826923755379833, 0.8418377753765164, 0.8406394514637516, 0.8505833076953702, 0.7614195032084725, 0.7721595609477503, 0.7530128855010384, 0.7293061732444831, 0.716492897451896, 0.8590446617057265, 0.8830648893439444, 0.9211743005324183, 1.0, 0.8899797403107225, 0.8719169996572712, 0.8377547905280185, 0.801984884906999, 0.8393703699045802, 0.8587417989399954, 0.8043019649775249, 0.8514006210153156, 0.9007446733919229, 0.8553505733941933, 0.8943810770048923, 0.7561044588556226, 0.7857725695010209, 0.7976294442569345, 0.7877940583288722, 0.8488701228242724, 0.924080482710293, 0.7888085330179059, 0.9425793333230162, 0.9512859221149897, 0.912781021956968, 0.9922579388970356, 0.789181051346119, 0.9257621191761576, 0.8812506300552742, 0.8061189994948479, 0.9376339850476134, 0.9879377097434784, 0.8781312955295036, 0.9592718733653092, 0.9464056213655561, 0.8881378453751012, 0.9152206411687777, 0.8916285874581138, 0.8359201275599709, 0.9581875297790513, 0.899369676480699, 0.9641915557667762, 0.9658903332362907, 0.8651933277634162]
    # measuredlist2=[0.8098770838027838, 0.9214473005813058, 0.8718054720900787, 0.8909159381682994, 0.780378362691307, 0.8119801671293684, 0.7792243010229981, 0.8546165421046978, 0.8967206808990659, 0.9217784923791327, 0.83339233177039, 0.8553038217786451, 0.8681926066543364, 0.8330205514889278, 0.8539644736483759, 0.7643363015090813, 0.7763513872203555, 0.8450044787709192, 0.8423605751946072, 0.7722979861130356, 0.8802658529850557, 0.8228554830085483, 0.7789732907359939, 0.8134466724364617, 0.831990619331732, 0.8589299160261373, 0.7597316199984057, 0.8282377163605572, 0.8199000374438661, 0.8046460389494862, 0.804858039789964, 0.8083854208163526, 0.7695652079486799, 0.7874280688036063, 0.8270742790801194, 0.7681865739228522, 0.8670435735027638, 0.7820079870038082, 0.7844596464908518, 0.7190580770225753, 0.8178041277286491, 0.7446501631257505, 0.7795698771262717, 0.781914964307222, 0.7313009561667089, 0.8507626088185593, 0.8101960377924639, 0.7633610024044364, 0.7560162945496963, 0.7368991575079712, 0.7905672666051566, 0.7853159979794906, 0.8168396874160726, 0.976668130248296, 0.8074165433112575, 0.7842319296256534, 0.8290591436050959, 0.7662756945462678, 0.7689759227819059, 0.7775490769494817, 0.7401073878658679, 0.762366258478892, 0.75828093125589, 0.7797617054545386, 0.7988989658395187, 0.7406498981555845, 0.7597026211638555, 0.7676743943763381, 0.788214347755202, 0.8454575354516397, 0.8057903484210865, 0.8548628917034603, 0.7654652367283032, 0.7619694609751332, 0.7536293982980932, 0.7443701801939085, 0.7199827514515834, 0.8386708512936877, 0.9130400704119778, 0.920458422343496, 0.9552921830619624, 0.9183414079062909, 0.8825770893059693, 0.8212670386936471, 0.8790244888000758, 0.839629893601451, 0.8182695284480476, 0.8958318213343268, 0.8323118109971056, 0.908082612825902, 0.86016973967743, 0.8928538257599739, 0.7782564318148238, 0.7402774491786323, 0.7771000859570663, 0.7660208793585418, 0.8173956291847729, 0.940964613213378, 0.7848856822815536, 0.977984712126078, 1.0, 0.8919570853028123, 0.988727042918249, 0.8499239152053589, 0.8848418798268246, 0.8789121261646691, 0.889533841772203, 0.9745501851591928, 0.961572858223188, 0.8913104659806671, 0.9495902410177925, 0.9932038645351515, 0.8760841660896588, 0.9854024008603892, 0.8763726906634863, 0.9102006676684343, 0.9244785092139659, 0.9839165606521393, 0.962051317739882, 0.9544798566398178, 0.8539223285134097]
    # measuredlist=[0.8709085916239567, 0.9389686328376317, 0.8779979902843721, 0.9012332971416162, 1.0, 0.9088192686778999, 0.9353477205833058, 0.9089938626802632, 0.8393348426920108, 0.8761970739104595, 0.8545730892453299]
    # #
    # #measuredlist=[0.9300170174544179, 1.0, 0.9597044728343397, 0.9718463081191726, 0.9154011535922214, 0.8446235008539856, 0.9534148882978251, 0.8701368750106212, 0.8959055162170217, 0.9552763426158957, 0.8707611052005036]
    #
    # measuredlist=[0.8010836243371336, 0.7925957366539216, 0.8308107894050137, 0.8321871900306828, 0.83280636951376,
    #  0.7656245376341643, 0.8527367802790912, 0.8985650827367335, 0.8974411633650258, 0.8716031769619065,
    #  0.8794384890610221, 0.9927494599958977, 0.9229885132702126, 0.8918461582143806, 0.9670496758713669,
    #  0.8422577797524411, 1.0, 0.7869678626082859, 0.9642245176296137, 0.7270624691903079, 0.8703860914705239,
    #  0.7206753849526155]
    #
    # measuredlist=[0.8458248759356296, 0.9007403695713692, 0.8555780438839258, 0.9116795014944963, 0.8864210382218866, 0.9126009601555499, 0.874320507786373, 0.8529404309959503, 0.891751924807843, 0.8068755571244631, 0.8497482744833181, 0.6980496127821214, 0.8746804839264707, 0.7493692095356722, 0.685710071051083, 0.8306115743041198, 0.7734203940942133, 0.8883528119192184, 0.7847147097729433, 1.0, 0.8371126627157485, 0.8211449522989759]
    # measuredlist2=[0.7850297788992595, 0.7869894068004767, 0.8237558459945725, 0.8946511658135978, 0.8185320562929762, 0.7447381676065544, 0.8544198634703762, 0.8933542062091352, 0.8927482692197861, 0.8384455469051734, 0.8998930235073549, 1.0, 0.9258413270286047, 0.8684055039430273, 0.9691477622625362, 0.8496937752725295, 0.9785446646606133, 0.7847060308590896, 0.9347076231622599, 0.7460130046751626, 0.8496622775912533, 0.7102290915257042]
    # measuredlist=[0.7118106785873429, 0.7983101862667219, 0.7900755021608689, 0.8205180871518976, 0.842442160688628, 0.8737540272881076, 0.9287817343998996, 0.8562948113545984, 0.8747168596489356, 1.0, 0.8694339878764249, 0.9356706553458648, 0.9178416257944285, 0.9591882770508905, 0.9676547104053351, 0.8722969065922964, 0.9786821482786959, 0.8554488943974882, 0.8149484398106636, 0.7768923886417015, 0.8891364178613942, 0.6750072505492534]
    # measuredlist2=[0.8536399554850009, 0.8638895169130791, 0.8650159311974134, 0.9421833197562194, 0.8790028238070394, 0.8604751799082978, 0.8723190448745259, 0.9387204231266675, 0.9179905927792058, 0.8955465766475578, 0.9217883733790184, 1.0, 0.9166226482624508, 0.9129982545473033, 0.9157505847782118, 0.8816114699653914, 0.9206224594003133, 0.9178999523514598, 0.8834760337245426, 0.8984919363218331, 0.9363174746171669, 0.7568062755619968]
    # measuredlist3=[0.9270386871548318, 0.921155784051533, 0.8885137890054752, 0.7742110288346699, 0.8362387573191185, 0.9681424919215458, 0.864556583992129, 0.9259528819623865, 0.9418915311477792, 0.9286738267436224, 0.8986299643580855, 0.7511326985975121, 0.8474403845978867, 0.8439845183333645, 0.855446833812117, 0.891823793603063, 0.9552067005073611, 0.8411693016205936, 1.0, 0.8012947183895032, 0.8962540265033896, 0.9717202189036921]

    # measuredlist=[0.6779465545373562, 0.6150801511650839, 0.6130768501105833, 0.5824954905011182, 0.5514477327875978, 0.7240655880882882, 0.6210378018146047, 0.6449895730496915, 0.7516917512864715, 0.7422689008177269, 0.6582462759704218, 0.7438539048791968, 0.7560836963413683, 0.6800468833758463, 0.7143488121536868, 0.6845685441732295, 0.7944913606345259, 0.6790252381600709, 0.7370179363901167, 0.6297208048568544, 0.7033674542487165, 0.6224315577592067, 0.6107979954096224, 0.50656461647551, 0.6201704589441868, 0.6385951297779655, 0.5997991224607488, 0.5960649813006268, 0.6852731671144451, 0.6926833741306551, 0.7424246506355198, 0.7672985767920896, 0.6833387797804188, 0.7690471138101335, 0.6493781421949228, 0.7422141428282996, 0.670119317382584, 0.6223948035775905, 0.7323522992697427, 0.5463765497546198, 0.66957479053174, 0.5563726016582691, 0.6989705313548028, 0.5693063309679769, 0.6604527077141423, 0.6114293494084609, 0.6287895692227738, 0.6917763602905764, 0.5916561293903062, 0.5828075251476862, 0.6869789539104437, 0.6708095154281776, 0.7379930906764542, 0.7287750562622637, 0.7143786463217212, 0.7725776451253693, 0.7979648681424556, 0.7432514625666653, 0.7872616807703965, 0.6278912059829617, 0.7805221187093184, 0.6505449382850013, 0.8683598606914547, 0.6548140805975514, 0.7309609250026747, 0.6196752996342006, 0.633488177912184, 0.6468734193326272, 0.6092334945870012, 0.6997073200960399, 0.6768249060641954, 0.73925075551509, 0.6200206728361493, 0.7595019426317132, 0.743897009203398, 0.7925832482971173, 0.6642383242979558, 0.8035973153326084, 0.7637159699601395, 0.7456281529005583, 0.9246464053488982, 0.6746418988095064, 0.7581010158296918, 0.6215810061752749, 0.7860116052646472, 0.6076794115991733, 0.7719672820535034, 0.6354879242496612, 0.6474317813034557, 0.5391496749134109, 0.6937692496784825, 0.6631690526089622, 0.6623605681777667, 0.6399364499450656, 0.6939815208561136, 0.7203852753598209, 0.8394105138216925, 0.8848133187898712, 0.760379336122842, 0.779610252630903, 0.7690659067601208, 0.7884782168983745, 0.8112166960145396, 0.5991913307288966, 0.7662363403647863, 0.5997587207082663, 0.6684596956761879, 0.5760501603512601, 0.7467673637433292, 0.5510288194141598, 0.6145788538293184, 0.5806984869241336, 0.6473143217359312, 0.6360312148236403, 0.6878545653263055, 0.5737296663158108, 0.6886854603623923, 0.7076432299443745, 0.7659879892852051, 0.7137301224799028, 0.7048415821764533, 0.7690509244777657, 0.8421088115339671, 0.7241053551642741, 0.7441517264531973, 0.6186352945078069, 0.7948504185515245, 0.6153073564811301, 0.6578514391209201, 0.583107361950104, 0.6511661525633547, 0.614281915785161, 0.6507201447016969, 0.6351346520879935, 0.6783656642255866, 0.698730483571164, 0.6418676883232792, 0.727968722106391, 0.6072885307678975, 0.8218865482126241, 0.7025090681867997, 0.5765085666004656, 0.7551085547925422, 0.7999482051116144, 0.787489776942727, 0.6498070406008187, 0.7783875203057212, 0.7019343596593693, 0.8150675507979264, 0.700097232138639, 0.6883082397992636, 0.4840178449224803, 0.6901602099405991, 0.6378121818756521, 0.7625341725597371, 0.7239515623721867, 0.6706164318480807, 0.7528297475412542, 0.6717741214179098, 0.6978791405076484, 0.7529513005594597, 0.747163466843332, 0.7983471637103556, 0.7019616924876522, 0.7028890350403735, 0.8348545346004652, 0.9348225945501862, 0.7657226150573394, 0.8360628009113122, 0.7384111815771898, 1.0, 0.7449069231542462, 0.8539428209389169, 0.6553987945930302, 0.6898135182676259, 0.6231075780950148, 0.6890590546788089, 0.6445201080642923, 0.6028186651682678, 0.7574314888273168, 0.6579604075940784, 0.7101563161610456, 0.8444029201429766, 0.7477617684873331, 0.6719592084363535, 0.8771202926143704, 0.8383188999516884, 0.786782913471311, 0.7737776258065919, 0.7941179209251188, 0.7629463211963905, 0.8051545784127844, 0.944817912501443, 0.7230200889992934, 0.8680338186845125, 0.6218931288536889, 0.8009152504184753, 0.5906788592979718, 0.7369271268153728, 0.765643785768457, 0.7574574939023843, 0.8011250156454336, 0.7207997258248097, 0.6892987573900473, 0.8031345800902803, 0.8372103642424483, 0.868324890307288, 0.6700058432132213, 0.7764380872859471, 0.9254685997463972, 0.7987011525993306, 0.7931816627265682, 0.8330786971164912, 0.8292371961754192, 0.8166190814797692, 0.7130599199203235, 0.8031059772364715, 0.6275406244663868, 0.7142258921037792, 0.6258054852037418, 0.7368641452389341, 0.6883979776707048, 0.8218049045260123, 0.7884143754160681, 0.7749529192202089, 0.6628784611536485, 0.779021041502109, 0.8338177456318485, 0.961654899518357, 0.7702750752224269, 0.7075216719856968, 0.9825279389436465, 0.9177462530104897, 0.8310007553458665, 0.8025774101706945, 0.7581557953056152, 0.9620406272673769, 0.7125432785910017, 0.7873079134351687, 0.7439319024069886, 0.8186997782710868, 0.7196046676947558]
    # measuredlist2=[0.7571000445563187, 0.756100569731273, 0.8611352575595841, 0.8879005581823359, 0.7971124515570044, 0.6691255509625587, 0.7064237852249895, 0.7929378885346315, 0.6840815636216785, 0.6747273619009989, 0.7589324947835843, 0.7892078816531597, 0.8684869579011817, 0.6368731453811509, 0.7431011684302363, 0.8366341836480687, 0.7726305535179028, 0.6660601364611713, 0.7619424953280909, 0.9079274683898618, 0.7494979306261639, 0.7530676988243828, 0.7203178216879066, 0.9087604335395847, 0.6872689930846588, 0.8788150081521289, 0.7157565974718576, 0.767752951399196, 0.8212369706035028, 0.7320372395713414, 0.6117302561869196, 0.6789689061875057, 0.7900044960974375, 0.6622466301780812, 0.9356720596356518, 0.7373011817530443, 0.7670955544168163, 0.7365780258759459, 0.7295380774679703, 0.7549429591036756, 0.7836338738114592, 0.8285801898240043, 0.8197286550693412, 0.7348100616333295, 0.6503706123729415, 0.7748888554693959, 0.867898017352398, 0.7943961353036124, 0.7919328544827469, 0.8080093307898798, 0.8863187666784058, 0.777806716538956, 0.8202858566894762, 0.6683646454650616, 0.6751250972903298, 0.6997577361669705, 0.8344801432261032, 0.6744671712388728, 0.6637349465730352, 0.7560933300288825, 0.7339397804874994, 0.6764004628828703, 0.6921961437538525, 0.8203779774063494, 0.6948176539476834, 0.705147966793316, 0.6250715805547405, 0.6778784659054109, 0.7663909483477624, 0.795240702116589, 0.7814938439144239, 0.7100074431035778, 0.8300240582526405, 0.8221046892332197, 0.8747861105633787, 0.8473478217071859, 0.8670388980531619, 0.7163032552056957, 0.7453919746108688, 0.6375905156440829, 0.6831546294804034, 0.8059366460083482, 0.7163391106106841, 0.7655988641390961, 0.6837518203724529, 0.8231748980450035, 0.7655122087525782, 0.679377048790179, 0.8204044143873576, 0.7614127116568363, 0.7444067412626697, 0.7669820764407514, 0.7838332461166732, 0.782213574735996, 0.7892825416391457, 0.7810040221310081, 0.6315838477412249, 0.7695082873023279, 0.7289962328177558, 0.7353378435046362, 0.7746562991726248, 0.7490043063962739, 0.713618405844073, 0.7826341268346703, 0.7493914625160517, 0.8289353873139892, 0.8298868223842221, 1.0, 0.7847908436650509, 0.8320045558464504, 0.7615777568912827, 0.7951739893461736, 0.7579502842467739, 0.884969535534551, 0.6844031833628588, 0.7698707827498733, 0.8104046836176324, 0.829365800125769, 0.8213103262877408, 0.7475657081093484, 0.8333191678837768, 0.5941133384424693, 0.7269617733628059, 0.797643329961133, 0.7366167220334452, 0.7433385175956577, 0.7615184308374622, 0.7849098535618131, 0.8084207236459112, 0.8841580207125993, 0.8136783198708337, 0.7862628160509091, 0.6571646188099101, 0.8021213520188012, 0.786015920478842, 0.7842562867147187, 0.7771163209300165, 0.744021106057045, 0.8046675011331893, 0.8381202923825856, 0.8113355095690689, 0.8365927085859995, 0.7859325123756717, 0.7482565561088007, 0.8140745207980427, 0.6618189433168151, 0.671043186440081, 0.8894903364145711, 0.7827985579399255, 0.7375128204717586, 0.8501710309985364, 0.8244940904911344, 0.7423448547462999, 0.6670900923247824, 0.7722551250561571, 0.7137389215382864, 0.801458512131841, 0.8920057522691519, 0.9772437702286557, 0.7751530017916165, 0.8068025660103727, 0.8631796161972134, 0.9751859678941476, 0.8593584037085286, 0.8804686016150078, 0.7769542144984795, 0.8570837269910072, 0.7506052811533686, 0.6380898208551649, 0.8389482072772966, 0.7020941039827304, 0.7815960387906957, 0.7269915176410863, 0.8581001104084038, 0.7489714326532614, 0.8735863169340928, 0.8193217115261179, 0.8304180600874131, 0.8028061979322196, 0.8861142825803168, 0.9132676008036095, 0.7760045270518665, 0.7736647682687987, 0.7924014329627491, 0.9048710784850846, 0.7500344263751396, 0.8356788354992168, 0.6923116098411886, 0.8708421681852253, 0.6418770513255744, 0.726468468643436, 0.6719634576370118, 0.7304415530299971, 0.8139772782646397, 0.907962550833819, 0.8537925676212371, 0.7897900546997334, 0.7074446955373364, 0.7904266801566819, 0.7862954609036438, 0.7799854682850129, 0.8916119424474203, 0.8376496624988241, 0.7857766849539234, 0.7787190293636215, 0.7812026224791546, 0.7563368837528057, 0.766263498443721, 0.9138779710868595, 0.7010295305841033, 0.908503944550496, 0.6061324778979914, 0.678940024874961, 0.7731687715624562, 0.8135050728446162, 0.8730844209397438, 0.8407202593598397, 0.8589183319961563, 0.8844626997726804, 0.782809581097517, 0.8669283105983733, 0.9180388087954675, 0.7155038948385724, 0.8649913794710942, 0.8132269999719152, 0.819511254807994, 0.8113144514776288, 0.8619601601046862, 0.8311161190858011, 0.7441888243231426, 0.9592849634591839, 0.7103264310914154, 0.8010986878719877, 0.724114856038682, 0.8038867996240722, 0.9022787119273258, 0.8416325882325373, 0.8388136921346091, 0.8098486731043707, 0.7844885702934052, 0.8121393538530061, 0.8093487994849281]
    #
    # #measuredlist=np.array(measuredlist)**0.5
    # measuredlist = (np.array(measuredlist)*np.array(measuredlist2))**0.5#*np.array(measuredlist3))**0.5#*np.array(measuredlist3))**0.5
    # measuredlist=np.sqrt(measuredlist/np.max(measuredlist))
    # print(len(measuredlist),"measurelist",measuredlist)
    # double_amps_in = (1/measuredlist)
    # double_amps_in=double_amps_in/np.max(double_amps_in)
    # #double_amps_in = double_amps_in[::-1] #flip horixontally only
    # double_amps_in_mirrored = []
    #
    # for i in range(0, len(double_amps_in), 2*n):
    #     double_amps_in_mirrored.extend(double_amps_in[i:i + 2*n][::-1])
    # double_amps_in=double_amps_in_mirrored
    # # amps=double_amps_in_mirrored

    double_amps_in=tuple(double_amps_in)

    #amps=(1.0,)*11
    #amps=(1.0,1.0,1.0,1.0,1.0)
    phases =tuple(0.0 for i in range(n * m))#(0,0.2,0.8,0.6,0.2,0.8,0.6,0,0.3,0.1)#tuple(0 for i in range(n * m))#(0,0.2,0.8,0.6,0.2,0.8,0.6,0)#(0,0.5,0,0.5)#tuple(0 for i in range(n * m))
    #phases = tuple(np.random.rand() for i in range(n * m))
    #phases= (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
    #phases=(0.0,0.5,)*60+(0.0,)
    # phases=np.array(phases)
    # for iii in range(n):
    #     for jjj in range(m):
    #         phases[jjj*n+iii]=6.5*((((iii+0.5)/n)-0.5)**2 +((jjj+0.5)/m-0.5)**2)
    print("phases0",phases)
    tem01_0=True
    x_pitch0=0.0089

    #phases = (0.5,0.4,0.3,0.2,0.1,0.0,0.1,0.2,0.3,0.4,0.5)
    #phases=(0.0,0.0,)*24+(0.0,)
    #For 21 ions
    #amps=list(([np.float64(0.14610046552680364), np.float64(1.1244909272813193e-09), np.float64(0.1639618888492886), np.float64(0.1639618876547513), np.float64(0.12690333764061396), np.float64(0.12690333661326864), np.float64(0.1269033375643812), np.float64(0.1461004652640547), np.float64(0.3196454695730479), np.float64(0.20934807080102394), np.float64(0.2525495568937741), np.float64(0.25254955657585504), np.float64(0.25254955703927556), np.float64(0.3196454696552676), np.float64(0.3196454692308991), np.float64(0.2525495557898602), np.float64(0.3196454704537126), np.float64(0.2093480714054639), np.float64(0.2093480696100511), np.float64(0.20934807110683318), np.float64(0.12690333778482576)]))
    #For 47 ions
    #amps=list([np.float64(0.04586915924100187), np.float64(0.0737158218486801), np.float64(1.7255334088047454e-10), np.float64(0.0737158216568236), np.float64(0.14632802444498844), np.float64(0.14632802425786048), np.float64(0.07047564049967088), np.float64(0.1397719421223463), np.float64(0.13977194218107503), np.float64(0.14632802458290414), np.float64(0.07047564062225226), np.float64(0.103546642494763), np.float64(0.18046184232764007), np.float64(0.08095818482328287), np.float64(0.1804618424515428), np.float64(0.045869159295217454), np.float64(0.15047272603228182), np.float64(0.07047564043911027), np.float64(0.10354664249300768), np.float64(0.13977194212480343), np.float64(0.10354664243863597), np.float64(0.22106155008059786), np.float64(0.2210615499737462), np.float64(0.18046184223521522), np.float64(0.14632802447973453), np.float64(0.08095818476931864), np.float64(0.10820207063869679), np.float64(0.10820207064264364), np.float64(0.07371582188085807), np.float64(0.22106155006897915), np.float64(0.07371582175215913), np.float64(0.1397719422967504), np.float64(0.1440219418384983), np.float64(0.14402194185591574), np.float64(0.2227630594106869), np.float64(0.22276305958322093), np.float64(0.07047564027944482), np.float64(0.15810114977095455), np.float64(0.15810114993268837), np.float64(0.15047272594532118), np.float64(0.15810114986479626), np.float64(0.10354664247580193), np.float64(0.22106154998023197), np.float64(0.15810114991473614), np.float64(0.18046184244441152), np.float64(0.22276305947986294), np.float64(0.22276305954786257)])
    #amps=amps/np.max(amps)

    #n, m, amps, phases, uni_spacing, x_pitch0, tem01_0,xarblist0, yarblist0, anglearblist0 = choose_pattern("tem00_47_1_u",scalepad)
    uni_spacing, x_pitch0, tem01_0, xarblist0, yarblist0, anglearblist0 = True,0.0063,True,None,None,None

    #double_amps_in=np.repeat(np.array(amps), 2)
    double_amps_in=tuple(double_amps_in)
    amps=list(amps)
    phases=list(phases)
    #for kk in range(0,n*m):
    #    if np.mod(kk,3)==0:
    #        amps[kk]=0.0#np.mod(kk,3)*0.5
    #        phases[kk]=0.0
    amps=tuple(amps)
    phases=tuple(phases)
    #phases=(0.0,0.0,0.0,0.0,0.0)#,0.3,0.05,0.9 )#,0.62,0.12,0.0,0.3,0.35,0.9,0.2)
    #phases=(0.0,0.0,0.2,0.0,0.0)
    amps_guess = amps
    # MDS #Target curve compensation
    curvature_list=np.linspace(0.75,4,1)
    for curvature_num in curvature_list:
        print("curvature_num: ",curvature_num)
        global_variables.curvature.append(curvature_num)
        a = 2*(-4.282766e-06)#(0.1027)*(1/24.0)* #-4.90392e-06  # -4.03456842e-06#-3.733022642775838e-06(1/scalepad**3)*
        b = 2070#mds oct2070#2095#2071#1937.23#2070  # 2070 original
        c = 2*(-4.482766e-06)#(0.1027)*(1/24.0)*  # -3.9478368e-06#-3.893853882563125e-06A(1/scalepad**3)*
        d = 1672#mds oct1672 #1628#1573.195#1672  # 1712 #1672 original
        a=1.15*np.pi/(1272*scalepad)*curvature_num#*0.6#1.2
        c=1.49*np.pi/(1024*scalepad)*curvature_num#*0.6 #1.49
        d_off=0
        b_off=0
        target_phase = np.zeros((int(1024*scalepad), int(1272*scalepad)))
        for i in range(int(1024*scalepad)):
            for j in range(int(1272*scalepad)):
                # target_phase[i][j]=c*((i-512)*104+1500-d)**2/2+a*((j-636)*84+2000-b)**2/2
                # target_phase[i][j] = c * ((i - 511) * 53.187*1.13 + 1573.195 - d) ** 2 / 2 + a * (
                #            (j - 635.5) * 41.045*1.13 + 1937.23 - b) ** 2 / 2  # MDS
                #target_phase[i][j] = c * ((i - 511) * 53.651 + 1672 - d) ** 2 / 2 + a * (
                #       (j - 635.5) * 41.4555 + 2070 - b) ** 2 / 2  # MDS Enter center of the beam here
                target_phase[i][j] = c * (i - ((1024-d_off)*scalepad/2.0)+0.5)** 2 + a *(j - ((1272-b_off)*scalepad/2.0)+0.5) ** 2  # MDS Enter center of the beam here
        #target_phase = target_phase[462:562, 586:686]
        #target_phase = target_phase[362:662, 486:786]
        #plt.imshow(target_phase)
        #plt.show()
        target_phase = slm.pad_border(target_phase, (int(1024*scalepad), int(1272*scalepad)))
        # plt.imshow(target_phase)
        # plt.title("targetphase")
        # plt.show()

        # MDS
        plotshow = False
        wu_1x4_full = mod.wu_algorithm2D(n=n, m=m,phase_tem_compensation=np.exp(1j * np.array(target_phase)),#MDS
                                    M=40, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases,
                                    x_pitch=x_pitch0, plots=plotshow,
                                    input_profile=profile.Profile.input_gaussian(beam_size=(0.55*(1/scalepad), 0.55*(1/scalepad)),
                                                                                 size=(int(1024*scalepad), int(1272*scalepad))), phase_memory=True,
                                    tem01=tem01_0,res_factor=scalepad, uni_spacing=uni_spacing,
                                    xarblist0=xarblist0, yarblist0=yarblist0, anglearblist0=anglearblist0,double_amps_in=double_amps_in)
    #wu_1x4_full=np.zeros((int(1024*scalepad), int(1272*scalepad)))
    #wu_1x4_full=phase_rotate(wu_1x4_full,1/180*np.pi)
    #wu_1x4_full[25:,25:]=wu_1x4_full[:-25,:-25]
    np.save(r'curvature_plots/curvature.npy', global_variables.curvature)
    np.save(r'curvature_plots/final_eff_curvature.npy',global_variables.final_eff_curvature)
    np.save(r'curvature_plots/final_ion_fidelity_curvature.npy', global_variables.final_ion_fidelity_curvature)

    if plotshow:
        plt.imshow(wu_1x4_full)
        plt.title('wu_1x4_full')
        plt.show()
    print("phase done")
    if scalepad==1:
        wu_1x4=wu_1x4_full
        mod.add_phase(wu_1x4_full)
    else:
        wu_1x4=wu_1x4_full[int(1024*scalepad/2)-512:int(1024*scalepad/2)+512,int(1272*scalepad/2)-636:int(1272*scalepad/2)+636]
        mod.add_phase(wu_1x4)
    phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\phaseTest.txt'
    np.savetxt(phase_path, wu_1x4[:1024,:1024])  #REMOVE THE CLIPPING
    wu_1x4_phase=np.mod(wu_1x4+np.pi,2*np.pi)*255/(2*np.pi)
    total_phase = np.mod(mod.phase + np.pi, 2 * np.pi) * 255 / (2 * np.pi)
    phase_path_p = r'Z:\Lab Rice\Experimental Projects\SLM\sample images\wu1x4_phase.bmp'
    cv2.imwrite(phase_path_p,wu_1x4_phase) #total_phase)#
    phase_path_p= r'C:\Python programs\slm_comp_pytorch\images\wu1x4_phase.bmp'
    cv2.imwrite(phase_path_p,total_phase)#wu_1x4_phase)
    wu_1x4_intensity=(profile.Profile.input_gaussian(beam_size=(0.4, 0.4),size=(1024, 1272)))*255
    phase_path_i = r'Z:\Lab Rice\Experimental Projects\SLM\sample images\wu1x4_intensity.bmp'
    cv2.imwrite(phase_path_i,wu_1x4_intensity)
    phase_path_i = r'C:\Python programs\slm_comp_pytorch\images\wu1x4_intensity.bmp'
    cv2.imwrite(phase_path_i,wu_1x4_intensity)
    if plotshow:
        plt.imshow(mod.phase)
        plt.title("modphase")
        plt.show()
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wu_1x4_intensity*np.exp(np.multiply(1j,mod.phase))), norm="ortho"))))
        axs[0].set_title("mod fft abs")
        #plt.show()
        axs[1].imshow((1/(2*np.pi))*np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wu_1x4_intensity*np.exp(np.multiply(1j,mod.phase))), norm="ortho"))))
        axs[1].set_title("mod fft phase")
        plt.show()
    fftout=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(wu_1x4_intensity*np.exp(np.multiply(1j,mod.phase))), norm="ortho"))
    flat_max_index = np.argmax(np.abs(fftout))
    rowfftmax, colfftmax = np.unravel_index(flat_max_index, fftout.shape)
    if plotshow:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(np.abs(fftout[rowfftmax,:]))
        axs[0].set_title("absolute")
        axs[1].plot((1/(2*np.pi))*np.angle(fftout[rowfftmax,:]))
        axs[1].set_title("angle")
        plt.show()

    area_abb=720#1022#720
    patch_size_abb=25#15
    num_patch_abb=28#40#48
    matrix_correction = np.zeros((area_abb, area_abb))
    for i in range(num_patch_abb):
        for j in range(num_patch_abb):
            if 0:#correction[i][j]==0:
                print('x')
                for m in range(patch_size_abb):
                    for n in range(patch_size_abb):
                         matrix_correction[15 * i + m][15 * j + n]=-mod.phase[212+15 * i + m][372+15 * j + n]
            else:
                for m in range(patch_size_abb):
                    for n in range(patch_size_abb):
                        matrix_correction[patch_size_abb * i + m][patch_size_abb * j + n] = correction[i][j]*0.0

    np.save('testing_tem01_4_4_u.npy', mod.phase)
    mod.phase = np.load('testing_tem01_4_4_u.npy')


    if True: #Original 06 Feb, choosing only the central part
        mod.phase = mod.phase + slm.pad_border(matrix_correction, (1024, 1272))  #UNCOMMENT FOR CORRECTION
        # phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01.txt'
        # np.savetxt(phase_path,mod.phase)
        area_size=(720,720)#(720,720)(1022,1022)#
        mod.phase = mod.phase[512 - int(area_size[0] / 2):512 + int(area_size[0] / 2),
                    636 - int(area_size[1] / 2):636 + int(area_size[1] / 2)]
        mod.phase = slm.pad_border(mod.phase, (1024, 1272))
        if True:
            for m in range(int(area_size[0])):
                for n in range(int(area_size[1])):
                    if np.sqrt((m - int(area_size[0] / 2)) ** 2 + (n - int(area_size[1] / 2)) ** 2) > int(area_size[0] / 2):
                        mod.phase[512 - int(area_size[0] / 2) + m][636 - int(area_size[1] / 2) + n] = 0


    # mod.phase[512:1024, :] = mod.phase[512:1024, :] +wu_1x4[512:1024, :]
    # mod.phase[0:512, :] =mod.phase[0:512, :]+mod.gradient(mod.beams[0][1], angle=-0.04* np.pi / 180, axis=0, wavelength=wavelength)[0:512, :]+\
    # mod.gradient(mod.beams[0][1], angle=0.01 * np.pi / 180, axis=1, wavelength=wavelength)[0:512, :]
    # mod.phase[0:512, :] = mod.phase[0:512, :] + mod.gradient(mod.beams[0][1], angle=-0.5 * np.pi / 180, axis=0,
    #                                                          wavelength=wavelength)[0:512, :] + \
    #                       mod.gradient(mod.beams[0][1], angle=0.01 * np.pi / 180, axis=1, wavelength=wavelength)[0:512,
    #                       :]

    # mod.phase=mod.phase[212:812,336:936]
    # phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01.txt'
    # mod.phase = slm.pad_border(mod.phase, (1024,1272))
    # np.savetxt(phase_path, mod.phase)

    mod.phaseToBMP(mod.phase, name='phase_mask0_0', color=False, correction=False, wavelength=411)
    mod.phase = mod.phase + 120 / 180 * np.pi
    mod.phaseToBMP(mod.phase, name='phase_mask0_120', color=False, correction=False, wavelength=411)
    mod.phase = mod.phase + 120 / 180 * np.pi
    mod.phaseToBMP(mod.phase, name='phase_mask0_240', color=False, correction=False, wavelength=411)
    if plotshow:
        plt.imshow(mod.phase)
        plt.title("is it")
        plt.colorbar()
        plt.show()



    # plt.figure(figsize=(8, 6))
    # plt.imshow(temnm, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title('phase_compensation')
    # plt.show()


    #wu_1x4 = mod.wu_algorithm(n=len(amps), M=30, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases, x_pitch=0.01, plots=True, res_factor=1, input_profile=profile.Profile.input_gaussian(beam_size=(0.4*0.8, 0.4), size=np.array(mod.size)), phase_memory=True)
    # wu_1x4 = mod.wu_algorithm2D(n=n,m=m,
    # M=30, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases,
    # x_pitch=0.004, plots=False, res_factor=1,
    # input_profile=profile.Profile.input_gaussian(beam_size=(0.4, 0.4),
    #      size=np.array(mod.size)), phase_memory=True,tem01=False)

    return mod

def create_phase_padding(wavelength=411, phases = (0.,0.,0.,0.),amps=(1.,1.,1.,1.)):
    size_x_l=1024*1
    size_y_l=1272*1
    # mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
    # target = mod.BMPToAmp(path=r'C:/Users/RiceT/Documents/FlirCamera/pictures/images.bmp', norm=True)
    # #target = cp.array(resize(target, (40,20)))
    #
    # target =cp.array(slm.pad_border(target, (1024,1272)))
    # arb2d = mod.wu_image(N=100, wavelength=411, input_profile=cp.array(profile.Profile.input_gaussian(beam_size=(0.4, 0.4), size=np.array((1024,1272)))), target=target)
    # grady = mod.gradient(mod.beams[0][1], angle=0.12 * np.pi / 180, axis=0, wavelength=wavelength)
    #
    # # temnm = mod.temnm(mod.beams[0][1], n=1, m=0)
    # mod.add_phase(grady)
    # mod.add_phase(arb2d)

    # matrix_correction = np.zeros((400, 400))
    # for i in range(16):
    #     for j in range(16):
    #         if 1:
    #             for m in range(25):
    #                 for n in range(25):
    #                     matrix_correction[25 * i + m][25 * j + n] = correction[i][j]
    #
    # mod.phase = mod.phase + slm.pad_border(matrix_correction, (1024,1272))
    #
    # mod.phaseToBMP(mod.phase, name='test_halloween', color=False, correction=False, wavelength=411)
    # mod.add_phase(-mod.phase)
    mod = ArrayModulator(beams=1, size=((size_x_l,size_y_l)),correction_path='images/%dcorrwithLUT.bmp' % wavelength)



    phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase.txt'
    correction = np.loadtxt(phase_path)
    grady = mod.gradient(mod.beams[0][1], angle=0.00* np.pi / 180, axis=0, wavelength=wavelength) #yangle=0.12
    gradx = mod.gradient(mod.beams[0][1], angle=0.00* np.pi / 180, axis=1, wavelength=wavelength)
    #temnm = mod.temnm(mod.beams[0][1], n=1, m=1)
    # gradx=rotate(g)
    mod.add_phase(grady)
    mod.add_phase(gradx)
    #mod.add_phase(temnm)
    # mod.add_phase(temnm)#/(0.7624614246165774+2.9579199832069927)*np.pi)

    #UNCOMMENT THIS FOR TEM01...?
    #phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01.txt'
    #phase_mask_tem01 = np.loadtxt(phase_path)
    #mod.add_phase(np.array(phase_mask_tem01))

    # plt.figure(figsize=(8, 6))
    # plt.imshow(np.array(phase_mask_tem01), cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title('phase of slm')
    # plt.show()
    n = 10
    m = 1
    amps = tuple(1. for i in range(n * m))
    phases = tuple(0 for i in range(n * m))#(0,0.2,0.8,0.6,0.2,0.8,0.6,0)#(0,0.5,0,0.5)#tuple(0 for i in range(n * m))
    #phases=(0.0,0.2,0.0,0.0)
    amps_guess = amps
    # MDS #Target curve compensation
    a = 4.282766e-06 #-4.90392e-06  # -4.03456842e-06#-3.733022642775838e-06
    b = 2070#2095#2071#1937.23#2070  # 2071
    c = 4.482766e-06  # -3.9478368e-06#-3.893853882563125e-06A
    d = 1672#1628#1573.195#1672  # 1712
    target_phase = np.zeros((size_x_l, size_y_l))
    for i in range(size_x_l):
        for j in range(size_y_l):
            # target_phase[i][j]=c*((i-512)*104+1500-d)**2/2+a*((j-636)*84+2000-b)**2/2
            # target_phase[i][j] = c * ((i - 511) * 53.187*1.13 + 1573.195 - d) ** 2 / 2 + a * (
            #            (j - 635.5) * 41.045*1.13 + 1937.23 - b) ** 2 / 2  # MDS
            target_phase[i][j] = c * ((i - size_x_l/2) * 53.651 + 1672 - d) ** 2 / 2 + a * (
                    (j - size_y_l/2) * 41.4555 + 2070 - b) ** 2 / 2  # MDS Enter center of the beam here
    target_phase = target_phase[int(size_x_l/2)-50:int(size_x_l/2)+50, int(size_y_l/2)-50:int(size_y_l/2)+50]
    target_phase = slm.pad_border(target_phase, (size_x_l, size_y_l))
    # MDS
    wu_1x4 = mod.wu_algorithm2D(n=n, m=m,phase_tem_compensation=np.exp(1j * np.array(target_phase)),#MDS
                                M=40, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases,
                                x_pitch=0.0085, plots=True,
                                input_profile=profile.Profile.input_gaussian(beam_size=(0.6, 0.6),
                                                                             size=(size_x_l,size_y_l)), phase_memory=True,
                                res_factor=1,
                                tem01=False)
    # wu_1x4=phase_rotate(wu_1x4,6.5/180*np.pi)
    plt.show()
    mod.add_phase(wu_1x4)
    phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\phaseTest.txt'
    np.savetxt(phase_path, wu_1x4)
    wu_1x4_phase=np.mod(wu_1x4+np.pi,2*np.pi)*255/(2*np.pi)
    phase_path_p = r'Z:\Lab Rice\Experimental Projects\SLM\sample images\wu1x4_phase.bmp'
    cv2.imwrite(phase_path_p,wu_1x4_phase)
    wu_1x4_intensity=(profile.Profile.input_gaussian(beam_size=(0.6, 0.6),size=(size_x_l, size_y_l)))*255
    phase_path_i = r'Z:\Lab Rice\Experimental Projects\SLM\sample images\wu1x4_intensity.bmp'
    cv2.imwrite(phase_path_i,wu_1x4_intensity)


    matrix_correction = np.zeros((720, 720))
    for i in range(48):
        for j in range(48):
            if 0:#correction[i][j]==0:
                print('x')
                for m in range(15):
                    for n in range(15):
                         matrix_correction[15 * i + m][15 * j + n]=-mod.phase[212+15 * i + m][372+15 * j + n]
            else:
                for m in range(15):
                    for n in range(15):
                        matrix_correction[15 * i + m][15 * j + n] = correction[i][j]


    mod.phase = mod.phase + slm.pad_border(matrix_correction, (1024, 1272))
    # phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01.txt'
    # np.savetxt(phase_path,mod.phase)
    area_size=(720,720)
    mod.phase = mod.phase[int(size_x_l/2) - int(area_size[0] / 2):int(size_x_l/2) + int(area_size[0] / 2),
                int(size_y_l/2) - int(area_size[1] / 2):int(size_y_l/2) + int(area_size[1] / 2)]
    mod.phase = slm.pad_border(mod.phase, (size_x_l, size_y_l))
    for m in range(int(area_size[0])):
        for n in range(int(area_size[1])):
            if np.sqrt((m - int(area_size[0] / 2)) ** 2 + (n - int(area_size[1] / 2)) ** 2) > int(area_size[0] / 2):
                mod.phase[int(size_x_l/2) - int(area_size[0] / 2) + m][int(size_y_l/2) - int(area_size[1] / 2) + n] = 0


    # mod.phase[512:1024, :] = mod.phase[512:1024, :] +wu_1x4[512:1024, :]
    # mod.phase[0:512, :] =mod.phase[0:512, :]+mod.gradient(mod.beams[0][1], angle=-0.04* np.pi / 180, axis=0, wavelength=wavelength)[0:512, :]+\
    # mod.gradient(mod.beams[0][1], angle=0.01 * np.pi / 180, axis=1, wavelength=wavelength)[0:512, :]
    # mod.phase[0:512, :] = mod.phase[0:512, :] + mod.gradient(mod.beams[0][1], angle=-0.5 * np.pi / 180, axis=0,
    #                                                          wavelength=wavelength)[0:512, :] + \
    #                       mod.gradient(mod.beams[0][1], angle=0.01 * np.pi / 180, axis=1, wavelength=wavelength)[0:512,
    #                       :]

    # mod.phase=mod.phase[212:812,336:936]
    # phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01.txt'
    # mod.phase = slm.pad_border(mod.phase, (1024,1272))
    # np.savetxt(phase_path, mod.phase)

    mod.phaseToBMP(mod.phase, name='phase_mask0_0', color=False, correction=False, wavelength=411)
    mod.phase = mod.phase + 120 / 180 * np.pi
    mod.phaseToBMP(mod.phase, name='phase_mask0_120', color=False, correction=False, wavelength=411)
    mod.phase = mod.phase + 120 / 180 * np.pi
    mod.phaseToBMP(mod.phase, name='phase_mask0_240', color=False, correction=False, wavelength=411)
    plt.imshow(mod.phase)
    plt.title("is it")
    plt.colorbar()
    plt.show()

    return mod

def create_phase_xie():
    try:

        wavelength = 411
        mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
        grady = mod.gradient(mod.beams[0][1], angle=0.12 * np.pi / 180, axis=0, wavelength=wavelength)
        gradx = mod.gradient(mod.beams[0][1], angle=0.00 * np.pi / 180, axis=1, wavelength=wavelength)

        phase = gradx + grady
        centre = [int(phase.shape[0]), int(phase.shape[1])]

        block_size = [25, 25]
        centre = [int((phase.shape[0]) / 2), int((phase.shape[1]) / 2)]
        phase_slice = slm.pad_border(phase[centre[0] - block_size[0]:centre[0] + block_size[0] + 1,
                                     centre[1] - block_size[1]:centre[1] + block_size[1] + 1], (1024,1272))

        target_shape = (1024,1272)
        target_center = np.array(target_shape) / 2

    except Exception as e:
        print('error', e)

    shift = [-175 ,-25]
                # shift=[-int(self.text_frame_set_1.text())* 25 + 200,-int(self.text_frame_set_2.text())* 25 + 200]
    output = np.zeros(shape=target_shape)
    phase_center = [shift[0], shift[1]]

    phase_shift_0 = phase[centre[0] - block_size[0] + shift[0]:centre[0] + block_size[0] + 1 + shift[0],
                                centre[1] - block_size[1] + shift[1]:centre[1] + block_size[1] + 1 + shift[1]]

                # phase_shift_0=phase_shift_0+correction[number_x[i]][number_y[j]]

    phi = 0
    phase_shift = phase_shift_0 + phi
    output[:phase_shift.shape[0], :phase_shift.shape[1]] = phase_shift

    output = np.roll(output, np.array(target_center - phase_center - block_size, dtype=np.int64),
                                 axis=(0, 1))

    phase_rota = phase_slice + output
    mod.add_phase(phase_rota)
    mod.phaseToBMP(mod.phase, name=f'correction_{shift[0]}_{shift[1]}_shift_{phi}', color=False, correction=False, wavelength=411)
def phase_rotate(phase,angle):

    centre=[int(len(phase)/2),int(len(phase[0])/2)]
    print('centre',centre)
    new_phase=np.zeros(np.shape(phase))# (1024,1272))
    for i in range(new_phase.shape[0]):
        for j in range(new_phase.shape[1]):
            l=np.sqrt((i-centre[0])**2+(j-centre[1])**2)
            if i!=centre[0]:
                if i>centre[0]:
                    angle0=np.arctan((j-centre[1])/(i-centre[0]))+angle
                if i<centre[0]:
                    angle0=np.arctan((j-centre[1])/(i-centre[0]))+angle+np.pi

            else:
                if  y>= centre[1]:
                    angle0=np.pi/2+angle
                else:
                    angle0 = -np.pi / 2 + angle
            x=centre[0]+l*np.cos(angle0)
            y=centre[1]+l*np.sin(angle0)
            # if y<centre[1] and x<centre[0]:
            #     print('i=',i,'j=',j,'x=',x,'y=',y)
            if x>0 and y>0 and x<new_phase.shape[0]-1 and y <new_phase.shape[1]-1:
                new_phase[i][j]=new_phase[i][j]+\
                    phase[math.floor(x)][math.floor(y)]*(x-math.floor(x))*(y-math.floor(y))
                new_phase[i][j] = new_phase[i][j] + \
                     phase[math.floor(x)][math.ceil(y)] * (x - math.floor(x)) * (math.ceil(y)-y)
                new_phase[i][j] = new_phase[i][j] + \
                                  phase[math.ceil(x)][math.ceil(y)] *  (math.ceil(x)-x) *  (math.ceil(y)-y)
                new_phase[i][j] = new_phase[i][j] + \
                                  phase[math.ceil(x)][math.floor(y)] * (math.ceil(x)-x) *  (y-math.floor(y))
            # else:
            #
            #     print('i=', i, 'j=', j, 'x=', x, 'y=', y)
    return new_phase
# Simulate a phase pattern from create_phase() using some resolution factor res_factor
def test_phase(mod=None, input_profile=None, size= (1024,1272), res_factor=2, wavelength=411):
    if input_profile is None:
        input_profile = cp.array(profile.Profile.input_gaussian(beam_size=np.array((0.4, 0.4)) / res_factor, size=np.array(np.array(size)*res_factor, dtype=np.uint)))
    input_profile /= cp.sqrt(cp.sum(cp.abs(input_profile)**2))

    if mod is None:
        mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
        zernike = mod.zernike_sum(mod.beams[0][1] / 8)
        zernike = slm.pad_border(zernike, mod.phase.shape)
        mod.add_phase(zernike)

    slm_field = input_profile * cp.exp(1j * cp.array(slm.pad_border(mod.phase, input_profile.shape)))
    image_field = mod.propagate(slm_field)

    image_mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
    image_mod.set_phase(image_field.get())

    image_mod.fieldtoBMP(slm_field.get(), name='arraymodtest_slm', wavelength=wavelength, color=True, show=False)
    image_mod.fieldtoBMP(image_field.get(), name='arraymodtest_image', wavelength=wavelength, color=True, show=False)

    plot_gradient(image_field, coord=True)
    plt.show()


# Load a phase pattern into an ArrayModulator instance
def load_mod(phase_path=None, corr_path='images/%dcorrwithLUT.bmp', wavelength=411, correction=True):
    if phase_path is None:
        phase_path = tkinter.filedialog.askopenfilename(title='Select Phase Pattern')
    mod = ArrayModulator(beams=1, correction_path=corr_path % wavelength)
    mod.set_phase(mod.BMPToPhase(phase_path, wavelength=wavelength))


    # if without using the slm gui, uncomment the next two lines and use load_mod() instead of create_phase()
    # after generating the undeflected phase pattern
    # mod.add_phase(np.pi/2)
    # mod.phaseToBMP(mod.phase, name='tem10 uniform phase array 0.006x 0.04 deflected shifted', color=False, correction=True, wavelength=411)

    if correction:
        mod.add_phase(-1 * mod.correction)
    return mod


def choose_pattern(patternIn,scalepad):
    if patternIn =="tem00_1":
        n=1; m=1;
        amps=tuple(1. for i in range(n * m))
        phases=tuple(0. for i in range(n * m))
        uni_spacing=True
        x_pitch0=0.0063
        tem01_0=False
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn =="tem01_1":
        n=1; m=1;
        amps=tuple(1. for i in range(n * m))
        phases=tuple(0. for i in range(n * m))
        uni_spacing=True
        x_pitch0=0.0063
        tem01_0=True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn =="tem00_4_1_u":
        n=4; m=1;
        amps=tuple(1. for i in range(n * m))
        phases=tuple(0. for i in range(n * m))
        uni_spacing=True
        x_pitch0=0.0063
        tem01_0=False
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))


    if patternIn =="tem00_4_1_a":
        n=4; m=1;
        amps=tuple(1. for i in range(n * m))
        phases=tuple((0.0,0.5,)*2)
        uni_spacing=True
        x_pitch0=0.0063
        tem01_0=False
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn =="tem00_4_1_arb":
        n=4; m=1;
        amps=tuple(1. for i in range(n * m))
        phases=tuple((0.0,0.0,)*2)
        uni_spacing=False
        x_pitch0=0.0063
        tem01_0=True
        xarblist = np.multiply(0.0000, np.array([-0.0, -0.77742, -3.70982, -2.61698]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.77742, -3.70982, -2.61698]))
        anglearblist = np.zeros(np.size(xarblist))+90.0

    if patternIn == "tem00_5_1_u":
        n = 5;
        m = 1;
        amps = tuple(1. for i in range(n * m))
        phases = tuple(0. for i in range(n * m))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem01_5_1_a":
        n = 5;
        m = 1;
        amps = tuple(1. for i in range(n * m))
        phases = tuple((0.0, 0.5,) * 2 + (0.0,))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem01_4_1_u":
        n = 4;
        m = 1;
        amps = tuple(1. for i in range(n * m))
        phases = tuple(0. for i in range(n * m))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))


    if patternIn == "tem00_11_1_u":
        n = 11;
        m = 1;
        amps = tuple(1. for i in range(n * m))
        phases = tuple(0. for i in range(n * m))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem01_11_1_u":
        n = 11;
        m = 1;
        amps = tuple(1. for i in range(n * m))
        phases = tuple(0. for i in range(n * m))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem01_11_1_a":
        n = 11;
        m = 1;
        amps = tuple(1. for i in range(n * m))
        phases = tuple((0.0, 0.5,) * 5 + (0.0,))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem00_4_4_u":
        n = 4;
        m = 4;
        amps = tuple(1. for i in range(n * m))
        phases = tuple(0. for i in range(n * m))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = False
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem00_4_4_a":
        n = 4;
        m = 4;
        amps = tuple(1. for i in range(n * m))
        phases = tuple((0.0, 0.5,) * 8)
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = False
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem01_4_4_u":
        n = 4;
        m = 4;
        amps = tuple(1. for i in range(n * m))
        phases = tuple(0. for i in range(n * m))
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem01_4_4_a":
        n = 4;
        m = 4;
        amps = tuple(1. for i in range(n * m))
        phases = tuple((0.0, 0.5,) * 8)
        uni_spacing = True
        x_pitch0 = 0.0063
        tem01_0 = True
        xarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        yarblist = np.multiply(0.0100, np.array([-0.0, -0.47742, -3.70982, -2.61698, -2.95564]))
        anglearblist = np.zeros(np.size(xarblist))

    if patternIn == "tem01_47_1_u":
        n = 47;
        m = 1;
        amps=list([np.float64(0.04586915924100187), np.float64(0.0737158218486801), np.float64(1.7255334088047454e-10), np.float64(0.0737158216568236), np.float64(0.14632802444498844), np.float64(0.14632802425786048), np.float64(0.07047564049967088), np.float64(0.1397719421223463), np.float64(0.13977194218107503), np.float64(0.14632802458290414), np.float64(0.07047564062225226), np.float64(0.103546642494763), np.float64(0.18046184232764007), np.float64(0.08095818482328287), np.float64(0.1804618424515428), np.float64(0.045869159295217454), np.float64(0.15047272603228182), np.float64(0.07047564043911027), np.float64(0.10354664249300768), np.float64(0.13977194212480343), np.float64(0.10354664243863597), np.float64(0.22106155008059786), np.float64(0.2210615499737462), np.float64(0.18046184223521522), np.float64(0.14632802447973453), np.float64(0.08095818476931864), np.float64(0.10820207063869679), np.float64(0.10820207064264364), np.float64(0.07371582188085807), np.float64(0.22106155006897915), np.float64(0.07371582175215913), np.float64(0.1397719422967504), np.float64(0.1440219418384983), np.float64(0.14402194185591574), np.float64(0.2227630594106869), np.float64(0.22276305958322093), np.float64(0.07047564027944482), np.float64(0.15810114977095455), np.float64(0.15810114993268837), np.float64(0.15047272594532118), np.float64(0.15810114986479626), np.float64(0.10354664247580193), np.float64(0.22106154998023197), np.float64(0.15810114991473614), np.float64(0.18046184244441152), np.float64(0.22276305947986294), np.float64(0.22276305954786257)])
        amps=amps/np.max(amps)
        phases = tuple(1. for i in range(n * m))
        uni_spacing = False
        x_pitch0 = 0.0063
        tem01_0 = True
        yarblist = np.multiply(0.001875, np.array(
            [np.float64(-3.9153676373864976), np.float64(1.5958924653913373), np.float64(-0.24789189083258933),
             np.float64(1.595892454424498), np.float64(3.51795397976523), np.float64(-4.013737758115343),
             np.float64(5.433885056698667), np.float64(-7.892236416750612), np.float64(7.396452641781203),
             np.float64(3.5179539685022996), np.float64(-5.929668845960811), np.float64(9.858898722101284),
             np.float64(-11.662620477684406), np.float64(7.604459696761918), np.float64(11.16683669238147),
             np.float64(3.4195838537920924), np.float64(-0.24789188859652656), np.float64(5.433885069420214),
             np.float64(-10.354682503336655), np.float64(7.396452629423975), np.float64(9.858898720238939),
             np.float64(-2.4206606397110795), np.float64(1.9248768489256154), np.float64(11.166836707272271),
             np.float64(-4.013737754913859), np.float64(-8.10024348265401), np.float64(11.77623483845139),
             np.float64(-12.272018621227247), np.float64(-2.091676238293306), np.float64(1.9248768589685479),
             np.float64(-2.091676245975052), np.float64(-7.8922364189119945), np.float64(15.793584660572392),
             np.float64(-16.289368441609966), np.float64(6.418611290583689), np.float64(6.4186112993839775),
             np.float64(-5.9296688497419225), np.float64(14.079456475296887), np.float64(14.079456473275863),
             np.float64(-0.24789189579311033), np.float64(-14.57524025456667), np.float64(-10.354682507001325),
             np.float64(-2.420660631514017), np.float64(-14.575240261414221), np.float64(-11.662620484105023),
             np.float64(-6.914395073486592), np.float64(-6.914395074523181)]))

        xarblist = np.multiply(0.001875 * 1272*scalepad / (1024*scalepad), np.array(
            [np.float64(-0.14402684160131637), np.float64(3.2612895167361255), np.float64(-0.14402684360513102),
             np.float64(-3.5493432086919947), np.float64(6.648329080209887), np.float64(-6.936382769095556),
             np.float64(-3.2167285170025828), np.float64(-6.427674579803346), np.float64(6.13962088627167),
             np.float64(-6.9363827787988335), np.float64(2.92867482361023), np.float64(-3.4794570763251618),
             np.float64(-7.900517078211293), np.float64(-0.14402685630384013), np.float64(-7.90051707820557),
             np.float64(-0.1440268471400142), np.float64(6.857957322617093), np.float64(2.9286748167110237),
             np.float64(-3.4794570715019115), np.float64(-6.427674580670901), np.float64(3.1914033701945352),
             np.float64(-10.874976276753143), np.float64(-10.874976280995716), np.float64(7.612463373661678),
             np.float64(6.648329082223555), np.float64(-0.14402684149046635), np.float64(-0.14402685135616655),
             np.float64(-0.1440268484059051), np.float64(3.2612895170856055), np.float64(10.586922582716923),
             np.float64(-3.5493432031882444), np.float64(6.139620889764666), np.float64(-0.14402685297854273),
             np.float64(-0.14402685441374838), np.float64(-10.29534867565972), np.float64(10.007294978375237),
             np.float64(-3.2167285072119336), np.float64(4.110977719807727), np.float64(-4.399031427141683),
             np.float64(-7.146011014941164), np.float64(-4.399031426941241), np.float64(3.191403379424469),
             np.float64(10.586922585459911), np.float64(4.110977723488051), np.float64(7.61246337791134),
             np.float64(10.007294980894486), np.float64(-10.295348673951082)]))

        anglearblist = np.zeros(np.size(xarblist))
        anglearblist = 90.0 + np.array(
            [np.float64(-9.926423015743774e-08), np.float64(-96.57596690631269), np.float64(-51.219935504870456),
             np.float64(96.57596691240425), np.float64(-106.87386975001493), np.float64(73.12613025480017),
             np.float64(120.83287506611262), np.float64(62.292384093209634), np.float64(-117.70761592154263),
             np.float64(106.87386967786495), np.float64(-59.1671249054077), np.float64(141.5075167118805),
             np.float64(60.58666674512219), np.float64(179.99999991737988), np.float64(119.41333328276657),
             np.float64(-179.9999999614754), np.float64(-90.00000002999623), np.float64(-120.83287522494996),
             np.float64(38.492483255639144), np.float64(117.70761594769567), np.float64(-141.50751686104718),
             np.float64(87.46163284209767), np.float64(92.53836713809196), np.float64(-119.41333332292612),
             np.float64(-73.12613029305714), np.float64(-6.447747537362172e-08), np.float64(179.9999999162039),
             np.float64(8.47137792365958e-09), np.float64(-83.42403312567035), np.float64(-92.53836716528025),
             np.float64(83.4240331320487), np.float64(-62.292384073423236), np.float64(179.9999999049066),
             np.float64(1.0753090160404035e-07), np.float64(101.58840969182394), np.float64(-101.58840970353917),
             np.float64(59.16712467795351), np.float64(-147.379221652306), np.float64(147.3792216222973),
             np.float64(89.99999996590905), np.float64(32.62077833488603), np.float64(-38.49248320555845),
             np.float64(-87.46163286328525), np.float64(-32.620778365282355), np.float64(-60.586666716420815),
             np.float64(-78.41159029358), np.float64(78.41159029393553)])

    if patternIn == "tem00_47_1_u":
        n = 47;
        m = 1;
        #amps=list([np.float64(0.04586915924100187), np.float64(0.0737158218486801), np.float64(1.7255334088047454e-10), np.float64(0.0737158216568236), np.float64(0.14632802444498844), np.float64(0.14632802425786048), np.float64(0.07047564049967088), np.float64(0.1397719421223463), np.float64(0.13977194218107503), np.float64(0.14632802458290414), np.float64(0.07047564062225226), np.float64(0.103546642494763), np.float64(0.18046184232764007), np.float64(0.08095818482328287), np.float64(0.1804618424515428), np.float64(0.045869159295217454), np.float64(0.15047272603228182), np.float64(0.07047564043911027), np.float64(0.10354664249300768), np.float64(0.13977194212480343), np.float64(0.10354664243863597), np.float64(0.22106155008059786), np.float64(0.2210615499737462), np.float64(0.18046184223521522), np.float64(0.14632802447973453), np.float64(0.08095818476931864), np.float64(0.10820207063869679), np.float64(0.10820207064264364), np.float64(0.07371582188085807), np.float64(0.22106155006897915), np.float64(0.07371582175215913), np.float64(0.1397719422967504), np.float64(0.1440219418384983), np.float64(0.14402194185591574), np.float64(0.2227630594106869), np.float64(0.22276305958322093), np.float64(0.07047564027944482), np.float64(0.15810114977095455), np.float64(0.15810114993268837), np.float64(0.15047272594532118), np.float64(0.15810114986479626), np.float64(0.10354664247580193), np.float64(0.22106154998023197), np.float64(0.15810114991473614), np.float64(0.18046184244441152), np.float64(0.22276305947986294), np.float64(0.22276305954786257)])

        amps=np.ones(n*m)
        #For lowest frequency mode:
        #amps=[0.43550768, 0.0986392264, -0.662154125, 0.0986391642, 0.0045716072, 0.00457164989, -0.184431014, 0.0325558886, 0.0325558831, 0.00457161484, -0.184431222, 0.00885743664, -0.00249365467, 0.017034352, -0.00249365329, 0.435507493, -0.0311821007, -0.184431089, 0.00885742628, 0.0325558626, 0.00885745146, 0.00148158316, 0.001481586, -0.00249365588, 0.00457164224, 0.0170344252, -0.00304184863, -0.00304185232, 0.0986391744, 0.00148158744, 0.0986391122, 0.0325559091, 0.000183384162, 0.000183384449, -0.00251860317, -0.00251860399, -0.184431147, -0.000374508224, -0.000374507055, -0.0311820799, -0.000374506377, 0.0088574411, 0.00148158461, -0.000374507545, -0.00249365726, -0.0025186075, -0.00251860667]
        #mode k=15
        #amps=[-0.0495873, -0.08770398, -0.23574773, -0.08770431, 0.04674978, 0.04675015, 0.06266671, -0.26662445, -0.26662539, 0.04675009, 0.0626663, 0.12200535, 0.02950778, 0.34848904, 0.02950776, -0.04958726, 0.24313973, 0.06266635, 0.12200521, -0.26662441, 0.1220065, -0.04098828, -0.04098828, 0.02950784, 0.04674984, 0.34848908, -0.28176742, -0.28176711, -0.08770398, -0.04098836, -0.08770431, -0.26662542, 0.0385085, 0.03850846, 0.04273947, 0.04273967, 0.06266665, 0.00119341, 0.00119356, 0.2431388, 0.00119357, 0.12200635, -0.04098836, 0.00119341, 0.02950786, 0.04273967, 0.04273947]
        amps=np.array(amps)
        phases = tuple(1. for i in range(n * m))
        #phases= np.where(amps >= 0, 0.0, 0.5)
        amps=(np.abs(amps))
        amps=np.ones(n*m)
        amps=amps/np.max(amps)
        uni_spacing = False
        x_pitch0 = 0.0063
        tem01_0 = False
        yarblist = np.multiply(0.001875* 1272*scalepad / (1024*scalepad), np.array(
            [np.float64(-3.9153676373864976), np.float64(1.5958924653913373), np.float64(-0.24789189083258933),
             np.float64(1.595892454424498), np.float64(3.51795397976523), np.float64(-4.013737758115343),
             np.float64(5.433885056698667), np.float64(-7.892236416750612), np.float64(7.396452641781203),
             np.float64(3.5179539685022996), np.float64(-5.929668845960811), np.float64(9.858898722101284),
             np.float64(-11.662620477684406), np.float64(7.604459696761918), np.float64(11.16683669238147),
             np.float64(3.4195838537920924), np.float64(-0.24789188859652656), np.float64(5.433885069420214),
             np.float64(-10.354682503336655), np.float64(7.396452629423975), np.float64(9.858898720238939),
             np.float64(-2.4206606397110795), np.float64(1.9248768489256154), np.float64(11.166836707272271),
             np.float64(-4.013737754913859), np.float64(-8.10024348265401), np.float64(11.77623483845139),
             np.float64(-12.272018621227247), np.float64(-2.091676238293306), np.float64(1.9248768589685479),
             np.float64(-2.091676245975052), np.float64(-7.8922364189119945), np.float64(15.793584660572392),
             np.float64(-16.289368441609966), np.float64(6.418611290583689), np.float64(6.4186112993839775),
             np.float64(-5.9296688497419225), np.float64(14.079456475296887), np.float64(14.079456473275863),
             np.float64(-0.24789189579311033), np.float64(-14.57524025456667), np.float64(-10.354682507001325),
             np.float64(-2.420660631514017), np.float64(-14.575240261414221), np.float64(-11.662620484105023),
             np.float64(-6.914395073486592), np.float64(-6.914395074523181)]))

        xarblist = np.multiply(0.001875 * 1272*scalepad / (1024*scalepad)
                               ,np.array(
            [np.float64(-0.14402684160131637), np.float64(3.2612895167361255), np.float64(-0.14402684360513102),
             np.float64(-3.5493432086919947), np.float64(6.648329080209887), np.float64(-6.936382769095556),
             np.float64(-3.2167285170025828), np.float64(-6.427674579803346), np.float64(6.13962088627167),
             np.float64(-6.9363827787988335), np.float64(2.92867482361023), np.float64(-3.4794570763251618),
             np.float64(-7.900517078211293), np.float64(-0.14402685630384013), np.float64(-7.90051707820557),
             np.float64(-0.1440268471400142), np.float64(6.857957322617093), np.float64(2.9286748167110237),
             np.float64(-3.4794570715019115), np.float64(-6.427674580670901), np.float64(3.1914033701945352),
             np.float64(-10.874976276753143), np.float64(-10.874976280995716), np.float64(7.612463373661678),
             np.float64(6.648329082223555), np.float64(-0.14402684149046635), np.float64(-0.14402685135616655),
             np.float64(-0.1440268484059051), np.float64(3.2612895170856055), np.float64(10.586922582716923),
             np.float64(-3.5493432031882444), np.float64(6.139620889764666), np.float64(-0.14402685297854273),
             np.float64(-0.14402685441374838), np.float64(-10.29534867565972), np.float64(10.007294978375237),
             np.float64(-3.2167285072119336), np.float64(4.110977719807727), np.float64(-4.399031427141683),
             np.float64(-7.146011014941164), np.float64(-4.399031426941241), np.float64(3.191403379424469),
             np.float64(10.586922585459911), np.float64(4.110977723488051), np.float64(7.61246337791134),
             np.float64(10.007294980894486), np.float64(-10.295348673951082)]))

        anglearblist = np.zeros(np.size(xarblist))
        anglearblist = 90.0 + np.array(
            [np.float64(-9.926423015743774e-08), np.float64(-96.57596690631269), np.float64(-51.219935504870456),
             np.float64(96.57596691240425), np.float64(-106.87386975001493), np.float64(73.12613025480017),
             np.float64(120.83287506611262), np.float64(62.292384093209634), np.float64(-117.70761592154263),
             np.float64(106.87386967786495), np.float64(-59.1671249054077), np.float64(141.5075167118805),
             np.float64(60.58666674512219), np.float64(179.99999991737988), np.float64(119.41333328276657),
             np.float64(-179.9999999614754), np.float64(-90.00000002999623), np.float64(-120.83287522494996),
             np.float64(38.492483255639144), np.float64(117.70761594769567), np.float64(-141.50751686104718),
             np.float64(87.46163284209767), np.float64(92.53836713809196), np.float64(-119.41333332292612),
             np.float64(-73.12613029305714), np.float64(-6.447747537362172e-08), np.float64(179.9999999162039),
             np.float64(8.47137792365958e-09), np.float64(-83.42403312567035), np.float64(-92.53836716528025),
             np.float64(83.4240331320487), np.float64(-62.292384073423236), np.float64(179.9999999049066),
             np.float64(1.0753090160404035e-07), np.float64(101.58840969182394), np.float64(-101.58840970353917),
             np.float64(59.16712467795351), np.float64(-147.379221652306), np.float64(147.3792216222973),
             np.float64(89.99999996590905), np.float64(32.62077833488603), np.float64(-38.49248320555845),
             np.float64(-87.46163286328525), np.float64(-32.620778365282355), np.float64(-60.586666716420815),
             np.float64(-78.41159029358), np.float64(78.41159029393553)])


    if patternIn == "tem01_11_1_u_h":
        n = 11;
        m = 1;
        amps = tuple(1. for i in range(n * m))
        phases = (0.0,0.5,)*5+(0.0,)
        uni_spacing = False
        x_pitch0 = 0.0063
        tem01_0 = True
        yarblist = np.multiply(0.001875*2.5, np.array([-8.35757,-6.292246,-4.5597,-2.97758512,-1.47280,0,1.47280,2.97758512,4.5597,6.292246,8.35757]))
        xarblist = np.multiply(0.0000, np.zeros(np.size(yarblist)))
        anglearblist = np.zeros(np.size(xarblist))+90.0

    return n,m,amps,phases,uni_spacing,x_pitch0,tem01_0,xarblist,yarblist,anglearblist






if __name__ == '__main__':
    if 1:
        # phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\compensation_phase_list.txt'
        # phase_information =[-2.213106427780366,-0.5808628956448176,-0.7924700487307755,-3.0426621474672846]
        # phase_information = -np.array(phase_information) / 2 / np.pi / 10
        # np.savetxt(phase_path, np.array(phase_information))
        #
        # paths = [r"C:/Users/RiceT/Documents/FlirCamera/pictures/image_intensity/image_d06_h13_m20_s34.bmp",
        #          r"C:/Users/RiceT/Documents/FlirCamera/pictures/average_image1.bmp",
        #          r"C:/Users/RiceT/Documents/FlirCamera/pictures/average_image2.bmp",
        #          r"C:/Users/RiceT/Documents/FlirCamera/pictures/average_image3.bmp"]
        create_phase()
    if 5<1:
        # phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\compensation_phase_list.txt'
        # phase_information =[-2.213106427780366,-0.5808628956448176,-0.7924700487307755,-3.0426621474672846]
        # phase_information = -np.array(phase_information) / 2 / np.pi / 10
        # np.savetxt(phase_path, np.array(phase_information))

        # paths = [r"C:/Users/RiceT/Documents/FlirCamera/pictures/image_intensity/image_d06_h13_m20_s34.bmp",
        #          r"C:/Users/RiceT/Documents/FlirCamera/pictures/average_image1.bmp",
        #          r"C:/Users/RiceT/Documents/FlirCamera/pictures/average_image2.bmp",
        #          r"C:/Users/RiceT/Documents/FlirCamera/pictures/average_image3.bmp"]
        # create_phase()

        a = -3.733022642775838e-06
        b = 1776
        c = -3.893853882563125e-06
        d = 1790
        wavelength=411
        mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
        target_phase=np.zeros((1024,1272))
        for i in range(1024):
            for j in range(1272):
                #target_phase[i][j]=c*((i-512)*104+1500-d)**2/2+a*((j-636)*84+2000-b)**2/2
                target_phase[i][j]=c*((i-512)*39+1500-d)**2/2+a*((j-636)*42+2000-b)**2/2 #MDS
        target_phase=target_phase[462:562,586:686]
        target_phase=slm.pad_border(target_phase, (1024, 1272))



        mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)

        grady = mod.gradient(mod.beams[0][1], angle=0.12 * np.pi / 180, axis=0, wavelength=wavelength)
        gradx = mod.gradient(mod.beams[0][1], angle=0.00 * np.pi / 180, axis=1, wavelength=wavelength)
        mod.add_phase(grady)
        mod.add_phase(gradx)


        n = 4
        m = 1
        amps = tuple(1. for i in range(n * m))
        phases = tuple(
            0 for i in range(n * m))  # (0,0.2,0.8,0.6,0.2,0.8,0.6,0)#(0,0.5,0,0.5)#tuple(0 for i in range(n * m))
        amps_guess = amps

        #np.exp(1j * np.array(target_phase))

        wu_1x4 = mod.wu_algorithm2D(n=n, m=m,phase_tem_compensation=np.exp(1j * np.array(target_phase)),
                                    M=20, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases,
                                    x_pitch=0.0085, plots=False,
                                    input_profile=profile.Profile.input_gaussian(beam_size=(0.4, 0.4),
                                                                                 size=(1024, 1272)), phase_memory=True,
                                    tem01=False)
        # wu_1x4=phase_rotate(wu_1x4,6.5/180*np.pi)
        mod.add_phase(wu_1x4)

        # mod.phase = mod.phase + slm.pad_border(matrix_correction, (1024, 1272))
        mod.phaseToBMP(mod.phase, name='phase_mask0_0', color=False, correction=False, wavelength=411)
        mod.phase = mod.phase + 120 / 180 * np.pi
        mod.phaseToBMP(mod.phase, name='phase_mask0_120', color=False, correction=False, wavelength=411)
        mod.phase = mod.phase + 120 / 180 * np.pi
        mod.phaseToBMP(mod.phase, name='phase_mask0_240', color=False, correction=False, wavelength=411)


