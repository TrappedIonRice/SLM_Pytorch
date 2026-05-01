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

import runsettings

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


    def wu_algorithm2D(self,  amps_tem_compensation=None,phase_tem_compensation=None ,N=40, M=20, m=4,n=5, wavelength=411, name='wu_1x4', amps=(1., 1., 1., 1.), amps_guess=(1., 1., 1., 1.), phases=(0, 0, 0, 0), x_pitch=0.004, input_profile=None, plots=True, res_factor=1, phase_memory=False, figs=(None, None, None), tem01=False, uni_spacing=True ,xarblist0=None, yarblist0=None, anglearblist0=None,double_amps_in=None,start_phase=None):
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
                          phase_memory=phase_memory, start_phase=start_phase ,size=size, tem01=tem01, uni_spacing=uni_spacing, xarblist0=xarblist0, yarblist0=yarblist0, anglearblist0=anglearblist0,double_amps_in=double_amps_in)
        phase = outer.iterate_updatedloop_outer(N=2, M=M) #n=30
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

    scalepad=runsettings.scalepad_global
    n =47
    m =1
    amps = tuple(1. for i in range(n * m))#(1.0,1.0,1.0,1.0)#(1.0,0.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0)#
    double_amps_in = tuple([ampsx for ampsx in amps for _ in range(2)])#tuple(1. for i in range(2*n * m))


    #measuredlist=[0.8531352946187093, 0.7965294058557149, 1.0, 0.9056240544937595, 0.8748861876340257, 0.8663608365416605, 0.9077016560699039, 0.86821876752473, 0.8248658156979294, 0.6983283390477941, 0.7952508141701927, 0.8094983854616743, 0.7903482970284168, 0.7738481958912674, 0.7383839722443436, 0.7662365771419362, 0.8158100304795636, 0.8918589353756115, 0.8222871211876964, 0.884762054158084, 0.8601225786773393, 0.8454749030650931]
    measuredlist2=[0.8439610534276695, 0.8626851866420653, 0.9613728815634297, 0.8370505772693067, 0.7976215386243023, 0.7272200261399524, 0.8843721890673902, 0.7652473563704804, 0.735611641461746, 0.7317724408134628, 0.8012249561715696, 0.7629908867703901, 0.8025495380864411, 0.7529227939179908, 0.7386182076182133, 0.7301463655860104, 0.752013749293335, 0.6495402011888561, 0.7528243337722006, 0.8312058753381851, 0.7873976033858706, 0.8973458487709258, 0.7661927633686102, 0.8498648331627829, 0.779744421004564, 0.8013437645578035, 0.742979402524888, 0.8582395512782881, 0.8139379027187258, 0.6992737028065514, 0.801688485269596, 0.7856274091932658, 0.6965259086905338, 0.8321128883454936, 0.7296815934611363, 0.8895954654670905, 0.7774336822075429, 0.7155252605492178, 0.7628928128729398, 0.8409767764320486, 0.7976417687116896, 0.818345119452142, 0.858139873417437, 0.8376485605442431, 0.7401180676302305, 0.7639943190791177, 0.7982869207817094, 0.7587283283759984, 0.8830673423736239, 0.7172435953416376, 0.8377015875391478, 0.7762430587338411, 0.768522637740328, 0.8061859746878641, 0.7640252118927628, 0.7380574894200498, 0.7777565842621581, 0.7588193263098969, 0.8063191867345135, 0.759482504342968, 0.7846042541858476, 0.771394305154472, 0.7102110341778709, 0.8088158261062368, 0.7278822463101617, 0.776426687124368, 0.8073335466572992, 0.8204337677294465, 0.7201278842918146, 0.7554826942550525, 0.6793144260679339, 0.9063384357617273, 0.765336863748648, 0.8392399843839916, 0.9568488370641683, 0.7450934670064467, 0.7733809255562966, 0.9085178448806229, 0.6932162200604267, 0.76232149712717, 0.711310423805241, 0.7267880603686045, 0.8047463431070357, 0.7061712782836904, 0.783711382495797, 0.750361909032542, 0.8081694332017018, 0.8182994964326, 0.6923650812232011, 0.734839549446165, 0.7967240573560702, 0.8154977507268304, 0.7310035723954929, 0.8584152227436842, 0.7400364511106677, 0.7982066352029915, 0.7762824164883824, 0.7968486240276771, 0.8178517370893573, 0.7082998275970521, 0.7808696705812836, 0.7024927753390086, 0.7287571799946306, 0.7386484691848404, 0.8311235423362958, 0.6726372634638207, 0.8145191096032367, 0.6905071755479925, 0.6848079497274668, 0.8279279919153429, 0.7442369244532655, 0.7458940752793893, 0.7422680897998716, 0.8063515173809087, 0.7861232143450657, 0.7789889231700932, 0.6584021067313115, 0.7787443867176093, 0.6986115771227788, 0.6940272622710506, 0.6279715480370148, 1.0, 0.6576487173578844, 0.7821989463729385, 0.7368474207554861, 0.7822788130104631, 0.7649368437271488, 0.7513093912685785, 0.7460723477959191, 0.7025831156042928, 0.7014735769641653, 0.7726481653506901, 0.7693982518191721, 0.7088651934681977, 0.7648572581058583, 0.8473189780255985, 0.7907797934213189, 0.7988951494438092, 0.7933204204157108, 0.8060992307628103, 0.7799575742174109, 0.789795583764972, 0.7964472096231661, 0.8673222733320495, 0.7534371615561765, 0.8362361387604655, 0.6783262142589426, 0.7536409965259158, 0.6869144791315301, 0.9253971853296549, 0.7441980009515781, 0.8254703727303171, 0.7576287690489834, 0.8670919681225449, 0.8794487189459809, 0.7058151005154403, 0.8898121350851149, 0.758124885589459, 0.6806792451823834, 0.8057663443701517, 0.7511736033868067, 0.8063742398151694, 0.793153796913961, 0.8443779398679583, 0.7976287608132413, 0.7755789600480689, 0.9062314922166134, 0.8174579294245384, 0.7541289829308583, 0.7290278480513168, 0.7949886384377413, 0.7572856185196428, 0.7709241096019154, 0.8312691171175512, 0.8025606275096362, 0.8248655195348354, 0.8392191252268959, 0.7628140141638818, 0.7931916314187635, 0.7772359672801299, 0.7899423245630034, 0.7538989010999313, 0.8172527861219311, 0.7606822920087257, 0.7816133188164227, 0.7985252590534052, 0.8184958858804244, 0.792578997496643, 0.7647720756818502, 0.6848843527934926, 0.7599594743602001, 0.7490743439570227, 0.7760391276980807, 0.7889003685488903, 0.7876878927124881, 0.8121305022911635, 0.698283096960008, 0.8313359484901781, 0.7653687554185183, 0.7223620810740264, 0.8792943734798431, 0.8393211761202118, 0.7818317226349316, 0.8144480380898957, 0.793939947000011, 0.6868367137597653, 0.7548384072698431, 0.7306181375587661, 0.7781845756408456, 0.7638881732894014, 0.7648141295780039, 0.8413434120345797, 0.7495226109151425, 0.7765700092287351, 0.8347730490322277, 0.7973920237333181, 0.8171586303888663, 0.8109799011070817, 0.7625128272276072, 0.862117368377655, 0.8705922087358537, 0.8275677513551134, 0.7667924313209972, 0.7776400007909217, 0.7415450204326322, 0.7767376315032826, 0.778040035843423, 0.647438690808642, 0.8059137749511315, 0.7539099977102225, 0.6769877983314755, 0.8059237605757764, 0.7156089771765973, 0.7787934420879995, 0.6940121861912185, 0.8625199950422447, 0.8168857444394598, 0.710787190739411, 0.7839255957729706, 0.7528756989906588, 0.7670883104325101, 0.8241212441604612]

    measuredlist=[0.632504727316183, 0.74221263238833, 0.5472931407458179, 0.720769271636983, 0.7103826608091045, 0.6874895119086634, 0.6597922788670774, 0.7024769966398199, 0.7171996578487929, 0.7509112587714439, 0.8577608799674568, 0.7129537997538127, 0.7477033540436261, 0.7343727346167244, 0.7372792809451528, 0.6466667101362992, 0.6724981442980958, 0.6850164885301279, 0.6576188996624474, 0.590525636503704, 0.6138496926394443, 0.6692234636683427, 0.6734942581806581, 0.7942318080787671, 0.5786732055821474, 0.8191456365989699, 0.7373784866467074, 0.832085111645632, 0.7459095012053728, 0.8726734189313937, 0.6978954965956659, 0.7770853513641811, 0.8432537675833632, 0.7710342181052521, 0.8157105621000523, 0.6757334330890911, 0.6680278236915477, 0.7498299461114295, 0.7694737026977996, 0.6863904419968377, 0.8346654022293726, 0.6075989763210845, 0.6993707284576504, 0.6635328512001768, 0.6447627057220283, 0.7281677081984262, 0.5837411271099842, 0.8159712760913391, 0.6666824203302492, 0.8096972696765636, 0.7768008115651236, 0.823372676023289, 0.7326946800068251, 0.9116306290326215, 0.7651642876137646, 0.8000976791904257, 0.7086894916281453, 0.7121712473945566, 0.7426854710639262, 0.748249861246241, 0.7920276388272526, 0.6586993667663767, 0.7433461745036558, 0.6143475140137798, 0.744205419304656, 0.6849788743809461, 0.7160019449918611, 0.7317027238666645, 0.6302841954095478, 0.7835140701034238, 0.675247997219633, 0.8199529171205944, 0.78594499953211, 0.8319351853045408, 0.6961292774918736, 0.7009837320677007, 0.6851665289555271, 0.7142187570266799, 0.773868298323742, 0.808087643637054, 0.7667309867528952, 0.7468582060282518, 0.7193181448134423, 0.7868628782571436, 0.6852656346034632, 0.6207231514781667, 0.6563039654283187, 0.7862897898487802, 0.7402940364013737, 0.7502918672478149, 0.7162773361258393, 0.8110098420615568, 0.7157570991824932, 0.7326649749630004, 0.7382644072824486, 0.6656003858891105, 0.7420623156941532, 0.7299465024068958, 0.7701283892866504, 0.8028096640729515, 0.7659050355645056, 0.7387809669914372, 0.6992717607376288, 0.7530587955510576, 0.7481006469321604, 0.758881270955608, 0.7653848814366017, 0.6548473341483544, 0.6686849365818643, 0.7238444147813384, 0.6838659781605613, 0.7525695196902161, 0.7170317507895054, 0.7753345312839339, 0.7254222969855986, 0.8620006851560853, 0.6850194034202574, 0.7533235254779244, 0.8370171964539761, 0.6770952513724702, 0.7323089102251403, 0.624930870979584, 0.8085918401027767, 0.8033647585022814, 0.8430789060326486, 0.6883108589548097, 0.7646393100796411, 0.8301790754618856, 0.7499930226257013, 0.6767888219339816, 0.7685635090816075, 0.7255032769286154, 0.7295752367000536, 0.7631255714012278, 0.7073686753885252, 0.795792754427735, 0.7088561174985516, 0.8121641745570493, 0.693324705922316, 0.7211164657420711, 0.7619804916338307, 0.8730541953314872, 0.8413745143048068, 0.7798597111419171, 0.7279721054365978, 0.7514717543756976, 0.7524277816517736, 0.7778964021159943, 0.7738898009489442, 0.8324000968705971, 0.7605994099864103, 0.7395273152564691, 0.68522059639375, 0.7004640926730887, 0.670378309387473, 0.776780715147842, 0.6219314606309502, 0.8243986092232166, 0.6303325549040807, 0.7821819102900314, 0.7181924102328597, 0.7811686762193608, 0.7459229555202183, 0.8589145688716736, 0.8602603763838317, 0.7644246593773374, 0.813393869497995, 0.7932293722026679, 0.802981505952565, 0.6995499059817331, 0.7676601653206244, 0.7445435104764601, 0.7660061685295165, 0.6378001887207059, 0.7637546770533364, 0.7160434088045327, 0.6860636210784584, 0.7332832476846994, 0.6324384179342322, 0.8411079997538832, 0.7044412231496331, 0.8739636354099238, 0.723643199519024, 0.8079766818260546, 0.7871780501331185, 0.7312210499177713, 0.913853413468334, 0.8756973426916782, 0.8728708697121148, 0.7721965144484135, 0.7022348943714777, 0.7541931294428097, 0.818120886016197, 0.8517241890891958, 0.7755565189257413, 0.6625593861687018, 0.7655791154180073, 0.6672534825840446, 0.7167111209297253, 0.8278913053759407, 0.5752027450116548, 0.7897924817001141, 0.7019637640168679, 0.9128484556040659, 0.7868826079953832, 0.7365625176843668, 0.6898088627590386, 0.878924213446765, 0.9029162525474673, 0.8913215035826677, 0.8980777015606233, 0.7935783451381668, 0.7240257419434458, 0.7136932387805015, 0.7888267819963285, 0.7147869348638384, 0.8175036421098715, 0.6779523161064486, 0.6850350663256842, 0.7338839807215004, 0.7849337867230668, 0.893726476873141, 0.6210648014049747, 0.80741837017956, 0.7316639822180279, 0.8352689587495369, 0.8216252152035795, 0.8382216915945585, 0.8017623214151004, 1.0, 0.8650022626065533, 0.8200485177790263, 0.9125833894218249, 0.7198008514750206, 0.758003313069323, 0.734246447041459, 0.7385023933282259, 0.7882844150755184, 0.7886925324025916, 0.6916434081299881, 0.7876019092412673, 0.7599322495935638]

    # measuredlist = (np.array(measuredlist)*np.array(measuredlist2))**0.5#*np.array(measuredlist3))**0.5#*np.array(measuredlist3))**0.5
    if False: #measuredlist feedback
        measuredlist=(measuredlist/np.max(measuredlist))**1#np.sqrt
        #measuredlist=(np.array(measuredlist/np.max(measuredlist))**0.5)*(np.array(measuredlist2/np.max(measuredlist2))**0.5)
        print(len(measuredlist),"measurelist",measuredlist)
        double_amps_in = (1/measuredlist)
        double_amps_in=double_amps_in/np.max(double_amps_in)
        #double_amps_in = double_amps_in[::-1] #flip horixontally only
        double_amps_in_mirrored = []
        #
        for i in range(0, len(double_amps_in), 2*n):
            double_amps_in_mirrored.extend(double_amps_in[i:i + 2*n][::-1])
        double_amps_in=double_amps_in_mirrored
        # amps=double_amps_in_mirrored

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
    # amps=list([np.float64(0.04586915924100187), np.float64(0.0737158218486801), np.float64(1.7255334088047454e-10), np.float64(0.0737158216568236), np.float64(0.14632802444498844), np.float64(0.14632802425786048), np.float64(0.07047564049967088), np.float64(0.1397719421223463), np.float64(0.13977194218107503), np.float64(0.14632802458290414), np.float64(0.07047564062225226), np.float64(0.103546642494763), np.float64(0.18046184232764007), np.float64(0.08095818482328287), np.float64(0.1804618424515428), np.float64(0.045869159295217454), np.float64(0.15047272603228182), np.float64(0.07047564043911027), np.float64(0.10354664249300768), np.float64(0.13977194212480343), np.float64(0.10354664243863597), np.float64(0.22106155008059786), np.float64(0.2210615499737462), np.float64(0.18046184223521522), np.float64(0.14632802447973453), np.float64(0.08095818476931864), np.float64(0.10820207063869679), np.float64(0.10820207064264364), np.float64(0.07371582188085807), np.float64(0.22106155006897915), np.float64(0.07371582175215913), np.float64(0.1397719422967504), np.float64(0.1440219418384983), np.float64(0.14402194185591574), np.float64(0.2227630594106869), np.float64(0.22276305958322093), np.float64(0.07047564027944482), np.float64(0.15810114977095455), np.float64(0.15810114993268837), np.float64(0.15047272594532118), np.float64(0.15810114986479626), np.float64(0.10354664247580193), np.float64(0.22106154998023197), np.float64(0.15810114991473614), np.float64(0.18046184244441152), np.float64(0.22276305947986294), np.float64(0.22276305954786257)])
    # amps=amps/np.max(amps)

    n, m, amps, phases, uni_spacing, x_pitch0, tem01_0,xarblist0, yarblist0, anglearblist0 = choose_pattern("tem00_47_1_u",scalepad)
    #uni_spacing, x_pitch0, tem01_0, xarblist0, yarblist0, anglearblist0 = True,0.0068,True,None,None,None

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
    #curvature_list=np.linspace(0.75,4,1)
    curvature_list = np.linspace(runsettings.curvature, 4, 1)
    for curvature_num in curvature_list:
        print("curvature_num: ",curvature_num)
        global_variables.curvature.append(curvature_num)
        a = 2*(-4.282766e-06)#(0.1027)*(1/24.0)* #-4.90392e-06  # -4.03456842e-06#-3.733022642775838e-06(1/scalepad**3)*
        b = 2070#mds oct2070#2095#2071#1937.23#2070  # 2070 original
        c = 2*(-4.482766e-06)#(0.1027)*(1/24.0)*  # -3.9478368e-06#-3.893853882563125e-06A(1/scalepad**3)*
        d = 1672#mds oct1672 #1628#1573.195#1672  # 1712 #1672 original
        a=1.15*np.pi/(1272*scalepad)*curvature_num*1.02#*0.6#1.2
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

        #To simulate bowman paper grid's LG00 type phase pattern:
        # for i in range(int(1024 * scalepad)):
        #     for j in range(int(1272 * scalepad)):
        #         target_phase[i][j] +=( np.arctan2((j - ((1272 - b_off) * scalepad / 2.0) + 0.5), (
        #                     i - ((1024 - d_off) * scalepad / 2.0) + 0.5)) ) # MDS Enter center of the beam here

        #target_phase = target_phase[462:562, 586:686]
        #target_phase = target_phase[362:662, 486:786]

        target_phase = slm.pad_border(target_phase, (int(1024*scalepad), int(1272*scalepad)))
        # plt.imshow(target_phase)
        # plt.title("targetphase")
        # plt.show()

        # MDS
        plotshow = runsettings.plot_images_global
        wu_1x4_full = mod.wu_algorithm2D(n=n, m=m,phase_tem_compensation=np.exp(1j * np.array(target_phase)),#MDS
                                    M=60, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases,
                                    x_pitch=x_pitch0, plots=plotshow,
                                    input_profile=profile.Profile.input_gaussian(beam_size=(0.45*(1/scalepad), 0.45*(1/scalepad)),
                                                                                 size=(int(1024*scalepad), int(1272*scalepad))), phase_memory=True,
                                    tem01=tem01_0,res_factor=scalepad, uni_spacing=uni_spacing,
                                    xarblist0=xarblist0, yarblist0=yarblist0, anglearblist0=anglearblist0,double_amps_in=double_amps_in)
        # # runsettings.learning_rate=0.02
        # mod2 = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
        # wu_1x4_full = mod2.wu_algorithm2D(n=n, m=m,phase_tem_compensation=np.exp(1j * np.array(target_phase)*0.0),#MDS
        #                             M=60, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases,
        #                             x_pitch=x_pitch0, plots=plotshow,
        #                             input_profile=profile.Profile.input_gaussian(beam_size=(0.55*(1/scalepad), 0.55*(1/scalepad)),
        #                                                                          size=(int(1024*scalepad), int(1272*scalepad))), phase_memory=True,
        #                             tem01=tem01_0,res_factor=scalepad, uni_spacing=uni_spacing,
        #                             xarblist0=xarblist0, yarblist0=yarblist0, anglearblist0=anglearblist0,double_amps_in=double_amps_in,start_phase=wu_1x4_full1 )

    #wu_1x4_full=np.zeros((int(1024*scalepad), int(1272*scalepad)))
    #wu_1x4_full=phase_rotate(wu_1x4_full,1/180*np.pi)
    #wu_1x4_full[25:,25:]=wu_1x4_full[:-25,:-25]
    np.save(r'curvature_plots/curvature.npy', global_variables.curvature)
    np.save(r'curvature_plots/final_eff_curvature.npy',global_variables.final_eff_curvature)
    np.save(r'curvature_plots/final_ion_fidelity_curvature.npy', global_variables.final_ion_fidelity_curvature)

    if plotshow:
        plt.imshow(np.mod(wu_1x4_full+np.pi,2*np.pi))
        plt.colorbar()
        plt.title('wu_1x4_full')
        plt.show()
    print("phase done")
    if scalepad==1:
        wu_1x4=wu_1x4_full
        mod.add_phase(wu_1x4_full)
    else:
        wu_1x4=wu_1x4_full[int(1024*scalepad/2)-512:int(1024*scalepad/2)+512,int(1272*scalepad/2)-636:int(1272*scalepad/2)+636]
        mod.add_phase(wu_1x4)
    if plotshow:
        plt.imshow(np.mod(wu_1x4 + np.pi, 2 * np.pi))
        plt.title('SLM Phase')
        plt.colorbar(label ='Phase (radian)')
        plt.show()
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
                        matrix_correction[patch_size_abb * i + m][patch_size_abb * j + n] = correction[i][j]

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
        #amps=np.array(tuple(1. for i in range(n * m)))
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
        amps=[-0.0495873, -0.08770398, -0.23574773, -0.08770431, 0.04674978, 0.04675015, 0.06266671, -0.26662445, -0.26662539, 0.04675009, 0.0626663, 0.12200535, 0.02950778, 0.34848904, 0.02950776, -0.04958726, 0.24313973, 0.06266635, 0.12200521, -0.26662441, 0.1220065, -0.04098828, -0.04098828, 0.02950784, 0.04674984, 0.34848908, -0.28176742, -0.28176711, -0.08770398, -0.04098836, -0.08770431, -0.26662542, 0.0385085, 0.03850846, 0.04273947, 0.04273967, 0.06266665, 0.00119341, 0.00119356, 0.2431388, 0.00119357, 0.12200635, -0.04098836, 0.00119341, 0.02950786, 0.04273967, 0.04273947]
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
        phases =  tuple(1. for i in range(n * m))#(0.0,0.5,)*5+(0.0,)# tuple(1. for i in range(n * m))#
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


