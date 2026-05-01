import random
import time

import matplotlib.pyplot
import numpy as np
from networkx import efficiency
from sympy import false
from sympy.physics.quantum.density import fidelity

import runsettings
from profile_v1 import Profile, temnm, laserbeamsizefromimage, propagate
from slm_v1 import SLM
from slm_v1 import rescale, pad_border
from matplotlib import pyplot as plt
import laserbeamsize as lbs
from InversePhase import inverse_phase
import scipy.stats
import sys
#from zoomfft2d import ZoomFFT2D
from scipy.signal import ZoomFFT
import copy
import torch
import global_variables
import runsettings

try:
    import cupy as cp
except ImportError:
    cp = np
    print("cupy not installed. Using numpy.")

pi = cp.pi

@cp.fuse()
def make_U_alpha(target1,Ac1,Pc1, mask1, S1,onesbox1, fb1,fbmix1):
    U_alpha1=((target1 * mask1 * S1) - fb1 * Ac1*cp.exp(1j*Pc1) * (1.0 - mask1 * S1))#*onesbox1 + Ac1*cp.exp(1j*Pc1)*(1.0-onesbox1)
    return ((1.0-fbmix1) * U_alpha1 + fbmix1 * Ac1*cp.exp(1j*Pc1)) * onesbox1 + Ac1*cp.exp(1j*Pc1) * (1.0- onesbox1)

def make_U_beta(target1,Ac1,Pc1, mask1, S1,onesbox1, fb1,fbmix1):
    U_beta1=((target1 * mask1 * (1.0-S1)) - fb1 * Ac1*cp.exp(1j*Pc1) * (1.0 - mask1 * (1.0-S1)))#*onesbox1 + Ac1*cp.exp(1j*Pc1)*(1.0-onesbox1)
    return ((1.0-fbmix1) * U_beta1 + fbmix1 * Ac1*cp.exp(1j*Pc1)) * onesbox1 + Ac1*cp.exp(1j*Pc1) * (1.0- onesbox1)


def gauss2Dforfit(xygrid, A, wx, wy, x0, y0, O):
    xgrid, ygrid = xygrid
    g = A * np.exp(-(xgrid - x0) ** 2 / (2*wx**2)) * np.exp(-(ygrid - y0) ** 2 / (2*wy**2)) + 0
    #plt.imshow(abs(g))
    #plt.title("2daussfit")
    return g.ravel()

def gauss2Dforplot(xgrid,ygrid, A, wx, wy, x0, y0, O):
    #xgrid, ygrid = xygrid
    g = A * np.exp(-(xgrid - x0) ** 2 / (2*wx**2)) * np.exp(-(ygrid - y0) ** 2 / (2*wy**2)) + 0
    print(g)
    plt.imshow(abs(g))
    plt.title("Gaussian 2D plot with prev. fitted parameters")
    plt.show()
    
def gauss2Dlinefit(xgrid,ygrid, A, wx, wy, x0, y0, O):
    #xgrid, ygrid = xygrid
    g = A * np.exp(-(xgrid - x0) ** 2 / (2*wx**2)) * np.exp(-(ygrid - y0) ** 2 / (2*wy**2)) + 0
    print(g)
    plt.plot(g[:,int(g.shape[1]*0.5)])
    plt.show()
    plt.imshow(abs(g))
    plt.title("linefit")
    plt.show()

def gauss_intensity_linefit(x, A, wx, x0, O):
    #xgrid, ygrid = xygrid
    g = A * np.exp(-2*(x - x0) ** 2 / (wx**2))  + O
    return g



import cupy as cp



def make_ion_mask_cp(shape, spots, radius):
    ny, nx = shape  # (rows, cols)

    spots = cp.asarray(spots, dtype=cp.float32)
    spots = spots[:, ::-1]  # (y, x) -> (x, y)

    ion_mask = cp.zeros((ny, nx), dtype=cp.bool_)
    r2 = radius * radius

    x = cp.arange(nx)
    y = cp.arange(ny)

    eventrack=1
    for x_spot, y_spot in spots:
        if np.mod(eventrack,2)==0:
            x_temp=x_spot
            y_temp=y_spot
        if np.mod(eventrack,1)==0:
            #x_spot=(x_temp+x_spot)*0.5
            #y_spot = (y_temp + y_spot) * 0.5
            dx2 = (x - x_spot) ** 2          # shape (nx,)
            dy2 = (y - y_spot) ** 2          # shape (ny,)

            # Broadcasting: (ny, 1) + (1, nx) -> (ny, nx)
            ion_mask |= (dy2[:, None] + dx2[None, :] <= r2)
        #eventrack+=1

    # dx2 = (x - nx*0.5) ** 2  # shape (nx,)   #For a circle at the center
    # dy2 = (y - ny*0.5) ** 2  # shape (ny,)
    #
    # # Broadcasting: (ny, 1) + (1, nx) -> (ny, nx)
    # ion_mask |= (dy2[:, None] + dx2[None, :] <= r2)
    # plt.imshow(np.abs(ion_mask.get()))
    # plt.title("ion_mask")
    # plt.show()
    return ion_mask


# Wrapper class for iterative fourier transform algorithms
class IFTA:

    # Initialize an IFTA for an SLM of specified size, with specified input and target light fields
    def __init__(self, size=(1024,1272), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, waist=0.01):
        self.size = size
        self.waist = waist

        # Initial SLM phase is random
        self.input = input

        
        self.slm_field = cp.array(input) * cp.exp(2j * pi * cp.random.random_sample(size))

        self.A = cp.abs(self.slm_field)
        self.B = cp.zeros(size)
        self.phi = cp.angle(self.slm_field)
        self.psi = cp.zeros(size)

        # Keep track of iterations and deviations from target phase and amplitude for each iteration
        self.it = []
        self.B_dev = []
        self.psi_dev = []
        self.min_dev = (0, self.A, self.phi, self.B, self.psi)

        self.target = target[0]
        self.image_field = cp.array(cp.zeros(size))
        self.spots = target[1]

        self.wavelength = wavelength
        self.f = f

    # Propagate the SLM plane light field to the image plane
    def propagate(self, slm_field=None):
        if slm_field is None:
            slm_field = self.slm_field
        # t = time.time()
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))
        # print('FFT time:' + str(time.time() - t))

        self.B = cp.abs(self.image_field)
        self.psi = cp.angle(self.image_field)
        return self.image_field

    # Optimization method
    def opt(self):
        pass

    # Backpropagate the image plane light field to the SLM plane
    def backpropagate(self, image_field=None):
        if image_field is None:
            image_field = self.image_field

        # t = time.time()
        backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_field), norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = cp.abs(backprop)
        self.phi = cp.angle(backprop)

        self.slm_field = self.input * cp.exp(1j * self.phi)

    # Iterate n times and optimize at each step, keeping track of the amplitude and phase deviations at each step, as
    # well as the iteration with minimum error
    def iterate(self, n):
        for i in range(n):
            self.propagate()

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / (2 * pi))
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < cp.min(cp.array(self.B_dev[:-1])):
                self.min_dev = (i + 1, cp.copy(self.A), cp.copy(self.phi), cp.copy(self.B), cp.copy(self.psi),
                                    self.B_dev[-1], self.psi_dev[-1])

            self.opt()
            self.backpropagate()

            print('Step %d' % i)
        return self.slm_field, self.image_field

    # Plot and save the results of the IFTA run
    def save_pattern(self, name, slm, correction=False, plots=(0, 1, 2, 3, 4, 5), min=False, target=True, show=(0, 1, 2, 3, 4, 5), wavelength=413, field=False):
        if field:
            slm.fieldtoBMP(self.slm_field.get(), name=name + '_input', color=(0 in plots), show=(0 in show), correction=correction, wavelength=wavelength)

            slm.fieldtoBMP(self.image_field.get(), name=name + '_output', color=(1 in plots), show=(1 in show), correction=correction, wavelength=wavelength)

            if target:
                slm.fieldtoBMP(self.target.get(), name=name + '_target', color=(2 in plots), show=(2 in show), correction=correction, wavelength=wavelength)
        else:
            if min:
                slm.ampToBMP(cp.abs(self.input).get(), name=name + '_input_amp', color=(0 in plots), show=(0 in show))
                slm.phaseToBMP(self.min_dev[2].get(), name=name + '_input_phase', correction=correction, color=(1 in plots), show=(1 in show), wavelength=wavelength)

                slm.ampToBMP(self.min_dev[3].get(), name=name + '_output_amp', color=(2 in plots), show=(2 in show))
                slm.phaseToBMP(self.min_dev[4].get(), name=name + '_output_phase', color=(3 in plots), show=(3 in show), wavelength=wavelength)
            else:

                slm.ampToBMP(cp.abs(self.slm_field).get(), name=name + '_input_amp', color=(0 in plots), show=(0 in show))
                slm.phaseToBMP(cp.angle(self.slm_field).get(), name=name + '_input_phase', correction=correction, color=(1 in plots), show=(1 in show), wavelength=wavelength)

                slm.ampToBMP(cp.abs(self.image_field).get(), name=name + '_output_amp', color=(2 in plots), show=(2 in show))
                slm.phaseToBMP(cp.angle(self.image_field).get(), name=name + '_output_phase', color=(3 in plots), show=(3 in show), wavelength=wavelength)

            if target:
                slm.ampToBMP(cp.abs(self.target).get(), name=name + '_target_amp', color=(4 in plots), show=(4 in show))
                slm.phaseToBMP(cp.angle(self.target).get(), name=name + '_target_phase', color=(5 in plots), show=(5 in show), wavelength=wavelength)

    # Take the average of the field in a circle of radius centered at pos
    def avg(self, field, pos, radius):
        avg = 0
        pts = 0
        #print("new crop")
        for x in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
            for y in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) <= radius ** 2:
                    avg += field[x + int(pos[0])][y + int(pos[1])]
                    #print("xpos",x + int(pos[0]),"ypos",y + int(pos[1]),"fieldang:",field[x + int(pos[0])][y + int(pos[1])])
                    pts += 1
        #plt.imshow(cp.abs(field).get())
        #plt.title("full field")
        #plt.show()
        #plt.imshow(np.abs(field[int(-radius * self.size[0]) + int(pos[0]):int(radius * self.size[0]) + 1+ int(pos[0]),int(-radius * self.size[0]) + int(pos[1]):int(radius * self.size[0]) + 1+ int(pos[1])].get()))
        #plt.title("avg")
        #plt.show()
        return avg / pts

    def beammax(self, field, pos, radius):
        avg = 0
        pts = 0
        maxtemp=0
        # print("new crop")
        for x in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
            for y in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) <= radius ** 2:
                    if cp.abs(field[x + int(pos[0])][y + int(pos[1])]) > maxtemp:
                        maxtemp= cp.abs(field[x + int(pos[0])][y + int(pos[1])])
        # plt.imshow(cp.abs(field).get())
        # plt.title("full field")
        # plt.show()
        # plt.imshow(np.abs(field[int(-radius * self.size[0]) + int(pos[0]):int(radius * self.size[0]) + 1+ int(pos[0]),int(-radius * self.size[0]) + int(pos[1]):int(radius * self.size[0]) + 1+ int(pos[1])].get()))
        # plt.title("maxtemp")
        # plt.show()
        return maxtemp

    def beammax_torch_orig(self, field, pos, radius):
        avg = 0
        pts = 0
        maxtemp=0
        # print("new crop")
        for x in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
            for y in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) <= radius ** 2:
                    if torch.abs(field[x + int(pos[0])][y + int(pos[1])]) > maxtemp:
                        maxtemp= torch.abs(field[x + int(pos[0])][y + int(pos[1])])
        #
        # H, W = field.shape
        # y0, x0 = pos
        #
        # r = int(max(1, 3))
        #
        # x_min = max(0, int(np.round(x0 - r)))
        # x_max = min(W - 1, int(np.round(x0 + r)))
        # y_min = max(0, int(np.round(y0 - r)))
        # y_max = min(H - 1, int(np.round(y0 + r)))
        #
        # patch = field[y_min:y_max + 1, x_min:x_max + 1]
        # I = torch.abs(patch) ** 2
        #
        # plt.imshow(np.abs(patch.cpu().numpy()))
        # plt.show()
        return maxtemp

    def beammax_torch(self, field, spot, waist=1):
        """
        field: complex tensor [H, W]
        spot: (x0, y0) float pixel coordinates
        waist: window radius in pixels
        """
        if field is None:
            field = self.image_field
        if spot is None:
            spot = self.spots

        H, W = field.shape
        y0, x0 = spot

        r = int(max(1, waist))

        x_min = max(0, int(x0 - r))
        x_max = min(W - 1, int(x0 + r))
        y_min = max(0, int(y0 - r))
        y_max = min(H - 1, int(y0 + r))

        patch = field[y_min:y_max + 1, x_min:x_max + 1]
        I = torch.abs(patch) ** 2

        plt.imshow(np.abs(patch.cpu().numpy()))
        plt.show()


        # ---- Quadratic fit for intensity peak ----
        yy, xx = torch.meshgrid(
            torch.arange(y_min, y_max + 1, device=field.device),
            torch.arange(x_min, x_max + 1, device=field.device),
            indexing='ij'
        )

        xx = xx.float()
        yy = yy.float()

        X = torch.stack([
            xx.flatten() ** 2,
            yy.flatten() ** 2,
            xx.flatten() * yy.flatten(),
            xx.flatten(),
            yy.flatten(),
            torch.ones_like(xx.flatten())
        ], dim=1)

        Y = I.flatten().unsqueeze(1)

        coeffs, *_ = torch.linalg.lstsq(X, Y)
        a, b, c, d, e, f = coeffs.squeeze()

        denom = 4 * a * b - c ** 2 + 1e-12
        x_peak = (c * e - 2 * b * d) / denom
        y_peak = (c * d - 2 * a * e) / denom

        I_peak = (
                a * x_peak ** 2 + b * y_peak ** 2 + c * x_peak * y_peak
                + d * x_peak + e * y_peak + f
        )

        amp_peak = torch.sqrt(torch.clamp(I_peak, min=0.0))

        # ---- Phase estimate (weighted complex average) ----
        # Use intensity as weight for robustness
        weighted_complex = torch.sum(patch * I)
        weight_sum = torch.sum(I) + 1e-12

        mean_complex = weighted_complex / weight_sum
        phase_peak = torch.angle(mean_complex)

        return amp_peak, phase_peak
    
    def gaussfitlineint(self,field,pos,radius):
        avg = 0
        pts = 0
        radiusx=radius* self.size[0]
        radiusy = radius * self.size[1]
        #print("xpos",pos[0],"ypos",pos[1])
        #print("radiusx",radiusx,"ry",radiusy)
        #print(int(pos[0]-radiusx),int(pos[0]+radiusx))
        field_crop=cp.array(field[int(pos[0]-radiusx)+1:int(pos[0]+radiusx)+1,int(pos[1]-radiusy)+1:int(pos[1]+radiusy)+1]).get()
        xlist=np.arange(int(pos[0]-radiusx)+1,int(pos[0]+radiusx)+1,1)
        ylist = np.arange(int(pos[1] - radiusy)+1, int(pos[1] + radiusy)+1, 1)
        int_crop=np.abs(field_crop)**2
        gausslinex=np.zeros(len(xlist))
        gaussliney = np.zeros(len(ylist))
        for ii in range(0,len(xlist),1):
            gausslinex[ii]=np.sum(int_crop[ii,:])
        for ii in range(0, len(ylist), 1):
            gaussliney[ii] = np.sum(int_crop[:,ii])


        p0forgaussfitx = [np.abs(field[int(pos[0]), int(pos[1])].get()) ** 2, radiusx * 0.2, int(pos[0]),0]
        try:
            popt, popc = scipy.optimize.curve_fit(gauss_intensity_linefit, xlist,
                                                  ydata=gausslinex,
                                                  p0=p0forgaussfitx, maxfev=5000)
            x_intensity=popt[0];
        except RuntimeError as e:
            print("xgauss fit error:", e)
            x_intensity = np.max(gausslinex);
        #plt.plot(xlist,gausslinex)
        #plt.plot(xlist,gauss_intensity_linefit(xlist,*popt))
        #plt.show()
        p0forgaussfity = [np.abs(field[int(pos[0]), int(pos[1])].get()) ** 2, radiusy * 0.2, int(pos[1]),0]
        try:
            popt, popc = scipy.optimize.curve_fit(gauss_intensity_linefit, ylist,
                                                  ydata=gaussliney,
                                                  p0=p0forgaussfity, maxfev=5000)
            y_intensity=popt[0]
        except RuntimeError as e:
            y_intensity = np.max(gaussliney);
            print("ygauss fit error:",e)
        print("x_intensity",x_intensity,"y_intensity",y_intensity)


        #plt.plot(ylist,gaussliney)
        #plt.plot(ylist, gauss_intensity_linefit(ylist, *popt))
        #plt.show()
        #plt.imshow(int_crop)
        #plt.show()
        return (x_intensity)#+y_intensity)*0.5

    def gaussfitlineint_phase(self,field,pos,radius):
        avg = 0
        pts = 0
        radiusx=radius* self.size[0]
        radiusy = radius * self.size[1]
        #print("xpos",pos[0],"ypos",pos[1])
        #print("radiusx",radiusx,"ry",radiusy)
        #print(int(pos[0]-radiusx),int(pos[0]+radiusx))
        field_crop=cp.array(field[int(pos[0]-radiusx)+1:int(pos[0]+radiusx)+1,int(pos[1]-radiusy)+1:int(pos[1]+radiusy)+1]).get()
        xlist=np.arange(int(pos[0]-radiusx)+1,int(pos[0]+radiusx)+1,1)
        ylist = np.arange(int(pos[1] - radiusy)+1, int(pos[1] + radiusy)+1, 1)
        int_crop=np.abs(field_crop)**2
        gausslinex=np.zeros(len(xlist))
        gaussliney = np.zeros(len(ylist))
        for ii in range(0,len(xlist),1):
            gausslinex[ii]=np.sum(int_crop[ii,:])
        for ii in range(0, len(ylist), 1):
            gaussliney[ii] = np.sum(int_crop[:,ii])


        p0forgaussfitx = [np.abs(field[int(pos[0]), int(pos[1])].get()) ** 2, radiusx * 0.2, int(pos[0]),0]
        try:
            popt, popc = scipy.optimize.curve_fit(gauss_intensity_linefit, xlist,
                                                  ydata=gausslinex,
                                                  p0=p0forgaussfitx)
            x_intensity=popt[0];
            xpos=popt[2]
        except RuntimeError as e:
            print("xgauss phasefit error:",e)
            x_intensity=np.max(gausslinex);
            xpos=np.average(xlist)
        #plt.plot(xlist,gausslinex)
        #plt.plot(xlist,gauss_intensity_linefit(xlist,*popt))
        p0forgaussfity = [np.abs(field[int(pos[0]), int(pos[1])].get()) ** 2, radiusy * 0.2, int(pos[1]),0]
        try:
            popt, popc = scipy.optimize.curve_fit(gauss_intensity_linefit, ylist,
                                                  ydata=gaussliney,
                                                  p0=p0forgaussfity)
            ypos=popt[2]
        except RuntimeError as e:
            print("ygauss phasefit error:",e)
            x_intensity=np.max(gaussliney);
            ypos=np.average(ylist)

        if 'popc' in locals() and popc[2][2] < 0.001 and xpos<=field.shape[0] and ypos<=field.shape[1] and xpos>=0 and ypos>=0:
            print("pops[2][2]",popc[2][2])
            gaussangle = np.angle(field[int(xpos), int(ypos)])
        else:
            gaussangle = np.angle(field[int(pos[0]), int(pos[1])])

        #plt.plot(ylist,gaussliney)
        #plt.plot(ylist, gauss_intensity_linefit(ylist, *popt))
        #plt.show()
        #plt.imshow(int_crop)
        #plt.show()
        return (gaussangle)#+y_intensity)*0.5

            
        

    def gaussfit(self, field, pos, radius):
        avg = 0
        pts = 0
        radiusx=radius* self.size[0]
        radiusy = radius * self.size[1]
        field_crop=cp.array(field[int(pos[0]-radiusx):int(pos[0]+radiusx),int(pos[1]-radiusy):int(pos[1]+radiusy)]).get()
        xlist=np.arange(int(pos[0]-radiusx),int(pos[0]+radiusx),1)
        ylist = np.arange(int(pos[1] - radiusy), int(pos[1] + radiusy), 1)
        xlistg1, ylistg1 = np.meshgrid(xlist, ylist)
        xlistg = xlistg1.ravel()
        ylistg = ylistg1.ravel()
        try:
            #popt,popc=scipy.optimize.curve_fit(gauss2Dforfit,np.array((xlistg,ylistg)),ydata=np.abs(field_crop.ravel())**2,p0=[np.abs(field[int(pos[0]),int(pos[1])].get())**2, radiusx*0.5, radiusy*0.5, int(pos[0]), int(pos[1]), 0])
            p0forgaussfit = [np.abs(field[int(pos[0]), int(pos[1])].get()) ** 2, radiusy*0.2, radiusx*0.2, int(pos[0]),
                             int(pos[1]), 0]
            popt, popc = scipy.optimize.curve_fit(gauss2Dforfit, np.array((xlistg1, ylistg1)),
                                                  ydata=(np.abs(field_crop) ** 2).ravel(),
                                                  p0=p0forgaussfit)#,options={'ftol': 1e-3, 'xtol': 1e-3, 'gtol': 1e-3} )

            if False: #printing popt gaussian 2D values
                #print("p0",p0forgaussfit)
                #print("pos[0,1]",pos[0],pos[1])
                #print("xlist,ylist",xlistg,ylistg)
                print("popt",popt)
                plt.imshow(np.abs(field_crop)**2)
                plt.title("cropped field**2")
                plt.show()
                field_crop_ints=np.abs(field_crop)**2
                plt.plot(field_crop_ints[int(field_crop_ints.shape[1]*0.5),:])
                xlist=np.arange(int(pos[0]-radiusx),int(pos[0]+radiusx),1)
                ylist = np.arange(int(pos[1] - radiusy), int(pos[1] + radiusy), 1)
                xlistg2, ylistg2 = np.meshgrid(xlist, ylist)
                gauss2Dlinefit(xlistg2,ylistg2, *popt)
                plt.show()
                print("lineplotdone")
                gaussfitout=gauss2Dforfit(np.array((ylistg, xlistg)),*popt)
                plt.imshow(gaussfitout.reshape(len(xlist),len(ylist)))
                plt.title("reshaped fit")
                plt.show()
                gauss2Dforplot(xlistg2, ylistg2, *p0forgaussfit)
                plt.show()
                print("first auss plotdone")
                gauss2Dforplot(xlistg2, ylistg2,*popt)
                plt.title("fit")
                plt.show()
            returnval = popt[0]
            print(type(popt[0]))
            if popc[3][3] > 1 or popc[4][4] > 1:
                for x in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                    for y in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                        if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) <= radius ** 2:
                            avg += field[x + int(pos[0])][y + int(pos[1])]
                            pts += 1
                returnval = (avg / pts).get()
                
        except RuntimeError as e:
            for x in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                for y in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                    if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) <= radius ** 2:
                        avg += field[x + int(pos[0])][y + int(pos[1])]
                        pts += 1
            returnval= (avg / pts).get()
            print(type(avg/pts))
            
        return returnval
        #return avg / pts
        #return popt[0]
    
    def gaussfitphase(self, field, pos, radius):
        avg = 0
        pts = 0
        pos[0]=pos[0]+1
        pos[1]=pos[1]+1
        radiusx=radius* self.size[0]
        radiusy = radius * self.size[1]
        field_crop=cp.array(field[int(pos[0]-radiusx):int(pos[0]+radiusx),int(pos[1]-radiusy):int(pos[1]+radiusy)]).get()
        xlist=np.arange(int(pos[0]-radiusx),int(pos[0]+radiusx),1)
        ylist = np.arange(int(pos[1] - radiusy), int(pos[1] + radiusy), 1)
        xlistg, ylistg = np.meshgrid(xlist, ylist)
        xlistg = xlistg.ravel()
        ylistg = ylistg.ravel()
        try:
            popt,popc=scipy.optimize.curve_fit(gauss2Dforfit,np.array((xlistg,ylistg)),ydata=np.abs(field_crop.ravel())**2,p0=[np.abs(field[int(pos[0]),int(pos[1])].get())**2, radiusx, radiusy, int(pos[0]), int(pos[1]), 0])
            print("popt",popt,"popc3,4",popc[3][3],popc[4][4])
            if popc[3][3] < 1 and popc[4][4] < 1:
                gaussangle=np.angle(field[int(popt[3]),int(popt[4])])
            else:
                gaussangle = np.angle(field[int(pos[0]), int(pos[1])])
                
            #print("gaussangle", gaussangle)
        except RuntimeError as e:
            gaussangle=np.angle(field[int(pos[0]), int(pos[1])])

        return gaussangle
    

    # Return the phase and amplitude of set of beams at positions spots
    def beams(self, field=None, spots=None, waist=0.003):
        if field is None:
            field = self.image_field
        if spots is None:
            spots = self.spots
        #print("spots",spots)
        #return cp.array([self.avg(field, m, waist) for m in spots])
        #return cp.array([self.gaussfit(field, m, waist) for m in spots])
        #return cp.array([self.gaussfitlineint(field, m, waist) for m in spots])
        return cp.array([self.beammax(field, m, waist) for m in spots])
    def beams_phase(self, field=None, spots=None, waist=0.001):
        if field is None:
            field = self.image_field
        if spots is None:
            spots = self.spots
        #return cp.array([self.avg(np.angle(field), m, waist) for m in spots])
        #return cp.array([self.gaussfitphase(field, m, waist) for m in spots])
        #return cp.array([self.gaussfitlineint_phase(field, m, waist) for m in spots])
        return cp.array([np.angle(field[int(m[0])][int(m[1])]) for m in spots]) #Direct phase at spot position


    # Return the phase and amplitude of set of beams at positions spots
    def beams_torch(self, field=None, spots=None, waist=0.003):
        if field is None:
            field = self.image_field
        if spots is None:
            spots = self.spots
        #print("spots",spots)
        #return cp.array([self.avg(field, m, waist) for m in spots])
        #return cp.array([self.gaussfit(field, m, waist) for m in spots])
        #return cp.array([self.gaussfitlineint(field, m, waist) for m in spots])
        return (([self.beammax_torch_orig(field, m, waist) for m in spots]))

    def beams_phase_torch(self, field=None, spots=None, waist=0.001):
        if field is None:
            field = self.image_field
        if spots is None:
            spots = self.spots
        #return cp.array([self.avg(np.angle(field), m, waist) for m in spots])
        #return cp.array([self.gaussfitphase(field, m, waist) for m in spots])
        #return cp.array([self.gaussfitlineint_phase(field, m, waist) for m in spots])
        return ([torch.angle(field[int(m[0])][int(m[1])]) for m in spots]) #Direct phase at spot position


    # Calculate phase error
    def dev_phase(self, spots=None, waist=0.01, target=None):
        if spots is None:
            spots = self.spots
        if target is None:
            target = [0 for _ in spots]
        target = cp.array(target) / 2 / pi
        phases = cp.array([(cp.angle(self.avg(self.image_field, m, waist)) + 2 * pi) % (2 * pi) for m in spots]) / 2 / pi
        # return cp.max([cp.max([min(cp.abs(p1 - p2), 2 * pi - cp.abs(p1 - p2)) for p2 in phases]) for p1 in phases])
        # return scipy.stats.circstd(phases, high=pi, low=-pi)
        return cp.max(cp.min(cp.array([(phases - target + 1) % 1, (target - phases + 1) % 1])))

    # Calculate amplitude error
    def dev_amp(self, spots=None, waist=0.01, target=None):
        if spots is None:
            spots = self.spots
        if target is None:
            target = cp.array([1 for _ in spots])
        # print(spots)
        amps = cp.array([cp.abs(self.avg(cp.abs(self.image_field), m, waist)) for m in spots])
        amps /= cp.max(amps)
        print("amps", amps, "target", target)
        return cp.max(cp.abs(amps - target))

    # Calculate intensity error
    def dev_intensity(self, spots=None, waist=0.01, target=None):
        if spots is None:
            spots = self.spots
        if target is None:
            target = cp.array([1 for _ in spots])
        intensities = cp.array([self.avg(cp.abs(self.image_field)**2, m, waist) for m in spots])
        intensities /= cp.max(intensities)
        return cp.max(cp.abs(intensities - target))


# Implement the WGS algorithm (intensity control only)
class WGS(IFTA):

    def __init__(self, size=(1024,1272), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, start_phase=None, reference=None, consider_phase=False, waist=0.01):
        super().__init__(size, input, target, wavelength, f, waist)
        self.consider_phase = consider_phase

        self.input = input
        if start_phase is not None:
            self.slm_field = input * cp.exp(1j * pi * start_phase)
            self.A = cp.abs(self.slm_field)
            self.phi = cp.angle(self.slm_field)
        if reference is None:
            self.reference = cp.zeros(size)
        else:
            self.reference = reference

        self.g = cp.array([cp.ones(self.B.shape)])

        self.h = cp.array([cp.ones(self.psi.shape)])

    def opt(self):
        # print(self.spots)
        # print(self.B)
        # print(self.g.shape)
        avg_B = cp.sum(cp.array([self.B[m[0], m[1]] for m in self.spots])) / len(self.spots)
        g = cp.zeros(self.B.shape)
        for m in self.spots:
            g[m[0], m[1]] += (avg_B / self.B[m[0], m[1]]) * self.g[-1][m[0], m[1]]
        # g = cp.sum([(avg_B / self.B[-1][m[0], m[1]])[m[0], m[1]] for m in self.spots]) * self.g[-1]
        self.g = cp.append(self.g, [g], axis=0)


        if self.consider_phase:
            avg_psi = cp.sum(cp.array([self.psi[m[0], m[1]] for m in self.spots])) / len(self.spots)
            h = cp.zeros(self.psi.shape)
            for m in self.spots:
                h[m[0], m[1]] += (avg_psi / self.psi[m[0], m[1]]) * self.h[-1][m[0], m[1]]
            self.h = cp.append(self.h, [h], axis=0)

    def backpropagate(self, image_field=None):
        # t = time.time()
        if not self.consider_phase:
            backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(
                cp.array(self.g[-1]) * cp.array(self.target) * cp.exp(1j * self.psi)),
                                                   norm="ortho"))
        else:
            backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(self.g[-1] * self.target * cp.exp(1j * (self.psi + self.h[-1]))),
                                                   norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = cp.abs(backprop)
        self.phi = cp.angle(backprop)

        self.slm_field = cp.array(self.input) * cp.exp(1j * self.phi)

    def iterate(self, n):
        for i in range(n):
            self.propagate()

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / pi)
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < cp.min(cp.array(self.B_dev[:-1])):
                self.min_dev = (i + 1, cp.copy(self.A), cp.copy(self.phi), cp.copy(self.B), cp.copy(self.psi),
                                    self.B_dev[-1], self.psi_dev[-1])

            self.opt()
            self.backpropagate()

            print('Step %d' % i)
        return self.slm_field, self.image_field


# Unfinished IFTA
class CostOptimizer(WGS):

    def __init__(self, size=cp.array((1024,1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 start_phase=cp.zeros((1024,1272)), wavelength=413e-9, f=100):
        super().__init__(size, input, target, wavelength, f)
        self.slm_field = input * cp.exp(1j * start_phase)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.slm_field), norm="ortho"))
        self.cost = self.cost_fxn()

    def cost_fxn(self, field=None, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        if field is None:
            field = self.image_field

        beams = self.beams(field=field, spots=spots, waist=waist)
        avg_I = cp.mean(cp.abs(beams)**2) / cp.max(cp.abs(beams)**2)
        design_I = cp.ones(beams.shape)

        gamma = cp.sum(cp.abs(beams)**2 * design_I) / cp.sum(design_I**2)

        sigma = cp.sqrt(cp.sum(cp.abs(beams)**2 - gamma * design_I)**2 / len(beams))

        f = 0.5

        return -avg_I + f * sigma

    def opt(self):
        phi = cp.angle(self.slm_field)
        px = cp.array(cp.random.rand(2) * self.size, dtype=cp.uint)

        phi[px[0], px[1]], phi[px[0], px[1]] = cp.random.rand() * 2 * pi
        slm_field = self.input * cp.exp(1j * phi)
        image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))
        cost = self.cost_fxn(field=image_field)

        if cost < self.cost:
            self.slm_field = slm_field
            self.image_field = image_field
            self.cost = cost

        return cost

    def gradient(self, phase_1D, h=0.01):
        phase = cp.reshape(phase_1D, (-1, self.size[1]))

        current = self.cost_fxn(field=self.propagate(slm_field=phase))

        grad = cp.zeros(phase_1D.shape)

        for i in range(len(phase)):
            for j in range(len(phase[i])):
                phase[i, j] += h
                grad[i * len(phase) + j] = (self.cost_fxn(field=self.propagate(slm_field=phase)) - current) / h
                phase[i, j] -= h


# Unfinished IFTA
class OutputOutput(IFTA):
    def __init__(self, size=cp.array((1024,1272)), input=Profile.input_gaussian(), target=Profile.gaussian_array(1, 5),
                 wavelength=413e-9, f=100, waist=0.01, beta=1):
        super().__init__(size, input, target, wavelength, f, waist)
        self.propagate()

        self.a = cp.copy(self.slm_field)
        self.b = cp.copy(self.image_field)
        self.a_prime = []
        self.b_prime = []

        self.b_driving = []

        self.beta = beta

    def propagate(self, slm_field=None):
        if slm_field is None:
            slm_field = self.slm_field

        # t = time.time()
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))
        # print('FFT time:' + str(time.time() - t))

        self.B = cp.abs(self.image_field)
        self.psi = cp.angle(self.image_field)
        return self.image_field

    def opt(self):
        self.b_driving = cp.abs(self.target) * (2 * cp.exp(1j * cp.angle(self.b_prime)) -
                                                     cp.exp(1j * cp.angle(self.b))) - self.b

        return self.b_prime + self.beta * self.b_driving

    def constraints(self, slm_field):
        if slm_field is None:
            slm_field = self.slm_field

        return cp.abs(self.input) * cp.angle(slm_field)

    def backpropagate(self, image_field=None):
        if image_field is None:
            image_field = self.image_field

        # t = time.time()
        backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_field), norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = cp.abs(backprop)
        self.phi = cp.angle(backprop)

        self.slm_field = self.input * cp.exp(1j * self.phi)
        return self.slm_field

    def iterate(self, n):
        for i in range(n):
            self.a = self.backpropagate(image_field=self.b)
            self.a_prime = self.constraints(slm_field=self.a)
            self.b_prime = self.propagate(slm_field=self.a_prime)
            self.b = self.opt()

            self.slm_field = self.a_prime
            self.A = cp.abs(self.slm_field)
            self.phi = cp.angle(self.slm_field)

            self.image_field = self.b_prime
            self.B = cp.abs(self.image_field)
            self.psi = cp.angle(self.image_field)

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / (2 * pi))
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < cp.min(cp.array(self.B_dev[:-1])):
                self.min_dev = (i + 1, cp.copy(self.A), cp.copy(self.phi), cp.copy(self.B), cp.copy(self.psi),
                                self.B_dev[-1], self.psi_dev[-1])

            print('Step %d' % i)
        return self.slm_field, self.image_field


# Unfinished IFTA
class ThreeStep(IFTA):
    def __init__(self):
        super().__init__()


# Implement the algorithm in the Wu paper (phase and intensity control)
class Wu(IFTA):
    def __init__(self, size=(1024,1272), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, waist=0.001, array=True, target_beams=None, res_factor=1, start_phase=None, phase_memory=False,phaseconstant=False,P_c_old=None,outer_num=0.0,target_orig=None,tem01=False,target_New=None):
        # print('wu input', np.size(input))
        super().__init__(size=size, input=input, target=target, wavelength=wavelength, f=f, waist=waist)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if start_phase is None:
            print("startphase is none")
            if target is None:
            #start_phase = 2 * pi * cp.random.random_sample(size)
                start_phase = 2 * pi * cp.random.normal(0,0.2,size)
            else:
                start_phase = 2 * pi * cp.random.normal(0, 0.1, size)
                phase_path_tem01 = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01_nocorr.txt'
                #phase_path_tem01 = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01_scalepad4_withcorr.txt'
                phase_mask_tem01 = pad_border(np.loadtxt(phase_path_tem01),size)
                phase_mask_tem01=cp.flip(phase_mask_tem01, axis=1)
                #phase_mask_tem01[:, phase_mask_tem01.shape[1] // 2 :]=0
                #phase_mask_tem01[:, :phase_mask_tem01.shape[1] // 2] =np.pi
                phase_path_5arb_063 = r'C:\Users\RiceT\Documents\SLM_computation\images\tem00_arb5_063.txt'
                phase_mask_5arb_063 = pad_border(np.loadtxt(phase_path_5arb_063), size)
                start_phase = cp.angle(cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(self.target), norm="ortho")))#+cp.asarray(phase_mask_tem01)
                #start_phase =cp.asarray(phase_mask_tem01)#cp.asarray(phase_mask_5arb_063)+
                #start_phase[:,int(len(start_phase[0])*0.5)-40:int(len(start_phase[0])*0.5)+40]=2 * pi * cp.random.random_sample(start_phase[:,int(len(start_phase[0])*0.5)-40:int(len(start_phase[0])*0.5)+40].shape)
                phase_path_47_Tcurve = r'C:\Users\RiceT\Documents\SLM_computation\images\Tcurve_47ions.txt'
                start_phase = cp.asarray(pad_border(np.loadtxt(phase_path_47_Tcurve), size))
                phaseTest11101beams=r'C:\Users\RiceT\Documents\SLM_computation\images\phaseTest47beams_every3nulled.txt'
                                     # phaseTest11101beams.txt
                start_phase = cp.asarray(pad_border(np.loadtxt(phaseTest11101beams), size))
                start_phase = 2 * pi * cp.random.normal(0, 0.1, size)
                if False:
                    print("size",size)
                    start_phase_x=cp.linspace(-1,1,size[1])
                    start_phase_y = cp.linspace(-1, 1, size[0])
                    start_phase_xx,start_phase_yy=cp.meshgrid(start_phase_x,start_phase_y)
                    start_phase+=0*(start_phase_xx**2+start_phase_yy**2)
                    plt.imshow(np.abs(start_phase.get()))
                    plt.title("start_phase")
                    plt.show()
            Target_orig_fft=cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(cp.asarray(target_orig[0])), norm="ortho"))
            #Target_orig_fft=Target_orig_fft/cp.max(Target_orig_fft)
            estimated_slm_phase=cp.angle(Target_orig_fft)+2 * pi * cp.random.normal(0,0.1*(1-cp.abs(Target_orig_fft/cp.max(cp.abs(Target_orig_fft)))),self.size)
            start_phase=estimated_slm_phase
            #start_phase= 2 * pi * cp.random.normal(0, 0.1, size)
            # plt.imshow(np.angle(Target_orig_fft.get()))
            # plt.figure()
            # plt.imshow(np.abs(Target_orig_fft.get()))
            # plt.figure()
            # plt.imshow((start_phase.get()))
            # plt.show()

        if False: #For simple curved phase

            local_y = cp.arange(size[0]) - (size[0] / 2.0) + 0.5
            local_x = cp.arange(size[1]) - (size[1] / 2.0) + 0.5

            start_phase = runsettings.start_phase_curve * (local_y[:, None] ** 2 + local_x[None, :] ** 2) #0.00002
            start_phase = cp.asarray(start_phase)
            # #curved start phase:
            # for i in range(size[0]):
            #     for j in range(size[1]):
            #         start_phase[i][j] = (0.00002 * (i - (size[0] / 2.0) + 0.5) ** 2 + 0.00002 * (
            #                     j - (size[1] / 2.0) + 0.5) ** 2)  # MDS Enter center of the beam here

        if False: #Use axicon https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-5-5875
            local_y = cp.arange(size[0]) - (size[0] / 2.0) + 0.5
            local_x = cp.arange(size[1]) - (size[1] / 2.0) + 0.5
            slm_fft_angle=cp.angle(Target_orig_fft)
            normalised_slm_fft_magnitude=cp.abs(Target_orig_fft/cp.max(cp.abs(Target_orig_fft)))
            #axicon_phase = runsettings.start_phase_curve * cp.sqrt((local_y[:, None] ** 2 + local_x[None, :] ** 2))
            #Using a tilt
            axicon_phase = runsettings.start_phase_curve * ((local_y[:, None] + local_x[None, :]))
            random_factor = 0.3*cp.random.random(self.size)
            use_axicon = random_factor > ( normalised_slm_fft_magnitude)
            use_estimated = ~use_axicon
            start_phase[use_axicon] = axicon_phase[use_axicon]
            start_phase[use_estimated] = slm_fft_angle[use_estimated]
        if False: #S1 method from Clark https://opg.optica.org/oe/fulltext.cfm?uri=oe-24-6-6249
            local_y = cp.arange(size[0]) - (size[0] / 2.0) + 0.5
            local_x = cp.arange(size[1]) - (size[1] / 2.0) + 0.5
            slm_fft_angle = cp.angle(Target_orig_fft)
            normalised_slm_fft_magnitude = cp.abs(Target_orig_fft / cp.max(cp.abs(Target_orig_fft)))
            tilt_phase = runsettings.start_phase_curve * ((local_y[:, None] + local_x[None, :]))
            start_phase = normalised_slm_fft_magnitude* cp.mod((slm_fft_angle + tilt_phase),2*cp.pi) - tilt_phase


        start_phase = cp.asarray(start_phase)
        self.p = cp.asarray(start_phase)
        # plt.imshow(np.abs(np.mod(start_phase.get(),2*np.pi)))
        # plt.title("start_phase")
        # plt.figure()

        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.slm_field), norm="ortho"))

        # plt.imshow(np.abs(self.image_field.get()))
        # plt.title("start_image_field_abs")
        # plt.show()

        self.S = cp.zeros(size)
        self.S[:, self.S.shape[1] // 2 +6 :] = 1#+300
        #self.S[self.S.shape[0] // 2:, :] = 1

        if False:
            self.S1 = cp.zeros(size)
            self.S1[:self.S1.shape[0] // 2, :self.S1.shape[1] // 2] = 1
           # self.S1[1018:1030,1244:1256]=1

            self.S2 = cp.zeros(size)
            self.S2[:self.S2.shape[0] // 2, self.S2.shape[1] // 2:] = 1
           # self.S2[1018:1030, 1258:1270] = 1

            self.S3 = cp.zeros(size)
            self.S3[self.S3.shape[0] // 2:, :self.S3.shape[1] // 2] = 1
           # self.S3[1018:1030, 1276:1285] = 1

            self.S4 = cp.zeros(size)
            self.S4[self.S4.shape[0] // 2:, self.S4.shape[1] // 2:] = 1
           # self.S4[1018:1030, 1286:1299] = 1

       # self.S5 = cp.zeros(size)
       # self.S5[1018:1030, 1228:1242] = 1

       # self.S6 = cp.zeros(size)
       # self.S6[1018:1030, 1300:1312] = 1
        self.A_c_old= cp.zeros(size)
        self.P_c_old = cp.zeros(size)
        self.stepstartflag=0
        self.totalsteperr=[]
        self.stepnum=0
        self.outer_num=0
        self.phaseconstant=phaseconstant
        self.P_c_old=P_c_old
        self.target_orig = target_orig
        self.A_t_orig=cp.abs(target_orig[0])
        self.P_t_orig = cp.angle(target_orig[0])

        # Performance trackers
        self.eff = []
        self.nonunif = []
        self.phase_err = []

        self.res_factor = res_factor
        # print(self.spots)


        self.ones_box = cp.zeros(size)

        self.ones_box[int(size[0] / 2 - size[0] / 6):int(size[0] / 2 + size[0] / 6), int(size[1] / 2 - size[1] / 6):int(size[1] / 2 + size[1] / 6)] = 1
        #self.ones_box = cp.ones(size)
        if True: #Having a circular region of interest
            local_y = np.arange(size[0]) - (size[0] / 2.0) + 0.5
            local_x = np.arange(size[1]) - (size[1] / 2.0) + 0.5

            self.ones_box = cp.asarray(((local_y[:, None] ** 2 + local_x[None, :] ** 2) < (80*runsettings.scalepad_global)**2).astype(int))
            print("res_factor",res_factor)
        # plt.imshow(np.abs(self.ones_box.get()))
        # plt.show()



        if target!=None:
            print("target",target[1])
            print("target", np.max(cp.array(target[1])[:,0]))
            print("target", np.min(cp.array(target[1])[:,1]))
            self.Sin= cp.zeros(target[0].shape)
            extra=30
            self.Sin[cp.min(cp.array(target[1])[:,0])-extra:cp.max(cp.array(target[1])[:,0])+extra,cp.min(cp.array(target[1])[:,1])-extra:cp.max(cp.array(target[1])[:,1])+extra]=1
            self.Sout=cp.ones(target[0].shape)
            self.Sout=self.Sout-self.Sin
            #plt.imshow(self.Sout)
            #plt.show()

        # self.I = cp.eye(self.size[0], M=self.size[1], k=0)
        self.I = cp.ones(size)
        #self.SO1= self.I -(self.S1+self.S2+self.S3+self.S4)#+self.S5+self.S6)

        self.A_t = cp.abs(self.target)
        self.P_t = cp.angle(self.target)

        # Generate a mask with 0 everywhere except where the target pattern is, to divide the image into target and noise areas
        self.mask = cp.where(cp.abs(self.target) > 1e-2, cp.ones(size), cp.zeros(size))
        # slm.ampToBMP(cp.abs(self.mask), name='mask', color=True, show=False)
        #plt.imshow(self.mask.get())
        #plt.show()
        #Get the spot positions and create a circle around it
        print(cp.multiply(cp.asarray(target_orig[1]),self.res_factor))
        self.ion_mask = make_ion_mask_cp(size, spots=cp.multiply(cp.asarray(target_orig[1]), self.res_factor),radius=10)#50

        if tem01==True:
            spots_temp=cp.multiply(cp.asarray(target_orig[1]), self.res_factor)
            self.ion_mask_small = make_ion_mask_cp(size, spots=cp.array([(spots_temp[kk]+spots_temp[kk+1])/2.0 for kk in range(0,len(spots_temp),2)]),radius=2)
            # plt.imshow(self.ion_mask_small.get())
            # plt.show()
        else:
            spots_temp=cp.multiply(cp.asarray(target_orig[1]), self.res_factor)
            self.ion_mask_small = make_ion_mask_cp(size, spots=spots_temp,radius=2)
            # plt.imshow(self.ion_mask_small.get())
            # plt.show()

        #self.ones_box = self.ion_mask
        if True: #Maximum efficiency check

            Target_orig_fft=cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(cp.asarray(target_orig[0])), norm="ortho"))
            #Target_orig_fft=Target_orig_fft/cp.max(Target_orig_fft)
            print("Limit Schatrz", cp.mean(cp.abs(Target_orig_fft))**2/cp.mean(cp.abs(Target_orig_fft)**2))
            # plt.imshow(cp.abs(Target_orig_fft).get())
            # plt.show()
            estimated_slm_phase=cp.angle(Target_orig_fft)+2 * pi * cp.random.normal(0,0.05*(1-cp.abs(Target_orig_fft/cp.max(cp.abs(Target_orig_fft)))),self.size)


            Estimated_image = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.abs(self.input)*cp.exp(1j*estimated_slm_phase)), norm="ortho"))
            Maximum_possible_efficiency_orig= cp.sum((cp.abs(self.input))*(cp.abs(Target_orig_fft)))/(cp.sum(cp.abs(self.input)**2)*cp.sum(cp.abs(Target_orig_fft)**2))
            Maximum_possible_efficiency= cp.sum((cp.abs(self.input))*(cp.abs(Target_orig_fft)))**2/(cp.sum(cp.abs(self.input)**2)*cp.sum(cp.abs(Target_orig_fft)**2))
            Maximum_possible_efficiency_complex= cp.sum((cp.conj(self.input*cp.exp(1j*(cp.angle(Target_orig_fft)))))*((Target_orig_fft)))**2/(cp.sum(cp.abs(self.input)**2)*cp.sum(cp.abs(Target_orig_fft)**2))

            Estimated_efficiency = cp.sum(cp.abs((cp.conj(Estimated_image)) * (cp.asarray(target_orig[0])) ))/ (
                        cp.sum(cp.abs(Estimated_image) ** 2) * cp.sum(cp.abs(cp.asarray(target_orig[0])) ** 2))
            
            Estimated_Final_fidelity = ((cp.abs(
                cp.sum(cp.conj(Estimated_image) * (cp.asarray(target_orig[0])) * cp.asarray(self.ones_box))) ** 2 / (
                                       (cp.abs(cp.sum(cp.conj(Estimated_image) * Estimated_image * cp.asarray(
                                           self.ones_box)))) * cp.abs(cp.sum(
                                   cp.conj((cp.asarray(target_orig[0]))) * (cp.asarray(target_orig[0])) * cp.asarray(
                                       self.ones_box))))).item())

            Estimated_Final_efficiency = cp.sum(cp.abs(cp.conj(Estimated_image) * Estimated_image) * cp.asarray(self.mask)) / cp.sum(
                cp.abs(cp.conj(Estimated_image) * Estimated_image))

            Estimated_Final_ion_fidelity = ((cp.abs(
                cp.sum(cp.conj(Estimated_image) * (cp.asarray(target_orig[0])) * cp.asarray(self.ion_mask))) ** 2 / (
                                           (cp.abs(cp.sum(cp.conj(Estimated_image) * Estimated_image * cp.asarray(
                                               self.ion_mask)))) * cp.abs(cp.sum(cp.conj((cp.asarray(target_orig[0]))) * (
                                       cp.asarray(target_orig[0])) * cp.asarray(self.ion_mask))))).item())

            #print("type(target_New[0]): ",type(target_New[0]))
            print("Estimated fidelities, tot, fidelity, ion: ",Estimated_Final_fidelity,Estimated_Final_efficiency, Estimated_Final_ion_fidelity)
            # Now ensure target_New[0] is a cupy array
            if False:
                target_New = cp.asarray(target_orig)#cp.asarray(target_New[0])
                # Now perform the division to normalize
                target_New[0] = cp.divide(target_New[0], cp.sqrt(cp.sum(cp.abs(target_New[0]) ** 2) + 1e-12))
                Estimated_efficiency_new_target = cp.sum(cp.abs((cp.conj(Estimated_image)) * (cp.asarray(target_New[0])) ))/ (
                            cp.sum(cp.abs(Estimated_image) ** 2) * cp.sum(cp.abs(cp.asarray(target_New[0])) ** 2))

            print("Maximum_possible_efficiency",Maximum_possible_efficiency,"square",Maximum_possible_efficiency**2)
            print("Maximum_possible_efficiency_complex", Maximum_possible_efficiency_complex, "square_complex",
                  Maximum_possible_efficiency_complex ** 2)
            print("Estimated_efficiency",Estimated_efficiency)#,"Estimated eff New target",Estimated_efficiency_new_target)
        # plt.imshow(np.abs(Target_orig_fft.get()))
        # plt.title("target_orig_fft")
        # plt.figure()
        # plt.imshow(np.abs(self.input.get()))
        # plt.figure()
        # plt.imshow(np.abs(self.input.get())-np.abs(Target_orig_fft.get()))
        # plt.figure()
        # plt.imshow(np.angle(np.exp(1j*estimated_slm_phase).get()))
        # plt.title("estimated phase")
        # plt.figure()
        # plt.imshow(np.abs(cp.asarray(target_orig[0]).get()))
        # plt.title("Actual Target image")
        # plt.figure()
        # # plt.imshow(np.abs(cp.asarray(target_New[0]).get()))
        # # plt.title("New Target image")
        # # plt.figure()
        # plt.imshow(np.abs(Estimated_image.get()))
        # plt.title("estimated image")
        # plt.figure()
        # plt.imshow(np.angle(cp.asarray(target_orig[0]).get()))
        # plt.title("Actual Target image angle")
        # plt.figure()
        # plt.imshow(np.angle(Estimated_image.get()))
        # plt.title("estimated image angle")
        # plt.figure()
        # plt.plot(np.angle(Estimated_image[1024,:].get()))
        # plt.title("estimated image angle")
        # plt.figure()
        # plt.plot(np.abs(Estimated_image[1024,:].get()))
        # plt.title("estimated image")
        # plt.show()

        self.S=torch.as_tensor(self.S)
        self.I=torch.as_tensor(self.I)



        if target_beams is None:
            self.target_amp = cp.array([cp.abs(self.avg(field=self.target, pos=spot, radius=waist)) for spot in self.spots])
            self.target_phase = cp.array([cp.angle(self.avg(field=self.target, pos=spot, radius=waist)) for spot in self.spots])
        else:
            self.target_amp = cp.abs(target_beams)
            self.target_phase = cp.angle(target_beams)

        # True if the target is an array of beams
        self.array = array
        # #MDS commenting out this part to avoid having a starting nonunif term
        # if self.array:
        #     self.nonunif.append(self.dev_amp(waist=0.001, target=self.target_amp))#original
        #     self.phase_err.append(self.dev_phase(waist=0.001, target=self.target_phase))#original
        #     #self.nonunif.append(cp.mean(cp.abs(cp.abs(self.beams(waist=0.0015)) - self.amps())))#MDS added
        #     #self.phase_err.append(cp.mean(cp.abs(self.beams_phase(waist=0.0005) - self.phases())))#MDS added
        # else:
        #     self.nonunif.append(self.nonuniformity())
        #     self.phase_err.append(self.phase_error())

        self.phase_memory = phase_memory

    # Execute a single iteration of the algorithm
    #Trying to implement adaptive GSW from https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-2-1412
    def step_adaptive(self): #original
        #old_p=self.p
        if np.mod(int(self.stepnum/2),2)==0:
            fb=0.2#MDS added`
        else:
            fb=0.2 #original plus
        cp._default_memory_pool.free_all_blocks()
        U_c = self.image_field
        if 5<1:
            plt.imshow(np.abs(self.image_field.get())**2)
            plt.xlim(560,700)
            plt.ylim(500,520)
            plt.colorbar()
            plt.title("image")
            plt.show()
            plt.imshow(np.angle(self.image_field.get()))
            plt.xlim(560,700)
            plt.ylim(500,520)
            plt.colorbar()
            plt.title("image")
            plt.show()
            plt.imshow(np.angle(self.slm_field.get()))
            plt.title("slm")
            plt.show()
        # if self.res_factor != 1:
        #     U_c = pad_border(U_c, U_c.shape * 2)
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)
        ones_box=cp.zeros(A_c.shape)
        xbox_min=int(A_c.shape[0]/2 - A_c.shape[0]/4)
        xbox_max = int(A_c.shape[0] / 2 + A_c.shape[0] / 6)
        ybox_min = int(A_c.shape[1] / 2 - A_c.shape[1] / 6)
        ybox_max = int(A_c.shape[1] / 2 + A_c.shape[1] / 6)
        
        ones_box[xbox_min:xbox_max,ybox_min:ybox_max]=1
        #ones_box = cp.ones(A_c.shape)

        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S + P_c * (self.I - self.S)
        U_alpha = A_alpha * cp.exp(1j * P_alpha)
        if self.stepstartflag!=0:
            print("feedback")
            #U_alpha = A_alpha * cp.exp(1j * P_alpha) - fb*self.A_c_old* cp.exp(1j * self.P_c_old) *(self.I-self.mask) #Used so far 16/12/2025
            U_alpha = A_alpha * cp.exp(1j * P_alpha) - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask)
            #U_alpha= np.multiply(U_alpha,cp.ones(self.size))-fb*self.U_alpha_old*(cp.ones(self.size)-self.mask) #MDS added
        #else:
        #    self.stepstartflag=1
        U_alpha_boxed=(0.9*U_alpha+0.1*A_c*cp.exp(1j*P_c))*ones_box+ A_c*cp.exp(1j*P_c)*(self.I-ones_box)
        #u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha), norm="ortho"))
        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha_boxed), norm="ortho"))
        
       #zoomfft_Func=ZoomFFT2D(self.size, self.size, (0,0), (2,2), None, direction="forward")
       # u_alpha = cp.asarray(zoomfft_Func(U_alpha.get()))
       # u_alpha =u_alpha/cp.max(cp.abs(u_alpha))
        #fig, axs = plt.subplots(2, 1)
        #axs[0].imshow(np.abs(u_alpha.get()))
        #axs[0].set_title("u alpha orig")
        # plt.show()
        #axs[1].imshow(np.abs(uf_alpha.get()))
        #axs[1].set_title("u alpha fft")
        #plt.show()
        p_alpha = cp.angle(u_alpha)

        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S) + P_c * self.S
        U_beta = A_beta * cp.exp(1j * P_beta)

        if self.stepstartflag!=0:
            #U_beta = A_beta * cp.exp(1j * P_beta) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (self.I - self.mask)#Used so far 16/12/2025
            U_beta = A_beta * cp.exp(1j * P_beta) - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask)
            #U_beta= np.multiply(U_beta,cp.ones(self.size))-fb*self.U_beta_old*(cp.ones(self.size)-self.mask) #MDS added

        else:
            self.stepstartflag=1
        U_beta_boxed = (0.9*U_beta +0.1*A_c * cp.exp(1j * P_c))* ones_box + A_c * cp.exp(1j * P_c) * (self.I - ones_box)
        self.A_c_old =A_c
        self.P_c_old = P_c
        #u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta), norm="ortho"))
        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta_boxed), norm="ortho"))
        
        #zoomfft_Func=ZoomFFT2D(self.size, self.size, (0,0), (2,2), None, direction="forward")
        #u_beta = cp.asarray(zoomfft_Func(U_beta.get()))
        #u_beta =u_beta/cp.max(cp.abs(u_beta))
        #fig, axs = plt.subplots(2, 1)
        #axs[0].imshow(np.abs(u_beta.get()))
        #axs[0].set_title("u_beta orig")
        # plt.show()
        #axs[1].imshow(np.abs(uf_beta.get()))
        #axs[1].set_title("u _beta fft")
        #plt.show()
        
        p_beta = cp.angle(u_beta)

        self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field)**2)) #Used so far
        self.image_field *= cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2))
        print("Target sum", cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2)))
        print("image sum", cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2)))
        
        #plt.imshow(np.abs(self.image_field.get()))
        #plt.title("image after")
        #plt.show()
        #original
        #self.totalsteperr.append(cp.average(cp.abs(cp.abs(self.image_field * ones_box) - cp.abs(self.A_t * ones_box))))
        #self.totalsteperr.append(cp.average(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)-cp.abs((self.A_t/cp.max(self.A_t))*ones_box))))
        #Bowman definition
        self.totalsteperr.append((cp.sum(cp.abs(cp.conj(self.A_t*cp.exp(1j*self.P_t))*self.image_field))**2)/(cp.sum(cp.abs(self.A_t) ** 2)))
        #plt.imshow(cp.abs(self.image_field).get())
        #plt.title("imagefield plain")
        #plt.show()
        #plt.imshow(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)).get())
        #plt.title("imagefield")
        #plt.show()
        #plt.imshow(cp.abs(cp.abs((self.A_t/cp.max(self.A_t))*ones_box)).get())
        #plt.title("Target A_t")
        #plt.show()
        #plt.imshow(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)-cp.abs((self.A_t/cp.max(self.A_t))*ones_box)).get())
        #plt.title("totalsteperror")
        #plt.show()
        print("totalerror",self.totalsteperr[-1])
        #plt.show(block=False)
        #plt.pause(1)
        #plt.imshow(np.angle(self.slm_field.get()))
        #plt.title("slm after")
        
        #plt.show()

    def step(self):  # original
        # old_p=self.p
        if np.mod(int(self.stepnum / 2), 2) == 0:
            fb = 0.0  # MDS added`
            fbmix=0.0
        else:
            fb = 0.0# original plus
            fbmix = 0.0
        #cp._default_memory_pool.free_all_blocks()
        U_c = self.image_field
        if 5 < 1:
            plt.imshow(np.abs(self.image_field.get()) ** 2)
            plt.xlim(560, 700)
            plt.ylim(500, 520)
            plt.colorbar()
            plt.title("image")
            plt.show()
            plt.imshow(np.angle(self.image_field.get()))
            plt.xlim(560, 700)
            plt.ylim(500, 520)
            plt.colorbar()
            plt.title("image")
            plt.show()
            plt.imshow(np.angle(self.slm_field.get()))
            plt.title("slm")
            plt.show()
        # if self.res_factor != 1:
        #     U_c = pad_border(U_c, U_c.shape * 2)
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)
        if self.phaseconstant==True and self.P_c_old is not None and self.P_c_old.size > 0:
            P_c = self.P_c_old
        ones_box = cp.zeros(A_c.shape)
        xbox_min = int(A_c.shape[0] / 2 - A_c.shape[0] / 6)
        xbox_max = int(A_c.shape[0] / 2 + A_c.shape[0] / 6)
        ybox_min = int(A_c.shape[1] / 2 - A_c.shape[1] / 6)
        ybox_max = int(A_c.shape[1] / 2 + A_c.shape[1] / 6)

        ones_box[xbox_min:xbox_max, ybox_min:ybox_max] = 1
        #ones_box = cp.ones(A_c.shape)

        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S + P_c * (self.I - self.S)
        U_alpha = A_alpha * cp.exp(1j * P_alpha)
        if self.stepstartflag != 0:
            print("feedback")
            #U_alpha = A_alpha * cp.exp(1j * P_alpha) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (self.I - self.mask) #original 16/12/2025
            U_alpha = A_alpha * cp.exp(1j * P_alpha)*self.mask - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask)  #Use fb=1.0 for this
            # U_alpha= np.multiply(U_alpha,cp.ones(self.size))-fb*self.U_alpha_old*(cp.ones(self.size)-self.mask) #MDS added
            #U_alpha_boxed =make_U_alpha(self.target, A_c,P_c, self.mask, self.S,self.ones_box, fb,fbmix)
        #else:
        #    self.stepstartflag=1
        U_alpha_boxed = ((1-fbmix) * U_alpha + fbmix * A_c * cp.exp(1j * P_c)) * ones_box + A_c * cp.exp(1j * P_c) * (
                    self.I - ones_box)
        if False:
            plt.imshow( np.abs(U_alpha_boxed.get()))
            plt.title(" U_alpha_boxed")
        # u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha), norm="ortho"))
        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha_boxed), norm="ortho"))

        # zoomfft_Func=ZoomFFT2D(self.size, self.size, (0,0), (2,2), None, direction="forward")
        # u_alpha = cp.asarray(zoomfft_Func(U_alpha.get()))
        # u_alpha =u_alpha/cp.max(cp.abs(u_alpha))
        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(np.abs(u_alpha.get()))
        # axs[0].set_title("u alpha orig")
        # plt.show()
        # axs[1].imshow(np.abs(uf_alpha.get()))
        # axs[1].set_title("u alpha fft")
        # plt.show()
        p_alpha = cp.angle(u_alpha)

        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S) + P_c * self.S
        U_beta = A_beta * cp.exp(1j * P_beta)

        if self.stepstartflag != 0:
            #U_beta = A_beta * cp.exp(1j * P_beta) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (self.I - self.mask)#Used so far 16/12/2025
            U_beta = A_beta * cp.exp(1j * P_beta)*self.mask - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask) #Use it to negate whatever output in the same round, more effective #Use fb=1.0 for this
            # U_beta= np.multiply(U_beta,cp.ones(self.size))-fb*self.U_beta_old*(cp.ones(self.size)-self.mask) #MDS added
            #U_beta_boxed = make_U_beta(self.target, A_c,P_c, self.mask, self.S, self.ones_box, fb,fbmix)
        else:
            self.stepstartflag = 1
        U_beta_boxed = ((1.0-fbmix) * U_beta + fbmix * A_c * cp.exp(1j * P_c)) * ones_box + A_c * cp.exp(1j * P_c) * (
                    self.I - ones_box)
        if False:#self.stepnum==0:
            plt.figure()
            plt.imshow( np.abs(U_beta_boxed.get()))
            plt.title(" U_beta_boxed")
            plt.show()
        self.A_c_old = A_c
        self.P_c_old = P_c
        # u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta), norm="ortho"))
        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta_boxed), norm="ortho"))

        # zoomfft_Func=ZoomFFT2D(self.size, self.size, (0,0), (2,2), None, direction="forward")
        # u_beta = cp.asarray(zoomfft_Func(U_beta.get()))
        # u_beta =u_beta/cp.max(cp.abs(u_beta))
        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(np.abs(u_beta.get()))
        # axs[0].set_title("u_beta orig")
        # plt.show()
        # axs[1].imshow(np.abs(uf_beta.get()))
        # axs[1].set_title("u _beta fft")
        # plt.show()

        p_beta = cp.angle(u_beta)

        self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2))  # Used so far
        #self.image_field *= cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2))
        print("Target sum", cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2)))
        print("image sum", cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2)))

        # plt.imshow(np.abs(self.image_field.get()))
        # plt.title("image after")
        # plt.show()
        # original
        # self.totalsteperr.append(cp.average(cp.abs(cp.abs(self.image_field * ones_box) - cp.abs(self.A_t * ones_box))))
        # self.totalsteperr.append(cp.average(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)-cp.abs((self.A_t/cp.max(self.A_t))*ones_box))))
        # Bowman definition has outside of box as well
        #self.totalsteperr.append((cp.sum(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * self.image_field)*ones_box)) / (
        #    cp.sum((cp.abs(self.A_t) ** 2)*ones_box)))
        print("full fidelity:",(cp.sum(cp.abs(cp.conj(self.image_field) * self.image_field)*ones_box)) / (
            cp.sum((cp.abs(self.image_field) ** 2)*ones_box)) )
        image_field_insidebox=self.image_field / cp.sqrt(cp.sum(cp.abs(self.image_field*ones_box) ** 2))
        print("fidelity inside box:",(cp.sum(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * image_field_insidebox)*ones_box)) / (
            cp.sum((cp.abs(self.A_t) ** 2)*ones_box)) )
        print("fidelity inside box_orig:",(cp.sum(cp.abs(cp.conj(self.A_t_orig * cp.exp(1j * self.P_t_orig)) * image_field_insidebox)*ones_box)) / (
            cp.sum((cp.abs(self.A_t_orig) ** 2)*ones_box)) )
        # Bowman definition only inside the box
        #self.totalsteperr.append((cp.sum(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * image_field_insidebox)*ones_box)) / (
        #    cp.sum((cp.abs(self.A_t) ** 2)*ones_box)))
        self.totalsteperr.append((cp.sum(cp.abs(cp.conj(self.A_t_orig * cp.exp(1j * self.P_t_orig)) * image_field_insidebox)*ones_box)) / (
            cp.sum((cp.abs(self.A_t_orig) ** 2)*ones_box)))



        # plt.imshow(cp.abs(self.image_field).get())
        # plt.title("imagefield plain")
        # plt.show()
        # plt.imshow(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)).get())
        # plt.title("imagefield")
        # plt.show()
        # plt.imshow(cp.abs(cp.abs((self.A_t/cp.max(self.A_t))*ones_box)).get())
        # plt.title("Target A_t")
        # plt.show()
        # plt.imshow(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)-cp.abs((self.A_t/cp.max(self.A_t))*ones_box)).get())
        # plt.title("totalsteperror")
        # plt.show()
        print("totalerror", self.totalsteperr[-1])
        # plt.show(block=False)
        # plt.pause(1)
        # plt.imshow(np.angle(self.slm_field.get()))
        # plt.title("slm after")
        # plt.show()
        if False:#np.mod(self.stepnum,10)==0:
            plt.imshow(cp.abs(self.image_field).get())
            plt.title("image")
            plt.show()
            plt.imshow(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t))).get())
            plt.title("target")
            plt.show()
            plt.imshow(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * self.image_field).get())
            plt.title("fidelity")
            plt.show()

    def step_updatedloop_old(self):  # original
        # old_p=self.p
        if np.mod(int(self.stepnum / 2), 2) == 0:
            fb = 0.0  # MDS added`
            fbmix=0.0
        else:
            fb = 0.0# original plus
            fbmix = 0.0
        #cp._default_memory_pool.free_all_blocks()
        U_c = self.image_field
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)
        if self.phaseconstant==True and self.P_c_old is not None and self.P_c_old.size > 0:
            P_c = self.P_c_old
        if False:
            ones_box = cp.zeros(A_c.shape)
            xbox_min = int(A_c.shape[0] / 2 - A_c.shape[0] / 4)
            xbox_max = int(A_c.shape[0] / 2 + A_c.shape[0] / 4)
            ybox_min = int(A_c.shape[1] / 2 - A_c.shape[1] / 4)
            ybox_max = int(A_c.shape[1] / 2 + A_c.shape[1] / 4)

            ones_box[xbox_min:xbox_max, ybox_min:ybox_max] = 1
            #ones_box = cp.ones(A_c.shape)

        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S + P_c * (self.I - self.S)
        U_alpha = A_alpha * cp.exp(1j * P_alpha)
        if self.stepstartflag != 0:
            print("feedback")
            #U_alpha = A_alpha * cp.exp(1j * P_alpha) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (self.I - self.mask) #original 16/12/2025
            U_alpha = A_alpha * cp.exp(1j * P_alpha)*self.mask - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask)  #Use fb=1.0 for this
            # U_alpha= np.multiply(U_alpha,cp.ones(self.size))-fb*self.U_alpha_old*(cp.ones(self.size)-self.mask) #MDS added
            #U_alpha_boxed =make_U_alpha(self.target, A_c,P_c, self.mask, self.S,self.ones_box, fb,fbmix)
        #else:
        #    self.stepstartflag=1
        U_alpha_boxed = ((1-fbmix) * U_alpha + fbmix * A_c * cp.exp(1j * P_c)) * self.ones_box + A_c * cp.exp(1j * P_c) * (
                    self.I - self.ones_box)
        if False:
            plt.imshow( np.abs(U_alpha_boxed.get()))
            plt.title(" U_alpha_boxed")
        # u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha), norm="ortho"))
        print("fft starttime",time.time())
        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha_boxed), norm="ortho"))
        print("fft end time", time.time())
        # zoomfft_Func=ZoomFFT2D(self.size, self.size, (0,0), (2,2), None, direction="forward")
        # u_alpha = cp.asarray(zoomfft_Func(U_alpha.get()))
        # u_alpha =u_alpha/cp.max(cp.abs(u_alpha))
        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(np.abs(u_alpha.get()))
        # axs[0].set_title("u alpha orig")
        # plt.show()
        # axs[1].imshow(np.abs(uf_alpha.get()))
        # axs[1].set_title("u alpha fft")
        # plt.show()
        p_alpha = cp.angle(u_alpha)

        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S) + P_c * self.S
        U_beta = A_beta * cp.exp(1j * P_beta)

        if self.stepstartflag != 0:
            #U_beta = A_beta * cp.exp(1j * P_beta) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (self.I - self.mask)#Used so far 16/12/2025
            U_beta = A_beta * cp.exp(1j * P_beta)*self.mask - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask) #Use it to negate whatever output in the same round, more effective #Use fb=1.0 for this
            # U_beta= np.multiply(U_beta,cp.ones(self.size))-fb*self.U_beta_old*(cp.ones(self.size)-self.mask) #MDS added
            #U_beta_boxed = make_U_beta(self.target, A_c,P_c, self.mask, self.S, self.ones_box, fb,fbmix)
        else:
            self.stepstartflag = 1
        U_beta_boxed = ((1.0-fbmix) * U_beta + fbmix * A_c * cp.exp(1j * P_c)) * self.ones_box + A_c * cp.exp(1j * P_c) * (
                    self.I - self.ones_box)
        if False:#self.stepnum==0:
            plt.figure()
            plt.imshow( np.abs(U_beta_boxed.get()))
            plt.title(" U_beta_boxed")
            plt.show()
        self.A_c_old = A_c
        self.P_c_old = P_c
        # u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta), norm="ortho"))
        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta_boxed), norm="ortho"))

        # zoomfft_Func=ZoomFFT2D(self.size, self.size, (0,0), (2,2), None, direction="forward")
        # u_beta = cp.asarray(zoomfft_Func(U_beta.get()))
        # u_beta =u_beta/cp.max(cp.abs(u_beta))
        # fig, axs = plt.subplots(2, 1)
        # axs[0].imshow(np.abs(u_beta.get()))
        # axs[0].set_title("u_beta orig")
        # plt.show()
        # axs[1].imshow(np.abs(uf_beta.get()))
        # axs[1].set_title("u _beta fft")
        # plt.show()

        p_beta = cp.angle(u_beta)

        self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2))  # Used so far
        #self.image_field *= cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2))
        if False:
            print("Target sum", cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2)))
            print("image sum", cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2)))

        # plt.imshow(np.abs(self.image_field.get()))
        # plt.title("image after")
        # plt.show()
        # original
        # self.totalsteperr.append(cp.average(cp.abs(cp.abs(self.image_field * ones_box) - cp.abs(self.A_t * ones_box))))
        # self.totalsteperr.append(cp.average(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)-cp.abs((self.A_t/cp.max(self.A_t))*ones_box))))
        # Bowman definition has outside of box as well
        #self.totalsteperr.append((cp.sum(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * self.image_field)*ones_box)) / (
        #    cp.sum((cp.abs(self.A_t) ** 2)*ones_box)))
        image_field_insidebox = self.image_field / cp.sqrt(cp.sum(cp.abs(self.image_field * self.ones_box) ** 2))
        if False:
            print("full fidelity:",(cp.sum(cp.abs(cp.conj(self.image_field) * self.image_field)*self.ones_box)) / (
                cp.sum((cp.abs(self.image_field) ** 2)*self.ones_box)) )

            print("fidelity inside box:",(cp.sum(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * image_field_insidebox)*self.ones_box)) / (
                cp.sum((cp.abs(self.A_t) ** 2)*self.ones_box)) )
            print("fidelity inside box_orig:",(cp.sum(cp.abs(cp.conj(self.A_t_orig * cp.exp(1j * self.P_t_orig)) * image_field_insidebox)*self.ones_box)) / (
                cp.sum((cp.abs(self.A_t_orig) ** 2)*self.ones_box)) )
        # Bowman definition only inside the box
        #self.totalsteperr.append((cp.sum(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * image_field_insidebox)*ones_box)) / (
        #    cp.sum((cp.abs(self.A_t) ** 2)*ones_box)))
        self.totalsteperr.append((cp.sum(cp.abs(cp.conj(self.A_t_orig * cp.exp(1j * self.P_t_orig)) * image_field_insidebox)*self.ones_box)) / (
            cp.sum((cp.abs(self.A_t_orig) ** 2)*self.ones_box)))
        #print("totalerror", self.totalsteperr[-1])



        # plt.imshow(cp.abs(self.image_field).get())
        # plt.title("imagefield plain")
        # plt.show()
        # plt.imshow(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)).get())
        # plt.title("imagefield")
        # plt.show()
        # plt.imshow(cp.abs(cp.abs((self.A_t/cp.max(self.A_t))*ones_box)).get())
        # plt.title("Target A_t")
        # plt.show()
        # plt.imshow(cp.abs(cp.abs((self.image_field/cp.max(self.image_field))*ones_box)-cp.abs((self.A_t/cp.max(self.A_t))*ones_box)).get())
        # plt.title("totalsteperror")
        # plt.show()


        if False:#np.mod(self.stepnum,10)==0:
            plt.imshow(cp.abs(self.image_field).get())
            plt.title("image")
            plt.show()
            plt.imshow(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t))).get())
            plt.title("target")
            plt.show()
            plt.imshow(cp.abs(cp.conj(self.A_t * cp.exp(1j * self.P_t)) * self.image_field).get())
            plt.title("fidelity")
            plt.show()

    import torch
    import time

    def step_updatedloop_torch(self):
        # 1. Setup Feedback Constants
        if (self.stepnum // 2) % 2 == 0:
            #fb, fbmix = 0.2+0.6*(self.outer_num/100.0), 0.6+0.1*(self.outer_num/100.0)
            fb, fbmix= runsettings.fb_global ,runsettings.fbmix_global#0.9-0.6*(self.outer_num/50.0), 0.4+0.2*(self.outer_num/50.0)
        else:
            #fb, fbmix = 0.2+0.6*(self.outer_num/100.0), 0.6+0.1*(self.outer_num/100.0)
            fb, fbmix = runsettings.fb_global,runsettings.fbmix_global#0.9-0.6*(self.outer_num/50.0), 0.4+0.2*(self.outer_num/50.0)

        # 2. Extract Amplitude and Phase
        U_c = self.image_field
        A_c = torch.abs(U_c)
        P_c = torch.angle(U_c)

        # Use persistent phase if enabled
        if self.phaseconstant and self.P_c_old is not None and self.P_c_old.numel() > 0:
            P_c = self.P_c_old
        # if self.stepnum==10:
        #     plt.imshow(self.A_t.abs().detach().cpu().numpy())
        #     plt.show()

        # 3. Calculate Alpha Field (Target Intensity + Current Phase)
        # A_alpha, P_alpha are real-valued tensors
        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S + P_c * (self.I - self.S)

        # torch.polar is the high-performance equivalent of A * exp(1j * P)
        U_alpha = torch.polar(A_alpha.real, P_alpha.real)

        if True:#self.stepstartflag != 0:
            # Apply mask-based feedback
            U_alpha = (U_alpha * self.mask) - (fb * torch.polar(A_c.real, P_c.real) * (self.I - self.mask))

        # Box Constraints
        U_alpha_boxed = ((1.0 - fbmix) * U_alpha + fbmix * torch.polar(A_c.real, P_c.real)) * self.ones_box + \
                        torch.polar(A_c.real, P_c.real) * (self.I - self.ones_box)

        # 4. Calculate Beta Field
        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S) + P_c * self.S
        U_beta = torch.polar(A_beta.real, P_beta.real)

        if True:#self.stepstartflag != 0:
            U_beta = (U_beta * self.mask) - (fb * torch.polar(A_c.real, P_c.real) * (self.I - self.mask))
        else:
            self.stepstartflag = 1

        U_beta_boxed = ((1.0 - fbmix) * U_beta + fbmix * torch.polar(A_c.real, P_c.real)) * self.ones_box + \
                       torch.polar(A_c.real, P_c.real) * (self.I - self.ones_box)

        # Cache for next iteration
        self.A_c_old = A_c
        self.P_c_old = P_c

        # 5. Batched IFFT (Compute Alpha and Beta back-propagation in one GPU call)
        # This is 2x faster than doing them separately
        U_stack = torch.stack([U_alpha_boxed, U_beta_boxed])
        # PyTorch FFTs require fftshift/ifftshift manual handling if not using centered grids
        u_stack = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U_stack, dim=(-2, -1)), norm="ortho"),
                                      dim=(-2, -1))

        p_alpha = torch.angle(u_stack[0])
        p_beta = torch.angle(u_stack[1])

        # 6. Combine and Propagate
        # SLM Phase = angle(exp(j*alpha) + exp(j*beta))
        self.p = torch.angle(torch.polar(torch.ones_like(p_alpha).real, p_alpha.real) +
                             torch.polar(torch.ones_like(p_beta).real, p_beta.real))

        self.slm_field = self.input * torch.polar(torch.ones_like(self.p).real, self.p.real)

        # Forward FFT to Image Plane
        self.image_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(self.slm_field), norm="ortho"))

        # Normalize Energy
        self.image_field /= torch.sqrt(torch.sum(torch.abs(self.image_field) ** 2))

        # 7. Error / Fidelity Calculation
        if np.mod(self.stepnum, 10) == 0:
            with torch.no_grad():
                # image_field_insidebox normalization
                denom_box = torch.sqrt(torch.sum(torch.abs(self.image_field * self.ones_box) ** 2))
                image_field_insidebox = self.image_field / denom_box

                # Fidelity calculation based on your "A_t_orig" logic
                target_complex = torch.polar(self.A_t_orig.real, self.P_t_orig.real)
                num = torch.sum(torch.abs(torch.conj(target_complex) * image_field_insidebox) * self.ones_box)
                den = torch.sum((torch.abs(self.A_t_orig) ** 2) * self.ones_box)

                #self.totalsteperr.append((num / den).item())

                self.totalsteperr.append((torch.abs(torch.sum(torch.conj(self.image_field) * torch.as_tensor(self.target_orig[0]) * self.ones_box)) ** 2 / (
                            (torch.abs(torch.sum(torch.conj(self.image_field) * self.image_field * self.ones_box))) * torch.abs(
                        torch.sum(torch.conj(torch.as_tensor(self.target_orig[0])) * torch.as_tensor(self.target_orig[0]) * self.ones_box)))).item())

                efficiency=torch.sum(torch.abs(torch.conj(self.image_field) * self.image_field) * self.mask)/torch.sum(torch.abs(torch.conj(self.image_field) * self.image_field))
                self.eff.append(efficiency.detach().cpu().item())
            if np.mod(self.stepnum,10)==0:
                #plt.imshow(self.image_field.abs().detach().cpu().numpy())
                #plt.title("image_field")
                # plt.figure()
                #
                # plt.imshow((self.A_t).abs().detach().cpu().numpy())
                # plt.title("target")
                # plt.show()
                print("error {:.3f} eff {:.3f}".format(self.totalsteperr[-1],self.eff[-1]))


    def step_fbchanging(self):
        if np.mod(int(self.stepnum / 2), 2) == 0:
            fb = self.fb_initial  # Use dynamic feedback scaling
        else:
            fb = self.fb_initial  # Same for now, but could change dynamically based on conditions

        cp._default_memory_pool.free_all_blocks()
        U_c = self.image_field

        # Normalize image and amplitude to prevent excessive scaling issues during optimization
        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2))  # Normalize image
        self.image_field *= cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2))  # Scale by target amplitude

        # Compute the amplitude and phase of the current field
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)

        ones_box = cp.zeros(A_c.shape)
        xbox_min = int(A_c.shape[0] / 2 - A_c.shape[0] / 4)
        xbox_max = int(A_c.shape[0] / 2 + A_c.shape[0] / 4)
        ybox_min = int(A_c.shape[1] / 2 - A_c.shape[1] / 4)
        ybox_max = int(A_c.shape[1] / 2 + A_c.shape[1] / 4)

        ones_box[xbox_min:xbox_max, ybox_min:ybox_max] = 1

        # Update amplitude and phase using the target and feedback mechanism
        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S + P_c * (self.I - self.S)
        U_alpha = A_alpha * cp.exp(1j * P_alpha)

        if self.stepstartflag != 0:
            # Adjust U_alpha with feedback
            U_alpha = A_alpha * cp.exp(1j * P_alpha) * self.mask - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask)
        else:
            self.stepstartflag = 1

        U_alpha_boxed = (0.8 * U_alpha + 0.2 * A_c * cp.exp(1j * P_c)) * ones_box + A_c * cp.exp(1j * P_c) * (
                    self.I - ones_box)

        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha_boxed), norm="ortho"))

        p_alpha = cp.angle(u_alpha)

        # Update the beta field
        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S) + P_c * self.S
        U_beta = A_beta * cp.exp(1j * P_beta)

        if self.stepstartflag != 0:
            U_beta = A_beta * cp.exp(1j * P_beta) * self.mask - fb * A_c * cp.exp(1j * P_c) * (self.I - self.mask)
        else:
            self.stepstartflag = 1

        U_beta_boxed = (0.8 * U_beta + 0.2 * A_c * cp.exp(1j * P_c)) * ones_box + A_c * cp.exp(1j * P_c) * (
                    self.I - ones_box)

        self.A_c_old = A_c
        self.P_c_old = P_c

        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta_boxed), norm="ortho"))

        p_beta = cp.angle(u_beta)

        # Now combine both phases (alpha and beta) to update the phase
        self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))

        # Update the SLM field with the new phase
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        # Normalize the field after the update
        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2))  # Normalize image again
        self.image_field *= cp.sqrt(cp.sum(cp.abs(self.A_t) ** 2))  # Scale by target amplitude again

        # Calculate error using L2 norm for both amplitude and phase
        amplitude_error = cp.sum(cp.abs(cp.abs(self.image_field) - cp.abs(self.A_t)) ** 2)
        phase_error = cp.sum(cp.angle(self.image_field) - cp.angle(self.A_t)) ** 2

        total_error = amplitude_error + phase_error

        self.totalsteperr.append(total_error)

        print("Amplitude Error:", amplitude_error)
        print("Phase Error:", phase_error)
        print("Total Error:", total_error)

        # Optionally, visualize results at different steps
        if self.stepnum % 10 == 0:  # Only visualize every 10 steps to save computation
            plt.imshow(cp.abs(self.image_field).get())
            plt.title("Image Amplitude")
            plt.show()

            plt.imshow(cp.angle(self.image_field).get())
            plt.title("Image Phase")
            plt.show()

            plt.imshow(cp.abs(self.slm_field).get())
            plt.title("SLM Field")
            plt.show()

        return total_error  # You can return error for optimization tracking

    def step_GSW(self): #only amplitude control
        # old_p=self.p

        if np.mod(int(self.stepnum / 4), 2) == 0:
            fb = 0.00  # MDS added
        else:
            fb = 0.0
        cp._default_memory_pool.free_all_blocks()
        U_c = self.image_field
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)
        ones_box = cp.zeros(A_c.shape)
        xbox_min = int(A_c.shape[0] / 2 - A_c.shape[0] / 4)
        xbox_max = int(A_c.shape[0] / 2 + A_c.shape[0] / 4)
        ybox_min = int(A_c.shape[1] / 2 - A_c.shape[1] / 4)
        ybox_max = int(A_c.shape[1] / 2 + A_c.shape[1] / 4)

        ones_box[xbox_min:xbox_max, ybox_min:ybox_max] = 1
        #ones_box = cp.ones(A_c.shape)

        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S*self.mask + P_c * self.S*(self.I-self.mask)+ P_c * (self.I - self.S)
        U_alpha = A_alpha * cp.exp(1j * P_alpha)
        if self.stepstartflag != 0:
            print("feedback")
            U_alpha = A_alpha * cp.exp(1j * P_alpha) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (
                        self.I - self.mask)

        U_alpha_boxed = U_alpha * ones_box + A_c * cp.exp(1j * P_c) * (self.I - ones_box)
        # u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha), norm="ortho"))
        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha_boxed), norm="ortho"))
        p_alpha = cp.angle(u_alpha)

        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S)*self.mask + P_c * (self.I - self.S)*(self.I-self.mask)+ P_c * self.S
        U_beta = A_beta * cp.exp(1j * P_beta)

        if self.stepstartflag != 0:
            U_beta = A_beta * cp.exp(1j * P_beta) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (self.I - self.mask)
        else:
            self.stepstartflag = 1
        U_beta_boxed = U_beta * ones_box + A_c * cp.exp(1j * P_c) * (self.I - ones_box)
        self.A_c_old = A_c
        self.P_c_old = P_c
        # u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta), norm="ortho"))
        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta_boxed), norm="ortho"))

        p_beta = cp.angle(u_beta)
        #GSW
        u_outGSW= cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(self.A_t*cp.exp(1j * P_c)), norm="ortho"))

        #self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))
        #GSW
        self.p=cp.angle(u_outGSW)
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2))

        # plt.imshow(np.abs(self.image_field.get()))
        # plt.title("image after")
        # plt.show()
        self.totalsteperr.append(cp.average(cp.abs(cp.abs(self.image_field) - cp.abs(self.A_t))))
        print("totalerror", self.totalsteperr[-1])
        # plt.show(block=False)
        # plt.pause(1)
        # plt.imshow(np.angle(self.slm_field.get()))
        # plt.title("slm after")

        # plt.show()
    def step_GradientDescent(self):  # original
        # old_p=self.p

        if np.mod(int(self.stepnum / 4), 2) == 0:
            fb = 0.00  # MDS added
        else:
            fb = 0.0
        cp._default_memory_pool.free_all_blocks()
        U_c = self.image_field
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)
        ones_box = cp.zeros(A_c.shape)
        xbox_min = int(A_c.shape[0] / 2 - A_c.shape[0] / 6)
        xbox_max = int(A_c.shape[0] / 2 + A_c.shape[0] / 6)
        ybox_min = int(A_c.shape[1] / 2 - A_c.shape[1] / 6)
        ybox_max = int(A_c.shape[1] / 2 + A_c.shape[1] / 6)

        ones_box[xbox_min:xbox_max, ybox_min:ybox_max] = 1
        #ones_box = cp.ones(A_c.shape)

        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S*self.mask + P_c * self.S*(self.I-self.mask)+ P_c * (self.I - self.S)
        U_alpha = A_alpha * cp.exp(1j * P_alpha)
        if self.stepstartflag != 0:
            print("feedback")
            U_alpha = A_alpha * cp.exp(1j * P_alpha) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (
                        self.I - self.mask)

        U_alpha_boxed = U_alpha * ones_box + A_c * cp.exp(1j * P_c) * (self.I - ones_box)
        # u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha), norm="ortho"))
        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha_boxed), norm="ortho"))
        p_alpha = cp.angle(u_alpha)

        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S)*self.mask + P_c * (self.I - self.S)*(self.I-self.mask)+ P_c * self.S
        U_beta = A_beta * cp.exp(1j * P_beta)

        if self.stepstartflag != 0:
            U_beta = A_beta * cp.exp(1j * P_beta) - fb * self.A_c_old * cp.exp(1j * self.P_c_old) * (self.I - self.mask)
        else:
            self.stepstartflag = 1
        U_beta_boxed = U_beta * ones_box + A_c * cp.exp(1j * P_c) * (self.I - ones_box)
        self.A_c_old = A_c
        self.P_c_old = P_c
        # u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta), norm="ortho"))
        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta_boxed), norm="ortho"))

        p_beta = cp.angle(u_beta)
        #GSW
        #u_outGSW= cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(self.A_t*cp.exp(1j * P_c)), norm="ortho"))

        self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))
        #GSW
        #self.p=cp.angle(u_outGSW)
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2))

        # plt.imshow(np.abs(self.image_field.get()))
        # plt.title("image after")
        # plt.show()
        self.totalsteperr.append(cp.average(cp.abs(cp.abs(self.image_field) - cp.abs(self.A_t))))
        print("totalerror", self.totalsteperr[-1])
        # plt.show(block=False)
        # plt.pause(1)
        # plt.imshow(np.angle(self.slm_field.get()))
        # plt.title("slm after")

        # plt.show()
    def step_MDS_cross(self): #cross_MDS
        U_c = self.image_field
        # if self.res_factor != 1:
        #     U_c = pad_border(U_c, U_c.shape * 2)
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)
        
        #A_alpha = self.A_t * self.S1 + A_c * self.S2 + self.A_t * self.S3 + A_c * self.S4
        #P_alpha = self.P_t * self.S1 + P_c * self.S2 + self.P_t * self.S3 + P_c * self.S4
        #A_alpha = self.A_t * self.S1 + self.A_t * self.S2 + A_c * self.S3 + A_c * self.S4
        #P_alpha = self.P_t * self.S1 + P_c * self.S2 + self.P_t * self.S3 + P_c * self.S4
        A_alpha = self.A_t * self.S1 + A_c * self.S2 + A_c * self.S3 + self.A_t * self.S4#)*self.Sin+ A_c *self.Sout#+A_c * self.SO1# + self.A_t * self.S5+ self.A_t * self.S6
        P_alpha = (self.P_t * self.S1 + P_c * self.S2 + P_c * self.S3 + self.P_t * self.S4)*self.Sin+ P_c *self.Sout   # +P_c * self.SO1# + P_c * self.S5 + P_c * self.S6
        #A_alpha = self.A_t * self.S1 + A_c * self.S2 + self.A_t * self.S3 + A_c * self.S4
        #P_alpha = P_c * self.S1 + self.P_t * self.S2 + P_c * self.S3 + self.P_t * self.S4
        U_alpha = A_alpha * cp.exp(1j * P_alpha)
        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha), norm="ortho"))
        p_alpha = cp.angle(u_alpha)

        #A_beta = A_c * self.S1 + self.A_t * self.S2 + A_c * self.S3 + self.A_t * self.S4
        #P_beta = P_c * self.S1 + self.P_t * self.S2 + P_c * self.S3 + self.P_t * self.S4
        #A_beta = A_c * self.S1 + A_c * self.S2 + self.A_t * self.S3 + self.A_t * self.S4
        #P_beta = P_c * self.S1 + self.P_t * self.S2 + P_c * self.S3 + self.P_t * self.S4
        A_beta = A_c * self.S1 + self.A_t * self.S2 + self.A_t * self.S3 + A_c * self.S4#)*self.Sin+ A_c *self.Sout # +A_c * self.SO1# + self.A_t * self.S5+ self.A_t * self.S6
        P_beta = (P_c * self.S1 + self.P_t * self.S2 + self.P_t * self.S3 + P_c * self.S4)*self.Sin+ P_c *self.Sout# +P_c * self.SO1# + P_c * self.S5 + P_c * self.S6
        #A_beta = A_c * self.S1 + self.A_t * self.S2 + A_c * self.S3 + self.A_t * self.S4
        #P_beta = self.P_t * self.S1 + P_c * self.S2 + self.P_t * self.S3 + P_c * self.S4
        U_beta = A_beta * cp.exp(1j * P_beta)
        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta), norm="ortho"))
        p_beta = cp.angle(u_beta)

        self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field) ** 2))

    # Execute the algorithm for N iterations
    def iterate(self, N):
        self.stepnum=0
        self.fb_initial = 1.0
        for n in range(N):
            print(str(n) + ' ', end='')
            cp._default_memory_pool.free_all_blocks()
            p_temp = self.p
            slm_temp = self.slm_field
            im_temp = self.image_field
            print(n, "step start:",str(time.time()))
            self.step()
            print(n, "step end:", str(time.time()))
            #self.step_GSW()
            self.stepnum+=1
            #self.step_MDS_cross()
            #self.eff.append(float(cp.asnumpy(self.eta())))
            self.eff.append(self.eta())
            print("shape trage",self.target)
            if True:
                image_field_box=cp.multiply(self.image_field,self.ones_box)
                target_updated = cp.where(self.mask == 1,
                                             self.target_orig[0] * (self.target / (image_field_box + 1e-8)) ** (0.25),
                                             self.target_orig[0])

                # target_updated[0]=target_orig_box*(target_updated[0]/(image_field_box+ 1e-8))**(0.4)#*(target_updated[0]
                target_updated *= np.sqrt(
                    np.sum(np.abs(self.target_orig[0]) ** 2) / np.sum(np.abs(target_updated) ** 2))
                self.target=target_updated
                self.A_t = cp.abs(self.target)
                self.P_t = cp.angle(self.target)
                #Do phase and amplitude separately

                #start_phase=wu.p

            # Keep track of the phase and amplitude errors/nonuniformities
            if self.array:
                #self.nonunif.append(float(cp.asnumpy(self.dev_amp(waist=0.001, target=self.target_amp))))
                #self.phase_err.append(float(cp.asnumpy(self.dev_phase(waist=0.001, target=self.target_phase))))
                self.nonunif.append(self.dev_amp(waist=0.001, target=self.target_amp))#original changed by MDS
                self.phase_err.append(self.dev_phase(waist=0.001, target=self.target_phase))#original changed by MDS
                #self.nonunif.append(cp.mean(cp.abs(cp.abs(self.beams(waist=0.0015))-self.amps())))
                #self.phase_err.append(cp.mean(cp.abs(self.beams_phase(waist=0.0005)-self.phases())))
            else:
                #self.nonunif.append(float(cp.asnumpy(self.nonuniformity())))
                #self.phase_err.append(float(cp.asnumpy(self.phase_error())))
                self.nonunif.append(self.nonuniformity())
                self.phase_err.append(self.phase_error())

            if self.phase_memory and 0.01 > self.nonunif[-1] > self.nonunif[-2]:
                self.p = p_temp
                self.slm_field = slm_temp
                self.image_field = im_temp
                print('Local minimum in amplitude convergence', end='')
                #break
        #ot(cp.array(self.totalsteperr).get())
        #plt.title("totalsteperr")
        #plt.show()
        # self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

    def iterate_updatedloopWu(self, N):
        self.stepnum = 0
        #self.totalsteperr = []

        # 1. --- ONE-TIME CONVERSION ---
        # Convert all physics constants to Tensors before the loop
        # This prevents the CPU from re-uploading them to GPU at every step
        with torch.no_grad():
            vars_to_convert = [
                'image_field', 'A_t', 'S', 'I', 'P_t', 'mask',
                'ones_box', 'input', 'A_t_orig', 'P_t_orig', 'P_c_old','ion_mask','ion_mask_small'
            ]
            for var in vars_to_convert:
                val = getattr(self, var, None)
                if val is not None:
                    # Use as_tensor to handle CuPy or NumPy inputs
                    tensor_val = torch.as_tensor(val, device=self.device,
                                                 dtype=torch.float32 if "P_" in var or var in ['S', 'I', 'mask',
                                                                                               'ones_box','ion_mask','ion_mask_small'] else torch.complex64)
                    setattr(self, var, tensor_val)



            # 2. --- ITERATIVE LOOP ---
            for n in range(N):
                # Only sync and time if you are specifically debugging; otherwise, remove for speed
                # torch.cuda.synchronize()
                self.step_updatedloop_torch()
                self.stepnum += 1
            # 1. Apply the box mask
            print("type",type(self.outer_num))
            #if self.outer_num == 0:
            A_t_exponent=runsettings.exp_amp_global_wu#max(0.15+self.outer_num*0.002,0.05)
            A_t_exponent_diff=runsettings.exp_amp_global_diff_wu
            # if self.outer_num<50:
            P_t_exponent=runsettings.exp_phase_global_wu #max(0.15+self.outer_num*0.002,0.05)
            # else:
            #     P_t_exponent=0.0
            image_field_box = self.image_field * self.ones_box
            print("type ion_mask",type(self.ion_mask))
            image_field_mask = (self.image_field * self.ion_mask)#self.mask)
            image_field_mask =image_field_mask *torch.sqrt(torch.sum(torch.abs(self.A_t_orig)**2)/torch.sum(torch.abs(image_field_mask)**2))
            #self.target[0] = torch.where( self.mask == 1,self.target_orig_box * (self.target[0] / (torch.abs(image_field_box) + 1e-8)) ** 0.25,self.target_orig_box)
            #self.A_t=torch.where( self.mask == 1,self.A_t * (self.A_t_orig / (torch.abs(image_field_box) + 1e-18)) ** (0.1*(1-self.outer_num*(1/100))),self.A_t_orig) #Decreasing feedback
            #self.A_t=torch.where( self.mask == 1,self.A_t * (self.A_t_orig / (torch.abs(image_field_box) + 1e-18)) ** (0.9),self.A_t_orig) #Decreasing feedback with box
            #uncomment below
            # self.A_t = torch.where(self.mask == 1,
            #                        self.A_t * (self.A_t_orig / (torch.abs(image_field_mask) + 1e-18)) ** (A_t_exponent),
            #                        self.A_t_orig)

            # epsW = 0.02  # threshold (tune this)
            # magW = torch.abs(image_field_mask)
            # alphaW = magW / (magW + epsW)  # smoothly goes 0 → 1
            #
            # ratio_updateW = self.A_t * (self.A_t_orig / (magW + 1e-6)) ** A_t_exponent
            # diff_updateW = self.A_t +  (self.A_t_orig - magW)*0.15#*A_t_exponent
            #
            # self.A_t =alphaW * ratio_updateW + (1 - alphaW) * diff_updateW
            # self.A_t = torch.where(self.ion_mask == 1,
            #                        alphaW * ratio_updateW + (1 - alphaW) * diff_updateW,
            #                        self.A_t_orig)

            mag = torch.abs(image_field_mask)
            eps = 1e-6

            # smooth blending (ratio ↔ difference)
            alpha = mag / (mag + 0.02)

            ratio_update = self.A_t * (self.A_t_orig / (mag + eps)) ** A_t_exponent
            diff_update = self.A_t + (self.A_t_orig - mag)*A_t_exponent_diff

            A_new = alpha * ratio_update + (1 - alpha) * diff_update

            gamma_out = runsettings.wu_gamma_out#0.8#0.85  # suppress outside-mask energy
            beta_in = runsettings.wu_beta_in#1.2  # boost inside-mask correction

            A_new = torch.where(self.mask == 1,
                                A_new * beta_in,
                                A_new * gamma_out)
            orig_norm = torch.norm(self.A_t_orig)
            current_norm = torch.norm(A_new)

            A_new *= (orig_norm / (current_norm + 1e-8)) ** 0.5
            self.A_t = A_new




            # self.P_t = torch.where(self.mask == 1, self.P_t_orig + (self.P_t - (torch.angle(image_field_box))) * 0.00,
            #                      self.P_t_orig)
            print("A_t_orig_max",torch.max(torch.abs(self.A_t_orig)))
            # testing=torch.where(torch.abs(self.A_t_orig)>0.02,torch.abs(self.A_t_orig),0)
            # plt.imshow(testing.detach().cpu().numpy())
            # plt.show()
            # self.P_t = torch.where(torch.abs(self.A_t_orig)>0.02, self.P_t + (self.P_t_orig - (torch.angle(image_field_box))) * 0.01,
            #                      self.P_t_orig)

            mask_phasefb = torch.abs(self.A_t_orig) > 0.01

            phase_est_phasefb = torch.angle(image_field_box)

            delta_phasefb = self.P_t_orig - phase_est_phasefb

            # global phase offset via complex correlation
            global_offset_phasefb = torch.angle(
                torch.sum(torch.exp(1j * delta_phasefb)[mask_phasefb])
            )

            phase_diff_phasefb = torch.angle(
                torch.exp(1j * (delta_phasefb - global_offset_phasefb))
            )

            self.P_t = torch.where(
                mask_phasefb,
                self.P_t + (P_t_exponent) * phase_diff_phasefb,
                self.P_t_orig
            )



            orig_norm = torch.norm(self.A_t_orig)
            current_norm = torch.norm(self.A_t)
            self.A_t *= (orig_norm / (current_norm + 1e-18))

            # self.target[0]=self.A_t*torch.exp(1j*self.P_t)
            #
            # # 3. Re-normalize the intensity to match the original energy
            # # This keeps the target values from drifting during the feedback iterations
            # orig_energy = torch.sum(torch.abs(self.target_orig[0]) ** 2)
            # current_energy = torch.sum(torch.abs(self.target[0]) ** 2)
            #
            # # Apply normalization factor in-place
            # self.target[0] *= torch.sqrt(orig_energy / (current_energy + 1e-8))


        # 3. --- CONVERT BACK TO CUPY ---
        # Only do this once after all N iterations are done

        self.image_field = cp.asarray(self.image_field)
        self.slm_field = cp.asarray(self.slm_field)
        self.p = cp.asarray(self.p)

    def iterate_old_recovery(self, N):
        self.stepnum = 0
        #self.totalsteperr = []

        # 1. --- ONE-TIME CONVERSION ---
        # Convert all physics constants to Tensors before the loop
        # This prevents the CPU from re-uploading them to GPU at every step
        with torch.no_grad():
            vars_to_convert = [
                'image_field', 'A_t', 'S', 'I', 'P_t', 'mask',
                'ones_box', 'input', 'A_t_orig', 'P_t_orig', 'P_c_old'
            ]
            for var in vars_to_convert:
                val = getattr(self, var, None)
                if val is not None:
                    # Use as_tensor to handle CuPy or NumPy inputs
                    tensor_val = torch.as_tensor(val, device=self.device,
                                                 dtype=torch.float32 if "P_" in var or var in ['S', 'I', 'mask',
                                                                                               'ones_box'] else torch.complex64)
                    setattr(self, var, tensor_val)

            # 2. --- ITERATIVE LOOP ---
            for n in range(N):
                # Only sync and time if you are specifically debugging; otherwise, remove for speed
                # torch.cuda.synchronize()
                self.step_updatedloop_torch()
                self.stepnum += 1
            # 1. Apply the box mask
            image_field_box = self.image_field * self.ones_box
            image_field_mask = (self.image_field * self.mask)
            image_field_mask =image_field_mask *torch.sqrt(torch.sum(torch.abs(self.A_t_orig)**2)/torch.sum(torch.abs(image_field_mask)**2))
            if False: #old feedbackon outer_updated
                #self.target[0] = torch.where( self.mask == 1,self.target_orig_box * (self.target[0] / (torch.abs(image_field_box) + 1e-8)) ** 0.25,self.target_orig_box)
                #self.A_t=torch.where( self.mask == 1,self.A_t * (self.A_t_orig / (torch.abs(image_field_box) + 1e-18)) ** (0.1*(1-self.outer_num*(1/100))),self.A_t_orig) #Decreasing feedback
                #self.A_t=torch.where( self.mask == 1,self.A_t * (self.A_t_orig / (torch.abs(image_field_box) + 1e-18)) ** (0.9),self.A_t_orig) #Decreasing feedback with box
                self.A_t = torch.where(self.mask == 1,
                                       self.A_t * (self.A_t_orig / (torch.abs(image_field_mask) + 1e-18)) ** (0.15),
                                       self.A_t_orig)

                self.P_t = torch.where(self.mask == 1, self.P_t_orig + (self.P_t - (torch.angle(image_field_box))) * 0.00,
                                     self.P_t_orig)



                # orig_norm = torch.norm(self.A_t_orig)
                # current_norm = torch.norm(self.A_t)
                # self.A_t *= (orig_norm / (current_norm + 1e-18))

            # self.target[0]=self.A_t*torch.exp(1j*self.P_t)
            #
            # # 3. Re-normalize the intensity to match the original energy
            # # This keeps the target values from drifting during the feedback iterations
            # orig_energy = torch.sum(torch.abs(self.target_orig[0]) ** 2)
            # current_energy = torch.sum(torch.abs(self.target[0]) ** 2)
            #
            # # Apply normalization factor in-place
            # self.target[0] *= torch.sqrt(orig_energy / (current_energy + 1e-8))


        # 3. --- CONVERT BACK TO CUPY ---
        # Only do this once after all N iterations are done

        self.image_field = cp.asarray(self.image_field)
        self.slm_field = cp.asarray(self.slm_field)
        self.p = cp.asarray(self.p)





    def iterate_Gradient(self, N, wuamps0,wuphases0,target_phase_curve):
        self.stepnum = 0
        #self.totalsteperr = []
        loss_track=[]
        plot_loss_overlap_complex=[]
        plot_err_mean=[]
        plot_eff_minus1=[]

        # 1. --- ONE-TIME CONVERSION ---
        # Convert all physics constants to Tensors before the loop
        # This prevents the CPU from re-uploading them to GPU at every step
        with torch.no_grad():
            vars_to_convert = [
                'image_field', 'A_t', 'S', 'I', 'P_t', 'mask',
                'ones_box', 'input', 'A_t_orig', 'P_t_orig', 'P_c_old','ion_mask','ion_mask_small'
            ]
            for var in vars_to_convert:
                val = getattr(self, var, None)
                if val is not None:
                    # Use as_tensor to handle CuPy or NumPy inputs
                    tensor_val = torch.as_tensor(val, device=self.device,
                                                 dtype=torch.float32 if "P_" in var or var in ['S', 'I', 'mask',
                                                                                               'ones_box'] else torch.complex64)
                    setattr(self, var, tensor_val)





            # 2. --- ITERATIVE LOOP ---
        lr = runsettings.learning_rate # 5
        epochs = 2000
        target_loss=0.1e-1#+ 5e-1#*0.005
        patience=2000
        best_loss=float('inf')
        patience_counter=0
        phi_slm = torch.tensor(self.p,device=self.device, dtype=torch.float32,requires_grad=True )
        # plt.imshow(torch.abs(phi_slm).detach().cpu().numpy())
        # plt.title("phi_slm")
        # plt.show()
        A_target_amp=torch.as_tensor(self.A_t_orig,device=self.device, dtype=torch.float32)
        P_target_phase=torch.as_tensor(self.P_t_orig,device=self.device, dtype=torch.float32)
        A_in=torch.as_tensor(cp.abs(self.input),device=self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam([phi_slm], lr=lr)

        target_normalised = self.target_orig[0] / torch.sqrt(torch.abs(
            torch.sum(torch.conj(
                torch.as_tensor(self.target_orig[0])) * torch.as_tensor(
                self.target_orig[0]) * self.ones_box)))

        amps_target_orig = (
            [torch.abs(torch.as_tensor(target_normalised[int(np.round(spotnum[0])), int(np.round(spotnum[1]))])) for
             spotnum in
             self.spots])
        amps_target_orig = torch.as_tensor(amps_target_orig, device=self.device)
        amps_target_orig = amps_target_orig / torch.max(amps_target_orig)
        print("self.spots",type(self.spots),self.spots)
        # Use cp.atleast_1d to ensure they aren't "unsized" 0-rank objects
        spots_tensor = torch.stack([torch.as_tensor(cp.atleast_1d(a), device=self.device) for a in self.spots])

        for epoch in range(epochs):
            # if epoch>500:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 0.004
            # if epoch>1000:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 0.004
            optimizer.zero_grad()

            U_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A_in * torch.exp(1j * phi_slm))))

            U_out = U_out / (self.size[0] * self.size[1])  # normalize FFT
            U_out = U_out / torch.sqrt(torch.sum(torch.abs(U_out) ** 2)) #Checking if this is better

            A_out = torch.abs(U_out)
            A_out = A_out / (A_out.max() + 1e-18)  # normalize to 1
            P_out = torch.angle(U_out)

            # Cost function: MSE of amplitude inside ROI
            # loss = torch.sum((A_out - A_target_amp) ** 2 * mask) / mask.sum() #Is it only for amplitude?
            Efficiency_intensity_box = torch.sum((A_out) ** 2 * self.ones_box) / torch.sum((A_out) ** 2)
            Efficiency_beam_intensity = torch.sum((A_out) ** 2 * self.mask) / torch.sum((A_out) ** 2)

            # loss_orig = (10**2)*((1 - (
            #     torch.sum((torch.abs(A_out * A_target_amp) + 1e-8) * torch.cos(P_out - P_target_phase) * self.ones_box)) / ((
            #         (torch.sqrt(torch.sum(torch.conj(A_out) * A_out * self.ones_box))) * torch.sqrt(
            #     torch.sum(torch.conj(A_target_amp) * A_target_amp * self.ones_box))))) ** 2)

            #Difference:
                    # phase_diff =((A_target_amp/torch.sqrt(torch.sum(torch.conj(A_target_amp) * A_target_amp * self.ones_box)))**2)* torch.angle(torch.exp(1j * (P_out - P_target_phase)))
                    #
                    # loss_amp = torch.mean((((A_out/torch.sqrt(torch.sum(torch.conj(A_out) * A_out * self.ones_box))) - (A_target_amp/torch.sqrt(torch.sum(torch.conj(A_target_amp) * A_target_amp * self.ones_box)))) ** 2 * self.ones_box))
                    # loss_phase = torch.mean(phase_diff ** 2 * self.ones_box)
                    #
                    # loss_difference = 10e6 * (loss_amp + loss_phase)

            #Complexoverlap:
            out_complex = A_out * torch.exp(1j * P_out)
            target_complex = A_target_amp * torch.exp(1j * P_target_phase)

            out_complex = out_complex * self.ones_box # self.ion_mask#
            target_complex = target_complex * self.ones_box #self.ion_mask#

            num = torch.sum(torch.conj(target_complex) * out_complex)

            den = torch.sqrt(torch.sum(torch.abs(out_complex) ** 2)) * \
                  torch.sqrt(torch.sum(torch.abs(target_complex) ** 2))

            eta = num / (den + 1e-12)

            loss_complexoverlap = 100 * (1 - torch.real(eta))

            out_complex_small = A_out * torch.exp(1j * P_out) * self.ion_mask_small#
            target_complex_small = A_target_amp * torch.exp(1j * P_target_phase) * self.ion_mask_small#

            num_small = torch.sum(torch.conj(target_complex_small) * out_complex_small)

            den_small = torch.sqrt(torch.sum(torch.abs(out_complex_small) ** 2)) * \
                  torch.sqrt(torch.sum(torch.abs(target_complex_small) ** 2))

            eta_small = num_small / (den_small + 1e-12)

            loss_complexoverlap_small = 100 * (1 - torch.real(eta_small))

            # loss_ionpos = ((1 - (
            #     torch.sum((torch.abs(A_out * A_target_amp) + 1e-18) * torch.cos(P_out - P_target_phase) * self.ion_mask)) / ((
            #         (torch.sqrt(torch.sum(torch.conj(A_out) * A_out * self.ion_mask))) * torch.sqrt(
            #     torch.sum(torch.conj(A_target_amp) * A_target_amp * self.ion_mask))))) ** 2)

            #self.image_field = U_out #original used
            self.image_field = U_out / torch.sqrt(torch.sum(torch.abs(U_out) ** 2))

            # amps_current = (
            # [torch.abs(self.image_field[int(np.round(spotnum[0])), int(np.round(spotnum[1]))]) for spotnum in
            #  self.spots]) #used originally
            coords_round = torch.round(spots_tensor).long()
            amps_current = torch.abs(self.image_field[coords_round[:, 0], coords_round[:, 1]])

            # # phase_current = (
            # # [torch.angle(self.image_field[int(np.round(spotnum[0])), int(np.round(spotnum[1]))]) for spotnum in
            # #  self.spots])
            #amps_current = torch.stack(amps_current)
            amps_current = amps_current / torch.max(amps_current)
            # # # phase_current = torch.stack(phase_current)
            # # # print("amps_current", amps_current)
            # # # print("phase_current", phase_current)
            # # # print("amps_current", amps_current)
            # # # print("phase_current", phase_current)
            # #
            # # target_normalised = self.target_orig[0] / torch.sqrt(torch.abs(
            # #     torch.sum(torch.conj(
            # #         torch.as_tensor(self.target_orig[0])) * torch.as_tensor(
            # #         self.target_orig[0]) * self.ones_box)))
            # #
            # # amps_target_orig = (
            # #     [torch.abs(torch.as_tensor(target_normalised[int(np.round(spotnum[0])), int(np.round(spotnum[1]))])) for
            # #      spotnum in
            # #      self.spots])
            # # amps_target_orig = torch.as_tensor(amps_target_orig, device=self.device)
            # # amps_target_orig = amps_target_orig / torch.max(amps_target_orig)
            # #
            # #
            # #
            err = torch.abs(amps_current) - torch.abs(torch.as_tensor(amps_target_orig,device=self.device))#-torch.as_tensor(wuamps0)  # amplitude error for each beam
            # # err_phase = torch.as_tensor(phase_current,device=self.device) - torch.as_tensor(wuphases0 - cp.array(
            # #     target_phase_curve),device=self.device)  # phase error for each beam
            #


            # Fidelity_loss = ((torch.abs(
            #     torch.sum(torch.conj(self.image_field) * (torch.asarray(A_target_amp*torch.exp(1j*P_target_phase))) * torch.asarray(self.ones_box))) ** 2 / (
            #                            (torch.abs(torch.sum(torch.conj(self.image_field) * self.image_field * torch.asarray(
            #                                self.ones_box)))) * torch.abs(torch.sum(
            #                        torch.conj((torch.asarray(A_target_amp*torch.exp(1j*P_target_phase)))) * (torch.asarray(A_target_amp*torch.exp(1j*P_target_phase))) * torch.asarray(
            #                            self.ones_box))))))

            #loss = loss_orig#+0.005*torch.mean(torch.abs(err))#+torch.max(torch.tensor(0.0),0.10-Efficiency_beam_intensity)+torch.mean(torch.abs(err))*10#+ torch.relu(0.1-Efficiency_beam_intensity)#+0.05*(1-Efficiency_beam_intensity)
            #loss=torch.mean(torch.abs(err))*10 + torch.mean(torch.abs(err_phase))*10 #loss_ionpos.real+torch.max(torch.tensor(0.0),0.05-Efficiency_beam_intensity)+
            #loss= loss_ionpos.real+0.01*torch.mean(torch.abs(err))#-0.01*torch.log(Efficiency_beam_intensity + 1e-18)#+0.1*torch.mean(torch.abs(err)).real #-0.001*torch.log(Efficiency_beam_intensity + 1e-18)
            ##loss=(1-Fidelity_loss)+1*torch.max(torch.tensor(0.0),0.12-Efficiency_beam_intensity)
            #loss=(loss_difference)


            #MAIN LOSS FUNCTION. Uncomment the rest of the loss components to give weightage to other metrics
            loss=((loss_complexoverlap))#(target_loss/0.05)*torch.mean(torch.abs(err)**2)##+(target_loss/0.05)*torch.mean(torch.abs(err)))#+*torch.std(torch.abs(amps_current)))#+5*torch.mean(torch.abs(err)))#+runsettings.efficiency_limit_scale*4*(torch.max(torch.tensor(0.0),runsettings.efficiency_limit-Efficiency_beam_intensity)))
                  #+5*(loss_complexoverlap_small)+5*torch.mean(torch.abs(err))*runsettings.efficiency_limit_scale+runsettings.efficiency_limit_scale*4*(torch.max(torch.tensor(0.0),runsettings.efficiency_limit-Efficiency_beam_intensity)))#+torch.mean(torch.abs(err))
            #loss=10*torch.mean(torch.abs(err))+torch.max(torch.tensor(0.0),loss_complexoverlap-0.05)
            # if np.mod(int(epoch/100), 2) == 0:
            #     loss=loss_complexoverlap#0.1*loss_complexoverlap+(1-Efficiency_beam_intensity)*0.02+2*torch.mean(torch.abs(err))
            # else:
            #     loss=(1-Efficiency_beam_intensity)*0.2#2*torch.mean(torch.abs(err))
            # alpha_l = epoch / epochs
            # loss = (1 - alpha_l) * loss_complexoverlap + (alpha_l) * torch.mean(torch.abs(err)) * (target_loss/0.005)# + alpha_l*(torch.max(torch.tensor(0.0),0.48-Efficiency_beam_intensity))
            # if epoch<200:
            #     loss = loss_complexoverlap#(1-alpha_l) * loss_complexoverlap + (alpha_l) * (1-Efficiency_beam_intensity)*1
            # else:
            #     loss=20*(target_loss/0.005)*torch.mean(torch.abs(err)**2)#loss_complexoverlap+torch.max(torch.tensor(0.0),0.30-Efficiency_beam_intensity)*1
            #loss= loss_complexoverlap*((1-Efficiency_beam_intensity))**50

            if False:  # best loss
                if epoch == 0:
                    best_loss = loss
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_phi_slm = phi_slm.detach().clone()  # Save the current phi_slm if loss improves

            # Backprop
            loss.backward()
            optimizer.step()
            # phi_slm = torch.clamp(phi_slm, min=-2 * torch.pi, max=2 * torch.pi)
            if (epoch + 1) % 200 == 0:

                with torch.no_grad():
                    self.totalsteperr.append((torch.abs(
                        torch.sum(torch.conj(self.image_field) * torch.as_tensor(
                            self.target_orig[0]) * self.ones_box)) ** 2 / (
                                                      (torch.abs(torch.sum(torch.conj(
                                                          self.image_field) * self.image_field * self.ones_box))) * torch.abs(
                                                  torch.sum(torch.conj(
                                                      torch.as_tensor(self.target_orig[0])) * torch.as_tensor(
                                                      self.target_orig[0]) * self.ones_box)))).item())

                    efficiency = torch.sum(
                        torch.abs(torch.conj(self.image_field) * self.image_field) * self.mask) / torch.sum(
                        torch.abs(torch.conj(self.image_field) * self.image_field))
                    #self.eff.append(efficiency.detach().cpu().numpy()) #original revert back for type
                    self.eff.append(efficiency.detach().cpu().item())

                    amps_current = (
                    [torch.abs(self.image_field[int(np.round(spotnum[0])), int(np.round(spotnum[1]))]) for spotnum in
                     self.spots])
                    phase_current = (
                    [torch.angle(self.image_field[int(np.round(spotnum[0])), int(np.round(spotnum[1]))]) for spotnum in
                     self.spots])
                    target_normalised=self.target_orig[0]/ torch.sqrt(torch.abs(
                                                  torch.sum(torch.conj(
                                                      torch.as_tensor(self.target_orig[0])) * torch.as_tensor(
                                                      self.target_orig[0]) * self.ones_box)))

                    amps_target_orig = (
                    [torch.abs(torch.as_tensor(target_normalised[int(np.round(spotnum[0])), int(np.round(spotnum[1]))])) for spotnum in
                     self.spots])
                    amps_target_orig=torch.as_tensor(amps_target_orig,device=self.device)
                    amps_target_orig=amps_target_orig/torch.max(amps_target_orig)

                    amps_current = torch.as_tensor(amps_current,device=self.device)
                    amps_current = amps_current / torch.max(amps_current)
                    phase_current = torch.as_tensor(phase_current,device=self.device)
                    # print("amps_current", amps_current)
                    # print("phase_current", phase_current)
                    # print("amps_current", amps_current)
                    # print("phase_current", phase_current)
                    err = torch.abs(amps_current) - torch.abs(torch.as_tensor(amps_target_orig,device=self.device))#torch.as_tensor(wuamps0)  # amplitude error for each beam
                    # print("ampos_current", amps_current)
                    # print("ampos_target_orig", amps_target_orig)
                    err_phase = torch.as_tensor(phase_current,device=self.device) - torch.as_tensor(wuphases0 - cp.array(
                        target_phase_curve),device=self.device)  # phase error for each beam
                    amps_error=torch.mean(torch.abs(err))
                    phase_error=torch.mean(torch.abs(err_phase))
                    self.nonunif.append(amps_error.detach().cpu().item())
                    self.phase_err.append(phase_error.detach().cpu().numpy().item())

                    plot_err_mean.append(torch.mean(torch.abs(err)).detach().cpu().numpy().item())
                    plot_loss_overlap_complex.append(loss_complexoverlap.detach().cpu().numpy().item())
                    plot_eff_minus1.append((1-Efficiency_beam_intensity).detach().cpu().numpy().item())


                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Efficiency: {efficiency.item():.6f}')
                print('loss', loss.item(), 'amps_error', torch.mean(torch.abs(err)).item(), 'phase_error',
                      torch.mean(torch.abs(err_phase)).item())
                print('loss contributors',loss_complexoverlap.item(),(1-Efficiency_beam_intensity.item()),torch.mean(torch.abs(err)).item())
                print('loss contri scaled', 0.1*loss_complexoverlap.item(), (1 - Efficiency_beam_intensity.item())*0.02,
                      torch.mean(torch.abs(err)).item()*2.0)
                loss_track.append(loss.item())

                print("amps_current", amps_current)

                #print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}' + " fidelity:", fidelity_track[-1],
                #      " eff:", efficiency_track[-1])

                if False:#with torch.no_grad(): #Amplitude phase, not uniformity tracker
                    amps_current = self.beams_torch(waist=0.0015)
                    if isinstance(amps_current, list):
                        amps_current = torch.stack(amps_current)
                    amps_current = amps_current / torch.max(amps_current)
                    phase_current = self.beams_phase_torch(waist=0.0015)
                    print("amps_current_old", amps_current)
                    print("phase_current_old", phase_current)
                    # amps_current, phase_current = zip(*[self.beammax_torch(self.image_field, spotnum,1)for spotnum in self.spots])
                    amps_current = ([torch.abs(self.image_field[int(np.round(spotnum[0])),int(np.round(spotnum[1]))]) for spotnum in self.spots])
                    amps_current = torch.stack(amps_current)
                    amps_current = amps_current / torch.max(amps_current)
                    # phase_current = torch.stack(phase_current)
                    print("amps_current", amps_current)
                    # print("phase_current", phase_current)
                    err = torch.abs(amps_current) - torch.as_tensor(wuamps0)  # amplitude error for each beam
                    err_phase = torch.as_tensor(phase_current,device=self.device) - torch.as_tensor(wuphases0 - cp.array(
                        target_phase_curve),device=self.device)  # phase error for each beam
                    print('amps_error', torch.mean(torch.abs(err)), 'phase_error', torch.mean(torch.abs(err_phase)))

            if loss.item() < target_loss:
            #if loss_complexoverlap.item() < target_loss:
                print(f"Early stopping at epoch {epoch + 1}, loss reached {loss.item():.6f},loss_overlap {loss_complexoverlap.item():.6f}")
                break
            #
            # Convergence check based on patience
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0  # Reset counter if improvement is seen
            else:
                patience_counter += 1  # Increment counter if no improvement

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}, loss stopped improving.")
                break



        # 3. --- CONVERT BACK TO CUPY ---
        # Only do this once after all N iterations are done
        with torch.no_grad():
            self.image_field = cp.asarray(U_out.detach())
            self.slm_field = cp.asarray(A_in.detach()*torch.exp(1j*phi_slm.detach()))
            self.p = cp.asarray(phi_slm.detach())

            # plt.plot(self.nonunif.get())
            # plt.show()
            print("amps_current", amps_current)
            amps_current_avg = torch.mean(amps_current)
            amps_current_std = torch.std(amps_current)

            print("non-uniformity:",amps_current_std / amps_current_avg,"std deviation:",amps_current_std)
            runsettings.amps_current_std=float(amps_current_std.detach().cpu().numpy())

            # fig, ax1 = plt.subplots(figsize=(10, 6))
            # ax1.plot(plot_loss_overlap_complex, color='tab:blue', label='Loss')
            # ax1.set_ylabel('Loss', color='tab:blue')
            # ax2=ax1.twinx()
            # ax2.plot(plot_eff_minus1, color='tab:red', label='Efficiency')
            # ax1.set_ylabel('Loss', color='tab:blue')
            # ax3 = ax1.twinx()
            # ax3.spines['right'].set_position(('outward', 60))
            # ax3.plot(plot_err_mean, color='tab:green', label='Non-Uniformity')
            # ax3.set_ylabel('Non-Uniformity', color='tab:green')
            # plt.show()


    def iterate_Gradient_staged_optimisation(self, N, wuamps0, wuphases0, target_phase_curve):
        import torch
        import numpy as np
        import cupy as cp

        self.stepnum = 0
        loss_track = []

        # --- ONE-TIME CONVERSION TO TORCH ---
        with torch.no_grad():
            vars_to_convert = [
                'image_field', 'A_t', 'S', 'I', 'P_t', 'mask',
                'ones_box', 'input', 'A_t_orig', 'P_t_orig', 'P_c_old', 'ion_mask','ion_mask_small'
            ]
            for var in vars_to_convert:
                val = getattr(self, var, None)
                if val is not None:
                    tensor_val = torch.as_tensor(
                        val,
                        device=self.device,
                        dtype=torch.float32 if "P_" in var or var in ['S', 'I', 'mask', 'ones_box'] else torch.complex64
                    )
                    setattr(self, var, tensor_val)

        # --- INITIALIZE VARIABLES ---
        phi_slm = torch.tensor(self.p, device=self.device, dtype=torch.float32, requires_grad=True)
        A_in = torch.as_tensor(cp.abs(self.input), device=self.device, dtype=torch.float32)
        A_target_amp = torch.as_tensor(self.A_t_orig, device=self.device, dtype=torch.float32)
        P_target_phase = torch.as_tensor(self.P_t_orig, device=self.device, dtype=torch.float32)
        ones_box = self.ones_box
        mask = self.mask

        optimizer = torch.optim.Adam([phi_slm], lr=0.01)

        # --- PRECOMPUTE NORMALIZED TARGET ---
        target_tensor = torch.as_tensor(self.target_orig[0], device=self.device)
        target_norm = target_tensor / torch.sqrt(
            torch.abs(torch.sum(torch.conj(target_tensor) * target_tensor * ones_box)) + 1e-12)

        # --- VECTORIZE SPOT INDEXES ---
        spots_idx = torch.tensor(np.round(self.spots), dtype=torch.long, device=self.device)
        spots_idx[:, 0] = torch.clamp(spots_idx[:, 0], 0, self.size[0] - 1)
        spots_idx[:, 1] = torch.clamp(spots_idx[:, 1], 0, self.size[1] - 1)

        amps_target_orig = torch.abs(target_norm[spots_idx[:, 0], spots_idx[:, 1]])
        amps_target_orig = amps_target_orig / (torch.max(amps_target_orig) + 1e-12)

        # --- DEFINE STAGES ---
        stages = [
            {'name': 'complex_only', 'epochs': 500, 'w_complex': 1.0, 'w_amp': 0.0},
            {'name': 'amplitude_only', 'epochs': 300, 'w_complex': 0.05, 'w_amp': 1.0},
            {'name': 'combined', 'epochs': 50, 'w_complex': 0.1, 'w_amp': 1.0}
        ]

        for stage in stages:
            for epoch in range(stage['epochs']):
                optimizer.zero_grad()

                # --- FORWARD PROPAGATION ---
                U_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A_in * torch.exp(1j * phi_slm))))
                U_out = U_out / (self.size[0] * self.size[1])
                U_out = U_out / torch.sqrt(torch.sum(torch.abs(U_out) ** 2) + 1e-12)

                A_out = torch.abs(U_out)
                A_out = A_out / (torch.max(A_out) + 1e-18)
                P_out = torch.angle(U_out)

                # --- COMPLEX OVERLAP LOSS ---
                out_complex = A_out * torch.exp(1j * P_out) * ones_box
                target_complex = A_target_amp * torch.exp(1j * P_target_phase) * ones_box
                eta = torch.sum(torch.conj(target_complex) * out_complex) / (
                        torch.sqrt(torch.sum(torch.abs(out_complex) ** 2) + 1e-12) *
                        torch.sqrt(torch.sum(torch.abs(target_complex) ** 2) + 1e-12)
                )
                loss_complexoverlap = 100 * (1 - torch.real(eta))

                # --- SPOT AMPLITUDE ERROR ---
                amps_current = torch.abs(U_out[spots_idx[:, 0], spots_idx[:, 1]])
                amps_current = amps_current / (torch.max(amps_current) + 1e-12)
                amp_error = torch.mean(torch.abs(amps_current - amps_target_orig))

                # --- TOTAL LOSS FOR THIS STAGE ---
                loss = stage['w_complex'] * loss_complexoverlap + stage['w_amp'] * amp_error

                # --- BACKPROPAGATION ---
                loss.backward()
                optimizer.step()

                # --- WRAP PHASE TO [-π, π] safely ---
                phi_slm.data[:] = torch.remainder(phi_slm.data + np.pi, 2 * np.pi) - np.pi

                # --- TRACKING AND PRINTING ---
                if (epoch + 1) % 100 == 0 or stage['name'] != 'combined':
                    with torch.no_grad():
                        self.image_field = U_out / torch.sqrt(torch.sum(torch.abs(U_out) ** 2) + 1e-12)
                        efficiency = torch.sum(torch.abs(U_out) ** 2 * mask) / (
                                    torch.sum(torch.abs(U_out) ** 2) + 1e-12)
                        self.eff.append(efficiency.detach().cpu().item())
                        self.nonunif.append(amp_error.item())
                    print(
                        f"Stage: {stage['name']}, Epoch {epoch + 1}/{stage['epochs']}, Loss: {loss.item():.6f}, Efficiency: {efficiency.item():.6f}")

        # --- CONVERT BACK TO CUPY ---
        with torch.no_grad():
            self.image_field = cp.asarray(U_out.detach())
            self.slm_field = cp.asarray(A_in.detach() * torch.exp(1j * phi_slm.detach()))
            self.p = cp.asarray(phi_slm.detach())

    # Diffraction efficiency
    def eta(self):
        return cp.sum(cp.abs(self.mask * self.image_field)**2) / cp.sum(cp.abs(self.image_field)**2)

    # Diffraction efficiency
    def I_a(self):
        return cp.sum(self.mask * cp.abs(self.image_field)**2) / cp.sum(self.mask)

    # Calculate amplitude nonuniformity
    def nonuniformity(self):
        # I_a = self.I_a()
        # I_a = cp.sum(self.mask * cp.abs(self.image_field)**2) / cp.sum(self.mask)
        # I_t = cp.sum(self.mask * cp.abs(self.target)**2) / cp.sum(self.mask)
        # normalized_image = self.image_field * cp.sum(cp.abs(self.input)) / cp.sum(cp.abs(self.image_field))
        return cp.sum(self.mask * cp.abs(cp.abs(self.image_field) - cp.abs(self.target))**2)\
               / cp.sum(self.mask * cp.abs(self.target)**2)

    # Calculate spot nonuniformity
    def spot_nonuniformity(self):
        return

    # Calculate phase error
    def phase_error(self):
        return cp.sum(self.mask * cp.abs(self.image_field)**2 * cp.abs(cp.angle(self.image_field) - cp.angle(self.target)))\
               / cp.sum(self.mask * cp.abs(self.image_field)**2 * cp.pi)

    def phases(self):
        return cp.array([cp.angle(self.avg(field=self.image_field, pos=spot, radius=self.waist)) for spot in self.spots])

    def amps(self):
        return cp.array([cp.abs(self.avg(field=self.image_field, pos=spot, radius=self.waist)) for spot in self.spots])


class OuterLoop_Gen:
    def __init__(self, slm=None, size=(1024,1272), input_profile=None, target_profile=None, wavelength=411, name='bs', start_phase=None, phase_memory=True):
        if input_profile is None:
            input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
        self.input_profile = input_profile
        self.input_profile = cp.array(self.input_profile)
        self.input_profile /= cp.sqrt(cp.sum(cp.abs(self.input_profile) ** 2))

        self.target_profile = target_profile[0]
        self.spots = target_profile[1]

        # Performance trackers
        self.eff = []
        self.nonunif = []
        self.phase_err = []

        # Record each algorithm run
        self.wus = []
        self.results = []
        self.target_record = [cp.copy(self.target_profile)]

        self.wavelength = wavelength
        self.size = size
        self.start_phase = start_phase
        self.phase_memory = phase_memory
        self.name = name
        if slm is None:
            slm = SLM()
        self.slm = slm

        self.min_it = 0

    def iterate(self, N, M):
        start_time = time.time()

        for i in range(M):
            print('Iteration: ' + str(i))

            # res = test_amps(n=self.n, N=N, size=self.size, beams0=self.amps0 * cp.exp(1j * self.phases0),
            #                 beams=self.amps * cp.exp(1j * self.phases),
            #                 x_pitch=self.x_pitch, input_profile=self.input_profile, start_phase=self.start_phase,
            #                 phase_memory=self.phase_memory)

            wu = Wu(input=self.input_profile, target=(self.target_profile, self.spots), size=size, array=False,
                    start_phase=self.start_phase, phase_memory=self.phase_memory)
            wu.iterate(N)
            print()

            self.results.append(wu.image_field / np.max(np.abs(wu.image_field)))

            res = [wu, wu.image_field, wu.eta(), wu.nonuniformity(), wu.phase_error()]

            if True:
                rel_result = self.target_record[0] / cp.abs(wu.image_field)
                self.target_profile *= cp.abs(rel_result) ** 0.25
            else:

                err = cp.abs(self.results[-1]) - cp.abs(self.target_record[0])
                grad = cp.abs(self.results[-1]) - cp.abs(self.results[-2])
                grad /= cp.abs(self.target_record[-1]) - cp.abs(self.target_record[-2])
                self.target_profile -= 0.1 * grad * err

            # Update the target amplitudes to compensate the nonuniformity of the previous iteration
            # print('Calculated amplitudes: ' + str(np.abs(self.results[-1]) / np.max(np.abs(self.results[-1]))))
            # err = cp.abs(self.results[-1]) - self.amps0
            # rel_result = self.amps0 / self.results[-1]
            # for j in range(len(self.amps)):
            #     if self.amps0[j] >= 0.25:
            #         self.amps[j] *= cp.abs(rel_result[j]) ** 0.25
            # self.amps /= cp.max(self.amps)
            # for j in range(len(self.amps)):
            #     if i > 0 and self.amps0[j] < 0.25 and self.amps_record[-1][j] != self.amps_record[-2][j]:
            #         grad = cp.abs(self.results[-1][j]) - cp.abs(self.results[-2][j])
            #         grad /= self.amps_record[-1][j] - self.amps_record[-2][j]
            #         self.amps[j] -= 0.1 * grad * err[j]
            # print('Updated target amplitudes: ' + str(self.amps))
            # self.amps_record.append(cp.copy(self.amps))
            self.target_record.append(cp.copy(self.target_profile))

            self.start_phase = cp.angle(res[0].slm_field)

            self.wus.append(res[0])

            self.eff.append(res[2])
            self.nonunif.append(res[3])
            self.phase_err.append(res[4])
            print('Diffraction Efficiency: ' + str(self.eff[-1]))
            print('Amplitude error: ' + str(self.nonunif[-1]))
            print('Phase error: ' + str(self.phase_err[-1]))
            print()

        # Find the minimum nonuniformity run of the Wu algorithm
        self.min_it = self.nonunif.index(min(self.nonunif))
        self.min_it = len(self.nonunif) - 1

        print('Time to run: ' + str(time.time() - start_time))

        print('')
        print('----Minimum iteration----')
        print('Diffraction Efficiency: ' + str(self.eff[self.min_it]))
        print('Amplitude error: ' + str(self.nonunif[self.min_it]))
        print('Phase error: ' + str(self.phase_err[self.min_it]))
        print('min_it: ' + str(self.min_it))

        return self.wus[self.min_it].slm_field

    def plot(self, show=False, figs=(None, None, None)):
        plots= ()
        if show:
            plots = (0, 1, 2, 3, 4, 5)
        self.wus[self.min_it].save_pattern(name=self.name, slm=self.slm, target=True, correction=True, show=(), wavelength=self.wavelength,
                                 field=True, plots=plots)

        # print(figs[2])
        plot_gradient(self.wus[self.min_it].image_field, fig=figs[2])

        # Plot the efficiency, nonuniformity and phase error for the best iteration
        if figs[0] is None:
            plt.figure()
            plt.clf()
            plotter = plt
        else:
            figs[0].axes.cla()
            plotter = figs[0].axes
        plotter.plot([n for n in range(len(self.wus[self.min_it].eff))], cp.array(self.wus[self.min_it].eff).get(), label='Efficiency')
        plotter.plot([n for n in range(len(self.wus[self.min_it].nonunif))], cp.array(self.wus[self.min_it].nonunif).get(),
                 label='Amplitude error')
        plotter.plot([n for n in range(len(self.wus[self.min_it].phase_err))], cp.array(self.wus[self.min_it].phase_err).get(),
                 label='Phase error')
        if figs[0] is None:
            plotter.xlabel('Inner Iteration')
            plotter.title('Inner Loop Convergence')
            plotter.xlim(0, len(self.wus[0].eff))
        else:
            plotter.set_xlabel('Inner Iteration')
            plotter.set_title('Inner Loop Convergence')
            plotter.set_xlim(0, len(self.wus[0].eff))
        plotter.grid(True)
        plotter.legend()
        if figs[0] is None:
            plotter.pause(.001)
            plotter.savefig('images/inner_convergence.png')

        # Plot the final efficiency, nonuniformity and phase error across all iterations
        if figs[1] is None:
            plt.figure()
            plt.clf()
            plotter = plt
        else:
            figs[1].axes.cla()
            plotter = figs[1].axes
        plotter.plot([m for m in range(len(self.wus))], cp.array(self.eff).get(), label='Efficiency')
        plotter.plot([m for m in range(len(self.wus))], cp.array(self.nonunif).get(), label='Amplitude error')
        plotter.plot([m for m in range(len(self.wus))], cp.array(self.phase_err).get(), label='Phase error')
        if figs[1] is None:
            plotter.xlabel('Outer Iteration')
            plotter.title('Outer Loop Convergence')
            plotter.xlim(0, len(self.wus))
        else:
            plotter.set_xlabel('Outer Iteration')
            plotter.set_title('Outer Loop Convergence')
            plotter.set_xlim(0, len(self.wus))
        plotter.grid(True)
        plotter.legend()
        if figs[1] is None:
            plotter.savefig('images/outer_convergence.png')

        if show:
            plt.show()


class OuterLoop_MDS:
    def __init__(self, slm=None, amps_tem_compensation=None, phase_tem_compensation=None, m=4, n=5, size=(1024, 1272),
                 wavelength=411, name='wu_1x5',
                 amps=(1., 1., 1., 1., 1.), amps_guess=(1., 1., 1., 1., 1.), phases=(0, 0, 0, 0, 0), x_pitch=0.004,y_pitch=0.004,
                 input_profile=None, start_phase=None, phase_memory=False, tem01=False, imag=True):

        # Initialize input profile
        if input_profile is None:
            input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
        self.input_profile = input_profile
        self.input_profile = cp.array(self.input_profile)
        self.input_profile /= cp.sqrt(cp.sum(cp.abs(self.input_profile) ** 2))
        self.phase_tem_compensation = phase_tem_compensation

        # amps=(1.,)*m*n
        # amps_guess=(1.,)*m*n

        # Target array amplitudes and phases
        # print('tem01',tem01)
        if tem01:
            double_amps = []
            double_amps_guess = []
            double_phases = []
            for amp in amps:
                double_amps.append(amp)
                double_amps.append(amp)
            amps = np.array(double_amps)
            amps_guess = amps
            # for amp_guess in amps_guess:
            #     double_amps_guess.append(amp_guess)
            #     double_amps_guess.append(amp_guess)
            # amps_guess = np.array(double_amps_guess)
            for phase in phases:
                double_phases.append(phase)
                double_phases.append(phase)  #
            phases = np.array(double_phases)

        self.amps = cp.array(amps)
        self.amps0 = cp.copy(self.amps)
        self.amps = cp.array(amps_guess)
        self.m = m
        self.phases = 2 * pi * cp.array(phases)
        self.phases0 = cp.copy(self.phases)

        # Performance trackers
        self.eff = []
        self.nonunif = []
        self.phase_err = []

        # Record each algorithm run
        self.wus = []
        self.results = [self.amps]
        self.results_phase = [self.phases]
        self.amps_record = [self.amps]
        self.record_phase = [self.phases]

        self.n = n
        self.x_pitch = x_pitch
        self.y_pitch = y_pitch
        self.wavelength = wavelength
        self.size = size
        self.start_phase = start_phase
        self.phase_memory = phase_memory
        self.tem01 = tem01
        self.name = name
        if slm is None:
            slm = SLM()
        self.slm = slm
        self.imag = imag

        self.min_it = 0

    def iterate(self, N=50, M=30):
        start_time = time.time()

        self.phases0 = cp.copy(self.phases)
        self.middle = [[0], [0], [[512, 636]]]
        self.middle_last = [[0], [0], [[512, 636]]]
        target = Profile.target_output_array_MDS(self.m, self.n, middlecompensate=self.middle, center=(0, 0),
                                                   input_profile=self.input_profile.get(), x_pitch=self.x_pitch,
                                                   y_pitch=self.y_pitch,
                                                   amps=self.amps.get(),
                                                   phases=self.phases.get(),
                                                   size=np.array((1024, 1272)),
                                                   tem=self.tem01, double_amps=self.tem01)
        if not (self.phase_tem_compensation is None):
            target[0] = cp.array(target[0]) * cp.array(self.phase_tem_compensation)
            target_phase_curve = []
            plt.imshow(cp.angle(target[0]).get())  # MDS
            plt.show()
            for i in target[1]:
                # target_phase_curve.append(cp.angle(target[0][i[0]][i[1]])) #MDS
                target_phase_curve.append(cp.angle(self.phase_tem_compensation[i[0]][i[1]]))
            print('target phase_curve 2pi*', cp.array(target_phase_curve) / 2 / np.pi)  # MDS
        else:
            target_phase_curve = np.zeros((len(self.amps)))

        for i in range(M):
            self.middle = [[0], [0], [[512, 636]]]
            print('Iteration: ' + str(i))
            print('amps0', self.amps0)

            res = test_amps(m=self.m, n=self.n, N=N, size=self.size,
                            beams0=self.amps0 * cp.exp(1j * self.phases0),
                            beams=self.amps * cp.exp(1j * self.phases),
                            x_pitch=self.x_pitch, input_profile=self.input_profile,
                            start_phase=self.start_phase,
                            phase_memory=self.phase_memory, tem01=self.tem01, middlecompensate=self.middle,
                            phase_tem_compensation=self.phase_tem_compensation)
            self.results.append(res[1] / np.max(np.abs(res[1])))
            self.results_phase.append(res[5])

            self.middle[0] = res[6]

            self.middle[2] = res[8]
            self.middle[1] = cp.array(res[7])  # -cp.array( [0.33845852,0.51371972,0.2987232])
            # print('middlecompensate', self.middle[1])
            # print(np.angle(-1))
            # print(np.angle(self.results[-1]))
            # print((self.results[-1]))

            # Update the target amplitudes to compensate the nonuniformity of the previous iteration
            print('Calculated phases: 2pi * ' + str(np.angle(self.results[-1]) / 2 / np.pi))
            print('Calculated amplitudes: ' + str(np.abs(self.results[-1]) / np.max(np.abs(self.results[-1]))))

            err = cp.abs(self.results[-1]) - self.amps0  # amplitude error for each beam
            err_phase = (self.results_phase[-1]) - self.phases0 - cp.array(
                target_phase_curve)  # phase error for each beam
            print("phases0 MDS", self.phases0)
            count = 0
            for m in err_phase:
                if self.tem01 and count % 2 == 0:
                    err_phase[count] += np.pi
                if err_phase[count] < 0 and np.abs(err_phase[count]) > np.pi:
                    err_phase[count] += 2 * np.pi
                if err_phase[count] > 0 and np.abs(err_phase[count]) > np.pi:
                    err_phase[count] -= 2 * np.pi
                count += 1
            err_all = 1 * np.mean(np.abs(err)) + 0.1 * np.mean(
                np.abs(err_phase))  # set weight for uniform amlitude or for specificphase
            print('amps_error', np.mean(np.abs(err)), 'phase_error', np.mean(np.abs(err_phase)))
            # now we put weight on the phase
            rel_result = self.amps0 / self.results[-1]
            for j in range(len(self.amps)):
                if self.amps0[j] >= 0.25:
                    self.amps[j] *= cp.abs(rel_result[j]) ** 0.25
            self.amps /= cp.max(self.amps)
            # print(self.results_phase[-1])
            for j in range(len(self.amps)):
                if i > 0 and self.amps0[j] < 0.25 and self.amps_record[-1][j] != self.amps_record[-2][j]:
                    grad = cp.abs(self.results[-1][j]) - cp.abs(self.results[-2][j])
                    grad /= self.amps_record[-1][j] - self.amps_record[-2][j]
                    self.amps[j] -= 0.1 * grad * err[j]
                if i >= 0:
                    self.phases[j] -= err_phase[j] * 0.1
                    # print('chan')
                # if i > 0 and np.abs(self.results_phase[-1][j] -self.results_phase[-2][j])>0.0001:
                #
                #     grad = self.record_phase[-1][j] - self.record_phase[-2][j]
                #     grad /= cp.abs(self.results_phase[-1][j]) - cp.abs(self.results_phase[-2][j])
                #     self.phases[j] += 0.1 * grad * err_phase[j]
                #     print('suc')
                # else:
                #     if i > 0:
                #         print(self.results_phase[-1])
            print('Updated target amplitudes: ' + str(self.amps))
            self.amps_record.append(cp.copy(self.amps))
            self.record_phase.append(cp.copy(self.phases))

            self.start_phase = cp.angle(res[0].slm_field)

            self.wus.append(res[0])
            # if i==34:
            #
            #     plt.figure(figsize=(8, 6))
            #     plt.imshow(((cp.abs(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
            #     plt.colorbar()
            #     plt.title('wu result')
            #     plt.show()

            # plt.figure(figsize=(8, 6))
            # plt.imshow(((cp.angle(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            # plt.title('target phase')
            # plt.show()

            self.eff.append(res[2])
            # self.nonunif.append(res[3])
            self.nonunif.append(err_all)
            self.phase_err.append(res[4])
            print('Diffraction Efficiency: ' + str(self.eff[-1]))
            print('Amplitude error: ' + str(self.nonunif[-1]))
            print('Phase error: ' + str(self.phase_err[-1]))
            print()

        # Find the minimum nonuniformity run of the Wu algorithm
        self.min_it = self.nonunif.index(min(self.nonunif))
        # self.min_it = len(self.nonunif) - 1

        print('Time to run: ' + str(time.time() - start_time))

        print('')
        print('----Minimum iteration----')
        print('Diffraction Efficiency: ' + str(self.eff[self.min_it]))
        print('Total error: ' + str(self.nonunif[self.min_it]))
        print('Phase error: ' + str(self.phase_err[self.min_it]))
        print('\nTarget phases: 2pi * ' + str(self.phases0 / 2 / pi))
        print('Actual phases: 2pi * ' + str(((self.wus[self.min_it].phases() + 2 * pi) % (2 * pi)) / 2 / pi))
        print('\nTarget amplitudes: ' + str(self.amps0))
        print('Actual amplitudes: ' + str(self.wus[self.min_it].amps() / cp.max(self.wus[self.min_it].amps())))

        return self.wus[self.min_it].slm_field

    def iterate_MDS(self, N=50, M=30):
        start_time = time.time()

        self.phases0 = cp.copy(self.phases)
        self.middle = [[0], [0], [[512, 636]]]
        self.middle_last = [[0], [0], [[512, 636]]]
        target = Profile.target_output_array_MDS(self.m, self.n, middlecompensate=self.middle, center=(0, 0),
                                                 input_profile=self.input_profile.get(), x_pitch=self.x_pitch,
                                                 y_pitch=self.y_pitch,
                                                 amps=self.amps.get(),
                                                 phases=self.phases.get(),
                                                 size=np.array((1024, 1272)),
                                                 tem=self.tem01, double_amps=self.tem01)
        if not (self.phase_tem_compensation is None):
            target[0] = cp.array(target[0]) * cp.array(self.phase_tem_compensation)
            target_phase_curve = []
            for i in target[1]:
                # target_phase_curve.append(cp.angle(target[0][i[0]][i[1]])) #MDS
                target_phase_curve.append(cp.angle(self.phase_tem_compensation[i[0]][i[1]]))
            print('target phase_curve 2pi*', cp.array(target_phase_curve) / 2 / np.pi)  # MDS
        else:
            target_phase_curve = np.zeros((len(self.amps)))

        for i in range(M):
            self.middle = [[0], [0], [[512, 636]]]
            print('Iteration: ' + str(i))
            print('amps0', self.amps0)

            res = test_amps_MDS(m=self.m, n=self.n, N=N, size=self.size,
                            beams0=self.amps0 * cp.exp(1j * self.phases0),
                            beams=self.amps * cp.exp(1j * self.phases),
                            x_pitch=self.x_pitch,y_pitch=self.y_pitch, input_profile=self.input_profile,
                            start_phase=self.start_phase,
                            phase_memory=self.phase_memory, tem01=self.tem01, middlecompensate=self.middle,
                            phase_tem_compensation=self.phase_tem_compensation)
            self.results.append(res[1] / np.max(np.abs(res[1])))
            self.results_phase.append(res[5])

            self.middle[0] = res[6]

            self.middle[2] = res[8]
            self.middle[1] = cp.array(res[7])  # -cp.array( [0.33845852,0.51371972,0.2987232])
            # print('middlecompensate', self.middle[1])
            # print(np.angle(-1))
            # print(np.angle(self.results[-1]))
            # print((self.results[-1]))

            # Update the target amplitudes to compensate the nonuniformity of the previous iteration
            print('Calculated phases: 2pi * ' + str(np.angle(self.results[-1]) / 2 / np.pi))
            print('Calculated amplitudes: ' + str(np.abs(self.results[-1]) / np.max(np.abs(self.results[-1]))))

            err = cp.abs(self.results[-1]) - self.amps0  # amplitude error for each beam
            err_phase = (self.results_phase[-1]) - self.phases0 - cp.array(
                target_phase_curve)  # phase error for each beam
            print("phases0 MDS", self.phases0)
            count = 0
            for m in err_phase:
                if self.tem01 and count % 2 == 0:
                    err_phase[count] += np.pi
                if err_phase[count] < 0 and np.abs(err_phase[count]) > np.pi:
                    err_phase[count] += 2 * np.pi
                if err_phase[count] > 0 and np.abs(err_phase[count]) > np.pi:
                    err_phase[count] -= 2 * np.pi
                count += 1
            err_all = 1 * np.mean(np.abs(err)) + 1 * np.mean(
                np.abs(err_phase))  # set weight for uniform amlitude or for specificphase
            print('amps_error', np.mean(np.abs(err)), 'phase_error', np.mean(np.abs(err_phase)))
            # now we put weight on the phase
            rel_result = self.amps0 / self.results[-1]
            for j in range(len(self.amps)):
                if self.amps0[j] >= 0.25:
                    self.amps[j] *= cp.abs(rel_result[j]) ** 0.25
            self.amps /= cp.max(self.amps)
            # print(self.results_phase[-1])
            for j in range(len(self.amps)):
                if i > 0 and self.amps0[j] < 0.25 and self.amps_record[-1][j] != self.amps_record[-2][j]:
                    grad = cp.abs(self.results[-1][j]) - cp.abs(self.results[-2][j])
                    grad /= self.amps_record[-1][j] - self.amps_record[-2][j]
                    self.amps[j] -= 0.1 * grad * err[j]
                if i >= 0:
                    self.phases[j] -= err_phase[j] * 0.1
                    # print('chan')
                # if i > 0 and np.abs(self.results_phase[-1][j] -self.results_phase[-2][j])>0.0001:
                #
                #     grad = self.record_phase[-1][j] - self.record_phase[-2][j]
                #     grad /= cp.abs(self.results_phase[-1][j]) - cp.abs(self.results_phase[-2][j])
                #     self.phases[j] += 0.1 * grad * err_phase[j]
                #     print('suc')
                # else:
                #     if i > 0:
                #         print(self.results_phase[-1])
            print('Updated target amplitudes: ' + str(self.amps))
            self.amps_record.append(cp.copy(self.amps))
            self.record_phase.append(cp.copy(self.phases))

            self.start_phase = cp.angle(res[0].slm_field)

            self.wus.append(res[0])
            # if i==34:
            #
            #     plt.figure(figsize=(8, 6))
            #     plt.imshow(((cp.abs(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
            #     plt.colorbar()
            #     plt.title('wu result')
            #     plt.show()

            # plt.figure(figsize=(8, 6))
            # plt.imshow(((cp.angle(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            # plt.title('target phase')
            # plt.show()

            self.eff.append(res[2])
            # self.nonunif.append(res[3])
            self.nonunif.append(err_all)
            self.phase_err.append(res[4])
            print('Diffraction Efficiency: ' + str(self.eff[-1]))
            print('Amplitude error: ' + str(self.nonunif[-1]))
            print('Phase error: ' + str(self.phase_err[-1]))
            print()

        # Find the minimum nonuniformity run of the Wu algorithm
        self.min_it = self.nonunif.index(min(self.nonunif))
        # self.min_it = len(self.nonunif) - 1

        print('Time to run: ' + str(time.time() - start_time))

        print('')
        print('----Minimum iteration----')
        print('Diffraction Efficiency: ' + str(self.eff[self.min_it]))
        print('Amplitude error: ' + str(self.nonunif[self.min_it]))
        print('Phase error: ' + str(self.phase_err[self.min_it]))
        print('\nTarget phases: 2pi * ' + str(self.phases0 / 2 / pi))
        print('Actual phases: 2pi * ' + str(((self.wus[self.min_it].phases() + 2 * pi) % (2 * pi)) / 2 / pi))
        print('\nTarget amplitudes: ' + str(self.amps0))
        print('Actual amplitudes: ' + str(self.wus[self.min_it].amps() / cp.max(self.wus[self.min_it].amps())))

        return self.wus[self.min_it].slm_field

    def plot(self, show=False, figs=(None, None, None)):
        plots = ()
        if show:
            plots = (0, 1, 2, 3, 4, 5)
        self.wus[self.min_it].save_pattern(name=self.name, slm=self.slm, target=False, correction=True, show=(),
                                           wavelength=self.wavelength,
                                           field=True, plots=plots)

        # print(figs[2])
        plot_gradient(self.wus[self.min_it].image_field, fig=figs[2], intensity=True, imag=self.imag)

        # Plot the efficiency, nonuniformity and phase error for the best iteration
        if figs[0] is None:
            plt.figure()
            plt.clf()
            plotter = plt
        else:
            figs[0].axes.cla()
            plotter = figs[0].axes
        plotter.plot([n for n in range(len(self.wus[self.min_it].eff))], cp.array(self.wus[self.min_it].eff).get(),
                     label='Efficiency')
        plotter.plot([n for n in range(len(self.wus[self.min_it].nonunif))],
                     cp.array(self.wus[self.min_it].nonunif).get(),
                     label='Amplitude error')
        plotter.plot([n for n in range(len(self.wus[self.min_it].phase_err))],
                     cp.array(self.wus[self.min_it].phase_err).get(),
                     label='Phase error')
        if figs[0] is None:
            plotter.xlabel('Inner Iteration')
            plotter.title('Inner Loop Convergence')
            plotter.xlim(0, len(self.wus[0].eff))
        else:
            plotter.set_xlabel('Inner Iteration')
            plotter.set_title('Inner Loop Convergence')
            plotter.set_xlim(0, len(self.wus[0].eff))
        plotter.grid(True)
        plotter.legend()
        if figs[0] is None:
            plotter.pause(.001)
            plotter.savefig('images/inner_convergence.png')

        # Plot the final efficiency, nonuniformity and phase error across all iterations
        if figs[1] is None:
            plt.figure()
            plt.clf()
            plotter = plt
        else:
            figs[1].axes.cla()
            plotter = figs[1].axes
        plotter.plot([m for m in range(len(self.wus))], cp.array(self.eff).get(), label='Efficiency')
        plotter.plot([m for m in range(len(self.wus))], cp.array(self.nonunif).get(), label='Amplitude error')
        plotter.plot([m for m in range(len(self.wus))], cp.array(self.phase_err).get(), label='Phase error')
        if figs[1] is None:
            plotter.xlabel('Outer Iteration')
            plotter.title('Outer Loop Convergence')
            plotter.xlim(0, len(self.wus))
        else:
            plotter.set_xlabel('Outer Iteration')
            plotter.set_title('Outer Loop Convergence')
            plotter.set_xlim(0, len(self.wus))
        plotter.grid(True)
        plotter.legend()
        if figs[1] is None:
            plotter.savefig('images/outer_convergence.png')

        if show:
            plt.show()


class OuterLoop:
    def __init__(self, slm=None, amps_tem_compensation=None,phase_tem_compensation=None , m=4, n=5, size=(1024,1272), wavelength=411, name='wu_1x5',
                 amps=(1., 1., 1., 1., 1.), amps_guess=(1., 1., 1., 1., 1.), phases=(0, 0, 0, 0, 0), x_pitch=0.004,
                 input_profile=None, start_phase=None, phase_memory=False, tem01=False, imag=True, uni_spacing=True, xarblist0=None, yarblist0=None, anglearblist0=None,double_amps_in=None):

        # Initialize input profile
        if input_profile is None:
            input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
        self.input_profile = input_profile
        self.input_profile = cp.array(self.input_profile)
        self.input_profile /= cp.sqrt(cp.sum(cp.abs(self.input_profile) ** 2))
        self.phase_tem_compensation=phase_tem_compensation

        # amps=(1.,)*m*n
        # amps_guess=(1.,)*m*n

        
        # Target array amplitudes and phases
        # print('tem01',tem01)
        if tem01:
            double_amps = []
            double_amps_guess = []
            double_phases = []
            if double_amps_in is None:
                for amp in amps:
                    double_amps.append(amp)
                    double_amps.append(amp)
            else:
                double_amps=np.array(double_amps_in)
            print("double_amps first output",double_amps)
            amps = np.array(double_amps)
            amps_guess =amps
            # for amp_guess in amps_guess:
            #     double_amps_guess.append(amp_guess)
            #     double_amps_guess.append(amp_guess)
            # amps_guess = np.array(double_amps_guess)
            for phase in phases:
                double_phases.append(phase)
                double_phases.append(phase)#
            phases = np.array(double_phases)



        self.amps = cp.array(amps)
        self.amps0 = cp.copy(self.amps)
        self.amps = cp.array(amps_guess)
        self.m=m
        self.phases = 2 * pi * cp.array(phases)
        self.phases0 = cp.copy(self.phases)

        # Performance trackers
        self.eff = []
        self.nonunif = []
        self.phase_err = []

        # Record each algorithm run
        self.wus = []
        self.wusField = []
        self.wusImageField = []
        self.wuseff=[]
        self.wusnonunif=[]
        self.wusphase_err=[]
        self.wusfidelity_err = []
        self.P_c_old=None
        
        
        self.results = [self.amps]
        self.results_phase=[self.phases]
        self.amps_record = [self.amps]
        self.record_phase=[self.phases]

        self.n = n
        self.x_pitch = x_pitch
        self.wavelength = wavelength
        self.size = size
        self.start_phase = start_phase
        self.phase_memory = phase_memory
        self.tem01 = tem01
        self.name = name
        if slm is None:
            slm = SLM()
        self.slm = slm
        self.imag = imag

        self.min_it = 0

        self.uni_spacing=uni_spacing
        self.xarblist0=xarblist0
        self.yarblist0=yarblist0
        self.anglearblist0=anglearblist0

    def iterate(self, N=50, M=30):
        start_time = time.time()
        
        self.phases0=cp.copy(self.phases)
        self.middle =  [[0], [0],[[512,636]]]
        self.middle_last=[[0], [0],[[512,636]]]
        target = Profile.target_output_array_bokai(self.m, self.n,  middlecompensate=self.middle, center=(0, 0),
                                                   input_profile=self.input_profile.get(), x_pitch=self.x_pitch,
                                                   y_pitch=self.x_pitch,
                                                   amps=self.amps.get(),
                                                   phases=self.phases.get(),
                                                   size=self.size,#np.array((1024, 1272)),
                                                   tem=self.tem01, double_amps=self.tem01)
        target_orig = Profile.target_output_array_bokai_Offcenter2Darb(self.m, self.n, middlecompensate=self.middle, center=(0, 0),
                                                 input_profile=self.input_profile.get(), x_pitch=self.x_pitch,
                                                 y_pitch=self.x_pitch,
                                                 amps=self.amps.get(), phases=self.phases.get(),
                                                 size=self.size,
                                                 tem=self.tem01, double_amps=self.tem01)
        plt.imshow(np.abs(target_orig[0].get()))
        plt.plot()
        if not (self.phase_tem_compensation is None):
            target[0] = cp.array(target[0]) * cp.array(self.phase_tem_compensation)
            target_phase_curve = []
            for i in target[1]:
                #target_phase_curve.append(cp.angle(target[0][i[0]][i[1]])) #MDS
                target_phase_curve.append(cp.angle(self.phase_tem_compensation[i[0]][i[1]]))
            print('target phase_curve 2pi*', cp.array(target_phase_curve)/2/np.pi)#MDS
        else:
            target_phase_curve=np.zeros((len(self.amps)))

        Mmax=M
        phaseconstant=False
        for i in range(Mmax):
            self.middle = [[0], [0], [[512, 636]]]
            print('Iteration: ' + str(i))
            print('amps0',self.amps0)
            #print("iteration:",i,"statrphase",self.start_phase)
            if i==0:
                Ntemp=2
            else:
                Ntemp=N
            if i>4:
                phaseconstant=True
            res = test_amps(m=self.m,n=self.n, N=Ntemp, size=self.size,
                            beams0=self.amps0 * cp.exp(1j * self.phases0),
                            beams=self.amps * cp.exp(1j * self.phases),
                            x_pitch=self.x_pitch, input_profile=self.input_profile,
                            start_phase=self.start_phase,
                            phase_memory=self.phase_memory, tem01=self.tem01,middlecompensate=self.middle,phase_tem_compensation=self.phase_tem_compensation,phaseconstant=phaseconstant,P_c_old=self.P_c_old,target_orig=target_orig)
            self.results.append(res[1] / np.max(np.abs(res[1])))
            self.results_phase.append(res[5])
            self.P_c_old=res[14]

            self.middle[0] = res[6]

            self.middle[2] = res[8]
            self.middle[1] = cp.array(res[7])#-cp.array( [0.33845852,0.51371972,0.2987232])
            # print('middlecompensate', self.middle[1])
            # print(np.angle(-1))
            # print(np.angle(self.results[-1]))
            # print((self.results[-1]))
           

            # Update the target amplitudes to compensate the nonuniformity of the previous iteration
            print('Calculated phases: 2pi * ' + str(np.angle(self.results[-1])/2/np.pi))
            print('Calculated amplitudes: ' + str(np.abs(self.results[-1]) / np.max(np.abs(self.results[-1]))))

            err = cp.abs(self.results[-1]) - self.amps0 #amplitude error for each beam
            err_phase=(self.results_phase[-1]) - self.phases0-cp.array(target_phase_curve) #phase error for each beam

            # For nulling beams from 47 (considering only the 27 not nulled beams)

            #for kk in range(0,47):
            #    if np.mod(kk, 3) == 0:
            #        err[kk] =0.0
            #        err_phase[kk]=0.0 #phase error for each beam

            calc_full=self.results[-1]*np.exp(1j*self.results_phase[-1])
            target_full=self.amps*np.exp(1j*(self.phases0+cp.array(target_phase_curve)))


            count = 0
            for m in err_phase:
                if self.tem01 and count%2==1:  #Original 0 defined for old linear convention. Changing to 1 for 2Darb convention
                    #err_phase[count] +=  np.pi #original
                    err_phase[count] -= np.pi
                if err_phase[count] < 0 and np.abs(err_phase[count])>np.pi:
                    err_phase[count] += 2 * np.pi
                if err_phase[count] > 0 and np.abs(err_phase[count])>np.pi:
                    err_phase[count] -= 2 * np.pi
                count += 1
            
            print("phases0 MDS",self.phases0)
            print("Calculated phases0 MDS", self.results_phase[-1])
            print("err_phase",err_phase)
            print("cp.array(target_phase_curve)",cp.array(target_phase_curve))
            
            err_all =  1*np.mean(np.abs(err))+1*np.mean(np.abs(err_phase))#set weight for uniform amlitude or for specificphase
            print('amps_error',np.mean(np.abs(err)),'phase_error',np.mean(np.abs(err_phase)))
            print('amps_error list',err )
            print('amps before err', self.amps)
            # now we put weight on the phase
            rel_result = self.amps0 / self.results[-1]
            for j in range(len(self.amps)):
                if self.amps0[j] >= 0.005:
                    self.amps[j] *= cp.abs(rel_result[j]) ** 0.15#0.25 #MDS
                    #self.amps[j] = self.amps[j]+0.1* cp.abs(err[j])#*cp.abs(rel_result[j]) ** (-0.25)#Chane here
                    #self.phases[j] -= err_phase[j] * 0.1
                if i >= 0 and self.amps0[j] >= 0.25:
                    self.phases[j] -= err_phase[j] * 0.1
            self.amps /= cp.max(self.amps)
            print('amps after err',self.amps)

            # print(self.results_phase[-1])
            if False: #original feedback 15/12/2025
                for j in range(len(self.amps)):
                    if i > 0 and self.amps0[j] < 0.25:# and self.amps_record[-1][j] != self.amps_record[-2][j]:
                        grad = cp.abs(self.results[-1][j]) - cp.abs(self.results[-2][j])
                        #grad /= self.amps_record[-1][j] - self.amps_record[-2][j]
                        #self.amps[j] -= 0.1 * grad * err[j] #was 0.1
                        self.amps[j] -= -0.1*err[j]
                        print("zero amp")
                    if i>=0:
                        self.phases[j] -= err_phase[j] *0.1 #Chnage here for phase
                        # print('chan')
            print("Here input")
            if True:
                for j in range(len(self.amps)):
                    if False:
                        if i > 0 and self.amps0[j] < 0.25 and  self.amps0[j] > 0.05:  # and self.amps_record[-1][j] != self.amps_record[-2][j]:
                            grad = cp.abs(self.results[-1][j]) - cp.abs(self.results[-2][j])
                            # grad /= self.amps_record[-1][j] - self.amps_record[-2][j]
                            # self.amps[j] -= 0.1 * grad * err[j] #was 0.1
                            self.amps[j] -= -0.1 * err[j]
                            print(" close to zero amp")
                        if i >= 0 and self.amps0[j] > 0.05:
                            self.phases[j] -= err_phase[j] * 0.1  # Chnage here for phase
                            # print('chan')
                    if i > 0 and self.amps0[j] < 0.0005:# and self.amps0[j] < 0.05:
                        #if i==1:
                        #    self.amps[j]=0.01
                        print("zero amp")
                        grad = cp.abs(self.results[-1][j])**2 - cp.abs(self.results[-2][j])**2
                        #grad /= (self.amps_record[-1][j] - self.amps_record[-2][j])
                        self.amps[j] =self.amps0[j]
                        self.phases[j] = (self.phases0[j])#(1*self.phases0[j])#(self.results_phase[-1][j])*(self.results_phase[-1][j]-self.results_phase[-2][j])#self.results_phase[-1][j]

                # if i > 0 and np.abs(self.results_phase[-1][j] -self.results_phase[-2][j])>0.0001:
                #     
                #     grad = self.record_phase[-1][j] - self.record_phase[-2][j]
                #     grad /= cp.abs(self.results_phase[-1][j]) - cp.abs(self.results_phase[-2][j])
                #     self.phases[j] += 0.1 * grad * err_phase[j]
                #     print('suc')
                # else:
                #     if i > 0:
                #         print(self.results_phase[-1])
            print('Updated target amplitudes: ' + str(self.amps))
            print('Updated target phases: ' + str(self.phases))
            self.amps_record.append(cp.copy(self.amps))
            self.record_phase.append(cp.copy(self.phases))
            
            #plt.imshow(np.abs(res[0].image_field.get())**2)
            #plt.title("image_field_at end")
            #plt.colorbar()
            #plt.show()
            
            #original with wu as an object
            #self.start_phase = cp.angle(res[0].slm_field)#+2 * pi * cp.random.normal(0,0.2,self.size)#MDS added extra random noise for the next round
            self.start_phase = cp.angle(res[0])+2 * pi * cp.random.normal(0,0.00,self.size)#MDS added extra random noise for the next round
            #self.start_phase =res[0]
            #self.wus.append(res[0]) #Original passing the whole object
            self.wusField.append(res[0])
            self.wuseff.append(res[9])
            self.wusnonunif.append(res[10])
            self.wusphase_err.append(res[11])
            self.wusImageField.append(res[12])
            self.wusfidelity_err.append(res[13])
            # if i==34:
            #
            #     plt.figure(figsize=(8, 6))
            #     plt.imshow(((cp.abs(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
            #     plt.colorbar()
            #     plt.title('wu result')
            #     plt.show()

            # plt.figure(figsize=(8, 6))
            # plt.imshow(((cp.angle(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            # plt.title('target phase')
            # plt.show()

            self.eff.append(res[2])
            # self.nonunif.append(res[3])
            self.nonunif.append(err_all)#err err_phase
            #self.phase_err.append(res[4])
            self.phase_err.append(np.mean(np.abs(err_phase)))
            print('Diffraction Efficiency: ' + str(self.eff[-1]))
            print('Total error: ' + str(self.nonunif[-1]))
            print('Phase error: ' + str(self.phase_err[-1]))
            print("M", M,"i,",i)
            if False: #self.nonunif[-1]<0.00001: For stop based on error
                break
            else:
                if i>M:
                    print("Not converged, extra iteration added")

        # Find the minimum nonuniformity run of the Wu algorithm
        self.min_it = self.nonunif.index(min(self.nonunif))
        # self.min_it = len(self.nonunif) - 1

        print('Time to run: ' + str(time.time() - start_time))

        print('')
        print('----Minimum iteration----')
        print('Diffraction Efficiency: ' + str(self.eff[self.min_it]))
        print('Amplitude error: ' + str(self.nonunif[self.min_it]))
        print('Phase error: ' + str(self.phase_err[self.min_it]))
        print('\nTarget phases: 2pi * ' + str(self.phases0 / 2 / pi))
        # Previous wus.phases()
        #print('Actual phases: 2pi * ' + str(((self.wus[self.min_it].phases() + 2 * pi) % (2 * pi)) / 2 / pi))
        #Original wus
        #print('Actual phases: 2pi * ' + str(((self.wus[self.min_it].beams_phase() + 2 * pi) % (2 * pi)) / 2 / pi))

        print('\nTarget amplitudes: ' + str(self.amps0))
        #Previous wus.amps()
        #print('Actual amplitudes: ' + str(self.wus[self.min_it].amps() / cp.max(self.wus[self.min_it].amps())))
        #Original wus
        #print('Actual amplitudes: ' + str(cp.abs(self.wus[self.min_it].beams()) / cp.max(cp.abs(self.wus[self.min_it].beams()))))

        return self.wusField[self.min_it]#self.wus[self.min_it].slm_field #Original with wus as the whole object

    def iterate_updatedloop_outer(self, N=50, M=30):

        self.phases0 = cp.copy(self.phases)

        target_orig = Profile.target_output_array_bokai_Offcenter2Darb(self.m, self.n,
                                                                       center=(0.00, 0.00),
                                                                       input_profile=self.input_profile.get(),
                                                                       x_pitch=self.x_pitch,
                                                                       y_pitch=self.x_pitch,
                                                                       amps=self.amps.get(), phases=self.phases.get(),
                                                                       size=self.size,
                                                                       tem=self.tem01, double_amps=self.tem01, uni_spacing=self.uni_spacing, xarblist0=self.xarblist0, yarblist0=self.yarblist0, anglearblist0=self.anglearblist0)

        # target_New = Profile.target_output_array_bokai_Offcenter2Darb(self.m, self.n,
        #                                                                center=(0, 0),
        #                                                                input_profile=Profile.input_gaussian(beam_size=(0.25*(1/2), 0.25*(1/2)),
        #                                                                      size=(int(1024*2), int(1272*2))),
        #                                                                x_pitch=self.x_pitch,
        #                                                                y_pitch=self.x_pitch,
        #                                                                amps=self.amps.get(), phases=self.phases.get(),
        #                                                                size=self.size,
        #                                                                tem=self.tem01, double_amps=self.tem01)





        print((self.phase_tem_compensation))
        print(type(self.phase_tem_compensation))

        if not (self.phase_tem_compensation is None):
            target_orig[0] = cp.array(target_orig[0]) * cp.array(self.phase_tem_compensation)
            target_phase_curve = []
            for i in target_orig[1]:
                # target_phase_curve.append(cp.angle(target[0][i[0]][i[1]])) #MDS
                target_phase_curve.append(cp.angle(self.phase_tem_compensation[int(i[0])][int(i[1])]))
            print('target phase_curve 2pi*', cp.array(target_phase_curve) / 2 / np.pi)  # MDS
        else:
            target_phase_curve = np.zeros((len(self.amps)))
        target_orig[0] = cp.array(target_orig[0])

        beams0 = self.amps0 * cp.exp(1j * self.phases0)
        beams = self.amps * cp.exp(1j * self.phases)
        P_c_old=[]
        A_c_old=[]

        target = copy.deepcopy(target_orig)

        # plt.imshow(np.abs(target_orig[0].get()))
        # plt.title("target_orig")
        # plt.figure()
        # plt.plot(np.abs(target_orig[0][1024,:].get()))
        # plt.figure()
        # plt.imshow(np.angle(target_orig[0].get()))
        # plt.title("target_orig_angle")
        # plt.show()

        wu = Wu(input=self.input_profile, target=target, size=self.size,
                target_beams=cp.array([cp.abs(beams0)[i] * cp.exp(cp.angle(beams0)[i]) for i in range(len(beams0))]),
                start_phase=self.start_phase, phase_memory=self.phase_memory, phaseconstant=False, P_c_old=P_c_old,
                target_orig=target_orig,tem01=self.tem01)#,target_New=target_New)

        if True:
            wu.A_t=torch.as_tensor(wu.A_t)
            wu.P_t = torch.as_tensor(wu.P_t)
            wu.P_t_orig = torch.as_tensor(wu.P_t_orig)
            wu.A_t_orig = torch.as_tensor(wu.A_t_orig)
            if False:
                if isinstance(wu.target_orig, cp.ndarray):
                    wu.target_orig = torch.as_tensor(wu.target_orig.get())
                else:
                    wu.target_orig = torch.as_tensor(wu.target_orig)
                wu.target=torch.as_tensor(wu.target)
        # plt.imshow(np.abs(self.input_profile.get()))
        # plt.title("input")
        # plt.show()


        method_algorithm=runsettings.method
        if method_algorithm=="wu": #Wu
            print("Method: ", method_algorithm)
            for m in range(M):
                print("M_step_outer ",m)
                wu.outer_num=m
                wu.iterate_updatedloopWu(N)
                # runsettings.exp_amp_global_wu=(runsettings.exp_amp_global_wu)**1
                # if m<100:
                #     runsettings.exp_phase_global_wu = (runsettings.exp_phase_global_wu)**1
                # else:
                #     runsettings.exp_phase_global_wu=0.0

                if True: #Compare these metric to the ones below
                    amps_current = wu.beams(waist=0.0015)
                    amps_current=amps_current/cp.max(amps_current)
                    phase_current = wu.beams_phase(waist=0.0015)
                    print("amps_current", amps_current)
                    print("phase_current", phase_current)
                    err = cp.abs(amps_current) - self.amps0  # amplitude error for each beam
                    err_phase = (phase_current) - self.phases0 - cp.array(
                        target_phase_curve)  # phase error for each beam
                    print('amps_error', np.mean(np.abs(err)), 'phase_error', np.mean(np.abs(err_phase)), 'amps_current_std', cp.std(amps_current), 'phase_current_std', cp.std(phase_current))

                print("wueff:", wu.eff, "| nonunif:", wu.nonunif, "| phase_err:", wu.phase_err, "| image_field:",
                      wu.image_field, "| totalsteperr:", wu.totalsteperr)
                self.wuseff.append(wu.eff)
                self.wusnonunif.append(wu.nonunif)
                self.wusphase_err.append(wu.phase_err)
                self.wusImageField.append(wu.image_field)
                self.wusfidelity_err.append(wu.totalsteperr)
            runsettings.amps_current_std=cp.std(amps_current)

        if method_algorithm=="grad": #Gradient
            print("Method: ", method_algorithm)
            wu.iterate_Gradient(N,self.amps0,self.phases0,target_phase_curve)
            self.wusImageField.append(wu.image_field)
            self.wuseff.append(wu.eff)
            self.wusnonunif.append(wu.nonunif)
            self.wusphase_err.append(wu.phase_err)
            self.wusfidelity_err.append(wu.totalsteperr)

        if method_algorithm=="wu_old_recovery": #Wu
            print("Method: ", method_algorithm)
            for m in range(M):
                print("M_step_outer ",m)
                wu.outer_num=m
                wu.iterate_old_recovery(N)

                if True: #Outer loop is effetively here
                    amps_current = wu.beams(waist=0.0015)
                    amps_current=amps_current/cp.max(amps_current)
                    phase_current = wu.beams_phase(waist=0.0015)
                    print("amps_current", amps_current)
                    print("phase_current", phase_current)
                    err = cp.abs(amps_current) - self.amps0  # amplitude error for each beam
                    err_phase = (phase_current) - self.phases0 - cp.array(
                        target_phase_curve)  # phase error for each beam
                    print('amps_error', np.mean(np.abs(err)), 'phase_error', np.mean(np.abs(err_phase)))

                    for j in range(len(self.amps)):
                        if self.amps0[j] >= 0.005:
                            self.amps[j] *= cp.abs(self.amps0[j]/amps_current[j]) ** 0.15  # 0.25 #MDS
                            # self.amps[j] = self.amps[j]+0.1* cp.abs(err[j])#*cp.abs(rel_result[j]) ** (-0.25)#Chane here
                            # self.phases[j] -= err_phase[j] * 0.1
                        if m >= 0 and self.amps0[j] >= 0.25:
                            self.phases[j] -= err_phase[j] * 0.1#0.1
                    self.amps /= cp.max(self.amps)
                    print('amps after err', self.amps)

                    target_current = Profile.target_output_array_bokai_Offcenter2Darb(self.m, self.n,
                                                                                      center=(0, 0),
                                                                                      input_profile=self.input_profile.get(),
                                                                                      x_pitch=self.x_pitch,
                                                                                      y_pitch=self.x_pitch,
                                                                                      amps=self.amps.get(),
                                                                                      phases=self.phases.get(),
                                                                                      size=self.size,
                                                                                      tem=self.tem01,
                                                                                      double_amps=self.tem01,
                                                                                      uni_spacing=self.uni_spacing,
                                                                                      xarblist0=self.xarblist0,
                                                                                      yarblist0=self.yarblist0,
                                                                                      anglearblist0=self.anglearblist0)

                    wu.target = target_current[0]
                    wu.A_t = torch.as_tensor(wu.target)
                    wu.P_t = torch.as_tensor(wu.target)




                self.wuseff.append(wu.eff)
                self.wusnonunif.append(wu.nonunif)
                self.wusphase_err.append(wu.phase_err)
                self.wusImageField.append(wu.image_field)
                self.wusfidelity_err.append(wu.totalsteperr)



        if method_algorithm=="wu_old_old": #Wu

            Mmax = M
            phaseconstant = False
            for i in range(Mmax):
                print('Iteration: ' + str(i))
                print('amps0', self.amps0)
                # print("iteration:",i,"statrphase",self.start_phase)
                if i == 0:
                    Ntemp = 2
                else:
                    Ntemp = N
                if i > 4:
                    phaseconstant = True
                res = test_amps(m=self.m, n=self.n, N=Ntemp, size=self.size,
                                beams0=self.amps0 * cp.exp(1j * self.phases0),
                                beams=self.amps * cp.exp(1j * self.phases),
                                x_pitch=self.x_pitch, input_profile=self.input_profile,
                                start_phase=self.start_phase,
                                phase_memory=self.phase_memory, tem01=self.tem01, #middlecompensate=self.middle,
                                phase_tem_compensation=self.phase_tem_compensation, phaseconstant=phaseconstant,
                                P_c_old=self.P_c_old, target_orig=target_orig)
                self.results.append(res[1] / np.max(np.abs(res[1])))
                self.results_phase.append(res[5])
                self.P_c_old = res[14]

                self.middle[0] = res[6]

                self.middle[2] = res[8]
                self.middle[1] = cp.array(res[7])  # -cp.array( [0.33845852,0.51371972,0.2987232])
                # print('middlecompensate', self.middle[1])
                # print(np.angle(-1))
                # print(np.angle(self.results[-1]))
                # print((self.results[-1]))

                # Update the target amplitudes to compensate the nonuniformity of the previous iteration
                print('Calculated phases: 2pi * ' + str(np.angle(self.results[-1]) / 2 / np.pi))
                print('Calculated amplitudes: ' + str(np.abs(self.results[-1]) / np.max(np.abs(self.results[-1]))))

                err = cp.abs(self.results[-1]) - self.amps0  # amplitude error for each beam
                err_phase = (self.results_phase[-1]) - self.phases0 - cp.array(
                    target_phase_curve)  # phase error for each beam

                # For nulling beams from 47 (considering only the 27 not nulled beams)

                # for kk in range(0,47):
                #    if np.mod(kk, 3) == 0:
                #        err[kk] =0.0
                #        err_phase[kk]=0.0 #phase error for each beam

                calc_full = self.results[-1] * np.exp(1j * self.results_phase[-1])
                target_full = self.amps * np.exp(1j * (self.phases0 + cp.array(target_phase_curve)))

                count = 0
                for m in err_phase:
                    if self.tem01 and count % 2 == 1:  # Original 0 defined for old linear convention. Changing to 1 for 2Darb convention
                        # err_phase[count] +=  np.pi #original
                        err_phase[count] -= np.pi
                    if err_phase[count] < 0 and np.abs(err_phase[count]) > np.pi:
                        err_phase[count] += 2 * np.pi
                    if err_phase[count] > 0 and np.abs(err_phase[count]) > np.pi:
                        err_phase[count] -= 2 * np.pi
                    count += 1

                print("phases0 MDS", self.phases0)
                print("Calculated phases0 MDS", self.results_phase[-1])
                print("err_phase", err_phase)
                print("cp.array(target_phase_curve)", cp.array(target_phase_curve))

                err_all = 1 * np.mean(np.abs(err)) + 1 * np.mean(
                    np.abs(err_phase))  # set weight for uniform amlitude or for specificphase
                print('amps_error', np.mean(np.abs(err)), 'phase_error', np.mean(np.abs(err_phase)))
                print('amps_error list', err)
                print('amps before err', self.amps)
                # now we put weight on the phase
                rel_result = self.amps0 / self.results[-1]
                for j in range(len(self.amps)):
                    if self.amps0[j] >= 0.005:
                        self.amps[j] *= cp.abs(rel_result[j]) ** 0.15  # 0.25 #MDS
                        # self.amps[j] = self.amps[j]+0.1* cp.abs(err[j])#*cp.abs(rel_result[j]) ** (-0.25)#Chane here
                        # self.phases[j] -= err_phase[j] * 0.1
                    if i >= 0 and self.amps0[j] >= 0.25:
                        self.phases[j] -= err_phase[j] * 0.1
                self.amps /= cp.max(self.amps)
                print('amps after err', self.amps)

                # print(self.results_phase[-1])
                if False:  # original feedback 15/12/2025
                    for j in range(len(self.amps)):
                        if i > 0 and self.amps0[j] < 0.25:  # and self.amps_record[-1][j] != self.amps_record[-2][j]:
                            grad = cp.abs(self.results[-1][j]) - cp.abs(self.results[-2][j])
                            # grad /= self.amps_record[-1][j] - self.amps_record[-2][j]
                            # self.amps[j] -= 0.1 * grad * err[j] #was 0.1
                            self.amps[j] -= -0.1 * err[j]
                            print("zero amp")
                        if i >= 0:
                            self.phases[j] -= err_phase[j] * 0.1  # Chnage here for phase
                            # print('chan')
                print("Here input")
                if True:
                    for j in range(len(self.amps)):
                        if False:
                            if i > 0 and self.amps0[j] < 0.25 and self.amps0[
                                j] > 0.05:  # and self.amps_record[-1][j] != self.amps_record[-2][j]:
                                grad = cp.abs(self.results[-1][j]) - cp.abs(self.results[-2][j])
                                # grad /= self.amps_record[-1][j] - self.amps_record[-2][j]
                                # self.amps[j] -= 0.1 * grad * err[j] #was 0.1
                                self.amps[j] -= -0.1 * err[j]
                                print(" close to zero amp")
                            if i >= 0 and self.amps0[j] > 0.05:
                                self.phases[j] -= err_phase[j] * 0.1  # Chnage here for phase
                                # print('chan')
                        if i > 0 and self.amps0[j] < 0.0005:  # and self.amps0[j] < 0.05:
                            # if i==1:
                            #    self.amps[j]=0.01
                            print("zero amp")
                            grad = cp.abs(self.results[-1][j]) ** 2 - cp.abs(self.results[-2][j]) ** 2
                            # grad /= (self.amps_record[-1][j] - self.amps_record[-2][j])
                            self.amps[j] = self.amps0[j]
                            self.phases[j] = (self.phases0[
                                j])  # (1*self.phases0[j])#(self.results_phase[-1][j])*(self.results_phase[-1][j]-self.results_phase[-2][j])#self.results_phase[-1][j]

                    # if i > 0 and np.abs(self.results_phase[-1][j] -self.results_phase[-2][j])>0.0001:
                    #
                    #     grad = self.record_phase[-1][j] - self.record_phase[-2][j]
                    #     grad /= cp.abs(self.results_phase[-1][j]) - cp.abs(self.results_phase[-2][j])
                    #     self.phases[j] += 0.1 * grad * err_phase[j]
                    #     print('suc')
                    # else:
                    #     if i > 0:
                    #         print(self.results_phase[-1])
                print('Updated target amplitudes: ' + str(self.amps))
                print('Updated target phases: ' + str(self.phases))
                self.amps_record.append(cp.copy(self.amps))
                self.record_phase.append(cp.copy(self.phases))

                # plt.imshow(np.abs(res[0].image_field.get())**2)
                # plt.title("image_field_at end")
                # plt.colorbar()
                # plt.show()

                # original with wu as an object
                # self.start_phase = cp.angle(res[0].slm_field)#+2 * pi * cp.random.normal(0,0.2,self.size)#MDS added extra random noise for the next round
                self.start_phase = cp.angle(res[0]) + 2 * pi * cp.random.normal(0, 0.00,
                                                                                self.size)  # MDS added extra random noise for the next round
                # self.start_phase =res[0]
                # self.wus.append(res[0]) #Original passing the whole object
                self.wusField.append(res[0])
                self.wuseff.append(res[9])
                self.wusnonunif.append(res[10])
                self.wusphase_err.append(res[11])
                self.wusImageField.append(res[12])
                self.wusfidelity_err.append(res[13])
                # if i==34:
                #
                #     plt.figure(figsize=(8, 6))
                #     plt.imshow(((cp.abs(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
                #     plt.colorbar()
                #     plt.title('wu result')
                #     plt.show()

                # plt.figure(figsize=(8, 6))
                # plt.imshow(((cp.angle(res[0].image_field)).get()), cmap='viridis', interpolation='nearest')
                # plt.colorbar()
                # plt.title('target phase')
                # plt.show()

                self.eff.append(res[2])
                # self.nonunif.append(res[3])
                self.nonunif.append(err_all)  # err err_phase
                # self.phase_err.append(res[4])
                self.phase_err.append(np.mean(np.abs(err_phase)))
                print('Diffraction Efficiency: ' + str(self.eff[-1]))
                print('Total error: ' + str(self.nonunif[-1]))
                print('Phase error: ' + str(self.phase_err[-1]))
                print("M", M, "i,", i)
                if False:  # self.nonunif[-1]<0.00001: For stop based on error
                    break
                else:
                    if i > M:
                        print("Not converged, extra iteration added")

            # Find the minimum nonuniformity run of the Wu algorithm
            self.min_it = self.nonunif.index(min(self.nonunif))
            # self.min_it = len(self.nonunif) - 1

            print('Time to run: ' + str(time.time() - start_time))

            print('')
            print('----Minimum iteration----')
            print('Diffraction Efficiency: ' + str(self.eff[self.min_it]))
            print('Amplitude error: ' + str(self.nonunif[self.min_it]))
            print('Phase error: ' + str(self.phase_err[self.min_it]))
            print('\nTarget phases: 2pi * ' + str(self.phases0 / 2 / pi))
            # Previous wus.phases()
            # print('Actual phases: 2pi * ' + str(((self.wus[self.min_it].phases() + 2 * pi) % (2 * pi)) / 2 / pi))
            # Original wus
            # print('Actual phases: 2pi * ' + str(((self.wus[self.min_it].beams_phase() + 2 * pi) % (2 * pi)) / 2 / pi))

            print('\nTarget amplitudes: ' + str(self.amps0))
            # Previous wus.amps()
            # print('Actual amplitudes: ' + str(self.wus[self.min_it].amps() / cp.max(self.wus[self.min_it].amps())))
            # Original wus
            # print('Actual amplitudes: ' + str(cp.abs(self.wus[self.min_it].beams()) / cp.max(cp.abs(self.wus[self.min_it].beams()))))

        print(type(target_orig[0]))
        Final_fidelity =((cp.abs(cp.sum(cp.conj(wu.image_field) * (cp.asarray(target_orig[0])) * cp.asarray(wu.ones_box))) ** 2 / (
                (cp.abs(cp.sum(cp.conj(wu.image_field) * wu.image_field * cp.asarray(wu.ones_box)))) * cp.abs(cp.sum(cp.conj((cp.asarray(target_orig[0]))) * (cp.asarray(target_orig[0])) * cp.asarray(wu.ones_box))))).item())

        Final_efficiency=cp.sum(cp.abs(cp.conj(wu.image_field) * wu.image_field) * cp.asarray(wu.mask))/cp.sum(cp.abs(cp.conj(wu.image_field) * wu.image_field))
        
        Final_ion_fidelity =((cp.abs(cp.sum(cp.conj(wu.image_field) * (cp.asarray(target_orig[0])) * cp.asarray(wu.ion_mask))) ** 2 / (
                (cp.abs(cp.sum(cp.conj(wu.image_field) * wu.image_field * cp.asarray(wu.ion_mask)))) * cp.abs(cp.sum(cp.conj((cp.asarray(target_orig[0]))) * (cp.asarray(target_orig[0])) * cp.asarray(wu.ion_mask))))).item())


        print("Final check: Fidelity = {:.8f}, Efficiency = {:.8f}, Ion Fidelity = {:.8f}".format(Final_fidelity, Final_efficiency,Final_ion_fidelity))
        print("Final check: Fid= {:.8f}, Eff= {:.8f}, Ion Fid= {:.8f}".format(Final_fidelity,
                                                                                                  Final_efficiency,
                                                                                                  Final_ion_fidelity))
        global_variables.final_eff_curvature.append(Final_efficiency.get())
        global_variables.final_ion_fidelity_curvature.append(Final_ion_fidelity)
        runsettings.Final_box_fidelity = Final_fidelity
        runsettings.Final_efficiency = Final_efficiency
        runsettings.Final_ion_fidelity = Final_ion_fidelity
        self.min_it=len(self.wusImageField)-1
        return wu.slm_field#self.wusField[self.min_it]  # self.wus[self.min_it].slm_field #Original with wus as the whole object


    def plot(self, show=False, figs=(None, None, None)):
        plots= ()
        if show:
            plots = (0, 1, 2, 3, 4, 5)
        #self.wus[self.min_it].save_pattern(name=self.name, slm=self.slm, target=False, correction=False, show=(), wavelength=self.wavelength,
        #                         field=True, plots=plots)

        # print(figs[2])
        #plot_gradient(self.wus[self.min_it].image_field, fig=figs[2], intensity=True, imag=self.imag)#orig wus object
        plot_gradient(self.wusImageField[self.min_it], fig=figs[2], intensity=True, imag=self.imag,phase_tem_compensation=self.phase_tem_compensation)

        # Plot the efficiency, nonuniformity and phase error for the best iteration
        if figs[0] is None:
            plt.figure()
            plt.clf()
            plotter = plt
        else:
            figs[0].axes.cla()
            plotter = figs[0].axes
        #Original with wu as object
        #plotter.plot([n for n in range(len(self.wus[self.min_it].eff))], cp.array(self.wus[self.min_it].eff).get(), label='Efficiency')
        # plotter.plot([n for n in range(len(self.wuseff[self.min_it]))], cp.array(self.wusnonunif[self.min_it]).get(), label='nonunif')
        #
        #
        # plotter.plot([n for n in range(len(self.wusfidelity_err[self.min_it]))], cp.array(self.wusphase_err[self.min_it]).get(),
        #          label='phase error')
        print("type(self.wusnonunif[self.min_it])",type(self.wusnonunif[self.min_it]))
        #print("type(self.wusnonunif[self.min_it][0])", type(self.wusnonunif[self.min_it][0]))
        #plotter.plot([n for n in range(len(self.wuseff[self.min_it]))], cp.array(self.wuseff[self.min_it]).get(), label='Efficiency')
        plotter.plot([n for n in range(len(self.wuseff[self.min_it]))], np.array(self.wuseff[self.min_it]),
                     label='Efficiency')
        plotter.plot([n for n in range(len(self.wusnonunif[self.min_it]))], cp.array(self.wusnonunif[self.min_it]).get(),
                     label='Nonuniformity')


        plotter.plot([n for n in range(len(self.wusfidelity_err[self.min_it]))], cp.array(self.wusfidelity_err[self.min_it]).get(),
                 label='Total fidelity error')

        if figs[0] is None:
            plotter.xlabel('Inner Iteration')
            plotter.title('Inner Loop Convergence')
 #           plotter.xlim(0, len(self.wus[0].eff))
        else:
            plotter.set_xlabel('Inner Iteration')
            plotter.set_title('Inner Loop Convergence')
 #           plotter.set_xlim(0, len(self.wus[0].eff))
        plotter.grid(True)
        plotter.legend()
        if figs[0] is None:
            plotter.pause(.001)
            plotter.savefig('images/inner_convergence.png')
        # Plot the final efficiency, nonuniformity and phase error across all iterations
        if figs[1] is None:
            plt.figure()
            plt.clf()
            plotter = plt
        else:
            figs[1].axes.cla()
            plotter = figs[1].axes
        #plotter.plot([m for m in range(len(self.wusField))], cp.array(self.eff).get(), label='Efficiency')
        #plotter.plot([m for m in range(len(self.wusField))], cp.array(self.nonunif).get(), label='Amplitude error')
        #plotter.plot([m for m in range(len(self.wusField))], cp.array(self.phase_err).get(), label='Phase error')

        plotter.plot([m for m in range(len(self.eff))], cp.array(self.eff).get(), label='Efficiency')
        plotter.plot([m for m in range(len(self.nonunif))], cp.array(self.nonunif).get(), label='Amplitude error')
        plotter.plot([m for m in range(len(self.phase_err))], cp.array(self.phase_err).get(), label='Phase error')

        if figs[1] is None:
            plotter.xlabel('Outer Iteration')
            plotter.title('Outer Loop Convergence')
            plotter.xlim(0, len(self.wusField))
        else:
            plotter.set_xlabel('Outer Iteration')
            plotter.set_title('Outer Loop Convergence')
            plotter.set_xlim(0, len(self.wusField))
        plotter.grid(True)
        plotter.legend()
        if figs[1] is None:
            plotter.savefig('images/outer_convergence.png')

        if show:
            plt.show()


# Generate single TEM01 beam
def tem01(slm, size=(0.05, 0.05)):
    wgs = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)))
    wgs.phi[-1] = slm.half()
    wgs.slm_field = wgs.input * cp.exp(1j * wgs.phi[-1])
    wgs.propagate()
    wgs.save_pattern('TEM01', slm, correction=True)

    return wgs


# WGS 2D array of beams
def array2D(slm, n=4, m=4, x_pitch=0.02, y_pitch=0.02, size=(0.05, 0.05)):
    wgs = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)),
              target=Profile.spot_array(m, n, x_pitch=y_pitch, y_pitch=x_pitch))
    wgs.iterate(20)
    wgs.save_pattern('%dx%d' % (n, m), slm)

    return wgs


# WGS 1D array of beams
def array1D(slm, it=20, tries=1, n=5, pitch=0.004, size=(0.05, 0.05), start=None, ref=None, consider_phase=False, waist=0.01, plots=(0, 1, 2, 3), add_noise=False):
    wgs = []
    min = 0
    start = [start]
    for i in range(tries):
        if add_noise:
            start.append(start[0] + (cp.random.random_sample(slm.size) - 0.5) * 0.1)
        wgs.append(WGS(input=Profile.input_gaussian(beam_size=np.array(size)),
                       target=Profile.spot_array(1, n, y_pitch=pitch, center=(0.05, 0)), start_phase=start[-1], reference=None, consider_phase=consider_phase, waist=waist))
        wgs[-1].iterate(it)
        if wgs[min].min_dev[6] > wgs[-1].min_dev[6]:
            min = i
    wgs[min].save_pattern('1x%d' % n, slm, correction=True, plots=plots, min=True)
    wgs[min].A, wgs[min].phi, wgs[min].B, wgs[min].psi = wgs[min].min_dev[1:5]
    wgs[min].slm_field = wgs[min].A * cp.exp(wgs[min].phi * 1j)
    wgs[min].image_field = wgs[min].B * cp.exp(wgs[min].psi * 1j)

    print('Minimum amplitude non-uniformity at iteration %d out of %d' % (wgs[min].min_dev[0], it))
    print('Min Amplitude non-uniformity: %f' % wgs[min].min_dev[5])
    print('Phase non-uniformity: %f*Pi' % wgs[min].min_dev[6])
    print('Phases in last iteration: ' + str([cp.angle(wgs[min].avg(wgs[min].image_field, m, waist)) / pi for m in wgs[min].spots]) + ' * pi')

    plt.figure(1)
    plt.clf()
    plt.plot(wgs[min].it, wgs[min].B_dev, label='Amplitude non-uniformity')
    plt.plot(wgs[min].it, wgs[min].psi_dev, label='Phase non-uniformity')
    plt.title('Convergence')
    plt.ylabel('Non-uniformity')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.legend()
    plt.show()

    return wgs[min]


# WGS 2D array of TEM01 beams
def tem01_2D(slm, n=1, m=5, wgs=None, size=(0.05, 0.05)):
    if wgs is None:
        wgs = array2D(slm, n, m)

    wgs2 = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)))
    wgs2.phi[-1] = slm.add(wgs.phi[-1], slm.half())
    wgs2.slm_field = wgs2.input * cp.exp(1j * wgs2.phi[-1])
    wgs2.propagate()
    wgs2.save_pattern('%dx%dTEM01' % (n, m), slm)
    slm.phaseToBMP(wgs2.phi[-1], name='%dx%dTEM01_input_phase' % (n, m), correction=True)


# Simulate interference pattern of WGS beam array
def interfere_wgs(slm):
    array1x5 = array1D(slm, 5, pitch=0.1)
    reference = Profile(field=Profile.input_gaussian(beam_type=0))
    interference = Profile(field=reference.field + array1x5.image_field)
    reference.save(slm, 'reference')
    interference.save(slm, 'interference_wgs')
    return interference.field


# Simulate reference interference pattern
def interfere_ref(slm, phases=np.array([0, 0.35, 0, 1 - 0.35]), global_phase=0.25):
    """
    :param slm: SLM
    :param phases: phases of each beam, normalized between 0 and 1
    :param global_phase: global_phase of the entire beam array, normalized between 0 and 1
    """
    phases = 2 * pi * phases
    phases += 2 * pi * global_phase
    array1x5 = Profile(field=Profile.target_output_array(1, len(phases), input_profile=Profile.input_gaussian(beam_size=(0.03, 0.03)), x_pitch=0.1, phases=phases)[0]).field
    reference = Profile(field=Profile.input_gaussian(beam_size=(0.8, 0.8)))
    reference.field /= np.max(reference.field)
    array1x5 /= np.max(np.abs(array1x5))
    array1x5 *= 0.5
    interference = Profile(field=reference.field + array1x5)

    reference.field *= 0.5
    array1x5 *= 0.5
    interference.field *= 0.5

    inter = np.abs(interference.field) ** 2
    intensity = np.abs(array1x5) ** 2
    ref = np.abs(reference.field) ** 2

    slm.ampToBMP(ref, name='reference', color=False)
    slm.ampToBMP(intensity, name='intensity', color=False)
    slm.ampToBMP(inter, name='interference', color=False)
    np.savetxt('images/reference.txt', ref)
    np.savetxt('images/intensity.txt', intensity)
    np.savetxt('images/interference.txt', inter)

    # slm.phaseToBMP(np.angle(array1x5), name='beam array phase', color=True, wavelength=411)
    inter *= 255
    intensity *= 255
    ref *= 255
    inter = np.array(inter, dtype=np.uint8)
    intensity = np.array(intensity, dtype=np.uint8)
    ref = np.array(ref, dtype=np.uint8)
    inter = np.array(inter, dtype=np.float64)
    intensity = np.array(intensity, dtype=np.float64)
    ref = np.array(ref, dtype=np.float64)
    cosphi = (inter - intensity - ref) / (2 * np.sqrt(intensity * ref))
    cosphi = np.nan_to_num(cosphi)
    cosphi = np.clip(cosphi, -1, 1)
    phase = np.arccos(cosphi)
    phase = np.nan_to_num(phase)
    phase -= 2 * pi * 0.25

    print(np.max(cosphi), np.min(cosphi))
    print(np.max(phase), np.min(phase))
    slm.phaseToBMP(phase, name='beam array extracted phase', color=True, wavelength=411)

    # plt.figure()
    pk = np.where(intensity == np.max(intensity))
    ref_dat = ref[pk[0], :][0]
    intensity_dat = intensity[pk[0], :][0]
    interference_dat = inter[pk[0], :][0]
    phase_dat = phase[pk[0], :][0]
    ax = plt.figure().add_subplot(111)
    twinax = plt.twinx()
    ax.plot([i for i in range(len(ref_dat))], ref_dat, label='Reference')
    ax.plot([i for i in range(len(intensity_dat))], intensity_dat, label='Intensities')
    ax.plot([i for i in range(len(interference_dat))], interference_dat, label='Interference')
    twinax.plot([i for i in range(len(phase_dat))], phase_dat / 2 / pi, label='Phase', c='r')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Intensity')
    ax.set_title('1x4  Phase Beam Array')
    twinax.set_ylabel('Phase ($2\\pi$ radians)')
    ax.legend()
    twinax.legend()



# Generate 1D array using OutputOutput algorithm (unfinished)
def array1D_OutputOutput(slm, it=20, n=5, pitch=0.02, size=(0.05, 0.05), start=None, ref=None, waist=0.01, plots=(0, 1, 2, 3)):

    oo = OutputOutput(input=Profile.input_gaussian(beam_size=cp.array(size)),
                      target=Profile.gaussian_array(1, 5, x_pitch=pitch, waist=(waist, waist)), waist=waist)

    print(oo.spots)

    oo.iterate(it)

    oo.save_pattern('1x%d' % n, slm, correction=True, plots=plots, min=True)

    print('Minimum amplitude non-uniformity at iteration %d out of %d' % (oo.min_dev[0], it))
    print('Min Amplitude non-uniformity: %f' % oo.min_dev[5])
    print('Phase non-uniformity: %f*Pi' % oo.min_dev[6])

    plt.figure(1)
    plt.clf()
    plt.plot(oo.it, oo.B_dev, label='Amplitude non-uniformity')
    plt.plot(oo.it, oo.psi_dev, label='Phase non-uniformity')
    plt.title('Image Convergence')
    plt.ylabel('Non-uniformity')
    plt.xlabel('Elapsed iterations')
    plt.yscale('log')
    plt.legend()
    plt.show()

    return oo


def test_amps_MDS(m=4, n=5, N=60, size=(1024, 1272), beams0=(1., 1., 1., 1., 1.), beams=(1., 1., 1., 1., 1.), x_pitch=0.004,y_pitch=0.004,
              input_profile=None, start_phase=None, phase_memory=False, tem01=False,
              middlecompensate=[[0], [0], [[512, 636]]], phase_tem_compensation=None):
    if input_profile is None:
        input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
        input_profile = cp.array(input_profile)
        input_profile /= cp.sqrt(cp.sum(cp.abs(input_profile) ** 2))

    # Generate a target array
    # print('beasms',cp.angle(beams).get())
    # target = Profile.target_output_array(m, n, center=(0, 0), input_profile=input_profile.get(), x_pitch=x_pitch, y_pitch=x_pitch,
    #                                      amps=cp.abs(beams).get(), phases=cp.angle(beams).get(), size=np.array(size),
    #                                      tem=tem01, double_amps=tem01)
    target = Profile.target_output_array_MDS(m, n, middlecompensate=middlecompensate, center=(0, 0),
                                               input_profile=input_profile.get(), x_pitch=x_pitch,
                                               y_pitch=y_pitch,
                                               amps=cp.abs(beams).get(), phases=cp.angle(beams).get(),
                                               size=np.array(size),
                                               tem=tem01, double_amps=tem01)
    if not (phase_tem_compensation is None):
        target[0] = cp.array(target[0]) * cp.array(phase_tem_compensation)
    target[0] = cp.array(target[0])
    # target_phase_curve=[]
    # for i in target[1]:
    #     target_phase_curve.append(cp.angle(target[0][i[0]][i[1]]))
    # print('target phase',target_phase_curve)

    # Run the Wu algorithm
    wu = Wu(input=input_profile, target=target, size=size,
            target_beams=cp.array([cp.abs(beams0)[i] * cp.exp(cp.angle(beams0)[i]) for i in range(len(beams0))]),
            start_phase=start_phase, phase_memory=phase_memory)
    wu.iterate(N)
    print()

    result = wu.beams(waist=0.0015)
    result_phase = wu.beams_phase(waist=0.0015)

    # Store the efficiency, nonuniformity and phase error for each iteration
    eff = wu.eta()
    nonunif = wu.dev_amp(waist=0.001, target=cp.abs(beams0))
    phase_err = wu.dev_phase(waist=0.001, target=cp.angle(beams0))
    print("beams:", cp.angle(beams0))  # MDS

    compensate_phase = []
    compensate_amps = []
    shift = []
    for i in range(len(beams0) - 1):
        if (tem01 == 1 and i % 2 != 0) or tem01 == 0:
            shift.append(((target[1][i][0] + target[1][i + 1][0]) / 2, (target[1][i][1] + target[1][i + 1][1]) / 2))

            compensate_phase.append(cp.angle((wu.avg(cp.abs(wu.image_field), (
            (target[1][i][0] + target[1][i + 1][0]) / 2, (target[1][i][1] + target[1][i + 1][1]) / 2),
                                                     0.001)) + 2 * pi) % (2 * pi))
            compensate_amps.append(cp.abs((wu.avg(cp.abs(wu.image_field), (
            (target[1][i][0] + target[1][i + 1][0]) / 2, (target[1][i][1] + target[1][i + 1][1]) / 2),
                                                  0.001))) / cp.abs(
                (wu.avg(cp.abs(wu.image_field), target[1][i], 0.001))))

    print('middle phase', compensate_phase)
    print('middle amps', compensate_amps)

    return [wu, result, eff, nonunif, phase_err, result_phase, compensate_phase, compensate_amps, shift]


# Test a set of target amplitudes and phases using the inner/Wu algorithm
def test_amps(m=4,n=5, N=60, size=(1024,1272), beams0=(1., 1., 1., 1., 1.), beams=(1., 1., 1., 1., 1.), x_pitch=0.004,
              input_profile=None, start_phase=None, phase_memory=False, tem01=False,middlecompensate=[[0], [0],[[512,636]]],phase_tem_compensation=None,phaseconstant=False,P_c_old=None,outeriter_num=0.0,target_orig=None):
    if input_profile is None:
        #input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
        input_profile = Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size))
        input_profile = cp.array(input_profile)
        input_profile /= cp.sqrt(cp.sum(cp.abs(input_profile)**2))

    # Generate a target array
    # print('beasms',cp.angle(beams).get())
    #target = Profile.target_output_array(m, n, center=(0, 0), input_profile=input_profile.get(), x_pitch=x_pitch, y_pitch=x_pitch,
    #                                     amps=cp.abs(beams).get(), phases=cp.angle(beams).get(), size=np.array(size),
    #                                     tem=tem01, double_amps=tem01)
    #target = Profile.target_output_array_bokai(m, n,middlecompensate=middlecompensate,center=(0, 0), input_profile=input_profile.get(), x_pitch=x_pitch,
    #                                     y_pitch=x_pitch,
    #                                     amps=cp.abs(beams).get(), phases=cp.angle(beams).get(), size=np.array(size),
    #                                     tem=tem01, double_amps=tem01)
    #target = Profile.target_output_array_bokai_Offcenter(m, n,middlecompensate=middlecompensate,center=(0, 0), input_profile=input_profile.get(), x_pitch=x_pitch,
    #                                     y_pitch=x_pitch,
    #                                     amps=cp.abs(beams).get(), phases=cp.angle(beams).get(), size=np.array(size),
    #                                     tem=tem01, double_amps=tem01) #    Use this!! Commented 09/12/2025
    target = Profile.target_output_array_bokai_Offcenter2Darb(m, n,middlecompensate=middlecompensate,center=(0, 0), input_profile=input_profile.get(), x_pitch=x_pitch,
                                         y_pitch=x_pitch,
                                         amps=cp.abs(beams).get(), phases=cp.angle(beams).get(), size=np.array(size),
                                         tem=tem01, double_amps=tem01) #    Commented 09/12/2025
    if not (phase_tem_compensation is None):
        target[0] = cp.array(target[0])*cp.array(phase_tem_compensation)
    target[0] = cp.array(target[0])
    # target_phase_curve=[]
    # for i in target[1]:
    #     target_phase_curve.append(cp.angle(target[0][i[0]][i[1]]))
    # print('target phase',target_phase_curve)
    # plt.imshow(np.abs(target[0].get()))
    # plt.title("target")
    # plt.figure()
    # plt.imshow(np.angle(target[0].get()))
    # plt.title("target angle")
    # plt.show()

    # Run the Wu algorithm
    wu = Wu(input=input_profile, target=target, size=size,
            target_beams=cp.array([cp.abs(beams0)[i] * cp.exp(cp.angle(beams0)[i]) for i in range(len(beams0))]),
            start_phase=start_phase, phase_memory=phase_memory,phaseconstant=phaseconstant,P_c_old=P_c_old,target_orig=target_orig)
    wu.iterate(N)
    print()
    print("start beam")
    result = wu.beams(waist=0.0015)# (waist=0.001)#MDS changed to 0.004, was 0.001 initiallt
    print("beamsphase start here")
    result_phase = wu.beams_phase(waist=0.0015)# (waist=0.001)#MDS changed to 0.004, was 0.001 initiallt
    #print("beamsphase end here")
    # Store the efficiency, nonuniformity and phase error for each iteration
    eff = wu.eta()
    nonunif = wu.dev_amp(waist=0.001, target=cp.abs(beams0))
    phase_err = wu.dev_phase(waist=0.001, target=cp.angle(beams0))
    print("beams:",cp.angle(beams0)) #  MDS
    
    compensate_phase=[]
    compensate_amps=[]
    shift=[]
    for i in range(len(beams0)-1):
        if (tem01==1 and i%2!=0) or tem01==0:
            shift.append(((target[1][i][0]+target[1][i+1][0])/2,(target[1][i][1]+target[1][i+1][1])/2))
            
            compensate_phase.append(cp.angle((wu.avg(cp.abs(wu.image_field),((target[1][i][0]+target[1][i+1][0])/2,(target[1][i][1]+target[1][i+1][1])/2), 0.001)) + 2 * pi) % (2 * pi))      
            compensate_amps.append(cp.abs((wu.avg(cp.abs(wu.image_field), ((target[1][i][0]+target[1][i+1][0])/2,(target[1][i][1]+target[1][i+1][1])/2), 0.001)))/cp.abs((wu.avg(cp.abs(wu.image_field), target[1][i], 0.001))))
    
    print('middle phase', compensate_phase)
    print('middle amps', compensate_amps)
    #plt.imshow(np.angle(wu.slm_field.get()))
    #plt.title("wu.slm_field")
    #plt.show(block=False)
    #plt.pause(2)
    #plt.imshow(np.abs(wu.image_field.get()))
    #plt.title("wu.slm_field")
    #plt.show()
    #plt.pause(2)
    
    # Original passing full wu object
    #return [wu, result, eff, nonunif, phase_err,result_phase,compensate_phase,compensate_amps,shift]
    return [wu.slm_field, result, eff, nonunif, phase_err,result_phase,compensate_phase,compensate_amps,shift,wu.eff,wu.nonunif,wu.phase_err,wu.image_field,wu.totalsteperr,wu.P_c_old]

# def gradient_descent_1darray(M=20, amps=None, phases=None, ):


# Plot the horizontal electric field gradient at the center of the image
def plot_gradient(field, coord=False, axis=0, target=None, fig=None, intensity=False, norm_coords=False, imag=True, norm=False, grid=False,phase_tem_compensation=None):
    # print(fig)
    if True:
        plt.imshow(np.abs(field.get())**2)
        plt.title("Final field intensity")
        plt.colorbar()
        plt.show()
        plt.imshow(np.abs(field.get()))
        plt.title("Final field magnitude")
        plt.colorbar()
        plt.figure()
        plt.plot(np.abs(field[1024,:].get()))
        plt.show()
        plt.imshow(np.angle(field.get()))
        plt.title("Final field angle")
        plt.colorbar()
        plt.show()
        if phase_tem_compensation is not None:
            plt.imshow(np.mod(np.angle(field.get())-np.angle(phase_tem_compensation),2*np.pi))
            plt.title("Final field angle - phase tem compensation")
            plt.colorbar()
            plt.show()
    if coord:
        coord = np.where(np.abs(field) == np.max(np.abs(field)))[0][0]
        # print(coord)
    else:
        coord = np.where(np.abs(field) == np.max(np.abs(field)))[0][0]
        #coord = len(field) / 2 - 1
    # E_t = np.real(target_field[len(target_field) / 2 - 1, :].get())
    E_im = field[coord, :].get()
    E_im /= np.max(np.abs(E_im))
    if norm:
        E_im /= np.max(np.abs(E_im))
    # plt.figure()

    plt.figure()
    plt.plot(np.arange(len(E_im)),np.abs(E_im))
    plt.plot(np.arange(len(E_im)), np.where(np.abs(E_im)>0.1,np.mod(np.angle(E_im)-np.angle(phase_tem_compensation[int(coord), :])+1,2*np.pi),0))
    plt.title("Phase crossection")

    if fig is None:
        plt.figure()
        plt.clf()
        plotter = plt
    else:
        fig.axes.cla()
        plotter = fig.axes
    x_axis = np.array([i - len(field[1]) / 2 for i in range(len(field[1]))])
    xgrad_axis = np.array([i + 0.5 - len(field[1]) / 2 for i in range(len(field[1]) - 1)])
    if norm_coords:
        x_axis *= 2 / len(field[1])
        xgrad_axis *= 2 / len(field[1])
    if imag:
        plotter.plot(x_axis, np.real(E_im), label='E field (real)')
        plotter.plot(x_axis, np.imag(E_im), label='E field (imag)')
        plotter.plot(x_axis, np.abs(E_im), label='E field (abs)')
    else:
        plotter.plot(x_axis, np.real(E_im), label='$\\vec E$')
    # plt.plot([i + 0.5 for i in range(len(target_field[1]) - 1)], [E_t[i + 1] - E_t[i] for i in range(len(E_t) - 1)],
    #          label='E gradient (target)')
    grad = np.array([E_im[i + 1] - E_im[i] for i in range(len(E_im) - 1)])
    if norm:
        grad /= 2 / len(field[1])
    if imag:
        plotter.plot(xgrad_axis, np.real(grad), label='E x-gradient (real)')
        plotter.plot(xgrad_axis, np.imag(grad), label='E x-gradient (imag)')
    else:
        plotter.plot(xgrad_axis, np.real(grad), label='$\\frac{d\\vec E}{dx}$')
    if intensity:
        intens = np.abs(E_im)**2
        intens *= np.max([np.max(np.real(E_im)), np.max(np.imag(E_im))]) / np.max(intens)
        plotter.plot(x_axis, intens, label='Intensity')
    plotter.legend()
    if grid:
        plotter.grid()
    if fig is None:
        plotter.xlabel('$x$')
        plotter.title('TEM$_{01}$ Electric Field Cross-section')
        plotter.xlim(np.min(x_axis), np.max(x_axis))
        plotter.pause(0.001)
    else:
        plotter.set_xlabel('$x$')
        plotter.set_xlim(np.min(x_axis), np.max(x_axis))
        plotter.set_title('Electric Field Cross-section')


# Generate an array of beams using the Wu algorithm
def wu(slm, N=40, M=20, n=5, plot_each=False, size=(1024, 1272), res_factor=1, wavelength=413, show=True, name='wu_1x5',
       amps=(1., 1., 1., 1., 1.), amps_guess=(1., 1., 1., 1., 1.), phases=(0, 0, 0, 0, 0), x_pitch=0.004, input_profile=None, plots=True,
       start_phase=None, phase_memory=False):

    # Keep track of elapsed time
    start_time = time.time()

    # Initialize input profile
    if input_profile is None:
        input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
    input_profile = cp.array(input_profile)
    input_profile /= cp.sqrt(cp.sum(cp.abs(input_profile)**2))

    # Target array amplitudes and phases
    amps = cp.array(amps)
    amps0 = cp.copy(amps)
    amps = cp.array(amps_guess)

    phases = 2 * pi * cp.array(phases)
    phases0 = cp.copy(phases)

    # Performance trackers
    eff = []
    nonunif = []
    phase_err = []

    # Record each algorithm run
    wus = []
    results = []
    amps_record = [amps]

    # Run the Wu algorithm M times
    for i in range(M):
        print('Iteration: ' + str(i))

        res = test_amps(n=n, N=N, size=size, beams0=amps0 * cp.exp(1j * phases0), beams=amps * cp.exp(1j * phases),
                        x_pitch=x_pitch, input_profile=input_profile, start_phase=start_phase,
                        phase_memory=phase_memory)
        # result = res[1]
        results.append(res[1] / np.max(np.abs(res[1])))

        # Update the target amplitudes to compensate the nonuniformity of the previous iteration
        print('Calculated amplitudes: ' + str(np.abs(results[-1]) / np.max(np.abs(results[-1]))))
        err = cp.abs(results[-1]) - amps0
        # print('Amplitude errors: ' + str(err))
        rel_result = amps0 / results[-1]
        # rel_result = 1 + amps0 - results[-1]
        # for i in range(len(amps)):
        #     if amps0[i] > 0.25:
        #         amps[i] *= cp.abs(rel_result[i])**0.25
        #     else:
        #         amps[i] = 0.54
        # if i > 0:
        #     grad = cp.abs(results[-1] - results[-2]) / (amps_record[-1] - amps_record[-2])
        print("amps0",amps0)
        for j in range(len(amps)):
            if amps0[j] >= 0.25:
                amps[j] *= cp.abs(rel_result[j])**0.25
        amps /= cp.max(amps)
        for j in range(len(amps)):
            if i > 0 and amps0[j] < 0.25 and amps_record[-1][j] != amps_record[-2][j]:
                # rel = 1 + amps0[j] - cp.abs(results[-1][j])
                # amps[j] *= rel**0.25
                grad = cp.abs(results[-1][j]) - cp.abs(results[-2][j])
                # if amps_record[-1] != amps_record[-2]:
                grad /= amps_record[-1][j] - amps_record[-2][j]
                amps[j] -= 0.1 * grad * err[j]
        # amps /= cp.max(amps)
        # amps[2] = 0.54
        print('Updated target amplitudes: ' + str(amps))
        amps_record.append(cp.copy(amps))

        start_phase = cp.angle(res[0].slm_field)

        wus.append(res[0])

        eff.append(res[2])
        nonunif.append(res[3])
        phase_err.append(res[4])
        print('Diffraction Efficiency: ' + str(eff[-1]))
        print('Amplitude error: ' + str(nonunif[-1]))
        print('Phase error: ' + str(phase_err[-1]))
        print()

        if plot_each:
            plt.plot([n for n in range(N)], wu.eff, label='Efficiency')
            plt.plot([n for n in range(N)], wu.nonunif, label='Amplitude Nonuniformity')
            plt.plot([n for n in range(N)], wu.phase_err, label='Phase error')
            plt.xlabel('Inner Iteration')
            plt.xlim(0, N)
            plt.grid(True)
            plt.legend()

            plt.show()

    # Find the minimum nonuniformity run of the Wu algorithm
    min_it = nonunif.index(min(nonunif))

    print('Time to run: ' + str(time.time() - start_time))

    print('')
    print('----Minimum iteration----')
    print('Diffraction Efficiency: ' + str(eff[min_it]))
    print('Amplitude error: ' + str(nonunif[min_it]))
    print('Phase error: ' + str(phase_err[min_it]))
    print('\nTarget phases: 2pi * ' + str(phases0 / 2 / pi))
    print('Actual phases: 2pi * ' + str(((wus[min_it].phases() + 2 * pi) % (2 * pi)) / 2 / pi))
    print('\nTarget amplitudes: ' + str(amps0))
    print('Actual amplitudes: ' + str(wus[min_it].amps() / cp.max(wus[min_it].amps())))

    # Save algorithm results
    if plots:
        wus[min_it].save_pattern(name=name, slm=slm, target=False, correction=True, show=(), wavelength=wavelength,
                                 field=True)

        plot_gradient(wus[min_it].image_field)

        # Plot the efficiency, nonuniformity and phase error for the best iteration
        plt.figure()
        plt.clf()
        plt.plot([n for n in range(len(wus[min_it].eff))], cp.array(wus[min_it].eff).get(), label='Efficiency')
        plt.plot([n for n in range(len(wus[min_it].nonunif))], cp.array(wus[min_it].nonunif).get(),
                 label='Amplitude error')
        plt.plot([n for n in range(len(wus[min_it].phase_err))], cp.array(wus[min_it].phase_err).get(),
                 label='Phase error')
        plt.xlabel('Inner Iteration')
        plt.title('Inner Loop Convergence')
        plt.xlim(0, N)
        plt.grid(True)
        plt.legend()
        plt.pause(.001)
        plt.savefig('images/inner_convergence.png')

        # Plot the final efficiency, nonuniformity and phase error across all iterations
        plt.figure()
        plt.clf()
        plt.plot([m for m in range(M)], cp.array(eff).get(), label='Efficiency')
        plt.plot([m for m in range(M)], cp.array(nonunif).get(), label='Amplitude error')
        plt.plot([m for m in range(M)], cp.array(phase_err).get(), label='Phase error')
        plt.xlabel('Outer Iteration')
        plt.title('Outer Loop Convergence')
        plt.xlim(0, M)
        plt.grid(True)
        plt.legend()
        plt.savefig('images/outer_convergence.png')

        if show:
            plt.show()

    return wus[min_it].slm_field


def wu_temnm(slm, n=3, m=4, N=100, slm_waist=np.array([0.05, 0.05]), plot_each=False, size=(1024, 1272)):
    ## Initialize input beam
    input_field = cp.array(Profile.input_gaussian(beam_size=slm_waist))
    # image_waist = 1 / (np.pi * slm_waist)

    ## Find natural waist at image plane to determine target waist
    laserbeam = laserbeamsizefromimage(slm, cp.abs(propagate(input_field)).get() ** 2)
    image_waist = np.array([laserbeam[2], laserbeam[3]])
    # print(image_waist)

    ## Generate target field
    target_slm_field = temnm(slm, n=n, m=m, w=slm_waist)
    target_slm_field /= cp.max(target_slm_field)
    target_image_field = temnm(slm, n=n, m=m, w=image_waist)
    target_image_field /= cp.max(target_image_field)

    wu = Wu(input=input_field, target=[target_image_field, [[int(0.5 * size[1]), int(0.5 * size[0])]]], array=False)

    wu.iterate(N)
    print()

    print('Diffraction Efficiency: ' + str(wu.eta()))
    print('Amplitude nonuniformity: ' + str(wu.dev_amp(waist=0.001)))
    print('Phase error: ' + str(wu.phase_error()))
    print()

    wu.save_pattern(name='wu_tem01', slm=slm, target=True, correction=False, show=(), field=True)

    plt.figure()
    plt.clf()
    plt.plot([n for n in range(N)], cp.array(wu.eff).get(), label='Efficiency')
    plt.plot([n for n in range(N)], cp.array(wu.nonunif).get(), label='Amplitude Nonuniformity')
    plt.plot([n for n in range(N)], cp.array(wu.phase_err).get(), label='Phase error')
    plt.xlabel('Inner Iteration')
    plt.xlim(0, N)
    plt.grid(True)
    plt.legend()
    plt.pause(.001)
    plt.savefig('images/inner_convergence.png')
    plt.show()


if __name__ == '__main__':

    size = (1024,1272)
    slm = SLM(size=size, wavelength=411)

    # array1D(slm)
    # pass

    # tem01(slm, size=(0.02, 0.02))
    input_size = (0.05, 0.05)

    # target, spots = Profile.gaussian_array(1, 5, amps=cp.exp(cp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * cp.pi * 1j), y_pitch=0.1, waist=(0.05, 0.05))
    #
    # # array4x4 = array2D(slm, 4, 4, x_pitch=0.08, y_pitch=0.08, size=(0.05, 0.05))
    # start_phase = cp.random.random_sample(slm.size)
    # start_phase = inverse_phase(slm, color=True, target=target, spots=spots, input_size=input_size)
    # # start_phase = start_phase + cp.random.random_sample(slm.size) * 0.5
    # # start_phase = (start_phase + slm.half())
    # slm.phaseToBMP(start_phase, '1x5_start_phase', color=True)
    # array1x5 = array1D(slm, it=20, tries=1, n=5, pitch=0.1, size=input_size, consider_phase=False, plots=(2, 3), start=start_phase, add_noise=False, waist=0.05)
    #
    # phases = -1 * cp.angle(array1x5.beams())/2
    # updated_amps = cp.exp(1j * phases)
    # updated_target, spots = Profile.gaussian_array(1, 5, amps=updated_amps)
    #
    # start_phase = inverse_phase(slm, color=True, target=updated_target, spots=spots, input_size=input_size)
    # array1x5 = array1D(slm, it=20, tries=1, n=5, pitch=0.1, size=input_size, consider_phase=False, plots=(2, 3),
    #                    start=start_phase, add_noise=False, waist=0.05)




    # start_phase = slm.BMPToPhase('images/blaze_grating.bmp')
    # ifta = IFTA(input=Profile.input_gaussian(beam_size=input_size))
    # ifta.phi = start_phase
    # ifta.slm_field = cp.abs(ifta.slm_field) * cp.exp(1j * ifta.phi)
    # ifta.propagate()
    # ifta.save_pattern('blaze_grat',slm)




    # array1x5_2 = array1D(slm, it=30, n=5, pitch=0.1, size=(0.05, 0.05), start=array1x5.phi, consider_phase=True, plots=(2, 3))

    #
    # tem01_2D(slm, 1, 5, array1x5)
    # tem01_2D(slm, 4, 4, array4x4)
    # interfere_ref(slm)


    # size = (0.05, 0.05)
    # reference = Profile(Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)) +
    #                     Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)))
    # interference = Profile(field=reference.field + array1x5.image_field)
    #
    # interfere_wgs(slm)

    # array1x5_oo = array1D_OutputOutput(slm, it=100, n=5, pitch=0.1, size=(0.05, 0.05), waist=0.05, plots=(2, 3))

    # wu(slm, N=40, M=20, size=size, wavelength=411)
    # wu_temnm(slm)

    # ifta = IFTA(input=cp.array(Profile.input_gaussian(beam_size=(0.05, 0.05))))
    # ifta.slm_field = cp.abs(ifta.slm_field)
    # ifta.propagate(ifta.slm_field)
    # ifta.backpropagate(ifta.image_field)
    # ifta.save_pattern('test', slm)
    # interfere_ref(slm)
    # plt.show()


    # input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
    # target_profile = Profile.target_output_array(n=1, m=5, input_profile=input_profile.get(), tem=True, x_pitch=0.02)
    # target_profile = [cp.array(target_profile[0]), target_profile[1]]
    # size = np.array(np.array((1024,1272)), dtype=np.uint)
    # outer = OuterLoop(slm=slm, input_profile=input_profile, n=4, wavelength=411, name='bs', amps=(1., 1., 1., 1.),
    #                         amps_guess=(1., 1., 1., 1.), phases=(0, 0, 0, 0), x_pitch=0.02, phase_memory=True, size=size, tem01=True)
    # phase = outer.iterate(N=40, M=30)
    # outer.plot(show=True)

    # slm = SLM()
    # input_size = np.array((0.2, 0.2))
    # input_profile = Profile.input_gaussian(beam_size=input_size, pos=np.array((0, 0)), size=slm.size, amp=1)
    #
    # target_array = Profile.target_output_array(1, 1, input_profile=input_profile, tem=True, x_pitch=0.1,
    #                                            double_amps=True,
    #                                            amps=(1, 1), phases=(0, 0))
    # print(target_array[1])

    slm_square = SLM(size=(1024,1272))

    prof = temnm(slm_square, n=0, m=0, w=(0.5, 0.5))

    # slm.fieldtoBMP(input_profile, 'input_profile', wavelength=411, color=True, correction=False)
    slm_square.fieldtoBMP(prof.get(), 'TEM$_{01}$ Beam Profile', wavelength=411, color=True, correction=False, show=False, sat=True, norm=True)

    plot_gradient(prof, coord=True, norm_coords=True, imag=False, norm=True, grid=True)
    plt.show()

    array1D(slm)
