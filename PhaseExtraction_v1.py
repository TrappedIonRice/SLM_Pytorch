import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import skimage.color
import scipy
import cv2
from slm_v1 import SLM
import tkinter.filedialog

pi = np.pi


def semicircle(x, x0, y0, r):
    return -1 * np.sqrt(r**2 - (x - x0)**2) + y0


# Import intensity, reference and interference profiles
def import_profs(slm=SLM(), paths=None, name=''):
    if paths is None:
        intensity = slm.BMPToAmp(path=tkinter.filedialog.askopenfilename(title=name + 'Intensity'), norm=False)
        ref = slm.BMPToAmp(path=tkinter.filedialog.askopenfilename(title=name + 'Reference'), norm=False)
        x_proj = slm.BMPToAmp(path=tkinter.filedialog.askopenfilename(title=name + 'Interference'), norm=False)
        y_proj = slm.BMPToAmp(path=tkinter.filedialog.askopenfilename(title=name + 'Interference + pi/2'), norm=False)
    else:
        intensity = slm.BMPToAmp(path=paths[0], norm=False)
        ref = slm.BMPToAmp(path=paths[1], norm=False)
        x_proj = slm.BMPToAmp(path=paths[2], norm=False)
        y_proj = slm.BMPToAmp(path=paths[3], norm=False)

    return [intensity, ref, x_proj, y_proj]


# Extract a 2d phase map from input profiles
def phasemap_2d(profs):
    try:
        sigma = 1
        intensity = cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
        ref = cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
        x_proj = cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
        y_proj = cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # intensity = profs[0]#cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # ref = profs[1]#cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # x_proj = profs[2]#cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # y_proj = profs[3]#cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)
    
        cosphi_x = (x_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
        cosphi_y = (y_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
        cosphi_x = np.clip(cosphi_x, -1, 1)
        cosphi_y = np.clip(cosphi_y, -1, 1)
        phi_x = (np.arccos(cosphi_x) + 2 * np.pi) % (2 * np.pi)
        phi_y = (np.arccos(cosphi_y) + 2 * np.pi) % (2 * np.pi)
    
        phasex = np.array([phi_x, 2 * np.pi - phi_x])   #because we can't distinguish 'angle' and '2 * np.pi - angle' from cos function only
        phasey = np.array([phi_y, 2 * np.pi - phi_y + np.pi / 2])
    
        phasey = ((phasey - np.pi / 2) + 2 * np.pi) % (2 * np.pi)     #phasey was pi/2 shifted from phasex

        ## min = [[0, 0], 2 * np.pi]
        ## min = np.ones(phasex.shape) * 2 * np.pi
        min = np.ones(phasex[0].shape) * 2 * np.pi    # phasex[0].shape= (3000, 4000)
        min_loc = np.zeros((phasex[0].shape[0], phasex[0].shape[1], 2))  #defining the shape of min_loc
        # print('len',min_loc.shape)
    
        #rayan
    
        phase = np.zeros((phasex[0].shape[0], phasex[0].shape[1]))
        value1 = np.zeros(phase.shape)
        value4 = np.zeros(phase.shape)

        for i in range(phasex[0].shape[0]):
            for j in range(phasex[0].shape[1]):
                value1[i][j] = np.abs(phasex[0][i][j] - phasey[0][i][j])
                # value2[i][j] = phasex[0][i][j] - phasey[1][i][j]
                # value3[i][j] = phasex[1][i][j] - phasey[0][i][j]
                value4[i][j] = np.abs(phasex[1][i][j] - phasey[1][i][j])
                values = [value1[i][j], value4[i][j]]
                min= np.min(values)


                # if  values[0] == min:
                #     phase[i][j] =(phasex[0][i][j] + phasey[0][i][j]) /2
                # if values[1] == min:
                #     phase[i][j] = (phasex[1][i][j]+ phasey[1][i][j])  /2


                if  values[0] == min:
                    phase[i][j] = (phasex[0][i][j])
                elif values[1] == min:
                    phase[i][j] = (phasex[1][i][j])



        #method2
        # for i in range(len(phasex)):
        #     for j in range(len(phasey)):
        #         val = np.abs(phasex[i] - phasey[j])
        #
        #         min_loc[val < min] = np.array([i, j])
        #
        #         # min_loc = np.where(val < min, np.array([i, j]), min_loc)
        #
        #         min = np.where(val < min, val, min)
        #         # min[val < min] = val    #cannot use this here since min is a 2-dimensional input array
        # # print('k',min_loc)
        # # print(np.average(min))
        # phase = np.zeros(min.shape)
        # for i in range(len(min)):
        #     for j in range(len(min[i])):
        #         #print(min_loc[i, j, 0])
        #         # phase[i, j] = (phasex[int(min_loc[i, j, 0]), i, j] + phasey[int(min_loc[i, j, 0]), i, j]) / 2
        #         # phase[i, j] = phasex[int(min_loc[i, j, 0]), i, j]
        #
        #         phase[i, j] = phasey[int(min_loc[i, j, 1]), i, j]
        #         #print('test', phasey[int(min_loc[i, j, 1]), i, j], phasey.shape)


    except Exception as e:
        print("phase extract error",e)
    return phase

def phasemap_2di(profs):
    if 1:
        sigma = 1
        intensity = cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
        ref = cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
        x_proj = cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
        y_proj = cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # intensity = profs[0]  # cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # ref = profs[1]  # cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # x_proj = profs[2]  # cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # y_proj = profs[3]  # cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)
        m = 1
        index = np.argmax(intensity)
        cooridnates = np.unravel_index(index, intensity.shape)
        x=[]
        intensity_plot=[]
        ref_plot=[]
        x_proj_plot=[]
        y_proj_plot=[]
        for i in range(intensity.shape[1]):
            x.append(i)
            ref_plot.append(ref[cooridnates[0]][i])
            intensity_plot.append(intensity[cooridnates[0]][i])
            x_proj_plot.append(x_proj[cooridnates[0]][i])
            y_proj_plot.append(y_proj[cooridnates[0]][i])
        plt.plot(x,intensity_plot,label='intensity',color='blue')
        plt.plot(x, ref_plot, label='ref', color='red')
        plt.plot(x, x_proj_plot, label='x_proj', color='green')
        plt.plot(x, y_proj_plot, label='y_proj', color='brown')
        plt.title('1 dimension plot')
        plt.xlabel('x axis')
        plt.ylabel('intensity')
        plt.legend()
        plt.grid()
        plt.show()
            
        cosphi_x = (x_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
        sinphi = (y_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
        plt.figure(figsize=(8, 6))
        plt.imshow(cosphi_x, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.figure(figsize=(8, 6))
        plt.imshow(sinphi, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        # plt.figure(figsize=(8, 6))
        # plt.imshow(np.abs(cosphi_x), cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        cosphi_x = np.clip(cosphi_x, -1, 1)
        sinphi = np.clip(sinphi, -1, 1)
        phase = np.zeros((cosphi_x.shape[0], cosphi_x.shape[1]))
        for i in range(cosphi_x.shape[0]):
            for j in range(cosphi_x.shape[1]):
                if sinphi[i][j]<0:
                    if cosphi_x[i][j] ==1 or cosphi_x[i][j] ==-1:
                        if cosphi_x[i][j]<0:
                            phase[i][j] = np.pi+ np.arcsin(-sinphi[i][j])
                            # phase[i][j] = ((5*np.pi/2-np.arccos(sinphi[i][j])) )

                        if cosphi_x[i][j]>0:
                            # phase[i][j] =( np.arccos(sinphi[i][j])-pi/2)
                            phase[i][j] = np.arcsin(sinphi[i][j])+2*np.pi

                    else:
                        phase[i][j]= (2 * np.pi - np.arccos(cosphi_x[i][j])) #+ 2 * np.pi)) % (2 * np.pi)
                if sinphi[i][j]>=0:
                    if cosphi_x[i][j] ==1 or cosphi_x[i][j] ==-1:
                        if cosphi_x[i][j] < 0:
                            # phase[i][j] = ((3* np.pi / 2 - np.arccos(sinphi[i][j])) ) #% (2 * np.pi)
                            phase[i][j] = ((np.pi-np.arcsin(sinphi[i][j])))  # % (2 * np.pi)
                        if cosphi_x[i][j] > 0:
                            # phase[i][j] = (2*np.pi - (np.pi/2-np.arccos(sinphi[i][j])))
                            phase[i][j] = np.arcsin(sinphi[i][j])
                    else:
                        phase[i][j] = np.arccos(cosphi_x[i][j])
      
        # plt.figure(figsize=(8, 6))
        # plt.imshow(phase, cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
    return  phase
def phasemap_2d2(profs):
    if 1:
        # sigma = 1
        # intensity = cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # ref = cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # x_proj = cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
        # y_proj = cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)
        intensity = profs[0]  # cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
        ref = profs[1]  # cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
        x_proj = profs[2]  # cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
        y_proj = profs[3]  # cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)

        cosphi_x = (x_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
        cosphi_y = (y_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
        # plt.figure(figsize=(8, 6))
        # plt.imshow(np.abs(cosphi_x), cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        cosphi_x = np.clip(cosphi_x, -1, 1)
        cosphi_y = np.clip(cosphi_y, -1, 1)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(np.abs(cosphi_x), cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        # phasex = np.array([phi_x,
        #                    2 * np.pi - phi_x])  # because we can't distinguish 'angle' and '2 * np.pi - angle' from cos function only
        # phasey = np.array([phi_y - np.pi / 2, 2 * np.pi - phi_y + np.pi / 2])

        # phasey = ((phasey - np.pi / 2) + 2 * np.pi) % (2 * np.pi)     #phasey was pi/2 shifted from phasex
        #print('shape', phasex[0].shape[0])
        ## min = [[0, 0], 2 * np.pi]
        ## min = np.ones(phasex.shape) * 2 * np.pi
        # min = np.ones(phasex[0].shape) * 2 * np.pi  # phasex[0].shape= (3000, 4000)
        # min_loc = np.zeros((phasex[0].shape[0], phasex[0].shape[1], 2))  # defining the shape of min_loc
        # print('len',min_loc.shape)



        phase = np.zeros((cosphi_x.shape[0], cosphi_x.shape[1]))
        # value1 = np.zeros(phase.shape)
        # value4 = np.zeros(phase.shape)

        # for i in range(len(phasex)):
        #     for j in range(len(phasey)):
        #         val = np.abs(phasex[i] - phasey[j])
        #         print(min_loc.shape)
        #         min_loc[val < min] = np.array([i, j])
        #
        #         # min_loc = np.where(val < min, np.array([i, j]), min_loc)
        #
        #         print('b', min_loc)
        #         min = np.where(val < min, val, min)
        #         # min[val < min] = val    #cannot use this here since min is a 2-dimensional input array

        # rayan
        for i in range(cosphi_x.shape[0]):
            for j in range(cosphi_x.shape[1]):
                if cosphi_y[i][j]<0:
                    phase[i][j]= ((2 * np.pi - np.arccos(cosphi_x[i][j])) + 2 * np.pi) % (2 * np.pi)
                if cosphi_y[i][j]>0:
                    phase[i][j] = (( np.arccos(cosphi_x[i][j])) + 2 * np.pi) % (2 * np.pi)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(np.abs(phase), cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.show()


    # except Exception as e:
    #     print("phase extract error", e)
    return phase


def fit_semicircle(data):
    fit = scipy.optimize.curve_fit(semicircle, data[0], data[1], p0=[2000, 2000, 3000])
    popt, pcov = fit[0], fit[1]
    fitdata = np.array([data[0], semicircle(data[0], *popt)])

    return fitdata, popt


def subtract_semicircle(x_dat, phase_dat, intensity_dat):
    pd = np.copy(phase_dat)
    xd = np.copy(x_dat)
    x_dat = x_dat[3000:3900]
    phase_dat = phase_dat[3000:3900]
    intensity_dat = intensity_dat[3000:3900]
    phase_dat = np.where(phase_dat < 0.4 * 2 * np.pi, phase_dat + 2 * np.pi, phase_dat)
    data = [[], []]
    # dat = []
    for i in range(len(phase_dat)):
        if intensity_dat[i] > 0.05 * np.max(intensity_dat):
            data[0].append(x_dat[i])
            data[1].append(phase_dat[i])
            # dat.append(pd[i])
    data = np.array(data)
    fitdata, popt = fit_semicircle(data)

    fitdata = np.array([data[0], semicircle(data[0], *popt)])

    plt.figure()
    plt.scatter(xd, pd / 2 / np.pi, label='phase data', marker='.')
    plt.plot(fitdata[0], fitdata[1] / 2 / np.pi, label='fit data', c='r')
    plt.legend()
    plt.plot()

# def fit_int(slm, phase, profs):


# Plot a 2d phase map
def plot_phasemap(slm, phase, profs, figs=(None, None, None), colorbars=(True, True), threshold=0.3, name='1x4 Beam Array', maps=True):
    sigma = 1.5
    profs[0] = cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
    profs[1] = cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
    profs[2] = cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
    profs[3] = cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)
    pk = np.where(profs[0] == np.max(profs[0]))
    print(pk)
    # pk = [825]
    # pk = [[1526]]
    field = np.sqrt(profs[0]) * np.exp(1j * phase)
    # print(np.max(np.abs(field)))
    ref_dat = profs[1][pk[0], :][0]
    intensity_dat = profs[0][pk[0], :][0]
    x_proj_dat = profs[2][pk[0], :][0]
    y_proj_dat = profs[3][pk[0], :][0]
    phase_dat = phase[pk[0], :][0]
    phase_dat = np.where(intensity_dat > threshold * np.max(profs[0]), phase_dat, 0)

    # phase_dat1 = list(phase_dat)
    #
    # for i in range(len(phase_dat)):
    #     if intensity_dat[i] < threshold * np.max(intensity_dat):
    #
    #             # np.delete(phase_dat, i)
    #         phase_dat1.remove(phase_dat[i])




    # phase_dat[:1667] = 0
    # phase_dat[1779:2121] = 0
    # phase_dat[]
    # phase_dat = np.where(phase_dat < 0.4, phase_dat + 2 * np.pi, phase_dat)
    # print(intensity_dat)
    if figs[2] is not None:
        ax = figs[2].axes
        twinax = ax.twinx()
    else:
        ax = plt.figure().add_subplot(111)
        twinax = plt.twinx()
    pixel_size = 1.85
    ax.plot([i * pixel_size for i in range(len(ref_dat))], ref_dat, label='Reference')
    twinax.plot([i * pixel_size for i in range(len(phase_dat))], phase_dat / 2 / pi, label='Phase', c='r')
    ax.plot([i * pixel_size for i in range(len(intensity_dat))], intensity_dat, label='Intensities')

    ax.plot([i * pixel_size for i in range(len(x_proj_dat))], x_proj_dat, label='Interference x')
    # ax.plot([i * pixel_size for i in range(len(y_proj_dat))], y_proj_dat, label='Interference y')

    ax.plot([i * pixel_size for i in range(len(x_proj_dat))], x_proj_dat, label='Interference 1')
    # ax.plot([i * pixel_size for i in range(len(y_proj_dat))], y_proj_dat, label='Interference 2')

    ax.set_xlabel('$x$ ($\mu$m)')
    ax.set_ylabel('Intensity')
    ax.set_title(name)
    twinax.set_ylabel('Phase ($2\\pi$ radians)')
    twinax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    twinax.legend(loc='upper right')
    np.savetxt('phase_dat.txt', phase_dat)
    np.savetxt('intensity_dat.txt', intensity_dat)
    # subtract_semicircle([i * 1.85 for i in range(len(ref_dat))], phase_dat, intensity_dat)
    # plt.show()
    if maps:
        slm.fieldtoBMP(field, name, color=True, wavelength=411, show=False,
                       extent=(0, (field.shape[1] - 1) * 1.85, 0, (field.shape[0] - 1) * 1.85), units=' ($\mu$m)', figure=figs[0], colorbar=not colorbars[0])
        slm.phaseToBMP(phase, name, color=True, wavelength=411, show=False,
                       extent=(0, (phase.shape[1] - 1) * 1.85, 0, (phase.shape[0] - 1) * 1.85), units=' ($\mu$m)',
                       figure=figs[1], colorbar=not colorbars[1])
        
    # slm.phaseToBMP(phase, '1x4 Beam Array', color=True, wavelength=411, show=False)

    # plt.clf()
    # plt.plot([i * pixel_size for i in range(len(phase_dat1))], phase_dat1 / 2/ pi, label='Phase', c='r')
    # plt.xlabel('$x$ ($\mu$m)')
    # plt.ylabel('Phase ($\\pi$ radians)')
    # plt.ylim(0, 1)
    # ax.legend(loc='upper right')
    # np.savetxt('phase_dat.txt', phase_dat1)
    # plt.show()



    # plt.plot([i for i in range(len(intensity_dat))], intensity_dat / 50, label='Intensities')
    # plt.plot([i for i in range(len(phase_dat))], phase_dat, label='Phase')
    # plt.xlabel('X (px)')
    # plt.title('1x4  Phase Beam Array')
    # plt.legend()

    # slm.phaseToBMP(phase, 'Phase', wavelength=411, color=True, show=False)
    #
    # slm.ampToBMP(intensity, '1x4 Beam Array', color=True, show=False)
    # slm.ampToBMP(interference, '1x4 Beam Array Interference', color=True, show=True)
    # plt.show()
    
# def getsigma(slm, phase, profs):





if __name__ == '__main__':
    slm = SLM()


    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2.bmp",
    #          r"Z:\Lab Rice\Experimental Projects\SLM\camera images\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-2.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-0.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-1.Bmp"]

    # paths = [r"C:\Users\raywe\OneDrive\Desktop\slm\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2.bmp",
    #          r"C:\Users\raywe\OneDrive\Desktop\slm\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-2.Bmp",
    #          r"C:\Users\raywe\OneDrive\Desktop\slm\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-0.Bmp",
    #          r"C:\Users\raywe\OneDrive\Desktop\slm\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-1.Bmp"]

    paths = [r"C:\Users\raywe\OneDrive\Desktop\int tem beam without ref expander 3 plus nd 1 to ref.bmp",
             r"C:\Users\raywe\OneDrive\Desktop\int tem beam without ref expander 3 plus nd 1 to ref-08202024174321-1.Bmp",
             r"C:\Users\raywe\OneDrive\Desktop\int tem beam without ref expander 3 plus nd 1 to ref-08202024174321-2.Bmp",
             r"C:\Users\raywe\OneDrive\Desktop\int tem beam without ref expander 3 plus nd 1 to ref-08202024174320-0.Bmp"
             ]

    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 alternating phase array intensity corrected.bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 alternating phase array corrected-05302024183605-0.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 alternating phase array corrected-05302024183605-1.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 alternating phase array corrected-05302024183605-2.Bmp"]
    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array intensity.bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array-05302024140434-0.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array-05302024140434-1.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array-05302024140434-2.Bmp"]
    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array intensity corrected.bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array corrected-05302024143517-0.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array corrected-05302024143518-1.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 uniform phase array corrected-05302024143518-2.Bmp"]
    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 alternating phase array corrected 2 intensity.bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 alternating phase array corrected 2-05302024181153-0.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 alternating phase array corrected 2-05302024181153-1.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem00 alternating phase array corrected 2-05302024181154-2.Bmp"]
    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 single int.bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/single tem01-05302024221443-1.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/single tem01-05302024221444-2.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/single tem01-05302024221443-0.Bmp"]



    # paths = [r"R:\gp31\Lab Rice\Experimental Projects\SLM\camera images\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2.bmp",
    #          r"R:\gp31\Lab Rice\Experimental Projects\SLM\camera images\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-2.Bmp",
    #          r"R:\gp31\Lab Rice\Experimental Projects\SLM\camera images\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-0.Bmp",
    #          r"R:\gp31\Lab Rice\Experimental Projects\SLM\camera images\uniform array tem00 beam tem00 with ref expander 3 plus nd 1 to ref 3 try 2-08202024191427-1.Bmp"]


    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array intensity corrected.bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array corrected-05302024151507-1.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array corrected-05302024151507-2.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array corrected-05302024151507-0.Bmp"]


    # paths = [r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array intensity.bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array-05302024150003-1.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array-05302024150003-2.Bmp",
    #          r"Z:/Lab Rice/Experimental Projects/SLM/camera images/tem01 uniform phase array-05302024150003-0.Bmp"]

    paths = [r"Z:/Lab Rice/Experimental Projects/SLM/data/tem00 single intensity hama.bmp",
             r"Z:/Lab Rice/Experimental Projects/SLM/data/tem00 single intensity hama-07012024180011-1.Bmp",
             r"Z:/Lab Rice/Experimental Projects/SLM/data/tem00 single intensity hama-07012024180012-2.Bmp",
             r"Z:/Lab Rice/Experimental Projects/SLM/data/tem00 single intensity hama-07012024180011-0.Bmp"]

    # phase_dat = np.loadtxt('phase_dat.txt')
    # intensity_dat = np.loadtxt('intensity_dat.txt')
    # x_dat = np.array([i for i in range(len(phase_dat))])
    # subtract_semicircle(x_dat, phase_dat, intensity_dat)

    matplotlib.use('qtagg')

    # profs = np.load('profs.npy')
    # phase = np.loadtxt('phase.txt')

    profs = import_profs(slm=slm, paths=paths)
    np.save('profs', profs)

    phase = phasemap_2d(profs)
    np.savetxt('phase.txt', phase)

    plot_phasemap(slm, phase, profs, threshold=0.3, name='TEM00 uniform array', maps=True)
    plt.show()
