import matplotlib.pyplot as plt
import numpy as np
import profile_v1 as profile
from PIL import Image
import slm_v1 as slm

def curvatureview():
    file_path = 'curvature_plots/curvature_tem01_2_1_u.npy'
    curvature = np.load(file_path)

    file_path = 'curvature_plots/final_eff_curvature_tem01_2_1_u.npy'
    efficiency = np.load(file_path)

    file_path = 'curvature_plots/final_ion_fidelity_curvature_tem01_2_1_u.npy'
    fidelity = np.load(file_path)
    print(curvature)
    plt.plot(curvature,efficiency)
    plt.plot(curvature,fidelity)
    plt.show()


def datfilecreation():
    input_file = "zemax_files/phaseTest.txt"
    output_file = "zemax_files/SLM_phaseTest.dat"

    with open(input_file, "r") as f:
        content = f.read().split()
    num_count=0
    with open(output_file, "w") as f:
        f.write("1024 1024 0.0125 0.0125 0 0.0 0.0\n")
        for number in content:
            num_count+=1
            #f.write(f"{np.mod(np.float32(number)+2*np.pi,2*np.pi)-np.pi} 0.0 0.0 0.0 0\n")
            f.write(f"{(np.mod(np.float32(num_count*0.0001)+2*np.pi,2*np.pi)-np.pi)*(2*np.pi)} 0.0 0.0 0.0 0\n")

    print("Done.")
    print(num_count)
    data = np.loadtxt("zemax_files/phaseTest.txt")

    print("Shape:", data.shape)

    plt.figure()
    plt.imshow(data, cmap='jet')
    plt.colorbar(label='Value')
    plt.title("2D Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def ffttest():
    for ii in (6,):
        pixeltot=2048*10
        imagecoord=np.linspace(-1,1,pixeltot)
        slmplanecoord=np.linspace(-1*(pixeltot),1*(pixeltot),pixeltot)
        imageplane=np.exp(1j*(imagecoord**2)*ii*10)
        slmplane=np.fft.fftshift(np.fft.fft(np.fft.fftshift(imageplane)))
        slmplane=slmplane/np.max(slmplane)
        plt.figure()
        plt.title((ii,"slmplane"))
        plt.plot(slmplanecoord,np.angle(slmplane))
        plt.plot(slmplanecoord,np.abs(slmplane))
        plt.figure()
        plt.title((ii,"imageplane"))
        plt.plot(imagecoord,np.angle(imageplane))
        plt.plot(imagecoord,np.abs(imageplane))
        plt.show()

def trialplot():
    # x1=(0,2,4,6,8,10)
    # x2=(1,3,5,7,9)
    # y1=(12,53,96,84,35,5)
    # y2=(17,76,116,76,17)

    x1=np.array((1000,800,600,400))-300.0
    y1=(122.9,171,285,818)
    plt.plot(1/x1,y1)
    #plt.plot(x2,y2)
    plt.show()


def fftpropagatetest():
    scalepad=2
    intensity_in=profile.Profile.input_gaussian(beam_size=(0.55 * (1 / scalepad), 0.55 * (1 / scalepad)), pos=(0.062,0.00),
                                   size=(int(1024 * scalepad), int(1272 * scalepad)))
    plt.imshow(np.abs(intensity_in))

    phase_path_p = r'C:\Python programs\slm_comp_pytorch\images\wu1x4_phase.bmp'
    phase_slm=np.asarray(Image.open(phase_path_p))
    phase_in=slm.pad_border(phase_slm, (1024*scalepad, 1272*scalepad))
    plt.imshow(np.abs(phase_in),alpha=0.1)
    plt.show()
    plt.figure()
    field_out=np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(intensity_in * np.exp(np.multiply(1j, phase_in*2*np.pi/255))), norm="ortho"))
    plt.imshow(np.abs(field_out))
    plt.show()


if __name__ == '__main__':
    #datfilecreation()
    #curvatureview()
    #ffttest()
    #trialplot()
    fftpropagatetest()
