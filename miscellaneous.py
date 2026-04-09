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

def testing_phase():
    scalepad=2
    curvature_list=np.linspace(0*0.75,4,1)
    for curvature_num in curvature_list:
        print("curvature_num: ",curvature_num)
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

        #To simulate bowman paper grid's phase pattern:
        for i in range(int(1024 * scalepad)):
            for j in range(int(1272 * scalepad)):
                target_phase[i][j] = np.arctan2((j - ((1272-b_off)*scalepad/2.0)+0.5), (i - ((1024-d_off)*scalepad/2.0)+0.5))  # MDS Enter center of the beam here

    plt.imshow(target_phase)
    plt.show()

if __name__ == '__main__':
    #datfilecreation()
    #curvatureview()
    #ffttest()
    #trialplot()
    #fftpropagatetest()
    testing_phase()
