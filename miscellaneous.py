import matplotlib.pyplot as plt
import numpy as np
import profile_v1 as profile
from PIL import Image
import slm_v1 as slm
from scipy.stats import binned_statistic_2d

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


def pandastesting():
    import pandas as pd
    import os

    file_name = 'data_storage_14_04_26.csv'

    # 1. Load existing data or create a new file
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col=0)
    else:
        # Initialize with at least one column/row or empty
        df = pd.DataFrame()

    # 2. Choose row and column names for data insertion
    rownum=4
    row_name = str(rownum)
    col_name = 'Fidelity'
    data_value = 0.95

    # 3. Insert/Update data at the intersection
    df.loc[row_name, col_name] = data_value

    rownum=5
    row_name = str(rownum)
    col_name = 'Efficiency'
    data_value = 0.98

    # 3. Insert/Update data at the intersection
    df.loc[row_name, col_name] = data_value

    # 4. Save back to the file
    df.to_csv(file_name)

def plotfromdatastorage():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("data_storage.csv")



    plt.scatter(
        #np.log10(1 - (df["Fidelity_ion"])),
        df["Fidelity_ion"],
        df["Efficiency"],
    )
    plt.show()

    df_filtered1 = df[np.isclose(df["start_phase_curve"], 0.01)]

    fig, ax1 = plt.subplots()

    # First scale (Left)
    ax1.scatter(df_filtered1["Curvature"],
        df_filtered1["Fidelity_ion"], color='blue')
    ax1.set_ylabel('Fidelity', color='blue')

    # Second scale (Right)
    ax2 = ax1.twinx()
    ax2.scatter(df_filtered1["Curvature"],
        df_filtered1["Efficiency"], color='red')
    ax2.set_ylabel('Efficiency', color='red')

    plt.show()



    curvatures = np.linspace(0.0, 1.0, 11)
    colors = plt.cm.viridis(np.linspace(0, 1, len(curvatures)))  # colormap


    for curv_value, color in zip(curvatures, colors):
        tol = 1e-6

        df_filtered = df[np.isclose(df["Curvature"], curv_value, atol=tol)]
        df_filtered = df_filtered[df_filtered["Fidelity_ion"] > 0]

        plt.scatter(
            np.log10(1-(df_filtered["Fidelity_ion"])),
            df_filtered["Efficiency"],
            color=color,
            label=f"{curv_value:.2f}",
            alpha=0.7
        )

    #plt.xscale("log")

    plt.xlabel("Ion Fidelity (log10(1-F))")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Ion Fidelity for different Curvatures")

    plt.legend(title="Curvature", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def pickleplot():
    import pickle
    import cupy as cp
    with open('nonunif_temp.pkl', 'rb') as file:
        data = pickle.load(file)
    print(data)
    print(len(data))
    print(type(data))
    print(type(data[0]))
    print(data[0].shape)
    print(data[0])
    #plt.plot(data.get())
    plt.plot([np.asarray(x.get()) for x in data])
    plt.show()


import numpy as np


def add_zernike_phase(n, m, shape, amplitude, focus_factor=1.0):
    import math
    """
    Generates a Zernike polynomial phase screen and adds it to the mod object.

    Parameters:
    - mod: The object containing the add_phase method.
    - shape: Tuple (rows, cols) defining the grid dimensions.
    - n: Radial degree (non-negative integer).
    - m: Azimuthal frequency (integer, |m| <= n, and n - m must be even).
    - amplitude: Scaling factor for the phase.
    """
    # 1. Create normalized coordinate grid matching your aspect ratio logic
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1 * shape[0] / shape[1], 1 * shape[0] / shape[1], shape[0])
    xx, yy = np.meshgrid(x, y)

    # 2. Convert to polar coordinates
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    # Mask out coordinates outside the unit circle (Zernikes are defined on a unit disk)
    mask = r <= 1.0

    # 3. Compute radial polynomial R_n^m(r) using the standard formula
    radial_poly = np.zeros_like(r)

    if (n - abs(m)) % 2 == 0 and n >= abs(m):
        for k in range(0, (n - abs(m)) // 2 + 1):
            numerator = (-1) ** k * math.factorial(n - k)
            denominator = (
                    math.factorial(k) *
                    math.factorial((n + abs(m)) // 2 - k) *
                    math.factorial((n - abs(m)) // 2 - k)
            )
            radial_poly += (numerator / denominator) * (r ** (n - 2 * k))
    else:
        raise ValueError(
            f"Invalid Zernike indices indices (n={n}, m={m}). "
            f"n must be >= |m| and (n - |m|) must be even."
        )

    # 4. Compute angular component and combine
    if m >= 0:
        zernike_matrix = radial_poly * np.cos(m * theta)
    else:
        zernike_matrix = radial_poly * np.sin(abs(m) * theta)

    # Apply mask so it's only active within the unit circle (optional but standard)
    zernike_matrix = zernike_matrix * mask

    # 5. Scale by amplitude and add to your system
    zernike_phase = amplitude * zernike_matrix
    #mod.add_phase(zernike_phase)

    return zernike_phase


def fftlens():
    import numpy as np
    from PIL import Image
    import ArrayModulator_v1

    wavelength=411
    # Open the BMP image
    scalepad=2
    wu_1x4_full =np.load(runsettings.directory_SLM_computation + r'\images\wu_1x4_full.npy')
    wu_1x4 = wu_1x4_full[
        int(1024 * scalepad / 2) - 512:int(1024 * scalepad / 2) + 512, int(1272 * scalepad / 2) - 636:int(
            1272 * scalepad / 2) + 636]
    # plt.imshow(wu_1x4)
    # plt.show()
    shape=wu_1x4.shape
    mod = ArrayModulator_v1.ArrayModulator(beams=1 ,correction_path='images/%dcorrwithLUT.bmp' % wavelength)
    #original
    #phase_path = runsettings.directory_SLM_computation + r'\Filrcamera\pictures\correction_phase.txt'
    phase_path = runsettings.directory_SLM_computation + r'\Filrcamera\pictures\correction_phase_May_13_sys2usable.txt'
    correction = np.loadtxt(phase_path)
    grady = mod.gradient(mod.beams[0][1], angle=0.20* np.pi / 180, axis=0, wavelength=wavelength) #1315#yangle=0.12 #yangle=0.1310 for expt
    gradx = mod.gradient(mod.beams[0][1], angle=-0.27* np.pi / 180, axis=1, wavelength=wavelength) #-0.12 original



    mod.add_phase(grady)
    mod.add_phase(gradx)
    mod.add_phase(wu_1x4)


    if False: #Apply lens
        focus =2000
        x=np.linspace(-1,1,shape[1])
        y = np.linspace(-1*shape[0]/shape[1], 1*shape[0]/shape[1], shape[0]) #Having it as a square
        xx,yy=np.meshgrid(x,y)
        print(shape)
        lens_phase=-2*np.pi*(shape[1]/focus)*( xx**2+yy**2)

        # plt.imshow(np.mod(lens_phase,2*np.pi))
        # plt.show()
        mod.add_phase(lens_phase)

    if False:
        # zernike_polynomial= add_zernike_phase(1,1,grady.shape,1)
        # # plt.imshow(zernike_polynomial)
        # # plt.show()
        # mod.add_phase(zernike_polynomial)
        zernike_polynomial= add_zernike_phase(1,-1,grady.shape,-2)
        # plt.imshow(zernike_polynomial)
        # plt.show()
        mod.add_phase(zernike_polynomial)
        #
        # zernike_polynomial= add_zernike_phase(4,0,grady.shape,2)
        # # plt.imshow(zernike_polynomial)
        # # plt.show()
        # mod.add_phase(zernike_polynomial)
        #
        # zernike_polynomial= add_zernike_phase(3,1,grady.shape,2)
        # # plt.imshow(zernike_polynomial)
        # # plt.show()
        # mod.add_phase(zernike_polynomial)




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
                        matrix_correction[patch_size_abb * i + m][patch_size_abb * j + n] = correction[i][j] # 26/05/13 gt: multiply by zero for Hamamatsu corr (see also below at mod.phase)
    if True: #Original 06 Feb, choosing only the central part
        mod.phase = mod.phase + slm.pad_border(matrix_correction, (1024, 1272))  #UNCOMMENT FOR CORRECTION
        # phase_path = runsettings.directory_SLM_computation + r'\images\tem01.txt'
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

    mod.phaseToBMP(mod.phase, name='phase_mask0_0', color=False, correction=False, wavelength=411) # 26/05/13 gt: set correction to True for HM correction
    mod.phase = mod.phase + 120 / 180 * np.pi
    mod.phaseToBMP(mod.phase, name='phase_mask0_120', color=False, correction=False, wavelength=411) # same as above
    mod.phase = mod.phase + 120 / 180 * np.pi
    mod.phaseToBMP(mod.phase, name='phase_mask0_240', color=False, correction=False, wavelength=411) # same as above
    # if True:
    #     plt.imshow(mod.phase)
    #     plt.title("is it")
    #     plt.colorbar()
    #     plt.show()






    # intensity=np.array(Image.open(runsettings.directory_SLM_Pytorch + r'\images\wu1x4_intensity.bmp'))
    #
    # x=np.linspace()
    # X,Y = np.meshgrid(x,y)
    #
    # # Convert the image data directly to a NumPy array
    # fftout=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(intensity*np.exp(1j*phase*2*np.pi/255))))
    # plt.imshow(np.abs(fftout))
    # plt.show()


import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def plot_fidelity_vs_efficiency_old(jsonl_path='JSONoutput/data_storage_zero_curve_temp.jsonl'):
    # 1. Read the JSONL file line-by-line safely
    if not os.path.exists(jsonl_path):
        print(f"Error: The file {jsonl_path} does not exist.")
        return

    data_runs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data_runs.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(data_runs)

    # 2. Extract values from complex entries if they were converted to dictionaries
    def clean_numeric_list(val):
        if not isinstance(val, list):
            return []
        cleaned = []
        for x in val:
            if isinstance(x, dict) and 'real' in x:
                cleaned.append(x['real'])  # Plotting the real magnitude/component
            elif isinstance(x, (int, float)):
                cleaned.append(x)
        return cleaned

    # Apply cleanup to our list columns
    df['fidelity_ion_list'] = df['fidelity_ion_list'].apply(clean_numeric_list)
    df['efficiency_ion_list'] = df['efficiency_ion_list'].apply(clean_numeric_list)

    # Ensure curvature is clean for grouping
    df['Curvature'] = df['Curvature'].fillna('Unknown')

    # 3. Group by curvature to launch separate plot windows
    grouped = df.groupby('Curvature')

    fids_all_curve=[]
    effs_all_curve=[]

    for curvature_val, group in grouped:
        # Create a brand new plot window for this specific curvature
        fids_each_curve=[]
        effs_each_curve=[]
        plt.figure(figsize=(9, 6))

        # Track if we actually plotted any valid multi-element lines
        lines_plotted = 0
        loss_spots_each_curve=[]
        loss_local_std_each_curve=[]
        loss_spots_phase_each_curve=[]
        fids_last_each_curve=[]

        # Loop through every separate data entry (run) within this curvature group
        for idx, row in group.iterrows():
            fids = row['fidelity_ion_list']
            effs = row['efficiency_ion_list']
            # loss_spots = row['loss_spots']
            # loss_spots_phase = row['loss_spots_phase']
            # loss_local_phase_std = row['loss_local_phase_std']
            #
            # loss_spots_each_curve.append(loss_spots)
            # loss_local_std_each_curve.append(loss_local_phase_std)
            # loss_spots_phase_each_curve.append(loss_spots_phase)
            if len(fids) > 0:
                fids_last_each_curve.append(fids[-1])
            else:
                continue

            fids_each_curve.extend(fids)
            effs_each_curve.extend(effs)



            #Make sure we have data arrays that match in size and have elements to plot
            if len(fids) > 0 and len(fids) == len(effs):
                run_label = f"Run {idx} (ID: {row.get('slurm_id', 'N/A')})"

                # Plot the run as a scatter-line (both markers and connecting lines)
                plt.plot(effs, fids, marker='o', linestyle='-', alpha=0.7, label=run_label)
                lines_plotted += 1
        # plt.figure()
        # plt.scatter(fids_last_each_curve,loss_spots_each_curve,label="amp non-uniformity")
        # plt.scatter(fids_last_each_curve, loss_spots_phase_each_curve,label="phase non-uniformity")
        # plt.scatter(fids_last_each_curve, loss_local_std_each_curve, label="local phase non-uniformity within beam")
        # plt.title(f"curvature = {curvature_val}")
        # plt.legend()
        # plt.show()

        # Configure the plot window properties
        plt.title(f"Fidelity vs Efficiency — Curvature: {curvature_val}", fontsize=14, fontweight='bold')
        plt.xlabel("Ion Efficiency List", fontsize=12)
        plt.ylabel("Ion Fidelity List", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        #plt.yscale('log')

        # Only show a legend if there's a reasonable number of runs to avoid overcrowding
        if 0 < lines_plotted <= 15:
            plt.legend(loc='best', fontsize=9)
        elif lines_plotted > 15:
            plt.legend([f"{lines_plotted} total runs"], loc='upper right')

        plt.tight_layout()
        fids_all_curve.append(fids_each_curve)
        effs_all_curve.append(effs_each_curve)






    # 4. Display all generated plot windows simultaneously
    plt.figure()
    # num=0
    # for curvature_val, group in grouped:
    #     optimal_pairs, (boundary_fids, boundary_effs)=find_max_optimization_boundary(fids_all_curve[num], effs_all_curve[num], plot=True)
    #     print(optimal_pairs)
    #     num=num+1
    # plt.show()
    data = []
    num=0
    y_min, y_max = 0.998, 1.0  # Adjust these to crop your Y-axis

    X, Y, Z = [], [], []
    for curvature_val, group in grouped:
        _, (fids, effs),efficient_indices  = find_max_optimization_boundary(fids_all_curve[num], effs_all_curve[num], plot=False)
        X.extend(np.full_like(fids, curvature_val))
        Y.extend(fids)
        Z.extend(effs)
        num=num+1

    # Convert to arrays and apply the fidelity mask
    X, Y, Z = map(np.array, [X, Y, Z])
    mask = (Y >= y_min) & (Y <= y_max)

    # --- 2. Plot only the masked data ---
    # vmin and vmax automatically lock the colorbar limits to your visible data range
    scatter = plt.scatter(X[mask], Y[mask], c=Z[mask], cmap='viridis', marker='s', s=150,
                          vmin=Z[mask].min(), vmax=Z[mask].max())

    plt.colorbar(scatter, label='Max Efficiency')
    plt.ylim(y_min, y_max)
    plt.xlabel('Curvature')
    plt.ylabel('Fidelity')
    plt.show()

    scatter2 = plt.scatter(Y[mask], Z[mask], c=X[mask], cmap='viridis',#, marker='s', s=150,
                          vmin=X[mask].min(), vmax=X[mask].max())

    plt.colorbar(scatter2, label='Curvature')
    #plt.ylim(y_min, y_max)
    plt.xlabel('Fidelity')
    plt.ylabel('Efficiency')
    plt.show()
    mask = (Y >= 0.9994) & (Y <= 0.99942)
    plot3 = plt.scatter(X[mask], Z[mask])

    plt.colorbar(plot3, label='Efficiency=0.9994')
    #plt.ylim(y_min, y_max)
    plt.xlabel('Curvature')
    plt.ylabel('Efficiency')
    plt.show()


def find_max_optimization_boundary_old(fids, effs, plot=True):
    """
    Finds the pairs that maximize both fidelity and efficiency (Pareto Frontier)
    and plots the 2D maximum boundary.

    Parameters:
    -----------
    fids : list or numpy array
        List of fidelity values
    effs : list or numpy array
        List of efficiency values
    plot : bool
        If True, renders the 2D boundary scatter plot

    Returns:
    --------
    optimal_pairs : list of tuples
        Unsorted list of (fidelity, efficiency) pairs on the frontier
    boundary_coords : tuple of numpy arrays
        Sorted (fids, effs) arrays perfect for plotting a sequential line
    """
    # 1. Convert inputs to a unified 2D numpy array
    scores = np.vstack((fids, effs)).T
    n_points = scores.shape[0]

    # 2. Vectorized Pareto optimal selection
    # An item is Pareto optimal if no other item dominates it
    is_efficient = np.ones(n_points, dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            # Keep points if they are greater in at least one dimension, or equal to the current point
            is_efficient[is_efficient] = np.any(scores[is_efficient] > c, axis=1) | np.all(scores[is_efficient] == c,
                                                                                           axis=1)

    # 3. Extract optimal subsets
    optimal_fids = scores[is_efficient, 0]
    optimal_effs = scores[is_efficient, 1]
    optimal_pairs = list(zip(optimal_fids, optimal_effs))

    # 4. Sort coordinates by fidelity so the boundary line draws sequentially
    sort_idx = np.argsort(optimal_fids)
    boundary_fids = optimal_fids[sort_idx]
    boundary_effs = optimal_effs[sort_idx]

    # 5. Optional Plotting
    if plot:
        #plt.figure(figsize=(8, 5))
        # Plot sub-optimal data
        plt.scatter(fids, effs, color='lightgray',alpha=0.7, label='Sub-optimal Pairs')
        # Plot maximum frontier points
        plt.scatter(boundary_fids, boundary_effs, #color='darkorange',
                    s=50, zorder=3, label='Pareto Pairs')
        # Connect with the maximum boundary line
        plt.plot(boundary_fids, boundary_effs, #color='darkorange',
                 linestyle='-', linewidth=2, label='Maximum Boundary')

        plt.xlabel('Fidelity')
        plt.ylabel('Efficiency')
        plt.title('Fidelity vs Efficiency Optimization Frontier')
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        #plt.tight_layout()

    return optimal_pairs, (boundary_fids, boundary_effs)


def find_max_optimization_boundary(fids, effs, plot=True):
    """
    Finds the pairs that maximize both fidelity and efficiency (Pareto Frontier)
    and returns the indices alongside the coordinates.
    """
    scores = np.vstack((fids, effs)).T
    n_points = scores.shape[0]

    is_efficient = np.ones(n_points, dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] > c, axis=1) | np.all(scores[is_efficient] == c,
                                                                                           axis=1)

    optimal_fids = scores[is_efficient, 0]
    optimal_effs = scores[is_efficient, 1]
    optimal_pairs = list(zip(optimal_fids, optimal_effs))

    # Track the exact efficient indices
    efficient_indices = np.where(is_efficient)[0]

    sort_idx = np.argsort(optimal_fids)
    boundary_fids = optimal_fids[sort_idx]
    boundary_effs = optimal_effs[sort_idx]

    if plot:
        plt.scatter(fids, effs, color='lightgray', alpha=0.7, label='Sub-optimal Pairs')
        plt.scatter(boundary_fids, boundary_effs, s=50, zorder=3, label='Pareto Pairs')
        plt.plot(boundary_fids, boundary_effs, linestyle='-', linewidth=2, label='Maximum Boundary')
        plt.xlabel('Fidelity')
        plt.ylabel('Efficiency')
        plt.title('Fidelity vs Efficiency Optimization Frontier')
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)

    return optimal_pairs, (boundary_fids, boundary_effs), efficient_indices


def plot_boundary_heatmap(jsonl_path='JSONoutput/data_storage_11by11_curvature_change.jsonl',
                          target_variable='Efficiency', y_min=0.999, y_max=1.0, pixel_height=0.00005):
    if not os.path.exists(jsonl_path):
        print(f"Error: The file {jsonl_path} does not exist.")
        return None

    data_runs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data_runs.append(json.loads(line))
    df = pd.DataFrame(data_runs)

    def clean_numeric_list(val):
        if not isinstance(val, list):
            return []
        return [x['real'] if isinstance(x, dict) and 'real' in x else x for x in val if
                isinstance(x, (int, float, dict))]

    df['fidelity_ion_list'] = df['fidelity_ion_list'].apply(clean_numeric_list)
    df['efficiency_ion_list'] = df['efficiency_ion_list'].apply(clean_numeric_list)

    grouped = df.groupby('Curvature')
    X, Y, Z,High_eff = [], [], [],[]

    for curvature_val, group in grouped:
        fids_each_curve = []
        effs_each_curve = []
        run_mapping = []

        for idx, row in group.iterrows():
            fids = row['fidelity_ion_list']
            effs = row['efficiency_ion_list']
            fids_each_curve.extend(fids)
            effs_each_curve.extend(effs)
            run_mapping.extend([row] * len(fids))

        if not fids_each_curve:
            continue

        # Unpack the new efficient_indices rule here
        _, _, efficient_indices = find_max_optimization_boundary(fids_each_curve, effs_each_curve, plot=False)

        # Pull values out strictly by their validated array indices
        for exact_idx in efficient_indices:
            f = fids_each_curve[exact_idx]
            origin_row = run_mapping[exact_idx]

            if target_variable in origin_row:
                X.append(curvature_val)
                Y.append(f)
                Z.append(origin_row[target_variable])
                High_eff.append(effs_each_curve[exact_idx])

    if not X:
        print(f"Error: No data extracted for target variable '{target_variable}'. Check your JSON keys.")
        return None

    X, Y, Z,High_eff = map(np.array, [X, Y, Z,High_eff])
    mask = (Y >= y_min) & (Y <= y_max)
    X_m, Y_m, Z_m,High_eff_mask = X[mask], Y[mask], Z[mask],High_eff[mask]

    if len(Z_m) == 0:
        print("No data points found inside the visible Y range.")
        return None

    unique_curves = np.sort(np.unique(X_m))
    col_width = 0.05#np.min(np.diff(unique_curves)) if len(unique_curves) > 1 else 0.05

    # Set plot titles cleanly
    title_dict = {
        'Efficiency': 'Efficiency',
        'loss_spots': 'Amplitude non-uniformity',
        'loss_spots_phase': 'Phase non-uniformity',
        'loss_local_phase_std': 'Phase local non-uniformity'
    }
    plot_title = title_dict.get(target_variable, f"{target_variable.replace('_', ' ').title()}")

    # Toggle to True if you want to switch back to block layouts instead of quick scatters
    render_blocks = True
    if True:
        if target_variable == 'Efficiency':
            Z_m=High_eff_mask
        fig, ax = plt.subplots(figsize=(8, 6))
        if render_blocks:
            cmap = plt.get_cmap('plasma' if 'loss' in target_variable.lower() else 'viridis')
            norm = plt.Normalize(vmin=Z_m.min(), vmax=Z_m.max())
            ax.bar(x=X_m, height=pixel_height, width=col_width, bottom=Y_m - (pixel_height / 2), color=cmap(norm(Z_m)),
                   edgecolor='none', align='center')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=plot_title)
            ax.set_ylabel('Fidelity')
        else:
            scatter = ax.scatter(Y_m, Z_m, c=X_m, cmap='viridis')
            fig.colorbar(scatter, ax=ax, label='Curvature')
            ax.set_ylabel(plot_title)
            ax.set_xlabel('Fidelity')

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Curvature' if render_blocks else 'Fidelity')
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.3)

        return fig
    else:
        if target_variable=='Efficiency':
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(Y_m, High_eff_mask, c=X_m, cmap='viridis')
            ax.title.set_text('Efficiency')
            ax.set_xlabel('Fidelity')
            ax.set_ylabel('Efficiency')
            # cbar = fig.colorbar()
            # cbar.set_label("Curvature")
            return fig
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(Y_m, Z_m, c=X_m, cmap='viridis')
            plt.title(plot_title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Fidelity')
            ax.set_ylabel(plot_title)
            # cbar = fig.colorbar()
            # cbar.set_label("Curvature")
            return fig

def plot_boundary_heatmap_old(jsonl_path='JSONoutput/data_storage_11by11_curvature_change.jsonl',
                          target_variable='Efficiency',
                          y_min=0.999, y_max=1.0,
                          pixel_height=0.00005):
    if not os.path.exists(jsonl_path):
        print(f"Error: The file {jsonl_path} does not exist.")
        return

    # 1. Read JSONL dataset safely
    data_runs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data_runs.append(json.loads(line))
    df = pd.DataFrame(data_runs)

    # Clean numeric lists to extract real components out of possible dictionaries
    def clean_numeric_list(val):
        if not isinstance(val, list): return []
        return [x['real'] if isinstance(x, dict) and 'real' in x else x for x in val if
                isinstance(x, (int, float, dict))]

    df['fidelity_ion_list'] = df['fidelity_ion_list'].apply(clean_numeric_list)
    df['efficiency_ion_list'] = df['efficiency_ion_list'].apply(clean_numeric_list)

    grouped = df.groupby('Curvature')

    X, Y, Z = [], [], []

    # 2. Extract Pareto Frontier elements using the true combinations
    for curvature_val, group in grouped:
        fids_each_curve = []
        effs_each_curve = []
        run_mapping = []

        # Flatten arrays while tracking which row/run index each point came from
        for idx, row in group.iterrows():
            fids = row['fidelity_ion_list']
            effs = row['efficiency_ion_list']

            fids_each_curve.extend(fids)
            effs_each_curve.extend(effs)
            run_mapping.extend([row] * len(fids))  # Keep pointer to full run metadata row

        if not fids_each_curve:
            continue

        # Use find_max_optimization_boundary to figure out WHICH combinations are optimal
        # We pass the flattened data to evaluate the global curve boundary for this curvature
        optimal_pairs, _ = find_max_optimization_boundary(fids_each_curve, effs_each_curve, plot=False)
        optimal_set = set(optimal_pairs)

        # Re-verify and harvest the targeted variable for every single matching Pareto pair
        for f, e, origin_row in zip(fids_each_curve, effs_each_curve, run_mapping):
            if (f, e) in optimal_set and target_variable in origin_row:
                X.append(curvature_val)
                Y.append(f)
                Z.append(origin_row[target_variable])

    if not X:
        print(f"Error: No data extracted for target variable '{target_variable}'. Check your JSON keys.")
        return

    X, Y, Z = map(np.array, [X, Y, Z])

    # Apply dynamic visibility mask based on Y limits
    mask = (Y >= y_min) & (Y <= y_max)
    X_m, Y_m, Z_m = X[mask], Y[mask], Z[mask]

    if len(Z_m) == 0:
        print("No data points found inside the visible Y range.")
        return

    # 3. Calculate optimal column widths to make blocks blend horizontally
    unique_curves = np.sort(np.unique(X_m))
    col_width = np.min(np.diff(unique_curves)) if len(unique_curves) > 1 else 0.05

    if target_variable=='Efficiency':
        plot_title='Efficiency'
    elif target_variable=='loss_spots':
        plot_title='Amplitude non-uniformity'
    elif target_variable=='loss_spots_phase':
        plot_title='Phase non-uniformity'
    elif target_variable=='loss_local_phase_std':
        plot_title='Phase local non-uniformity'
    else:
        f'{target_variable.replace("_", " ").title()}'

    # 4. Generate the Plot
    if False:
        fig,ax= plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('plasma' if 'loss' in target_variable.lower() else 'viridis')
        norm = plt.Normalize(vmin=Z_m.min(), vmax=Z_m.max())

        # Build the rectangular flat blocks mapping Y to the center of each bar
        ax.bar(x=X_m, height=pixel_height, width=col_width, bottom=Y_m - (pixel_height / 2),
                color=cmap(norm(Z_m)), edgecolor='none', align='center')

        # Correct colorbar configuration linking specifically to our active axes
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # fig.colorbar(sm, ax=plt.gca(), label=target_variable.replace('_', ' ').title())
        fig.colorbar(sm, ax=plt.gca(), label=plot_title)


        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Curvature')
        ax.set_ylabel('Fidelity')# of Pareto Pairs')
        #ax.set_title(f'{target_variable.replace("_", " ").title()} for Pareto-Optimal Pairs', fontsize=12, fontweight='bold')
        ax.set_title(plot_title, fontsize=14, fontweight='bold')

        ax.grid(True, linestyle=':', alpha=0.3)
        #plt.show()

        return fig
    else:
        fig,ax = plt.subplots(figsize=(10, 6))
        ax.scatter(Y_m,Z_m,c=X_m)
        return fig


def heatmap_dataplot_for_optimised_pairs():
    fig1 = plot_boundary_heatmap(jsonl_path='JSONoutput/data_storage_11by11_curvature_change.jsonl',
                          target_variable='Efficiency',
                          y_min=0.999, y_max=1.0,
                          pixel_height=0.00005)
    fig2 = plot_boundary_heatmap(jsonl_path='JSONoutput/data_storage_11by11_curvature_change.jsonl',
                          target_variable='loss_spots',
                          y_min=0.999, y_max=1.0,
                          pixel_height=0.00005)
    fig3 = plot_boundary_heatmap(jsonl_path='JSONoutput/data_storage_11by11_curvature_change.jsonl',
                          target_variable='loss_spots_phase',
                          y_min=0.999, y_max=1.0,
                          pixel_height=0.00005)
    fig3 = plot_boundary_heatmap(jsonl_path='JSONoutput/data_storage_11by11_curvature_change.jsonl',
                          target_variable='loss_local_phase_std',
                          y_min=0.999, y_max=1.0,
                          pixel_height=0.00005)
    plt.show()


if __name__ == '__main__':

    #datfilecreation()
    #curvatureview()
    #ffttest()
    #trialplot()
    #fftpropagatetest()
    #testing_phase()
    #pandastesting()
    #fftlens()
    #plotfromdatastorage() #csv file
    #pickleplot()
    plot_fidelity_vs_efficiency_old() #For the pareto curve
    #heatmap_dataplot_for_optimised_pairs()
