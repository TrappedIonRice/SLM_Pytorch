# importing matplotlib modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import struct
import math

import numpy as np
from PIL import Image


def zbf_func():
    print("heyoo")
    # INPUTS FROM USER
    # filepath_read = input image full filename. Greyscale or viridis colour scale are supported
    # filepath_write = output ZBF full filename
    # X_size_mm = Full X size of the beam array in mm
    # Wave_mm = Wavelength in mm
    # Text_bool = Boolean. Write 1 to create a ZBF text or 0 to creata a ZBF binary
    filepath_read = r'C:\Python programs\slm_comp_pytorch\images\wu1x4_intensity.bmp'
    filepath_read_p = r'C:\Python programs\slm_comp_pytorch\images\wu1x4_phase.bmp'
    filepath_write = r'C:\Users\RiceT\Documents\Zemax\POP\BEAMFILES\LaserBeamSlm_tem01_5_1_0p72_000p200_negCorrect.ZBF'
    #filepath_write = r'C:\Python programs\slm_comp_pytorch\images\zemax_files\LaserBeamSlm_tem01_11_1.ZBF'
    pad_num=1 #odd number only, pad_num+1 is the actual scaling
    X_size_mm = 13*2#*(pad_num+1)*2# for padding #13 original #has to be integer number, otherwise there seems to be a problem.
    Wave_mm = 0.000411
    Text_bool = 0

    def next_largest_power_of_2(n):
        """
            Function that returns the next largest power of 2 value
            For example, if n=20, then it will return 32 (2^5)

            Parameters
            ----------
            n value

            Returns
            -------
            integer

            """
        return pow(2, int(math.log(n) / math.log(2)) + 1)

    # Read Images
    # img = mpimg.imread(filepath_read)

    # Convert to grey scale
    im = np.array(Image.open(filepath_read).convert('L'))  # you can pass multiple arguments in single line
    im_p = np.array(Image.open(filepath_read_p).convert('L'))
    # plt.imshow(im)
    # plt.colorbar()
    # plt.show()

    # im = np.pad(im,
    #             ((im.shape[0] *pad_num//2, im.shape[0] *pad_num//2),
    #              (im.shape[1] *pad_num//2, im.shape[1] *pad_num//2)),
    #             mode='constant', constant_values=0)
    #
    # im_p = np.pad(im_p,
    #               ((im_p.shape[0]*pad_num // 2, im_p.shape[0]*pad_num // 2),
    #                (im_p.shape[1]*pad_num // 2, im_p.shape[1] *pad_num// 2)),
    #               mode='constant', constant_values=0)

    plt.imshow(im_p)
    plt.title("im_p")
    plt.show()




    size = im.shape
    Number_Y_pixels = size[0]
    Number_X_pixels = size[1]
    print("size",size)

    # The array size in POP are power of 2, for example 32 (2^5), 64 (2^6),
    # The array size must be a power of 2 so need to resize

    padding_y = next_largest_power_of_2(Number_Y_pixels) - Number_Y_pixels
    padding_x = next_largest_power_of_2(Number_X_pixels) - Number_X_pixels
    print("mds16")
    padding_y_left = int(padding_y / 2)
    padding_y_right = padding_y - padding_y_left
    padding_x_left = int(padding_x / 2)
    padding_x_right = padding_x - padding_x_left

    im = np.pad(im, ((padding_y_left, padding_y_right), (padding_x_left, padding_x_right)), 'constant',
                constant_values=(0))
    im_p = np.pad(im_p, ((padding_y_left, padding_y_right), (padding_x_left, padding_x_right)), 'constant',
                  constant_values=(0))

    # If the original size of the image was an even number of pixels
    if padding_y_left != padding_y_right:
        column = np.zeros(next_largest_power_of_2(Number_Y_pixels), 1)
        im = np.vstack([im, column])
        im_p = np.vstack([im_p, column])
    if padding_x_left != padding_x_right:
        row = np.zeros(next_largest_power_of_2(Number_X_pixels), 1)
        im = np.hstack([im, row])
        im_p = np.hstack([im_p, row])

    # Finally we flipped the array
    im = np.flipud(im)
    im_p = np.flipud(im_p)

    # Now the array size is a power of 2
    size = im.shape
    Number_Y_pixels = size[0]
    Number_X_pixels = size[1]

    Y_size_mm = Number_Y_pixels * X_size_mm / Number_X_pixels

    # width_byte = width.to_bytes((width.bit_length() + 7) // 8, byteorder='little')
    # height_byte = width.to_bytes((height.bit_length() + 7) // 8, byteorder='little')

    # what needs to be inside the ZBF File
    Version = 1
    Nx = Number_X_pixels
    Ny = Number_Y_pixels
    IsPol = 0  # Will set to unpolarized - The "is polarized" flag; 0 for unpolarized, 1 for polarized.
    Units = 0  # for mm
    Unused = 0  # Unused
    Xspacing = X_size_mm / Number_X_pixels
    Yspacing = Y_size_mm / Number_Y_pixels
    X_ZPos = 0
    X_ZR = 0
    X_W0 = 0
    Y_ZPos = 0
    Y_ZR = 0
    Y_W0 = 0
    Wave = Wave_mm
    Indx = 1
    Rec_Eff = 0
    Sys_Eff = 0

    print("mds34")

    # Normalize the max of the image to 1
    max = 0
    for i in range(0, Number_X_pixels, 1):
        for j in range(0, Number_Y_pixels, 1):
            if (im[i, j]) > max:
                max = im[i, j]

    max_p = 0
    for i in range(0, Number_X_pixels, 1):
        for j in range(0, Number_Y_pixels, 1):
            if (im_p[i, j]) > max_p:
                max_p = im_p[i, j]

    if max == 0:
        # print('Error max of the image = 0')
        sys.exit("Error max of the image = 0")

    if (Text_bool):
        file = open(filepath_write, "w")  # Overwrites the file if the file exists
        # Writing the header text lines
        file.write("A" + '\n')
        file.write(str(Version) + '\n')
        file.write(str(Number_X_pixels) + '\n')
        file.write(str(Number_Y_pixels) + '\n')
        file.write(str(IsPol) + '\n')
        file.write(str(Units) + '\n')
        for i in range(1, 5, 1):  # 4 unused integers
            file.write(str(Unused) + '\n')
        file.write(str(Xspacing) + '\n')  # d is for double for 8 bytes
        file.write(str(Yspacing) + '\n')
        file.write(str(X_ZPos) + '\n')
        file.write(str(X_ZR) + '\n')
        file.write(str(X_W0) + '\n')
        file.write(str(Y_ZPos) + '\n')
        file.write(str(Y_ZR) + '\n')
        file.write(str(Y_W0) + '\n')
        file.write(str(Wave) + '\n')
        file.write(str(Indx) + '\n')
        file.write(str(Rec_Eff) + '\n')
        file.write(str(Sys_Eff) + '\n')
        for i in range(1, 9, 1):  # 8 unused doubles
            file.write(str(Unused) + '\n')
        print("mds46")
        # First we write Ex
        for i in range(0, Number_X_pixels, 1):
            for j in range(0, Number_Y_pixels, 1):
                Ex_r = math.sqrt((im[i, j]) / 2 / max) * np.cos(2 * np.pi * (im_p[i, j]) / max_p)  # MDS
                Ex_i = math.sqrt((im[i, j]) / 2 / max) * np.sin(2 * np.pi * (im_p[i, j]) / max_p)
                file.write(str(Ex_r) + '\n')
                file.write(str(Ex_i) + '\n')

        # We don't write Ey since the beam is not polarized
        # Then we write Ey
        # for j in range(0, Number_X_pixels, 1):
        #    for i in range(0, Number_Y_pixels, 1):
        #        Ey_r = math.sqrt((im[j, i]) / 2 * max)
        #        Ey_i = 0
        #        file.write(str(Ey_r) + '\n')
        #        file.write(str(Ey_i) + '\n')

    else:
        file = open(filepath_write, "wb")  # Overwrites the file if the file exists
        # Writing the header binary lines
        file.write(Version.to_bytes(4, byteorder='little'))
        file.write(Number_X_pixels.to_bytes(4, byteorder='little'))
        file.write(Number_Y_pixels.to_bytes(4, byteorder='little'))
        file.write(IsPol.to_bytes(4, byteorder='little'))
        file.write(Units.to_bytes(4, byteorder='little'))
        for i in range(1, 5, 1):  # 4 unused integers
            file.write(Unused.to_bytes(4, byteorder='little'))
        file.write(struct.pack('d', Xspacing))  # d is for double for 8 bytes
        file.write(struct.pack('d', Yspacing))
        file.write(struct.pack('d', X_ZPos))
        file.write(struct.pack('d', X_ZR))
        file.write(struct.pack('d', X_W0))
        file.write(struct.pack('d', Y_ZPos))
        file.write(struct.pack('d', Y_ZR))
        file.write(struct.pack('d', Y_W0))
        file.write(struct.pack('d', Wave))
        file.write(struct.pack('d', Indx))
        file.write(struct.pack('d', Rec_Eff))
        file.write(struct.pack('d', Sys_Eff))
        for i in range(1, 9, 1):  # 8 unused doubles
            file.write(struct.pack('d', Unused))
        print("mds67")
        # First we write Ex
        for i in range(0, Number_X_pixels, 1):
            for j in range(0, Number_Y_pixels, 1):
                Ex_r = math.sqrt((im[i, j]) / 2 / max) * np.cos(2 * np.pi * (im_p[i, j])  / max_p)  # MDS
                Ex_i = math.sqrt((im[i, j]) / 2 / max) * np.sin(2 * np.pi * (im_p[i, j])  / max_p)
                file.write(struct.pack('d', Ex_r))
                file.write(struct.pack('d', Ex_i))

        # We don't write Ey since the beam is not polarized
        # Then we write Ey
        # for j in range(0, Number_X_pixels, 1):
        #    for i in range(0, Number_Y_pixels, 1):
        #        Ey_r = math.sqrt((im[j, i]) / 2 / max)
        #        Ey_i = 0
        #        file.write(struct.pack('d', Ey_r))
        #        file.write(struct.pack('d', Ey_i))

    file.close()

    print('The file %s has been created!' % filepath_write)
