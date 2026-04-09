import tkinter.filedialog
import matplotlib.pyplot as plt
import numpy
from PIL import Image as im
import numpy as np
from slm_v1 import SLM
import scipy
import laserbeamsize as lbs
from skimage.transform import resize
# from IFTA import plot_gradient
from datetime import datetime

try:
    import cupy as cp
except ImportError:
    cp = np
    print("cupy not installed. Using numpy.")

pi = np.pi

HERMITE_KERNEL_SRC = """
__device__ double hermite_val(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return 2.0 * x;

    // Recurrence relation: H_N(x) = 2x*H_{N-1}(x) - 2(N-1)*H_{N-2}(x)
    double h_n_minus_2 = 1.0; 
    double h_n_minus_1 = 2.0 * x; 
    double h_n = 0.0;

    for (int i = 2; i <= n; ++i) {
        h_n = 2.0 * x * h_n_minus_1 - 2.0 * (i - 1) * h_n_minus_2;
        h_n_minus_2 = h_n_minus_1;
        h_n_minus_1 = h_n;
    }
    return h_n;
}
"""

# Compile the kernel using ElementwiseKernel
# Inputs: n (degree), x (input array). Output: out (result array).
hermite_gpu_func = cp.ElementwiseKernel(
    'int32 n, T x',
    'T out',
    'out = hermite_val(n, x)',
    'hermite_gpu_func',
    preamble=HERMITE_KERNEL_SRC
)

# Gives the complex amplitude of an arbitrary HG beam at an x, y, z position
def hermite_beam(x, y, n=0, m=1, z=0, w=0.1, R=cp.infty, zR=cp.infty, k=0, amp=1):
    return amp * scipy.special.hermite(n)(np.sqrt(2) * x / w) * cp.exp(-x**2 / w**2) * scipy.special.hermite(m)(np.sqrt(2) * y / w) * cp.exp(-y**2 / w**2) * cp.exp(-1j * (k * z - (1 + n + m) * cp.arctan(z / zR) + k * (x**2 + y**2) / (2 * R)))


# Fill a plane with an n, m mode HG beam
def temnm(slm, n=0, m=6, amp=1, w=(0.03, 0.1)):
    w=(w[0]/slm.size[0],w[1]/slm.size[0])
    field = cp.fromfunction(lambda i, j: amp * scipy.special.hermite(n)(np.sqrt(2) * (2 * (i / slm.size[0] - 0.5)) / w[0]) * np.exp(-(2 * (i / slm.size[0] - 0.5))**2 / w[0]**2) * scipy.special.hermite(m)(np.sqrt(2) * (2 * (j / slm.size[1] - 0.5)) / w[1]) * np.exp(-(2 * (j / slm.size[1] - 0.5))**2 / w[1]**2), slm.size)
    return field

def temnm_Offcenter(slm, n=0, m=6, amp=1, w=(0.03, 0.1),x0=0.5,y0=0.5):
    w=(w[0]/slm.size[0],w[1]/slm.size[0])
    field = cp.fromfunction(lambda i, j: amp * scipy.special.hermite(n)(np.sqrt(2) * (2 * (i / slm.size[0] - x0)) / w[0]) * np.exp(-(2 * (i / slm.size[0] - x0))**2 / w[0]**2) * scipy.special.hermite(m)(np.sqrt(2) * (2 * (j / slm.size[1] - y0)) / w[1]) * np.exp(-(2 * (j / slm.size[1] - y0))**2 / w[1]**2), slm.size)
    return field


def temnm_Offcenter2Darb(slm, n=1, m=0, amp=1, w=(0.03, 0.1), x0=0.5, y0=0.5, angle=0):
    # 1. Parse dimensions and normalize widths
    rows, cols = slm.size[0], slm.size[1]
    # Normalization as per your snippet
    w_norm = (w[0] / rows, w[1] / rows) #Original 20/01S/2026
    #w_norm = (np.sqrt(2)*0.25*w[0] / rows, np.sqrt(2)*0.25*w[1] / rows)# For gaussian
    w_norm= (2/(2*pi*0.55*1024*0.5), 2/(2*pi*0.55*1024*0.5))
    # if w[0]>10:  Uncomment only for target_New
    #     w_norm = (2 / (2 * pi * 0.25 * 1024 * 0.5), 2 / (2 * pi * 0.25 * 1024 * 0.5))
    print("rows",rows,"cols",cols)

    # 2. Define the ROI (Square of length 10 * w_norm[0] around x0, y0)
    half_len = 5.0 * w_norm[0]

    # Calculate pixel boundaries and clip to SLM limits
    x_start = int(max(0, (x0 - half_len) * cols))
    x_end = int(min(cols, (x0 + half_len) * cols))
    y_start = int(max(0, (y0 - half_len) * rows))
    y_end = int(min(rows, (y0 + half_len) * rows))

    # Initialize the full field with zeros
    field = cp.zeros((rows, cols), dtype=cp.float64)

    # If the window is outside the SLM, return early
    if x_start >= x_end or y_start >= y_end:
        return field

    # 3. Create sub-grid for the ROI only
    # i: x-coordinates (columns), j: y-coordinates (rows)
    i_sub = cp.linspace(x_start / cols, (x_end - 1) / cols, (x_end - x_start), dtype=cp.float64)
    j_sub = cp.linspace(y_start / rows, (y_end - 1) / rows, (y_end - y_start), dtype=cp.float64)
    I_sub, J_sub = cp.meshgrid(i_sub, j_sub)

    # 4. Rotation Logic (only on the ROI)
    angle_rad = cp.deg2rad(angle-90.0)
    cos_a, sin_a = cp.cos(angle_rad), cp.sin(angle_rad)

    I_rot = cos_a * (I_sub - x0) + sin_a * (J_sub - y0) + x0
    J_rot = -sin_a * (I_sub - x0) + cos_a * (J_sub - y0) + y0
    if False:
        # 5. Hermite and Gaussian calculations (only on the ROI)
        x_in = np.sqrt(2.0) * 2*(I_rot - x0) / w_norm[0]
        y_in = np.sqrt(2.0) * 2*(J_rot - y0) / w_norm[1]

        hx = cp.empty_like(x_in)
        hy = cp.empty_like(y_in)
        hermite_gpu_func(n, x_in, hx)
        hermite_gpu_func(m, y_in, hy)

        gx = cp.exp(-(2*(I_rot - x0)) ** 2 / (w_norm[0] ** 2 ))
        gy = cp.exp(-(2*(J_rot - y0)) ** 2 / (w_norm[1] ** 2 ))

    # 6. Assign sub-grid result to the main field
    #field[y_start:y_end, x_start:x_end] = amp * hx * gx * hy * gy #Original 20/01/2026
    #field[y_start:y_end, x_start:x_end] = amp * ((I_rot - x0) / (2 * w_norm[0])) * cp.exp(-((I_rot - x0) ** 2 / (2 * w_norm[0] ** 2) + (J_rot - y0) ** 2 / (2 * w_norm[1] ** 2)))
    if m==1:  ##ORIGINAL
        field[y_start:y_end, x_start:x_end] = amp * (2*(I_rot - x0) / ( w_norm[0])) * cp.exp(-((I_rot - x0) ** 2 / ( 2*w_norm[0] ** 2) + (J_rot - y0) ** 2 / ( 2*w_norm[1] ** 2)))
    if m==0:
        field[y_start:y_end, x_start:x_end] = amp * cp.exp(-((I_rot - x0) ** 2 / (w_norm[0] ** 2) + (J_rot - y0) ** 2 / (w_norm[1] ** 2)))
        print("gaussian here")
    # plt.imshow(cp.abs(field/cp.max(cp.abs(field))).get())
    # plt.title(("field normalised",w_norm))
    # plt.show()
    return field


def temnm_Offcenter2Darb_old(slm, n=1, m=0, amp=1, w=(0.03, 0.1), x0=0.5, y0=0.5, angle=0):
    """
    Generate a rotated TEM01 field on an SLM grid.

    Parameters:
    - slm: Object with a 'size' attribute representing the SLM's pixel grid.
    - n, m: Hermite polynomial orders for the x and y directions (n=1, m=0 for TEM01).
    - amp: Amplitude of the field.
    - w: Tuple of standard deviations for the Gaussian envelopes in the x and y directions.
    - x0, y0: Center position of the beam in the x and y directions (normalized).
    - angle: Rotation angle in degrees for the beam.

    Returns:
    - field: The rotated TEM01 field as a 2D array.
    """
    # Normalize the width based on the SLM's size
    w = (w[0] / slm.size[0], w[1] / slm.size[0])

    # Create mesh grid for coordinates
    i = cp.linspace(0, 1, slm.size[1])  # x-coordinates (normalized)
    j = cp.linspace(0, 1, slm.size[0])  # y-coordinates (normalized)

    # Create 2D grid of coordinates
    I, J = cp.meshgrid(i, j)

    # Apply rotation to the coordinates (counterclockwise)
    angle_rad = cp.deg2rad(angle)  # Convert angle to radians
    cos_angle = cp.cos(angle_rad)
    sin_angle = cp.sin(angle_rad)

    # Rotate the coordinates (I, J) by the given angle
    I_rot = cos_angle * (I - x0) + sin_angle * (J - y0) + x0
    J_rot = -sin_angle * (I - x0) + cos_angle * (J - y0) + y0

    # Compute the field directly using the 2D arrays I_rot and J_rot
    hermite_x = scipy.special.hermite(n)(np.sqrt(2) * (2 * (I_rot - x0)) / w[0])
    gaussian_x = np.exp(-(2 * (I_rot - x0))**2 / w[0]**2)

    hermite_y = scipy.special.hermite(m)(np.sqrt(2) * (2 * (J_rot - y0)) / w[1])
    gaussian_y = np.exp(-(2 * (J_rot - y0))**2 / w[1]**2)

    # Combine everything to form the field
    field = amp * hermite_x * gaussian_x * hermite_y * gaussian_y

    #plt.imshow(field)
    #plt.title("rotated TEM01")
    #plt.show()

    return field #cp.asarray(field)  # Return the field as a CuPy array (for GPU acceleration)


# propagate SLM-plane field to image plane
def propagate(slm_field):
    return cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))


# backpropagate image plane field to SLM plane
def backpropagate(image_field):
    return cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_field), norm="ortho"))


# Fit gaussian beam position, waist and elliptical angle
def laserbeamsizefromimage(slm, intensity):
    x, y, dx, dy, phi = lbs.beam_size(intensity)
    # print(slm.size)
    # print("The center of the beam ellipse is at (%.1f, %.1f)" % (x, y))
    # print("The ellipse diameter (closest to horizontal) is %.4f" % (dx / slm.size[1]))
    # print("The ellipse diameter (closest to   vertical) is %.4f" % (dy / slm.size[0]))
    # print("The ellipse is rotated %.0f° ccw from horizontal" % (phi * 180 / np.pi))
    # # plt.figure()
    # # lbs.plot_image_analysis(intensity)
    # plt.pause(0.001)
    # print(dx, dy)
    return [x, y, dx , dy , phi]
    # lbs.beam_size_plot(intensity)
    # plt.show()


def calibrated_temnm(slm, input_field, n=0, m=0):
    input_field = cp.array(input_field)
    ## Find natural waist at image plane to determine target waist
    inputbeam = laserbeamsizefromimage(slm, cp.abs(input_field).get() ** 2)
    slm_waist = np.array([inputbeam[2], inputbeam[3]])
    laserbeam = laserbeamsizefromimage(slm, cp.abs(propagate(input_field)).get() ** 2)
    image_waist = np.array([laserbeam[2], laserbeam[3]])
    # print(image_waist)

    ## Generate target field
    target_slm_field = temnm(slm, n=n, m=m, w=slm_waist)
    target_slm_field /= cp.max(target_slm_field)
    target_image_field = temnm(slm, n=n, m=m, w=image_waist)
    target_image_field /= cp.max(target_image_field)

    return target_slm_field, target_image_field

def calibrated_temnm_Offcenter(slm, input_field, n=0, m=0,x0i=0.5,y0i=0.5):
    input_field = cp.array(input_field)
    ## Find natural waist at image plane to determine target waist
    inputbeam = laserbeamsizefromimage(slm, cp.abs(input_field).get() ** 2)
    slm_waist = np.array([inputbeam[2], inputbeam[3]])
    laserbeam = laserbeamsizefromimage(slm, cp.abs(propagate(input_field)).get() ** 2)
    image_waist = np.array([laserbeam[2], laserbeam[3]])
    # print(image_waist)
    ## Generate target field
    target_slm_field = temnm(slm, n=n, m=m, w=slm_waist)
    target_slm_field /= cp.max(target_slm_field)
    target_image_field = temnm_Offcenter(slm, n=n, m=m, w=image_waist,x0=x0i,y0=y0i)
    target_image_field /= cp.max(target_image_field)

    return target_slm_field, target_image_field


def calibrated_temnm_Offcenter2Darb(slm, input_field, n=0, m=0,x0i=0.5,y0i=0.5,angle=0):
    input_field = cp.array(input_field)
    ## Find natural waist at image plane to determine target waist
    inputbeam = laserbeamsizefromimage(slm, cp.abs(input_field).get() **2)
    slm_waist = np.array([inputbeam[2], inputbeam[3]])
    laserbeam = laserbeamsizefromimage(slm, cp.abs(propagate(input_field)).get() **2)
    image_waist = np.array([laserbeam[2], laserbeam[3]])
    print("image_waist",image_waist, "size", (propagate(input_field)).shape)
    print("slm_waist",slm_waist, "size", (input_field).shape)

    ## Generate target field
    target_slm_field = temnm(slm, n=n, m=m, w=slm_waist)
    target_slm_field /= cp.max(target_slm_field)
    target_image_field = temnm_Offcenter2Darb(slm, n=n, m=m, w=image_waist,x0=x0i,y0=y0i,angle=angle)
    target_image_field /= cp.max(target_image_field)

    return target_slm_field, target_image_field

def shift_array(arr, shift=(0, 0)):
    shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
    # out = np.roll(arr, shift, axis=(0, 1))
    out = np.zeros(arr.shape)
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i + shift[0] > 0 and j + shift[1] > 0:
                out[i, j] = arr[i + shift[0], j + shift[1]]
    return out


def stipple(im, size, threshold=0.5):
    im /= cp.max(im)
    im *= im > threshold
    return cp.array(resize(im.get(), size))


# All coordinates and amplitudes normalized to [-1, 1]
class Profile:

    def __init__(self, size=np.array( (1024,1272)), field=None):

        if field is None:
            field = Profile.input_gaussian(size=size)
        self.field = field

        self.amp = np.abs(self.field)
        self.phase = np.angle(self.field)

        self.size = self.field.shape

    def save(self, slm, name):
        slm.fieldtoBMP(self.field, name=name, color=True, correction=False, wavelength=411)
        # slm.phaseToBMP(cp.array(self.phase), name=name + '_phase', color=True)

    # Generate single gaussian beam
    @staticmethod
    def input_gaussian(beam_type=0, amp=1.0, beam_size=np.array((0.5, 0.5)), pos=np.array((0, 0)),
                       size=np.array( (1024,1272)), mesh=False):

        if beam_type == 0:
            # beam_size[0] *= size[0] / size[1]
            #beam_size = np.array([beam_size[0] * size[0] / size[1], beam_size[1]])
            beam_size_0 = numpy.float64(beam_size[0] * size[0] / size[1])
            beam_size_1 = numpy.float64(beam_size[1])

            xx = np.exp(-(np.linspace(-1, 1, size[1]) - pos[1])**2 / beam_size_0**2)
            yy = np.exp(-(np.linspace(-1, 1, size[0]) - pos[0])**2 / beam_size_1**2)
        else:
            xx = np.ones(size[1])
            yy = np.ones(size[0])

        beams = np.meshgrid(xx, yy)
        # print(amp)
        amp_profile = amp * beams[0] * beams[1]
        # phase_profile = np.ones(size) * np.exp(2j * pi)

        if mesh:
            return amp_profile, beams

        return amp_profile

    # Generate array of single pixel spots
    @staticmethod
    def spot_array(n, m, center=np.array((0, 0)), x_pitch=0.1, y_pitch=0.1, size=np.array( (1024,1272)),uni_spacing=True, xarblist0=None, yarblist0=None, anglearblist0=None):
        amp_profile = np.zeros(size)
        spots = np.array([[0, 0]])
        if False:
            if m == 10:
                xarblist = np.multiply(0.0045, [0.29519436, -2.1632774, -0.61761615, -2.63331531, 2.66932284, 1.19780158,
                                               2.34280933, 0.93160313, -0.38616102, -1.63636632])

                yarblist = np.multiply(0.0045, [-2.74000583, -1.98098021, -0.91034715, 0.31423362, 1.06906318, -0.14289163,
                                               -1.66711298, 2.59357481, 1.03003746, 2.43442249])
               # yarblist= np.multiply(0.0035, [-7.84937698,-5.74095644, - 3.96527944, - 2.33190280,
               #  - 0.769912269, 0.774888067,2.33524185,  3.96421253,
               #  5.74425264, 7.85006738])
               # xarblist=np.multiply(0.0000, [-7.84937698,-5.74095644, - 3.96527944, - 2.33190280,
               #  - 0.769912269, 0.774888067,2.33524185,  3.96421253,
               #  5.74425264, 7.85006738])
                #yarblist=np.multiply(0.0085,[-4.00000000,-3.1111111,-2.22222222,-1.33333333,-0.444444444,0.44444444,1.33333333,2.22222222,3.11111111,4.00000000])
                #xarblist = np.multiply(0.0000, [-4.00000000, -3.1111111, -2.22222222, -1.33333333, -0.444444444, 0.44444444,
                 #                               1.33333333, 2.22222222, 3.11111111, 4.00000000])
            if m==80:
                print("80 ION ARBITRARY")
                xarblist = np.multiply(0.0075,np.array([
                    -8.43580e-01, 8.43570e-01, -1.78623e+00, -2.95118e+00, -4.04263e+00,
                    -4.55658e+00, -6.42640e-01, -3.88401e+00, -2.03045e+00, -3.60662e+00,
                    -2.28609e+00, -4.64404e+00, -4.47901e+00, -3.35112e+00, -3.58618e+00,
                    -2.58774e+00, -1.73573e+00, -2.64848e+00, -8.46630e-01, -8.60300e-01,
                    8.46630e-01, -2.60708e+00, -0.00000e+00, -1.72420e+00, -1.65403e+00,
                    -2.73003e+00, -1.43440e+00, -3.53453e+00, -5.46590e-01, -1.89447e+00,
                    -1.08796e+00, -9.25810e-01, -3.06206e+00, -9.87610e-01, -1.92357e+00,
                    -2.95357e+00, 5.46590e-01, -1.94037e+00, -0.00000e+00, -7.81550e-01,
                    6.42630e-01, 1.00000e-05, 1.92355e+00, 0.00000e+00, 3.06205e+00,
                    9.87620e-01, 1.08795e+00, 1.94038e+00, -0.00000e+00, 9.25820e-01,
                    2.03044e+00, 0.00000e+00, 3.35112e+00, 8.81560e-01, 2.28608e+00,
                    1.78623e+00, 1.65402e+00, 0.00000e+00, 2.58774e+00, 1.72421e+00,
                    1.73573e+00, 8.60300e-01, 3.58618e+00, 2.64848e+00, 2.60708e+00,
                    4.64404e+00, 3.60663e+00, 3.53454e+00, 4.55658e+00, 2.73003e+00,
                    4.47901e+00, 1.89448e+00, 2.95117e+00, 7.81560e-01, 4.04263e+00,
                    3.88402e+00, 2.95358e+00, 1.43439e+00, -0.00000e+00, -8.81560e-01
                ]))

                # Equilibrium v positions
                yarblist = np.multiply(0.0075,np.array([
                    1.51036, 1.51036, -1.90279, 2.48373, 2.40894, -1.27778, 4.55962, -2.6666,
                    3.03158, -0.58862, 1.7559, -0.01069, 1.208, 1.46229, 0.43761, 0.7287,
                    0.0775, -0.26196, 0.49489, -0.46223, 0.49489, -1.26103, 0.93977, -0.88536,
                    1.05042, -2.36841, 2.25253, -1.64725, 2.67604, -2.96843, 3.49901, -2.41803,
                    3.56142, -3.44371, 4.21462, -3.50584, 2.67604, -4.13428, 3.6061, -4.60435,
                    4.55962, -3.86231, 4.21463, -2.85736, 3.56143, -3.4437, 3.49901, -4.13428,
                    1.90366, -2.41802, 3.03159, -1.9025, 1.4623, -1.43358, 1.7559, -1.90278,
                    1.05042, -0.95794, 0.7287, -0.88535, 0.0775, -0.46223, 0.43762, -0.26195,
                    -1.26102, -0.01068, -0.58861, -1.64725, -1.27777, -2.3684, 1.20801, -2.96842,
                    2.48374, -4.60434, 2.40895, -2.66659, -3.50584, 2.25254, -0.01408, -1.43359
                ]))
                #for ilen in np.arange(0, len(xarblist), 1):
                #    amp_profile[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])] = 1.0
                #    spots = np.append(spots,
                #                      [[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])]],
                #                      axis=0)

            if m==27:
                print("27 ION ARBITRARY")
                yarblist = np.multiply(1,np.array([-0.0035806, -0.0278236, -0.0196273, -0.0221673, -0.0233508, -0.0152492,
                                                        -0.0148453, -0.0110340, -0.0073132, -0.0196273, -0.0148453, -0.0000000,
                                                        -0.0073132, -0.0042195, -0.0042195, 0.0042195, 0.0042195, 0.0129463,
                                                        0.0000000, 0.0148453, 0.0196273, 0.0035806, 0.0152492, 0.0233508,
                                                        0.0311525, 0.0196273, 0.0071222]))

                xarblist = np.multiply(1,np.array([-0.0066115, -0.0082616, -0.0064758, 0.0150647, 0.0000016, 0.0000016,
                                                        -0.0122012, 0.0059688, -0.0131891, 0.0064789, 0.0122044, -0.0135962,
                                                        0.0131923, -0.0208378, 0.0208410, -0.0208378, 0.0208411, 0.0197154,
                                                        0.0135994, 0.0122044, 0.0064789, 0.0066147, 0.0000016, 0.0000016,
                                                        0.0000016, -0.0064758, 0.0000016]))

            if m==471:
                print("47 ION ARBITRARY original")
                yarblist = np.multiply(0.0100,np.array([
                    -0.0, -0.47742, -3.70982, -2.61698, -2.95564, -4.15366, -3.11344,
                    -0.94963, -3.70982, -1.4712, -2.03323, -1.97937, -1.4712, -0.9751,
                    -2.61698, -2.95565, -1.97937, -0.0, -0.9751, -0.5626, -0.5626,
                    -1.72617, -1.72617, 0.5626, 0.5626, 1.72617, 1.72617, 0.9751,
                    0.0, 1.4712, 1.97937, 2.95564, 2.61698, 1.97937, 0.47742, 0.47742,
                    1.4712, 2.03323, 3.70982, 0.9751, 3.11344, 4.15366, 2.95564,
                    2.61698, 3.70982, 0.94963, -0.47742
                ]))

                xarblist = np.multiply(0.0100,np.array([
                    2.10000e-04, -8.81530e-01, -1.10155e+00, -8.63440e-01, 2.00862e+00,
                    2.10000e-04, 2.10000e-04, 2.10000e-04, 1.10197e+00, -7.95410e-01,
                    2.10000e-04, -1.62683e+00, 7.95840e-01, -1.75855e+00, 8.63860e-01,
                    -2.00820e+00, 1.62725e+00, -1.81283e+00, 1.75897e+00, -2.77838e+00,
                    2.77880e+00, -2.62830e+00, 2.62872e+00, -2.77838e+00, 2.77881e+00,
                    -2.62829e+00, 2.62872e+00, -1.75855e+00, 1.81325e+00, -7.95410e-01,
                    1.62725e+00, -2.00820e+00, 8.63860e-01, -1.62683e+00, 8.81960e-01,
                    -8.81530e-01, 7.95840e-01, 2.10000e-04, 1.10197e+00, 1.75898e+00,
                    2.10000e-04, 2.10000e-04, 2.00862e+00, -8.63440e-01, -1.10155e+00,
                    2.10000e-04, 8.81960e-01
                ]))

                anglearblist=np.zeros(np.size(xarblist))
                anglearblist = np.linspace(0, 90, np.size(xarblist))

                #for ilen in np.arange(0, len(xarblist), 1):
                #    amp_profile[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])] = 1.0
                #    spots = np.append(spots,[[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])]],axis=0)

            if m==21:
                print("21 ION ARBITRARY")
                yarblist = np.multiply(0.001875,np.array([np.float64(6.708568615511826), np.float64(-0.262762385917175), np.float64(-0.2627623876367533), np.float64(-0.2627623726313839), np.float64(-3.9787779198443207), np.float64(3.4532531546293823), np.float64(3.4532531612685697), np.float64(-7.2340933766251165), np.float64(2.4554469988851824), np.float64(10.302280034326998), np.float64(6.7001766595219925), np.float64(-7.225701413292778), np.float64(-7.2257013918070765), np.float64(-2.9809717635010053), np.float64(-2.9809717401406544), np.float64(6.7001766330484065), np.float64(2.4554470066368053), np.float64(-10.82780479869527), np.float64(10.302280064731704), np.float64(-10.827804811283903), np.float64(-3.9787779276714565)]))

                xarblist = np.multiply(0.001875,np.array([np.float64(-0.27031919668926946), np.float64(-0.2703192027491966), np.float64(3.852933695501573), np.float64(-4.3935720993598375), np.float64(-2.653070484049261), np.float64(2.112432071173244), np.float64(-2.6530704661615263), np.float64(-0.2703192151264276), np.float64(-7.59933960103416), np.float64(-2.762623815822687), np.float64(5.147918410790347), np.float64(5.147918417326782), np.float64(-5.688556850496022), np.float64(7.058701183257072), np.float64(-7.599339607460892), np.float64(-5.688556837031819), np.float64(7.058701182855652), np.float64(-2.762623812895744), np.float64(2.221985355102846), np.float64(2.2219853624173616), np.float64(2.112432060276719)]))

                anglearblist=np.zeros(np.size(xarblist))
                anglearblist = 90.0+np.array([np.float64(3.2497317447095186e-07), np.float64(51.985112459380616), np.float64(90.00000013747618), np.float64(-90.00000013576972), np.float64(-129.63307681088895), np.float64(50.3669228667596), np.float64(-50.36692274477683), np.float64(-179.99999992104995), np.float64(-78.7241520628606), np.float64(-37.31081421621002), np.float64(54.20370788013457), np.float64(125.79629270525027), np.float64(-125.7962927700287), np.float64(101.27584752298303), np.float64(-101.27584745668526), np.float64(-54.20370812373931), np.float64(78.72415232455614), np.float64(-142.68918503165065), np.float64(37.31081373020869), np.float64(142.68918544144535), np.float64(129.63307702606372)])

                #for ilen in np.arange(0, len(xarblist), 1):
                #    amp_profile[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])] = 1.0
                #    spots = np.append(spots,[[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])]],axis=0)

            if m==47:
                print("47 ION 2d")
                yarblist = np.multiply(0.001875,np.array([np.float64(-3.9153676373864976), np.float64(1.5958924653913373), np.float64(-0.24789189083258933), np.float64(1.595892454424498), np.float64(3.51795397976523), np.float64(-4.013737758115343), np.float64(5.433885056698667), np.float64(-7.892236416750612), np.float64(7.396452641781203), np.float64(3.5179539685022996), np.float64(-5.929668845960811), np.float64(9.858898722101284), np.float64(-11.662620477684406), np.float64(7.604459696761918), np.float64(11.16683669238147), np.float64(3.4195838537920924), np.float64(-0.24789188859652656), np.float64(5.433885069420214), np.float64(-10.354682503336655), np.float64(7.396452629423975), np.float64(9.858898720238939), np.float64(-2.4206606397110795), np.float64(1.9248768489256154), np.float64(11.166836707272271), np.float64(-4.013737754913859), np.float64(-8.10024348265401), np.float64(11.77623483845139), np.float64(-12.272018621227247), np.float64(-2.091676238293306), np.float64(1.9248768589685479), np.float64(-2.091676245975052), np.float64(-7.8922364189119945), np.float64(15.793584660572392), np.float64(-16.289368441609966), np.float64(6.418611290583689), np.float64(6.4186112993839775), np.float64(-5.9296688497419225), np.float64(14.079456475296887), np.float64(14.079456473275863), np.float64(-0.24789189579311033), np.float64(-14.57524025456667), np.float64(-10.354682507001325), np.float64(-2.420660631514017), np.float64(-14.575240261414221), np.float64(-11.662620484105023), np.float64(-6.914395073486592), np.float64(-6.914395074523181)]))

                xarblist = np.multiply(0.001875*size[1]/size[0],np.array([np.float64(-0.14402684160131637), np.float64(3.2612895167361255), np.float64(-0.14402684360513102), np.float64(-3.5493432086919947), np.float64(6.648329080209887), np.float64(-6.936382769095556), np.float64(-3.2167285170025828), np.float64(-6.427674579803346), np.float64(6.13962088627167), np.float64(-6.9363827787988335), np.float64(2.92867482361023), np.float64(-3.4794570763251618), np.float64(-7.900517078211293), np.float64(-0.14402685630384013), np.float64(-7.90051707820557), np.float64(-0.1440268471400142), np.float64(6.857957322617093), np.float64(2.9286748167110237), np.float64(-3.4794570715019115), np.float64(-6.427674580670901), np.float64(3.1914033701945352), np.float64(-10.874976276753143), np.float64(-10.874976280995716), np.float64(7.612463373661678), np.float64(6.648329082223555), np.float64(-0.14402684149046635), np.float64(-0.14402685135616655), np.float64(-0.1440268484059051), np.float64(3.2612895170856055), np.float64(10.586922582716923), np.float64(-3.5493432031882444), np.float64(6.139620889764666), np.float64(-0.14402685297854273), np.float64(-0.14402685441374838), np.float64(-10.29534867565972), np.float64(10.007294978375237), np.float64(-3.2167285072119336), np.float64(4.110977719807727), np.float64(-4.399031427141683), np.float64(-7.146011014941164), np.float64(-4.399031426941241), np.float64(3.191403379424469), np.float64(10.586922585459911), np.float64(4.110977723488051), np.float64(7.61246337791134), np.float64(10.007294980894486), np.float64(-10.295348673951082)]))

                anglearblist=np.zeros(np.size(xarblist))
                anglearblist = 90.0+np.array([np.float64(-9.926423015743774e-08), np.float64(-96.57596690631269), np.float64(-51.219935504870456), np.float64(96.57596691240425), np.float64(-106.87386975001493), np.float64(73.12613025480017), np.float64(120.83287506611262), np.float64(62.292384093209634), np.float64(-117.70761592154263), np.float64(106.87386967786495), np.float64(-59.1671249054077), np.float64(141.5075167118805), np.float64(60.58666674512219), np.float64(179.99999991737988), np.float64(119.41333328276657), np.float64(-179.9999999614754), np.float64(-90.00000002999623), np.float64(-120.83287522494996), np.float64(38.492483255639144), np.float64(117.70761594769567), np.float64(-141.50751686104718), np.float64(87.46163284209767), np.float64(92.53836713809196), np.float64(-119.41333332292612), np.float64(-73.12613029305714), np.float64(-6.447747537362172e-08), np.float64(179.9999999162039), np.float64(8.47137792365958e-09), np.float64(-83.42403312567035), np.float64(-92.53836716528025), np.float64(83.4240331320487), np.float64(-62.292384073423236), np.float64(179.9999999049066), np.float64(1.0753090160404035e-07), np.float64(101.58840969182394), np.float64(-101.58840970353917), np.float64(59.16712467795351), np.float64(-147.379221652306), np.float64(147.3792216222973), np.float64(89.99999996590905), np.float64(32.62077833488603), np.float64(-38.49248320555845), np.float64(-87.46163286328525), np.float64(-32.620778365282355), np.float64(-60.586666716420815), np.float64(-78.41159029358), np.float64(78.41159029393553)])

        if uni_spacing == False:
            xarblist = xarblist0; yarblist = yarblist0; anglearblist = anglearblist0;
            for ilen in np.arange(0, len(xarblist), 1):
                amp_profile[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])] = 1.0
                #spots = np.append(spots,[[int((xarblist[ilen] + 0.5) * size[0]), int((yarblist[ilen] + 0.5) * size[1])]],axis=0) #MDS original integer spots
                spots = np.append(spots,[[((xarblist[ilen] + 0.5) * size[0]), ((yarblist[ilen] + 0.5) * size[1])]],axis=0)


        if uni_spacing:
            for i in np.linspace(-0.5 * n * x_pitch + center[0], 0.5 * n * x_pitch + center[0], n, endpoint=True):
                if n <= 1:
                    i = 0
                if m > 1:
                    for j in np.linspace(-0.5 * m * y_pitch + center[1], 0.5 * m * y_pitch + center[1], m, endpoint=True):
                        # x = (i + 0.5) * size[0]
                        amp_profile[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])] = 1.0
                        spots = np.append(spots, [[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])]], axis=0)
                else:
                    amp_profile[int((i + 0.5) * size[0]), int(0.5 * size[1])] = 1.0
                    spots = np.append(spots, [[int((i + 0.5) * size[0]), int(0.5 * size[1])]], axis=0)
            anglearblist = np.linspace(90,90,np.shape(spots[1:])[0])
            #anglearblist=np.array([90,-90,90,-90,90])
        print("spots", spots)
        print("spots[1:]", spots[1:])
        print("anglearblist", anglearblist)
        return [amp_profile * np.exp(2j * pi), spots[1:], anglearblist]

    #For 2 beams with one at the center
    def spot_array_MDS(n=1, m=1, center=np.array((0, 0)), x_pitch=0.1, y_pitch=0.1, size=np.array( (1024,1272))):
        amp_profile = np.zeros(size)
        spots = np.array([[0, 0]])
        #Central beam
        amp_profile[int((0.5) * size[0]), int(0.5 * size[1])] = 1.0
        spots = np.append(spots, [[int((0.5) * size[0]), int(0.5 * size[1])]], axis=0)
        for i in np.linspace((n-1) * x_pitch + center[0], 0 * x_pitch + center[0], (n-1), endpoint=True):
            for j in np.linspace(m * y_pitch + center[1], 0 * y_pitch + center[1], m, endpoint=True):
                amp_profile[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])] = 1.0
                spots = np.append(spots, [[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])]], axis=0)
        return [amp_profile * np.exp(2j * pi), spots[1:]]

    # Generate array of gaussian beams
    @staticmethod
    def gaussian_array(n, m, waist=(0.02, 0.02), center=np.array((0, 0)), x_pitch=0.1, y_pitch=0.1,
                       size=np.array( (1024,1272)), amps=None):

        if amps is None:
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)
        # print(amps)

        spot_array, spots,spot_angles = Profile.spot_array(n, m, center, x_pitch, y_pitch, size)
        amp = np.copy(spot_array)
        for i in range(len(spots)):
            amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])

        amp -= spot_array
        # amp = np.abs(amp)
        amp /= np.max(np.abs(amp))
        return [amp, spots]

    # Generate a target output array of gaussian beams
    @staticmethod
    def target_output_array(n, m, input_profile, compensate_tem=0,center=np.array((0, 0)),
                              x_pitch=0.1, y_pitch=0.1, size=np.array( (1024,1272)), amps=None, phases=None,
                              global_phase=0, tem=False, double_amps=False):

        if amps is None :
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)
        
        if phases is None :
            
            phases = [global_phase for _ in range(n * m)]
        phases = np.array(phases)
        # print(amps)

        # input_profile = Profile.input_gaussian(beam_size=input_waist, pos=input_center, size=size, amp=1)
        transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(input_profile), norm="ortho"))
        if tem:
            slm = SLM(size=size)
            target_slm_field, transform = calibrated_temnm(slm, input_profile, n=0, m=1)
            transform = transform
            
            
            # if double_amps:
            #     separation = np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2

        spot_array, spots = Profile.spot_array(n, m, center, y_pitch, x_pitch, size)
        amp = cp.zeros(transform.shape, dtype=np.complex128)
        print('phase=',phases)
        
        
        for i in range(len(spots)):
            
            # shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
            shift = (int(spots[i][0] - size[0] / 2), int(spots[i][1] - size[1] / 2))
            if double_amps:
                half_amps = cp.ones(transform.shape,dtype=cp.complex128)
                half_amps[:, :len(half_amps[0]) // 2] = amps[2 * i]* cp.exp(1j * phases[2 * i])
                # print(2 * i)
                # print(amps[2 * i])
                # print(2 * i + 1)
                # print(amps)
                half_amps[:, len(half_amps[0]) // 2:] = amps[2 * i + 1]* cp.exp(1j * phases[2 * i])
                
                # print(amps[2 * i + 1])
                transform_shifted = transform * half_amps
                
                amp += cp.roll(transform_shifted, shift, axis=(0, 1)) 
            else:
                # print(type(shift))
                # print(type(amp))
                # print(type(transform))
                amp += cp.array(np.roll(transform, (shift), axis=(0, 1))) * cp.array(amps[i] * np.exp(1j * phases[i]))
            # amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])
        
            # slm.ampToBMP(np.abs(amp), 'shifted_transform', True)
        # amp -= spot_array
        # amp = np.abs(amp)
        # amp /= np.max(np.abs(amp))
        # total_input = np.sum(np.abs(input_profile))
        if tem and double_amps:
            # print('tem and doubleamps')
            # print(transform.shape)
            # print(np.where(np.abs(transform) == np.max(np.abs(transform)))[1][0])
            # print(len(transform[0]) / 2)
            double_spots = []
            separation = int(np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2)
            # print(separation)
            for spot in spots:
                double_spots.append([spot[0], spot[1] - separation])
                double_spots.append([spot[0], spot[1] + separation])
            spots = double_spots
           
        amp *= np.sqrt(np.sum(np.abs(input_profile)**2) / np.sum(np.abs(amp)**2))
        # print(spots)
       
        return [amp, spots]

   #Two beam scan
    def target_output_array_MDS(n, m, input_profile, middlecompensate=[[0], [0], [[512, 636]]],
                                  center=np.array((0, 0)),
                                  x_pitch=0.1, y_pitch=0.1, size=np.array((1024, 1272)), amps=None, phases=None,
                                  global_phase=0, tem=False, double_amps=False):

        if amps is None:
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)

        if phases is None:
            phases = [global_phase for _ in range(n * m)]
        phases = np.array(phases)
        # print(amps)
        # input_profile = Profile.input_gaussian(beam_size=input_waist, pos=input_center, size=size, amp=1)
        transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(input_profile), norm="ortho"))
        if tem:
            slm = SLM(size=size)
            target_slm_field, transform = calibrated_temnm(slm, input_profile, n=0, m=1)
            transform = transform

            # if double_amps:
            #     separation = np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2

        spot_array, spots = Profile.spot_array_MDS(n, m, center, y_pitch, x_pitch, size)

        amp = cp.zeros(transform.shape, dtype=np.complex128)

        for i in range(len(spots)):

            # shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
            shift = (int(spots[i][0] - size[0] / 2), int(spots[i][1] - size[1] / 2))
            # print(shift)
            if double_amps:
                half_amps = cp.ones(transform.shape, dtype=cp.complex128)
                half_amps[:, :len(half_amps[0]) // 2] = amps[2 * i] * cp.exp(1j * phases[2 * i])
                # print(2 * i)
                # print(amps[2 * i])
                # print(2 * i + 1)
                # print(amps)
                half_amps[:, len(half_amps[0]) // 2:] = amps[2 * i + 1] * cp.exp(1j * phases[2 * i])

                # print(amps[2 * i + 1])
                transform_shifted = transform * half_amps

                amp += cp.roll(transform_shifted, shift, axis=(0, 1))
            else:
                # print(type(shift))
                # print(type(amp))
                # print(type(transform))
                amp += cp.array(np.roll(transform, (shift), axis=(0, 1))) * cp.array(amps[i] * np.exp(1j * phases[i]))
            # amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])

            # slm.ampToBMP(np.abs(amp), 'shifted_transform', True)
        # amp -= spot_array
        # amp = np.abs(amp)
        # amp /= np.max(np.abs(amp))
        # total_input = np.sum(np.abs(input_profile))
        if tem and double_amps:
            # print('tem and doubleamps')
            # print(transform.shape)
            # print(np.where(np.abs(transform) == np.max(np.abs(transform)))[1][0])
            # print(len(transform[0]) / 2)
            double_spots = []
            separation = int(np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2)
            # print(separation)
            for spot in spots:
                double_spots.append([spot[0], spot[1] - separation])
                double_spots.append([spot[0], spot[1] + separation])
            spots = double_spots
        # plt.figure(figsize=(8, 6))
        # plt.imshow(np.abs(amp.get()), cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.title('wu result')
        # plt.show()

        # amp=amp+amp1
        amp *= np.sqrt(np.sum(np.abs(input_profile) ** 2) / np.sum(np.abs(amp) ** 2))
        print("spots",spots)

        return [amp, spots]

    def target_output_array_bokai(n, m, input_profile, middlecompensate=[[0], [0],[[512,636]]], center=np.array((0, 0)),
                            x_pitch=0.1, y_pitch=0.1, size=np.array((1024, 1272)), amps=None, phases=None,
                            global_phase=0, tem=False, double_amps=False):
    
        if amps is None:
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)

        if phases is None:
            phases = [global_phase for _ in range(n * m)]
        phases = np.array(phases)
        # print(amps)
        # input_profile = Profile.input_gaussian(beam_size=input_waist, pos=input_center, size=size, amp=1)
        transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(input_profile), norm="ortho"))
        if tem:
            slm = SLM(size=size)
            target_slm_field, transform = calibrated_temnm(slm, input_profile, n=0, m=1)
            transform = transform

            # if double_amps:
            #     separation = np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2

        spot_array, spots,spot_angles = Profile.spot_array(n, m, center, y_pitch, x_pitch, size)

        amp = cp.zeros(transform.shape, dtype=np.complex128)
        
        for i in range(len(spots)):

            # shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
            shift = (int(spots[i][0] - size[0] / 2), int(spots[i][1] - size[1] / 2))
            # print(shift)
            if double_amps:
                half_amps = cp.ones(transform.shape, dtype=cp.complex128)
                half_amps[:, :len(half_amps[0]) // 2] = amps[2 * i] * cp.exp(1j * phases[2 * i])
                # print(2 * i)
                # print(amps[2 * i])
                # print(2 * i + 1)
                # print(amps)
                half_amps[:, len(half_amps[0]) // 2:] = amps[2 * i + 1] * cp.exp(1j * phases[2 * i])

                # print(amps[2 * i + 1])
                transform_shifted = transform * half_amps

                amp += cp.roll(transform_shifted, shift, axis=(0, 1))
            else:
                # print(type(shift))
                # print(type(amp))
                # print(type(transform))
                amp += cp.array(np.roll(transform, (shift), axis=(0, 1))) * cp.array(amps[i] * np.exp(1j * phases[i]))

            # amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])

            # slm.ampToBMP(np.abs(amp), 'shifted_transform', True)
        # amp -= spot_array
        # amp = np.abs(amp)
        # amp /= np.max(np.abs(amp))
        # total_input = np.sum(np.abs(input_profile))
        if tem and double_amps:
            # print('tem and doubleamps')
            # print(transform.shape)
            # print(np.where(np.abs(transform) == np.max(np.abs(transform)))[1][0])
            # print(len(transform[0]) / 2)
            double_spots = []
            separation = int(np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2)#MDS added +1
            # print(separation)
            for spot in spots:
                double_spots.append([spot[0], spot[1] - separation])
                double_spots.append([spot[0], spot[1] + separation])
            spots = double_spots
        # plt.figure(figsize=(8, 6))
        # plt.imshow(np.abs(amp.get()), cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.title('wu result')
        # plt.show()
        if False: #For force tilt
            xytiltgrid = np.ones((1024, 11))
            for ii in range(0, 11, 1):
                xytiltgrid[:, ii] = -5 + ii
            #plt.imshow(xytiltgrid)
            #plt.show()
            amp[:, 630:641] = amp[:, 630:641] * cp.array(np.exp(-1j * 0.166 * xytiltgrid))
        # amp=amp+amp1
        amp *= np.sqrt(np.sum(np.abs(input_profile) ** 2) / np.sum(np.abs(amp) ** 2))
        # print(spots)

        return [amp, spots]

    def target_output_array_bokai_Offcenter(n, m, input_profile, middlecompensate=[[0], [0], [[512, 636]]],
                                  center=np.array((0, 0)),
                                  x_pitch=0.1, y_pitch=0.1, size=np.array((1024, 1272)), amps=None, phases=None,
                                  global_phase=0, tem=False, double_amps=False):

        if amps is None:
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)

        if phases is None:
            phases = [global_phase for _ in range(n * m)]
        phases = np.array(phases)
        # print(amps)
        # input_profile = Profile.input_gaussian(beam_size=input_waist, pos=input_center, size=size, amp=1)
        #transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(input_profile), norm="ortho"))

        slm = SLM(size=size)

        if tem:
            target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=1, x0i=0.5, y0i=0.5)
            transform = transform
        else:
            target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=0, x0i=0.5, y0i=0.5)
            transform = transform
        #plt.imshow(np.abs(transform.get()))
        #plt.title("transform abs")
        #plt.show()
        #if tem:
        #    slm = SLM(size=size)
        #    target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=1)
        #    transform = transform

            # if double_amps:
            #     separation = np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2

        spot_array, spots,spot_angles = Profile.spot_array(n, m, center, y_pitch, x_pitch, size)

       # plt.imshow(np.abs(spot_array))
        #plt.title("spotarray")
        #plt.show()
        amp = cp.zeros(transform.shape, dtype=np.complex128)
        print("spot_array",spot_array,"spots",spots)
        spotnum=0
        print("start spot",datetime.now())
        laserbeam_waisttem = laserbeamsizefromimage(slm, cp.abs(propagate(input_profile)).get() ** 2)
        image_waisttem = np.array([laserbeam_waisttem[2], laserbeam_waisttem[3]])
        #starts here for uniform spacing, independent of spot_array
        if True:
            for i in np.linspace(-0.5 * n * x_pitch + center[0], 0.5 * n * x_pitch + center[0], n, endpoint=True):
                if n <= 1:
                    i = 0
                if m > 1:
                    for j in np.linspace(-0.5 * m * y_pitch + center[1], 0.5 * m * y_pitch + center[1], m, endpoint=True):
                        x0ic=(i + 0.5); y0ic=(j + 0.5)
                        #target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=0, x0i=x0ic,
                        #                                                         y0i=y0ic)
                        if tem:
                            transform_tem = temnm_Offcenter(slm, n=0, m=1, w=image_waisttem, x0=x0ic, y0=y0ic)
                            transform_tem /= cp.max(transform_tem)
                            if double_amps:
                                half_amps = cp.ones(transform.shape, dtype=cp.complex128)
                                #half_amps[:, :len(half_amps[0]) // 2] = amps[2 * i] * cp.exp(1j * phases[2 * i])
                                #half_amps[:, len(half_amps[0]) // 2:] = amps[2 * i + 1] * cp.exp(1j * phases[2 * i])
                                half_amps[:, :int(y0ic*len(half_amps[1]))] = amps[2 * spotnum] * cp.exp(1j * phases[2 * spotnum])
                                half_amps[:, int(y0ic*len(half_amps[1])):] = amps[2 * spotnum + 1] * cp.exp(1j * phases[2 * spotnum+1])
                                amp += transform_tem * cp.array(half_amps)#[spotnum])

                            else:
                                amp+=transform_tem* cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        else:
                            transform_tem = temnm_Offcenter(slm, n=0, m=0, w=image_waisttem, x0=x0ic, y0=y0ic)
                            transform_tem /= cp.max(transform_tem)
                            amp+=transform_tem* cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        spotnum+=1
                else:
                    x0ic = (i + 0.5)
                    #target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=0, x0i=x0ic,
                    #                                                         y0i=0.5)
                    if tem: 
                        transform_tem = temnm_Offcenter(slm, n=0, m=1, w=image_waisttem, x0=x0ic, y0=0.5)
                        transform_tem /= cp.max(transform_tem)
                        amp += transform_tem * cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        spotnum += 1
                    else:
                        transform_tem = temnm_Offcenter(slm, n=0, m=0, w=image_waisttem, x0=x0ic, y0=0.5)
                        transform_tem /= cp.max(transform_tem)
                        amp += transform_tem * cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        spotnum += 1
            if tem and double_amps:
                double_spots = []
                separation = int(np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2+1)  # MDS added +1
                # print(separation)
                for spot in spots:
                    double_spots.append([spot[0], spot[1] - separation])
                    double_spots.append([spot[0], spot[1] + separation+1])# MDS added +1
                spots = double_spots

        # False starts here old method of defining in integer spot points
        if False:
            for i in range(len(spots)):
    
                # shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
                shift = (int(spots[i][0] - size[0] / 2), int(spots[i][1] - size[1] / 2))
                # print(shift)
                if double_amps:
                    half_amps = cp.ones(transform.shape, dtype=cp.complex128)
                    half_amps[:, :len(half_amps[0]) // 2] = amps[2 * i] * cp.exp(1j * phases[2 * i])
                    # print(2 * i)
                    # print(amps[2 * i])
                    # print(2 * i + 1)
                    # print(amps)
                    half_amps[:, len(half_amps[0]) // 2:] = amps[2 * i + 1] * cp.exp(1j * phases[2 * i])
    
                    # print(amps[2 * i + 1])
                    transform_shifted = transform * half_amps
    
                    amp += cp.roll(transform_shifted, shift, axis=(0, 1))# old
    
                else:
                    # print(type(shift))
                    # print(type(amp))
                    # print(type(transform))
                    amp += cp.array(np.roll(transform, (shift), axis=(0, 1))) * cp.array(amps[i] * np.exp(1j * phases[i]))
    
                # amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])
    
                # slm.ampToBMP(np.abs(amp), 'shifted_transform', True)
            # amp -= spot_array
            # amp = np.abs(amp)
            # amp /= np.max(np.abs(amp))
            # total_input = np.sum(np.abs(input_profile))
            if tem and double_amps:
                # print('tem and doubleamps')
                # print(transform.shape)
                # print(np.where(np.abs(transform) == np.max(np.abs(transform)))[1][0])
                # print(len(transform[0]) / 2)
                double_spots = []
                separation = int(np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2)  # MDS added +1
                # print(separation)
                for spot in spots:
                    double_spots.append([spot[0], spot[1] - separation])
                    double_spots.append([spot[0], spot[1] + separation])
                spots = double_spots
            # plt.figure(figsize=(8, 6))
            # plt.imshow(np.abs(amp.get()), cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            # plt.title('wu result')
            # plt.show()
            if False:  # For force tilt
                xytiltgrid = np.ones((1024, 11))
                for ii in range(0, 11, 1):
                    xytiltgrid[:, ii] = -5 + ii
                # plt.imshow(xytiltgrid)
                # plt.show()
                amp[:, 630:641] = amp[:, 630:641] * cp.array(np.exp(-1j * 0.166 * xytiltgrid))
            # amp=amp+amp1
        #False ends here old method of defining in integer spot points
        amp *= np.sqrt(np.sum(np.abs(input_profile) ** 2) / np.sum(np.abs(amp) ** 2))
        # print(spots)
        print("end",datetime.now())
        return [amp, spots]

    def target_output_array_bokai_Offcenter2Darb(n, m, input_profile, middlecompensate=[[0], [0], [[512, 636]]],
                                            center=np.array((0, 0)),
                                            x_pitch=0.1, y_pitch=0.1, size=np.array((1024, 1272)), amps=None,
                                            phases=None,
                                            global_phase=0, tem=False, double_amps=False, uni_spacing=True, xarblist0=None, yarblist0=None, anglearblist0=None):

        if amps is None:
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)


        if phases is None:
            phases = [global_phase for _ in range(n * m)]
        phases = np.array(phases)
        # print(amps)
        # input_profile = Profile.input_gaussian(beam_size=input_waist, pos=input_center, size=size, amp=1)
        # transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(input_profile), norm="ortho"))

        slm = SLM(size=size)

        input_field = cp.array(input_profile)
        ## Find natural waist at image plane to determine target waist
        inputbeam_tem = laserbeamsizefromimage(slm, cp.abs(input_field).get() ** 2)
        slm_waist_tem = np.array([inputbeam_tem[2], inputbeam_tem[3]])
        laserbeam_tem = laserbeamsizefromimage(slm, cp.abs(propagate(input_field)).get() ** 2)
        image_waist_tem = np.array([laserbeam_tem[2], laserbeam_tem[3]])

        if tem:
            target_slm_field, transform = calibrated_temnm_Offcenter2Darb(slm, input_profile, n=0, m=1, x0i=0.5, y0i=0.5,angle=90)
            transform = transform
            #target_slm_field45, transform45 = calibrated_temnm_Offcenter2Darb(slm, input_profile, n=0, m=1, x0i=0.5, y0i=0.5,angle=90)
            #transform45 = transform45
            #target_slm_field0, transform0 = calibrated_temnm_Offcenter2Darb(slm, input_profile, n=0, m=1, x0i=0.5, y0i=0.5,angle=90)
            #transform0 = transform0
            #target_slm_field25, transform25 = calibrated_temnm_Offcenter2Darb(slm, input_profile, n=0, m=1, x0i=0.5, y0i=0.5,angle=90)
            #transform25 = transform25
        else:
            #target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=0, x0i=0.5, y0i=0.5)
            #transform = transform
            target_slm_field, transform = calibrated_temnm_Offcenter2Darb(slm, input_profile, n=0, m=0, x0i=0.5, y0i=0.5,angle=90)
            transform = transform
        # plt.imshow(np.abs(transform.get()))
        # plt.title("transform abs")
        # plt.show()
        # if tem:
        #    slm = SLM(size=size)
        #    target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=1)
        #    transform = transform

        # if double_amps:
        #     separation = np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2

        spot_array, spots, spot_angles = Profile.spot_array(n, m, center, y_pitch, x_pitch, size, uni_spacing=uni_spacing, xarblist0=xarblist0, yarblist0=yarblist0, anglearblist0=anglearblist0)

        # plt.imshow(np.abs(spot_array))
        # plt.title("spotarray")
        # plt.show()
        amp = cp.zeros(transform.shape, dtype=np.complex128)
        print("spot_array", spot_array, "spots", spots)
        spotnum = 0
        print("start spot", datetime.now())
        laserbeam_waisttem = laserbeamsizefromimage(slm, cp.abs(propagate(input_profile)).get() ** 2)
        image_waisttem = np.array([laserbeam_waisttem[2], laserbeam_waisttem[3]])
        # starts here for uniform spacing, independent of spot_array
        if False:
            for i in np.linspace(-0.5 * n * x_pitch + center[0], 0.5 * n * x_pitch + center[0], n, endpoint=True):
                if n <= 1:
                    i = 0
                if m > 1:
                    for j in np.linspace(-0.5 * m * y_pitch + center[1], 0.5 * m * y_pitch + center[1], m,
                                         endpoint=True):
                        x0ic = (i + 0.5);
                        y0ic = (j + 0.5)
                        # target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=0, x0i=x0ic,
                        #                                                         y0i=y0ic)
                        if tem:
                            transform_tem = temnm_Offcenter(slm, n=0, m=1, w=image_waisttem, x0=x0ic, y0=y0ic)
                            transform_tem /= cp.max(transform_tem)
                            if double_amps:
                                half_amps = cp.ones(transform.shape, dtype=cp.complex128)
                                # half_amps[:, :len(half_amps[0]) // 2] = amps[2 * i] * cp.exp(1j * phases[2 * i])
                                # half_amps[:, len(half_amps[0]) // 2:] = amps[2 * i + 1] * cp.exp(1j * phases[2 * i])
                                half_amps[:, :int(y0ic * len(half_amps[1]))] = amps[2 * spotnum] * cp.exp(
                                    1j * phases[2 * spotnum])
                                half_amps[:, int(y0ic * len(half_amps[1])):] = amps[2 * spotnum + 1] * cp.exp(
                                    1j * phases[2 * spotnum + 1])
                                amp += transform_tem * cp.array(half_amps)  # [spotnum])

                            else:
                                amp += transform_tem * cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        else:
                            transform_tem = temnm_Offcenter(slm, n=0, m=0, w=image_waisttem, x0=x0ic, y0=y0ic)
                            transform_tem /= cp.max(transform_tem)
                            amp += transform_tem * cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        spotnum += 1
                else:
                    x0ic = (i + 0.5)
                    # target_slm_field, transform = calibrated_temnm_Offcenter(slm, input_profile, n=0, m=0, x0i=x0ic,
                    #                                                         y0i=0.5)
                    if tem:
                        transform_tem = temnm_Offcenter(slm, n=0, m=1, w=image_waisttem, x0=x0ic, y0=0.5)
                        transform_tem /= cp.max(transform_tem)
                        amp += transform_tem * cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        spotnum += 1
                    else:
                        transform_tem = temnm_Offcenter(slm, n=0, m=0, w=image_waisttem, x0=x0ic, y0=0.5)
                        transform_tem /= cp.max(transform_tem)
                        amp += transform_tem * cp.array(amps[spotnum] * np.exp(1j * phases[spotnum]))
                        spotnum += 1
            if tem and double_amps:
                double_spots = []
                separation = int(
                    np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2 + 1)  # MDS added +1
                # print(separation)
                for spot in spots:
                    double_spots.append([spot[0], spot[1] - separation])
                    double_spots.append([spot[0], spot[1] + separation + 1])  # MDS added +1
                spots = double_spots

        # False starts here old method of defining in integer spot points
        if True:
            for i in range(len(spots)):

                # shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
                shift = (int(spots[i][0] - size[0] / 2), int(spots[i][1] - size[1] / 2))
                # print(shift)
                if double_amps:
                    print("amps inside double_amps",amps)
                    half_amps = cp.ones(transform.shape, dtype=cp.complex128)
                    half_amps[:, :len(half_amps[0]) // 2] = amps[2 * i] * cp.exp(1j * phases[2 * i])
                    # print(2 * i)
                    # print(amps[2 * i])
                    # print(2 * i + 1)
                    # print(amps)
                    half_amps[:, len(half_amps[0]) // 2:] = amps[2 * i + 1] * cp.exp(1j * phases[2 * i])

                    # print(amps[2 * i + 1])
                    #Original
                    if False: #ORIGINAL Commented 09/12/2025
                        transform_shifted = transform * half_amps
                        amp += cp.roll(transform_shifted, shift, axis=(0, 1))  # old Commented 09/12/2025

                    if True: # Spot angle is defined
                        ior = cp.linspace(0, 1, slm.size[1])  # x-coordinates (normalized)
                        jor = cp.linspace(0, 1, slm.size[0])  # y-coordinates (normalized)

                        # Create 2D grid of coordinates
                        Ior, Jor = cp.meshgrid(ior, jor)

                        # Apply rotation to the coordinates (counterclockwise)
                        angle_rad = cp.deg2rad(spot_angles[i])  # Convert angle to radians
                        cos_angle = cp.cos(angle_rad)
                        sin_angle = cp.sin(angle_rad)

                        # Rotate the coordinates (I, J) by the given angle
                        x0or= spots[i][1]/size[1] ; y0or= spots[i][0]/size[0]
                        I_rot2 = cos_angle * (Ior - x0or) + sin_angle * (Jor - y0or) + x0or
                        J_rot2 = -sin_angle * (Ior - x0or) + cos_angle * (Jor - y0or) + y0or

                        half_amps = cp.ones(transform.shape, dtype=cp.complex128)
                        half_amps = cp.where((J_rot2>y0or), amps[2 * i] * cp.exp(1j * phases[2 * i]), amps[2 * i + 1] * cp.exp(1j * phases[2 * i]))


                    if True: # Creating each tem01 as rotated directly in the function
                        #target_slm_field, transform = calibrated_temnm_Offcenter2Darb(slm, input_profile, n=0, m=1,
                        #                                                              x0i=spots[i][1]/size[1], y0i=spots[i][0]/size[0], angle=spot_angles[i])
                        transform = temnm_Offcenter2Darb(slm, n=0, m=1, w=image_waist_tem,x0=spots[i][1]/size[1],y0=spots[i][0]/size[0],angle=spot_angles[i])
                        transform /= cp.max(transform)
                        transform_shifted = transform * half_amps
                        amp+=transform_shifted
                        #print("spot_angle_i",spot_angles[i])
                        #plt.imshow(cp.abs(half_amps).get())
                        #plt.title("halfamps")
                        #plt.show()
                        #plt.imshow(cp.abs(transform_shifted).get())
                        #plt.title("transform_shifted")
                        #plt.show()



                    if False: #For different fixed rotations
                        if i%4 ==0:
                            transform_shifted = transform * half_amps
                            amp += cp.roll(transform_shifted, shift, axis=(0, 1))  # Commented 09/12/2025
                        if i % 4 == 1:
                            transform_shifted = transform45 * half_amps
                            amp += cp.roll(transform_shifted, shift, axis=(0, 1))  # Commented 09/12/2025
                        if i%4 ==2:
                            transform_shifted = transform0 * half_amps
                            amp += cp.roll(transform_shifted, shift, axis=(0, 1))  # Commented 09/12/2025
                        if i%4 ==3:
                            transform_shifted = transform25 * half_amps
                            amp += cp.roll(transform_shifted, shift, axis=(0, 1))  # Commented 09/12/2025


                else:
                    # print(type(shift))
                    # print(type(amp))
                    # print(type(transform))
                    amp += cp.array(np.roll(transform, (shift), axis=(0, 1))) * cp.array(
                        amps[i] * np.exp(1j * phases[i]))

                # amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])

                # slm.ampToBMP(np.abs(amp), 'shifted_transform', True)
            # amp -= spot_array
            # amp = np.abs(amp)
            # amp /= np.max(np.abs(amp))
            # total_input = np.sum(np.abs(input_profile))
            if tem and double_amps:
                # print('tem and doubleamps')
                # print(transform.shape)
                # print(np.where(np.abs(transform) == np.max(np.abs(transform)))[1][0])
                # print(len(transform[0]) / 2)
                double_spots = []
                separation = int(np.where(transform == np.max(transform))[1][0] - len(transform[0]) / 2)  # MDS added +1
                separation = 2*1272*(2/(2*pi*0.55*1024*0.5))
                print(separation)
                if False: #Original: For TEM01 defining double positions for each lobe
                    for spot in spots:
                        double_spots.append([spot[0], spot[1] - separation])
                        double_spots.append([spot[0], spot[1] + separation])
                    spots = double_spots
                if True: #Modified: For TEM01 defining double positions for each lobe
                    for k in range(len(spots)):
                        angle_rad = cp.deg2rad(spot_angles[k])  # Convert angle to radians
                        cos_angle = cp.cos(angle_rad)
                        sin_angle = cp.sin(angle_rad)
                        # double_spots.append([spots[k][0]+cos_angle*3-1, spots[k][1] - sin_angle*3-1]) #-1 added   #old_fixed separation
                        # double_spots.append([spots[k][0]-cos_angle*3, spots[k][1] + sin_angle*3])
                        double_spots.append([spots[k][0]+cos_angle*separation, spots[k][1] - sin_angle*separation]) #new
                        double_spots.append([spots[k][0]-cos_angle*separation, spots[k][1] + sin_angle*separation])
                    spots = double_spots
                    print("spots after double amp",spots)
            # plt.figure(figsize=(8, 6))
            # plt.imshow(np.abs(amp.get()), cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            # plt.title('wu result')
            # plt.show()
            if False:  # For force tilt
                xytiltgrid = np.ones((1024, 11))
                for ii in range(0, 11, 1):
                    xytiltgrid[:, ii] = -5 + ii
                # plt.imshow(xytiltgrid)
                # plt.show()
                amp[:, 630:641] = amp[:, 630:641] * cp.array(np.exp(-1j * 0.166 * xytiltgrid))
            # amp=amp+amp1
        # False ends here old method of defining in integer spot points
        amp *= np.sqrt(np.sum(np.abs(input_profile) ** 2) / np.sum(np.abs(amp) ** 2))
        # print(spots)
        print("end", datetime.now())
        return [amp, spots]


if __name__ == '__main__':
    slm = SLM()
    input_size = np.array((0.05, 0.05))
    input_profile = Profile.input_gaussian(beam_size=input_size, pos=np.array((0, 0)), size=slm.size, amp=1)

    target_array = Profile.target_output_array(1, 2, input_profile=input_profile, tem=True, x_pitch=0.04, double_amps=True,
                                               amps=(1, 1, 1, 1), phases=(0, 0, 0.5 * 2 * pi, 0.5 * 2 * pi))
    # print(target_array[1])

    # slm.fieldtoBMP(input_profile, 'input_profile', wavelength=411, color=True, correction=False)
    slm.fieldtoBMP(target_array[0], 'target_array', wavelength=411, color=True, correction=False, show=True, sat=True, norm=True)
    # slm.phaseToBMP(np.angle(target_array[0]), 'target_array', color=True)

    # array_2d = Profile.spot_array(20, 20, x_pitch=0.005, y_pitch=0.005, size=np.array((512, 512)))[0]
    # slm.ampToBMP(np.abs(array_2d), 'spot_array_2d')
    # print(laserbeamsizefromimage(slm, np.abs(input_profile)**2))
    # print(laserbeamsizefromimage(slm, np.abs(propagate(cp.array(input_profile)).get())**2))

    # image = cp.array(slm.BMPToAmp(path=tkinter.filedialog.askopenfilename(title='Select Image'), norm=True))
    # stip = stipple(image, size=(100, 100))
    # amps = cp.ndarray.flatten(image)
    # amps *= amps > 0.5
    # target_array = Profile.target_output_array(100, 100, input_profile=input_profile, amps=amps.get(), tem=False, x_pitch=0.004, y_pitch=0.004)
    # slm.fieldtoBMP(target_array[0], 'target_array', wavelength=411, color=True, correction=False, show=True)

    # plot_gradient(target_array[0])

