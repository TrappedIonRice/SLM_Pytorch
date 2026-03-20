import matplotlib.colors
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import skimage.color
import scipy
import cv2

matplotlib.use('TkAgg')

pi = np.pi


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

# All coordinates normalized to [-1, 1]
# Class to represent an SLM
class SLM:

    def __init__(self, size=np.array((1024, 1272)), correction_path="images/413corrwithLUT.bmp", lut=None, pitch=12500,
                 wavelength=413):
        # LUT for number of phase increments on SLM pixels
        if lut is None:
            lut = {399: 93, 411: 102, 413: 103, 435:114}
        self.lut = lut

        self.size = np.array(size)
        self.pitch = pitch
        self.wavelength = wavelength

        # Correction pattern to be applied for the respective wavelength
        correction_image = im.open(correction_path)
        self.correction = np.array(correction_image) / lut[wavelength] * 2 * pi

        # self.correction_399 = np.array(correction_image) / lut[399] * 2 * pi

    # Convert normalized [-1, 1] coordinates to array indices
    def coord(self, r=np.array((0, 0))):
        shape = np.array([self.size[1], self.size[0]])
        r_px = np.array(shape * (r / 2 + 0.5), np.uint)
        return r_px

    def px_to_coords(self, r_px=np.array((0, 0))):
        shape = np.array([self.size[1], self.size[0]])
        r = np.array(2 * (r_px / shape - 0.5))
        return r

    #
    def shape_px(self, shape):
        shape_px = np.array([self.size[1], self.size[0]])
        return np.array(shape_px * shape / 2, np.uint)

    # def coords_to_realunits(self, coords=(0, 0), f=(np.inf, np.inf), wavelength=413e-9, eff_area=(12.8e-3, 15.9e-3)):
    #     r =

    # Return an otherwise flat phase pattern with a pi shift from one half to the other
    def half(self, center=0):
        center = np.array(self.size * (center / 2 + 0.5), np.uint)
        phase = np.zeros(self.size)
        phase[:, :center[1]] = pi
        return phase

    def half_smoothed(self, center=0, loc=636, scale=50):
        phase = self.half(center)
        # sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # return scipy.signal.convolve2d(phase, np.ones((50, 50)), mode='same')
        phase[:] = scipy.stats.logistic.cdf(np.arange(start=0, stop=self.size[1]), loc=loc, scale=scale)
        return phase * pi

    # Return a 2x2 checkerboard phase pattern (with pi shifts between squares)
    def quad(self, center=np.array((0, 0))):
        center = np.array(self.size * (center / 2 + 0.5), np.uint)
        phase = np.zeros(self.size)
        phase[:center[0], :center[1]] = pi
        phase[center[0]:, center[1]:] = pi
        return phase

    # Generate random phase pattern
    def random(self):
        return np.random.rand(self.size)

    # Add two phase patterns modulo 2pi
    def add(self, a, b):
        out = (a + b) % (2 * pi)
        return out

    def pixelate(self, field, factor=2):
        out = np.zeros(field.shape / factor)
        for i in range(len(out)):
            for j in range(len(out)):
                out[i][j] = np.average(field[i * factor:(i + 1) * factor][j * factor:(j + 1) * factor])
        return out

    # Convert grayscale phase pattern to a cyclic colormap for better visualization
    def cyclic_colormap(self, grayscale):
        # print(grayscale)
        hsv = np.append(np.expand_dims(grayscale, axis=2), np.ones((grayscale.shape[0], grayscale.shape[1], 2)), axis=2)
        # print(hsv)
        # rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return grayscale

    # Plot and save a phase pattern
    def phaseToBMP(self, phase, name='output', wavelength=413, correction=False, color=False, show=False, fig=None, location=None, figure=None, norm=False, units='', extent=None, colorbar=True):
        phase = (phase + 2 * pi) % (2 * pi)
        if correction:
            phase = self.add(phase, self.correction)

        if color:
            if units.__eq__(''):
                units = ' (px)'
            if norm:
                extent = (-1, 1, -1, 1)
                units = ''
            if figure is not None:
                phase[0, 0] = 0
                phase[-1, -1] = 2 * pi
                figure.axes.imshow(phase / 2 / pi, extent=extent, cmap='hsv')
                # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                figure.axes.set_xlabel('$x$' + units)
                figure.axes.set_ylabel('$y$' + units)
                figure.axes.set_title(name)
                # figure.axes.tight_layout()
                if colorbar:
                    figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes, label='$2\\pi$ radians')
            else:
                fig = plt.figure(fig, dpi=150, figsize=(5, 4))
                plt.clf()
                # fig = plt.figure(dpi=150)
                plt.imshow(phase / (2 * pi), cmap='hsv')
                plt.colorbar(label='$2\\pi$ radians')
                plt.xlabel('$x$' + units)
                plt.ylabel('$y$' + units)
                plt.title(name)
                fig.tight_layout()
                plt.pause(.001)
                if show:
                    plt.show()
                    # plt.draw()
                im.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_argb()).save('images/' + name + '_color.png')
                # return fig

        bmp_array = np.array(phase / (2 * pi) * self.lut[wavelength], dtype=np.uint8)
        if location is None:
            im.fromarray(bmp_array).save('images/' + name + '.bmp')
        else:
            im.fromarray(bmp_array).save(location)
    def phaseToBMP_correct(self, phase, name='output', wavelength=413, correction=False, color=False, show=False, fig=None, location=None, figure=None, norm=False, units='', extent=None, colorbar=True):
        phase = (phase + 2 * pi) % (2 * pi)
        if correction:
            phase = self.add(phase, self.correction)

        if color:
            if units.__eq__(''):
                units = ' (px)'
            if norm:
                extent = (-1, 1, -1, 1)
                units = ''
            if figure is not None:
                phase[0, 0] = 0
                phase[-1, -1] = 2 * pi
                figure.axes.imshow(phase / 2 / pi, extent=extent, cmap='hsv')
                # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                figure.axes.set_xlabel('$x$' + units)
                figure.axes.set_ylabel('$y$' + units)
                figure.axes.set_title(name)
                # figure.axes.tight_layout()
                if colorbar:
                    figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes, label='$2\\pi$ radians')
            else:
                fig = plt.figure(fig, dpi=150, figsize=(5, 4))
                plt.clf()
                # fig = plt.figure(dpi=150)
                plt.imshow(phase / (2 * pi), cmap='hsv')
                plt.colorbar(label='$2\\pi$ radians')
                plt.xlabel('$x$' + units)
                plt.ylabel('$y$' + units)
                plt.title(name)
                fig.tight_layout()
                plt.pause(.001)
                if show:
                    plt.show()
                    # plt.draw()
                im.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_argb()).save('images/' + name + '_color.png')
                # return fig

        bmp_array = np.array(phase / (2 * pi) * self.lut[wavelength], dtype=np.uint8)
        if location is None:
            im.fromarray(bmp_array).save('correction1/' + name + '.bmp')
        else:
            im.fromarray(bmp_array).save(location)

    # Plot and save an amplitude pattern
    def ampToBMP(self, amp, name='output', color=False, show=False, fig=None, figure=None, extent=None, norm=False, units=''):
        bmp_array = np.array(amp * 255, dtype=np.uint8)
        im.fromarray(bmp_array).save('images/' + name + '.bmp')

        if color:
            amp[0, 0] = 0
            if units.__eq__(''):
                units = ' (px)'
            if norm:
                extent = (-1, 1, -1, 1)
                units = ''
            if figure is not None:
                figure.axes.imshow(amp, extent=extent)
                # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                figure.axes.set_xlabel('$x$' + units)
                figure.axes.set_ylabel('$y$' + units)
                figure.axes.set_title(name)
                # figure.axes.tight_layout()
            else:
                fig = plt.figure(fig, dpi=240, figsize=(5, 4))
                plt.clf()
                # fig = plt.figure(dpi=240)
                plt.imshow(amp)
                plt.colorbar()
                plt.xlabel('$x$' + units)
                plt.ylabel('$y$' + units)
                plt.title(name)
                fig.tight_layout()
                plt.pause(.001)
                if show:
                    plt.show()
                    # plt.draw()
                im.frombytes('RGB', fig.canvas.get_width_height(),
                             fig.canvas.tostring_argb()).save('images/' + name + '_color.png')
            return fig

    # Plot and save a light field pattern: in HSV encoding, phase is encoded in H and intensity is encoded in V
    def fieldtoBMP(self, field, name='output', wavelength=413, correction=False, color=False, show=False, fig=None, figure=None, norm=False, sat=False, extent=None, units='', colorbar=True):
        field = field / np.max(np.abs(field))
        bmp_array = np.array([((np.angle(field) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi), np.abs(field)**2 if sat else np.ones(field.shape), np.ones(field.shape) if sat else np.abs(field)**2])
        bmp_array = np.swapaxes(bmp_array, 0, 2)
        bmp_array = np.swapaxes(bmp_array, 0, 1)
        bmp_array = np.array(skimage.color.convert_colorspace(bmp_array, 'HSV', 'RGB', channel_axis=-1) * 255, dtype=np.uint8)
        image = im.fromarray(bmp_array, mode='RGB')
        image.save('images/' + name + '.bmp')

        phase = np.angle(field)
        amp = np.abs(field)

        phase = (phase + 2 * pi) % (2 * pi)
        if correction:
            phase = self.add(phase, self.correction)
        phase_bmp_array = np.array(phase / (2 * pi) * self.lut[wavelength], dtype=np.uint8)
        im.fromarray(phase_bmp_array).save('images/' + name + '_phase.bmp')

        amp_bmp_array = np.array(((amp / np.max(amp)) ** 2) * 255, dtype=np.uint8)
        im.fromarray(amp_bmp_array).save('images/' + name + '_amp.bmp')

        if color:
            field[0, 0] = 0
            if units.__eq__(''):
                units = ' (px)'
            if norm:
                extent = (-1, 1, -1, 1)
                units = ''
            if figure is not None:
                figure.axes.imshow(image, extent=extent)
                # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                figure.axes.set_xlabel('$x$' + units)
                figure.axes.set_ylabel('$y$' + units)
                figure.axes.set_title(name)
                if colorbar:
                    figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes, label='$2\\pi$ radians')
                # figure.axes.tight_layout()
            else:
                fig = plt.figure(fig, dpi=150, figsize=(6, 5))
                field[0, 0] = 0
                plt.imshow(image, extent=extent)
                plt.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                plt.xlabel('$x$' + units)
                plt.ylabel('$y$' + units)
                plt.title(name)
                fig.tight_layout()
                plt.pause(.001)
                if show:
                    plt.show()
                    # plt.draw()
                im.frombytes('RGB', fig.canvas.get_width_height(),
                                 fig.canvas.tostring_argb()).save('images/' + name + '_color.png')
                return fig

    # Import a phase pattern from an image
    def BMPToPhase(self, path, wavelength=413):
        return np.array(im.open(path)) / self.lut[wavelength] * 2 * pi

    # Import an amplitude pattern from an image
    def BMPToAmp(self, path, norm=True):
        out = np.array(im.open(path), dtype=np.double)
        if len(out.shape) > 2:
            out = np.mean(out, axis=2)
            print(out.shape)
        if norm:
            return out / np.max(out)
        else:
            return out


# Rescale a light field profile by an integer (or inverse integer), either averaging pixels to decrease size or binning to increase size
def rescale(phase, res_factor=2):
    output = np.zeros(shape=np.array(np.array(phase.shape) * res_factor, dtype=np.uint))
    if res_factor > 1:
        for i in range(len(phase)):
            for j in range(len(phase[i])):
                output[i * res_factor:(i + 1) * res_factor, j * res_factor:(j + 1) * res_factor] = phase[i, j]
    elif res_factor < 1:
        res_factor = 1 / res_factor
        for i in range(len(output)):
            for j in range(len(output[i])):
                output[i, j] = np.mean(phase[int(i * res_factor):int((i + 1) * res_factor), int(j * res_factor):int((j + 1) * res_factor)])
    else:
        output = phase
    return output


# Pad the border of a light field profile with zeros
def pad_border(phase, target_shape):
    output = np.zeros(shape=target_shape)
    target_center = np.array(target_shape) / 2
    phase_center = np.array(phase.shape) / 2
    if output.shape[1] > phase.shape[1] or output.shape[0] > phase.shape[0]:
        output[:phase.shape[0], :phase.shape[1]] = phase
        output = np.roll(output, np.array(target_center - phase_center, dtype=np.int64), axis=(0, 1))
    else:
        phase = np.roll(phase, np.array(target_center - phase_center, dtype=np.int64), axis=(0, 1))
        output[:, :] = phase[:output.shape[0], :output.shape[1]]
    
    return output


if __name__ == '__main__':
    slm = SLM()

    intensity = slm.BMPToAmp(
        r"Z:\Lab Rice\Experimental Projects\SLM\camera images\1x4 tem00 array intensity.bmp", norm=False)
    ref = slm.BMPToAmp(
        r"Z:\Lab Rice\Experimental Projects\SLM\camera images\temp-05272024115536-0.Bmp",
        norm=False)
    x_proj = slm.BMPToAmp(
        r"Z:\Lab Rice\Experimental Projects\SLM\camera images\temp-05272024115537-1.Bmp", norm=False)
    y_proj = slm.BMPToAmp(
        r"Z:\Lab Rice\Experimental Projects\SLM\camera images\temp-05272024115537-2.Bmp", norm=False)

    pk = np.where(intensity == np.max(intensity))

    sigma = 1.5
    x_proj = cv2.GaussianBlur(x_proj, (0, 0), sigmaX=sigma, sigmaY=sigma)
    y_proj = cv2.GaussianBlur(y_proj, (0, 0), sigmaX=sigma, sigmaY=sigma)
    intensity = cv2.GaussianBlur(intensity, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ref = cv2.GaussianBlur(ref, (0, 0), sigmaX=sigma, sigmaY=sigma)

    cosphi_x = (x_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
    cosphi_y = (y_proj - intensity - ref) / (2 * np.sqrt(intensity * ref))
    cosphi_x = np.clip(cosphi_x, -1, 1)
    cosphi_y = np.clip(cosphi_y, -1, 1)
    phi_x = (np.arccos(cosphi_x) + 2 * np.pi) % (2 * np.pi)
    phi_y = (np.arccos(cosphi_y) + 2 * np.pi) % (2 * np.pi)

    phasex = np.array([phi_x, 2 * np.pi - phi_x])
    phasey = np.array([phi_y, 2 * np.pi - phi_y])

    phasey = ((phasey - np.pi / 2) + 2 * np.pi) % (2 * np.pi)

    # min = [[0, 0], 2 * np.pi]
    # min = np.ones(phasex.shape) * 2 * np.pi
    min = np.ones(phasex[0].shape) * 2 * np.pi
    min_loc = np.zeros((phasex[0].shape[0], phasex[0].shape[1], 2))
    for i in range(len(phasex)):
        for j in range(len(phasey)):
            val = np.abs(phasex[i] - phasey[j])
            print(val.shape)
            min_loc[val < min] = np.array([i, j])
            # min_loc = np.where(val < min, np.array([i, j]), min_loc)
            min = np.where(val < min, val, min)
            # min[val < min] = val
    print(np.average(min))
    phase = np.zeros(min.shape)
    for i in range(len(min)):
        for j in range(len(min[i])):
            # print(min_loc[i, j, 0])
            phase[i, j] = (phasex[int(min_loc[i, j, 0]), i, j] * 2) / 2

    field = np.sqrt(intensity) * np.exp(1j * phase)
    # print(np.max(np.abs(field)))
    ref_dat = ref[pk[0], :][0]
    intensity_dat = intensity[pk[0], :][0]
    x_proj_dat = x_proj[pk[0], :][0]
    y_proj_dat = y_proj[pk[0], :][0]
    phase_dat = phase[pk[0], :][0]
    phase_dat = np.where(intensity_dat > 0.1 * np.max(intensity), phase_dat, 0)
    #print(intensity_dat)
    ax = plt.figure().add_subplot(111)
    twinax = plt.twinx()
    ax.plot([i for i in range(len(ref_dat))], ref_dat, label='Reference')
    ax.plot([i for i in range(len(intensity_dat))], intensity_dat, label='Intensities')
    ax.plot([i for i in range(len(x_proj_dat))], x_proj_dat, label='Interference')
    # ax.plot([i for i in range(len(y_proj_dat))], y_proj_dat, label='Interference 2')
    twinax.plot([i for i in range(len(phase_dat))], phase_dat / 2 / pi, label='Phase', c='r')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Intensity')
    ax.set_title('1x4  Phase Beam Array')
    twinax.set_ylabel('Phase ($2\\pi$ radians)')
    twinax.set_ylim(0, 1)
    ax.legend()
    twinax.legend()
    # plt.show()
    slm.fieldtoBMP(field, '1x4 Beam Array', color=True, wavelength=411, show=False, extent=(0, (field.shape[1] - 1) / 1.85, 0, (field.shape[0] - 1) / 1.85), units=' ($\mu$m)')
    slm.phaseToBMP(phase, '1x4 Beam Array', color=True, wavelength=411, show=False)

    # plt.clf()
    # plt.plot([i for i in range(len(intensity_dat))], intensity_dat / 50, label='Intensities')
    # plt.plot([i for i in range(len(phase_dat))], phase_dat, label='Phase')
    # plt.xlabel('X (px)')
    # plt.title('1x4  Phase Beam Array')
    # plt.legend()

    # slm.phaseToBMP(phase, 'Phase', wavelength=411, color=True, show=False)
    #
    # slm.ampToBMP(intensity, '1x4 Beam Array', color=True, show=False)
    # slm.ampToBMP(interference, '1x4 Beam Array Interference', color=True, show=True)
    plt.show()

    # slm.ampToBMP(scipy.signal.convolve2d(intensity, np.ones((50, 50))), '1x4  Beam Array smoothed', color=True, show=True)

