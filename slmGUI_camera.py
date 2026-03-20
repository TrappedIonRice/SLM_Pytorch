import sys
import time
import tkinter.filedialog
from Filrcamera import  FlirCamSelector
import FlirCamController_slm
# print(sys.path)
# sys.path.append('C:/Users/RiceT/Documents/SLM_computation/Filrcamera')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from contextlib import redirect_stdout
import socket
from pathlib import  Path
import matplotlib
import random
import math
import qdarkstyle
import cv2
from qtGUI_camera import Ui_MainWindow
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from ArrayModulator import ArrayModulator, load_mod
from IFTA import plot_gradient
import profile
from skimage.transform import resize
import numpy as np
from PIL import Image as im
import  datetime
cupy_working = False
from scipy.optimize import minimize
try:
    import cupy as cp
    cupy_working = True
except ImportError:
    cp = np
    print("cupy not installed. Using numpy.")
import slm as slm
import scipy
from PhaseExtraction import phasemap_2d, plot_phasemap, import_profs
from scipy.optimize import curve_fit
import subprocess
import matplotlib.gridspec as gridspec
import time
from PIL import Image
import numpy as np
from ctypes import *
import copy

from skimage.restoration import unwrap_phase
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score
def func(x, a, b):#for linear function fit
    return a * (x - b)
def gaussian(x, amp, mean, stddev):#gausssian fit
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
def phase_rotate(phase,angle)\
:#use it when the slm is tilt and we see the beam array tilt in camera

    centre=[int(len(phase)/2),int(len(phase[0])/2)]
    print('centre',centre)
    new_phase=np.zeros( (1024,1272))
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
            if x>0 and y>0 and x<new_phase.shape[0]-1 and y <new_phase.shape[1]-1:
                new_phase[i][j]=new_phase[i][j]+\
                    phase[math.floor(x)][math.floor(y)]*(x-math.floor(x))*(y-math.floor(y))
                new_phase[i][j] = new_phase[i][j] + \
                     phase[math.floor(x)][math.ceil(y)] * (x - math.floor(x)) * (math.ceil(y)-y)
                new_phase[i][j] = new_phase[i][j] + \
                                  phase[math.ceil(x)][math.ceil(y)] *  (math.ceil(x)-x) *  (math.ceil(y)-y)
                new_phase[i][j] = new_phase[i][j] + \
                                  phase[math.ceil(x)][math.floor(y)] * (math.ceil(x)-x) *  (y-math.floor(y))
    return new_phase
def findfilepath(shift):
#use it to localize the address of photos taken in doing SLM optical abberation correction with scan(self)
    image_paths = []
    name1 = f'image_{shift[0]}_{shift[1]}_{0}.bmp'
    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
    image_paths.append(os.path.join(directory, name1))
    name2 = f'image_{shift[0]}_{shift[1]}_{120}.bmp'
    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
    image_paths.append(os.path.join(directory, name2))
    name3 = f'image_{shift[0]}_{shift[1]}_{240}.bmp'
    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
    image_paths.append(os.path.join(directory, name3))
    return image_paths
def findfilepath2(shift):
    # use it to localize the address of photos taken in measuring referrence veam with scan(self)
    image_paths = []
    name1 = f'image_{shift[0]}_{shift[1]}_{0}.bmp'
    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
    image_paths.append(os.path.join(directory, name1))
    name2 = f'image_{shift[0]}_{shift[1]}_{120}.bmp'
    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
    image_paths.append(os.path.join(directory, name2))
    name3 = f'image_{shift[0]}_{shift[1]}_{240}.bmp'
    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
    image_paths.append(os.path.join(directory, name3))
    return image_paths
def camera_success(shift):
# use it  in doing SLM optical abberation correction with scan(self)
#to tell if the linear fit is good enough that we can use this data
    path = findfilepath(shift)
    data1 = (np.array(im.open(path[0]), dtype=np.float64))
    data2 = (np.array(im.open(path[1]), dtype=np.float64))
    data3 = (np.array(im.open(path[2]), dtype=np.float64))
    p_real = data1 + data1 - data2 - data3
    p_image = data2 - data3
    p = p_real + 1j * p_image * np.sqrt(3)
    phase_ = np.angle(p)[1200:2200, 1500:2500]
    phase_unwrap = unwrap_phase(phase_)
    #doing linear fit
    x = np.arange(phase_unwrap.shape[0])
    y = np.arange(phase_unwrap.shape[1])
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    phase_flat = phase_unwrap.flatten()
    features = np.vstack((X_flat, Y_flat)).T
    model = LinearRegression()
    model.fit(features, phase_flat)
    phase_predict = model.predict((features))
    r_square = r2_score(phase_unwrap.flatten(), phase_predict)
    print('r_square=',r_square)
    c = model.intercept_
    print('phase=', c)
    # error=phase_unwrap.flatten()-phase_predict
    # error=error.reshape(X.shape)
    # phase_predict=phase_predict.reshape(X.shape)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(phase_predict, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title('error after')
    # plt.show()
    # plt.figure(figsize=(8, 6))
    # plt.imshow(error, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title('error after')
    # plt.show()
    # print(np.sqrt(mean_squared_error(phase_unwrap.flatten(),phase_predict)))
    return r_square
def search_x(image_amp,num_x,tem,x_range):
#doing slices when analyse phase extraction of a beam array in x axis , num_x is how many beams in x axis
    #find the max value of intensity
    points = []
    index = np.argmax(image_amp)
    cooridnates = np.unravel_index(index, image_amp.shape)
    max = image_amp[cooridnates]
    print("index",index, "maxcoor",max,"ccord",cooridnates)

    # to see how many points are bright in x axis
    for j in range(image_amp.shape[1] - 10):
        if image_amp[cooridnates[0]][j + 5] > max * 0.5 and int(x_range[0])<j + 5 and int(x_range[1])>j + 5:
            if  1:# image_amp[cooridnates[0]][j + 5] >image_amp[cooridnates[0]][j + 4] :

                points.append(j)

    #patch_area=200 #original MDS FOR HIGH MAGNIFICATION
    patch_area = 50
    if tem!=0:
        #patch_area = 100#original MDS FOR HIGH MAGNIFICATION
        patch_area = 10
    area = np.zeros(int(4000/patch_area))
    inten= np.zeros(int(4000/patch_area))
    #divide the 4000 into many patches, for the size 100, then 50 patches
    #to see how many bright points are there in each patches

    for i in points:
        area[math.floor(i / patch_area)] = area[math.floor(i / patch_area)] + 1
        inten[math.floor(i / patch_area)] = np.max(image_amp[:,i-50:i+50])
    # if the brightest value is not bright enough, then treat it as dark
    for i in range(int(4000 / patch_area)):
        if  inten[i] < max * 0.5:
            area[i]=0
    print(area)
    # how many bright points
    n = np.max(area)
    num = 0


    print(area)
    #find beam number patches(for example 4 patches)
    while num < num_x:
        n = n - 1
        num = 0
        points = []
        for i in range(int(4000/patch_area)):
            if area[i] > n :
                num = num + 1
                points.append(i)
    #in each patch search the brightest point, and find its radius
    centre = []
    print('point=', points)
    area = np.zeros((num_x, 4))
    for k in range(num_x):
        slice = image_amp[:, patch_area * points[k]:patch_area * (points[k] + 1)]
        index = np.argmax(slice)
        cooridnates = np.unravel_index(index, slice.shape)
        max = slice[cooridnates]

        y_min = 0
        y_max = slice.shape[0]
        for i in range(slice.shape[0]):
            if slice[i][cooridnates[1]] < max / 3 and i > y_min and i < cooridnates[0]:
                y_min = i
            if slice[i][cooridnates[1]] < max / 3 and i < y_max and i > cooridnates[0]:
                y_max = i

        area[k][0] = y_min
        area[k][1] = y_max
        centre.append([cooridnates[0], patch_area * points[k] + cooridnates[1]])
    print(centre)
    #divide the area again according to the position of these brightest point, which will be more accurate
    for k in range(num_x):
        if k == 0:
            area[k][2] = int(x_range[0])
        else:
            x = centre[k]
            xx = centre[k - 1]
            area[k][2] = int((x[1] + 2*xx[1]) / 3)
        if k == num_x - 1:
            area[k][3] =int(x_range[1])
        else:
            x = centre[k]
            xx = centre[k + 1]
            area[k][3] = int((2*x[1] + xx[1]) / 3)

    if area[0][2] < 0:
        area[0][2] = int(x_range[0])
    if area[num_x - 1][3] > 3999:
        area[num_x - 1][3] = int(x_range[1])


    print(area)
    return area#return the slice I divide
def search_y(image_amp,num_y):
# same as the x, but in a old version I have no update it
    points = []
    index = np.argmax(image_amp)
    cooridnates = np.unravel_index(index, image_amp.shape)
    max = image_amp[cooridnates]
    for j in range(image_amp.shape[0] - 10):
        if image_amp[j + 5][cooridnates[1]] > max * 0.5 :
                points.append(j)
    area = np.zeros(15)
    inten= np.zeros(15)
    #patch_area=200 #ORIGINAL MDS
    patch_area=200


    for i in points:
        area[math.floor(i / patch_area)] = area[math.floor(i / patch_area)] + 1
        inten[math.floor(i / patch_area)] = np.max(image_amp[i - 50:i + 50,:])
    # if the brightest value is not bright enough, then treat it as dark
    for i in range(int(3000 / patch_area)):
        if inten[i] < max * 0.5:
            area[i] = 0

    n = np.max(area)
    num = 0
    # print('point_1=', points)
    print(area)
    while num < num_y:
        n = n - 1
        num = 0
        points = []
        for i in range(15):
            if area[i] > n:
                num = num + 1
                points.append(i)
    centre = []
    print('point_2=', points)
    area = np.zeros((num_y, 4))
    for k in range(num_y):
        slice = image_amp[ 200 * points[k]:200 * (points[k] + 1),:]
        index = np.argmax(slice)
        cooridnates = np.unravel_index(index, slice.shape)
        max = slice[cooridnates]

        x_min = 0
        x_max = slice.shape[1]
        for i in range(slice.shape[1]):
            if slice[cooridnates[0]][i] < max / 3 and i > x_min and i < cooridnates[1]:
                x_min = i
            if slice[cooridnates[0]][i] < max / 3 and i < x_max and i > cooridnates[1]:
                x_max = i

        area[k][0] = x_min
        area[k][1] = x_max
        centre.append([ 200 * points[k] + cooridnates[0],cooridnates[1]])
    print(centre)
    for k in range(num_y):
        if k == 0:
            area[k][2] = 0
        else:
            x = centre[k]
            xx = centre[k - 1]
            area[k][2] = int((x[0] + 2*xx[0]) / 3)
        if k == num_y - 1:
            area[k][3] = 2999
        else:
            x = centre[k]
            xx = centre[k + 1]
            area[k][3] = int((2*x[0] + xx[0]) / 3)

    if area[0][2] < 0:
        area[0][2] = 0
    if area[num_y - 1][3] > 2999:
        area[num_y - 1][3] = 2999


    print(area)
    return area
class MplCanvas(FigureCanvas):
    #for polting graph
    def __init__(self, parent=None, width=5, height=4, dpi=75, y_scale="linear", dark=True):
        if dark:
            with plt.style.context("dark_background"):
                fig = Figure(figsize=(width, height), dpi=dpi)
                self.axes = fig.add_subplot(111, yscale=y_scale)
                fig.tight_layout()
                self.fig = fig
        else:
            fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = fig.add_subplot(111, yscale=y_scale)
            # fig.tight_layout()
            self.fig = fig
        super(MplCanvas, self).__init__(fig)


class Stream(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class slmGUI(Ui_MainWindow):
    def __init__(self, mainwindow):
        self.Lcoslib = windll.LoadLibrary(
            "Z:\Lab Rice\Experimental Projects\SLM\LCOS-SLM_Control_software_Sample_source_code\Python_sample_code\python_sample_code_64bit\Image_Control.dll")

        self.set_slm_low_noise()

        self.mainwindow = mainwindow
        self.setupUi(mainwindow)
        #MDS
        self.a_fit= 0# -3.733022642775838e-06
        self.b_fit = 1776
        self.c_fit = 0# -3.893853882563125e-06
        self.d_fit = 1790

        self.phase_extracted=[]
        self.amp_extracted=[]
        #MDS
        self.connect= False
        self.auto_save = False
        self.expotime = self.text_expotime.text()


        self.slm_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_slm = NavigationToolbar(self.slm_plot, mainwindow)
        self.slm_layout.addWidget(toolbar_slm)
        self.slm_layout.addWidget(self.slm_plot)

        self.image_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_image = NavigationToolbar(self.image_plot, mainwindow)
        self.image_layout.addWidget(toolbar_image)
        self.image_layout.addWidget(self.image_plot)

        self.inner_plot = MplCanvas(self, height=7, dpi=80, dark=False)
        toolbar_inner = NavigationToolbar(self.inner_plot, mainwindow)
        self.inner_layout.addWidget(toolbar_inner)
        self.inner_layout.addWidget(self.inner_plot)

        self.outer_plot = MplCanvas(self, height=7, dpi=80, dark=False)
        toolbar_outer = NavigationToolbar(self.outer_plot, mainwindow)
        self.outer_layout.addWidget(toolbar_outer)
        self.outer_layout.addWidget(self.outer_plot)

        self.eprof_plot = MplCanvas(self, height=7, dpi=80, dark=False)
        toolbar_eprof = NavigationToolbar(self.eprof_plot, mainwindow)
        self.eprof_layout.addWidget(toolbar_eprof)
        self.eprof_layout.addWidget(self.eprof_plot)

        self.phase_field_plot = MplCanvas(self, dpi=60, dark=False)
        toolbar_phase_field = NavigationToolbar(self.phase_field_plot, mainwindow)
        self.phase_field_layout.addWidget(toolbar_phase_field)
        self.phase_field_layout.addWidget(self.phase_field_plot)

        self.phase_phase_plot = MplCanvas(self, dpi=60, dark=False)
        toolbar_phase_phase = NavigationToolbar(self.phase_phase_plot, mainwindow)
        self.phase_phase_layout.addWidget(toolbar_phase_phase)
        self.phase_phase_layout.addWidget(self.phase_phase_plot)

        self.phase_prof_plot = MplCanvas(self, height=9, dpi=60, dark=False)
        self.toolbar_phase_prof = NavigationToolbar(self.phase_prof_plot, mainwindow)
        self.phase_prof_layout.addWidget(self.toolbar_phase_prof)
        self.phase_prof_layout.addWidget(self.phase_prof_plot)

        self.intensity_plot = MplCanvas(self, height=8, dpi=60, dark=False)
        toolbar_intensity = NavigationToolbar(self.intensity_plot, mainwindow)
        self.intensity_Vlayout = QtWidgets.QVBoxLayout()
        self.intensity_Vlayout.addWidget(toolbar_intensity)
        self.intensity_Vlayout.addWidget(self.intensity_plot)
        self.intensity_layout.addLayout(self.intensity_Vlayout)

        self.reference_plot = MplCanvas(self, height=8, dpi=60, dark=False)
        toolbar_reference = NavigationToolbar(self.reference_plot, mainwindow)
        self.reference_Vlayout = QtWidgets.QVBoxLayout()
        self.reference_Vlayout.addWidget(toolbar_reference)
        self.reference_Vlayout.addWidget(self.reference_plot)
        self.reference_layout.addLayout(self.reference_Vlayout)

        self.interference1_plot = MplCanvas(self, height=8, dpi=60, dark=False)
        toolbar_interference1 = NavigationToolbar(self.interference1_plot, mainwindow)
        self.interference1_Vlayout = QtWidgets.QVBoxLayout()
        self.interference1_Vlayout.addWidget(toolbar_interference1)
        self.interference1_Vlayout.addWidget(self.interference1_plot)
        self.interference1_layout.addLayout(self.interference1_Vlayout)

        self.interference2_plot = MplCanvas(self, height=8, dpi=60, dark=False)
        toolbar_interference2 = NavigationToolbar(self.interference2_plot, mainwindow)
        self.interference2_Vlayout = QtWidgets.QVBoxLayout()
        self.interference2_Vlayout.addWidget(toolbar_interference2)
        self.interference2_Vlayout.addWidget(self.interference2_plot)
        self.interference2_layout.addLayout(self.interference2_Vlayout)

        self.cameraphase_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_cameraphase = NavigationToolbar(self.cameraphase_plot, mainwindow)
        self.camera_layout.addWidget(toolbar_cameraphase)
        self.camera_layout.addWidget(self.cameraphase_plot)

        self.image_camera_plot_1 = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_image_camera_1 = NavigationToolbar(self.image_camera_plot_1, mainwindow)
        self.camera_layout.addWidget(toolbar_image_camera_1)
        self.camera_layout.addWidget(self.image_camera_plot_1)

        self.image_camera_plot_2 = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_image_camera_2 = NavigationToolbar(self.image_camera_plot_2, mainwindow)
        self.image_layout_camera.addWidget(toolbar_image_camera_2)
        self.image_layout_camera.addWidget(self.image_camera_plot_2)

        self.image_camera_plot_3 = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_image_camera_3 = NavigationToolbar(self.image_camera_plot_3, mainwindow)
        self.image_layout_camera.addWidget(toolbar_image_camera_3)
        self.image_layout_camera.addWidget(self.image_camera_plot_3)

        self.phase_plot2 = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_phase_plot2 = NavigationToolbar(self.phase_plot2, mainwindow)
        self.amps_layout.addWidget(toolbar_phase_plot2)
        self.amps_layout.addWidget(self.phase_plot2)

        self.phase_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_phase_plot = NavigationToolbar(self.phase_plot, mainwindow)
        self.phase_layout.addWidget(toolbar_phase_plot)
        self.phase_layout.addWidget(self.phase_plot)



        self.amplitude_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_amplitude_plot = NavigationToolbar(self.amplitude_plot, mainwindow)
        self.beam_amps_layout.addWidget(toolbar_amplitude_plot)
        self.beam_amps_layout.addWidget(self.amplitude_plot)

        self.tranverse_field_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_tranverse_field = NavigationToolbar(self.tranverse_field_plot, mainwindow)
        self.beam_field_layout.addWidget(toolbar_tranverse_field)
        self.beam_field_layout.addWidget(self.tranverse_field_plot)

        self.phase_1d_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_phase_1d = NavigationToolbar(self.phase_1d_plot, mainwindow)
        self.phase_measured_layout.addWidget(toolbar_phase_1d)
        self.phase_measured_layout.addWidget(self.phase_1d_plot)

        self.column_plot = MplCanvas(self, dark=False, height=2, width=3, dpi=100)
        toolbar_intensity_reference_1d = NavigationToolbar(self.column_plot, mainwindow)
        self.intensity_reference_layout.addWidget(toolbar_intensity_reference_1d)
        self.intensity_reference_layout.addWidget(self.column_plot)

        # self.verticalLayout_10.removeWidget(self.textBrowser)

        self.pushButton_alg_compute.clicked.connect(self.compute_alg)
        self.pushButton_open.clicked.connect(self.open_file)
        self.pushButton_save.clicked.connect(self.save_file)
        self.pushbutton_sim_apply.clicked.connect(self.apply_input)
        self.pushbutton_sim_compute.clicked.connect(self.propagate)
        self.pushButton_alg_apply.clicked.connect(self.apply_alg)
        self.pushButton_2dfile.clicked.connect(self.open_2d)
        self.pushButton_intensity.clicked.connect(self.open_intensity)
        self.pushButton_reference.clicked.connect(self.open_reference)
        self.pushButton_interference1.clicked.connect(self.open_interference1)
        self.pushButton_interference2.clicked.connect(self.open_interference2)
        self.pushButton_extractphase.clicked.connect(self.phase_extraction)
        self.pushButton_aberrationcorrection.clicked.connect(self.aberration_correction)

        self.pushbutton_camera_image.clicked.connect(lambda :self.take_image_show(0))# button 'take image'
        self.pushbutton_circle.clicked.connect(self.circle)# button 'start'
        self.pushButton_setexpo.clicked.connect(self.setexpo)# button 'set exposure'
        self.pushbutton_cut_off_camera.clicked.connect(self.disconnect)# button 'disconnect'
        self.pushbutton_save_data.clicked.connect(self.save_data)  # button 'disconnect'
        # button 'open phase ' 'apply phase'
        self.pushButton_openphase_1.clicked.connect(lambda :self.open_file_camera(1))
        self.pushButton_applyphase_1.clicked.connect(lambda :self.apply_phase(1))

        self.pushButton_openphase_2.clicked.connect(lambda: self.open_file_camera(2))
        self.pushButton_applyphase_2.clicked.connect(lambda: self.apply_phase(2))

        self.pushButton_openphase_3.clicked.connect(lambda: self.open_file_camera(3))
        self.pushButton_applyphase_3.clicked.connect(lambda: self.apply_phase(3))

        self.pushbutton_phase_generation.clicked.connect(self.phase_generation)
        self.pushbutton_scan.clicked.connect(self.scan)
        self.pushbutton_data_process.clicked.connect(self.dataprocess)
        self.pushbutton_patch_generate.clicked.connect(self.patch_generate)
        self.pushbutton_compensate.clicked.connect(self.compensate)# button 'analyse data'
        self.pushbutton_reset_para.clicked.connect(self.reset_para)# button 'reset'
        self.pushbutton_reference_scan.clicked.connect(self.reference_scan)
        self.pushbutton_reference_scan_analyse.clicked.connect(self.reference_scan_analyse)
        self.mark_for_slice=0
        self.mark_for_compen = 0

        # self.lineEdit_5.setText('411')
        self.textBrowser.moveCursor(QtGui.QTextCursor.Start)
        self.textBrowser.ensureCursorVisible()
        self.textBrowser.setLineWrapColumnOrWidth(500)
        # self.textBrowser.setLineWrapMode(QtGui.QTextEdit.FixedPixelWidth)

        # sys.stdout = Stream(newText=self.onUpdateText)

        # self.verticalLayout_2.addWidget(layout_slm)
        # self.verticalLayout_4.addWidget(layout_image)

        self.arb2d_filename = ''
        self.intensity_filename = ''
        self.reference_filename = ''
        self.interference1_filename = ''
        self.interference2_filename = ''

        try:
            with open('paths.txt', 'r') as f:
                lines = f.readlines()
                self.filename_sim.setText(lines[0].strip())

                self.arb2d_filename = lines[1].strip()
                self.filename_2d.setText(self.arb2d_filename)

                self.intensity_filename = lines[2].strip()
                self.intensity_path.setText(self.intensity_filename)

                self.reference_filename = lines[3].strip()
                self.reference_path.setText(self.reference_filename)

                self.interference1_filename = lines[4].strip()
                self.interference1_path.setText(self.interference1_filename)

                self.interference2_filename = lines[5].strip()
                self.interference2_path.setText(self.interference2_filename)
        except FileNotFoundError:
            with open('paths.txt', 'w') as f:
                f.write('\n\n\n\n\n\n\n\n')
        except IndexError:
            with open('paths.txt', 'w') as f:
                f.write('\n\n\n\n\n\n\n\n')
        self.image_colorbar_1_num = 0
        self.phase_colorbar_num = 0
        self.image_colorbar_1_num = 0
        self.image_colorbar_2_num = 0
        self.image_colorbar_3_num = 0
        self.phase_plot_colorbar_num=0
        self.actve_colorbars = [False, False, False, False]

        self.update_vals()

        self.slm_field = cp.zeros(shape=(self.shape[1], self.shape[0]))
        self.image_field = cp.zeros(shape=(self.shape[1], self.shape[0]))

        self.mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % self.wave, size=(self.shape[1], self.shape[0]))

        self.apply_input()
        self.propagate()

    def set_slm_low_noise(self):
        # Function signature: LcosSetFunction(slm_id, function_id, value)
        # function_id 1 = Drive Mode
        # value 1 = High Stability (Low Noise)
        # value 0 = Standard
        try:
            result = self.Lcoslib.LcosSetFunction(0, 1, 1)
            if result == 0:
                print("Successfully switched to High Stability Mode")
        except AttributeError:
            print("LcosSetFunction not found, trying next...")

    # Update log
    def onUpdateText(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    # Read user inputs
    def update_vals(self):
        self.corr = self.checkbox_correction.isChecked()
        self.wave = int(self.wavelength.text())
        self.input_waist = cp.array([float(self.input_waist_x.text()), float(self.input_waist_y.text())])
        self.shape = [int(self.slm_shape_x.text()), int(self.slm_shape_y.text())]
        self.res = float(self.res_factor.text())
        self.f = cp.array([float(self.f_x.text()), float(self.f_y.text())])

        self.bs = self.checkBox_1d_beamsplitter.isChecked()
        self.bs_n = int(self.num_beams.text())
        self.bs_x_pitch = float(self.x_pitch.text())
        self.bs_amps = cp.array([float(num) for num in self.amps.text().split(', ')])
        self.bs_amps_guess = cp.array([float(num) for num in self.amps_guess.text().split(', ')])
        self.bs_phases = cp.array([float(num) for num in self.phases.text().split(', ')])
        self.bs_N = int(self.N_1d.text())
        self.bs_M = int(self.M_1d.text())

        self.flat = self.checkBox_flat.isChecked()
        self.shift = float(self.flat_phase.text())

        self.hg = self.checkBox_hermite_gauss.isChecked()
        self.hg_n = int(self.tem_n.text())
        self.hg_m = int(self.tem_m.text())

        self.defl = self.checkBox_deflection.isChecked()
        self.def_ax = float(self.def_axis.text())
        self.def_ag = float(self.def_angle.text())

        self.rand = self.checkBox_randphase.isChecked()
        self.rand_op = float(self.rand_opacity.text())

        self.lg = self.checkBox_laguerre_gauss.isChecked()
        self.lg_l = int(self.laguerre_l.text())
        self.lg_p = int(self.laguerre_p.text())

        self.zern = self.checkBox_zernike.isChecked()
        self.zern_weights = ((int(self.zernike_n.text()), int(self.zernike_m.text())), int(self.zernike_weight.text()))
        self.zern_weights = (self.zern_weights,)

        self.arb2d = self.checkBox_arb2d.isChecked()
        self.arb2d_N = int(self.N_2d.text())
        self.arb2d_filename = self.filename_2d.text()

        self.intensity_filename = self.intensity_path.text()
        self.reference_filename = self.reference_path.text()
        self.interference1_filename = self.interference1_path.text()
        self.interference2_filename = self.interference2_path.text()

        self.expotime = self.text_expotime.text()

        self.phase_paths = [self.intensity_filename, self.reference_filename, self.interference1_filename, self.interference2_filename]

        with open('paths.txt', 'w') as f:
            f.write(self.filename_sim.text().strip() + '\n')
            f.write(self.arb2d_filename.strip() + '\n')
            f.write(self.intensity_filename.strip() + '\n')
            f.write(self.reference_filename.strip() + '\n')
            f.write(self.interference1_filename.strip() + '\n')
            f.write(self.interference2_filename.strip() + '\n')

    # Propagate slm field to the image plane
    def propagate(self):
        self.update_vals()

        self.image_field = self.mod.propagate(self.slm_field)
        self.mod.fieldtoBMP(self.image_field, name='Image Plane', wavelength=self.wave, color=True, show=False,
                             figure=self.image_plot, norm=True, sat=False, colorbar=not self.actve_colorbars[1])
        self.image_plot.draw()
        self.actve_colorbars[1] = True

        # temp = cp.where(cp.abs(self.image_field) / cp.max(cp.abs(self.image_field)) > 1, self.image_field, 0)
        plot_gradient(self.image_field, coord=True, fig=self.eprof_plot, intensity=True, imag=False, norm_coords=True)
        self.eprof_plot.draw()

    # Save a created phase pattern
    def save_file(self):
        self.update_vals()

        path = tkinter.filedialog.asksaveasfilename(title='Target Location', defaultextension='.bmp', filetypes=[('BMP', '*.bmp')],)
        self.filename_sim.setText(path)
        self.mod.phaseToBMP(slm.pad_border(cp.angle(self.slm_field), target_shape=self.mod.size), name=path[path.rfind('/'):], color=False, correction=self.corr, wavelength=self.wave, location=path)

    # Open a phase pattern to be simulated
    def open_file(self):
        self.update_vals()

        self.input_profile = self.input_prof()
        phase_path = tkinter.filedialog.askopenfilename(title='Select Phase Pattern', initialfile=self.filename_sim.text())
        if phase_path == '':
            return
        self.filename_sim.setText(phase_path)
        mod = load_mod(phase_path=phase_path, wavelength=self.wave, correction=self.corr)
        self.slm_field = self.input_profile * cp.exp(1j * cp.array(slm.pad_border(mod.phase, self.input_profile.shape)))
        self.mod.fieldtoBMP(self.slm_field, name='SLM Plane', wavelength=self.wave, color=True, show=False,
                            figure=self.slm_plot)
        self.slm_plot.draw()
        return self.slm_field

    def open_file_camera(self,option):
    # apply and show the phase when you select one
        self.update_vals()
        if option==1:
            filename=self.filename_phase_1.text()
        if option==2:
            filename=self.filename_phase_2.text()
        if option==3:
            filename=self.filename_phase_3.text()
        phase_path = tkinter.filedialog.askopenfilename(title='Select Phase Pattern', initialfile=filename)
        if phase_path == '':
            return
        if option == 1:
            units = ' (px)'
            name = 'phase 1'
            self.filename_phase_1.setText(phase_path)
            phase=np.array(im.open(phase_path))
            phase[0, 0] = 0
            phase[-1, -1] = 2 * np.pi
            figure= self.cameraphase_plot
            if cupy_working:
                figure.axes.imshow(phase / 2 / np.pi,  cmap='hsv')
            else:
                figure.axes.imshow(phase / 2 / np.pi,  cmap='hsv')
            # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
            figure.axes.set_xlabel('$x$' + units)
            figure.axes.set_ylabel('$y$' + units)
            figure.axes.set_title(name)
            # figure.axes.tight_layout()
            if self.phase_colorbar_num == 0:
                figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes,
                                            label='$2\\pi$ radians')
            self.phase_colorbar_num = 1
            figure.draw()
        if option == 2:
            units = ' (px)'
            name = 'phase 2'
            self.filename_phase_2.setText(phase_path)
            phase = np.array(im.open(phase_path))
            phase[0, 0] = 0
            phase[-1, -1] = 2 * np.pi
            figure = self.cameraphase_plot
            if cupy_working:
                figure.axes.imshow(phase / 2 / np.pi,  cmap='hsv')
            else:
                figure.axes.imshow(phase / 2 / np.pi, cmap='hsv')
            # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
            figure.axes.set_xlabel('$x$' + units)
            figure.axes.set_ylabel('$y$' + units)
            figure.axes.set_title(name)
            # figure.axes.tight_layout()
            if self.phase_colorbar_num == 0:
                figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes,
                                    label='$2\\pi$ radians')
            self.phase_colorbar_num = 1
            figure.draw()
        if option == 3:
            units = ' (px)'
            name = 'phase 3'
            self.filename_phase_3.setText(phase_path)
            phase = np.array(im.open(phase_path))
            phase[0, 0] = 0
            phase[-1, -1] = 2 * np.pi
            figure = self.cameraphase_plot
            if cupy_working:
                figure.axes.imshow(phase / 2 / np.pi,  cmap='hsv')
            else:
                figure.axes.imshow(phase / 2 / np.pi, cmap='hsv')
            # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
            figure.axes.set_xlabel('$x$' + units)
            figure.axes.set_ylabel('$y$' + units)
            figure.axes.set_title(name)
            # figure.axes.tight_layout()
            if self.phase_colorbar_num == 0:
                figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes,
                                    label='$2\\pi$ radians')
            self.phase_colorbar_num = 1
            figure.draw()
    def autoexpo(self):
    #set auto exposure
        if not self.connect:
            self.flir = FlirCamController_slm.FlirCamController()
            self.flir.initialize()
            self.flir.start_continue()
            self.flir.set_average_frames(1)
            self.connect = True
        self.flir.reset_exposure()
    def disconnect(self):
    #disconnect the camera
        if self.connect:
            self.flir.stop_continue()
            self.flir.close()
            self.connect=False

    def setexpo(self):
     # set the exposure time
        if not self.connect:
            self.flir = FlirCamController_slm.FlirCamController()
            self.flir.initialize()
            self.flir.start_continue()
            self.flir.set_average_frames(1)
            self.connect = True
        if not self.checkbox_auto_expo.isChecked():
            expotime=float(self.text_expotime.text())
            expotime = expotime * 1000
            print(expotime)
            self.flir.configure_exposure(expotime)

        else:
            self.autoexpo()

    def makeBmpArray(self,x, y, outArray):#not using
        inArray = copy.deepcopy(outArray)
        Image_Tiling = self.Lcoslib.Image_Tiling
        Image_Tiling.argtyes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_void_p, c_void_p]
        Image_Tiling.restype = c_int
        Image_Tiling(byref(inArray), self.imageWidth, self.imageHeight, self.imageHeight * self.imageWidth, x, y, byref(c_int(x * y)),
                     byref(outArray))
        return 0
    def showOn2ndDisplay(self, monitorNo, windowNo, x, xShift, y, yShift, array):#not using
        Window_Settings = self.Lcoslib.Window_Settings
        Window_Settings.argtypes = [c_int, c_int, c_int, c_int]
        Window_Settings.restype = c_int
        # print('monitor',monitorNo)
        Window_Settings(monitorNo, windowNo, xShift, yShift)
        Window_Array_to_Display = self.Lcoslib.Window_Array_to_Display
        Window_Array_to_Display.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
        Window_Array_to_Display.restype = c_int
        Window_Array_to_Display(array, x, y, windowNo, x * y)


        return 0
    
    def sent_phase_scan_MDS(self, item):
        t0=time.time()
        if self.load_already[item-1] == 0:
            for i in range(3):
                if (i+1)==item:
                    x = 1272
                    y = 1024
                    if item==1:
                        try:
                            filepath = self.correction_phase_1
                            im = Image.open(filepath)
                            self.imageWidth, self.imageHeight = im.size
                            im_gray = im.convert("L")
                            self.phase_array1=np.array(im_gray)
                            cv2.namedWindow("foo")
                            cv2.imshow("foo", np.reshape(np.asarray(self.phase_array1), (y, x)))
                            cv2.waitKey(1)
                        except Exception as e:
                            print('error-load:',e)
                    if item ==2:
                        try:
                            filepath = self.correction_phase_2  # self.filename_phase_1.text()
                            im = Image.open(filepath)

                            self.imageWidth, self.imageHeight = im.size
                            im_gray = im.convert("L")

                            self.phase_array2 = np.array(im_gray)
                            cv2.namedWindow("foo")
                            cv2.imshow("foo", np.reshape(np.asarray(self.phase_array2), (y, x)))
                            cv2.waitKey(1)
                        except Exception as e:
                            print('error-load:', e)
                    if item==3:
                        try:
                            filepath = self.correction_phase_3 
                            im = Image.open(filepath)
                            self.imageWidth, self.imageHeight = im.size
                            im_gray = im.convert("L")
                            self.phase_array3 = np.array(im_gray)
                            cv2.namedWindow("foo")
                            cv2.imshow("foo", np.reshape(np.asarray(self.phase_array3), (y, x)))
                            cv2.waitKey(1)
                        except Exception as e:
                            print('error-load:', e)




    def load_phase_scan_MDS(self,item):
        x = 1272
        y = 1024
        FARRAY=np.zeros([x,y])
        if item==1:

            try:
                self.phase_array1 = FARRAY#(0)

                filepath = self.correction_phase_1#self.filename_phase_1.text()
                im = Image.open(filepath)

                self.imageWidth, self.imageHeight = im.size
                im_gray = im.convert("L")
                self.phase_array1=np.array(im_gray)
                # im_array = np.array(im_gray)
                # for i in range(self.imageWidth):
                #     for j in range(self.imageHeight):
                #         #self.phase_array1[i + self.imageWidth * j] = im_array[j][i]  # -im_gray.getpixel((i,j))
                #         self.phase_array1[i,j] = im_array[j][i]  # -im_gray.getpixel((i,j))
            except Exception as e:
                print('error-load:',e)
        if item ==2:
            try:
                self.phase_array2 = FARRAY  # (0)

                filepath = self.correction_phase_2  # self.filename_phase_1.text()
                im = Image.open(filepath)

                self.imageWidth, self.imageHeight = im.size
                im_gray = im.convert("L")

                self.phase_array2 = np.array(im_gray)
            except Exception as e:
                print('error-load:', e)
        if item==3:
            try:
                self.phase_array3 = FARRAY  # (0)

                filepath = self.correction_phase_3  # self.filename_phase_1.text()
                im = Image.open(filepath)

                self.imageWidth, self.imageHeight = im.size
                im_gray = im.convert("L")
                self.phase_array3 = np.array(im_gray)
            except Exception as e:
                print('error-load:', e)
    def sent_phase_scan(self, item):# not using old version
        x = 1272
        y = 1024
        monitorNo = 2
        windowNo = 0
        xShift = 0
        yShift = 0
        if self.load_already[item-1] == 0:
            for i in range(3):
                if (i+1)==item:
                    self.load_phase_scan(i+1)

                    self.load_already[item-1] = 1

        if item==1:
            try:
                self.makeBmpArray( x, y, self.phase_array1)
                self.showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, self.phase_array1)
            except  Exception as e:
                print('error_sent1',e)
        if item==2:
            self.makeBmpArray( x, y, self.phase_array2)
            self.showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, self.phase_array2)
        if item==3:
            self.makeBmpArray(x, y, self.phase_array3)
            self.showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, self.phase_array3)
    def load_phase_scan(self,item):# not using old version
        x = 1272
        y = 1024
        monitorNo = 1
        windowNo = 0
        xShift = 0
        yShift = 0
        array_size = x * y
        FARRAY = c_uint8 * array_size
        if item==1:

            try:
                self.phase_array1 = FARRAY(0)

                filepath = self.correction_phase_1#self.filename_phase_1.text()
                im = Image.open(filepath)

                self.imageWidth, self.imageHeight = im.size
                im_gray = im.convert("L")
                im_array = np.array(im_gray)
                for i in range(self.imageWidth):
                    for j in range(self.imageHeight):
                        self.phase_array1[i + self.imageWidth * j] = im_array[j][i]  # -im_gray.getpixel((i,j))
            except Exception as e:
                print('error-load:',e)
        if item ==2:
            FARRAY = c_uint8 * array_size
            self.phase_array2 = FARRAY(0)
            filepath = self.correction_phase_2#self.filename_phase_2.text()
            im = Image.open(filepath)
            self.imageWidth, self.imageHeight = im.size
            im_gray = im.convert("L")
            im_array = np.array(im_gray)
            for i in range(self.imageWidth):
                for j in range(self.imageHeight):
                    self.phase_array2[i + self.imageWidth * j] = im_array[j][i]
        if item==3:
            FARRAY = c_uint8 * array_size
            self.phase_array3 = FARRAY(0)
            filepath = self.correction_phase_3#self.filename_phase_3.text()
            im = Image.open(filepath)
            self.imageWidth, self.imageHeight = im.size
            im_gray = im.convert("L")
            im_array = np.array(im_gray)
            for i in range(self.imageWidth):
                for j in range(self.imageHeight):
                    self.phase_array3[i + self.imageWidth * j] = im_array[j][i]
    def load_phase(self,item):# not using old version
        x = 1272
        y = 1024
        monitorNo = 1
        windowNo = 0
        xShift = 0
        yShift = 0
        array_size = x * y
        FARRAY = c_uint8 * array_size
        if item==1:

            try:
                self.phase_array1 = FARRAY(0)

                filepath = self.filename_phase_1.text()
                im = Image.open(filepath)

                self.imageWidth, self.imageHeight = im.size
                im_gray = im.convert("L")
                im_array = np.array(im_gray)
                for i in range(self.imageWidth):
                    for j in range(self.imageHeight):
                        self.phase_array1[i + self.imageWidth * j] = im_array[j][i]  # -im_gray.getpixel((i,j))
            except Exception as e:
                print('error-load:',e)
        if item ==2:
            FARRAY = c_uint8 * array_size
            self.phase_array2 = FARRAY(0)
            filepath = self.filename_phase_2.text()
            im = Image.open(filepath)
            self.imageWidth, self.imageHeight = im.size
            im_gray = im.convert("L")
            im_array = np.array(im_gray)
            for i in range(self.imageWidth):
                for j in range(self.imageHeight):
                    self.phase_array2[i + self.imageWidth * j] = im_array[j][i]
        if item==3:
            FARRAY = c_uint8 * array_size
            self.phase_array3 = FARRAY(0)
            filepath = self.filename_phase_3.text()
            im = Image.open(filepath)
            self.imageWidth, self.imageHeight = im.size
            im_gray = im.convert("L")
            im_array = np.array(im_gray)
            for i in range(self.imageWidth):
                for j in range(self.imageHeight):
                    self.phase_array3[i + self.imageWidth * j] = im_array[j][i]
    def sent_phase(self, item):# not using old version
        x = 1272
        y = 1024
        monitorNo = 2
        windowNo = 0
        xShift = 0
        yShift = 0
        if self.load_already[item-1] == 0:
            for i in range(3):
                if (i+1)==item:
                    self.load_phase(i+1)

                    self.load_already[item-1] = 1

        if item==1:
            try:

                self.makeBmpArray( x, y, self.phase_array1)
                self.showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, self.phase_array1)
            except  Exception as e:
                print('error_sent1',e)
        if item==2:

            self.makeBmpArray( x, y, self.phase_array2)
            self.showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, self.phase_array2)
        if item==3:

            self.makeBmpArray(x, y, self.phase_array3)
            self.showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, self.phase_array3)
    def findfilepath(self,folder_path):# search images in the file folder when take images
        image_paths = []

        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        image_paths.append(os.path.join(root, file))
        except Exception as e:
            print('filelocate error:',e)
        path_=[]
        for i in image_paths:
            name=os.path.basename(i)
            name=name[6:len(name)-4]
            if name>=self.starttime:
                path_.append(i)
        return path_
    def patch_generate(self):
        #is used when we want some phase emask with separate patchs when doing SLM self-interfere
        from ArrayModulator_v1 import ArrayModulator, load_mod
        import slm_v1 as slm
        try:
            wavelength = 411
            mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
            grady = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_y.text()) * np.pi / 180, axis=0,
                                 wavelength=wavelength)
            gradx = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_x.text()) * np.pi / 180, axis=1,
                                 wavelength=wavelength)

            phase = gradx + grady
            centre = [int(phase.shape[0]), int(phase.shape[1])]

            block_size = [int(self.text_patch_size_x.text()), int(self.text_patch_size_y.text())]
            centre = [int((phase.shape[0]) / 2), int((phase.shape[1]) / 2)]
            phase_slice = slm.pad_border(phase[centre[0] - block_size[0]:centre[0] + block_size[0] + 1,
                                         centre[1] - block_size[1]:centre[1] + block_size[1] + 1], (1024, 1272))

            target_shape = (1024, 1272)
            target_center = np.array(target_shape) / 2


        except Exception as e:
            print('error', e)


        shift = [int(self.text_shift_x.text()), int(self.text_shift_y.text())]
        output = np.zeros(shape=target_shape)
        phase_center = [shift[0], shift[1]]

        phase_shift_0 = phase[centre[0] - block_size[0] + shift[0]:centre[0] + block_size[0] + 1 + shift[0],
                        centre[1] - block_size[1] + shift[1]:centre[1] + block_size[1] + 1 + shift[1]]
        phi = 0
        phase_shift = phase_shift_0 + phi
        output[:phase_shift.shape[0], :phase_shift.shape[1]] = phase_shift

        output = np.roll(output, np.array(target_center - phase_center - block_size, dtype=np.int64),
                         axis=(0, 1))

        phase_rota = phase_slice + output
        mod.add_phase(phase_rota)
        mod.phaseToBMP(mod.phase, name=f'patch_{shift[0]}_{shift[1]}_shift_{phi}', color=False, correction=False,
                       wavelength=411)

    def dataprocess(self):
    # process data in SLM optical abberation correction
        try:
            length_y = int(int(self.text_scan_area_y.text()) / int(self.text_patch_size_y.text()))
            length_x = int(int(self.text_scan_area_x.text()) / int(self.text_patch_size_x.text()))
            number_x = list(range(length_x))
            number_y = list(range(length_y))
            block_size = [int(self.text_patch_size_x.text()), int(self.text_patch_size_y.text())]
            area_size = [int(int(self.text_scan_area_x.text()) / 2), int(int(self.text_scan_area_y.text()) / 2)]
            phase = np.zeros((length_x, length_y))
            p_amp_MDS_full= np.zeros((length_x, length_y))
            # print(length_x, length_y)
            position_error = []
            phase_tiltx = np.zeros((length_x, length_y))
            phase_tilty = np.zeros((length_x, length_y))
            radius = ((length_x ** 2 ) / 4)
            print('radius=', radius)
            for i in range(length_x):
                for j in range(length_y):
                    if ((i - length_x / 2) ** 2 + (j - length_y / 2) ** 2) < radius:
                        shift = [-number_x[i] * block_size[0] + area_size[0], - number_y[j] * block_size[1] + area_size[1]]
                        print('shift=',shift)
                        path = findfilepath(shift)
                        try:
                            data1 = (np.array(im.open(path[0]), dtype=np.float64))
                            data2 = (np.array(im.open(path[1]), dtype=np.float64))
                            data3 = (np.array(im.open(path[2]), dtype=np.float64))
                            #p_real = np.zeros((3000, 4000), dtype=np.float64) MDS
                            #p_real = np.zeros((1000, 400), dtype=np.float64)
                            p_real = data1 + data1 - data2 - data3
                            p_image = data2 - data3
                            p_inten=data1 + data2 + data3
                            index = np.argmax(p_inten)
                            cooridnates = np.unravel_index(index, p_inten.shape)

                            p = p_real + 1j * p_image * np.sqrt(3)
                            phase_ = np.angle(p)
                            p_amp_MDS=np.abs(p)

                            if False: #high magnification
                                xxx =516#interference center #732#cropped beam center#1482#beam center1565#1577#1520
                                yyy =170#cropped beam center#2004#beam center1985#1958#1950
                                sq_size=100
                            if True: #Low magnification
                                xxx =172#interference center #732#cropped beam center#1482#beam center1565#1577#1520
                                yyy =57#cropped beam center#2004#beam center1985#1958#1950
                                sq_size=34
                            phase_ = phase_[xxx - sq_size:xxx + sq_size, yyy - sq_size:yyy + sq_size]
                            phase_unwrap = unwrap_phase(phase_)
                            phase[number_x[i]][number_y[j]] = np.average(phase_unwrap)
                            p_amp_MDScrop=p_amp_MDS[xxx - sq_size:xxx + sq_size, yyy - sq_size:yyy + sq_size]
                            p_amp_MDS_full[number_x[i]][number_y[j]] = np.average(p_amp_MDScrop)
                        except Exception as e:
                            print(e)
                            phase[number_x[i]][number_y[j]] = np.nan
                            p_amp_MDS_full[number_x[i]][number_y[j]]=np.nan
                        # plt.figure(figsize=(8, 6))
                        # plt.imshow(p_inten[xxx - 500:xxx + 500, yyy - 500:yyy + 500], cmap='viridis',
                        #             interpolation='nearest')
                        # plt.colorbar()
                        # plt.title('phase wrapped')
                        # plt.show()
                        # #
                        # plt.figure(figsize=(8, 6))
                        # plt.imshow(phase_unwrap, cmap='viridis', interpolation='nearest')
                        # plt.colorbar()
                        # plt.title('phase unwrapped')
                        # plt.show()
                        #MDS
                        '''
                        x = np.arange(phase_unwrap.shape[0])
                        y = np.arange(phase_unwrap.shape[1])
                        X, Y = np.meshgrid(x, y)
                        X_flat = X.flatten()
                        Y_flat = Y.flatten()
                        phase_flat = phase_unwrap.flatten()
                        features = np.vstack((X_flat, Y_flat)).T
                        model = LinearRegression()
                        model.fit(features, phase_flat)
                        phase_predict = model.predict((features))
                        r_square = r2_score(phase_unwrap.flatten(), phase_predict)
                        if r_square < 0.97:
                            print('error', i, j)
                            print(r_square)
                            position_error.append(shift)
                            phase[number_x[i]][number_y[j]] = 0
                            phase_tiltx[number_x[i]][number_y[j]] = 0
                            phase_tilty[number_x[i]][number_y[j]] = 0
                        else:
                            a = model.coef_[0]
                            b = model.coef_[1]
                            c = model.intercept_
                            print(a, b, c)
                            print('r_square', r_square)
                            #phase[number_x[i]][number_y[j]] = c
                            phase[number_x[i]][number_y[j]] = a*sq_size+b*sq_size+c
                            phase_tiltx[number_x[i]][number_y[j]] = a
                            phase_tilty[number_x[i]][number_y[j]] = b
                        '''

            #phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase.txt'
            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase_MDS_crop_interferencecenter1.txt'

            np.savetxt(phase_path, np.array(phase))
            #phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_tilt_x_add.txt'
            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\corrections_tilt_x_add_MDS_crop_interferencebeamcenter1.txt'

            np.savetxt(phase_path, np.array(phase_tiltx))
            #phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_tilt_y_add.txt'
            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_tilt_y_add_MDS_crop_interferencebeamcenter1.txt'

            np.savetxt(phase_path, np.array(phase_tilty))

            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase_MDS_crop_amp1.txt'
            np.savetxt(phase_path, np.array(p_amp_MDS_full))
            plt.imshow(phase)
            plt.title(xxx*100000+yyy)
            plt.show()
            plt.imshow(p_amp_MDS_full)
            plt.show()
        except Exception  as e:
            print('error',e)
    def save_data(self):
        try:
            name='array'+self.text_array_number_1.text()+'_'+self.text_array_number_2.text()+' tem0'+self.text_tem_n.text()
            name =name+' phase'+self.text_targetphase.text()+'_'+time.strftime("%Y%m%d-%H%M%S")
            try:
                path=Path('C:/Users/RiceT/Documents/SLM_computation/data/'+name)
                path.mkdir()
            except Exception as e:
                print('foler '+'C:/Users/RiceT/Documents/SLM_computation/data/'+name+' already exsist')
            folder_path='C:/Users/RiceT/Documents/SLM_computation/data/'+name+'/'
            paths = [self.filename_phase_0.text(),
                     r"C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image1.bmp",
                     r"C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image2.bmp",
                     r"C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image3.bmp"]
            data0 = (np.array(im.open(paths[0]), dtype=np.float64))
            cv2.imwrite(folder_path+'intensity.bmp', data0)
            np.savetxt(folder_path + 'intensity.txt', np.array(data0))
            data1 = (np.array(im.open(paths[1]), dtype=np.float64))
            cv2.imwrite(folder_path + 'phase offset_0.bmp', data1)
            np.savetxt(folder_path + 'phase offset_0.txt', np.array(data1))
            data2 = (np.array(im.open(paths[2]), dtype=np.float64))
            cv2.imwrite(folder_path + 'phase offset_120.bmp', data2)
            np.savetxt(folder_path + 'phase offset_120.txt', np.array(data2))
            data3 = (np.array(im.open(paths[3]), dtype=np.float64))
            cv2.imwrite(folder_path + 'phase offset_240.bmp', data3)
            np.savetxt(folder_path + 'phase offset_240.txt', np.array(data3))
            data4 = (np.array(self.phase_extracted, dtype=np.float64))
            cv2.imwrite(folder_path + 'phase_extracted.bmp', data4)
            np.savetxt(folder_path + 'phase_extracted.txt', np.array(data4))
            data5 = (np.array(self.amp_extracted, dtype=np.float64))
            cv2.imwrite(folder_path + 'amp_extracted.bmp', data5)
            np.savetxt(folder_path + 'amp_extracted.txt', np.array(data5))

        except Exception as e:
            import traceback
            print('save data', e)
            traceback.print_exc()
    def reference_scan_analyse(self): #Finds center of beam
        directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\reference\intensity'
        shift=[0.0,0.0]
        now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(0)
        name = f"image_{now}.bmp"
        path3 = os.path.join(directory, name)
        data1 = (np.array(im.open(path3), dtype=np.float64))
        index = np.argmax(data1)
        cooridnates_center = np.unravel_index(index, data1.shape)
        # print(cooridnates)
        # print('ena4')

        radius = 10
        waist1 = int(radius)
        waist2 = int(radius)
        if cooridnates_center[0] < radius:
            waist1 = math.floor(cooridnates_center[0])
        if cooridnates_center[0] > 3000 - radius:
            waist1 = math.floor(3000 - cooridnates_center[0])
        if cooridnates_center[1] < radius:
            waist2 = math.floor(cooridnates_center[1])
        if cooridnates_center[1] > 4000 - radius:
            waist2 = math.floor(4000 - cooridnates_center[1])
        waist = np.min([waist1, waist2])
        print("center:", cooridnates_center)
        print(path3)

        shift=[0.02,0.02]
        now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(0)
        name = f"image_{now}.bmp"
        path1 = os.path.join(directory, name)
        print(path1)
        data2 = (np.array(im.open(path1), dtype=np.float64))
        print(cooridnates_center[0],cooridnates_center[1])
        data2[cooridnates_center[0] - radius:cooridnates_center[0] + radius, cooridnates_center[1] - radius:
                                                                             cooridnates_center[1] + radius] = 0
        index = np.argmax(data2)
        cooridnates2 = np.unravel_index(index, data2.shape)
        print("coord2:",cooridnates2)
        plt.imshow(data2)
        plt.colorbar()
        plt.show()
        plt.imshow(data1)
        plt.colorbar()
        plt.show()


##MDS refernce_scan
    def reference_scan(self):
        center_coordinates=[1672,2070]
        phase_collect=np.full((3000,4000), np.nan)
        from ArrayModulator_v1 import ArrayModulator, load_mod
        import slm_v1 as slm
        try:
            point_x = int(self.text_point_x.text())
            point_y = int(self.text_point_y.text())
            step_y = float((self.text_step_y.text()))
            step_x = float((self.text_step_x.text()))
            deflection_upperright_x = float((self.text_upperleft_x.text()))
            deflection_upperright_y = float((self.text_upperleft_y.text()))
        except Exception as e:
            print(e)
        if not self.connect:
            self.flir = FlirCamController_slm.FlirCamController()
            self.flir.initialize()
            self.flir.start_continue()
            self.flir.set_average_frames(1)
            self.connect = True
        print('em')
        try:
            wavelength = 411
            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase.txt'
            correction = np.loadtxt(phase_path)
            print(correction.shape)
            block_size = [int(self.text_patch_size_x.text()), int(self.text_patch_size_y.text())]
            area_size = [int(int(self.text_scan_area_x.text())), int(int(self.text_scan_area_y.text()))]
            length_y = int(int(self.text_scan_area_y.text()) / int(self.text_patch_size_y.text()))
            length_x = int(int(self.text_scan_area_x.text()) / int(self.text_patch_size_x.text()))

            matrix_correction = np.zeros((area_size[0], area_size[1]))
            print(area_size[0], area_size[1])
            print(length_x, length_y)
            radius = ((length_x ** 2) / 4)
            print('radius=', radius)
            for i in range(length_x):
                for j in range(length_y):
                    if ((i - length_x / 2) ** 2 + (j - length_y / 2) ** 2) < radius:
                        for m in range(block_size[0]):
                            for n in range(block_size[1]):
                                matrix_correction[block_size[0] * i + m][block_size[1] * j + n] = correction[i][j]
            matrix_correction = slm.pad_border(matrix_correction, (1024, 1272))

            mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
            self.correction_phase_1 = r"C:\Users\RiceT\Documents\SLM_computation\images\phase_mask_0.bmp"
            self.correction_phase_2 = r"C:\Users\RiceT\Documents\SLM_computation\images\phase_mask_120.bmp"
            self.correction_phase_3 = r"C:\Users\RiceT\Documents\SLM_computation\images\phase_mask_240.bmp"
            ena = np.zeros(3)
            data_fit = np.zeros((3000, 4000))
            x_data = []
            tilt_data_x = []
            y_data = []
            tilt_data_y = []
        except Exception as e:
            import traceback
            print('scan', e)
            traceback.print_exc()
        self.correction_phase_1 = r'C:/Users/RiceT/Documents/SLM_computation/images/phase_mask_0.bmp'
        self.correction_phase_2 = r'C:/Users/RiceT/Documents/SLM_computation/images/phase_mask_120.bmp'
        self.correction_phase_3 = r'C:/Users/RiceT/Documents/SLM_computation/images/phase_mask_240.bmp'
        number_x = list(range(point_x))
        random.shuffle(number_x)
        number_y = list(range(point_y))
        random.shuffle(number_y)
        print('number_x',number_x,'number_y',number_y)
        for i in range(point_x):
            # time.sleep(5)
            for j in range(point_y):
                print('em')
                try:
                    print('ena0',i,j)
                    shift = [-number_y[j] * step_y + deflection_upperright_y,
                             -number_x[i] * step_x + deflection_upperright_x]
                    print(shift)
                    if not(shift[0]==0.0 and shift[1]==0.0):
                        grady = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_y.text()) * np.pi / 180,
                                             axis=0, wavelength=wavelength)
                        gradx = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_x.text()) * np.pi / 180,
                                             axis=1, wavelength=wavelength)
                        mod.add_phase(grady) #Original deflections
                        mod.add_phase(gradx)
                        #grady = mod.gradient(mod.beams[0][1], angle=shift[0] * np.pi / 180, axis=0, wavelength=wavelength)
                        #gradx = mod.gradient(mod.beams[0][1], angle=shift[1] * np.pi / 180, axis=1, wavelength=wavelength)
                        #mod.add_phase(grady)  #Deflections for the reference beam scan
                        #mod.add_phase(gradx)

                        ##MDS adding compensation for testing the refernce scan
                        # MDS #Target curve compensation
                        a = 0#-4.03456842e-06  # -3.733022642775838e-06
                        b = 1937.23  # 1805#1700#1776
                        c = 0#-3.9478368e-06  # -3.893853882563125e-06
                        d = 1573.195  # 3400#1790
                        lin_c=6.08835452e-03
                        target_phase = np.zeros((1024, 1272))
                        for ir in range(1024):
                            for jr in range(1272):
                                # target_phase[i][j]=c*((i-512)*104+1500-d)**2/2+a*((j-636)*84+2000-b)**2/2
                                target_phase[i][j] = c * ((i - 511) * 53.187 + 1573.195 - d) ** 2 / 2 + a * (
                                        (j - 635.5) * 41.045 + 1937.23 - b) ** 2 / 2  # MDS
                        target_phase = target_phase[462:562, 586:686]
                        target_phase = slm.pad_border(target_phase, (1024, 1272))
                        print("shift:",shift[0],shift[1])

                        #wu2D for 2 beams of center and another.
                        wu_1x4 = mod.wu_algorithm2D_MDS(n=1, m=2,phase_tem_compensation=np.exp(1j * np.array(target_phase)),
                                                    M=15, name='wu_1x4', amps=(1.0,1.0), amps_guess=(1.0,1.0), phases=(0.0,0.0),
                                                    x_pitch=shift[0],y_pitch=shift[1], plots=False,
                                                    input_profile=profile.Profile.input_gaussian(beam_size=(0.35, 0.35),
                                                                                                 size=np.array(mod.size)),
                                                    phase_memory=True, tem01=False)
                        # wu_1x4 = phase_rotate(wu_1x4, 6.5 / 180 * np.pi)
                        mod.add_phase(wu_1x4)


                        mod.phase = mod.phase + matrix_correction
                        mod.phase = mod.phase[512 - int(area_size[0] / 2):512 + int(area_size[0] / 2),
                                    636 - int(area_size[1] / 2):636 + int(area_size[1] / 2)]
                        mod.phase = slm.pad_border(mod.phase, (1024, 1272))
                        # temnm = mod.temnm(mod.beams[0][1], n=1, m=1)
                        print('ena1')

                        mod.phaseToBMP(mod.phase, name='phase_mask_0', color=False, correction=False, wavelength=411)

                        mod.phase = mod.phase + 120 / 180 * np.pi
                        mod.phaseToBMP(mod.phase, name='phase_mask_120', color=False, correction=False, wavelength=411)
                        mod.phase = mod.phase + 120 / 180 * np.pi
                        mod.phaseToBMP(mod.phase, name='phase_mask_240', color=False, correction=False, wavelength=411)
                        mod.add_phase(-mod.phase)
                        print('ena2')
                        cooridnates = [0, 0]
                        r_square = 0
                        for p in range(1):
                            if 1:  # r_square<0.95 and cooridnates[0]!=1844:

                                self.load_already = np.zeros(3)
                                self.sent_phase_scan_MDS(1)

                                time.sleep(0.3)
                                self.flir.acquire_continue()
                                directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
                                now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(0)
                                name = f"image_{now}.bmp"
                                path = os.path.join(directory, name)
                                self.flir.file_save(path)

                                self.sent_phase_scan_MDS(2)

                                time.sleep(0.3)
                                self.flir.acquire_continue()
                                directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
                                now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(120)

                                name = f"image_{now}.bmp"
                                path = os.path.join(directory, name)
                                self.flir.file_save(path)
                                self.sent_phase_scan_MDS(3)
                                time.sleep(0.3)
                                self.flir.acquire_continue()
                                directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
                                now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(240)

                                name = f"image_{now}.bmp"
                                path = os.path.join(directory, name)
                                self.flir.file_save(path)

                                path = findfilepath2(shift)
                                data1 = (np.array(im.open(path[0]), dtype=np.float64))
                                data2 = (np.array(im.open(path[1]), dtype=np.float64))
                                data3 = (np.array(im.open(path[2]), dtype=np.float64))

                                # p_real = np.zeros((3000, 4000), dtype=np.float64)
                                p_real = data1 + data1 - data2 - data3
                                p_image = data2 - data3

                                p = p_real + 1j * p_image * np.sqrt(3)


                                phase_amps = data1 + data2 + data3
                                phase_amps[center_coordinates[0]-200:center_coordinates[0]+200,center_coordinates[1]-200:center_coordinates[1]+200]=0
                                print(shift)
                                #index = np.argmax(phase_amps)
                                index = np.argmax(cv2.GaussianBlur(phase_amps,ksize=(0,0),sigmaX=10,sigmaY=10))

                                cooridnates = np.unravel_index(index, phase_amps.shape)
                                print(cooridnates)
                                phase = np.angle(p)
                                # print('ena4')
                                phase_center_avg=cv2.GaussianBlur(phase,ksize=(0,0),sigmaX=10,sigmaY=10)[center_coordinates[0],center_coordinates[1]]
                                print("phase_center",phase_center_avg)
                                phase=np.mod(phase-phase_center_avg,2*np.pi)

                                radius = 200
                                waist1 = int(radius)
                                waist2 = int(radius)
                                if cooridnates[0] < radius:
                                    waist1 = math.floor(cooridnates[0])
                                if cooridnates[0] > 3000 - radius:
                                    waist1 = math.floor(3000 - cooridnates[0])
                                if cooridnates[1] < radius:
                                    waist2 = math.floor(cooridnates[1])
                                if cooridnates[1] > 4000 - radius:
                                    waist2 = math.floor(4000 - cooridnates[1])
                                waist = np.min([waist1, waist2])
                                # print(waist,waist1,waist2)
                                print("collect",cooridnates)
                                # plt.figure(figsize=(8, 6))
                                # plt.imshow(phase_amps, cmap='viridis', interpolation='nearest')
                                # plt.colorbar()
                                # plt.show()
                                #
                                # plt.figure(figsize=(8, 6))
                                # plt.imshow(phase, cmap='viridis', interpolation='nearest')
                                # plt.colorbar()
                                # plt.show()

                                waist = int(waist / 2)
                                phase_unwrap = (phase[cooridnates[0] - waist:cooridnates[0] + waist,
                                                cooridnates[1] - waist:cooridnates[1] + waist])
                                phase_collect[cooridnates[0] - waist:cooridnates[0] + waist,
                                                cooridnates[1] - waist:cooridnates[1] + waist]=phase[cooridnates[0] - waist:cooridnates[0] + waist,
                                                cooridnates[1] - waist:cooridnates[1] + waist]
                                # plt.figure(figsize=(8, 6))
                                # plt.imshow((phase[cooridnates[0] - waist:cooridnates[0] + waist,
                                #            cooridnates[1] - waist:cooridnates[1] + waist]), cmap='viridis', interpolation='nearest')
                                # plt.colorbar()
                                # plt.show()
                                x = np.arange(phase_unwrap.shape[0])
                                y = np.arange(phase_unwrap.shape[1])
                                X, Y = np.meshgrid(x, y)
                                X_flat = X.flatten()
                                Y_flat = Y.flatten()
                                phase_flat = phase_unwrap.flatten()
                                features = np.vstack((X_flat, Y_flat)).T
                                model = LinearRegression()
                                model.fit(features, phase_flat)
                                phase_predict = model.predict((features))
                                r_square = r2_score(phase_unwrap.flatten(), phase_predict)
                                a = model.coef_[0]
                                b = model.coef_[1]
                                print(a, b, 'r_square=', r_square)

                                if r_square > 0.82 and r_square < 1:
                                    # plt.figure(figsize=(8, 6))
                                    # plt.imshow(phase_unwrap, cmap='viridis', interpolation='nearest')
                                    # plt.colorbar()
                                    # plt.title('measure')
                                    # plt.figure(figsize=(8, 6))
                                    # plt.imshow(phase_predict.reshape(X.shape), cmap='viridis', interpolation='nearest')
                                    # plt.colorbar()
                                    # plt.title('fit')
                                    # plt.show()
                                    data_fit[cooridnates[0] - waist:cooridnates[0] + waist,
                                    cooridnates[1] - waist:cooridnates[1] + waist] = \
                                        phase[cooridnates[0] - waist:cooridnates[0] + waist,
                                        cooridnates[1] - waist:cooridnates[1] + waist]

                                    x_data.append(cooridnates[1])
                                    tilt_data_x.append(a)
                                    y_data.append(cooridnates[0])
                                    tilt_data_y.append(b)
                                    print(a, b)


                except Exception as e:
                    import traceback
                    print('scan', e)
                    traceback.print_exc()

        #MDS test added for testing
        phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\phasecollectfull_17_01_2025.txt'
        np.savetxt(phase_path, np.array(phase_collect))
        cv2.imwrite("phasecollectfull_17_01_2025.bmp", phase_collect)
        plt.imshow(phase_collect)
        plt.colorbar()
        plt.show()
        phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\phasecollectfull1_17_01_2025.txt'
        np.savetxt(phase_path, np.array(phase_collect))
        cv2.imwrite("phasecollectfull1_17_01_2025.bmp", phase_collect)
        # phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\fit_data.txt'
        # np.savetxt(phase_path, np.array(data_fit))
        #
        # plt.figure(figsize=(8, 6))
        # plt.imshow(data_fit, cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        #
        # # z_data = []
        # try:
        #
        #     tilt_data_x = np.array(tilt_data_x)
        #     x_data = np.array(x_data)
        #
        #     popt, pcov = curve_fit(func, x_data, tilt_data_x, p0=[-1 * 10 ** (-5), 0], maxfev=10000)
        #     self.a_fit, self.b_fit = popt
        #     y_fit = func(x_data, *popt)
        #
        #     # 计算 R²
        #     residuals = tilt_data_x - y_fit
        #     ss_res = np.sum(residuals ** 2)
        #     ss_total = np.sum((tilt_data_x - np.mean(tilt_data_x)) ** 2)
        #     r_squared = 1 - (ss_res / ss_total)
        #
        #     # 计算 RMSE
        #     rmse = np.sqrt(np.mean(residuals ** 2))
        #
        #     # 打印 R² 和 RMSE
        #     print(f"fit para a: {self.a_fit}, b: {self.b_fit}")
        #     print(f"R²: {r_squared}, RMSE: {rmse}")
        #
        #     tilt_data_y = np.array(tilt_data_y)
        #     y_data = np.array(y_data)
        #     popt, pcov = curve_fit(func, y_data, tilt_data_y, p0=[-1 * 10 ** (-5), 1500], maxfev=10000)
        #     self.c_fit, self.d_fit = popt
        #     y_fit = func(y_data, *popt)
        #
        #     # 计算 R²
        #     residuals = tilt_data_y - y_fit
        #     ss_res = np.sum(residuals ** 2)
        #     ss_total = np.sum((tilt_data_y - np.mean(tilt_data_y)) ** 2)
        #     r_squared = 1 - (ss_res / ss_total)
        #
        #     # 计算 RMSE
        #     rmse = np.sqrt(np.mean(residuals ** 2))
        #
        #     # 打印 R² 和 RMSE
        #     print(f"fit para c: {self.c_fit}, d: {self.d_fit}")
        #     print(f"R²: {r_squared}, RMSE: {rmse}")
        #     for i in range(3000):
        #         for j in range(4000):
        #             if data_fit[i][j] != 0:
        #                 data_fit[i][j] = (self.a_fit * (j - self.b_fit) ** 2 / 2 + self.c_fit * (
        #                         i - self.d_fit) ** 2 / 2) % (
        #                                          2 * np.pi)
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(data_fit, cmap='viridis', interpolation='nearest')
        #     plt.colorbar()
        #     plt.title('fit')
        #     plt.show()
        #
        #     # for y in y_data:
        #     #     x = []
        #     #     intensity_plot = []
        #     #     field_plot = []
        #     #     for i in range(data_fit.shape[1]-1):
        #     #         x.append(i)
        #     #         intensity_plot.append(data_fit[y][i+1]-data_fit[y][i])
        #     #
        #     #         field_plot.append(self.a_fit * (y - self.b_fit)  + self.c_fit * (
        #     #                          i - self.d_fit))
        #     #     plt.plot(x, intensity_plot, label='measure', color='blue')
        #     #     plt.plot(x, field_plot, label='expect', color='red')
        #     #     plt.show()
        #     phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\data_fit.txt'
        #
        #     np.savetxt(phase_path, data_fit)
        #
        #
        # except Exception as e:
        #     import traceback
        #     print('check', e)
        #     traceback.print_exc()

    def reference_scan_Original(self):
        from ArrayModulator_v1 import ArrayModulator, load_mod
        import slm_v1 as slm
        try:
            point_x=int(self.text_point_x.text())
            point_y=int(self.text_point_y.text())
            step_y=float((self.text_step_y.text()))
            step_x=float((self.text_step_x.text()))
            deflection_upperright_x=float((self.text_upperleft_x.text()))
            deflection_upperright_y = float((self.text_upperleft_y.text()))
        except Exception as e:
            print(e)
        if not self.connect:
            self.flir = FlirCamController_slm.FlirCamController()
            self.flir.initialize()
            self.flir.start_continue()
            self.flir.set_average_frames(1)
            self.connect = True
        print('em')
        try:
            wavelength=411
            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase.txt'
            correction = np.loadtxt(phase_path)
            print(correction.shape)
            block_size = [int(self.text_patch_size_x.text()), int(self.text_patch_size_y.text())]
            area_size = [int(int(self.text_scan_area_x.text())), int(int(self.text_scan_area_y.text()))]
            length_y = int(int(self.text_scan_area_y.text()) / int(self.text_patch_size_y.text()))
            length_x = int(int(self.text_scan_area_x.text()) / int(self.text_patch_size_x.text()))

            matrix_correction = np.zeros((area_size[0], area_size[1]))
            print(area_size[0],area_size[1])
            print(length_x, length_y)
            radius = ((length_x ** 2) / 4)
            print('radius=', radius)
            for i in range(length_x):
                for j in range(length_y):
                    if ((i - length_x / 2) ** 2 + (j - length_y / 2) ** 2) < radius:
                        for m in range(block_size[0]):
                            for n in range(block_size[1]):
                                matrix_correction[block_size[0] * i + m][block_size[1] * j + n] = correction[i][j]
            matrix_correction =slm.pad_border(matrix_correction, (1024, 1272))


            mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
            self.correction_phase_1 = r"C:\Users\RiceT\Documents\SLM_computation\images\phase_mask_0.bmp"
            self.correction_phase_2 = r"C:\Users\RiceT\Documents\SLM_computation\images\phase_mask_120.bmp"
            self.correction_phase_3 = r"C:\Users\RiceT\Documents\SLM_computation\images\phase_mask_240.bmp"
            ena=np.zeros(3)
            data_fit = np.zeros((3000,4000))
            x_data = []
            tilt_data_x = []
            y_data = []
            tilt_data_y = []
        except Exception as e:
            import traceback
            print('scan', e)
            traceback.print_exc()
        self.correction_phase_1=r'C:/Users/RiceT/Documents/SLM_computation/images/phase_mask_0.bmp'
        self.correction_phase_2 = r'C:/Users/RiceT/Documents/SLM_computation/images/phase_mask_120.bmp'
        self.correction_phase_3 = r'C:/Users/RiceT/Documents/SLM_computation/images/phase_mask_240.bmp'
        number_x = list(range(point_x))
        random.shuffle(number_x)
        number_y = list(range(point_y))
        random.shuffle(number_y)
        for i in range(point_x):
            # time.sleep(5)
            for j in range(point_y):
                print('em')
                try:
                    print('ena0')
                    shift = [-number_y[j]*step_y+deflection_upperright_y, -number_x[i]*step_x+deflection_upperright_x]
                    print(shift)
                    grady = mod.gradient(mod.beams[0][1], angle=shift[0] * np.pi / 180, axis=0, wavelength=wavelength)
                    gradx = mod.gradient(mod.beams[0][1], angle=shift[1]* np.pi / 180, axis=1, wavelength=wavelength)
                    mod.add_phase(grady)
                    mod.add_phase(gradx)
                    mod.phase = mod.phase + matrix_correction
                    mod.phase = mod.phase[512-int(area_size[0]/2):512+int(area_size[0]/2), 636-int(area_size[1]/2):636+int(area_size[1]/2)]
                    mod.phase = slm.pad_border(mod.phase, (1024, 1272))
                    # temnm = mod.temnm(mod.beams[0][1], n=1, m=1)
                    print('ena1')

                    mod.phaseToBMP(mod.phase, name='phase_mask_0', color=False, correction=False, wavelength=411)
                   
                    mod.phase = mod.phase + 120 / 180 * np.pi
                    mod.phaseToBMP(mod.phase, name='phase_mask_120', color=False, correction=False, wavelength=411)
                    mod.phase = mod.phase + 120 / 180 * np.pi
                    mod.phaseToBMP(mod.phase, name='phase_mask_240', color=False, correction=False, wavelength=411)
                    mod.add_phase(-mod.phase) #Clears the mod.phase array for the next iteration
                    print('ena2')
                    cooridnates=[0,0]
                    r_square=0
                    for p in range(1):
                        if 1:# r_square<0.95 and cooridnates[0]!=1844:

                            self.load_already = np.zeros(3)
                            self.sent_phase_scan_MDS(1)

                            time.sleep(0.3)
                            self.flir.acquire_continue()
                            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
                            now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(0)
                            name = f"image_{now}.bmp"
                            path = os.path.join(directory, name)
                            self.flir.file_save(path)

                            self.sent_phase_scan_MDS(2)

                            time.sleep(0.3)
                            self.flir.acquire_continue()
                            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
                            now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(120)

                            name = f"image_{now}.bmp"
                            path = os.path.join(directory, name)
                            self.flir.file_save(path)
                            self.sent_phase_scan_MDS(3)
                            time.sleep(0.3)
                            self.flir.acquire_continue()
                            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction2'
                            now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(240)

                            name = f"image_{now}.bmp"
                            path = os.path.join(directory, name)
                            self.flir.file_save(path)

                            path = findfilepath2(shift)
                            data1 = (np.array(im.open(path[0]), dtype=np.float64))
                            data2 = (np.array(im.open(path[1]), dtype=np.float64))
                            data3 = (np.array(im.open(path[2]), dtype=np.float64))

                            # p_real = np.zeros((3000, 4000), dtype=np.float64)
                            p_real = data1 + data1 - data2 - data3
                            p_image = data2 - data3

                            p = p_real + 1j * p_image * np.sqrt(3)

                            phase_amps= data1 + data2 + data3
                            index = np.argmax(phase_amps)
                            cooridnates = np.unravel_index(index, phase_amps.shape)
                            # print(cooridnates)
                            phase = np.angle(p)
                            # print('ena4')


                            radius=100
                            waist1= int(radius)
                            waist2 = int(radius)
                            if cooridnates[0]<radius:
                                waist1 = math.floor(cooridnates[0])
                            if cooridnates[0]>3000-radius:
                                waist1 = math.floor(3000-cooridnates[0])
                            if cooridnates[1]<radius:
                                waist2 = math.floor(cooridnates[1])
                            if cooridnates[1]>4000-radius:
                                waist2 = math.floor(4000-cooridnates[1])
                            waist=np.min([waist1,waist2])
                            # print(waist,waist1,waist2)
                            # print(cooridnates)
                            # plt.figure(figsize=(8, 6))
                            # plt.imshow(phase_amps, cmap='viridis', interpolation='nearest')
                            # plt.colorbar()
                            # plt.show()
                            #
                            # plt.figure(figsize=(8, 6))
                            # plt.imshow(phase, cmap='viridis', interpolation='nearest')
                            # plt.colorbar()
                            # plt.show()

                            waist=int(waist/2)
                            phase_unwrap=(phase[cooridnates[0]-waist:cooridnates[0]+waist,cooridnates[1]-waist:cooridnates[1]+waist])
                            # plt.figure(figsize=(8, 6))
                            # plt.imshow((phase[cooridnates[0] - waist:cooridnates[0] + waist,
                            #            cooridnates[1] - waist:cooridnates[1] + waist]), cmap='viridis', interpolation='nearest')
                            # plt.colorbar()
                            # plt.show()
                            x = np.arange(phase_unwrap.shape[0])
                            y = np.arange(phase_unwrap.shape[1])
                            X, Y = np.meshgrid(x, y)
                            X_flat = X.flatten()
                            Y_flat = Y.flatten()
                            phase_flat = phase_unwrap.flatten()
                            features = np.vstack((X_flat, Y_flat)).T
                            model = LinearRegression()
                            model.fit(features, phase_flat)
                            phase_predict = model.predict((features))
                            r_square = r2_score(phase_unwrap.flatten(), phase_predict)
                            a = model.coef_[0]
                            b = model.coef_[1]
                            print(a, b,'r_square=',r_square)



                            if r_square > 0.82 and r_square < 1:
                                # plt.figure(figsize=(8, 6))
                                # plt.imshow(phase_unwrap, cmap='viridis', interpolation='nearest')
                                # plt.colorbar()
                                # plt.title('measure')
                                # plt.figure(figsize=(8, 6))
                                # plt.imshow(phase_predict.reshape(X.shape), cmap='viridis', interpolation='nearest')
                                # plt.colorbar()
                                # plt.title('fit')
                                # plt.show()
                                data_fit[cooridnates[0] - waist:cooridnates[0] + waist,
                                cooridnates[1] - waist:cooridnates[1] + waist] = \
                                    phase[cooridnates[0] - waist:cooridnates[0] + waist,
                                    cooridnates[1] - waist:cooridnates[1] + waist]



                                x_data.append(cooridnates[1])
                                tilt_data_x.append(a)
                                y_data.append(cooridnates[0])
                                tilt_data_y.append(b)
                                print(a, b)


                except Exception as e:
                    import traceback
                    print('scan', e)
                    traceback.print_exc()
        #phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\fit_data.txt'
        phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\fit_data_MDS.txt'
        np.savetxt(phase_path, np.array(data_fit))

        plt.figure(figsize=(8, 6))
        plt.imshow(data_fit, cmap='viridis', interpolation='nearest')
        plt.colorbar()


        # z_data = []
        try:

            tilt_data_x = np.array(tilt_data_x)
            x_data = np.array(x_data)

            popt, pcov = curve_fit(func, x_data, tilt_data_x, p0=[-1 * 10 ** (-5), 0], maxfev=10000)
            self.a_fit, self.b_fit = popt
            y_fit = func(x_data, *popt)

            # 计算 R²
            residuals = tilt_data_x - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_total = np.sum((tilt_data_x - np.mean(tilt_data_x)) ** 2)
            r_squared = 1 - (ss_res / ss_total)

            # 计算 RMSE
            rmse = np.sqrt(np.mean(residuals ** 2))

            # 打印 R² 和 RMSE
            print(f"fit para a: {self.a_fit}, b: {self.b_fit}")
            print(f"R²: {r_squared}, RMSE: {rmse}")

            tilt_data_y = np.array(tilt_data_y)
            y_data = np.array(y_data)
            popt, pcov = curve_fit(func, y_data, tilt_data_y, p0=[-1 * 10 ** (-5), 1500], maxfev=10000)
            self.c_fit, self.d_fit = popt
            y_fit = func(y_data, *popt)

            # 计算 R²
            residuals = tilt_data_y - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_total = np.sum((tilt_data_y - np.mean(tilt_data_y)) ** 2)
            r_squared = 1 - (ss_res / ss_total)

            # 计算 RMSE
            rmse = np.sqrt(np.mean(residuals ** 2))

            # 打印 R² 和 RMSE
            print(f"fit para c: {self.c_fit}, d: {self.d_fit}")
            print(f"R²: {r_squared}, RMSE: {rmse}")
            for i in range(3000):
                for j in range(4000):
                    if data_fit[i][j] != 0:
                        data_fit[i][j] = ( self.a_fit * (j - self.b_fit) ** 2 / 2 +self.c_fit * (
                                i - self.d_fit) ** 2 / 2) % (
                                                 2 * np.pi)
            plt.figure(figsize=(8, 6))
            plt.imshow(data_fit, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('fit')
            plt.show()


            # for y in y_data:
            #     x = []
            #     intensity_plot = []
            #     field_plot = []
            #     for i in range(data_fit.shape[1]-1):
            #         x.append(i)
            #         intensity_plot.append(data_fit[y][i+1]-data_fit[y][i])
            #
            #         field_plot.append(self.a_fit * (y - self.b_fit)  + self.c_fit * (
            #                          i - self.d_fit))
            #     plt.plot(x, intensity_plot, label='measure', color='blue')
            #     plt.plot(x, field_plot, label='expect', color='red')
            #     plt.show()
            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\data_fit.txt'

            np.savetxt(phase_path, data_fit)


        except Exception as e:
            import traceback
            print('check', e)
            traceback.print_exc()



                
    def scan(self):
        # slm optical abberration correct
        # generating one moving patch and one stable centered patch, observing the interferrence fringe
        # finally get phase pattern from interferrence
        from ArrayModulator_v1 import ArrayModulator, load_mod
        import slm_v1 as slm
        if not self.connect:

            self.flir = FlirCamController_slm.FlirCamController()
            self.flir.initialize()
            self.flir.start_continue()
            self.flir.set_average_frames(1)
            self.connect = True
        # print('em')
        try:
            wavelength = 411
            mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
            grady = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_y.text()) * np.pi / 180, axis=0,
                                 wavelength=wavelength)
            gradx = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_x.text()) * np.pi / 180, axis=1,
                                 wavelength=wavelength)

            phase = gradx + grady
            #setting deflection
            centre = [int(phase.shape[0]), int(phase.shape[1])]
            block_size = [int(self.text_patch_size_x.text()), int(self.text_patch_size_y.text())]
            centre = [int((phase.shape[0]) / 2), int((phase.shape[1]) / 2)]
            phase_slice = slm.pad_border(phase[centre[0] - block_size[0]:centre[0] + block_size[0] + 1,
                                         centre[1] - block_size[1]:centre[1] + block_size[1] + 1], (1024, 1272))
            #phase_slice is th stable patch in the center
            target_shape = (1024, 1272)
            target_center = np.array(target_shape) / 2
            ena = np.zeros(3)

            length_y=int(int(self.text_scan_area_y.text())/int(self.text_patch_size_y.text()))
            length_x = int(int(self.text_scan_area_x.text()) / int(self.text_patch_size_x.text()))
            # print(length_x)
            number_x = list(range(length_x))
            random.shuffle(number_x)
            number_y = list(range(length_y))
            area_size=[int(int(self.text_scan_area_x.text())/2),int(int(self.text_scan_area_y.text())/2)]
            random.shuffle(number_y)
        except Exception as e:
            print('error', e)

        radius=((length_x**2)/4)
        print('radius=',radius)
        for i in range(length_x):
            for j in range(length_y):
                if 1:#((i-length_x/2)**2+(j-length_y/2)**2) < radius+40: #1:# MDS
                    try:

                        shift = [-number_x[i] * block_size[0] + area_size[0], - number_y[j] * block_size[1] + area_size[1]]
                        output = np.zeros(shape=target_shape)
                        phase_center = [shift[0], shift[1]]
                        phase_shift_0 = phase[centre[0] - block_size[0] + shift[0]:centre[0] + block_size[0] + 1 + shift[0],
                                        centre[1] - block_size[1] + shift[1]:centre[1] + block_size[1] + 1 + shift[1]]

                        # phase_shift_0 the moving patch
                        phi = 0 #phi is 0,120,240, phi is used for phase extraction
                        phase_shift = phase_shift_0 + phi
                        output[:phase_shift.shape[0], :phase_shift.shape[1]] = phase_shift

                        output = np.roll(output, np.array(target_center - phase_center - block_size, dtype=np.int64),
                                         axis=(0, 1))
                        phase_rota = phase_slice + output
                        name1 = f'correction_{shift[0]}_{shift[1]}_shift_{phi}.bmp'
                        mod.add_phase(phase_rota)
                        mod.phaseToBMP_correct(mod.phase, name=f'correction_{shift[0]}_{shift[1]}_shift_{phi}', color=False,
                                       correction=False, wavelength=411)
                        mod.add_phase(-mod.phase)
                        directory = r"C:\Users\RiceT\Documents\SLM_computation\correction1"
                        self.correction_phase_1 = os.path.join(directory, name1)


                        phi = 120
                        phase_shift = phase_shift_0 + phi / 180 * np.pi
                        # print(np.max(phase_shift))
                        output = np.zeros(shape=target_shape)
                        output[:phase_shift.shape[0], :phase_shift.shape[1]] = phase_shift
                        output = np.roll(output, np.array(target_center - phase_center - block_size, dtype=np.int64),
                                         axis=(0, 1))

                        phase_rota = phase_slice + output
                        name2 = f'correction_{shift[0]}_{shift[1]}_shift_{phi}.bmp'
                        mod.add_phase(phase_rota)
                        mod.phaseToBMP_correct(mod.phase, name=f'correction_{shift[0]}_{shift[1]}_shift_{phi}', color=False,
                                       correction=False, wavelength=411)
                        mod.add_phase(-mod.phase)
                        directory = r"C:\Users\RiceT\Documents\SLM_computation\correction1"
                        self.correction_phase_2 = os.path.join(directory, name2)
                        phi = 240
                        phase_shift = phase_shift_0 + phi / 180 * np.pi
                        # print(np.max(phase_shift))
                        output = np.zeros(shape=target_shape)
                        output[:phase_shift.shape[0], :phase_shift.shape[1]] = phase_shift

                        output = np.roll(output, np.array(target_center - phase_center - block_size, dtype=np.int64),
                                         axis=(0, 1))
                        phase_rota = phase_slice + output
                        mod.add_phase(phase_rota)
                        name3 = f'correction_{shift[0]}_{shift[1]}_shift_{phi}.bmp'
                        mod.phaseToBMP_correct(mod.phase, name=f'correction_{shift[0]}_{shift[1]}_shift_{phi}', color=False,
                                       correction=False, wavelength=411)
                        directory = r"C:\Users\RiceT\Documents\SLM_computation\correction1"
                        self.correction_phase_3 = os.path.join(directory, name3)
                        mod.add_phase(-mod.phase)
                    except Exception as e:
                        print('error', e)
                    print(shift)

                    self.load_already = np.zeros(3)
                    self.sent_phase_scan_MDS(1)
                    time.sleep(0.2)#0.3
                    self.flir.acquire_continue()
                    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
                    now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(0)
                    name = f"image_{now}.bmp"
                    path = os.path.join(directory, name)
                    self.flir.file_save(path)
                    self.sent_phase_scan_MDS(2)


                    time.sleep(0.2)#0.3
                    self.flir.acquire_continue()
                    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
                    now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(120)

                    name = f"image_{now}.bmp"
                    path = os.path.join(directory, name)
                    self.flir.file_save(path)
                    self.sent_phase_scan_MDS(3)
                    time.sleep(0.2)#0.3
                    self.flir.acquire_continue()
                    directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
                    now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(240)

                    name = f"image_{now}.bmp"
                    path = os.path.join(directory, name)
                    self.flir.file_save(path)
                    #Added MDS, delete patch files after taking data

                    for phidelete in (0,120,240):
                        deletename = f'correction_{shift[0]}_{shift[1]}_shift_{phidelete}.bmp'
                        deletedirectory = r"C:\Users\RiceT\Documents\SLM_computation\correction1"
                        file_to_delete = os.path.join(deletedirectory, deletename)
                        #file_to_delete = "my_file.txt"

                        # Check if the file exists before attempting to delete it
                        if os.path.exists(file_to_delete):
                            os.remove(file_to_delete)
                            print(f"File '{file_to_delete}' deleted successfully.")
                        else:
                            print(f"File '{file_to_delete}' does not exist.")


                    #if the fitting is not good ,then take picture again
                    if False:#np.abs(shift[0]) + np.abs(shift[1]) > 60: ###MDS changed this.
                    # if moving patch and center patch too close,the fit will not be good

                        fail=0
                        while camera_success(shift) < 0.98:
                            #teel if it's good or not by r_square
                            #if not good ,take picture again
                            fail=fail+1
                            if fail>6:
                                break
                                #try 6 times
                            print(shift, 'failed')
                            self.load_already = np.zeros(3)
                            self.sent_phase_scan_MDS(1)
                            time.sleep(0.3)
                            self.flir.acquire_continue()
                            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
                            now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(0)
                            name = f"image_{now}.bmp"
                            path = os.path.join(directory, name)
                            self.flir.file_save(path)
                            self.sent_phase_scan_MDS(2)
                            time.sleep(0.3)
                            self.flir.acquire_continue()
                            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
                            now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(120)
                            name = f"image_{now}.bmp"
                            path = os.path.join(directory, name)
                            self.flir.file_save(path)
                            self.sent_phase_scan_MDS(3)
                            time.sleep(0.3)
                            self.flir.acquire_continue()
                            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\correction1'
                            now = str(shift[0]) + '_' + str(shift[1]) + '_' + str(240)
                            name = f"image_{now}.bmp"
                            path = os.path.join(directory, name)
                            self.flir.file_save(path)
    def phase_generation(self):
        #button 'phase generate'
        try:
            length_y = int(int(self.text_scan_area_y.text()) / int(self.text_patch_size_y.text()))
            length_x = int(int(self.text_scan_area_x.text()) / int(self.text_patch_size_x.text()))
            block_size = [int(self.text_patch_size_x.text()), int(self.text_patch_size_y.text())]
            area_size = [int(int(self.text_scan_area_x.text())), int(int(self.text_scan_area_y.text()) )]
            from ArrayModulator_v1 import ArrayModulator, load_mod
            import slm_v1 as slm
            wavelength=411
            mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % wavelength)
            phase_path = r'C:\Users\RiceT\Documents\FlirCamera\pictures\correction_phase.txt'
            correction = np.loadtxt(phase_path)
            grady = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_y.text()) * np.pi / 180, axis=0, wavelength=wavelength)
            gradx = mod.gradient(mod.beams[0][1], angle=float(self.text_deflection_x.text()) * np.pi / 180, axis=1, wavelength=wavelength)
            # print(correction.shape)
            temnm = mod.temnm(mod.beams[0][1], n=int(self.text_tem_n.text()), m=int(self.text_tem_m.text()))
            mod.add_phase(grady)
            mod.add_phase(gradx)
            matrix_correction = np.zeros((area_size[0], area_size[1]))#from slm scan
            radius = ((length_x ** 2) / 4)
            # print('radius=', radius)
            #set correction to be round
            for i in range(length_x):
                for j in range(length_y):
                    if ((i - length_x / 2) ** 2 + (j - length_y / 2) ** 2) < radius:
                        for m in range(block_size[0]):
                            for n in range(block_size[1]):
                                matrix_correction[block_size[1] * i + m][block_size[0] * j + n] = correction[i][j]
            if self.checkbox_array_generation.isChecked():
                n = int(self.text_array_number_1.text())
                m = int(self.text_array_number_2.text())
                #set relative phase for compensation
                bungu = self.text_erroramps.text()
                start = 0
                end = 0
                self.relative_intensity = np.zeros(n*m)
                for read_num in range(n*m):
                    while end != len(bungu) and bungu[end] != ',':
                        end = end + 1
                    print(bungu[start:end])
                    self.relative_intensity[read_num] = float(bungu[start:end])
                    end = end + 1
                    start = end
                amps=tuple(0.+self.relative_intensity[i]  for i in range(n * m))
                bungu = self.text_errorphase.text()
                start = 0
                end = 0
                self.relative_phase = np.zeros(n*m)
                for read_num in range(n*m):
                    while end != len(bungu) and bungu[end] != ',':
                        end = end + 1
                    print(bungu[start:end])
                    self.relative_phase[read_num] = float(bungu[start:end])
                    end = end + 1
                    start = end
                phases =tuple(0.+self.relative_phase[i]  for i in range(n * m))
                if int(self.text_tem_n.text()) != 0:
                    tem01 = True
                    phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01.txt'
                    #phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\tem01withT.txt'
                    phase_mask_tem01 = np.loadtxt(phase_path)
                    #mod.add_phase(np.array(phase_mask_tem01))
                    tem01 = True
                else:
                    tem01 =False

                amps_guess = amps
                print(amps)

                #MDS #Target curve compensation
                a = -4.1734e-06#-4.03456842e-06#-3.733022642775838e-06
                b = 2070#2071
                c = -4.38394e-06#-3.9478368e-06#-3.893853882563125e-06A
                d = 1672#1712
                target_phase = np.zeros((1024, 1272))
                for i in range(1024):
                    for j in range(1272):
                        # target_phase[i][j]=c*((i-512)*104+1500-d)**2/2+a*((j-636)*84+2000-b)**2/2
                        #target_phase[i][j] = c * ((i - 511) * 53.187*1.13 + 1573.195 - d) ** 2 / 2 + a * (
                        #            (j - 635.5) * 41.045*1.13 + 1937.23 - b) ** 2 / 2  # MDS
                        target_phase[i][j] = c * ((i - 511) * 53.651 + 1672 - d) ** 2 / 2 + a * (
                                    (j - 635.5) * 41.4555 + 2070 - b) ** 2 / 2  # MDS Enter center of the beam here
                target_phase = target_phase[462:562, 586:686]
                target_phase = slm.pad_border(target_phase, (1024, 1272))
                #MDS
                #phase_tem_compensate added #phase_tem_compensation=np.exp(1j * np.array(target_phase)),
                # wu_1x4 = mod.wu_algorithm(n=len(amps), M=30, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases, x_pitch=0.01, plots=True, res_factor=1, input_profile=profile.Profile.input_gaussian(beam_size=(0.4*0.8, 0.4), size=np.array(mod.size)), phase_memory=True)
                wu_1x4 = mod.wu_algorithm2D(n=n, m=m,phase_tem_compensation=np.exp(1j * np.array(target_phase)),
                                            M=35, name='wu_1x4', amps=amps, amps_guess=amps_guess, phases=phases,
                                            x_pitch=float(self.text_x_pitch.text()), plots=False,
                                            input_profile=profile.Profile.input_gaussian(beam_size=(0.35, 0.35),
                                                                                         size=np.array(mod.size)),
                                            phase_memory=True, tem01=tem01)
                # wu_1x4 = phase_rotate(wu_1x4, 6.5 / 180 * np.pi)
                mod.add_phase(wu_1x4)
            else:
                mod.add_phase(temnm)
            mod.phase = mod.phase + slm.pad_border(matrix_correction, (1024, 1272))
            mod.phase = mod.phase[512-int(area_size[0]/2):512+int(area_size[0]/2), 636-int(area_size[1]/2):636+int(area_size[1]/2)]
            mod.phase =slm.pad_border(mod.phase, (1024, 1272))
            for m in range(int(area_size[0])):
                for n in range(int(area_size[1])):
                    if np.sqrt((m-int(area_size[0]/2))**2+(n-int(area_size[1]/2))**2)>int(area_size[0]/2):
                        mod.phase[512-int(area_size[0]/2)+m][636-int(area_size[1]/2)+n]=0
            phase_offset = np.pi / 4
            mod.phase = mod.phase + phase_offset
            mod.phaseToBMP(mod.phase, name='phase_mask_0', color=False, correction=False, wavelength=411)
            mod.phase = mod.phase + 120 / 180 * np.pi
            mod.phaseToBMP(mod.phase, name='phase_mask_120', color=False, correction=False, wavelength=411)
            mod.phase = mod.phase + 120 / 180 * np.pi
            mod.phaseToBMP(mod.phase, name='phase_mask_240', color=False, correction=False, wavelength=411)
            phase_path = r'C:\Users\RiceT\Documents\SLM_computation\images\phaseGUI.txt'
            np.savetxt(phase_path, mod.phase)
        except Exception as e:
            import traceback
            print('generation', e)
            traceback.print_exc()


    def calculate_average_image(self,image_paths):
    #not using now, to do average for images in the path
        avg_image = None
        count = 0
        try:
            for image_path in image_paths:
                print("imagepath:",image_path)

                img = Image.open(image_path)
                #img = img.convert('L')
                img_array = np.array(img)
                if avg_image is None:
                    avg_image = img_array
                else:
                    avg_image= avg_image/(count+1)*count+img_array/(count+1)

                count += 1
            if count > 0:
                try:
                    avg_image = avg_image.astype(np.uint8)  # Convert back to uint8 format
                except Exception as e:
                    print('error of average',e)
        except Exception as e:
            print('error of average', e)
        return avg_image

    def compensate(self):
    #button 'data analyze'
        from slm_v1 import SLM,pad_border
        try:

            num_x = int(self.text_array_number_1.text())
            num_y = int(self.text_array_number_2.text())
            if int(self.text_tem_n.text())!=0:# for tem01, beam number=n*m*2
                num_x=num_x*2
            paths = [self.filename_phase_0.text(),
                     r"C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image1.bmp",
                     r"C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image2.bmp",
                     r"C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image3.bmp"]
            slm = SLM()
            sigma = 1
            import PhaseExtraction_v1 as PhaseExtraction
            profs = PhaseExtraction.import_profs(slm=slm, paths=paths, name='')
            intensity = cv2.GaussianBlur(profs[0], (0, 0), sigmaX=sigma, sigmaY=sigma)
            data1 = cv2.GaussianBlur(profs[1], (0, 0), sigmaX=sigma, sigmaY=sigma)
            data2 = cv2.GaussianBlur(profs[2], (0, 0), sigmaX=sigma, sigmaY=sigma)
            data3 = cv2.GaussianBlur(profs[3], (0, 0), sigmaX=sigma, sigmaY=sigma)
            p_real = data1 + data1 - data2 - data3
            p_image = data2 - data3
            p = p_real + 1j * p_image * np.sqrt(3)
            phase = np.angle(p)  # -np.heaviside(p_real,0)*np.pi
            phase=unwrap_phase(phase)
            image_amp = intensity
            area=search_x(image_amp, num_x,int(self.text_tem_n.text()),[int(self.text_frame_set_1.text()),self.text_frame_set_2.text()])
            print("areaMDS",area)#mds
            plt.imshow(image_amp)
            plt.title("image_amp")
            plt.show()
            area_new=[]
            for k in range(num_x):
                memory=(search_y(image_amp[:, int(area[k][2]):int(area[k][3])],num_y))
                area_new.append([])
                for l in range(num_y):
                    area_new[k].append([memory[l][2],memory[l][3],int(area[k][2]),int(area[k][3])])
            area=area_new
            #area is the area slice for each beam
            print('area',area_new)
            waist = []
            for k in range(num_x):
                waist.append([])
                for l in range(num_y):
                    slice = image_amp[int(area[k][l][0]):int(area[k][l][1]), int(area[k][l][2]):int(area[k][l][3])]
                    index = np.argmax(slice)
                    cooridnates = np.unravel_index(index, slice.shape)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(slice, cmap='viridis', interpolation='nearest')
                    plt.colorbar()
                    plt.title('beam slice')
                    plt.show()

                    positiony = int(cooridnates[0])
                    positionx = int(cooridnates[1])
                    waist[k].append([positiony, positionx, 50, 50])
                    #waist radius can adjusted, now 50 is just for convinience


            phase_list = []
            intensity = []
            centre_x=[]
            centre_y = []
            for k in range(num_x):
                phase_list.append([])
                for l in range(num_y):
                    #I choose the20*20 area around the center to do average to get the intensity and phase for each beam
                    slice_inten = image_amp[
                                      int(area[k][l][0]) + int(waist[k][l][0]) - int(waist[k][l][2]/5):int(area[k][l][0]) + int(
                                          waist[k][l][0]) + int(
                                          waist[k][l][2]/5),
                                      int(area[k][l][2]) + int(waist[k][l][1]) - int(waist[k][l][3]/ 5):int(area[k][l][2]) + int(
                                          waist[k][l][1]) + int(
                                          waist[k][l][3] / 5)]
                    phi_ = phase[
                           int(area[k][l][0]) + int(waist[k][l][0]) - int(waist[k][l][2] / 5):int(
                               area[k][l][0]) + int(
                               waist[k][l][0]) + int(
                               waist[k][l][2] / 5),
                           int(area[k][l][2]) + int(waist[k][l][1]) - int(waist[k][l][3] / 5):int(
                               area[k][l][2]) + int(
                               waist[k][l][1]) + int(
                               waist[k][l][3] /5)]
                    phi_ = unwrap_phase(phi_)
                    phi_max = np.mean(phi_)
                    print('x=',int(area[k][l][2]) + int(waist[k][l][1]),'y=',int(area[k][l][0]) + int(waist[k][l][0]))
                    centre_x.append(int(area[k][l][2]) + int(waist[k][l][1]))
                    centre_y.append(int(area[k][l][0]) + int(waist[k][l][0]))

                    phase_list[k].append(phi_max)
                    intensity.append(1 / (np.mean(slice_inten)))

                    # plt.figure(figsize=(8, 6))
                    # plt.imshow(phi_, cmap='viridis', interpolation='nearest')
                    # plt.colorbar()
                    # plt.title('tilt')
                    # plt.show()

            if 1:#for subtraction of phase wavefront it is not used now
                a_fit=  0
                b_fit=  self.b_fit
                c_fit=0
                d_fit=self.d_fit
            # print(unwrap_phase(np.array(phase_list)))
            for k in range(num_x):
                for l in range(num_y):
                    phase_list[k][l]=phase_list[k][l]-a_fit/2*(int(area[k][l][2]) + int(waist[k][l][1])- b_fit)**2-c_fit/2*(int(area[k][l][0]) + int(waist[k][l][0])- d_fit)**2
            intensity=np.array(intensity)
            print('intensity=',intensity)
            phase_list=np.array(phase_list)%(2*np.pi)
            for i in range(phase.shape[0]):
                for j in range(phase.shape[1]):
                    phase[i][j]=phase[i][j]-a_fit/2*(j- b_fit)**2-c_fit/2*(i- d_fit)**2
            phase=((phase)%(2*np.pi))
            print((np.array(phase_list)))
            # plt.figure(figsize=(8, 6))
            # plt.imshow(phase, cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            # plt.title('phase')
            # plt.show()

            #plot complex elctric field with rgb image
            magnitude_normalized = (image_amp - np.min(image_amp) )/ (np.max(image_amp) - np.min(image_amp))
            self.amp_extracted=magnitude_normalized
            self.phase_extracted=phase
            phase_normalized = (phase ) / (2 * np.pi)
            hue = phase_normalized
            brightness = magnitude_normalized
            matrix_size = (3000, 4000)
            rgb_image = np.zeros((*matrix_size, 3))
            rgb_image[..., 0] =  brightness *hue #red
            rgb_image[..., 1] = 0 # green
            rgb_image[..., 2] = brightness *(1-hue)# blue
            figure = self.phase_plot2
            figure.axes.imshow(phase/2/np.pi, cmap='hsv')
            if self.mark_for_slice == 0:# add colorbar only at first time
                figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes,
                                    label='$2\\pi$ radians')
            figure.axes.set_title('phase_map')
            figure.draw()
            figure = self.phase_plot
            figure.axes.imshow(rgb_image, cmap='hsv')
            figure.axes.set_title('phase_intensity_map')
            figure.draw()


            phase_list = (phase_list - np.mean(phase_list))
            print(phase_list)
            #change the scale of phase into (*2pi)
            for i in range(num_x):
                for j in range(num_y):
                    if np.abs((phase_list[i][j]+2*np.pi)%(2*np.pi))>np.abs((phase_list[i][j])):
                        phase_list[i][j]=phase_list[i][j]/np.pi/2
                    else:
                        phase_list[i][j] = ((phase_list[i][j]+2*np.pi)%(2*np.pi))  / np.pi / 2

            index = np.argmax(image_amp)
            cooridnates = np.unravel_index(index, image_amp.shape)

            x = []
            intensity_plot = []
            field_plot = []
            phase_plot = []
            intensity_plot_2 = []
            field_plot_2 = []
            phase_plot_2 = []
            count=0
            length=num_y*num_x
            print('len=',length)
            print(centre_x)
            #get 1d plot and ignore the area out of radius
            for i in range(image_amp.shape[1]):
                x.append(i)
                intensity_plot.append(image_amp[centre_y[0]][i])
                intensity_plot_2.append(image_amp[centre_y[1]][i])
                if count< length-2 and i>centre_x[count+1]:
                    count=count+1

                if (np.abs(i-centre_x[count])<100 or np.abs(i-centre_x[count+1])<100) and magnitude_normalized[centre_y[0]][i]>0.2: #MDS using 100 instead of 180 and added the amplitude condition
                    phase_plot.append(phase[centre_y[0]][i]/np.pi)
                    field_plot.append(np.sqrt(image_amp[centre_y[0]][i]) * np.cos(phase[centre_y[0]][i]))
                    phase_plot_2.append(phase[centre_y[1]][i] / np.pi)
                    field_plot_2.append(np.sqrt(image_amp[centre_y[1]][i]) * np.cos(phase[centre_y[1]][i]))
                else:#set plot to be zero whenit's out of radius
                    phase_plot.append(0)
                    field_plot.append(0)
                    phase_plot_2.append(0)
                    field_plot_2.append(0)
            figure = self.amplitude_plot

            if self.mark_for_slice != 0:
                figure.fig.clf()
                figure = self.amplitude_plot
                figure.axes = figure.fig.add_subplot(111, yscale="linear")
            figure.axes.plot(x, intensity_plot, label='intensity_upper', color='blue')
            if num_y!=0:
                figure.axes.plot(x, intensity_plot_2, label='intensity_lower', color='red')
            figure.axes.set_xlabel('pixel')
            figure.axes.set_ylabel('intensity')
            figure.axes.set_title('beam intensity plot')
            figure.axes.legend(loc='upper left')
            figure.draw()

            figure = self.tranverse_field_plot

            if self.mark_for_slice != 0:
                figure.fig.clf()
                figure = self.tranverse_field_plot
                figure.axes = figure.fig.add_subplot(111, yscale="linear")
            figure.axes.plot(x, field_plot, label='field_upper', color='blue')
            if num_y!=0:
                figure.axes.plot(x, field_plot_2, label='field_lower', color='red')
            figure.axes.set_xlabel('pixel')
            figure.axes.set_ylabel('electrical field')
            figure.axes.set_title('tranverse electric field_plot')
            figure.axes.legend(loc='upper left')
            figure.draw()

            figure = self.phase_1d_plot

            if self.mark_for_slice != 0:
                figure.fig.clf()
                figure = self.phase_1d_plot
                figure.axes = figure.fig.add_subplot(111, yscale="linear")
            figure.axes.plot(x, phase_plot, label='phase_upper', color='blue')
            if num_y!=0:
                figure.axes.plot(x, phase_plot_2, label='phase_lower', color='red')
            figure.axes.set_xlabel('pixel')
            figure.axes.set_ylabel('extracted phase(/pi)')
            figure.axes.set_title('1d phase plot')
            figure.axes.legend(loc='upper left')
            figure.draw()
            print(phase_list)
            phase_show=(phase_list*2-2*phase_list[0][0]+2)%(2)
            figure = self.column_plot

            width=0.35
            if self.mark_for_slice != 0:
                figure.fig.clf()
                figure = self.column_plot
                figure.axes = figure.fig.add_subplot(111, yscale="linear")
            target_phase=np.array(self.target_phase)*2
            if int(self.text_tem_n.text())!=0:
                target_phase_show = np.repeat(target_phase, 2, axis=0).flatten()
            else:
                target_phase_show = target_phase.flatten()
            phase_show = np.array(phase_show).T.flatten()
            if int(self.text_tem_n.text()) != 0:
                count = 0
                for ena in target_phase_show:
                    if count % 2 != 0:
                        target_phase_show[count] = target_phase_show[count] + 1
                    count = count + 1
            x_plot = np.arange(len(target_phase_show))


            target_phase_show = (target_phase_show - target_phase_show[0] + 2) % 2
            for i in range(len(target_phase_show)):
                if (phase_show[i]-target_phase_show[i])>1 :
                    phase_show[i]=phase_show[i]-2
                if (phase_show[i]-target_phase_show[i])<-1 :
                    phase_show[i]=phase_show[i]+2
            print('phase_show=',phase_show)
            bars1=figure.axes.bar(x_plot-width/2, phase_show,width,color='b')

            bars2=figure.axes.bar(x_plot + width/2, target_phase_show,width,color='r')
            for bar in bars1:
                yval1=round(bar.get_height(),2)
                if yval1>0:
                    offset=0.5

                figure.axes.text(bar.get_x()+bar.get_width()/2,yval1,yval1,ha='center',va='bottom' if yval1>=0 else 'top')
            for bar in bars2:
                yval1=round(bar.get_height(),2)
                figure.axes.text(bar.get_x()+bar.get_width()/2,yval1,yval1,ha='center',va='bottom'if yval1>=0 else 'top')

            figure.axes.set_ylabel('extracted phase(/pi)')
            figure.axes.set_title('1d phase plot')
            figure.axes.set_xticks(x_plot)
            figure.draw()


            if self.mark_for_slice==0:


                self.mark_for_slice=1
            #     self.relative_phase =[]
            #     for i in range(num_x):
            #         for j in range(num_y):
            #             self.relative_phase.append(phase_list[i][j])
            # else:
            if int(self.text_tem_n.text()) != 0:
                phase_list = phase_list[::2, :]
                num_x=int(num_x/2)
            phase_list=phase_list-phase_list[0][0]
            bungu=self.text_erroramps.text()
            start = 0
            end = 0
            self.relative_intensity = np.zeros(num_x*num_y)
            for read_num in range(num_x*num_y):
                while end != len(bungu) and bungu[end] != ',':
                    end = end + 1
                # print(bungu[start:end])
                self.relative_intensity[read_num] = float(bungu[start:end])
                end = end + 1
                start = end
            bungu = self.text_errorphase.text()
            start = 0
            end = 0
            self.relative_phase = np.zeros(num_x*num_y)
            for read_num in range(num_x*num_y):
                while end != len(bungu) and bungu[end] != ',':
                    end = end + 1
                # print(bungu[start:end])
                self.relative_phase[read_num] = float(bungu[start:end])
                end = end + 1
                start = end
            for i in range(num_y*num_x):
                self.relative_intensity[i] = intensity[i]*self.relative_intensity[i]
            self.relative_intensity = self.relative_intensity / np.mean(self.relative_intensity)
            charamps=str(round(self.relative_intensity[0],3))
            for read_num in range(num_x*num_y-1):
                charamps =charamps+','+str(round(self.relative_intensity[read_num+1],3))
            self.text_erroramps.setText(charamps)
            print('phase_list=',phase_list)
            print('self.relative_phase=', self.relative_phase)
            
            for i in range(num_x):
                for j in range(num_y):
                    self.relative_phase[num_x*j+i]=(phase_list[i][j]-self.target_phase[num_x*j+i])+self.relative_phase[num_x*j+i]
                    self.relative_phase[num_x*j+i] =(self.relative_phase[num_x*j+i]+1)%1
            print('self.relative_phase=', self.relative_phase)
            charphase = str(round(self.relative_phase[0],3))
            for read_num in range(num_y*num_x - 1):
                charphase = charphase + ',' + str(round(self.relative_phase[read_num + 1],3))
            self.text_errorphase.setText(charphase)
            self.mark_for_compen=1
            # self.phase_generation()
        except Exception as e:
            print('compensate', e)
            import traceback
            traceback.print_exc()

    def reset_para(self):
        try:
            n = int(self.text_array_number_1.text())
            m = int(self.text_array_number_2.text())
            if self.checkbox_array_phase.isChecked():
                self.relative_phase=np.zeros(n*m)
                self.relative_intensity = np.ones(n * m)
                bungu='0'
                for i in range(n*m-1):
                    bungu=bungu+',0'
                self.text_targetphase.setText(bungu)
                self.text_errorphase.setText(bungu)
                bungu = '1'
                for i in range(n * m - 1):
                    bungu = bungu + ',1'
                self.text_targetamps.setText(bungu)
                self.text_erroramps.setText(bungu)
            else:
                bungu = self.text_targetamps.text()
                start = 0
                end = 0
                self.relative_intensity = np.zeros(n*m)
                for read_num in range(n*m):
                    while end != len(bungu) and bungu[end] != ',':
                        end = end + 1
                    # print(bungu[start:end])
                    self.relative_intensity[read_num] = float(bungu[start:end])
                    end = end + 1
                    start = end
                bungu = self.text_targetphase.text()
                start = 0
                end = 0
                self.relative_phase = np.zeros(n*m)
                for read_num in range(n*m):
                    while end != len(bungu) and bungu[end] != ',':
                        end = end + 1
                    # print(bungu[start:end])
                    self.relative_phase[read_num] = float(bungu[start:end])
                    end = end + 1
                    start = end
                # self.relative_intensity = np.ones(n * m)
                charamps = str(round(self.relative_intensity[0], 3))
                for read_num in range(n*m - 1):
                    charamps = charamps + ',' + str(round(self.relative_intensity[read_num + 1], 3))
                self.text_targetamps.setText(charamps)
                self.text_erroramps.setText(charamps)
                # self.relative_phase =  np.zeros(n * m)
                charamps = str(round(self.relative_phase[0], 3))
                for read_num in range(n * m - 1):
                    charamps = charamps + ',' + str(round(self.relative_phase[read_num + 1], 3))
                self.text_targetphase.setText(charamps)
                self.text_errorphase.setText(charamps)
            self.target_phase=self.relative_phase
            self.mark_for_compen = 0
            # self.mark_for_slice = 0
        except Exception as e:
            print('reset', e)
            import traceback
            traceback.print_exc()

    def circle(self):
        try:
            self.starttime = datetime.datetime.now().strftime("d%d_h%H_m%M_s%S")
            if not self.connect:
                self.flir = FlirCamController_slm.FlirCamController()
                self.flir.initialize()
                self.flir.start_continue()
                self.flir.set_average_frames(1)
                self.connect = True


            t0=time.time()
            self.load_already =np.zeros(3)
            ena = np.zeros(3)
            times=int(self.text_circle.text())
            self.phase=np.zeros((3000,4000))
            for loop in range(times):
                if self.checkbox_addlist_1.isChecked():
                    self.correction_phase_1=self.filename_phase_1.text()
                    self.sent_phase_scan_MDS(1)
                    print(time.time() - t0)

                    time.sleep(0.8)
                    self.take_image(1)
                print(time.time()-t0)
                if self.checkbox_addlist_2.isChecked():
                    self.correction_phase_2 = self.filename_phase_2.text()
                    self.sent_phase_scan_MDS(2)

                    time.sleep(0.3)
                    print(time.time() - t0)
                    self.take_image(2)
                print(time.time() - t0)
                if self.checkbox_addlist_3.isChecked():
                    self.correction_phase_3 = self.filename_phase_3.text()
                    self.sent_phase_scan_MDS(3)

                    time.sleep(0.3)
                    self.take_image(3)

                print(time.time() - t0)



                # p_real = np.zeros((3000, 4000), dtype=np.float64)
                # p_real = data1 + data1 - data2 - data3
                # p_image = data2 - data3
                #
                # p = p_real + 1j * p_image * np.sqrt(3)
                #
                # phase_amps = data1 + data2 + data3
                # phase = np.angle(p)
                # self.phase = self.phase+phase
                # plt.figure(figsize=(8, 6))
                # plt.imshow(self.phase, cmap='viridis', interpolation='nearest')
                # plt.colorbar()

            try:

                if self.checkbox_addlist_1.isChecked():
                    folder_path='C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/image1'
                    print('path')
                    image_path=self.findfilepath(folder_path)
                    print(image_path)
                    print('path')
                    avg_image=self.calculate_average_image(image_path)
                    print('avg')
                    # print(np.max(avg_image))

                    # avg_image_pil = Image.fromarray(avg_image)
                    phase_path='C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image1.bmp'
                    cv2.imwrite(phase_path, avg_image)
                    print('write')

                    units = ' (px)'
                    name = 'average 1'
                    phase = np.array(im.open(phase_path))
                    phase[0, 0] = 0
                    phase[-1, -1] = 2 * np.pi
                    figure = self.image_camera_plot_1

                    if cupy_working:
                        figure.axes.imshow(phase, cmap='hsv')
                    else:
                        figure.axes.imshow(phase , cmap='hsv')
                    # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                    figure.axes.set_xlabel('$x$' + units)
                    figure.axes.set_ylabel('$y$' + units)
                    figure.axes.set_title(name)

                    # figure.axes.tight_layout()
                    if self.image_colorbar_1_num == 0:
                        figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes,
                                            label='$2\\pi$ radians')
                    self.image_colorbar_1_num = 1

                    figure.draw()
                if self.checkbox_addlist_2.isChecked():
                    folder_path = 'C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/image2'
                    image_path = self.findfilepath(folder_path)
                    avg_image = self.calculate_average_image(image_path)

                    avg_image_pil = Image.fromarray(avg_image)
                    phase_path = 'C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image2.bmp'
                    cv2.imwrite(phase_path, avg_image)
                    units = ' (px)'
                    name = 'average 2'
                    phase = np.array(im.open(phase_path))
                    phase[0, 0] = 0
                    phase[-1, -1] = 2 * np.pi
                    figure = self.image_camera_plot_2
                    if cupy_working:
                        figure.axes.imshow(phase, cmap='hsv')

                    else:
                        figure.axes.imshow(phase, cmap='hsv')
                    # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                    figure.axes.set_xlabel('$x$' + units)
                    figure.axes.set_ylabel('$y$' + units)
                    figure.axes.set_title(name)
                    # figure.axes.tight_layout()
                    if self.image_colorbar_2_num == 0:

                        try:
                            figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes,
                                                label='$2\\pi$ radians')
                        except Exception as e:
                            print('colorbar e:',e)
                    self.image_colorbar_2_num = 1
                    figure.draw()
                if self.checkbox_addlist_3.isChecked():
                    folder_path = 'C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/image3'
                    image_path = self.findfilepath(folder_path)
                    avg_image = self.calculate_average_image(image_path)

                    avg_image_pil = Image.fromarray(avg_image)
                    phase_path = 'C:/Users/RiceT/Documents/SLM_computation/Filrcamera/pictures/average_image3.bmp'
                    cv2.imwrite(phase_path, avg_image)
                    units = ' (px)'
                    name = 'average 3'
                    phase = np.array(im.open(phase_path))
                    phase[0, 0] = 0
                    phase[-1, -1] = 2 * np.pi
                    figure = self.image_camera_plot_3
                    if cupy_working:
                        figure.axes.imshow(phase, cmap='hsv')
                    # figure.axes.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
                    figure.axes.set_xlabel('$x$' + units)
                    figure.axes.set_ylabel('$y$' + units)
                    figure.axes.set_title(name)
                        # figure.axes.tight_layout()
                    if self.image_colorbar_3_num == 0:
                        figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes,
                                                label='$2\\pi$ radians')
                    self.image_colorbar_3_num = 1
                    figure.draw()
            except Exception as e:
                import traceback

                traceback.print_exc()
                print('average',e)
        except Exception as e:
            import traceback
            print('circle', e)
            traceback.print_exc()


    def apply_phase(self,option):
        # if not self.connect:
        #     self.server_address = ('localhost', 51814)
        #     self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     self.server.bind(self.server_address)
        #     env = 'C:/Users/RiceT/.conda/envs/slm/python.exe'
        #     subprocess.Popen([env, 'C:/Users/RiceT/Documents/SLM_computation/Filrcamera/save_image.py'],
        #                      shell=True)
        #     time.sleep((0.2))
        #     self.server.listen(1)
        #     self.client_sock, address = self.server.accept()
        #     self.connect = True
        self.load_already=np.zeros(3)

        if option == 1:
            self.correction_phase_1 = self.filename_phase_1.text()
            self.sent_phase_scan_MDS(1)
        if option == 2:
            self.correction_phase_2 = self.filename_phase_2.text()

            self.sent_phase_scan_MDS(2)
        if option == 3:
            self.correction_phase_3 = self.filename_phase_3.text()
            self.sent_phase_scan_MDS(3)
        # time.sleep(0.8)
        # self.take_image_show(option)


    def take_image(self,item):
        if not self.connect:
            self.flir = FlirCamController_slm.FlirCamController()
            self.flir.initialize()
            self.flir.start_continue()
            self.flir.set_average_frames(1)
            self.connect = True
        # t0 = time.time()
        self.flir.acquire_continue()
        # print('show', time.time() - t0)
        # data_ = data[4:]
        if item == 1:
            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\image1'
            now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
        if item == 2:
            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\image2'
            now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
        if item== 3:
            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\image3'
            now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
        # if len(data_) > 1:
        #     directory = r'C:\Users\RiceT\Documents\SLM_computation\Filr camera\pictures\correction1'
        #     now = data_
        # print('show', time.time() - t0)

        name = f"image_{now}.bmp"
        path = os.path.join(directory, name)
        self.flir.file_save(path)
        # print('show', time.time() - t0)

        # message = path.encode('utf-8')
        # client_socket.sendto(message, ('localhost', 51814))
        # message = 'take'+str(item)
        # message=message.encode('utf-8')
        # self.client_sock.sendto(message, self.server_address)





    def take_image_show(self,option):

        if not self.connect:
            self.flir = FlirCamController_slm.FlirCamController()
            self.flir.initialize()
            self.flir.start_continue()
            self.flir.set_average_frames(1)
            self.connect = True

        if self.checkbox_save_next_image.isChecked():
            self.flir.acquire_continue()
            now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\image_intensity'
            name = f"image_{now}.bmp"
            path = os.path.join(directory, name)

        else:
            self.flir.acquire_continue()
            now = (datetime.datetime.now().strftime("d%d_h%H_m%M_s%S"))
            directory = r'C:\Users\RiceT\Documents\SLM_computation\Filrcamera\pictures\image0'
            name = f"image_{now}.bmp"
            path = os.path.join(directory, name)
        self.flir.file_save(path)
        if self.checkbox_save_next_image.isChecked():
            # self.slm_intensity_path = path
            self.filename_phase_0.setText(path)

            self.checkbox_save_next_image.setChecked(False)
            print(path)
        units = ' (px)'
        if option!=0:
            name = 'camera image'+str(option)
        else:
            name = 'camera image'
        try:
            phase = np.array(im.open(path))
        except Exception as e:
            print('error phase',e)
        phase[0, 0] = 0
        phase[-1, -1] = 2 * np.pi
        figure = self.image_camera_plot_1
        figure.axes.imshow(phase, cmap='hsv')
        if self.image_colorbar_1_num == 0:
            figure.fig.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=figure.axes, label='$2\\pi$ radians')
        figure.axes.set_xlabel('$x$' + units)
        figure.axes.set_ylabel('$y$' + units)

        figure.axes.set_title(name)


        self.image_colorbar_1_num=1

        figure.draw()

  

    # Open a target 2d image
    def open_2d(self):
        # print('heyo')
        self.update_vals()

        filename = tkinter.filedialog.askopenfilename(title='Select Image', initialfile=self.filename_2d.text())
        if self.arb2d_filename == '':
            return
        self.arb2d_filename = filename
        self.filename_2d.setText(self.arb2d_filename)

        self.update_vals()
        return self.arb2d_filename

    # Open an intensity image
    def open_intensity(self):
        # print('heyo')
        self.update_vals()

        filename = tkinter.filedialog.askopenfilename(title='Select Intensity Image', initialfile=self.intensity_path.text())
        if filename == '':
            return
        self.intensity_filename = filename
        self.intensity_path.setText(self.intensity_filename)

        amp = self.mod.BMPToAmp(self.intensity_filename)
        self.mod.ampToBMP(amp=amp, name='Intensity Profile', figure=self.intensity_plot, color=True, show=False, extent=(0, (amp.shape[1] - 1) * 1.85, 0, (amp.shape[0] - 1) * 1.85), units=' ($\mu$m)')
        self.intensity_plot.draw()

        self.update_vals()
        return self.intensity_filename

    # Open a reference image
    def open_reference(self):
        # print('heyo')
        self.update_vals()

        filename = tkinter.filedialog.askopenfilename(title='Select Reference Image', initialfile=self.reference_path.text())
        if filename == '':
            return
        self.reference_filename = filename
        self.reference_path.setText(self.reference_filename)

        amp = self.mod.BMPToAmp(self.reference_filename)
        self.mod.ampToBMP(amp=amp, name='Reference Profile', figure=self.reference_plot, color=True, show=False, extent=(0, (amp.shape[1] - 1) * 1.85, 0, (amp.shape[0] - 1) * 1.85), units=' ($\mu$m)')
        self.reference_plot.draw()

        self.update_vals()
        return self.reference_filename

    # Open a reference image
    def open_interference1(self):
        # print('heyo')
        self.update_vals()

        filename = tkinter.filedialog.askopenfilename(title='Select Interference Image', initialfile=self.interference1_path.text())
        if filename == '':
            return
        self.interference1_filename = filename
        self.interference1_path.setText(self.interference1_filename)

        amp = self.mod.BMPToAmp(self.interference1_filename)
        self.mod.ampToBMP(amp=amp, name='Interference Profile', figure=self.interference1_plot, color=True, show=False, extent=(0, (amp.shape[1] - 1) * 1.85, 0, (amp.shape[0] - 1) * 1.85), units=' ($\mu$m)')
        self.interference1_plot.draw()

        self.update_vals()
        return self.interference1_filename

    # Open a reference image
    def open_interference2(self):
        # print('heyo')
        self.update_vals()

        # print(self.interference2_path)
        filename = tkinter.filedialog.askopenfilename(title='Select Interference + pi/2 Image', initialfile=self.interference2_path.text())
        if filename == '':
            return
        self.interference2_filename = filename
        self.interference2_path.setText(self.interference2_filename)

        amp = self.mod.BMPToAmp(self.interference2_filename)
        self.mod.ampToBMP(amp=amp, name='Interference 2 Profile', figure=self.interference2_plot, color=True, show=False, extent=(0, (amp.shape[1] - 1) * 1.85, 0, (amp.shape[0] - 1) * 1.85), units=' ($\mu$m)')
        self.interference2_plot.draw()

        self.update_vals()
        return self.interference2_filename

    def phase_extraction(self):
        self.update_vals()
        self.progressBar_phase.setValue(0)

        # self.phase_paths = [r"Z:\Lab Rice\Experimental Projects\SLM\camera images\1x4 tem00 array intensity.bmp",
        #          r"Z:\Lab Rice\Experimental Projects\SLM\camera images\temp-05272024115536-0.Bmp",
        #          r"Z:\Lab Rice\Experimental Projects\SLM\camera images\temp-05272024115537-1.Bmp",
        #          r"Z:\Lab Rice\Experimental Projects\SLM\camera images\temp-05272024115537-2.Bmp"]

        profs = import_profs(slm=self.mod, paths=self.phase_paths)
        if cupy_working:
            profs = [prof.get() for prof in profs]
        self.progressBar_phase.setValue(40)
        phase = phasemap_2d(profs)
        self.progressBar_phase.setValue(80)
        self.phase_prof_layout.removeWidget(self.phase_prof_plot)
        self.phase_prof_layout.removeWidget(self.toolbar_phase_prof)
        self.phase_prof_plot = MplCanvas(self, height=9, dpi=60, dark=False)
        self.toolbar_phase_prof = NavigationToolbar(self.phase_prof_plot, self.mainwindow)
        self.phase_prof_layout.addWidget(self.toolbar_phase_prof)
        self.phase_prof_layout.addWidget(self.phase_prof_plot)
        plot_phasemap(slm=self.mod, phase=phase, profs=profs, figs=(self.phase_field_plot, self.phase_phase_plot, self.phase_prof_plot), colorbars=self.actve_colorbars[2:], name='Phase Profile')
        self.phase_field_plot.draw()
        self.phase_phase_plot.draw()
        self.phase_prof_plot.draw()
        self.actve_colorbars[2] = True
        self.actve_colorbars[3] = True

        self.progressBar_phase.setValue(100)

    def aberration_correction(self):
        self.slm_plot.draw()

    # Create an input light profile
    def input_prof(self, res=True):
        self.update_vals()
        if res:
            res = self.res
        else:
            res = 1
        input_profile = cp.array(profile.Profile.input_gaussian(beam_size=self.input_waist / res,
                                                                     size=[int(i * res) for i in self.mod.size]))
        input_profile /= cp.sqrt(cp.sum(cp.abs(input_profile) ** 2))

        return input_profile

    # Apply a flat phase input beam
    def apply_input(self):
        self.slm_field = self.input_prof()

        self.mod.fieldtoBMP(self.slm_field, name='SLM Plane', wavelength=self.wave, color=True, show=False,
                            figure=self.slm_plot, norm=False, sat=False, colorbar=not self.actve_colorbars[0], extent=(-self.slm_field.shape[1] / 2, self.slm_field.shape[1] / 2, -self.slm_field.shape[0] / 2, self.slm_field.shape[0] / 2))
        self.actve_colorbars[0] = True
        self.slm_plot.draw()

    # Add the algorithm generated phase pattern to the slm plane
    def apply_alg(self):
        self.input_profile = self.input_prof()

        self.slm_field = self.input_profile * cp.exp(1j * (cp.angle(self.slm_field) + cp.array(slm.pad_border(self.mod.phase, self.input_profile.shape))))
        # self.mod.phase = cp.angle(self.slm_field)

        self.mod.fieldtoBMP(self.slm_field, name='SLM Plane', wavelength=self.wave, color=True, show=False,
                             figure=self.slm_plot, norm=False, sat=False, colorbar=not self.actve_colorbars[0], extent=(-self.slm_field.shape[1] / 2, self.slm_field.shape[1] / 2, -self.slm_field.shape[0] / 2, self.slm_field.shape[0] / 2))
        self.actve_colorbars[0] = True
        self.slm_plot.draw()

    # Compute a phase pattern using the algorithm or some analytic generator
    def compute_alg(self):
        self.update_vals()
        # size = (1024, 1272)
        # print('Going')

        self.pushButton_alg_compute.setText("Computing")
        self.progressBar_alg.setValue(0)

        self.mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % self.wave, size=(self.shape[1], self.shape[0]))
        input_profile = self.input_prof(res=False)

        if self.bs:
            # with open('log.txt', 'w') as f:
            #     with redirect_stdout(f):
            # bs_args = (
            # self.bs_N, self.bs_M, self.bs_n, self.wave, 'bs', self.bs_amps, self.bs_amps_guess, self.bs_phases,
            # self.bs_x_pitch, input_profile, True, 1, True, (self.inner_plot, self.outer_plot, self.eprof_plot))

            # thread = QtCore.QThread()
            # process = SubThread(parent=None, ui=self, process='bs', wave=self.wave, shape=self.shape, args=bs_args)
            # process.moveToThread(thread)
            # thread.started.connect(process.initialize_in_thread)
            # thread.finished.connect(thread.deleteLater)
            # thread.start()
            # thread.start(n=self.bs_n, N=self.bs_N, M=self.bs_M, name='bs', amps=self.bs_amps, amps_guess=self.bs_amps_guess, phases=self.bs_phases, x_pitch=self.bs_x_pitch, plots=True, res_factor=1, input_profile=input_profile, phase_memory=True, figs=(self.inner_plot, self.outer_plot, self.eprof_plot))
            # print(self.bs_phases)
            bs = self.mod.wu_algorithm(n=self.bs_n, N=self.bs_N, M=self.bs_M, name='bs', amps=self.bs_amps,
                                       amps_guess=self.bs_amps_guess, phases=self.bs_phases, x_pitch=self.bs_x_pitch,
                                       plots=True, res_factor=1, input_profile=input_profile, phase_memory=True,
                                       figs=(self.inner_plot, self.outer_plot, self.eprof_plot),
                                       tem01=self.hg and self.hg_n == 1 and self.hg_m == 0, imag=False)
            self.mod.add_phase(bs)
            self.eprof_plot.draw()
            self.inner_plot.draw()
            self.outer_plot.draw()

        if self.flat:
            shift = self.mod.flat(self.mod.beams[0][1], self.shift * 2 * cp.pi)
            self.mod.add_phase(shift)

        if self.defl:
            grad = self.mod.gradient(self.mod.beams[0][1], angle=self.def_ag * cp.pi / 180, axis=self.def_ax, wavelength=self.wave)
            self.mod.add_phase(grad)

        if self.hg and not (self.hg_n == 1 and self.hg_m == 0 and self.bs):
            hg = self.mod.temnm(self.mod.beams[0][1], n=self.hg_n, m=self.hg_m)
            self.mod.add_phase(hg)

        if self.lg:
            lg = cp.array(self.mod.laguerre_gaussian(self.mod.beams[0][1], l=self.hg_n, p=self.hg_m))
            self.mod.add_phase(lg)

        if self.rand:
            rand = self.mod.rand(self.mod.beams[0][1])
            self.mod.add_phase(self.rand_op * rand)

        if self.zern:
            zern = self.mod.zernike_sum(self.mod.beams[0][1], self.zern_weights, waist=self.input_waist)
            print(self.zern_weights)
            print(self.mod.beams[0][1])
            zern = slm.pad_border(zern, self.mod.phase.shape)
            print('border padded')
            print(zern.shape)
            self.mod.add_phase(zern)
            print('phase added')

        if self.arb2d:
            target = self.mod.BMPToAmp(path=self.arb2d_filename, norm=True)
            if cupy_working:
                target = cp.array(resize(target.get(), self.mod.size))
            else:
                target = cp.array(resize(target, self.mod.size))
            # target = cp.array(slm.pad_border(target, self.mod.size))
            # with open('log.txt', 'w') as f:
            #     with redirect_stdout(f):
            arb2d = self.mod.wu_image(N=self.arb2d_N, wavelength=self.wave, input_profile=input_profile, target=target)
            self.mod.add_phase(arb2d)

        # self.alg_complete()
        self.progressBar_alg.setValue(100)
        self.pushButton_alg_compute.setText("Compute")

    # Run upon completion of the algorithm
    def alg_complete(self, phase=None):
        if phase is not None:
            self.mod.add_phase(phase)
            print('done!')
        self.eprof_plot.draw()
        self.inner_plot.draw()
        self.outer_plot.draw()

        self.progressBar_alg.setValue(100)
        self.pushButton_alg_compute.setText("Compute")


# For moving a time-consuming task to its own thread so it won't freeze the GUI
class SubThread(QtCore.QObject):
    done = QtCore.pyqtSignal(cp.ndarray)

    def __init__(self, parent=None, ui=None, process='bs', wave=411, shape=(1272, 1024), args=()):
        super().__init__(parent=parent)
        self.ui = ui
        self.process = process
        self.args = args
        self.wave = wave
        self.shape = shape
        self.mod = ArrayModulator(beams=1, correction_path='images/%dcorrwithLUT.bmp' % self.wave,
                       size=(self.shape[1], self.shape[0]))

    def initialize_in_thread(self):
        self.done.connect(self.ui.alg_complete)

        if self.process == 'bs':
            bs = self.mod.wu_algorithm(*self.args)
            self.done.emit(bs)
        elif self.process == '2d':
            im = self.mod.wu_image(*self.args)
            self.done.emit(im)


# Helper method for running the beamsplitter algorithm
def run_bs(mod, n, M, name, amps, amps_guess, phases, x_pitch, plots, res_factor, input_profile, phase_memory, figs):
    bs = mod.wu_algorithm(n=n, M=M, name=name, amps=amps, amps_guess=amps_guess,
                               phases=phases, x_pitch=x_pitch, plots=plots, res_factor=res_factor,
                               input_profile=input_profile, phase_memory=phase_memory, figs=figs)
    mod.add_phase(bs)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Create a new thread to receive input from the arduino, so it can run simultaneously as other processes
    # thread = QtCore.QThread()

    MainWindow = QtWidgets.QMainWindow()
    ui = slmGUI(MainWindow)
    MainWindow.show()
    # MainWindow.showMaximized()

    # Start the thread here, so we can receive input from the arduino
    # thread.start()

    app.exec_()

    # Quit the thread after the program finishes
    # thread.quit()

    # Save the plots from the finished run
    # ui.save_plots()

    sys.exit()