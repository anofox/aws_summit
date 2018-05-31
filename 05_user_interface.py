import glob
import numpy as np
import cv2
import sys
import matplotlib.pylab as plt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWebKitWidgets import QWebView

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType

from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QMessageBox, QMainWindow
import pyqtgraph as pg

from showcase.models import AutoencoderModel, DecissionTreeObjectClassifierModel, ClusteringObjectClassifierModel

try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)


def load_latest_online_model(encoder, online_glob_pattern, fallback_weights_file):
    file_list = sorted(glob.glob(online_glob_pattern))
    if not file_list:
        weights_file = fallback_weights_file
    else:
        weights_file = file_list[-1]
    print("Loading weights from file %s" % weights_file)
    encoder.load_model(weights_file=weights_file)

def norm_scale_img(raw, target_shape=(640 // 5, 480 // 5)):
    resized = cv2.resize(raw, target_shape)
    normalized = cv2.normalize(resized, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)


def combine_depth_and_rgb(d_ary, rgb_ary):
    d_ary = cv2.cvtColor(d_ary, cv2.COLOR_BGR2GRAY)
    rgbd_ary = np.concatenate((rgb_ary, np.atleast_3d(d_ary)), axis=2)
    return rgbd_ary


fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener((FrameType.Color | FrameType.Depth))
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)
device.start()


def retrieve_frame_and_update():
    try:
        rgbd_ary = retrieve_frame_from_kinect()
        x = encode_or_train_autoencoder(rgbd_ary[np.newaxis, :, :, :])
        update_or_learn_classification(x)
    except Exception as e:
        print("Error during frame update %s" % e)
        sys.exit(1)


def retrieve_frame_from_kinect():
    global depth_imi, color_imi

    frames = listener.waitForNewFrame()
    d_ary = frames["depth"].asarray()
    color_ary = cv2.cvtColor(frames["color"].asarray(), cv2.COLOR_BGR2RGB)

    norm_d_ary = norm_scale_img(cv2.cvtColor(d_ary, cv2.COLOR_GRAY2RGB))
    norm_color_ary = norm_scale_img(color_ary)
    rgbd_ary = combine_depth_and_rgb(norm_d_ary, norm_color_ary)

    depth_imi.setImage(cv2.flip(np.rot90(d_ary), flipCode=1))
    color_imi.setImage(cv2.flip(np.rot90(color_ary), flipCode=1))

    listener.release(frames)

    return rgbd_ary


def encode_or_train_autoencoder(rgbd_ary):
    global showcase_root, encoder, latent_imi

    is_encoder_learning = showcase_root.buttonTrainEncoder.isChecked()
    if is_encoder_learning:
        hist = encoder.online_fit(rgbd_ary)
        if hist:
            t = np.arange(len(hist[0]))
            encoder_status_plot.clear()
            encoder_status_plot.plot(t, hist[0], pen=pg.mkPen(width=2.5, color='y'))
            encoder_status_plot.plot(t, hist[1], pen=pg.mkPen(width=2.5, color='g'))

    x = encoder.encode(rgbd_ary)
    latent_imi.setImage(np.rot90(encoder.history))

    return x


def update_or_learn_classification(x):
    global showcase_root, inference_label, classifier_status_plot, classifier

    is_classifier_training = showcase_root.buttonLearn.isChecked()

    class_str = showcase_root.lineClassName.text()
    if is_classifier_training and class_str:
        x, y = classifier.online_fit(x, class_str)
        classifier_status_plot.clear()
        classifier_status_plot.plot(x, y, pen=None, symbol='o')
    else:
        pred_class, y_prob = classifier.predict_class(x)
        if pred_class:
            inference_label.setText(pred_class)
        else:
            inference_label.setText("Classification not successful")


@pyqtSlot()
def on_button_learn_click(btn):
    global showcase_root
    pass

@pyqtSlot()
def on_button_save_encoder_model_click(btn):
    global showcase_root, encoder
    weights_file = encoder.save_model_snapshot("ONLINE_autoenc_cnn_rgbd_%d.h5")
    QMessageBox.information(showcase_root, "Model saved", "Model saved to file %s" % weights_file)

@pyqtSlot()
def initialize_or_reset_encoder():
    global classifier
    print("Resetting encoder")
    encoder = AutoencoderModel(input_shape=(96, 128, 4))
    load_latest_online_model(encoder, fallback_weights_file="autoenc_cnn_rgbd.h5",
                             online_glob_pattern="ONLINE_autoenc_cnn_rgbd_*.h5")

@pyqtSlot()
def initialize_or_reset_classifier():
    global classifier
    print("Resetting classifier")
    classifier = ClusteringObjectClassifierModel()

class ShowcaseApp(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, flags=Qt.FramelessWindowHint)
        self.ui = loadUi('./showcase/train_app.ui', self)
        self.__connect_events()
        self.plot_widgets = self.__initialize_plot_and_result_widgets()

    def __connect_events(self):
        self.ui.buttonLearn.clicked.connect(on_button_learn_click)
        self.ui.buttonResetClassifier.clicked.connect(initialize_or_reset_classifier)
        self.ui.buttonResetEncoder.clicked.connect(initialize_or_reset_encoder)
        self.ui.buttonSaveEncoderModel.clicked.connect(on_button_save_encoder_model_click)

    def __initialize_plot_and_result_widgets(self):
        pg.setConfigOption('background', pg.mkColor("#004772"))

        colormap = plt.get_cmap("PiYG_r")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        depth_widget = self.ui.depthFrame
        depth_layout = QVBoxLayout()
        depth_widget.setLayout(depth_layout)
        depth_plot = pg.PlotWidget()
        depth_plot.hideAxis('left')
        depth_plot.hideAxis('bottom')
        depth_layout.addWidget(depth_plot)
        depth_imi = pg.ImageItem()
        depth_imi.setLookupTable(lut)
        depth_plot.addItem(depth_imi)

        color_widget = self.ui.colorFrame
        color_layout = QVBoxLayout()
        color_widget.setLayout(color_layout)
        color_plot = pg.PlotWidget()
        color_plot.hideAxis('left')
        color_plot.hideAxis('bottom')
        color_layout.addWidget(color_plot)
        color_imi = pg.ImageItem()
        color_plot.addItem(color_imi)

        latent_widget = self.ui.latentFrame
        latent_layout = QVBoxLayout()
        latent_widget.setLayout(latent_layout)
        latent_plot = pg.PlotWidget()
        latent_plot.hideAxis('left')
        latent_plot.hideAxis('bottom')
        latent_layout.addWidget(latent_plot)
        latent_imi = pg.ImageItem()
        latent_imi.setLookupTable(lut)
        latent_plot.addItem(latent_imi)

        inference_label = self.ui.inferenceLabel
        inference_label.setText("Starting up ... stay tuned")

        encoder_status_widget = self.ui.learnEncoderStatusWidget
        encoder_status_layout = QVBoxLayout()
        encoder_status_widget.setLayout(encoder_status_layout)
        encoder_status_plot = pg.PlotWidget()
        encoder_status_layout.addWidget(encoder_status_plot)

        classifier_status_widget = self.ui.classifierStatusWidget
        classifier_status_layout = QVBoxLayout()
        classifier_status_widget.setLayout(classifier_status_layout)
        classifier_status_plot = pg.PlotWidget()
        classifier_status_layout.addWidget(classifier_status_plot)

        return depth_imi, color_imi, latent_imi, inference_label, encoder_status_plot, classifier_status_plot

    def __toggleWindows(self):
        if self.slides:
            self.hide()
            self.slides.show()

    def keyPressEvent(self, event):
        if type(event)==QKeyEvent:
            print("Event key ", event.key())
            if event.key()==Qt.Key_P:
                print("Esc pressed, starting slides")
                self.__toggleWindows()
            event.accept()
        else:
            event.ignore()

class SlidesApp(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, flags=Qt.FramelessWindowHint)
        self.ui = loadUi("./showcase/slides_app.ui", self)

    def __toggleWindows(self):
        if self.showcase:
            self.hide()
            self.showcase.show()

    def keyPressEvent(self, event):
        if type(event)==QKeyEvent:
            print("Event key ", event.key())
            if event.key()==Qt.Key_P:
                print("Esc pressed, starting showcase")
                self.__toggleWindows()
            event.accept()
        else:
            event.ignore()


encoder = AutoencoderModel(input_shape=(96, 128, 4))
load_latest_online_model(encoder, fallback_weights_file="autoenc_cnn_rgbd.h5",
                                  online_glob_pattern="ONLINE_autoenc_cnn_rgbd_*.h5")
classifier = ClusteringObjectClassifierModel()

app = QApplication(sys.argv)
showcase_app = ShowcaseApp()
showcase_root = showcase_app.ui
slides_app = SlidesApp()
slides_app.showcase = showcase_app
showcase_app.slides = slides_app

depth_imi, color_imi, latent_imi, inference_label, encoder_status_plot, classifier_status_plot = showcase_app.plot_widgets

timer = QTimer()
timer.timeout.connect(retrieve_frame_and_update)

if __name__ == '__main__':
    showcase_root.show()
    timer.start(200)
    sys.exit(app.exec_())

device.stop()
device.close()

sys.exit(0)
