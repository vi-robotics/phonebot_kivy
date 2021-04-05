#!/usr/bin/env python3

# Standard library imports
import cv2
import numpy as np
import time

# Kivy imports
import kivy
from kivy.uix.image import Image
from kivy.properties import (NumericProperty, ListProperty,
                             BooleanProperty)
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.logger import Logger

# Custom imports
from camera_android_2 import CameraAndroid2
from phonebot.core.common.util import get_time


class FaceEstimator(object):
    def __init__(self,  distance_weight=1, area_weight=1):
        """A moving average tracker with heuristiscs based on the size of the face detected
    and the distance to the last detected face (to prefer tracking a single object when
    multiple detections are available). 

        Args:
            distance_weight (int, optional): [description]. Defaults to 1.
            area_weight (int, optional): [description]. Defaults to 1.
        """
        # TODO(yycho0108): Choice of these parameters should be better documented.
        self.num_avg = 5
        self.alpha = 0.1
        self.estimated_position = np.asarray([np.nan, np.nan])
        self.estimated_angular_position = np.asarray([np.nan, np.nan])
        self.dw = distance_weight
        self.aw = area_weight
        # TODO(yycho0108): Consistent member naming.
        self.last_detection_time_ = None

    def get_most_recent_detection(self):
        return (self.last_detection_time_, self.estimated_angular_position[0], self.estimated_angular_position[1])

    def update_from_measurement(self, detections, center_x, center_y, focal_length_x, focal_length_y):
        """Updates the position and angular estimates of the tracked face from a new measurement. 

        Args:
            detections (array): A nx4 array where each row is a face detection, and each column is an integer 
            representing x, y, w, h of each face detection. 
            center_x (int): The x value of the center of the image to measure angles relative to.
            center_y (int): The y value of the center of the image.
            focal_length_x (float): The x focal length in mm.
            focal_length_y (float): The y focal length in mm.
        """
        if len(detections) == 0:
            return

        max_linear_area = 0
        min_distance = float('inf')
        detection = [center_x, center_y]
        for (x, y, w, h) in detections:
            center_pt = np.asarray([x + w/2, y+h/2])
            new_linear_area = np.sqrt(w*h)
            centerDist = np.linalg.norm(center_pt - self.estimated_position)
            if centerDist*self.dw + new_linear_area*self.aw < min_distance*self.dw + max_linear_area * self.aw:
                detection = center_pt
                max_linear_area = new_linear_area

        detection = np.asarray(detection)

        if not np.all(np.isfinite(self.estimated_position)):
            # Initialization case, overwrite.
            self.estimated_position = detection
        else:
            # Simple moving average filter. Could be more complex, but not sure why.
            # self.estimated_position = 1/self.num_avg * detection + \
            #    (self.num_avg-1)/self.num_avg * self.estimated_position
            self.estimated_position = (
                (1.0 - self.alpha) * detection + self.alpha * self.estimated_position)

        # Calculate the angle of the face vector from the center vector, and make sure that the angle is signed.
        # FIXME(yycho0108): Why is there a negative sign here?
        # The choice of reference frame should be better documented.
        alpha_x = - np.arctan2(center_x -
                               self.estimated_position[0], focal_length_x)
        alpha_y = np.arctan2(
            center_y - self.estimated_position[1], focal_length_y)

        self.estimated_angular_position = np.asarray([alpha_x, alpha_y])
        self.last_detection_time_ = get_time()


class Camera2(Image):
    '''
    Note: This is a re-implementation of the original Camera class from kivy designed to 
    work with Andoid and include face tracking 
    (see https://kivy.org/doc/stable/_modules/kivy/uix/camera.html). 


    Camera2
    ======

    The :class:`Camera` widget is used to capture and display video from a camera.
    Once the widget is created, the texture inside the widget will be automatically
    updated. Our :class:`~kivy.core.camera.CameraBase` implementation is used under
    the hood::

        cam = Camera2()

    By default, the first camera found on your system is used. To use a different
    camera, set the index property::

        cam = Camera2(index=1)

    You can also select the camera resolution::

        cam = Camera2(resolution=(320, 240))

    .. warning::

        The camera texture is not updated as soon as you have created the object.
        The camera initialization is asynchronous, so there may be a delay before
        the requested texture is created.
    '''

    play = BooleanProperty(True)
    '''Boolean indicating whether the camera is playing or not.
    You can start/stop the camera by setting this property::

        # start the camera playing at creation (default)
        cam = Camera(play=True)

        # create the camera, and start later
        cam = Camera(play=False)
        # and later
        cam.play = True

    :attr:`play` is a :class:`~kivy.properties.BooleanProperty` and defaults to
    True.
    '''

    index = NumericProperty(-1)
    '''Index of the used camera, starting from 0.

    :attr:`index` is a :class:`~kivy.properties.NumericProperty` and defaults
    to -1 to allow auto selection.
    '''

    resolution = ListProperty([-1, -1])
    '''Preferred resolution to use when invoking the camera. If you are using
    [-1, -1], the resolution will be the default one::

        # create a camera object with the best image available
        cam = Camera()

        # create a camera object with an image of 320x240 if possible
        cam = Camera(resolution=(320, 240))

    .. warning::

        Depending on the implementation, the camera may not respect this
        property.

    :attr:`resolution` is a :class:`~kivy.properties.ListProperty` and defaults
    to [-1, -1].
    '''

    def __init__(self, cascade_scale_factor=1.5, cascade_min_neighbors=3, **kwargs):
        super(Camera2, self).__init__(**kwargs)
        Logger.info("PhoneBot: Loading Camera")
        self.start_time = time.time()

        # TODO(yycho0108): Remove?
        self.cascade_scale_factor = cascade_scale_factor
        self.cascade_min_neighbors = cascade_min_neighbors
        self.face_cascade = cv2.CascadeClassifier()
        self.face_cascade.load('haarcascade_frontalface_default.xml')

        # FIXME(yycho0108): Consistent naming conventions.
        self._camera = None

        if self.index == -1:
            self.index = 0

        fbind = self.fbind
        fbind('index', self._on_index)
        fbind('resolution', self._on_index)
        self._on_index()

        self.tracker = FaceEstimator()
        self.focal_length_x = None
        self.focal_length_y = None

        Logger.info("PhoneBot: Done Loading Camera")

    def on_tex(self, *l):
        buf = self._camera.grab_frame()
        if not buf:
            return
        frame = self.decode_frame(buf)
        # TODO (Max): Get native face detection working in place of this
        frame = self.process_frame(frame)
        buf = frame.tostring()

        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

        # display image from the texture
        self.canvas.ask_update()
        return True

    def decode_frame(self, buf):
        """
        NOTE: This is a re-implementation of Kivy's CoreCamera decode_frame function. 
        There was a bug in which (h + h / 2) was not returning an int, so the function
        failed. See https://github.com/kivy/kivy/blob/master/kivy/core/camera/camera_android.py
        for more details. 
        """

        w, h = self.resolution
        arr = np.fromstring(buf, 'uint8').reshape((int(w + w / 2), int(h)))
        arr = cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_NV21)
        arr = np.flipud(arr)
        return np.rot90(arr, 1)

    def process_frame(self, frame):
        # Logger.info("FPS: {}".format(1/(time.time() - self.start_time)))
        self.start_time = time.time()
        rows, cols, channel = frame.shape
        frame = cv2.UMat(frame)

        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # faces = self.face_cascade.detectMultiScale(
        #     gray, self.cascade_scale_factor, self.cascade_min_neighbors)
        faces = self._camera.get_face_detections()

        scale = np.asarray([cols, rows, cols, rows]) / 2000.0
        offset = np.asarray([cols/2, rows/2, 0, 0])
        for i, face in enumerate(faces):
            new_face = np.asarray(face) * scale + offset
            faces[i] = new_face

        self.tracker.update_from_measurement(
            faces,  self.resolution[0]/2, self.resolution[1]/2, self.focal_length_x, self.focal_length_y)

        # Visualize current measurements.
        for (x, y, w, h) in faces:
            x, y, w, h = [int(e) for e in (x, y, w, h)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if (np.all(np.isfinite(self.tracker.estimated_position))):
            cv2.circle(frame, (int(self.tracker.estimated_position[0]), int(
                self.tracker.estimated_position[1])), 60, (255, 0, 0), 2)

            cv2.putText(frame, "X Angle: {:.2f}, Y Angle: {:.2f}".format(self.tracker.estimated_angular_position[0],
                                                                         self.tracker.estimated_angular_position[1]), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        return cv2.UMat.get(frame)

    def _on_index(self, *largs):
        self._camera = None
        if self.index < 0:
            return
        if self.resolution[0] < 0 or self.resolution[1] < 0:
            return
        self._camera = CameraAndroid2(index=self.index,
                                      resolution=(self.resolution[1], self.resolution[0]), stopped=True)

        self._camera.bind(on_load=self._camera_loaded)
        Logger.info("PhoneBot: Binding Camera")
        horzAngle = self._camera._android_camera.getParameters(
        ).getHorizontalViewAngle() * np.pi / 180
        vertAngle = self._camera._android_camera.getParameters().getVerticalViewAngle() * \
            np.pi / 180
        Logger.info("PhoneBot: Horizontal = {}, Vertical = {}".format(
            horzAngle, vertAngle))
        self.focal_length_x = (
            0.5 * self.resolution[0]) / np.tan(0.5 * horzAngle)
        self.focal_length_y = (
            0.5 * self.resolution[1]) / np.tan(0.5 * vertAngle)

        if self.play:
            self._camera.start()
            self._camera.bind(on_texture=self.on_tex)

    def _camera_loaded(self, *largs):
        texture = self._camera.texture
        image_texture = Texture.create(
            size=(texture.height, texture.width), colorfmt='rgb')
        self.texture = image_texture
        self.texture_size = list(self.texture.size)

    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
        else:
            self._camera.stop()
