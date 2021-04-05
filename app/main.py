#!/usr/bin/env python3

from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.util import get_time, increment_time, get_time_scale
from phonebot.core.common.serial import encode
from phonebot.core.common.comm.client import SimpleClient
from phonebot.core.common.math.utils import alerp

import time
import numpy as np
import os
import zlib
from collections import deque
import threading

# Kivy
from jnius import autoclass, cast
from android.permissions import check_permission, request_permissions, Permission

from kivy.logger import Logger
from kivy.clock import Clock
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.utils import platform
from camera import Camera2

import phone_config as cfg
from robot import Phonebot

# Log
# TODO(yycho0108): Fix logging.
# import logging
# logging.root = Logger
from phonebot.core.common.logger import get_default_logger
logger = get_default_logger()

BleService = autoclass('main.com.phonebot.BluetoothLeService')
OrientationListener = autoclass('main.com.phonebot.OrientationListener')
PhoneBotCommandEncoder = autoclass('main.com.phonebot.PhoneBotCommandEncoder')
PhoneBotCommand = autoclass(
    'main.com.phonebot.PhoneBotCommandEncoder$PhoneBotCommand')


Builder.load_string('''
<FaceFollowWidget>:
    orientation: 'vertical'
    Camera2:
        index: 1
        id: camera
        resolution: (480, 640) # width, height
        play: False
        canvas.before:
            PushMatrix
            Scale:
                origin: self.center
                y: -1
        canvas.after:
            PopMatrix
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    ToggleButton:
        text: 'Connect to PhoneBot'
        on_press: app.on_ble(args)
        size_hint_y: None
        height: '48dp'
    ToggleButton:
        text: 'Start Demo'
        on_press: app.on_start_demo(args)
        size_hint_y: None
        height: '48dp'
    Slider:
        id: slider
        min: 0
        max: 180
        step: 1
        size_hint_y: None
        orientation: 'horizontal'
''')


class FaceFollowWidget(BoxLayout):
    """
    Layout for face follow demo.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def clamp(x, minx, maxx):
    return min(max(minx, x), maxx)


def get_data_dir():
    Environment = autoclass('android.os.Environment')
    PythonActivity = autoclass('org.kivy.android.PythonActivity')
    context = cast('android.content.Context', PythonActivity.mActivity)
    file_p = cast('java.io.File', context.getExternalFilesDir(
        Environment.DIRECTORY_DOWNLOADS))
    data_dir = str(file_p.getAbsolutePath())
    return data_dir


def get_angle_at_time(stamp_queue, angle_queue, stamp):
    # Validate queue.
    if len(stamp_queue) != len(angle_queue):
        return None
    if len(stamp_queue) <= 0:
        return None

    # Handle out-of-bounds cases.
    if stamp >= stamp_queue[-1]:
        return angle_queue[-1]
    if stamp <= stamp_queue[0]:
        return angle_queue[0]

    # Search stamp.
    rhs = np.searchsorted(stamp_queue, stamp)
    if rhs <= 0:
        return angle_queue[0]

    # Apply interpolation.
    stamp_prv = stamp_queue[rhs-1]
    stamp_nxt = stamp_queue[rhs]
    angle_prv = angle_queue[rhs-1]
    angle_nxt = angle_queue[rhs]
    alpha = (stamp - stamp_prv) / (stamp_nxt - stamp_prv)

    # Convert to numpy array.
    angle_prv = np.asarray(angle_prv)
    angle_nxt = np.asarray(angle_nxt)
    return alerp(angle_prv, angle_nxt, alpha)


class PhonebotApp(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_permissions_ = False
        self.orientation_listener_ = None
        self.ble_connection_ = None
        self.ble_service_ = None
        self.config_ = PhonebotSettings(
            camera_position=(cfg.camera_pos[0], cfg.camera_pos[1], 0.0)
        )
        self.state_transition_times = {
            'stand_up': 10,
            'walk': 15,
            'face_follow': 0
        }
        self.state_start_time = time.time()
        self.robot_ = None
        self.current_commands = None
        self.most_recent_commands = None
        self.filter_len = 10

        self.angle_stamp_queue_ = deque(maxlen=100)
        self.angle_queue_ = deque(maxlen=100)

        self._acquire_permissions()
        if not self.has_permissions_:
            raise PermissionError("Failed to get permissions")
        self.robot_ = Phonebot(self.config_)
        #self.data_dir_ = get_data_dir()
        #print('DATA DIR ================ {}'.format(self.data_dir_))
        #self.count_ = 0
        # self.client = SimpleClient('192.168.1.162')

        self.client = SimpleClient(cfg.ip_address)

    def _on_permissions(self, permissions, grans_results):
        has_permissions = True
        for perm in permissions:
            if not check_permission(perm):
                # TODO(yycho0108): consider bookkeeping enabled permissions.
                print('FAILED CHECK FOR = {}'.format(perm))
                has_permissions = False
        if has_permissions:
            self.has_permissions_ = True
        else:
            # Request again.
            request_permissions(permissions, self._on_permissions)

    def _acquire_permissions(self, timeout=10.0, timestep=0.1):
        # TODO(yycho0108): Don't define the list of permissions here.
        permissions = [Permission.READ_EXTERNAL_STORAGE,
                       Permission.WRITE_EXTERNAL_STORAGE,
                       Permission.CAMERA]

        if platform == 'android':
            request_permissions(permissions, self._on_permissions)
        else:
            self.has_permissions_ = True

        # Block execution until permission exists.
        request_start = time.time()
        while (time.time() - request_start) < timeout:
            if self.has_permissions_:
                print('Obtained Permissions')
                break
            time.sleep(timestep)

    def build(self):
        return FaceFollowWidget()

    def on_ble(self, btn):
        Logger.info(btn[0].state)
        if btn[0].state == "down":
            Logger.info("<ble-bind>")
            self.ble_connection_ = BleService.bindToContext(
                self.activity, cfg.ble_mac_address)
            self.ble_service_ = self.ble_connection_.getService()
            Logger.info(self.ble_service_)
            Logger.info("</ble-bind>")
        else:
            Logger.info("<ble-disconnect>")
            self.ble_service_.disconnect()
            self.ble_service_ = None
            self.ble_connection_ = None
            Logger.info("</ble-disconnect")

    def on_start(self):
        # Get android activity.
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        self.activity = PythonActivity.mActivity
        # Initialize all processing handlers.
        self.orientation_listener_ = OrientationListener(self.activity, True)
        self.orientation_listener_.registerListener()
        self.robot_.initialize()
        self.root.ids.slider.bind(value=self.on_slide)

        # Initialize data directory.
        # if not os.path.exists(str(App.user_data_dir)):
        #    print('user data dir', str(App.user_data_dir))
        #    os.makedirs(str(App.user_data_dir), exist_ok=True)

        Clock.schedule_interval(self.on_step, 1.0 / 10.0)
        threading.Thread(target=self.ble_loop).start()
        # Clock.schedule_interval(self.send_ble, 1.0/30)
        return super().on_start()

    def on_stop(self):
        # TODO(yycho0108): Appropriate cleanup sequence must be performed here.
        # This includes removing the ble connection and unregistering the orientation listener.
        return super().on_stop()

    def ble_loop(self):
        while True:
            self.send_ble()
            time.sleep((1.0 / 30.0))

    def send_ble(self):
        if (self.most_recent_commands is None):
            return
        if (self.current_commands is None):
            return
        self.current_commands = (self.filter_len - 1)/(self.filter_len) * \
            self.current_commands + (1/self.filter_len) * \
            self.most_recent_commands
        # TODO(yycho0108): Ensure that (somehow) the angles here are valid and spans 0-180 deg.
        # This may require referencing leg calibration parameters. Should we just naively add 90 deg?
        signs = [1, -1, 1, -1, 1, -1, 1, -1]
        commands = [s*x for (s, x) in zip(signs, self.current_commands)]
        command_bytes = [90 + int(e) for e in np.rad2deg(commands)]
        command_bytes = [clamp(e, 0, 180) for e in command_bytes]
        # print(command_bytes)
        command_data = PhoneBotCommandEncoder.encodeCommand(
            PhoneBotCommand.SET_LEG_POSITIONS, command_bytes)
        # print(command_data)
        if self.ble_service_ is not None:
            print("Sending Data!")
            self.ble_service_.sendData(command_data)

    def on_step(self, dt):
        # Log data.
        # with open('{}/state-{:04d}.txt'.format(self.data_dir_, self.count_), 'wb') as f:
        #    state = encode((get_time(), self.robot_.graph.encode()))
        #    cstate = zlib.compress(state)
        #    f.write(cstate)
        #    self.count_ += 1

        # Stream data.
        state = encode((get_time(), self.robot_.graph.encode()))
        cstate = zlib.compress(state)
        self.client.send(cstate)

        # TODO(yycho0108): Check the health of the connections here.
        # This includes checking whether the orientation listener is running,
        # and whether the bluetooth connection is still established.

        # Receive sensor input from the android device.
        # Angles has to be a list in order for the conversion to work.
        stamp = get_time()
        angles = [0 for _ in range(3)]
        delay_ns = self.orientation_listener_.getOrientationAngles(angles)
        # NOTE(yycho0108): For whatever reason the android convention returns
        # the angles in ZYX order.
        # Invert angle order to a sensible order (x-y-z)
        angles = [angles[2], angles[1], angles[0]]

        # Propagate sensor info into the main phonebot controller.
        self.robot_.update_angle(angles[0], angles[1])

        # Populate a queue of phonebot orientation angle queue.
        # This is to enable lookup for absolute face position at a given time.
        # FIXME(yycho0108): Technically, the more correct thing to do is resolve this
        # automatically by updating the local->body transform in the frame graph.
        delay_s = (delay_ns / 1e9)
        angle_stamp = stamp - delay_s * get_time_scale()
        if len(self.angle_stamp_queue_) <= 0 or (angle_stamp >= self.angle_stamp_queue_[-1]):
            self.angle_stamp_queue_.append(angle_stamp)
            self.angle_queue_.append(angles)

        # TODO(yycho0108): Get frame information here (face tracking target).
        face_stamp, x_angle, y_angle = self.root.ids.camera.tracker.get_most_recent_detection()
        if face_stamp is not None and np.all(np.isfinite([x_angle, y_angle])):
            print("Angles: X-> {} Y-> {}".format(x_angle, y_angle))
            angle = get_angle_at_time(
                self.angle_stamp_queue_, self.angle_queue_, face_stamp)
            print("Orientation: {}".format(angles))
            if angle is not None:
                self.robot_.set_target(
                    x_angle+0.5*angle[0], y_angle+0.5*angle[1], relative=False)

        print("State: {}".format(self.robot_.get_state()))

        # NOTE(yycho0108): Disable state transition for now.
        if (self.robot_.get_state() == 'stand_up'):
            if (time.time() - self.state_start_time >
                    self.state_transition_times['stand_up']):
                # Logger.info("<stand-up>")
                self.robot_.set_state('walk')
                self.state_start_time = time.time()
                print("Walking...")
                # Logger.info("</stand-up>")
        if (self.robot_.get_state() == 'walk'):
            if (time.time() - self.state_start_time >
                    self.state_transition_times['walk']):
                print("Face Following...")
                self.robot_.set_state('face_follow')
                self.state_start_time = time.time()
        if (self.robot_.get_state() == 'face_follow'):
            if (time.time() - self.state_start_time >
                    self.state_transition_times['face_follow']):
                print("Sitting Down...")
                self.robot_.set_state('sit_down')
                self.state_start_time = time.time()

        increment_time(.025)
        # Determine commands from the main phonebot controller.
        commands = self.robot_.step(get_time())

        self.most_recent_commands = np.asarray(commands)
        if (self.current_commands is None):
            self.current_commands = self.most_recent_commands

    def on_slide(self, slider, thing):
        command_bytes = [int(slider.value)] * 8
        signs = [1, -1, 1, -1, 1, -1, 1, -1]
        command_bytes = [int(90 + s*(x - 90))
                         for (s, x) in zip(signs, command_bytes)]
        command_data = PhoneBotCommandEncoder.encodeCommand(
            PhoneBotCommand.SET_LEG_POSITIONS, command_bytes)
        if self.ble_service_ is not None:
            print("Sending Data!")
            self.ble_service_.sendData(command_data)
        print(slider.value)

    def on_start_demo(self, btn):
        btn_state = btn[0].state == "down"

        if btn_state:
            self.state_start_time = time.time()
            self.robot_.set_state('stand_up')
        else:
            self.robot_.set_state('sit_down')

    def on_pause(self):
        # check orientation listener is active.
        if self.orientation_listener_ is not None:
            self.orientation_listener_.unregisterListener()
        return super().on_pause()

    def on_resume(self):
        # check orientation listener is inactive.
        if self.orientation_listener_ is not None:
            self.orientation_listener_.registerListener()
        return super().on_resume()


if __name__ == '__main__':
    PhonebotApp().run()
