#!/usr/bin/env python3

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform

import numpy as np
from camera import Camera2

import time

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera2:
        id: camera
        resolution: (640, 480) # width, height
        play: False
        canvas.before:
            PushMatrix
            Rotate:
                angle: 180
				origin: self.center
				# y: -1
        canvas.after:
            PopMatrix
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
''')

class CameraClick(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._request_android_permissions()

    @staticmethod
    def is_android():
        return platform == 'android'

    def _request_android_permissions(self):
        """
        Requests CAMERA permission on Android.
        """
        if not self.is_android():
            return
        from android.permissions import request_permission, Permission
        request_permission(Permission.CAMERA)

class PhonebotApp(App):
    def build(self):
        return CameraClick()
            
if __name__ == '__main__':
    PhonebotApp().run()
