#
#BSD 3-Clause License
#
#
#
#Copyright 2022 fortiss, Neuromorphic Computing group
#
#
#All rights reserved.
#
#
#
#Redistribution and use in source and binary forms, with or without
#
#modification, are permitted provided that the following conditions are met:
#
#
#
#* Redistributions of source code must retain the above copyright notice, this
#
#  list of conditions and the following disclaimer.
#
#
#
#* Redistributions in binary form must reproduce the above copyright notice,
#
#  this list of conditions and the following disclaimer in the documentation
#
#  and/or other materials provided with the distribution.
#
#
#
#* Neither the name of the copyright holder nor the names of its
#
#  contributors may be used to endorse or promote products derived from
#
#  this software without specific prior written permission.
#
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#


import mujoco
import glfw
import sys
from threading import Lock
import numpy as np
import time
import imageio
import cv2
from pathlib import Path

from .utils.esim import Esim_interface 
from upsampler import Upsampler

class MujocoViewer:
    def __init__(self, model, data, headless=False, render_every_frame=True, running_events=False, win_size=None):
        self.model = model
        self.data = data

        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._transparent = False
        self._contacts = False
        self._render_every_frame = render_every_frame
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menu = False

        # additions for the new feartures
        self._last_img = None
        self._esim = None
        self._upsampler = None
        self.count = 0

        # glfw init
        glfw.init()
        # if running_events:
        #     width, height = 128, 128
        # else:
            # width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
            # if win_size is not None:
            #     width, height = win_size[0]*2, win_size[1]*2
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        

        if headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(
            width // 2, height // 2, "mujoco", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        self._last_left_click_time = None
        self._last_right_click_time = None

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # get viewport
        self.viewport = mujoco.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []

    def iterateCameraView(self):
        self.cam.fixedcamid += 1
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def toggleBool(self, val):
        return not val

    def mulSimSpeed(self, val):
        self._run_speed *= val

    def winShape(self):
        return glfw.get_framebuffer_size(self.window)

    def emptyViewImg(self):
        window_shape = glfw.get_framebuffer_size(self.window)
        shape = window_shape[1], window_shape[0], 3
        return np.zeros(shape, dtype=np.uint8)
    
    def renderImg(self):
        img = self.emptyViewImg()
        mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
        return np.flipud(img)
    
    def toggleVOption(self, val):
        self.vopt.frame = val - self.vopt.frame

    def toogleTransparancy(self):
        self._transparent = self.toggleBool(self._transparent)
        multiplier = 5.0
        if self._transparent:
            self.model.geom_rgba[:, 3] /= multiplier
        else:
            self.model.geom_rgba[:, 3] *= multiplier

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_TAB:
            self.iterateCameraView()
        elif key == glfw.KEY_SPACE:
            self._paused = self.toggleBool(self._paused)
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT:
            self._advance_by_one_step = True
            self._paused = True
        elif key == glfw.KEY_S:
            self.mulSimSpeed(0.5)
        elif key == glfw.KEY_F:
            self.mulSimSpeed(2)
        elif key == glfw.KEY_D:
            self._render_every_frame = self.toggleBool(self._render_every_frame)
        # Capture screenshot
        elif key == glfw.KEY_T:
            imageio.imwrite(self._image_path % self._image_idx, self.renderImg())
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = self.toggleBool(self._contacts)
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        # Display coordinate frames
        elif key == glfw.KEY_E:
            self.toggleVOption(1)
        elif key == glfw.KEY_B:
            self.toggleVOption(3)
        elif key == glfw.KEY_H:
            self._hide_menu = self.toggleBool(self._hide_menu)
        # Make transparent
        elif key == glfw.KEY_R:
            self.toogleTransparancy()
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        # Quit
        if key == glfw.KEY_ESCAPE:
            self.close()

    def close(self):
        glfw.destroy_window(self.window)

    def isRLMousePressed(self):
        return self._button_left_pressed or self._button_right_pressed
    
    def isPressed(self, key):
        return glfw.get_key(self.window, key) == glfw.PRESS

    def isShiftPressed(self):
        return self.isPressed(glfw.KEY_LEFT_SHIFT) or self.isPressed(glfw.KEY_RIGHT_SHIFT)
    
    def currentAction(self):
        mod_shift = self.isShiftPressed()
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        
        return action

    def _cursor_pos_callback(self, window, x, y):
        if not self.isRLMousePressed():
            return
        
        action = self.currentAction()

        width, height = glfw.get_framebuffer_size(window)
        dx = self._scale * (x - self._last_mouse_x) / height
        dy = self._scale * (y - self._last_mouse_y) / height

        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx,
                    dy,
                    self.scn,
                    self.pert)
            else:
                mujoco.mjv_moveCamera(
                    self.model,
                    action,
                    dx,
                    dy,
                    self.scn,
                    self.cam)

        self.saveLastMouseCoord(x, y)

    def saveLastMouseCoord(self, x, y):
        self._last_mouse_x = x
        self._last_mouse_y = y

    def isButton(self, button, buttonType, act):
        return button == buttonType and act == glfw.PRESS
    
    def isDoubleMouseClick(self, _button_left_pressed, _last_left_click_time):
        _left_double_click_pressed = False
        if _button_left_pressed:
            time_now = glfw.get_time()

            if _last_left_click_time is None:
                _last_left_click_time = glfw.get_time()

            time_diff = (time_now - _last_left_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                _left_double_click_pressed = True

            _last_left_click_time = time_now

        return _left_double_click_pressed, _last_left_click_time
    
    def handlePerturbation(self, key):
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(
                    self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb

    def _mouse_button_callback(self, window, button, act, mods):
        x, y = glfw.get_cursor_pos(window)
        self.saveLastMouseCoord(x, y)

        self._button_left_pressed = self.isButton(button, glfw.MOUSE_BUTTON_LEFT, act)
        self._button_right_pressed = self.isButton(button, glfw.MOUSE_BUTTON_RIGHT, act)

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        self.handlePerturbation(key)

        # handle doubleclick
        self._left_double_click_pressed, self._last_left_click_time = self.isDoubleMouseClick(self._button_left_pressed, self._last_left_click_time)
        self._right_double_click_pressed, self._last_right_click_time = self.isDoubleMouseClick(self._button_right_pressed, self._last_right_click_time)

        if self._left_double_click_pressed or self._right_double_click_pressed:
            # determine selection mode
            selmode = 0
            if self._left_double_click_pressed:
                selmode = 1
            if self._right_double_click_pressed:
                selmode = 2
            if self._right_double_click_pressed and key:
                selmode = 3

            # find geom and 3D click point, get corresponding body
            selbody, selpnt, selskin = self.selectedBody(x, y)

            # set lookat point, start tracking is requested
            if selmode == 2 or selmode == 3:
                # set cam lookat
                if selbody >= 0:
                    self.cam.lookat = selpnt.flatten()
                # switch to tracking camera if dynamic body clicked
                if selmode == 3 and selbody > 0:
                    self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    self.cam.trackbodyid = selbody
                    self.cam.fixedcamid = -1
            # set body selection
            else:
                if selbody >= 0:
                    # record selection
                    self.pert.select = selbody
                    self.pert.skinselect = selskin
                    # compute localpos
                    vec = selpnt.flatten() - self.data.xpos[selbody]
                    mat = self.data.xmat[selbody].reshape(3, 3)
                    self.pert.localpos = self.data.xmat[selbody].reshape(
                        3, 3).dot(vec)
                else:
                    self.pert.select = 0
                    self.pert.skinselect = -1
            # stop perturbation on select
            self.pert.active = 0

        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def selectedBody(self, x, y):
        width, height = self.viewport.width, self.viewport.height
        aspectratio = width / height
        relx = x / width
        rely = (self.viewport.height - y) / height
        selpnt = np.zeros((3, 1), dtype=np.float64)
        selgeom = np.zeros((1, 1), dtype=np.int32)
        selskin = np.zeros((1, 1), dtype=np.int32)
        return mujoco.mjv_select(
                    self.model,
                    self.data,
                    self.vopt,
                    aspectratio,
                    relx,
                    rely,
                    self.scn,
                    selpnt,
                    selgeom,
                    selskin), selpnt, selskin


    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

        return
    
    def add_overlay(self, gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

    def createTopLeftOverlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT

        if self._render_every_frame:
            self.add_overlay(topleft, "", "")
        else:
            self.add_overlay(
                topleft,
                "Run speed = %.3f x real time" %
                self._run_speed,
                "[S]lower, [F]aster")
            
        self.add_overlay(
            topleft,
            "Ren[d]er every frame",
            "On" if self._render_every_frame else "Off")
        self.add_overlay(
            topleft, "Switch camera (#cams = %d)" %
            (self.model.ncam + 1), "[Tab] (camera ID = %d)" %
            self.cam.fixedcamid)
        self.add_overlay(
            topleft,
            "[C]ontact forces",
            "On" if self._contacts else "Off")
        self.add_overlay(
            topleft,
            "T[r]ansparent",
            "On" if self._transparent else "Off")
        if self._paused is not None:
            if not self._paused:
                self.add_overlay(topleft, "Stop", "[Space]")
            else:
                self.add_overlay(topleft, "Start", "[Space]")
                self.add_overlay(
                    topleft,
                    "Advance simulation by one step",
                    "[right arrow]")
        self.add_overlay(
            topleft,
            "Referenc[e] frames",
            "On" if self.vopt.frame == 1 else "Off")
        self.add_overlay(topleft, "[H]ide Menu", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            self.add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            self.add_overlay(topleft, "Cap[t]ure frame", "")
        self.add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

    def createBottomLeft(self):
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        self.add_overlay(
            bottomleft, "FPS", "%d%s" %
            (1 / self._time_per_render, ""))
        self.add_overlay(
            bottomleft, "Solver iterations", str(
                self.data.solver_iter + 1))
        self.add_overlay(
            bottomleft, "Step", str(
                round(
                    self.data.time / self.model.opt.timestep)))
        self.add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)

    def _create_overlay(self):
        self.createTopLeftOverlay()
        self.createBottomLeft()

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def update(self, overlay_on):
            # fill overlay items
            if overlay_on:
                self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.terminate()
                sys.exit(0)

            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                # overlay items
                if not self._hide_menu:
                    for gridpos, [t1, t2] in self._overlay.items():
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.viewport,
                            t1,
                            t2,
                            self.ctx)
                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

            # clear overlay
            self._overlay.clear()


    def render(self, overlay_on=True):        
        if self._paused:
            while self._paused:
                self.update(overlay_on)
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                self.update(overlay_on)
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []

        # apply perturbation (should this come before mj_step?)
        self.apply_perturbations()

    def make_current(self):
        glfw.make_context_current(self.window)

    def set_sim_speed(self, x):
        self._run_speed = x

    def change_camera(self,fixedcamid):
        self.cam.fixedcamid = fixedcamid
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def get_frame(self, fixedcamid):
        self.change_camera(fixedcamid)
        
        img = self.renderImg()

        # first two images are all black. It is for avoiding to capture them
        if np.all(img == 0):
            return None

        return img

    def save_img(self, img, path):
        path += "/%08d.png" % self._image_idx
        imageio.imwrite(path, img)

        self._image_idx += 1

    def save_events(self, e, path):
        path += "/%010d.npz" % self._image_idx
        np.savez(path, **e)

    # capture camera frame of a specified camera id and return img array and write in /tmp
    def capture_frame(self, fixedcamid, timestamp, save_it=False, path="/tmp"):
        img = self.get_frame(fixedcamid)
        if img is None:
            return None
        
        if save_it:
            raw_imgs_path = path + "/raw_imgs"
            Path(raw_imgs_path).mkdir(parents=True, exist_ok=True)
            self.save_img(img, raw_imgs_path)

            with open(path + "/timestamps.txt", "a") as f:
                f.write(str(timestamp)+"\n")

        return img


    # capture camera event of a specified camera id and return img array and write in /tmp (Prototype)
    def capture_event_prototype(self, fixedcamid, save_it=False, path="/tmp"):
        img_original = self.get_frame(fixedcamid)

        if img_original is None:
            return None

        gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        if self._last_img is None:
            self._last_img = gray_img
            return None
        else:
            img_sub = gray_img - self._last_img
            img = np.where(np.abs(img_sub) < 10, 0, 255).astype(np.uint8)
            self._last_img = gray_img

        if save_it:
            raw_imgs_path = path + "/sub_imgs"
            Path(raw_imgs_path).mkdir(parents=True, exist_ok=True)
            self.save_img(img, raw_imgs_path)

        return img

    # capture camera event of a specified camera id and return img array and write in /tmp
    def capture_event(self, fixedcamid, timestamp, save_it=False, path="/tmp"):
        img_original = self.get_frame(fixedcamid)

        if img_original is None:
            return None

        return self.process_img(img_original, timestamp, save_it, path)

    def init_esim(self, contrast_threshold_negative=0.1, contrast_threshold_positive=0.5, refractory_period_ns=1):
        self._esim = Esim_interface(contrast_threshold_negative, contrast_threshold_positive, refractory_period_ns)

    def process_img(self, img_original, timestamp, save_it, path):
        if self._esim is None:
            raise ValueError("Init esim first")

        if self.count < 2:
            self.count += 1
            return None

        gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        np_timestamp = np.array([timestamp])
        out = self._esim.img2e(gray_img, np_timestamp)

        if out is None:
            return None

        # save order is important
        e_img, e, num_e = out
        
        
        if save_it:
            events_path = path + "/events"
            event_imgs_path = path + "/event_imgs"
            Path(events_path).mkdir(parents=True, exist_ok=True)
            Path(event_imgs_path).mkdir(parents=True, exist_ok=True)
            self.save_events(e, events_path)
            self.save_img(e_img, event_imgs_path)

        return e_img, e, num_e

    # capture camera event of a specified camera id and return img array and write in /tmp
    # not done
    def capture_event_with_upsampling(self, fixedcamid, timestamp, path="/tmp"):

        upsampled_images, upsampled_timestamps = None, None

        img_original = self.get_frame(fixedcamid)

        if img_original is None:
            return None

        if not self._upsampler:
            self._upsampler = Upsampler(img_original, timestamp)
            self.process_img(img_original, timestamp, path)
        else:
             upsampled_images, upsampled_timestamps = self._upsampler.upsample_adaptively(img_original, timestamp)

        e_img_list = []
        e_list = []
        for I, t in zip(upsampled_images, upsampled_timestamps):
            e_img, e = self.process_img(I, t, path)
            e_img_list.append(e_img)
            e_list.append(e)

        return e_img_list, e_list


    # def close_win(self):
    #     print("######################################")
    #     # glfw.terminate()
    #     glfw.destroy_window(self.window)
    #     # self.window = glfw.create_window

