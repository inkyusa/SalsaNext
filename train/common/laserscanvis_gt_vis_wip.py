#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
import cv2


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, pred_label_names, gt_label_names, offset=0, semantics=True, instances=False):
    self.scan_pred = scan
    self.scan_gt = scan
    self.scan_names = scan_names
    self.pred_label_names = pred_label_names
    self.gt_label_names = gt_label_names
    self.offset = offset
    self.semantics = semantics
    self.instances = instances
    self.timer = vispy.app.Timer('auto', iterations=1e9, connect=self.on_timer, start=False)
    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    # add semantics
    if self.semantics:
      print("Using pred semantics in visualizer")
      self.sem_pred_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.sem_pred_view, 0, 1)
      self.sem_pred_vis = visuals.Markers()
      self.sem_pred_view.camera = 'turntable'
      self.sem_pred_view.add(self.sem_pred_vis)
      visuals.XYZAxis(parent=self.sem_pred_view.scene)
      # self.sem_view.camera.link(self.scan_view.camera)

      # print("Using gt in visualizer")
      # self.sem_gt_view = vispy.scene.widgets.ViewBox(
      #     border_color='white', parent=self.canvas.scene)
      # self.grid.add_widget(self.sem_gt_view, 0, 2)
      # self.sem_gt_vis = visuals.Markers()
      # self.sem_gt_view.camera = 'turntable'
      # self.sem_gt_view.add(self.sem_gt_vis)
      # visuals.XYZAxis(parent=self.sem_gt_view.scene)
      # self.sem_view.camera.link(self.scan_view.camera)

    if self.instances:
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 3)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.addr
    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024 
    self.canvas_H = 64
    if self.semantics:
      self.multiplier += 1
    if self.instances:
      self.multiplier += 1

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add semantics
    if self.semantics:
      self.sem_pred_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_pred_img_view, 1, 0)
      self.sem_pred_img_vis = visuals.Image(cmap='viridis')
      self.sem_pred_img_view.add(self.sem_pred_img_vis)

      # self.sem_gt_img_view = vispy.scene.widgets.ViewBox(
      #     border_color='white', parent=self.img_canvas.scene)
      # self.img_grid.add_widget(self.sem_gt_img_view, 2, 0)
      # self.sem_gt_img_vis = visuals.Image(cmap='viridis')
      # self.sem_gt_img_view.add(self.sem_gt_img_vis)

    # add instances
    if self.instances:
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 3, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    self.scan_pred.open_scan(self.scan_names[self.offset])
    self.scan_gt.open_scan(self.scan_names[self.offset])
    if self.semantics:
      self.scan_pred.open_label(self.pred_label_names[self.offset])
      self.scan_pred.colorize()
      self.scan_gt.open_label(self.gt_label_names[self.offset])
      self.scan_gt.colorize()

    # then change names
    title = "scan " + str(self.offset) + " of " + str(len(self.scan_names)-1)
    self.canvas.title = title
    self.img_canvas.title = title

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan_pred.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.scan_pred.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=1)

    # plot semantics
    if self.semantics:
      self.sem_pred_vis.set_data(self.scan_pred.points,
                            face_color=self.scan_pred.sem_label_color[..., ::-1],
                            edge_color=self.scan_pred.sem_label_color[..., ::-1],
                            size=1)
      # self.sem_gt_vis.set_data(self.scan_gt.points,
      #                       face_color=self.scan_gt.sem_label_color[..., ::-1],
      #                       edge_color=self.scan_gt.sem_label_color[..., ::-1],
      #                       size=1)

    # plot instances
    if self.instances:
      self.inst_vis.set_data(self.scan.points,
                             face_color=self.scan_pred.inst_label_color[..., ::-1],
                             edge_color=self.scan_pred.inst_label_color[..., ::-1],
                             size=1)

    # now do all the range image stuff
    # plot range image* self.multiplier
    data = np.copy(self.scan_pred.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())self.pred_label_names = pred_label_names
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    # print(data.max(), data.min())
    self.img_vis.set_data(data)
    self.img_vis.update()

    if self.semantics:
      self.sem_pred_img_vis.set_data(self.scan_pred.proj_sem_color[..., ::-1])
      self.sem_pred_img_vis.update()
      # self.sem_gt_img_vis.set_data(self.scan_gt.proj_sem_color[..., ::-1])
      # self.sem_gt_img_vis.update()

    if self.instances:
      self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
      self.inst_img_vis.update()

  # attaching of isoline filter via timer
  def on_timer(self, event):
      #image.attach(iso)
      self.offset += 1
      if self.offset >= len(self.scan_names):
            self.offset = 0
      self.update_scan()
      #print(f"offset = {self.offset}")

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= len(self.scan_names):
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset <= 0:
        self.offset = len(self.scan_names)-1
      self.update_scan()
    elif event.key == ' ':
      if self.timer.running:
        self.timer.stop()
      else:
        self.timer.start()
        
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
