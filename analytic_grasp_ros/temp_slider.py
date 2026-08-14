#!/usr/bin/env python3

import sys
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
)


def quaternion_from_euler(roll, pitch, yaw):
    """Convert roll/pitch/yaw in radians to quaternion."""

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class TransformPublisher(Node):

    def __init__(self):
        super().__init__('transform_slider_gui')

        self.tf_broadcaster = TransformBroadcaster(self)

        # ============================================================
        # DEFAULT TRANSFORM
        # ============================================================

        self.parent_frame = 'correction_frame'
        self.child_frame = 'camera_f_link'

        # Values in SI units / radians
        self.values_rad = {
            'x': 0.,
            'y': 0.,
            'z': 0.,
            'roll': 0.,
            'pitch': 0.,
            'yaw': 0.,
        }

        self.values = {
            'x': self.values_rad['x'],
            'y': self.values_rad['y'],
            'z': self.values_rad['z'],
            'roll': self.values_rad['roll']*180/math.pi,
            'pitch': self.values_rad['pitch']*180/math.pi,
            'yaw': self.values_rad['yaw']*180/math.pi,
        }

        # ============================================================
        # SLIDER LIMITS
        #
        # XYZ are meters.
        # RPY are degrees.
        # ============================================================

        # Limits
        self.limits = {
            'x': (self.values['x']-0.5, self.values['x']+0.5),
            'y': (self.values['y']-0.5, self.values['y']+0.5),
            'z': (self.values['z']-0.5, self.values['z']+0.5),
            'roll': (self.values['roll']-180.0, self.values['roll']+180.0),
            'pitch': (self.values['pitch']-180.0, self.values['pitch']+180.0),
            'yaw': (self.values['yaw']-180.0, self.values['yaw']+180.0),
        }

        # Save the initial transform so Reset returns here.
        self.initial_parent_frame = self.parent_frame
        self.initial_child_frame = self.child_frame
        self.initial_values = self.values.copy()

        # ============================================================
        # TF PUBLISHER
        # ============================================================

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_transform
        )

    def publish_transform(self):

        msg = TransformStamped()

        msg.header.stamp = self.get_clock().now().to_msg()

        parent = self.parent_frame.strip().lstrip('/')
        child = self.child_frame.strip().lstrip('/')

        if not parent or not child:
            return

        msg.header.frame_id = parent
        msg.child_frame_id = child

        # self.values_rad = {
        #     'x': self.values['x'],
        #     'y': self.values['y'],
        #     'z': self.values['z'],
        #     'roll': self.values['roll']/180*math.pi,
        #     'pitch': self.values['pitch']/180*math.pi,
        #     'yaw': self.values['yaw']/180*math.pi,
        # }

        # Translation
        msg.transform.translation.x = self.values['x']
        msg.transform.translation.y = self.values['y']
        msg.transform.translation.z = self.values['z']

        # Rotation
        roll = math.radians(self.values['roll'])
        pitch = math.radians(self.values['pitch'])
        yaw = math.radians(self.values['yaw'])

        qx, qy, qz, qw = quaternion_from_euler(
            roll,
            pitch,
            yaw
        )

        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(msg)


class TransformSliderGUI(QWidget):

    def __init__(self, ros_node):

        super().__init__()

        self.ros_node = ros_node

        self.setWindowTitle('TF Slider GUI')
        self.resize(750, 500)

        self.controls = {}

        self.build_gui()

    def build_gui(self):

        main_layout = QVBoxLayout()

        # ============================================================
        # FRAMES
        # ============================================================

        frame_group = QGroupBox('Frames')
        frame_layout = QGridLayout()

        frame_layout.addWidget(
            QLabel('Parent frame:'),
            0,
            0
        )

        self.parent_edit = QLineEdit(
            self.ros_node.parent_frame
        )

        frame_layout.addWidget(
            self.parent_edit,
            0,
            1
        )

        frame_layout.addWidget(
            QLabel('Child frame:'),
            1,
            0
        )

        self.child_edit = QLineEdit(
            self.ros_node.child_frame
        )

        frame_layout.addWidget(
            self.child_edit,
            1,
            1
        )

        # Update TransformPublisher immediately when edited.
        self.parent_edit.textChanged.connect(
            self.parent_frame_changed
        )

        self.child_edit.textChanged.connect(
            self.child_frame_changed
        )

        frame_group.setLayout(frame_layout)

        main_layout.addWidget(frame_group)

        # ============================================================
        # TRANSFORM
        # ============================================================

        transform_group = QGroupBox('Transform')
        transform_layout = QGridLayout()

        transform_layout.addWidget(
            QLabel('Axis'),
            0,
            0
        )

        transform_layout.addWidget(
            QLabel('Slider'),
            0,
            1
        )

        transform_layout.addWidget(
            QLabel('Value'),
            0,
            2
        )

        rows = [
            ('x', 'X', 'm'),
            ('y', 'Y', 'm'),
            ('z', 'Z', 'm'),
            ('roll', 'Roll', 'deg'),
            ('pitch', 'Pitch', 'deg'),
            ('yaw', 'Yaw', 'deg'),
        ]

        for row, (key, name, unit) in enumerate(rows, start=1):

            transform_layout.addWidget(
                QLabel(f'{name} ({unit})'),
                row,
                0
            )

            slider = QSlider(Qt.Horizontal)

            # --------------------------------------------------------
            # Slider resolution
            # --------------------------------------------------------

            if key in ('x', 'y', 'z'):
                scale = 1000
            else:
                scale = 10

            slider_min = int(
                self.ros_node.limits[key][0] * scale
            )

            slider_max = int(
                self.ros_node.limits[key][1] * scale
            )

            slider.setMinimum(slider_min)
            slider.setMaximum(slider_max)

            # --------------------------------------------------------
            # Spinbox
            # --------------------------------------------------------

            spinbox = QDoubleSpinBox()

            spinbox.setRange(
                self.ros_node.limits[key][0],
                self.ros_node.limits[key][1]
            )

            if key in ('x', 'y', 'z'):
                spinbox.setDecimals(3)
                spinbox.setSingleStep(0.01)
            else:
                spinbox.setDecimals(1)
                spinbox.setSingleStep(1.0)

            spinbox.setSuffix(f' {unit}')

            # --------------------------------------------------------
            # INITIAL VALUE
            # --------------------------------------------------------

            initial_value = self.ros_node.values[key]

            spinbox.setValue(initial_value)
            slider.setValue(
                int(round(initial_value * scale))
            )

            # --------------------------------------------------------
            # Store controls
            # --------------------------------------------------------

            self.controls[key] = {
                'slider': slider,
                'spinbox': spinbox,
                'scale': scale,
            }

            # --------------------------------------------------------
            # Connections
            # --------------------------------------------------------

            slider.valueChanged.connect(
                lambda value, k=key:
                self.slider_changed(k, value)
            )

            spinbox.valueChanged.connect(
                lambda value, k=key:
                self.spinbox_changed(k, value)
            )

            # --------------------------------------------------------
            # Layout
            # --------------------------------------------------------

            transform_layout.addWidget(
                slider,
                row,
                1
            )

            transform_layout.addWidget(
                spinbox,
                row,
                2
            )

        transform_group.setLayout(transform_layout)

        main_layout.addWidget(transform_group)

        # ============================================================
        # BUTTONS
        # ============================================================

        button_layout = QHBoxLayout()

        reset_button = QPushButton('Reset')

        reset_button.clicked.connect(
            self.reset
        )

        button_layout.addWidget(reset_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    # ================================================================
    # FRAME CALLBACKS
    # ================================================================

    def parent_frame_changed(self, value):

        self.ros_node.parent_frame = value

    def child_frame_changed(self, value):

        self.ros_node.child_frame = value

    # ================================================================
    # SLIDER -> VALUE
    # ================================================================

    def slider_changed(self, key, value):

        scale = self.controls[key]['scale']

        actual_value = value / scale

        self.ros_node.values[key] = actual_value

        spinbox = self.controls[key]['spinbox']

        spinbox.blockSignals(True)
        spinbox.setValue(actual_value)
        spinbox.blockSignals(False)

    # ================================================================
    # VALUE -> SLIDER
    # ================================================================

    def spinbox_changed(self, key, value):

        self.ros_node.values[key] = value

        slider = self.controls[key]['slider']
        scale = self.controls[key]['scale']

        slider_value = int(round(value * scale))

        slider.blockSignals(True)
        slider.setValue(slider_value)
        slider.blockSignals(False)

    # ================================================================
    # RESET
    # ================================================================

    def reset(self):

        # Restore frame names
        self.parent_edit.blockSignals(True)
        self.child_edit.blockSignals(True)

        self.parent_edit.setText(
            self.ros_node.initial_parent_frame
        )

        self.child_edit.setText(
            self.ros_node.initial_child_frame
        )

        self.parent_edit.blockSignals(False)
        self.child_edit.blockSignals(False)

        self.ros_node.parent_frame = (
            self.ros_node.initial_parent_frame
        )

        self.ros_node.child_frame = (
            self.ros_node.initial_child_frame
        )

        # Restore XYZ/RPY
        for key, initial_value in self.ros_node.initial_values.items():

            self.ros_node.values[key] = initial_value

            slider = self.controls[key]['slider']
            spinbox = self.controls[key]['spinbox']
            scale = self.controls[key]['scale']

            slider.blockSignals(True)
            spinbox.blockSignals(True)

            slider.setValue(
                int(round(initial_value * scale))
            )

            spinbox.setValue(initial_value)

            slider.blockSignals(False)
            spinbox.blockSignals(False)


def main(args=None):

    rclpy.init(args=args)

    # TransformPublisher owns all defaults and limits.
    ros_node = TransformPublisher()

    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)

    # ROS executor runs in background.
    ros_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )

    ros_thread.start()

    # Qt runs in main thread.
    app = QApplication(sys.argv)

    gui = TransformSliderGUI(ros_node)
    gui.show()

    try:
        exit_code = app.exec_()

    finally:

        executor.shutdown()

        ros_node.destroy_node()

        rclpy.shutdown()

        ros_thread.join(timeout=1.0)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()