# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

# Module GUI (licence libre, absolument non commercial)

from __future__ import annotations

import sys

import numpy as np
from Xlib.Xcursorfont import left_side
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QApplication, QHBoxLayout,
                               QLabel, QMainWindow, QSlider,
                               QVBoxLayout, QTabWidget,
                               QPushButton, QWidget, QComboBox, QLineEdit)
import Robot

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(12, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)

class ERGO(QMainWindow):
    def __init__(self, R = None):
        super().__init__()
        self.setWindowTitle("ERGO kinematics simulator")


        self.R = R or Robot.Robot(6,
                                       t_j=np.array(['R', 'T', 'R', 'R', 'T', 'R']),
                                       theta=np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                                                                        0.0]),
                                       alpha=np.array([np.pi/2,
                                                                        -np.pi/2,
                                                                        0.0,
                                                                        np.pi/2,
                                                                        -np.pi/2,
                                                                        0.0]),
                                       a=np.array([0.0, 0.0, 4, 3, 0.0, 0.0]),
                                       d=np.array([5, 0.2, 0.0, 0.0, 1.0, 1.0]),
                                       x=0.0,
                                       y=0.0,
                                       z=0.0)
        self.initial_theta = self.R.theta
        self.initial_d = self.R.d

        self.theta_prec = np.zeros(self.R.nb_j, dtype=float)
        self.d_prec = np.zeros(self.R.nb_j, dtype=float)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab_widget.addTab(self.tab1, "DH Selector")
        self.tab_widget.addTab(self.tab2, "Visualisation")
        self.layout.addWidget(self.tab_widget)

        self.initTab1()
        self.initTab2()

        # Affichage initial
        self.update_plot()

    def initTab1(self):
        layout = QVBoxLayout(self.tab1)
        self.dh_selectors = []
        for i in range(self.R.nb_j):
            hbox = QHBoxLayout()

            # Sélecteur du type de joint
            type_label = QLabel(f"Joint {i + 1} type:")
            type_selector = QComboBox()
            type_selector.addItems(["R", "T"])
            type_selector.setCurrentText(self.R.type_j[i])
            hbox.addWidget(type_label)
            hbox.addWidget(type_selector)

            # Paramètres de saisie DH
            for param, val in zip(["theta", "d", "a", "alpha"],
                                  [self.R.theta[i], self.R.d[i], self.R.a[i], self.R.alpha[i]]):
                param_label = QLabel(f"{param}:")
                param_input = QLineEdit(str(val))
                hbox.addWidget(param_label)
                hbox.addWidget(param_input)

            # Limites des joints
            min_label = QLabel("Min:")
            min_input = QLineEdit("0")
            max_label = QLabel("Max:")
            max_input = QLineEdit("360" if self.R.type_j[i] == "R" else "20")
            hbox.addWidget(min_label)
            hbox.addWidget(min_input)
            hbox.addWidget(max_label)
            hbox.addWidget(max_input)
            layout.addLayout(hbox)
            self.dh_selectors.append((type_selector, param_input, min_input, max_input))

        # Application des changements
        apply_btn = QPushButton("Apply DH Parameters")
        apply_btn.clicked.connect(self.apply_dh_parameters)
        layout.addWidget(apply_btn)


    def initTab2(self):
        layout = QVBoxLayout(self.tab2)
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)

        # Sliders layout
        self.sliders = []
        sliders_layout = QVBoxLayout()
        for i in range(self.R.nb_j):
            hbox = QHBoxLayout()
            if (self.R.type_j[i] == 'R'):
                label = QLabel(f"Theta {i + 1}")
            elif (self.R.type_j[i] == 'T'):
                label = QLabel(f"d {i + 1}")

            hbox.addWidget(label)
            slider = QSlider(Qt.Horizontal)

            if (self.R.type_j[i] == 'R'):
                slider.setRange(0, 360)
                slider.setValue(0)
                slider.setSingleStep(1)
                self.theta_prec[i] = slider.value()
            elif (self.R.type_j[i] == 'T'):
                slider.setRange(0, 20)
                slider.setValue(0)
                slider.setSingleStep(1)
                self.d_prec[i] = slider.value()

            slider.valueChanged.connect(self.update_plot)
            self.sliders.append(slider)
            hbox.addWidget(slider)
            sliders_layout.addLayout(hbox)
        layout.addLayout(sliders_layout)

    # Slots
    @Slot()
    def update_plot(self):
        # mise à jour des valeurs d'angles et de distance
        for i in range(self.R.nb_j):
            if (self.R.type_j[i] == 'R'):
                value = np.radians(self.sliders[i].value())
                delta = value - self.theta_prec[i]
                if value>self.theta_prec[i]:
                    self.R.set_theta(self.initial_theta[i] + delta , i)
                    self.theta_prec[i] = value

                elif (value == self.theta_prec[i]):
                    self.R.set_theta(self.R.theta[i], i)

                else:
                    self.R.set_theta(self.initial_theta[i] + delta , i)
                    self.theta_prec[i] = value
            elif (self.R.type_j[i] == 'T'):
                value = self.sliders[i].value()/10.0
                delta = value - self.d_prec[i]
                if value>self.d_prec[i]:
                    self.R.set_d(self.initial_d[i] + delta , i)
                    self.d_prec[i] = value

                elif (value == self.d_prec[i]):
                    self.R.set_d(self.R.d[i], i)

                else:
                    self.R.set_d(self.initial_d[i] + delta , i)
                    self.d_prec[i] = value

        # replot
        self.canvas.ax.clear()
        x_pts, y_pts, z_pts, _, quiv1, quiv2, quiv3 = self.R.cin_num_comp()

        for x, y, z in zip(x_pts, y_pts, z_pts):
            self.canvas.ax.plot(x, y, z, marker='o')

        # Quiver (orientation)
        self.canvas.ax.quiver(*quiv1, color='r', label="X", length = 5)
        self.canvas.ax.quiver(*quiv2, color='g', label="Y", length = 5)
        self.canvas.ax.quiver(*quiv3, color='b', label="Z", length = 5)

        self.canvas.ax.set_xlim(-10, 10)
        self.canvas.ax.set_ylim(-10, 10)
        self.canvas.ax.set_zlim(0, 20)
        self.canvas.ax.set_xlabel("X")
        self.canvas.ax.set_ylabel("Y")
        self.canvas.ax.set_zlabel("Z")
        self.canvas.ax.set_title("Robot Kinematics")
        self.canvas.ax.legend()
        self.canvas.draw()

    # en cours d'écriture
    @Slot()
    def apply_dh_parameters(self):
        for i, (type_selector, param_input, min_input, max_input) in enumerate(self.dh_selectors):
            self.R.type_j[i] = type_selector.currentText()
            if self.R.type_j[i] == 'R':
                self.R.theta[i] = np.radians(float(param_input.text()))
                self.R.d[i] = 0.0
            elif self.R.type_j[i] == 'T':
                self.R.d[i] = float(param_input.text())
                self.R.theta[i] = 0.0
        self.update_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ERGO()
    window.setFixedSize(1280, 800)
    window.show()
    sys.exit(app.exec())