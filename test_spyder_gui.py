# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:36:26 2020

@author: campo
"""
import sys
from PyQt5.QtWidgets import (QWidget, QGridLayout, QPushButton, QApplication, QLabel, QMainWindow, QMessageBox)
from matplotlib.image import *
from PyQt5 import QtGui
from PyQt5 import QtCore

class Window(QMainWindow):
    
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50,500,300)
        self.setWindowTitle("TdLog")
        self.home()

    def home(self):
        btn = QPushButton("Quit", self)
        btn.clicked.connect()
        btn.resize(100,100)
        btn.move(100,100)
        self.show()

def run():        
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())