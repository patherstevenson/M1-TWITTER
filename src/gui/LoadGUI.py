from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("Twitter Analyse")

        # setting geometry
        self.setGeometry(100, 100, 800, 600)

        # calling method
        self.UiComponents()

        # showing all the widgets
        self.show()

    # method for components
    def UiComponents(self):

        # creating a push button
        push = QPushButton("Search", self)

        # setting geometry to the push button
        push.setGeometry(0, 0, 80, 20)

        list_query = QListWidget()
        list_query.setGeometry(50,50,50,50)

        # creating a label
        label = QLabel("A TWITT EXAMPLE", self)

        # setting geometry to the label
        label.setGeometry(100, 160, 200, 50)

        # setting alignment to the label
        label.setAlignment(Qt.AlignCenter)
        # font
        font = QFont("Arial", 12)
        # setting font to the label
        label.setFont(font)

        # setting style sheet to the label
        label.setStyleSheet("QLabel"
                            "{"
                            "border : 2px solid black;"
                            "background : white;"
                            "}")
        # hiding the label
        label.hide()

        # adding action method to the push button
        push.clicked.connect(lambda: do_something())
        # method called by the push button when pressed
        def do_something():
            # unhide the label
            label.show()

# create pyqt5 app
App = QApplication(sys.argv)

# setting cursor flashtime
App.setCursorFlashTime(100)

# setting application object name
App.setObjectName("GfG")

# setting application display name
App.setApplicationDisplayName("GfG PyQt5")

# create the instance of our Window
window = Window()

# start the app
sys.exit(App.exec())
