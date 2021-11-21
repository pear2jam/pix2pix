from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.resize(800, 530)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        # font for buttons
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 20, 161, 51))
        self.pushButton.setSizeIncrement(QtCore.QSize(0, 0))
        self.pushButton.setFont(font)
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setStyleSheet("background : rgb(250, 255, 233); border-radius : 50; border : 2px solid black")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.draw_set)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 170, 320, 320))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(400, 300))
        self.label.pixmap().fill(QColor(255, 255, 255))
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.label.setStyleSheet("border : 2px solid black; background-color: white")
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(440, 170, 320, 320))
        self.label_2.setText("")
        self.label_2.pixmap = QPixmap()
        self.label_2.pixmap.fill(QColor(255, 255, 255))
        self.label_2.setStyleSheet("border : 2px solid black; background-color: white")
        self.label_2.setObjectName("label_2")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(230, 20, 161, 51))
        self.pushButton_2.setSizeIncrement(QtCore.QSize(0, 0))
        self.pushButton_2.clicked.connect(self.erase_set)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setAutoFillBackground(False)
        self.pushButton_2.setStyleSheet("background : rgb(250, 255, 233); border-radius : 50; border : 2px solid black")
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(410, 20, 161, 51))
        self.pushButton_3.setSizeIncrement(QtCore.QSize(0, 0))
        self.pushButton_3.setFont(font)
        self.pushButton_3.setAutoFillBackground(False)
        self.pushButton_3.setStyleSheet("background : rgb(250, 255, 233); border-radius : 50; border : 2px solid black")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.clear_canvas)

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(590, 20, 161, 51))
        self.pushButton_4.setSizeIncrement(QtCore.QSize(0, 0))
        self.pushButton_4.setFont(font)
        self.pushButton_4.setAutoFillBackground(False)
        self.pushButton_4.setStyleSheet("background : rgb(255, 233, 179); border-radius : 50; border : 2px solid black")
        self.pushButton_4.setObjectName("pushButton_4")
        #font for labels
        font = QtGui.QFont()
        font.setPointSize(21)

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(160, 110, 161, 51))
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(550, 110, 161, 51))
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.setCentralWidget(self.centralwidget)

        self.painter = QtGui.QPainter(self.label.pixmap())
        self.pen = QPen(QtCore.Qt.black)
        self.pen.setWidth(3)
        self.painter.setPen(self.pen)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pix2pix"))
        self.pushButton.setText(_translate("MainWindow", "Draw"))
        self.pushButton_2.setText(_translate("MainWindow", "Erase"))
        self.pushButton_3.setText(_translate("MainWindow", "Clear"))
        self.pushButton_4.setText(_translate("MainWindow", "Run"))
        self.label_3.setText(_translate("MainWindow", "Input"))
        self.label_4.setText(_translate("MainWindow", "Output"))

    def draw_set(self):
        self.pen = QPen(QtCore.Qt.black)
        self.pen.setWidth(3)

        self.painter.setPen(self.pen)

    def erase_set(self):
        self.pen = QPen(QtCore.Qt.white)
        self.pen.setWidth(18)
        self.painter.setPen(self.pen)

    def clear_canvas(self):
        self.label.pixmap().fill(QColor(255, 255, 255))
        self.label.update()

    def mouseMoveEvent(self, e):
        self.painter.drawPoint(e.x()-40, e.y()-180)
        self.update()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())