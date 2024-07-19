import sys
from PyQt5.QtWidgets import QApplication,QWidget, QPushButton, QLabel
from  Mywindow import mywindow
from fjsp import fjsp


if __name__ == '__main__':


    app = QApplication(sys.argv)
    w = mywindow()
    w.show()
    sys.exit(app.exec_())