from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget,QPushButton
from PyQt5 import QtWidgets,QtCore


import numpy as np

from fjsp import fjsp, optimize
from qtui.first import Ui_Form

import matplotlib
matplotlib.use("Qt5Agg") #声明使用pyqt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg #pyqt5的画布
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class MyMatplotlibFigure(FigureCanvasQTAgg):
    """
    创建一个画布类，并把画布放到FigureCanvasQTAgg
    """
    def __init__(self, width, heigh, dpi):
        # plt.rcParams['figure.facecolor'] = 'r'  # 设置窗体颜色
        # plt.rcParams['axes.facecolor'] = 'b'  # 设置绘图区颜色
        # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        # 这里还要注意，width, heigh可以直接调用参数，不能用self.width、self.heigh作为变量获取，因为self.width、self.heigh 在模块中已经FigureCanvasQTAgg模块中使用，这里定义会造成覆盖
        self.figs = Figure(figsize=(width, heigh),dpi=dpi)
        super(MyMatplotlibFigure, self).__init__(self.figs)  # 在父类种激活self.fig， 否则不能显示图像（就是在画板上放置画布）
        self.axes = self.figs.add_subplot(111)  # 添加绘图区





class mywindow(QWidget):

    my_signal = pyqtSignal(str) #创建自己想设置的信号


    def __init__(self):
        super().__init__()

        self.f = fjsp()

        self.init_ui()



    def init_ui(self):
        #导入ui转换的文件
        self.m_ui = Ui_Form()
        self.m_ui.setupUi(self)

        self.m_ui.comboBoxAl.activated.connect(self.selectal) #通过该方式链接信号和槽函数  算法选择
        self.m_ui.comboBoxIns.activated.connect(self.selectIns)

        self.m_ui.pushButtonrun.clicked.connect(self.run)  #运行算法

        self.m_ui.pushButtonclear.clicked.connect(self.output_clear) #清理输出信息

        #画布初始化 甘特图
        self.canvas = MyMatplotlibFigure(width=6, heigh=4, dpi=100)
        #设置图片大小
        width, height = self.m_ui.graphicsView.width(), self.m_ui.graphicsView.height()
        self.canvas.resize(width, height)

        # self.plotcos(self.canvas)
        graphicscene = QtWidgets.QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        graphicscene.addWidget(self.canvas)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.m_ui.graphicsView.setScene(graphicscene)

        #迭代曲线
        self.canvas_2 = MyMatplotlibFigure(width=3.5, heigh=3.5, dpi=100)

        width, height = self.m_ui.graphicsView_2.width(), self.m_ui.graphicsView_2.height()
        self.canvas_2.resize(width, height)

        # self.plotcos(self.canvas_2)
        graphicscene_2 = QtWidgets.QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        graphicscene_2.addWidget(self.canvas_2)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.m_ui.graphicsView_2.setScene(graphicscene_2)

        # self.hboxlayout = QtWidgets.QHBoxLayout(self.m_ui.graphicsView)
        # self.hboxlayout.addWidget(self.canvas)

        #创建新的线程
        self.thread_optimize = optimize(self.f, self)
        self.thread_optimize.optimi_finished.connect(self.picture_flush)

    def plotcos(self, canvas):
        # plt.clf()
        t = np.arange(0.0, 5.0, 0.01)
        s = np.cos(2 * np.pi * t)
        canvas.axes.plot(t, s)
        # canvas.figs.suptitle("sin")  # 设置标题

    def draw_gant(self):
        self.f.draw_gantte(self.canvas.axes)

    def draw_iter(self):
        self.f.draw_iter(self.canvas_2.axes)

    def picture_flush(self):

        #甘特图刷新
        print("优化完成")
        self.canvas.axes.cla()
        self.draw_gant()
        self.canvas.draw()
        # self.canvas.figs.canvas.flush_events()


        #迭代曲线刷新
        self.canvas_2.axes.cla()
        self.draw_iter()
        self.canvas_2.draw()
        # self.canvas_2.figs.canvas.flush_events()


        #最优解输出
        self.m_ui.textBrowser.append("输出最终解：" + str(self.f.best_individual))

    def selectal(self):  #算法选择

        self.f.selectal(self.m_ui.comboBoxAl.currentText())  #获取选择的算法名称
        self.m_ui.textBrowser.append("选择算法: "+ self.m_ui.comboBoxAl.currentText()) #文本输出信息追加到下一行
        #self.m_ui.textBrowser.setText("选择算法: "+ str(self.f.algorithm)) # 在ui中输出算法选择



    def selectIns(self): #算例选择
        self.f.selectIns(self.m_ui.comboBoxIns.currentIndex()) #获取选择算例索引
        self.m_ui.textBrowser.append("选择算例: " + self.m_ui.comboBoxIns.currentText())




    def output_clear(self):
        self.m_ui.textBrowser.clear()


    def run(self):  #算法运行
        self.thread_optimize.start()








