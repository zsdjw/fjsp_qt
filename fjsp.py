from Instance import  instance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PyQt5.Qt import QThread, pyqtSignal


class fjsp():
    def __init__(self):

        self.algorithm = "GA"
        self.Instance = 0  #算例索引

        # 算法参数
        self.iter = 500

        #优化结果
        self.best_individual = None


        #车间信息类
        self.s = None

        #迭代过程
        self.ob_v = None


    #画图
    def draw_gantte(self, axes):
        A_num = self.s.A_num
        M_num = self.s.M_num
        P_num = self.s.P_num
        J_num = self.s.J_num

        y = range(1, A_num + M_num + P_num + 1, 1)
        label_job = ['JOB' + str(i) for i in range(J_num)]
        label_agv = ['AGV' + str(i) for i in range(A_num)]
        label_peo = ['PEO' + str(i) for i in range(P_num)]
        label_machin = ['M' + str(i + 1) for i in range(M_num)]
        label = label_machin + label_agv + label_peo
        colors = ['xkcd:cyan', 'xkcd:olive', 'xkcd:gray', 'xkcd:pink', 'xkcd:brown', 'xkcd:purple', 'xkcd:green',
                  'xkcd:orange', 'xkcd:blue', 'xkcd:tan']

        hatch = ['//', 'xx', 'oo', '||', '--', '++']

        for i in range(J_num):  # 画出各个工件的开始 结束时间
            for j in range(len(self.s.Jobs[i].start)):
                axes.barh(self.s.Jobs[i].machines[j] + 1, self.s.Jobs[i].end[j] - self.s.Jobs[i].start[j],
                         left=self.s.Jobs[i].start[j], color=colors[i], edgecolor='black')
                # plt.text(x = ga.s[index].Jobs[i].start[j] + ((ga.s[index].Jobs[i].end[j] - ga.s[index].Jobs[i].start[j]) / 2 -0.25), y = ga.s[index].Jobs[i].machines[j] + 1 -0.2,
                #          s = i , size=15, fontproperties='Times New Roman')

        for i in range(J_num):
            for j in range(len(self.s.Jobs[i].agv_start)):
                axes.barh(M_num + self.s.Jobs[i].agv[j] + 1,
                         self.s.Jobs[i].agv_end[j] - self.s.Jobs[i].agv_start[j],
                         left=self.s.Jobs[i].agv_start[j], color=colors[i], edgecolor='black')
                # plt.text(x = ga.s[index].Jobs[i].agv_start[j] + ((ga.s[index].Jobs[i].agv_end[j] - ga.s[index].Jobs[i].agv_start[j]) / 2 - 0.25), y = machine_num + ga.s[index].Jobs[i].agv[j] + 1 -0.2,
                #          s =i  , size=15, fontproperties='Times New Roman')

        # 充电过程可视化
        for i in range(A_num):
            for j in range(len(self.s.Agvs[i].start['charge'])):
                axes.barh(M_num + i + 1,
                         self.s.Agvs[i].end['charge'][j] - self.s.Agvs[i].start['charge'][j],
                         left=self.s.Agvs[i].start['charge'][j], color='w', edgecolor='black')

                # plt.text(x = ga.s[index].Agvs[i].start['charge'][j] + ((ga.s[index].Agvs[i].end['charge'][j] - ga.s[index].Agvs[i].start['charge'][j]) / 2 - 1), y = ins.machine_num + i + 1 -0.1,
                #          s ="充电", size=12, fontproperties='SimHei')

                axes.text(x=self.s.Agvs[i].start['charge'][j] + (
                        (self.s.Agvs[i].end['charge'][j] - self.s.Agvs[i].start['charge'][j]) / 2),
                         y=M_num + i + 1, s="充电", size=9, fontproperties='SimHei',
                         verticalalignment="center", horizontalalignment="center")

        # 人工可视化
        for i in range(P_num):
            for j in range(len(self.s.Peos[i].start)):
                axes.barh(M_num + A_num + i + 1,
                         self.s.Peos[i].end[j] - self.s.Peos[i].start[j],
                         left=self.s.Peos[i].start[j], color=colors[self.s.Peos[i].jobs[j]],
                         edgecolor='black')
        #
        patches = [mpatches.Patch(color=colors[i], label=label_job[i]) for i in range(len(label_job))]
        axes.legend(handles=patches, loc=1)
        axes.set_yticks(y, label, color='black')  # 为y轴设置标号
        # # x = range(0,70,10)
        # # plt.xticks(x)
        #
        # # x、y轴标签
        axes.set_xlabel("时间", size=20, fontproperties='SimSun')
        # # plt.ylabel("机器", size=20, fontproperties='SimSun')
        axes.set_ylabel("       机器                  AGV      工人", size=24, fontproperties='SimSun')
        #
        # axes.show()

    def draw_iter(self,axes):
        t = np.arange(0, self.iter)
        s = np.array(self.ob_v)
        axes.plot(t, s)
        # self.canvas.figs.suptitle("sin")  # 设置标题

     # 与算法选择链接的槽函数
    def selectal(self, al_name): #选择算法
        self.algorithm = al_name
        print("算法选择为：",self.algorithm )


    def selectIns(self, Ins_idx): #选择算例
        self.Instance = Ins_idx
        print("选择的算例为",self.Instance)



    def run(self):

        # 算法文件导入
        if self.algorithm == "GA":
            from algorithm import GA

            class_name = f'ins{self.Instance}'  # 通过算例索引导入实际算例类
            ins_class = getattr(instance, class_name)
            ins = ins_class()
            #ins = instance.ins0()
            print(ins.job_state)


            #算法优化 获取最终解
            al = GA.ga(ins.job_state, ins.machine_num, ins.agv_num, ins.agv_time, ins.peo_num, ins.peo_time)
            self.best_individual, self.s, self.ob_v = al.main()  #算法运行
            print(self.s)



        elif self.algorithm == "PSO":
            from algorithm import PSO




# 迭代优化线程
class optimize(QThread):
    optimi_finished = pyqtSignal()
    def __init__(self, f, window):
        super().__init__()
        self.f = f
        self.window = window

    def run(self):

        print(self.f.algorithm)
        # 算法文件导入
        if self.f.algorithm == "GA":
            from algorithm import GA

            class_name = f'ins{self.f.Instance}'  # 通过算例索引导入实际算例类
            ins_class = getattr(instance, class_name)
            ins = ins_class()
            # ins = instance.ins0()
            print(ins.job_state)

            # 算法优化 获取最终解
            al = GA.ga(ins.job_state, ins.machine_num, ins.agv_num, ins.agv_time, ins.peo_num, ins.peo_time)
            print(self.f.s)
            self.f.best_individual, self.f.s, self.f.ob_v = al.main()  # 算法运行


        self.optimi_finished.emit()

