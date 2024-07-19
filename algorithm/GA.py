import random
from Scheduling import Scheduling
import copy
import numpy as np
import math
class ga:
    def __init__(self, job_state, machine_num, agv_num, agv_time, peo_num, peo_time):
        # 定义机器数量 工件数量 AGV小车数量 可加工机器集合等初始条件
        self.job_state = job_state  #是一个三维数组 一维表示工件数  二维表示各个工件的工序数 三维表示可加工机器集合    数组中的数字表示加工时长 不能加工为-1
        self.machine_num = machine_num  #机器数量  
        self.agv_num = agv_num          # agv数量
        self.job_num = len(self.job_state)
        self.agv_time = agv_time        #agv在各机器之间的搬运时间
        self.peo_num = peo_num
        self.peo_time = peo_time
        self.Pm = 0.2               #交叉变异因子
        self.Pc = 0.8
        self.Pop_size = 100        #种群大小
        self.iter = 500         # 迭代次数
        self.break_value = 30



        #创建一个调度类 创建 工件 机器 AGV类
        self.s = []

        #目标函数值
        self.ob_v = []
        #工件各工序可选的加工机器
        self.job_machine = [[[k for k in range(len(self.job_state[i][j]))
                              if self.job_state[i][j][k] != -1] for j in range(len(self.job_state[i]))]
                            for i in range(len(self.job_state))]

        # 将加工时间中的-1变成1000
        self.job_machine_time = [[[self.job_state[i][j][k] if self.job_state[i][j][k] != -1 else 1000
                                   for k in range(len(self.job_state[i][j]))] for j in range(len(self.job_state[i]))]
                                 for i in range(len(self.job_state))]

        self.travel_time = agv_time
        
        
        
       
        
        #生成初始种群
        self.population = [self.RCH() for i in range(self.Pop_size)]
        
        
        
    #生成染色体 机器编码 工序编码 以及 AGV编码
    def RCH(self):
        #生成工序编码
        j_code = []
        m_code = []
        a_code = []
        p_code = []
        record = [0 for i in range(len(self.job_state))] #记录工件在编码中出现的次数
        for i in range(len(self.job_state)):
            for x in range(len(self.job_state[i])):
                j_code.append(i)
        random.shuffle(j_code) # 打乱顺序 随机编码
        
        #机器编码
        for i in range(len(j_code)):  #j_code[i]是工件 record[j_code[i]]是工件的第几道工序
            m_code.append(random.choice(self.job_machine[j_code[i]][record[j_code[i]]]))
            record[j_code[i]] +=1
            
        #AGV编码
        for i in range(len(j_code)):
            a_code.append(random.randint(0, self.agv_num-1))  #随机从小车选取
            


        #人工编码
        for i in range(len(j_code)):
            p_code.append(random.randint(0, self.peo_num - 1))

        #四层编码
        return [j_code, m_code, a_code, p_code]

        # 全局初始化
    def global_initail(self):
        # 生成工序编码
        j_code = []
        m_code = []
        a_code = []
        p_code = []
        record = [0 for i in range(len(self.job_state))]  # 记录工件在编码中出现的次数

        for i in range(len(self.job_state)):
            for x in range(len(self.job_state[i])):
                j_code.append(i)
        random.shuffle(j_code)  # 打乱顺序 随机编码

        # 加工机器全局初始化
        record_g = [0 for i in range(len(self.job_state))]  # 记录工件在编码中出现的次数
        mt_g = [0] * self.machine_num
        for i in j_code:
            mt_l = [mt_g[j] + self.job_machine_time[i][record_g[i]][j] for j in range(self.machine_num)]
            record_g[i] = record_g[i] + 1
            ind = mt_l.index(min(mt_l))
            mt_g[ind] = mt_l[ind]
            m_code.append(ind)

            # AGV编码
        for i in range(len(j_code)):
            a_code.append(random.randint(0, self.agv_num - 1))  # 随机从小车选取

            # 人工编码
        for i in range(len(j_code)):
            p_code.append(random.randint(0, self.peo_num - 1))

            # 四层编码
        return [j_code, m_code, a_code, p_code]

        # 局部初始化

    def local_initial(self):
        # 生成工序编码
        j_code = []
        m_code = []
        a_code = []
        p_code = []
        record = [0 for i in range(len(self.job_state))]  # 记录工件在编码中出现的次数

        for i in range(len(self.job_state)):
            for x in range(len(self.job_state[i])):
                j_code.append(i)
        random.shuffle(j_code)  # 打乱顺序 随机编码

        # 加工机器局部初始化
        for i in j_code:
            m_l = self.job_machine_time[i][record[i]]
            ind = m_l.index(min(m_l))
            record[i] = record[i] + 1
            m_code.append(ind)

        # AGV编码
        for i in range(len(j_code)):
            a_code.append(random.randint(0, self.agv_num - 1))  # 随机从小车选取

        # 人工编码
        record = [0 for i in range(len(self.job_state))]
        for i in j_code:
            m_l = self.peo_time[i][record[i]]
            ind = m_l.index(min(m_l))
            record[i] = record[i] + 1
            m_code.append(ind)

        #工人随机编码
        # for i in range(len(j_code)):
        #     p_code.append(random.randint(0, self.peo_num - 1))

            # 四层编码
        return [j_code, m_code, a_code, p_code]

    def global_initail1(self):

        # 生成工序编码
        j_code = []
        m_code = []
        a_code = []
        p_code = []
        record = [0 for i in range(len(self.job_state))]  # 记录工件在编码中出现的次数

        for i in range(len(self.job_state)):
            for x in range(len(self.job_state[i])):
                j_code.append(i)
        random.shuffle(j_code)  # 打乱顺序 随机编码

        # 加工机器全局初始化
        m_code_tem = [[] for i in range(self.job_num)]
        mt_g = [0] * self.machine_num
        for i in range(len(self.job_machine_time)):
            for j in range(len(self.job_machine_time[i])):
                mt_l = [mt_g[_] + self.job_machine_time[i][j][_] for _ in range(self.machine_num)]
                ind = mt_l.index(min(mt_l))
                mt_g[ind] = mt_l[ind]
                m_code_tem[i].append(ind)


        for i in j_code:
            m_code.append(m_code_tem[i][record[i]])
            record[i]+=1

            # AGV编码
        for i in range(len(j_code)):
            a_code.append(random.randint(0, self.agv_num - 1))  # 随机从小车选取

            # 人工编码
        record = [0 for i in range(len(self.job_state))]
        for i in j_code:
            m_l = self.peo_time[i][record[i]]
            ind = m_l.index(min(m_l))
            record[i] = record[i] + 1
            m_code.append(ind)
        # for i in range(len(j_code)):
        #
        #     p_code.append(random.randint(0, self.peo_num - 1))

            # 四层编码
        return [j_code, m_code, a_code, p_code]


    def local_initial1(self):
        # 生成工序编码
        j_code = []
        m_code = []
        a_code = []
        p_code = []
        record = [0 for i in range(len(self.job_state))]  # 记录工件在编码中出现的次数

        for i in range(len(self.job_state)):
            for x in range(len(self.job_state[i])):
                j_code.append(i)
        random.shuffle(j_code)  # 打乱顺序 随机编码

        # 加工机器全局初始化
        m_code_tem = [[] for i in range(self.job_num)]
        for i in range(len(self.job_machine_time)):
            mt_g = [0] * self.machine_num
            for j in range(len(self.job_machine_time[i])):
                mt_l = [mt_g[_] + self.job_machine_time[i][j][_] for _ in range(self.machine_num)]
                ind = mt_l.index(min(mt_l))
                mt_g[ind] = mt_l[ind]
                m_code_tem[i].append(ind)

        for i in j_code:
            m_code.append(m_code_tem[i][record[i]])
            record[i] += 1

            # AGV编码
        for i in range(len(j_code)):
            a_code.append(random.randint(0, self.agv_num - 1))  # 随机从小车选取

            # 人工编码
        record = [0 for i in range(len(self.job_state))]
        for i in j_code:
            m_l = self.peo_time[i][record[i]]
            ind = m_l.index(min(m_l))
            record[i] = record[i] + 1
            m_code.append(ind)
        # for i in range(len(j_code)):
        #     p_code.append(random.randint(0, self.peo_num - 1))

            # 四层编码
        return [j_code, m_code, a_code, p_code]
    
    #选择
    # def Select(self,fitness):
    #     #轮盘赌选择
    #     new_pop = []  #被选中个体索引值
    #     pro = []   # 选中概率
    #     max_f = max(fitness)
    #     min_f = min(fitness)
    #     for i in range(len(fitness)):
    #         pro.append( (fitness[i] - min_f) / (max_f - min_f + 0.0001) ) #选中几率  增加一个极小值防止分母为0
    #
    #     temp = 0
    #     while len(new_pop) != len(fitness):
    #         if pro[temp] > random.random():
    #             new_pop.append(temp)  #记录种群索引值
    #         temp +=1
    #         if temp == len(fitness): # 当一轮循环后 种群没满 重新进行一轮选择
    #             temp = 0
    #
    #     return new_pop #返回索引值


    def Select(self, Fit_value):
        Fit = []
        for i in range(len(Fit_value)):
            fit = Fit_value[i]
            Fit.append(fit)
        Fit = np.array(Fit)
        idx = np.random.choice(np.arange(len(Fit_value)), size=len(Fit_value), replace=True,
                               p=(Fit) / (Fit.sum()))
        return idx
            

    def pox_crossover(self, code1, code2):
        # pox交叉。 要点：当交叉之后会改变工件的工序情况，保持工件各工序加工的机器和小车不变
        # 实现；利用栈， 主要思路是相同时出栈。 将一个染色体分为 记录选中基因位置的染色体 和 剔除选中基因的染色体
        # 这里可以优化的空间很大， 本来写的是列表在一起，为了不出现复制代码的提示打乱了。
        a = random.choice(list(set(code1[0])))  #random.choice() 随机返回列表中一个元素
        code1m = copy.deepcopy(code1[1])      #拷贝 1的机器编码 依次类推
        code2m = copy.deepcopy(code2[1])
        coda1 = copy.deepcopy(code1[2])
        coda2 = copy.deepcopy(code2[2])
        code1p = copy.deepcopy(code1[3])
        code2p = copy.deepcopy(code2[3])
        code11 = [i if i != a else -1 for i in code1[0]]
        code21 = [i if i != a else -1 for i in code2[0]]
        code11.reverse()
        code21.reverse()
        code22 = [a if i == a else -1 for i in code2[0]]
        code12 = [a if i == a else -1 for i in code1[0]]
        code1m.reverse()
        code2m.reverse()
        coda1.reverse()
        coda2.reverse()
        code1p.reverse()
        code2p.reverse()
        codem1 = []
        codem2 = []
        coa1 = []
        coa2 = []
        cop1 = []
        cop2 = []
        for i in range(len(code1[0])):
            codem1.append(code1[1][i])
            codem2.append(code2[1][i])
            coa1.append(code1[2][i])
            coa2.append(code2[2][i])
            cop1.append(code1[3][i])
            cop2.append(code2[3][i])
            while code12[i] == -1:
                code12[i] = code21.pop()
                codem1[i] = code2m.pop()
                coa1[i] = coda2.pop()
                cop1[i] = code1p.pop()
            while code22[i] == -1:
                code22[i] = code11.pop()
                codem2[i] = code1m.pop()
                coa2[i] = coda1.pop()
                cop2[i] = code2p.pop()
        return [code12, codem1, coa1, cop1], [code22, codem2, coa2, cop2]

        # 机器交叉

    def machine_crossover(self, code1, code2, n):
        le = list(range(0, len(code1[0])))
        r = random.sample(le, n)  # 随机选取两个点 进行机器交叉
        r.sort()  # 升序

        d1 = {'job': [], 'ope': []}  # 记录所取点相应的工件 工序
        d2 = {'job': [], 'ope': []}

        record_1 = [0] * self.job_num  # 记录工件工序
        record_2 = [0] * self.job_num
        for i in range(r[-1] + 1):

            record_1[code1[0][i]] += 1
            record_2[code2[0][i]] += 1

            if i in r:
                d1['job'].append(code1[0][i])
                d1['ope'].append(record_1[code1[0][i]] - 1)
                d2['job'].append(code2[0][i])
                d2['ope'].append(record_2[code2[0][i]] - 1)

        ind_c1 = [[]] * self.job_num  # 获取工序在编码中的索引
        ind_c2 = [[]] * self.job_num
        for i in range(len(code1[0])):
            ind_c1[code1[0][i]].append(i)
            ind_c2[code2[0][i]].append(i)

        for i in range(n):
            code2[1][ind_c2[d1['job'][i]][d1['ope'][i]]] = code1[1][ind_c1[d1['job'][i]][d1['ope'][i]]]
            code1[1][ind_c1[d2['job'][i]][d2['ope'][i]]] = code2[1][ind_c2[d2['job'][i]][d2['ope'][i]]]

        return code1, code2

    #变异
    def machine_Mutation(self, codes):
        # 机器随机变异
        rand = random.randint(0, len(codes[0])-1)
        pro = 0
        for i in range(rand):
            if codes[0][i] == codes[0][rand]:
                pro +=1
        codes[1][rand] = random.choice(self.job_machine[codes[0][rand]][pro])
        return codes


    def agv_mutation(self, codes):
        #小车随机变异
        rand = random.randint(0, len(codes[0])-1)  #随机选择一个基因改变搬运车
        codes[2][rand] = random.randint(0, self.agv_num-1)    #随机选择AGV
        return codes

    def peo_mutation(self, codes):
        #人工随机变异
        rand = random.randint(0, len(codes[0])-1)
        codes[3][rand] = random.randint(0, self.peo_num-1)
        return codes


    def job_mutation(self,codes):
        # 工件变异 随机交换两个基因位置
        # 要点：改变基因后其工序改变导致其加工机器可选集改变。  保存选中的两个工件的工件所选机器的顺序，交换位置之后仍用该顺序
        # 实现：用列表记录下顺序，当基因和选中的基因的一样时出栈列表中的机器和小车
        a = random.randint(0, len(codes[0]) - 1)
        b = random.randint(0, len(codes[0]) - 1)    # 随机选中两基因
        record1 = []
        recode2 = []    # 记录选中的工件的加工机器
        recorda1 = []
        recorda2 = []   # 记录选中工件的搬运小车
        recordp1 = []
        recordp2 = [] #记录选中的人工
        for i in range(len(codes[0])):
            if codes[0][i] == codes[0][a]:
                record1.append(codes[1][i])
                recorda1.append(codes[2][i])
                recordp1.append(codes[3][i])
            if codes[0][i] == codes[0][b]:
                recode2.append(codes[1][i])
                recorda2.append(codes[2][i])
                recordp2.append(codes[3][i])
        record1.reverse()
        recode2.reverse()
        recorda1.reverse()
        recorda2.reverse()
        recordp1.reverse()
        recordp2.reverse()
        tem = codes[0][a]
        codes[0][a] = codes[0][b]
        codes[0][b] = tem
        for i in range(len(codes[0])):
            if codes[0][i] == codes[0][a]:
                codes[1][i] = recode2.pop()
                codes[2][i] = recorda2.pop()
                codes[3][i] = recordp2.pop()
            if codes[0][i] == codes[0][b]:
                codes[1][i] = record1.pop()
                codes[2][i] = recorda1.pop()
                codes[3][i] = recordp1.pop()
        return codes


    def peo_fatigue(self, x_t , f_t, y, u, dt):

        #y u 分别为工人疲劳衰减率和疲劳恢复率 f_t为工人的疲劳值 范围在【0，1】
        #x_t表示工人在t时刻的工作状态 工作时为1 休息时为0   dt为工作或休息时间间隔

        f_t_1 = x_t * (1 - (1 - f_t) * (math.exp(-y * dt))) + (1 - x_t) * f_t * (math.exp(-u * dt))


        return f_t_1


    #求取工人剩余可工作时长
    def time_fatigue(self, y, f_0, f_e):

        t = (1 / y) * math.log((1 - f_0) / (1 - f_e), math.e)

        return t

    def rest_time(self, f_0, y, u, f_e, t_w):

        t = (math.log(f_0) - math.log(1 - ((1 - f_e) * math.exp(y * t_w)))) / u

        return t

    #解码
    def decode(self, codes):
        #记录工件的加工工序
        job_process_record = [0 for i in range(self.job_num)]
        
        s = Scheduling(self.job_num, self.machine_num, self.agv_num, self.peo_num)
        for i in range(len(codes[0])):

            job = codes[0][i]
            machine = codes[1][i]
            agv = codes[2][i]
            peo = codes[3][i]

            process = job_process_record[codes[0][i]]  # 当前工件的工序

            next_machine = machine + 1  # 将机器数加一 工件起始位置为0

            now_machine = s.Jobs[job].last_loc  # next_machine 和 now_machine 用来表示位置 初始位置为0 机器位置是machine+1

            if now_machine != next_machine:
                if s.Agvs[agv].battery > self.travel_time[s.Agvs[agv].last_loc][s.Jobs[job].last_loc] + \
                        self.travel_time[now_machine][next_machine] + self.travel_time[next_machine][-1]:
                    # 小车开始搬运时间 = max(小车结束时间 + 小车从结束位置前往工件处的时间， 工件上一工序结束加工时间)
                    agv_start_time = max(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][s.Jobs[job].last_loc],
                        s.Jobs[job].last_ot)

                    agv_end_time = agv_start_time + self.travel_time[now_machine][next_machine]
                    # 小车位置更新
                    agv_loc = next_machine
                    agv_battery = s.Agvs[agv].battery - (self.travel_time[s.Agvs[agv].last_loc][s.Jobs[job].last_loc] +
                                                         self.travel_time[now_machine][next_machine])

                    s.Agvs[agv].start['unload'].append(s.Agvs[agv].last_ot)
                    s.Agvs[agv].end['unload'].append(agv_start_time)
                    s.Agvs[agv].start['load'].append(agv_start_time)
                    s.Agvs[agv].end['load'].append(agv_end_time)

                    s.Agvs[agv].update(agv_start_time, agv_end_time, agv_loc, agv_battery)

                    # 工件到达机器时间
                    job_start_time = agv_end_time

                else:
                    agv_start_time = max(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1] + self.travel_time[-1][
                            s.Jobs[job].last_loc] + 5,
                        s.Jobs[job].last_ot)

                    agv_end_time = agv_start_time + self.travel_time[now_machine][next_machine]
                    # 小车位置更新
                    agv_loc = next_machine
                    agv_battery = 20 - self.travel_time[-1][s.Jobs[job].last_loc] - self.travel_time[now_machine][
                        next_machine]  # 20代表满电

                    # 使用字典记录agv装载、空载、充电 三种状态的时间
                    s.Agvs[agv].start['unload'].append(s.Agvs[agv].last_ot)
                    s.Agvs[agv].end['unload'].append(s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1])
                    s.Agvs[agv].start['charge'].append(s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1])
                    s.Agvs[agv].end['charge'].append(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1] + 5)
                    s.Agvs[agv].start['unload'].append(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1] + 5)
                    s.Agvs[agv].end['unload'].append(agv_start_time)
                    s.Agvs[agv].start['load'].append(agv_start_time)
                    s.Agvs[agv].end['load'].append(agv_end_time)

                    s.Agvs[agv].update(agv_start_time, agv_end_time, agv_loc, agv_battery)

                    job_start_time = agv_end_time  # 工件到达机器的时间


            else:
                job_start_time = s.Jobs[job].last_ot  # 工件上道工序的完工时间
                # 更新编码中运输小车的信息 时间和位置都不变
                s.Agvs[agv].update(s.Agvs[agv].last_ot, s.Agvs[agv].last_ot, s.Agvs[agv].last_loc, s.Agvs[agv].battery)
                agv_start_time = s.Agvs[agv].last_ot
                agv_end_time = s.Agvs[agv].last_ot

            if job_start_time >= s.Peos[peo].last_ot:

                dt = job_start_time - s.Peos[peo].last_ot
                ft = self.peo_fatigue(0, s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u, dt)
                s.Peos[peo].update_fatigue(ft)
                start_time = job_start_time

                # 疲劳值记录
                s.Peos[peo].ft_arr.append([start_time, ft])

                s.Peos[peo].ft_start['rest'].append(s.Peos[peo].last_ot)
                s.Peos[peo].ft_end['rest'].append(start_time)
            else:
                start_time = s.Peos[peo].last_ot

            # 获取工人的疲劳值 判断是否需要休息
            rem_t = self.time_fatigue(s.Peos[peo].y, s.Peos[peo].fatigue, s.Peos[peo].ind)
            if rem_t >= self.peo_time[job][process][peo]:
                peo_start_time = start_time
                s.Peos[peo].ft_start['work'].append(peo_start_time)
                s.Peos[peo].ft_end['work'].append(peo_start_time + self.peo_time[job][process][peo])




            else:
                rest_time = self.rest_time(s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u, s.Peos[peo].ind,
                                           self.peo_time[job][process][peo])
                peo_start_time = start_time + rest_time

                s.Peos[peo].ft_start['rest'].append(start_time)
                s.Peos[peo].ft_end['rest'].append(peo_start_time)

                f_t = self.peo_fatigue(0, s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u, rest_time)
                s.Peos[peo].ft_arr.append([peo_start_time, f_t])
                s.Peos[peo].update_fatigue(f_t)

                s.Peos[peo].ft_start['work'].append(peo_start_time)
                s.Peos[peo].ft_end['work'].append(peo_start_time + self.peo_time[job][process][peo])

            # peo_start_time = max(job_start_time, s.Peos[peo].last_ot)
            peo_end_time = peo_start_time + self.peo_time[job][process][peo]

            machine_start_time = max(peo_end_time, s.Machines[machine].last_ot)
            machine_end_time = machine_start_time + self.job_state[job][process][machine]
            job_process_record[job] += 1  # 记录工件已知加工工序数 工序+1

            # 更新 加工机器 工件 信息
            s.Jobs[job].update(machine_start_time, machine_end_time, machine, next_machine, agv, agv_start_time,
                               agv_end_time)

            s.Machines[machine].update(machine_start_time, machine_end_time)

            # 更新人工信息
            s.Peos[peo].update(peo_start_time, peo_end_time, job)

            fatigue = self.peo_fatigue(1, s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u,
                                       self.peo_time[job][process][peo])
            s.Peos[peo].update_fatigue(fatigue)

            # 记录疲劳值
            s.Peos[peo].ft_arr.append([peo_end_time, fatigue])


        #最后运输到卸载点的过程
        # job_time_list_id = sorted(range(self.job_num), key=lambda k: s.Jobs[k].last_ot)  #按最后时间排序 从小到大
        #
        # for i in job_time_list_id:
        #     agv_time_list_id = sorted(range(self.agv_num), key=lambda k: s.Agvs[k].last_ot)
        #     agv_chosen = agv_time_list_id[0]
        #
        #     if s.Agvs[agv_chosen].battery > self.travel_time[s.Agvs[agv_chosen].last_loc][s.Jobs[i].last_loc] + \
        #             self.travel_time[s.Jobs[i].last_loc][-2] + self.travel_time[-2][-1]:
        #         # 小车开始搬运时间 = max(小车结束时间 + 小车从结束位置前往工件处的时间， 工件上一工序结束加工时间)
        #         agv_start_time = max(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][s.Jobs[i].last_loc],
        #             s.Jobs[i].last_ot)
        #
        #         agv_end_time = agv_start_time + self.travel_time[s.Jobs[i].last_loc][-2]
        #         # 小车位置更新
        #         agv_loc = self.machine_num + 1
        #         agv_battery = s.Agvs[agv_chosen].battery - (self.travel_time[s.Agvs[agv_chosen].last_loc][s.Jobs[i].last_loc] +
        #                                              self.travel_time[s.Jobs[i].last_loc][-2])
        #
        #
        #
        #         s.Agvs[agv_chosen].start['unload'].append(s.Agvs[agv_chosen].last_ot)
        #         s.Agvs[agv_chosen].end['unload'].append(agv_start_time)
        #         s.Agvs[agv_chosen].start['load'].append(agv_start_time)
        #         s.Agvs[agv_chosen].end['load'].append(agv_end_time)
        #
        #         s.Agvs[agv_chosen].update(agv_start_time, agv_end_time, agv_loc, agv_battery)
        #
        #     else:
        #         agv_start_time = max(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1] + self.travel_time[-1][
        #                 s.Jobs[i].last_loc] + 5, #5为充电时长
        #             s.Jobs[i].last_ot)
        #
        #         agv_end_time = agv_start_time + self.travel_time[s.Jobs[i].last_loc][-2]
        #         # 小车位置更新
        #         agv_loc = self.machine_num + 1
        #         agv_battery = 20 - self.travel_time[-1][s.Jobs[i].last_loc] - self.travel_time[s.Jobs[i].last_loc][
        #             -2]
        #
        #         # 使用字典记录agv装载、空载、充电 三种状态的时间
        #         s.Agvs[agv_chosen].start['unload'].append(s.Agvs[agv_chosen].last_ot)
        #         s.Agvs[agv_chosen].end['unload'].append(s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1])
        #         s.Agvs[agv_chosen].start['charge'].append(s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1])
        #         s.Agvs[agv_chosen].end['charge'].append(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1] + 5)
        #         s.Agvs[agv_chosen].start['unload'].append(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1] + 5)
        #         s.Agvs[agv_chosen].end['unload'].append(agv_start_time)
        #         s.Agvs[agv_chosen].start['load'].append(agv_start_time)
        #         s.Agvs[agv_chosen].end['load'].append(agv_end_time)
        #
        #         s.Agvs[agv_chosen].update(agv_start_time, agv_end_time, agv_loc, agv_battery)
        #
        #     s.Jobs[i].agv.append(agv_chosen)
        #     s.Jobs[i].agv_start.append(agv_start_time)
        #     s.Jobs[i].agv_end.append(agv_end_time)
        #     s.Jobs[i].last_ot = agv_end_time

        #获取最大完工时间
        last_time=[]                      #获取每个工件的最大完工时间
        for i in range(len(s.Jobs)):
            last_time.append(s.Jobs[i].last_ot)
            
        fitness = 1 / max(last_time)
        
        return fitness


#解码
    def last_decode(self, codes):
        #记录工件的加工工序
        job_process_record = [0 for i in range(self.job_num)]
        
        s = Scheduling(self.job_num, self.machine_num, self.agv_num, self.peo_num)
        self.s.append(s) #如果要获得种群中适应度最好的解 需要获取索引值
        for i in range(len(codes[0])):

            job = codes[0][i]
            machine = codes[1][i]
            agv = codes[2][i]
            peo = codes[3][i]

            process = job_process_record[codes[0][i]]  # 当前工件的工序

            next_machine = machine + 1  # 将机器数加一 工件起始位置为0

            now_machine = s.Jobs[job].last_loc  # next_machine 和 now_machine 用来表示位置 初始位置为0 机器位置是machine+1

            if now_machine != next_machine:
                if s.Agvs[agv].battery > self.travel_time[s.Agvs[agv].last_loc][s.Jobs[job].last_loc] + \
                        self.travel_time[now_machine][next_machine] + self.travel_time[next_machine][-1]:
                    # 小车开始搬运时间 = max(小车结束时间 + 小车从结束位置前往工件处的时间， 工件上一工序结束加工时间)
                    agv_start_time = max(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][s.Jobs[job].last_loc],
                        s.Jobs[job].last_ot)

                    agv_end_time = agv_start_time + self.travel_time[now_machine][next_machine]
                    # 小车位置更新
                    agv_loc = next_machine
                    agv_battery = s.Agvs[agv].battery - (self.travel_time[s.Agvs[agv].last_loc][s.Jobs[job].last_loc] +
                                                         self.travel_time[now_machine][next_machine])

                    s.Agvs[agv].start['unload'].append(s.Agvs[agv].last_ot)
                    s.Agvs[agv].end['unload'].append(agv_start_time)
                    s.Agvs[agv].start['load'].append(agv_start_time)
                    s.Agvs[agv].end['load'].append(agv_end_time)

                    s.Agvs[agv].update(agv_start_time, agv_end_time, agv_loc, agv_battery)

                    # 工件到达机器时间
                    job_start_time = agv_end_time

                else:
                    agv_start_time = max(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1] + self.travel_time[-1][
                            s.Jobs[job].last_loc] + 5,
                        s.Jobs[job].last_ot)

                    agv_end_time = agv_start_time + self.travel_time[now_machine][next_machine]
                    # 小车位置更新
                    agv_loc = next_machine
                    agv_battery = 20 - self.travel_time[-1][s.Jobs[job].last_loc] - self.travel_time[now_machine][
                        next_machine]  # 20代表满电

                    # 使用字典记录agv装载、空载、充电 三种状态的时间
                    s.Agvs[agv].start['unload'].append(s.Agvs[agv].last_ot)
                    s.Agvs[agv].end['unload'].append(s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1])
                    s.Agvs[agv].start['charge'].append(s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1])
                    s.Agvs[agv].end['charge'].append(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1] + 5)
                    s.Agvs[agv].start['unload'].append(
                        s.Agvs[agv].last_ot + self.travel_time[s.Agvs[agv].last_loc][-1] + 5)
                    s.Agvs[agv].end['unload'].append(agv_start_time)
                    s.Agvs[agv].start['load'].append(agv_start_time)
                    s.Agvs[agv].end['load'].append(agv_end_time)

                    s.Agvs[agv].update(agv_start_time, agv_end_time, agv_loc, agv_battery)

                    job_start_time = agv_end_time  # 工件到达机器的时间


            else:
                job_start_time = s.Jobs[job].last_ot  # 工件上道工序的完工时间
                # 更新编码中运输小车的信息 时间和位置都不变
                s.Agvs[agv].update(s.Agvs[agv].last_ot, s.Agvs[agv].last_ot, s.Agvs[agv].last_loc, s.Agvs[agv].battery)
                agv_start_time = s.Agvs[agv].last_ot
                agv_end_time = s.Agvs[agv].last_ot

            if job_start_time >= s.Peos[peo].last_ot:

                dt = job_start_time - s.Peos[peo].last_ot
                ft = self.peo_fatigue(0, s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u, dt)
                s.Peos[peo].update_fatigue(ft)
                start_time = job_start_time

                # 疲劳值记录
                s.Peos[peo].ft_arr.append([start_time, ft])

                s.Peos[peo].ft_start['rest'].append(s.Peos[peo].last_ot)
                s.Peos[peo].ft_end['rest'].append(start_time)
            else:
                start_time = s.Peos[peo].last_ot

            # 获取工人的疲劳值 判断是否需要休息
            rem_t = self.time_fatigue(s.Peos[peo].y, s.Peos[peo].fatigue, s.Peos[peo].ind)
            if rem_t >= self.peo_time[job][process][peo]:
                peo_start_time = start_time
                s.Peos[peo].ft_start['work'].append(peo_start_time)
                s.Peos[peo].ft_end['work'].append(peo_start_time + self.peo_time[job][process][peo])




            else:
                rest_time = self.rest_time(s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u, s.Peos[peo].ind,
                                           self.peo_time[job][process][peo])
                peo_start_time = start_time + rest_time

                s.Peos[peo].ft_start['rest'].append(start_time)
                s.Peos[peo].ft_end['rest'].append(peo_start_time)

                f_t = self.peo_fatigue(0, s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u, rest_time)
                s.Peos[peo].ft_arr.append([peo_start_time, f_t])
                s.Peos[peo].update_fatigue(f_t)

                s.Peos[peo].ft_start['work'].append(peo_start_time)
                s.Peos[peo].ft_end['work'].append(peo_start_time + self.peo_time[job][process][peo])

            # peo_start_time = max(job_start_time, s.Peos[peo].last_ot)
            peo_end_time = peo_start_time + self.peo_time[job][process][peo]

            machine_start_time = max(peo_end_time, s.Machines[machine].last_ot)
            machine_end_time = machine_start_time + self.job_state[job][process][machine]
            job_process_record[job] += 1  # 记录工件已知加工工序数 工序+1

            # 更新 加工机器 工件 信息
            s.Jobs[job].update(machine_start_time, machine_end_time, machine, next_machine, agv, agv_start_time,
                               agv_end_time)

            s.Machines[machine].update(machine_start_time, machine_end_time)

            # 更新人工信息
            s.Peos[peo].update(peo_start_time, peo_end_time, job)

            fatigue = self.peo_fatigue(1, s.Peos[peo].fatigue, s.Peos[peo].y, s.Peos[peo].u,
                                       self.peo_time[job][process][peo])
            s.Peos[peo].update_fatigue(fatigue)

            # 记录疲劳值
            s.Peos[peo].ft_arr.append([peo_end_time, fatigue])

        # 最后运输到卸载点的过程
        # job_time_list_id = sorted(range(self.job_num), key=lambda k: s.Jobs[k].last_ot)  # 按最后时间排序 从小到大
        #
        # for i in job_time_list_id:
        #     agv_time_list_id = sorted(range(self.agv_num), key=lambda k: s.Agvs[k].last_ot)
        #     agv_chosen = agv_time_list_id[0]
        #
        #     if s.Agvs[agv_chosen].battery > self.travel_time[s.Agvs[agv_chosen].last_loc][s.Jobs[i].last_loc] + \
        #             self.travel_time[s.Jobs[i].last_loc][-2] + self.travel_time[-2][-1]:
        #         # 小车开始搬运时间 = max(小车结束时间 + 小车从结束位置前往工件处的时间， 工件上一工序结束加工时间)
        #         agv_start_time = max(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][s.Jobs[i].last_loc],
        #             s.Jobs[i].last_ot)
        #
        #         agv_end_time = agv_start_time + self.travel_time[s.Jobs[i].last_loc][-2]
        #         # 小车位置更新
        #         agv_loc = self.machine_num + 1
        #         agv_battery = s.Agvs[agv_chosen].battery - (
        #                     self.travel_time[s.Agvs[agv_chosen].last_loc][s.Jobs[i].last_loc] +
        #                     self.travel_time[s.Jobs[i].last_loc][-2])
        #
        #         s.Agvs[agv_chosen].start['unload'].append(s.Agvs[agv_chosen].last_ot)
        #         s.Agvs[agv_chosen].end['unload'].append(agv_start_time)
        #         s.Agvs[agv_chosen].start['load'].append(agv_start_time)
        #         s.Agvs[agv_chosen].end['load'].append(agv_end_time)
        #
        #         s.Agvs[agv_chosen].update(agv_start_time, agv_end_time, agv_loc, agv_battery)
        #
        #     else:
        #         agv_start_time = max(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1] +
        #             self.travel_time[-1][
        #                 s.Jobs[i].last_loc] + 5,  # 5为充电时长
        #             s.Jobs[i].last_ot)
        #
        #         agv_end_time = agv_start_time + self.travel_time[s.Jobs[i].last_loc][-2]
        #         # 小车位置更新
        #         agv_loc = self.machine_num + 1
        #         agv_battery = 20 - self.travel_time[-1][s.Jobs[i].last_loc] - self.travel_time[s.Jobs[i].last_loc][
        #             -2]
        #
        #         # 使用字典记录agv装载、空载、充电 三种状态的时间
        #         s.Agvs[agv_chosen].start['unload'].append(s.Agvs[agv_chosen].last_ot)
        #         s.Agvs[agv_chosen].end['unload'].append(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1])
        #         s.Agvs[agv_chosen].start['charge'].append(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1])
        #         s.Agvs[agv_chosen].end['charge'].append(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1] + 5)
        #         s.Agvs[agv_chosen].start['unload'].append(
        #             s.Agvs[agv_chosen].last_ot + self.travel_time[s.Agvs[agv_chosen].last_loc][-1] + 5)
        #         s.Agvs[agv_chosen].end['unload'].append(agv_start_time)
        #         s.Agvs[agv_chosen].start['load'].append(agv_start_time)
        #         s.Agvs[agv_chosen].end['load'].append(agv_end_time)
        #
        #         s.Agvs[agv_chosen].update(agv_start_time, agv_end_time, agv_loc, agv_battery)
        #
        #     s.Jobs[i].agv.append(agv_chosen)
        #     s.Jobs[i].agv_start.append(agv_start_time)
        #     s.Jobs[i].agv_end.append(agv_end_time)
        #     s.Jobs[i].last_ot = agv_end_time

        #获取最大完工时间
        last_time=[]                      #获取每个工件的最大完工时间
        for i in range(len(s.Jobs)):
            last_time.append(s.Jobs[i].last_ot)
            
        fitness = 1 / max(last_time)
        
        return fitness

    def oneloop(self, population):
        fitness = []
        for i in population:
            fit = self.decode(i)
            fitness.append(fit)

            
        list_id = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)  #获取排序的索引值
        new_popu = [self.RCH() for i in range(self.Pop_size)]
        j = 0
        for i in list_id[0:int(self.Pop_size*0.1)]:
            new_popu[j] =copy.deepcopy(population[i])
            j+=1
        #new_popu = [self.population[i] for i in list_id[0:10]]  #选取上一代适应度最高的10个 个体不进行交叉变异




        index = self.Select(fitness)

        for i in range(0, self.Pop_size-int(self.Pop_size*0.1)):
            new_popu[j] = copy.deepcopy(population[index[i]])
            j+=1

        for i in range(int(self.Pop_size*0.1), self.Pop_size):
            new = copy.deepcopy(new_popu[i])
            a = random.randint(int(self.Pop_size*0.1), self.Pop_size - 1)  # 随机返回 start-end中一个整数 包括start和end
            if self.Pc >= random.random():
                new, _ = self.pox_crossover(new, new_popu[a])
            # if self.Pc >= random.random():
            #     new,_ = self.machine_crossover(new, new_popu[a],2)
            if self.Pm >= random.random():
                new = self.job_mutation(new)
            if self.Pm >= random.random():
                new = self.machine_Mutation(new)
            if self.Pm >= random.random():
                new = self.peo_mutation(new)
            if self.Pm >= random.random():
                new = self.agv_mutation(new)

            new_popu[i] = new


        return new_popu

        
    
    # def main(self):
    #     for i in range(self.iter):
    #         x = self.population
    #         self.population = self.oneloop(x)
    #         fitness_set = []
    #         for j in self.population:
    #             fitness = self.decode(j)
    #             fitness_set.append(fitness)
    #         self.ob_v.append(1 / max(fitness_set))
    #         print(fitness_set.index(max(fitness_set)))
    #         print(1 / max(fitness_set))
    #
    #
    #
    #
    #     final_fitness = [] #对最后一代种群解码
    #     for i in self.population:
    #         fit_now = self.last_decode(i)
    #         final_fitness.append(fit_now)
    #
    #     return 1 / max(final_fitness), self.population[final_fitness.index(max(final_fitness))], final_fitness.index(max(final_fitness))

    def main(self):

        #全部随机初始化
        population = [self.RCH() for i in range(self.Pop_size)]

        # r g l  5 :3 :2   全局 随机 局部初始化相结合
        # population = []
        # for i in range(int(self.Pop_size * (5 / 10))):
        #     population.append(self.RCH())
        #
        # for i in range(int(self.Pop_size * (3 / 10))):
        #     population.append(self.global_initail1())
        #
        # for i in range(int(self.Pop_size * (2 / 10))):
        #     population.append((self.local_initial1()))


        for i in range(self.iter):

            fitness_set = []
            for j in population:
                fitness = self.decode(copy.deepcopy(j))
                fitness_set.append(fitness)
            self.ob_v.append(1 / max(fitness_set))
            print(1 / max(fitness_set))

            population = self.oneloop(population)



        final_fitness = []  # 对最后一代种群解码
        for i in population:
            fit_now = self.last_decode(i)
            final_fitness.append(fit_now)
        print(final_fitness)
        print('第二次' ,  final_fitness.index(max(final_fitness)))
        print('第二次' ,1 / max(final_fitness))

        return  population[final_fitness.index(max(final_fitness))], self.s[final_fitness.index(max(final_fitness))],self.ob_v