
"""
system model with multiple mec servers, which is consist of job and user.

Simulation based on SimPy:  https://simpy.readthedocs.io/en/latest/.

"""
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import simpy
import math
import csv
from math import *
from Offload_Strategy import OFFLOAD
from numpy import random as ra


__all__ = ["Job", "User", "UL", "MEC_L", "MEC"]

random.seed(1)

# As the number of users increases,the frequency of unloading must be reduced,
# otherwise it will be blocked

UN = 10 # the number of user
MECN = 4  # the number of user
CB = 3 # Channel bandwidth allocation rate
PEN = 1
PLIST = 10
PT_MEAN = 100.0 # the parameters for working time of mec
PT_SIGMA = 30.0
buffer = 500  # env中container
REPAIR_TIME = 30

class Job(object):
    def __init__(self, userID, jobID):
        self.userID = userID
        self.jobID = jobID
        self.jobTran = 0.0  # upload time
        self.jobDTran = 0.0  # download time
        self.jobRun = 0.0  # local processing time
        self.jobCPU = 0.0  # CPU and resource occurpy
        self.jobCEnergy = 0.0  # Transmission energy consumption
        self.jobLEnergy = 0.0  # Local execution energy consumption
        self.jobType = 'normal'  # Task type
        self.jobState = 'LW'  # Activation=act, printing=inh, local waiting=lw, local execution=lr, transmission=ts, remote waiting=rw, remote execution=rr, completion=cp, failure=fl
        self.jobValue = 1.0  # Task value

        # ---------- Dynamic changes during execution ----------
        self.jobRunLeft = 0.0  # Task remote execution time remaining
        self.jobTransLeft = 0.0  # Task transfer time remaining
        self.jobChannel = 0.0  # The obtained channel bandwidth uses Mbps

        # ---------- Execution completed record ----------
        self.jobBegin = 0.0  # The time when the task started
        self.jobFinish = 0.0  # The time when the task execution ends
        self.jobOffload = 0.0  # The time when the task started to unload
        self.jobRT =0.0  # Execution time
        self.jobTT = 0.0  # Transmission time
        self.jobAge = 0.0 # the time from the start of the offloading to the end of the execution


class User(object):
    def __init__(self, userID):
        self.userID = userID
        self.jobList = []  # User task list
        self.jobData = 0.0  # Task transfer data volume
        self.jobTrans = [20]  # Task transmission time-distribution-
        self.jobRuns = 20  # Task local execution time-distribution-
        self.jobCPU = 0.1  # Task CPU utilization
        self.jobNums = 50  # Initial number of tasks
        self.jobCEnergy = [20]  # Transmission energy consumption
        self.jobLEnergy = [20]  # local energy consumption
        self.jobDDL = 10  # Task deadline

        # ---------- location information -----------
        self.X = 0.0  # 纬度
        self.Y = 0.0  # 经度
        self.speed = 0.0
        self.trace = []  # User movement trajectory
        self.userPriority = [0, 0, 0, 0]  # User priority
        self.userMEC = [0, 0, 0, 0]  # User and mec connectivity 用户与mec的连接

        # ---------- Log -----------
        self.Throughout = 0.0  # Throughput
        self.CEnergy = 0.0  # Remote energy consumption
        self.LEnergy = 0.0  # Local energy consumption
        self.commTotal = 0.0  # When the transfer occurredeA
        self.userAge = 0.0  # user tasks are offloaded to the completion time

    def usersetting(self, uLng, uLat):
        self.jobNums = 10
        self.jobData = (UN - self.userID) * 8  # Decreasing data volume, kb
        self.jobRuns = (self.userID + 1) * 10  # job running time
        self.jobDDL = (self.userID + 1) * 12  # job deadline

        self.jobCPU = random.randint(1, UN) / (UN * 2)  # Increasing resource usage
        self.jobLEnergy = [(self.userID + 1) * 1.25 * i for i in range(7, 25)]
        self.X = uLng  # longitude
        self.Y = uLat  # latitude
        self.trace = [[self.X], [self.Y]]
        self.speed = random.choice([0, 0.2, 0.4, 0.6, 0.8, 1])  # speed，m/s 用户移动速度


    def setjobenergy(self, jid, jobtran):
        BDu = self.jobList[jid].jobChannel
        BDd = BDu / 2
        self.jobList[jid].jobTran = self.jobData / BDu  # Sampling from the transmission time distribution
        self.jobList[jid].jobDTran = self.jobData / BDd
        LET = BDu * 0.438 + 0.051 * BDd + 1.288  # 4G
        # WIFI = BDu*0.283 + 0.137*BDd + 0.132
        # self.JOB_LIST[jid].jobCEnergy = random.choice([LET,WIFI])*(jobtran/1000)
        self.jobList[jid].jobCEnergy = LET * (jobtran / 1000)

    def jobcreat(self, jobid, jobtype='normal'):
        jobrun = self.jobRuns  # Sampling from the execution time distribution
        onejob = Job(self.userID, jobid)
        onejob.jobRun = jobrun
        onejob.jobValue = jobrun
        onejob.jobType = jobtype
        onejob.jobCPU = self.jobCPU
        onejob.jobLEnergy = random.choice(self.jobLEnergy)
        return onejob

    def usercreat(self):
        onejob = self.jobcreat(0)
        self.jobList.append(onejob)

        for i in range(1, self.jobNums):
            onejob = self.jobcreat(i)
            self.jobList.append(onejob)

    def userprint(self):
        print("User %d totalfinish %.2f, energy %.2f , age %.2f." % (
        self.userID, self.Throughout, self.CEnergy, self.userAge))

    def usersend(self, x, y, mecid):  # User offloading algorithm
        jobid = -1
        distance = (self.X - x) ** 2 + (self.Y - y) ** 2

        if distance > 1e6:  # Not in range
            self.userMEC[mecid] = 0
            return -1
        else:
            self.userMEC[mecid] = 1

        if (sum(self.userMEC) >= 2) & (
                self.userPriority[mecid] < max(self.userPriority)):  # Multiple connection choices and not optimal
            return -1
        for i in range(len(self.jobList)):
            job = self.jobList[i]

            if job.jobState == 'LW':
                jobid = i
                self.jobappend()  # Ensure that there is a continuous task flow
                return jobid
        return jobid

    def userrun(self):  # Local execution of the most expensive tasks
        jobid = -1
        for i in range(len(self.jobList)):
            job = self.jobList[i]
            if job.jobState == 'LW':
                jobid = i
                return jobid
        return jobid

    def jobrefresh(self, env, fjob):
        jobID = fjob.jobID
        self.Throughout += 1
        self.jobList[jobID].jobFinish = env.now

    def jobappend(self):
        jid = len(self.jobList)
        onejob = self.jobcreat(jid)
        self.jobList.append(onejob)

    def runlocal(self, env):
        while True:
            jobID = self.userrun()
            if jobID == -1:

                self.jobappend()
                continue
            else:

                self.jobList[jobID].jobState = 'LR'  # Local execution
                self.jobList[jobID].jobBegin = env.now
                RUNNINGTIME = self.jobList[jobID].jobRun

                yield env.timeout(RUNNINGTIME)
                self.jobList[jobID].jobState = 'CP'  # Finished
                self.LEnergy += self.jobList[jobID].jobLEnergy

                self.jobrefresh(env, self.jobList[jobID])
                self.jobappend()


class UL(object):  # Total control of the movement of all users
    def __init__(self):
        self.userNum = UN
        self.userList = []  # user list
        self.frequncy = 3600.0

    def reset(self):
        self.userList = []

    def M_strait(self, userID, mecList, Y, X):  # Linear moving model
        userspeed = self.userList[userID].speed * self.frequncy / 100000
        angle_in_radians = random.uniform(0, 360)  # 0~360度夹脚
        self.userList[userID].X += (userspeed * math.cos(angle_in_radians))
        self.userList[userID].Y += (userspeed * math.sin(angle_in_radians))
        self.userList[userID].trace[0].append(self.userList[userID].X)
        self.userList[userID].trace[1].append(self.userList[userID].Y)

        # 用于移动后，判断该用户在那个服务器范围内，然后选择即离用户最近的MEC作为结果返回
        mecList = []
        belongMec = {}
        for mec in mecList:
            distance = mec.calcDistance('%.6f' % mec.mecY, '%.6f' % mec.mecX, '%.6f' % Y, '%.6f' % X)
            if distance < mec.radius :
                mecList.append(mec.mecID)
                belongMec[mec.mecID] = round(distance, 4)
                mec.SCORE += 1
            else:
                print('移动后用户没有在该服务器范围内')  # 即用户的任务就会失败

        # 距离排序 小 -> 大
        disSort = sorted(belongMec.items(), key=lambda item: item[1], reverse=False)
        for mecId, distance in disSort:
            mecList[mecId].includUsers = userID
            self.ul.userList[userID].userBelongMec = True
        # 任务会有时间上的延迟，用户之前所在的服务器与现在所在服务器，进行对比然后把任务结果返回
        # 给当前所在服务器

    def mobile(self, env, mecList):  # userrun is stratege functionname
        while True:
            yield env.timeout(self.frequncy)
            for u in self.userList:
                self.M_strait(u.userID, mecList, u.Y, u.X)

    def save_txt(self, list1, list2, save_path):
        if os.path.isfile(save_path):
            os.remove(save_path)
        with open(save_path, "a") as f:
            for i in range(len(list1)):
                f.write('{} {}\n'.format(list1[i], list2[i]))
            f.close()

    def drawtrace(self, name):  # 没有使用
        fig = plt.figure(figsize=(15, 6))
        plt.rcParams.update({'font.size': 6})
        plt.title("User move")
        l1 = []
        l2 = []
        for uid in range(len(self.userList)):
            u = self.userList[uid]
            for i in range(len(u.trace[0])):
                l1.append(u.trace[0][i])
                l2.append(u.trace[1][i])
            lle = copy.copy(l1)
            llq = copy.copy(l2)
            plt.plot(lle, llq)
            plt.show()
            save_path = 'usermove/user_' + str(uid) + '.data'
            self.save_txt(u.trace[0], u.trace[1], save_path)



class MEC_L(object):
    def __init__(self):
        self.mecNum = MECN
        self.mecList = []


class MEC(object):
    def __init__(self, mecid, userlist):
        # ---------- Basic Information ----------
        self.ID = mecid
        self.userNum = UN
        self.ul = userlist
        self.CHANNEL = 50.0  # Mps, KBpms total bandwidth
        self.RHO = 2.0  # Local and remote execution basic rate ratio ρ
        self.TIMER = 10  # System status refresh frequency
        self.Delta = 5  # Offloading frequency=TIMER*Delta
        self.CB = CB  # Channel bandwidth allocation factor

        self.mecX = 0.0  # mec location information
        self.mecY = 0.0
        self.radius = 0.0

        # ---------- Real-time change information
        self.jobPool = []  # Task list in progress
        self.transPool = []  # Transferring task list
        self.waitingList = []  # Remote waiting list
        self.priorityList = {}  # Priority list
        self.channelUsed = 0.0  # Channel occupied bandwidth
        self.sysTime = 0.0  # System last time
        self.sysCPU = 0.0  # The current CPU usage of the system
        self.ACTION = 0  # The last action taken by the system
        self.SCORE = 0.0  # System last score

        # ---------- log ------------
        self.offloadJob = []  # Task records that have been offloaded
        self.mecAge = 0.0  # Total age
        self.commTime = 0.0  # Total transmission time
        self.commEnergy = 0.0  # Total transmission energy consumption
        self.Run = 0.0  # Total execution time
        self.Throughout = 1  # Total offloading completed tasks
        self.Failure = 1

        # ---------- RL ----------
        self.REWARD = 0.0

        # ----------- MEC  ----------
        self.fail_time = 0 # start failure time
        self.jobFaultProb = 0.0  # The probability of task failure
        self.runRemCount = 0


    def setMEC(self, x, y, rho, channel, r):
        userNum = []
        coverNum = {}
        self.mecX = x  # longitude
        self.mecY = y
        self.RHO = rho  # Computing speed
        self.CHANNEL = channel  # Bandwidth, network resources
        self.radius = r  # coverage

        for u in self.ul.userList:
            distance = self.calcDistance('%.6f' % self.mecY, '%.6f' % self.mecX,
                                         '%.6f' % u.Y, '%.6f' % u.X)
            if distance < self.radius:
                userNum.append(u.userID)
                coverNum[u.userID] = round(distance, 4)


    # Calculate the distance between the edge server and the user
    def calcDistance(self, Lat_A, Lng_A, Lat_B, Lng_B):
        ra = 6378.140  # Equatorial radius (km)
        rb = 6356.755  # Polar radius (km)
        flatten = (ra - rb) / ra  # Oblateness of the earth
        rad_lat_A = radians(float(Lat_A))
        rad_lng_A = radians(float(Lng_A))
        rad_lat_B = radians(float(Lat_B))
        rad_lng_B = radians(float(Lng_B))
        pA = atan(rb / ra * tan(rad_lat_A))
        pB = atan(rb / ra * tan(rad_lat_B))
        xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
        c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
        c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (xx + dr)  # km
        return distance

    # ---------- System log ----------
    def writelog(self, env, fn, name, value, timeslot=5000):
        yield env.timeout(5000)
        f = open('./userNum/USER_' + str(fn) + '_' + str(name) + '_' + str(value) + '.data', 'w')
        oneline = 'TIMESLOT' + ',' + 'Throughout' + ',' + 'Failure' + ',' + 'Failurerate' + ',' + 'User' + \
                  ',' + 'mecAge' + ',' + 'Run' + ',' + 'commTotal' + ',' + 'commEnergy' + ',' + 'reward\n'
        f.write(oneline)
        f.close()
        while True:
            yield env.timeout(timeslot)
            age = 0.0
            run = 0.0
            throu = 0.0
            comm = 0.0
            energy = 0.0
            user = 0

            for i in range(self.userNum):
                user += self.ul.userList[i].userMEC[self.ID]

            sumreward = self.REWARD
            throu = self.Throughout
            fail = self.Failure
            age = self.mecAge / 1000
            run = self.Run
            comm = self.commTime / 1000
            energy = self.commEnergy
            sumreward = self.REWARD
            f = open('./userNum/USER_' + str(fn) + '_' + str(name) + '_' + str(value) + '.data', 'a')
            oneline = str(env.now / 1000) + ',' + str(throu) + ',' + str(fail) + ',' + str(fail / throu) + ',' + str(
                user) + ',' \
                      + str(age) + ',' + str(run) + ',' + str(comm) + ',' + str(energy) + ',' + str(sumreward) + '\n'
            f.write(oneline)
        f.close()

    def writeoffload(self, env,fn, name, value, timeslot=5000):
        yield env.timeout(5000)
        f1 = open('./userNum/JOB_' + str(fn) + '_' + str(name) + '_' + str(value) + '.data', 'w')
        titleline = ('No' + ',' + 'Uid' + ',' + 'Jid' + ',' + 'offloadtime' + ','
                     + 'commutime' + ',' + 'runtime' + ',' + 'energy' + ',' + 'AoI'
                     + ',' + 'state\n')
        f1.write(titleline)
        f1.close()
        i = 0
        while True:
            yield env.timeout(timeslot)
            f2 = f1 = open('./userNum/JOB_' + str(fn) + '_' + str(name) + '_' + str(value) + '.data', 'a')
            for j in self.offloadJob:
                oneline = str(i) + ',' + str(j.userID) + ',' + str(j.jobID) + ',' + str(j.jobOffload / 1000) + ','
                oneline += str(j.jobTran / 1000) + ',' + str(j.jobRun / 1000) + ',' + str(j.jobCEnergy) + ',' + str(
                    j.jobAge / 1000) + ',' + str(j.jobState) + '\n'
                i += 1
                f2.write(oneline)
            f2.close()



    # ---------- RL learning ----------
    def getstate(self):  # seven system status
        state = []
        state.append(self.channelUsed)
        state.append(self.sysCPU)
        state.append(len(self.jobPool))
        state.append(len(self.transPool))
        state.append(self.jobFaultProb)  # The probability that a task may fail
        uwait = 0.0
        utran = 0.0
        for i in self.jobPool:
            uwait += self.ul.userList[i[0]].jobList[i[1]].jobRunLeft
        for j in self.transPool:
            utran += self.ul.userList[j[0]].jobList[j[1]].jobTransLeft
        state.append(uwait)
        state.append(utran)
        state = np.array(state)
        return state

    def reset(self):
        # ---------- Real-time change information ----------
        self.jobPool = []  # Task list in progress
        self.transPool = []  # Transferring task list
        self.waitingList = []  # Remote waiting list
        self.channelUsed = 0.0  # Channel occupied bandwidth
        self.sysTime = 0.0  # System last time
        self.sysCPU = 0.0  # The current CPU usage of the system
        # ---------- log ----------
        self.offloadJob = []  # Task records that have been offloaded
        self.REWARD = 0.0

    # ---------- Channel interference ----------
    def channeldisturb(self, userID, jobID):  # Bandwidth allocation
        cl = self.CHANNEL - self.channelUsed
        cl = cl / self.CB
        self.channelUsed += cl
        jt = self.ul.userList[userID].jobData / cl
        self.ul.userList[userID].jobList[jobID].jobChannel = cl

        return jt

    def offloadOne(self, env, userID):  # offload a task
        jobID = self.ul.userList[userID].usersend(self.mecX, self.mecY, self.ID)
        if jobID == -1:
            return

        TRANSPOTTIME = self.channeldisturb(userID, jobID)  # Really required transmission time

        self.ul.userList[userID].jobList[jobID].jobOffload = env.now  # Task start transmission time
        self.ul.userList[userID].jobList[
            jobID].jobState = 'TS'  # Task status changes, start transmission  TS -transmission
        self.ul.userList[userID].jobList[jobID].jobAge = env.now  # Record moments
        self.ul.userList[userID].jobList[jobID].jobTT = TRANSPOTTIME  # Task real transmission time record
        self.ul.userList[userID].jobList[jobID].jobTransLeft = TRANSPOTTIME  # Remaining transmission time
        self.ul.userList[userID].setjobenergy(jobID, TRANSPOTTIME)  # Task transmission energy consumption calculation
        self.commEnergy += self.ul.userList[userID].jobList[
            jobID].jobCEnergy  # The user spends the total task to transfer energy
        self.transPool.append((userID, jobID))  # Task joins the transfer pool


    def time_per_part(self,mecID):
        re1 = random.normalvariate(PT_MEAN, PT_SIGMA)
        return re1

    def run_remote(self, env, mecID, waitingLen, repairman):  # Remote execution
        while True:
            worktime = self.time_per_part(mecID)
            while worktime:
                try:
                    start = env.now
                    yield env.timeout(worktime)
                    if self.sysCPU > 0.8:  # CPU overload
                        yield env.timeout(self.TIMER * 2) # System status refresh frequency
                        self.SCORE = -abs(self.SCORE)
                        continue
                    else:
                        yield waitingLen.get(1)  # Get a task from the waiting list
                        job = self.waitingList.pop()
                        userID = job['userID']
                        jobID = job['jobID']

                        self.jobPool.append((userID, jobID))  # Put tasks into the execution queue pool
                        self.sysCPU += (
                                    self.ul.userList[userID].jobList[jobID].jobCPU / self.RHO)  # Real resource usage
                        self.ul.userList[userID].jobList[jobID].jobState = 'RR'  # RR-Remote execution
                        self.ul.userList[userID].jobList[jobID].jobBegin = env.now
                        RUNNINGTIME = float(self.ul.userList[userID].jobList[
                                                jobID].jobRun) / self.RHO  # Really required execution time
                        self.ul.userList[userID].jobList[jobID].jobRT = RUNNINGTIME
                        self.ul.userList[userID].jobList[jobID].jobRunLeft = RUNNINGTIME
                        if self.fail_time > self.ul.userList[userID].jobList[jobID].jobAge:
                            if self.ul.userList[userID].jobList[jobID].jobAge + RUNNINGTIME > self.fail_time:
                                chazhi = self.fail_time - self.ul.userList[userID].jobList[jobID].jobAge
                                self.jobFaultProb = chazhi / RUNNINGTIME
                            else:
                                self.jobFaultProb = 1
                    self.runRemCount += 1
                    done_in = 0
                except simpy.Interrupt:
                    self.broken = True
                    done_in -= env.now - start
                    with repairman.request(priority=1) as req:
                        yield req
                        yield env.timeout(REPAIR_TIME)
                    self.broken = False

    # online algorithm
    def runRemoteByOther(self, env, waitingLen):
        while True:
            yield env.timeout(self.TIMER)
            if self.sysCPU > 0.8:
                yield env.timeout(self.TIMER * 2)
                self.SCORE = -abs(self.SCORE)
                continue
            else:
                yield waitingLen.get(1)
                job = self.waitingList.pop(0)
                userID = job.userID
                jobID = job.jobID
                self.jobPool.append((userID, jobID))
                self.sysCPU += self.ul.userList[userID].jobList[jobID].jobCPU

                self.ul.userList[userID].jobList[jobID].jobState = 'RR'
                self.ul.userList[userID].jobList[jobID].jobBegin = env.now
                RUNNINGTIME = float(self.ul.userList[userID].jobList[jobID].jobRun) / self.RHO
                self.ul.userList[userID].jobList[jobID].jobRT = RUNNINGTIME
                self.ul.userList[userID].jobList[jobID].jobRunLeft = RUNNINGTIME

    # ---------- Refresh system-transmission part ----------
    def refresh_trans(self, env, waitingLen):
        while True:
            yield env.timeout(self.TIMER)
            transpool = []
            for Jt in self.transPool:
                userID = Jt[0]
                jobID = Jt[1]
                onejob = self.ul.userList[userID].jobList[jobID]

                if onejob.jobTransLeft > self.TIMER:
                    transpool.append((userID, jobID))
                    self.ul.userList[userID].jobList[jobID].jobTransLeft = self.ul.userList[userID].jobList[
                                                                                 jobID].jobTransLeft - self.TIMER
                else:
                    self.ul.userList[userID].jobList[jobID].jobState = 'RW'  # RW - remote waiting
                    self.channelUsed -= self.ul.userList[userID].jobList[jobID].jobChannel

                    self.waitingList.append({'userID': userID, 'jobID': jobID})
                    self.ul.userList[userID].jobappend()
                    yield waitingLen.put(1)
            self.transPool = transpool

    # ---------- Refresh system-execution part -----------
    def refresh_sys(self, env, name='', value='', flag=1):
        if flag == 1:
            f = open('./userNum/ACTION_' + str(name) + '_' + str(value) + '.data', 'w')
            oneline = ('sysTime' + ',' + 'Action' + ',' + 'ChannelUsed' + ',' + 'TransJob' + ',' + 'CPU' + ',' +
                       'JobFaultProb' + ',' + 'RunningJob' + ',' + 'ActionQos\n')
            f.write(oneline)
            f.close()
        while True:
            yield env.timeout(self.TIMER)
            self.sysTime = env.now
            jobpool = []
            for Jr in self.jobPool:
                userID = Jr[0]
                jobID = Jr[1]
                onejob = self.ul.userList[userID].jobList[jobID]

                if onejob.jobRunLeft > self.TIMER:
                    jobpool.append((userID, jobID))
                    self.ul.userList[userID].jobList[jobID].jobRunLeft = self.ul.userList[userID].jobList[
                                                                               jobID].jobRunLeft - self.TIMER
                else:
                    if flag == 1:
                        f = open('./userNum/ACTION_' + str(name) + '_' + str(value) + '.data', 'a')
                        oneline = str(self.sysTime) + ',' + str(self.ACTION) + ',' + str(
                            self.channelUsed) + ',' + str(len(self.transPool)) + \
                                  ',' + str(self.sysCPU) + ',' + str(self.jobFaultProb) + ',' \
                                  + str(len(self.jobPool))
                        oneline += ',' + str(self.SCORE) + '\n'
                        f.write(oneline)
                    self.sysCPU -= (self.ul.userList[userID].jobCPU / self.RHO)
                    self.ul.userList[userID].jobrefresh(env, self.ul.userList[userID].jobList[jobID])
                    self.offloadJob.append(self.ul.userList[userID].jobList[jobID])
                    # ---------- log -----------
                    self.ul.userList[userID].jobList[jobID].jobAge = env.now - self.ul.userList[userID].jobList[
                        jobID].jobAge
                    self.mecAge += self.ul.userList[userID].jobList[jobID].jobAge
                    self.Run += self.ul.userList[userID].jobList[jobID].jobRun
                    self.commTime += self.ul.userList[userID].jobList[jobID].jobTT
                    self.Throughout += 1
                    # ---------- REWARD ----------
                    score = self.ul.userList[userID].jobList[jobID].jobRun / \
                            (self.ul.userList[userID].jobList[jobID].jobRT +
                             self.ul.userList[userID].jobList[jobID].jobCEnergy)

                    self.SCORE = score
                    self.ul.userList[userID].jobList[jobID].jobState = 'CP'  # CP-completion
                    self.REWARD += self.SCORE
                    self.priorityList[userID] = self.SCORE
                    self.ul.userList[userID].userPriority[self.ID] = self.SCORE
            self.jobPool = jobpool
        f.close()


    # ---------- offloading ----------
    def offloadDQ(self, env, waitingLen, ql):  # deep Q learning
        counter = 0
        while True:
            counter += 1
            yield env.timeout(self.TIMER)

            if (self.CHANNEL - self.channelUsed <= 1) or (self.sysCPU > 0.9):
                self.SCORE = -abs(self.SCORE)
                yield env.timeout(self.TIMER * self.Delta)
                continue
            else:
                observation = self.getstate()
                self.ACTION = ql.RL.choose_action(observation)
                if (counter < UN * 5) or (counter % 10 == 0):
                    pkey = random.sample([i for i in range(UN)], k=self.ACTION)
                else:
                    plist = sorted(self.priorityList.items(), key=lambda i: i[1], reverse=True)
                    pkey = [plist[i][0] for i in range(len(plist))][:self.ACTION]
                for i in range(len(pkey)):
                    userID = pkey[i]
                    self.offloadOne(env, userID)

    def time_to_failure(self):
        fault_time = ra.poisson(lam=100)
        return fault_time


    def break_machine(self, env, mecID, remote):
        while True:
            time = self.time_to_failure()
            yield env.timeout(time)
            self.fail_time = env.now
        if not self.broken:
            self.remote.interrupt()

    def randombin(self, action):
        userlist = list(bin(action).replace('0b', ''))
        zero = self.userNum - len(userlist)
        ll = [0 for i in range(zero)]
        for i in userlist:
            ll.append(int(i))
        return ll

    def channelDisturbByOL(self, userID, jobID, jobnum, channel):
        disturb = np.log2(1 + 1 / (self.CB + jobnum))
        cl = channel * disturb
        if self.channelUsed + cl > self.CHANNEL:
            return -1
        self.channelUsed += cl
        jt = self.ul.userList[userID].jobData / cl
        self.ul.userList[userID].jobList[jobID].jobChannel = cl
        return jt


    def offloadOneByOne(self, env, userID, jobnum, channel, x, y, mecID):
        jobID = self.ul.userList[userID].usersend(x, y, mecID)
        if jobID == -1:
            return

        TRANSPOTTIME = self.channelDisturbByOL(userID, jobID, jobnum, channel)
        if TRANSPOTTIME == -1:
            self.SCORE = -abs(self.SCORE)
            return
        self.ul.userList[userID].jobList[jobID].jobOffload = env.now  # Task start transmission time
        self.ul.userList[userID].jobList[
            jobID].jobState = 'TS'  # Task status changes, start transmission  TS -transmission
        self.ul.userList[userID].jobList[jobID].jobAge = env.now  # Record birthday moments
        self.ul.userList[userID].jobList[jobID].jobTT = TRANSPOTTIME  # Task real transmission time record
        self.ul.userList[userID].jobList[jobID].jobTransLeft = TRANSPOTTIME  # Remaining transmission time
        self.ul.userList[userID].setjobenergy(jobID, TRANSPOTTIME)  # Task transmission energy consumption calculation
        self.commEnergy += self.ul.userList[userID].jobList[
            jobID].jobCEnergy  # The user spends the total task to transfer energy
        self.transPool.append((userID, jobID))  # Task joins the transfer pool

    # ---------- online ----------
    def offloadOL(self, env, waitingLen):
        while True:
            if self.CHANNEL - self.channelUsed <= 1:
                self.SCORE = -abs(self.SCORE)
                yield env.timeout(self.TIMER * self.Delta * 2)
                continue
            yield env.timeout(self.TIMER * self.Delta)

            self.ACTION = random.randint(1, 2 ** self.userNum - 1)

            userlist = self.randombin(self.ACTION)
            jobnum = sum(userlist)
            channel = self.CHANNEL - self.channelUsed

            for i in range(len(userlist)):
                if userlist[i] == 1:
                    userID = i
                    self.offloadOneByOne(env, userID, jobnum, channel,
                                         self.ul.userList[userID].X,
                                         self.ul.userList[userID].Y,
                                         self.ID)


    def offline(self):
        score = 0.0
        action = 1
        for i in range(2 ** self.ul.userNum):
            userlist = self.randombin(i)
            score_ = 0
            jobnum = sum(userlist)
            channel = self.CHANNEL - self.channelUsed
            cl = 0
            for u in range(len(userlist)):
                if userlist[u] == 1:
                    userID = u
                    disturb = np.log2(1 + 1 / (self.CB + jobnum))
                    cl = channel * disturb
                    score_ += np.average(self.ul.userList[userID].jobRuns) / \
                              self.ul.userList[userID].jobData * cl
            if score_ > score:
                score = score_
                action = i
        return action

    # ---------- offline ----------
    def offloadOF(self, env, waitingLen):
        while True:
            if self.CHANNEL - self.channelUsed <= 1:
                self.SCORE = -abs(self.SCORE)
                yield env.timeout(self.TIMER * self.Delta * 2)
                continue
            yield env.timeout(self.TIMER * self.Delta)
            self.ACTION = self.offline()

            userlist = self.randombin(self.ACTION)
            jobnum = sum(userlist)
            channel = self.CHANNEL - self.channelUsed
            for i in range(len(userlist)):
                if userlist[i] == 1:
                    userID = i
                    self.offloadOneByOne(env, userID, jobnum, channel,
                                         self.ul.userList[userID].X,
                                         self.ul.userList[userID].Y,
                                         self.ID
                                         )


class SIMRUN(object):
    def __init__(self,env,mec,ql,repairman):
        self.env = env
        self.mec = mec
        self.broken = False

        waitingLen = simpy.Container(env, buffer, init=len(self.mec.waitingList))
        self.remote = env.process(self.mec.run_remote(self.env, self.mec.ID,waitingLen, repairman))
        env.process(self.mec.refresh_trans(env, waitingLen))
        sys0 = env.process(self.mec.refresh_sys(env, flag=0))
        self.env.process(self.mec.break_machine(env, self.mec.ID,self.remote))
        self.env.process(ql.step(self.mec, env))

""" RL simulation"""
class SIMRUN2(object):
    def __init__(self, env, mec, ql, name, repairman):
        self.env = env
        self.mec = mec
        self.name = name
        self.broken = False

        waitingLen = simpy.Container(env, buffer, init=len(self.mec.waitingList))
        self.remote = env.process(self.mec.run_remote(env, self.mec.ID,waitingLen, repairman))
        env.process(self.mec.refresh_trans(env, waitingLen))
        env.process(self.mec.refresh_sys(env, self.name, 'mec'+str(self.mec.ID)+'_'+str(self.mec.RHO)
                                         +'_'+str(self.mec.CHANNEL)))

        env.process(self.mec.break_machine(env, self.mec.ID,self.remote))
        env.process(self.mec.offloadDQ(env,waitingLen,ql))
        env.process(self.mec.writelog(env,name,'mec'+str(self.mec.ID),
                                      str(self.mec.RHO)+str(self.mec.CHANNEL)))
        env.process(self.mec.writeoffload(env,name, 'mec0', str(self.mec.RHO) + str(self.mec.CHANNEL)))

""" Non RL simulation"""
class SIMRUN3(object):
    def __init__(self, env, mec, methodName, name, repairman):
        self.env = env
        self.mec = mec
        self.methodName = methodName
        self.name = name
        self.broken = False


        waitingLen = simpy.Container(env, buffer, init=len(self.mec.waitingList))
        self.remote = env.process(self.mec.run_remote(env, self.mec.ID,waitingLen, repairman))

        env.process(self.mec.refresh_trans(env, waitingLen))


        env.process(self.mec.refresh_sys(env, self.name, 'mec'+str(self.mec.ID)+'_'+str(self.mec.RHO)
                                         +'_'+str(self.mec.CHANNEL)))

        env.process(self.mec.break_machine(env, self.mec.ID,self.remote))
        if methodName == 'offline':
            env.process(self.mec.offloadOF(env,waitingLen))
        elif methodName == 'online':
            env.process(self.mec.offloadOL(env,waitingLen))

        env.process(self.mec.writelog(env,name,'mec'+str(self.mec.ID),
                                      str(self.mec.RHO)+str(self.mec.CHANNEL)))

        env.process(self.mec.writeoffload(env,name, 'rho', self.mec.RHO))



