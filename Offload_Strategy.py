"""
offloading strategy
"""
import sysModel
# from RL_DQN import DQN
# from RL_DDQN import DDQN
from RL_DuelingDQN import DuelingDQN
# from RL_PRDQN import PRDQN
import simpy
import random
import xlwt
import numpy as np

userNum = 10
mecNum = 4
pen = 1
# name = 'DQN'
# name = 'DDQN'
name = 'DuelingDQN'
# name = 'PRDQN'
simTime = 800

LEPI = 100 # episode
jobDuration= 30



def other_job(env, repairman):
    while True:
        done_in = jobDuration
        while done_in:
            with repairman.request(priority=2) as req:
                yield req
                try:
                    start = env.now
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= env.now - start



class OFFLOAD(object):
    def __init__(self):
        self.name = name
        self.action_space = [str(i) for i in range(2)]
        self.n_actions = userNum  # 0-local，1-offload
        self.n_features = 7  # system state - getstate
        self.RL = DuelingDQN(self.n_actions, self.n_features,
                      learning_rate=0.03,
                      reward_decay=0.4,
                      e_greedy=0.9,  # exploration
                      replace_target_iter=200,
                      memory_size=20000,
                      # output_graph=True
                      )
        self.done = True
        self.stepcount = 0

    def reset(self):
        self.done = True
        self.stepcount = 0

    def printLost(self):
        self.RL.plot_lost(self.name, self.n_actions)

    def refreshstep(self, mec_, env_):
        while True:
            yield env_.timeout(mec_.TIMER)
            mec_.sysTime = env_.now
            jobpool = []
            for Jr in mec_.jobPool:
                userID = Jr[0]
                jobID = Jr[1]
                onejob = mec_.ul.userList[userID].jobList[jobID]

                if onejob.jobRunLeft > mec_.TIMER:
                    jobpool.append((userID, jobID))
                    mec_.ul.userList[userID].jobList[jobID].jobRunLeft = mec_.ul.userList[userID].jobList[
                                                                               jobID].jobRunLeft - mec_.TIMER
                else:
                    mec_.SYS_CPU -= (mec_.ul.userList[userID].jobCPU / mec_.RHO)
                    mec_.ul.userList[userID].jobrefresh(env_, mec_.ul.userList[userID].jobList[jobID])
                    mec_.offloadJob.append(mec_.ul.userList[userID].jobList[jobID])

                    mec_.ul.userList[userID].jobList[jobID].jobAge = env_.now - mec_.ul.userList[userID].jobList[
                        jobID].jobAge
                    mec_.Age += mec_.ul.userList[userID].jobList[jobID].jobAge
                    mec_.Run += mec_.ul.userList[userID].jobList[jobID].jobRun
                    mec_.commTime += mec_.ul.userList[userID].jobList[jobID].jobTT
                    mec_.Throughout += 1

                    failrate = mec_.Failure / mec_.Throughout

                    score = mec_.ul.userList[userID].jobList[jobID].jobRun / mec_.ul.userList[userID].jobList[
                        jobID].jobCEnergy

                    if (mec_.ul.userList[userID].jobList[jobID].jobAge > mec_.ul.userList[userID].jobList[
                        jobID].jobRun):
                        mec_.SCORE = -abs(score) * (1 - failrate)
                        mec_.ul.userList[userID].jobList[jobID].jobState = 'FL'
                        mec_.Failure += 1
                    else:
                        mec_.SCORE = score * (1 - failrate)
                        mec_.ul.userList[userID].jobList[jobID].jobState = 'CP'
                    mec_.REWARD += mec_.SCORE

                    mec_.priorityList[userID] = mec_.ACTION
                    mec_.ul.userList[userID].userPriority[mec_.ID] = mec_.ACTION


            mec_.jobPool = jobpool

    def step(self, mec_, env_):
        counter = 0
        while True:
            counter += 1
            yield env_.timeout(mec_.TIMER)

            if (mec_.CHANNEL - mec_.channelUsed <= 1) or (mec_.sysCPU > 0.9):
                mec_.SCORE = -abs(mec_.SCORE)
                yield env_.timeout(mec_.TIMER * mec_.Delta)
                continue
            else:
                observation = mec_.getstate()
                mec_.ACTION = self.RL.choose_action(observation)
                if (counter < userNum * 5) or (counter % 10 == 0):
                    pkey = random.sample([i for i in range(userNum)], k=mec_.ACTION)
                else:
                    plist = sorted(mec_.priorityList.items(), key=lambda i: i[1], reverse=True)
                    pkey = [plist[i][0] for i in range(len(plist))][:mec_.ACTION]
                for i in range(len(pkey)):
                    userID = pkey[i]
                    mec_.offloadOne(env_, userID)
                observation_ = mec_.getstate()
                reward = mec_.SCORE

                self.RL.store_transition(observation, mec_.ACTION, reward, observation_)
                if (self.stepcount > 40) and (self.stepcount % 4 == 0):
                    self.RL.learn()
                observation = observation_
                self.stepcount += 1

    def update(self, ql,RDSEED, uLng, uLat, sLng, sLat,
               rList):
        self.reset()
        for episode in range(LEPI):
            self.reset()
            print("episode %d step count %d" % (episode, self.stepcount))
            random.seed(RDSEED)
            ul = sysModel.UL()

            for i in range(ul.userNum):
                user = sysModel.User(i)
                user.usersetting(uLng[i], uLat[i])
                user.usercreat()
                ul.userList.append(user)

            mec_l = sysModel.MEC_L()
            rho = 2
            for j in range(mec_l.mecNum):
                mec = sysModel.MEC(j, ul)
                mec.setMEC(sLng[j], sLat[j], rho + j * 2, 50, rList[j])
                mec_l.mecList.append(mec)

            env = simpy.Environment()
            repairman = simpy.PreemptiveResource(env, capacity=1)

            env.process(ul.mobile(env, mec_l))  # user mobile

            edgeservers = [sysModel.SIMRUN(env, i, ql,repairman)
                           for i in mec_l.mecList]
            env.process(other_job(env,repairman))
            env.run(until=simTime)
        self.reset()

