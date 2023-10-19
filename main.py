"""
simulation with multiple servers and  users
This code is a modified version and the original code is from https://github.com/snsong/soCoM

"""

import sysModel
from Offload_Strategy import OFFLOAD
import random
import simpy
import xlwt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd

simTime = 144000  # simulation time is 40 hours
random_seed = 40
rho = 2 # the processing speed for mec
buffer = 500 # mec buffer
job_duration = 30
mecNum = 4

def other_job(env, repairman):
    while True:
        done_in = job_duration
        while done_in:
            with repairman.request(priority=2) as req:
                yield req
                try:
                    start = env.now
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= env.now - start


""" NON RL simulation """
def Simulation_NONRL(rho,name,methodName,userlng, userlat, sLng, sLat, rList):
    random.seed(random_seed)
    mec_l = sysModel.MEC_L()
    ul = sysModel.UL()

    for i in range(ul.userNum):
        user = sysModel.User(i)
        user.usersetting(userlng[i],userlat[i])
        user.usercreat()
        ul.userList.append(user)

    for j in range(mec_l.mecNum):
        mec = sysModel.MEC(j, ul)
        mec.RHO = rho * mec.RHO
        mec.setMEC(sLng[j], sLat[j], rho + j * 2, 50, rList[j])
        mec_l.mecList.append(mec)

    name += str(ul.userNum)

    print("Environment create for non rl")
    env = simpy.Environment()

    repairman = simpy.PreemptiveResource(env, capacity=1)
    env.process(ul.mobile(env, mec_l))  # user mobile

    edgeservers = [sysModel.SIMRUN3(env, i, methodName, name, repairman)
                   for i in mec_l.mecList]
    env.process(other_job(env, repairman))
    env.run(until=simTime)

    for user in ul.userList:
        user.userprint()

""" RL Simulation"""
def Simulation_RL(rho, ql, name, userlng, userlat, sLng,
               sLat,  rList):
    random.seed(random_seed)
    mec_l = sysModel.MEC_L()
    ul = sysModel.UL()
    uList = ul.userList

    for i in range(ul.userNum):
        user = sysModel.User(i)
        user.usersetting(userlng[i], userlat[i])
        user.usercreat()
        ul.userList.append(user)

    for j in range(mec_l.mecNum):
        mec = sysModel.MEC(j, ul)
        mec.setMEC(sLng[j], sLat[j], rho + j * 2, 50, rList[j])
        mec_l.mecList.append(mec)

    name += str(ul.userNum)
    print("Environment create for RL")
    env = simpy.Environment()


    repairman = simpy.PreemptiveResource(env, capacity=1)
    env.process(ul.mobile(env, mec_l))  # user mobile

    edgeservers = [sysModel.SIMRUN2(env, i, ql,name, repairman)
                   for i in mec_l.mecList]
    env.process(other_job(env, repairman))
    env.run(until=simTime)

    ul.drawtrace(name)  # Print user track
    for u in ul.userList:
        u.userprint()


# ---------- load data user and server location ---------
userData = pd.read_csv('./dataset/users-melbcbd-generated.csv')
userlng = userData['Longitude'].values.tolist()  # the longitude of user
userlat = userData['Latitude'].values.tolist()  # the latitude of user

df = pd.read_csv('./dataset/edgeResources-melbCBD.csv')
sLng = df['Longitude'].tolist()  # the longitude of mec
sLat = df['Latitude'].tolist()  # the latitude of mec

radiusList = [random.uniform(100, 500) for i in range(mecNum)]  # the coverage of mec


# ---------- NON RL ----------
# online = 'online' + str(sysModel.CB)+'_'
# Simulation_NONRL(rho,online,'online',userlng, userlat, sLng,
#                sLat, radiusList)
#
# offline = 'offline' + str(sysModel.CB)+'_'
# Simulation_NONRL(rho,offline,'offline',userlng, userlat, sLng,
#                sLat, radiusList)


# ---------- RL  ----------
print("BEGIN RL Learning!")
ql = OFFLOAD()
ql.update(ql,random_seed, userlng, userlat, sLng, sLat, radiusList)
ql.printLost()
Simulation_RL(rho, ql, ql.name, userlng, userlat, sLng, sLat,
            radiusList)
