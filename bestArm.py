import numpy as np
from matplotlib import pyplot as plt

import reformData0822 as AR


countA = AR.countA
countY = AR.countY

np.random.seed(0)
n_arms = 2
totalStep = 200
delta = 0.5
epsilon = 0.1
mu = [0.2, 0.4]


class env(object):
    def react(arm):
    #return the rewards
        if np.random.random() < mu[arm]: 
            return 1 
        else: 
            return 0


class ActionEliminationAgent(object):
    def __init__(self):
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0 for _ in range(n_arms)]
        self.arms = [arm for arm in range(n_arms)]
    def calc_ucb(self, arm, step):
        ucb = self.values[arm]
        ucb += np.sqrt(np.log(1 * n_arms * np.power(step, 1) / delta) / (2 * step))
        return ucb
    def calc_lcb(self, arm, step):
        lcb = self.values[arm]
        lcb -= np.sqrt(np.log(1 * n_arms * np.power(step, 1) / delta) / (2 * step))
        return lcb
    def get_arm(self, step):
        if step == 0:
            return self.arms, 1
        else:
            ucb = [self.calc_ucb(arm, step) for arm in range(n_arms)]
            maxUCB = 0
            armNum = len(self.arms)
            armHt = self.values.index(max(self.values))
            for i in self.arms:
                if i != armHt:
                    maxUCB = max(maxUCB, ucb[i])
            armLt = ucb.index(maxUCB)
            if armNum > 1:
                if self.calc_lcb(armHt,step) + epsilon > ucb[armLt]:
                    return armHt, 0
                else:
                    for i in self.arms:
                        if self.calc_lcb(armLt, step) > ucb[i]:
                            self.arms.remove(i)
                    return self.arms, 1
            elif armNum == 1: 
                return self.arms, 0
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]



class UCBAgent(object):
    def __init__(self):
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0 for _ in range(n_arms)]
    def calc_ucb(self, arm):
        ucb = self.values[arm]
        ucb += np.sqrt(np.log(5 * n_arms * np.power(sum(self.counts), 1) / 4 * delta) / (2 * self.counts[arm]))
        return ucb
    def calc_lcb(self, arm):
        lcb = self.values[arm]
        lcb -= np.sqrt(np.log(5 * n_arms * np.power(sum(self.counts), 1) / 4 * delta) / (2 * self.counts[arm]))
        return lcb
    def get_arm(self, step):
        if 0 in self.counts:
            arm = self.counts.index(0)
            return [arm], 1
        else:
            ucb = [self.calc_ucb(arm) for arm in range(n_arms)]
            armHt = self.values.index(max(self.values))
            armLt = ucb.index(max(ucb))
            if armLt == armHt:
                armLt = ucb.index(sorted(ucb)[-2])
            arm = armLt
            if self.calc_lcb(armHt) + epsilon > ucb[armLt]:
                return armHt, 0
            else: 
                return [arm], 1
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]

class LUCBAgent(object):
    def __init__(self):
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0 for _ in range(n_arms)]
    def calc_ucb(self, arm):
        ucb = self.values[arm]
        ucb += np.sqrt(np.log(5 * n_arms * np.power(sum(self.counts), 1) / 4 * delta) / (2 * self.counts[arm]))
        return ucb
    def calc_lcb(self, arm):
        lcb = self.values[arm]
        lcb -= np.sqrt(np.log(5 * n_arms * np.power(sum(self.counts), 1) / 4 * delta) / (2 * self.counts[arm]))
        return lcb
    def get_arm(self, step):
        if step == 0:
            return [arm for arm in range(n_arms)], 1, 0, 0
        else:
            ucb = [self.calc_ucb(arm) for arm in range(n_arms)]
            armHt = self.values.index(max(self.values))
            armLt = ucb.index(max(ucb))
            if armLt == armHt:
                armLt = ucb.index(sorted(ucb)[-2])
            arm = [armHt, armLt]
            if self.calc_lcb(armHt) + epsilon > ucb[armLt]:
                return armHt, 0, armHt, armLt
            else:
                return arm, 1, armHt, armLt
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]



def sim(Agent, **kwargs):
    selected_arms = []
    earned_rewards = []
    ucbs = []
    values =[]
    step = 0
    contn = 1
    agent = Agent(**kwargs)
    while contn == 1:
        arm, contn, ucb, value= agent.get_arm(step)
        ucbs.append(ucb)
        values.append(value)
        if contn == 1:
            for a in arm:
                reward = env.react(a)
                agent.sample(a, reward)
                earned_rewards.append(reward)
            selected_arms.append(arm)
            step += 1
        
    return selected_arms, earned_rewards, arm, step, ucbs, values




arms_ucb, reward_ucb, Barm_ucb, step_ucb, ucb_ucb, value_ucb= sim(LUCBAgent)
#arms_ae, reward_ae, Barm_ae, step_ae = sim(ActionEliminationAgent)

for a in range(len(ucb_ucb)):
    print(a, ucb_ucb[a], value_ucb[a])
