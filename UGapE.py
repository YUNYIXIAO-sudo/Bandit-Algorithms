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
            return [arm for arm in range(n_arms)], 1
        else:
            ucb = [self.calc_ucb(arm) for arm in range(n_arms)]
            armHt = self.values.index(max(self.values))
            armLt = ucb.index(max(ucb))
            if armLt == armHt:
                armLt = ucb.index(sorted(ucb)[-2])
            arm = [armHt, armLt]
            if self.calc_lcb(armHt) + epsilon > ucb[armLt]:
                return armHt, 0
            else:
                return arm, 1
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]



def sim(Agent, **kwargs):
    selected_arms = []
    earned_rewards = []
    step = 0
    contn = 1
    agent = Agent(**kwargs)
    while contn == 1:
        arm, contn = agent.get_arm(step)
        if contn == 1:
            for a in arm:
                reward = env.react(a)
                agent.sample(a, reward)
                earned_rewards.append(reward)
            selected_arms.append(arm)
            step += 1
    return selected_arms, earned_rewards, arm, step





arms_lucb, reward_lucb, Barm_lucb, step_lucb = sim(LUCBAgent)


print(step_lucb, Barm_lucb)
