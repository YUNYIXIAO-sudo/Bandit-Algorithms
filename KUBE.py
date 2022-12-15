import reformData0822 as AR

countA = AR.countA
countY = AR.countY

import numpy as np
from matplotlib import pyplot as plt


np.random.seed(0)

n_arms = 2
Budget = 1000
cost = [14, 5]


class env(object):
    def react(step, arm):
    #return the rewards
        if arm == 0:
            return countY[step][0]
        elif arm == 1:
            return countA[step][0]

    def opt(step):
        if env.react(step, 0) == 1 and env.react(step, 1) == 0:
            return 0, countY[step][0]
        else:
            return 1, countA[step][0]



class KUBEAgent(object):
    def __init__(self):
        self.step = 0
        self.B = Budget
        self.arms = []
        self.rewards = []
        self.counts = np.zeros(n_arms)
        self.expectedReward = np.zeros(n_arms)
    def dga(self, step):
        ucb = np.zeros(n_arms)
        e = np.zeros(n_arms)
        m = np.zeros(n_arms)
        li = []
        for i in range(n_arms):
            ucb[i] = self.expectedReward[i] + np.sqrt(2 * np.log(step) / self.counts[i])
            e[i] = ucb[i] / cost[i]
            li.append([i, e[i], cost[i]])
        li = sorted(li, key=lambda x: x[1], reverse=True)
        B = self.B
        for i in range(n_arms):
            m[i] = B // li[i][2]
            B -= m[i] * li[i][2]
        return m[0] / (m[0] + m[1]), [li[0][0], li[1][0]]
    def pull_arm(self):
        while self.B > min(cost):
            if self.step < n_arms:
                it = self.step
                self.sample(it, env.react(self.step, it))
            else:
                p, a = self.dga(self.step)
                if np.random.random() < p:
                    it = a[0]
                else:
                    it = a[1]
                self.sample(it, env.react(self.step, it))
            self.B -= cost[it]
            self.step += 1
    def sample(self, arm, reward):
        self.arms.append(arm)
        self.rewards.append(reward)
        self.counts[arm] += 1
        self.expectedReward[arm] = ((self.counts[arm] - 1) * self.expectedReward[arm] + reward) / self.counts[arm]
    def result(self):
        self.pull_arm()
        return self.arms, self.rewards, self.B

class RandomAgent(object):
    def __init__(self):
        self.arms = []
        self.rewards = []
        self.step = 0
        self.B = Budget
    def pull_arm(self):
        while self.B > min(cost):
            arm = np.random.randint(n_arms)
            self.sample(arm, env.react(self.step, arm))
            self.B -= cost[arm]
            self.step += 1
    def sample(self, arm, reward):
        self.arms.append(arm)
        self.rewards.append(reward)
    def result(self):
        self.pull_arm()
        return self.arms, self.rewards


class OracleAgent(object):
    def __init__(self):
        self.B = Budget
        self.step = 0
        self.arms = []
        self.rewards = []
    def pull_arm(self):
        while self.B > min(cost):
            a, r = env.opt(self.step)
            self.sample(a, r)
            self.B -= cost[a]
            self.step += 1
    def sample(self, arm, reward):
        self.arms.append(arm)
        self.rewards.append(reward)
    def result(self):
        self.pull_arm()
        return self.arms, self.rewards



def sim(Agent, N=1, **kwargs):
    agent = Agent(**kwargs)
    return agent.result()


arms_rd, reward_rd = sim(RandomAgent)
arms_kb, reward_kb, remainingB = sim(KUBEAgent)
arms_opt, reward_opt = sim(OracleAgent)


from collections import Counter

print(Counter(arms_rd))
print(Counter(arms_kb))
print(Counter(arms_opt))



