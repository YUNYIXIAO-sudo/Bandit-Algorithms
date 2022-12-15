import numpy as np
from matplotlib import pyplot as plt


np.random.seed(0)

n_arms = 2

class env(object):
    thetas = [0.3, 0.6]
    thetas2 = [0.5, 0,2]

    def react(arm, step):
        if step < totalStep // 2:
            return 1 if np.random.random() < env.thetas[arm] else 0
        


    def opt():
        return np.argmax(env.thetas)


class EpsilonGreedyAgent(object):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    def get_arm(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(n_arms)
        else:
            arm = np.argmax(self.values)
        return arm
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]

class AnnealingEpsilonGreedyAgent(object):
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    def get_arm(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(n_arms)
        else:
            arm = np.argmax(self.values)
        self.epsilon *= 0.99
        return arm
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]

class RandomAgent(object):
    def __init__(self):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    def get_arm(self):
        arm = np.random.randint(n_arms)
        return arm
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]

class OracleAgent(object):
    def __init__(self):
        self.arm = env.opt()
    def get_arm(self):
        return self.arm
    def sample(self, arm, reward):
        pass

class UCBAgent(object):
    def __init__(self):
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0 for _ in range(n_arms)]
    def calc_ucb(self, arm):
        ucb = self.values[arm]
        ucb += np.sqrt(np.log(sum(self.counts)) / (2 * self.counts[arm]))
        return ucb
    def get_arm(self):
        if 0 in self.counts:
            arm = self.counts.index(0)
        else:
            ucb = [self.calc_ucb(arm) for arm in range(n_arms)]
            arm = ucb.index(max(ucb))
        return arm
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]

class DUCBAgent(object):
    def __init__(self):
        self.upsilon = 0.5
        self.zeta = 0.7
        self.nk = 0
        self.n = [0 for _ in range(n_arms)]
        self.myu = [0 for _ in range(n_arms)]
        self.b = [0 for _ in range(n_arms)]
        self.ucb = [0 for _ in range(n_arms)]
    def calc_ucb(self, arm, reward, step):
        self.n[arm] = self.n[arm] + self.upsilon ** (step) 
        self.nk += self.upsilon ** (step)
        if reward == 1:
            #step starts from 0, step + 1 - 1 = step
            self.myu[arm] = self.myu[arm] + self.upsilon ** (step) #reward = 1
        for i in range(n_arms):
            self.b[i] = 2 * np.sqrt((self.zeta * np.log(self.nk) / self.n[i]))
            self.ucb[i] = self.myu[arm] / self.n[arm] + self.b[i]
    def get_arm(self, step):
        if step < n_arms:
            arm = step
        else:
            arm = np.argmax(self.ucb)
        return arm
    def sample(self, arm, reward, step):
        self.calc_ucb(arm, reward, step)


class SWUCBAgent(object):
    def __init__(self):
        self.tau = 4
        self.zeta = 0.7
        self.counts = [[0 for _ in range(totalStep)] for _ in range(n_arms)]
        self.values = [[0 for _ in range(totalStep)] for _ in range(n_arms)]
        self.n = [0 for _ in range(n_arms)]
        self.myu = [0 for _ in range(n_arms)]
        self.b = [0 for _ in range(n_arms)]
        self.ucb = [0 for _ in range(n_arms)]
    def calc_ucb(self, arm, reward, step):
        if step < self.tau + 1:
            self.n[arm] += 1
            self.myu[arm] += reward
            self.myu[arm] /= self.n[arm]
        else:
            self.n[arm] = self.n[arm] - self.counts[arm][step - self.tau + 1] + 1
            self.myu[arm] = (self.myu[arm] - self.values[arm][step - self.tau + 1] + reward) / self.n[arm]
        for i in range(n_arms):
            self.b[i] = np.sqrt((self.zeta * np.log(min(step + 1, self.tau)) / self.n[i]))
            self.ucb[i] = self.myu[i] + self.b[i]
    def get_arm(self, step):
        if step < n_arms:
            arm = step
        else:
            arm = np.argmax(self.ucb)
        return arm
    def sample(self, arm, reward, step):
        for i in range(n_arms):
            if i != arm:
                self.counts[i][step] = 0
                self.values[i][step] = 0
            else:
                self.counts[arm][step] = 1
                self.values[arm][step] = reward
        self.calc_ucb(arm, reward, step)        



class SoftmaxAgent(object):
    def __init__(self, tau=.05):
        self.tau = tau
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    def softmax_p(self):
        logit = self.values / self.tau
        logit = logit - np.max(logit) #オーバーフローを回避するためにlogitの最大値を引く
        p = np.exp(logit) / sum(np.exp(logit))
        return p
    def get_arm(self):
        arm = np.random.choice(n_arms, p=self.softmax_p())
        return arm
    def sample(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]


class AnnealingSoftmaxAgent(object):
    def __init__(self, tau=1000.):
        self.tau = tau
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    def softmax_p(self):
        logit = self.values / self.tau
        logit = logit - np.max(logit)
        p = np.exp(logit) / sum(np.exp(logit))
        return p
    def get_arm(self, step):
        arm = np.random.choice(n_arms, p=self.softmax_p())
        self.tau *= 0.9
        return arm
    def sample(self, arm, reward, step):
        self.counts[arm] += 1
        self.values[arm] = ((self.counts[arm] - 1) * self.values[arm] + reward) / self.counts[arm]

class BernoulliTSAgent(object):
    def __init__(self):
        self.counts = [0 for _ in range(n_arms)]
        self.wins = [0 for _ in range(n_arms)]
    def get_arm(self, step):
        beta = lambda N, a: np.random.beta(a + 1, N - a + 1)
        #観測回数N 報酬を得た回数a　→ 事後分布を得る
        result = [beta(self.counts[i], self.wins[i]) for i in range(n_arms)]
        arm = result.index(max(result))
        return arm
    def sample(self, arm, reward, step):
        self.counts[arm] += 1
        self.wins[arm] = self.wins[arm] + reward

totalStep = 200
def sim(Agent, N=200, T=totalStep, **kwargs):
    selected_arms = [[0 for _ in range(T)] for _ in range(N)]
    earned_rewards = [[0 for _ in range(T)] for _ in range(N)]

    for n in range(N):
        agent = Agent(**kwargs)
        for t in range(T):
            arm = agent.get_arm(T)
            reward = env.react(arm)
            agent.sample(arm, reward, T)
            selected_arms[n][t] = arm
            earned_rewards[n][t] = reward
    return np.array(selected_arms), np.array(earned_rewards)


arms_eg, reward_eg = sim(EpsilonGreedyAgent)
arms_rd, reward_rd = sim(RandomAgent)
arms_ae, reward_ae = sim(AnnealingEpsilonGreedyAgent)
arms_sm, reward_sm = sim(SoftmaxAgent)
arms_ts, reward_ts = sim(BernoulliTSAgent)
arms_ucb, reward_ucb = sim(UCBAgent)
arms_oc, reward_oc = sim(OracleAgent)
arms_as, reward_as = sim(AnnealingSoftmaxAgent)

#plt.plot(np.mean(arms_eg == env.opt(), axis=0), label=r'egreedy')
#plt.plot(np.mean(arms_rd == env.opt(), axis=0), label=r'random')
#plt.plot(np.mean(arms_ae == env.opt(), axis=0), label=r'aegreedy')
#plt.plot(np.mean(arms_sm == env.opt(), axis=0), label=r'softmax')
#plt.plot(np.mean(arms_ts == env.opt(), axis=0), label=r'thompson')
#plt.plot(np.mean(arms_ucb == env.opt(), axis=0), label=r'ucb')
#plt.plot(np.mean(arms_as == env.opt(), axis=0), label=r'Asoftmax')

plt.plot(np.mean(np.cumsum(reward_rd, axis=1), axis=0), label=r'random')
plt.plot(np.mean(np.cumsum(reward_ae, axis=1), axis=0), label=r'aegreedy')
plt.plot(np.mean(np.cumsum(reward_ts, axis=1), axis=0), label=r'thompson')
#plt.plot(np.cumsum(reward_rd), label=r'ucb')
plt.plot(np.mean(np.cumsum(reward_as, axis=1), axis=0), label=r'Asoftmax')

plt.xlabel(r'$t$')
plt.ylabel(r'$mathnn{E}[X(t) = X^*]$')
plt.legend()
plt.show()

#plt.plot(np.mean(arms_oc == env.opt(), axis=0), label=r'oracle')