# POLICY EVALUATION

## AIM
To implement and evaluate a given policy in a reinforcement learning environment using the iterative policy evaluation algorithm, and to analyze the resulting state-value function for different policies.

## PROBLEM STATEMENT
In reinforcement learning, the objective is to estimate the value function of a policy, which represents the expected long-term reward for each state when following that policy. The problem is to implement policy evaluation, apply it to different policies in a given environment (e.g., FrozenLake), compute their state-value functions, and compare the performance of the policies based on their values.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        prev_V = np.copy(V)
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * prev_V[next_state] * (1 - done))
            V[s] = v
        if np.max(np.abs(prev_V - V)) < theta:
            break
    return V
```

## OUTPUT:
### First Policy

![output](policy.png)
![output](po1.png)
### Second Policy
![output](policy2.png)
![output](po2.png)
### OUTPUT
![output](state.png)
![output](final.png)
## RESULT:


Thus the policy evaluation function has been successfully implemented and output is obtained.
