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

<img width="576" height="128" alt="Screenshot 2026-05-11 234308" src="https://github.com/user-attachments/assets/ebbccfaf-10f3-4934-952a-a21828a1f54a" />
<img width="716" height="30" alt="Screenshot 2026-05-11 234314" src="https://github.com/user-attachments/assets/6613ac8a-b3a1-4d81-817e-8c0a71a912ab" />

### Second Policy
<img width="615" height="176" alt="Screenshot 2026-05-11 234326" src="https://github.com/user-attachments/assets/a026d559-4550-4bdd-9bd2-97003c1b6980" />
<img width="923" height="52" alt="Screenshot 2026-05-11 234329" src="https://github.com/user-attachments/assets/6b0e3a81-703a-4984-9a1c-e9feecdbf257" />

### Output
<img width="650" height="233" alt="Screenshot 2026-05-11 234337" src="https://github.com/user-attachments/assets/6c26f79c-db54-4a8d-8949-84a5accfcf30" />
<img width="456" height="47" alt="Screenshot 2026-05-11 234341" src="https://github.com/user-attachments/assets/722ee8dd-81e9-402f-9b73-817ca7e8cb5c" />

## RESULT:


Thus the policy evaluation function has been successfully implemented and image is obtained.
