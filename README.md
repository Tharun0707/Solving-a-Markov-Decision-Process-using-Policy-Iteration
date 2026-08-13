# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)

num_states = env.observation_space.n
num_actions = env.action_space.n

gamma = 0.99
theta = 1e-10

V = np.zeros(num_states)

policy = np.zeros(num_states, dtype=int)

def policy_evaluation(policy):
    V = np.zeros(num_states)

    while True:
        delta = 0

        for state in range(num_states):
            old_value = V[state]
            action = policy[state]
            new_value = 0

            for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                new_value += probability * (
                    reward + gamma * V[next_state] * (not terminated)
                )

            V[state] = new_value
            delta = max(delta, abs(old_value - new_value))

        if delta < theta:
            break

    return V


def policy_improvement(V, policy):
    policy_stable = True
    new_policy = np.copy(policy)

    for state in range(num_states):

        action_values = np.zeros(num_actions)

        for action in range(num_actions):

            for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                action_values[action] += probability * (
                    reward + gamma * V[next_state] * (not terminated)
                )

        best_action = np.argmax(action_values)

        if best_action != policy[state]:
            policy_stable = False

        new_policy[state] = best_action

    return new_policy, policy_stable


def policy_iteration():
    policy = np.zeros(num_states, dtype=int)
    total_iterations = 0

    while True:
        V = policy_evaluation(policy)

        policy, policy_stable = policy_improvement(V, policy)

        total_iterations += 1

        if policy_stable:
            break

    return V, policy, total_iterations


V, optimal_policy, iterations = policy_iteration()

print("Total policy iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 6))

print("\nOptimal Policy:")

policy_symbols = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

for row in optimal_policy.reshape(4, 4):
    print(" ".join(policy_symbols[action] for action in row))

env.close()
```

## Output

<img width="405" height="239" alt="image" src="https://github.com/user-attachments/assets/c99d68ec-3791-4aaf-9a35-9af218c0032c" />


## Result

The Policy Iteration algorithm was successfully implemented to obtain the optimal state-value function and optimal policy for the FrozenLake-v1 environment.

## Inference

Policy Iteration successfully finds an optimal policy for the FrozenLake environment. The algorithm repeatedly evaluates the current policy and improves it by selecting the action with the highest expected value. The process stops when the policy becomes stable, indicating that further policy improvement does not change the selected actions.
