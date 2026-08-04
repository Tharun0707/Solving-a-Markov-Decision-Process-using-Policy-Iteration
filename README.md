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

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8

# -------------------------------------------------
# Helper Function: One-Step Lookahead
# -------------------------------------------------

def one_step_lookahead(env, state, V, gamma):

    action_values = np.zeros(env.action_space.n)

    for action in range(env.action_space.n):

        for probability, next_state, reward, terminated in env.P[state][action]:

            action_values[action] += probability * (
                reward + gamma * V[next_state] * (not terminated)
            )

    return action_values


# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------

def policy_evaluation(policy, env, gamma=0.99, theta=1e-8):

    V = np.zeros(env.observation_space.n)

    while True:

        delta = 0

        for state in range(env.observation_space.n):

            v = 0

            for action, action_prob in enumerate(policy[state]):

                for probability, next_state, reward, terminated in env.P[state][action]:

                    v += action_prob * probability * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            delta = max(delta, abs(v - V[state]))
            V[state] = v

        if delta < theta:
            break

    return V


# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------

def policy_improvement(V, env, gamma=0.99):

    policy = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )

    for state in range(env.observation_space.n):

        action_values = one_step_lookahead(env, state, V, gamma)

        best_action = np.argmax(action_values)

        policy[state][best_action] = 1.0

    return policy


# -------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration(env, gamma=0.99, theta=1e-8):

    policy = np.ones(
        (env.observation_space.n, env.action_space.n)
    ) / env.action_space.n

    iterations = 0

    while True:

        iterations += 1

        V = policy_evaluation(policy, env, gamma, theta)

        new_policy = policy_improvement(V, env, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    print("\nTotal policy iterations:", iterations)

    return policy, V


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(V):

    print("\nOptimal State-Value Function:")
    print(np.round(V.reshape(4,4),4))


def print_policy(policy):

    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)

    policy_grid = np.array(
        [action_symbols[a] for a in best_actions]
    ).reshape(4,4)

    print("\nOptimal Policy:")
    print(policy_grid)


# -------------------------------------------------
# Run Policy Iteration
# -------------------------------------------------

optimal_policy, optimal_value_function = policy_iteration(
    env,
    gamma=gamma,
    theta=theta
)

print("\nName:  GANESH D")
print("Register Number: 212223240035")

print_value_function(optimal_value_function)
print_policy(optimal_policy)

env.close()
```

## Output

<img width="374" height="296" alt="image" src="https://github.com/user-attachments/assets/e7a14168-9173-46ba-a463-a91ce508b67e" />


## Result

The Policy Iteration algorithm was successfully implemented to obtain the optimal state-value function and optimal policy for the FrozenLake-v1 environment.

## Inference

The Policy Iteration algorithm successfully found the optimal policy and optimal state-value function for the FrozenLake environment. By changing the discount factor to γ = 0.99, the algorithm still converged successfully, but the state-value function changed slightly because future rewards were given a little less importance. The optimal policy remained the same for this environment.
