#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("..")

gamma = 0.98 ### 0.8 / 0.98
lam = 0.0 ### 0.0 / 0.95

def gae(env, policy, *, gam=gamma, lam=lam):
    from mannequin import Trajectory, Adam
    from mannequin.basicnet import Input, Affine, Tanh
    from mannequin.gym import one_step

    def SimplePredictor(in_size, out_size):
        model = Input(in_size)
        for _ in range(2):
            model = Tanh(Affine(model, 64))
        model = Affine(model, out_size, init=0.1)
        return model

    # Assume continuous observation space
    value_predictor = SimplePredictor(env.observation_space.low.size, 1)

    def get_chunk():
        hist = []
        length = 2048

        # Run steps in the environment
        while len(hist) < length:
            hist.append(one_step(env, policy, include_next_obs=True))

        # Finish the last episode
        while not hist[-1][3]:
            hist.append(one_step(env, policy, include_next_obs=True))
            length += 1

        # Estimate value function for each state
        value = value_predictor(
            [hist[i][0] for i in range(length)]
        ).reshape(-1)

        # Estimate value function for each next state
        next_value = value_predictor(
            [hist[i][4] for i in range(length)]
        ).reshape(-1).copy()

        # Value of the last observation per episode
        # cannot be estimated using rewards
        for i in range(len(hist)):
            if hist[i][3] == True:
                next_value[i] = 0.

        # Compute advantages
        adv = np.zeros(length + 1, dtype=np.float32)
        for i in range(length-1, -1, -1):
            adv[i] = hist[i][2] + gam * next_value[i] - value[i]
            if not hist[i][3]:
                adv[i] += gam * lam * adv[i+1]
        adv = adv[:length]

        # Return a joined trajectory with advantages as rewards
        traj = Trajectory(
            [hist[i][0] for i in range(length)],
            [hist[i][1] for i in range(length)],
            adv
        )

        return traj

    return get_chunk

def run():
    from mannequin import RunningNormalize, Adam
    from mannequin.gym import NormalizedObservations
    from _env import build_env, get_progress
    from policy import stochastic_policy

    env = build_env()
    env = NormalizedObservations(env)

    policy = stochastic_policy(env)
    opt = Adam(policy.get_params(), horizon=10)
    normalize = RunningNormalize(horizon=2)
    get_chunk = gae(env, policy.sample)

    while get_progress() < 1.0:
        cur_step = get_progress(divide=False)[0]
        traj = get_chunk()
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)
        _, backprop = policy.logprob.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r), lr=0.01)
        policy.load_params(opt.get_value())

if __name__ == '__main__':
    run()
