#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("..")

gamma = 0.98 ### 0.8 / 0.98
lam = 0.0 ### 0.0 / 0.95
last_value_estimator = "zero" ### "zero" / "next_value" / "Zolna"
which_target_value = "use_rewards"

def gae(env, policy, *, gam=gamma, lam=lam):
    from mannequin import Trajectory, Adam
    from mannequin.basicnet import Input, Affine, LReLU
    from mannequin.gym import one_step

    def SimplePredictor(in_size, out_size):
        model = Input(in_size)
        for _ in range(2):
            model = LReLU(Affine(model, 64))
        model = Affine(model, out_size, init=0.1)

        opt = Adam(model.get_params(), horizon=10, lr=0.003)

        def sgd_step(inps, lbls):
            outs, backprop = model.evaluate(inps)
            opt.apply_gradient(backprop(lbls - outs))
            model.load_params(opt.get_value())

        model.sgd_step = sgd_step
        return model

    rng = np.random.RandomState()

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
        if last_value_estimator == "zero":
            for i in range(len(hist)):
                if hist[i][3] == True:
                    next_value[i] = 0.
        elif last_value_estimator == "next_value":
            # already done
            pass
        elif last_value_estimator == "Zolna":
            assert gamma > 0. and gamma <= 1. - 1e-6
            _counter = 0
            _total_reward = 0.
            for i in range(len(hist)):
                _counter += 1
                _total_reward += hist[i][2]
                if hist[i][3] == True:
                    next_value[i] = (_total_reward / _counter) / (1. - gamma)
                    _counter = 0
                    _total_reward = 0.
            del _counter, _total_reward
        else:
            raise ValueError(last_value_estimator)

        # Compute advantages
        adv = np.zeros(length + 1, dtype=np.float32)
        for i in range(length-1, -1, -1):
            adv[i] = hist[i][2] + gam * next_value[i] - value[i]
            if not hist[i][3]:
                adv[i] += gam * lam * adv[i+1]
        adv = adv[:length]

        # Compute target value
        if which_target_value == "use_advantages":
            # This seems biased
            target_value = adv + value
        elif which_target_value == "use_rewards":
            # Use empirical rewards and last predicted value
            target_value = np.zeros(length, dtype=np.float32)
            for i in range(length-1, -1, -1):
                if hist[i][3] or i == (length-1):
                    # Last step of the episode (or chunk)
                    target_value[i] = hist[i][2] + gam * next_value[i]
                else:
                    # The next step is a continuation
                    target_value[i]  = hist[i][2] + gam * target_value[i+1]
        else:
            raise ValueError(which_target_value)

        # Return a joined trajectory with advantages as rewards
        traj = Trajectory(
            [hist[i][0] for i in range(length)],
            [hist[i][1] for i in range(length)],
            adv
        )

        # Train the value predictor before returning
        learn_traj = Trajectory(
            traj.o,
            target_value.reshape(-1, 1)
        )
        for _ in range(300):
            batch = learn_traj[rng.randint(len(learn_traj), size=64)]
            value_predictor.sgd_step(batch.o, batch.a)

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
