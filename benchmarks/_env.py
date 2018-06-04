
import os
import sys
import gym
import gym.spaces
import numpy as np

sys.path.append("..")
from mannequin import bar
from mannequin.gym import PrintRewards, ClippedActions

get_progress = None

def build_env():
    finished_steps = 1
    global_max_steps = 1
    video_wanted = False

    global get_progress
    def get_progress(*, divide=True):
        if divide:
            return finished_steps / global_max_steps
        else:
            return finished_steps, global_max_steps

    if "N_VIDEOS" in os.environ:
        n_videos = int(os.environ["N_VIDEOS"])
    else:
        n_videos = 5

    class TrackedEnv(gym.Wrapper):
        def __init__(self, env, *,
                max_steps=400000,
                max_rew=500):
            nonlocal global_max_steps
            global_max_steps = max_steps

            def do_step(action):
                nonlocal finished_steps, video_wanted
                finished_steps += 1
                video_every = max(1, max_steps // n_videos)
                if finished_steps % video_every == video_every // 2:
                    video_wanted = True
                return self.env.step(action)
            self._step = do_step

            def pop_wanted(*_):
                nonlocal video_wanted
                ret, video_wanted = video_wanted, False
                return ret

            if isinstance(env.action_space, gym.spaces.Box):
                env = ClippedActions(env)

            n_lines = 0
            def print_line(s, r):
                nonlocal n_lines
                if n_lines < 100:
                    n_lines += 1
                    if "LOG_FILE" in os.environ:
                        with open(os.environ["LOG_FILE"], "a") as f:
                            f.write("%d %.2f\n" % (s, r))
                            f.flush()
                    else:
                        print("%8d steps:" % s, bar(r, max_rew),
                            flush=True)

            env = PrintRewards(env, print=print_line,
                every=max_steps // 100)

            if "VIDEO_DIR" in os.environ:
                env = gym.wrappers.Monitor(
                    env,
                    os.environ["VIDEO_DIR"],
                    video_callable=pop_wanted,
                    force=True
                )

            super().__init__(env)

    class DragCar(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            def do_step(action):
                self.unwrapped.state = (
                    self.unwrapped.state[0] * 0.99,
                    self.unwrapped.state[1]
                )
                return self.env._step(action)
            def do_reset():
                self.env._reset()
                self.unwrapped.state = (
                    np.random.choice(np.array([-0.4, 0.4]) - np.pi/6),
                    0
                )
                return np.array(self.unwrapped.state)
            self._step = do_step
            self._reset = do_reset

    class AxisOfEval(gym.Env):
        def __init__(self, width=20):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(low=-width, high=width, shape=(1,))
            self.width = width
        def _reset(self):
            self.s = 0
            self.counter = 0
            return (self.s,)
        def _step(self, action):
            def _reward(s):
                return 1 - 1 /(20*(1 + s**2))
            self.counter += 1
            self.s += (2*action - 1)
            return (self.s,), _reward(self.s), self.counter > self.width, None

    class AxisOfEvalMarkov(gym.Env):
        def __init__(self, width=20):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(low=np.array([-width,0]), high=np.array([width,width]))
            self.width = width
        def _reset(self):
            self.s = 0
            self.counter = 0
            return (self.s, self.counter)
        def _step(self, action):
            def _reward(s):
                return 1 - 1 /(20*(1 + s**2))
            self.counter += 1
            self.s += (2*action - 1)
            return (self.s, self.counter), _reward(self.s), self.counter > self.width, None

    configs = {
        "cartpole": lambda: TrackedEnv(gym.make("CartPole-v1"), max_steps=40000),
        "acrobot": lambda: TrackedEnv(gym.make("Acrobot-v1"), max_steps=80000),
        "car": lambda: TrackedEnv(gym.make("MountainCar-v0"), max_steps=80000, max_rew=200),
        "dragcar": lambda: TrackedEnv(DragCar(gym.make("MountainCar-v0")), max_steps=80000, max_rew=200),
        "walker": lambda: TrackedEnv(gym.make("BipedalWalker-v2"), max_rew=300),
        "lander": lambda: TrackedEnv(gym.make("LunarLanderContinuous-v2")),
        "axisOfEval": lambda: TrackedEnv(AxisOfEval(), max_steps=80000),
        "axisOfEval200": lambda: TrackedEnv(AxisOfEval(width=200), max_steps=80000),
        "axisOfEvalMarkov": lambda: TrackedEnv(AxisOfEvalMarkov(), max_steps=80000),
        "axisOfEvalMarkov200": lambda: TrackedEnv(AxisOfEvalMarkov(width=200), max_steps=80000),
    }

    if "ENV" in os.environ:
        return configs[os.environ["ENV"]]

    return configs["cartpole"]

build_env = build_env()
