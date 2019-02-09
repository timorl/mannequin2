
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
            self.step = do_step

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
            self.reset = env.reset

            super().__init__(env)

    class DragCar(gym.Wrapper):
        def __init__(self, env, drag=1., start_amp=0.4):
            super().__init__(env)
            def do_step(action):
                self.unwrapped.state = (
                    self.unwrapped.state[0],
                    self.unwrapped.state[1] * drag
                )
                return self.env.step(action)
            def do_reset():
                self.env.reset()
                self.unwrapped.state = (
                    np.random.choice(np.array([-start_amp, start_amp]) - np.pi/6),
                    0
                )
                return np.array(self.unwrapped.state)
            self.step = do_step
            self.reset = do_reset

    class RandomLengthCar(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.len_counter = 0
            self.max_len = min(1000, np.random.geometric(.005))
            def do_step(action):
                done = False
                self.len_counter += 1
                if self.len_counter >= self.max_len or self.unwrapped.state[0] >= .5:
                    done = True
                else:
                    done = False
                obs, rew, _, info = self.env.step(action)
                return obs, rew, done, info
            def do_reset():
                self.env.reset()
                self.len_counter = 0
                self.max_len = min(1000, np.random.geometric(.005))
                return self.unwrapped.state
            self.step = do_step
            self.reset = do_reset

    class ForceMarkov(gym.Wrapper):
        def __init__(self, env, max_steps=20):
            super().__init__(env)
            original_obs_space = env.observation_space
            if not isinstance(original_obs_space, gym.spaces.Box):
                print("The observation space has to be a Box if you want to force Makrov on it!")
                raise RuntimeError
            self.observation_space = gym.spaces.Box(
                low = np.append(original_obs_space.low, 0),
                high = np.append(original_obs_space.high, max_steps),
                dtype=np.float32)
            self.current_step = 0
            def do_step(action):
                result = self.env.step(action)
                self.current_step += 1
                return np.append(result[0], self.current_step), result[1], result[2] or self.current_step >= max_steps, result[3]
            def do_reset():
                state = self.env.reset()
                state = np.append(state, 0)
                self.current_step = 0
                return state
            self.step = do_step
            self.reset = do_reset

    #Last state becomes oversampled, might be strange.
    class ConstantLengthTrajectory(gym.Wrapper):
        def __init__(self, env, max_steps):
            super().__init__(env)
            self.current_step = 0
            self.last_state = None
            def do_step(action):
                self.current_step += 1
                if self.last_state is None:
                    result = self.env.step(action)
                    if result[2]:
                        self.last_state = result[0]
                    return result[0], result[1], self.current_step >= max_steps, result[3]
                return self.last_state, 0., self.current_step >= max_steps, None
            def do_reset():
                self.current_step = 0
                self.last_state = None
                return self.env.reset()
            self.step = do_step
            self.reset = do_reset

    class ModifyReward(gym.Wrapper):
        def __init__(self, env, bias=0.):
            super().__init__(env)
            def do_step(action):
                result = self.env.step(action)
                return result[0], result[1]+bias, result[2], result[3]
            def do_reset():
                return self.env.reset()
            self.step = do_step
            self.reset = do_reset

    class Gridworld(gym.Env):
        def __init__(self, width=20, scale=20):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(low=-width, high=width, shape=(1,), dtype=np.float32)
            self.width = width
            def reward(s):
                return 1 - 1 /(scale*(1 + s**2))
            def reset():
                self.s = 0
                self.counter = 0
                return (self.s,)
            def step(action):
                self.counter += 1
                self.s += (2*action - 1)
                return (self.s,), reward(self.s), self.counter >= self.width, None
            self.reset = reset
            self.step = step

    configs = {
        "gridworld20": lambda: TrackedEnv(Gridworld(scale=8), max_steps=80000),
        "gridworld20Markov": lambda: TrackedEnv(ForceMarkov(Gridworld(scale=8)), max_steps=80000),
        "gridworld200": lambda: TrackedEnv(Gridworld(width=200,scale=8), max_steps=80000),
        "gridworld200Markov": lambda: TrackedEnv(ForceMarkov(Gridworld(width=200, scale=8), max_steps=200), max_steps=80000),

        "car": lambda: TrackedEnv(gym.make("MountainCar-v0"), max_steps=80000, max_rew=200),
        "carMarkov": lambda: TrackedEnv(ForceMarkov(gym.make("MountainCar-v0"), max_steps=200), max_steps=80000, max_rew=200),
        "carConstLen1": lambda: TrackedEnv(ModifyReward(ConstantLengthTrajectory(gym.make("MountainCar-v0"), max_steps=200), bias=1), max_steps=80000, max_rew=200),
        "carConstLen1Markov": lambda: TrackedEnv(
            ForceMarkov(
                ModifyReward(ConstantLengthTrajectory(gym.make("MountainCar-v0"), max_steps=200), bias=1)
                ,max_steps=200)
            , max_steps=80000, max_rew=200),
        "carConstLen2": lambda: TrackedEnv(ModifyReward(ConstantLengthTrajectory(gym.make("MountainCar-v0"), max_steps=200), bias=2), max_steps=80000, max_rew=200),
        "carConstLen2Markov": lambda: TrackedEnv(
                ForceMarkov(
                    ModifyReward(ConstantLengthTrajectory(gym.make("MountainCar-v0"), max_steps=200), bias=2)
                    ,max_steps=200)
                , max_steps=80000, max_rew=200),

        "carBottom": lambda: TrackedEnv(DragCar(gym.make("MountainCar-v0"), drag=1., start_amp=0.), max_steps=80000, max_rew=200),
        "carBottomMarkov": lambda: TrackedEnv(ForceMarkov(DragCar(gym.make("MountainCar-v0"), drag=1., start_amp=0.), max_steps=200), max_steps=80000, max_rew=200),
        "carBottomConstLen1": lambda: TrackedEnv(ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=1., start_amp=0.), max_steps=200), bias=1), max_steps=80000, max_rew=200),
        "carBottomConstLen1Markov": lambda: TrackedEnv(
                ForceMarkov(
                    ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=1., start_amp=0.), max_steps=200), bias=1)
                    ,max_steps=200)
                , max_steps=80000, max_rew=200),
        "carBottomConstLen2": lambda: TrackedEnv(ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=1., start_amp=0.), max_steps=200), bias=2), max_steps=80000, max_rew=200),
        "carBottomConstLen2Markov": lambda: TrackedEnv(
                ForceMarkov(
                    ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=1., start_amp=0.), max_steps=200), bias=2)
                    ,max_steps=200)
                , max_steps=80000, max_rew=200),

        "randomlengthcar": lambda: TrackedEnv(RandomLengthCar(gym.make("MountainCar-v0")), max_steps=80000, max_rew=200),
        "randomlengthdragcar": lambda: TrackedEnv(RandomLengthCar(DragCar(gym.make("MountainCar-v0"), drag=1.)), max_steps=80000, max_rew=200),

        "dragcar99": lambda: TrackedEnv(DragCar(gym.make("MountainCar-v0"), drag=.99), max_steps=80000, max_rew=200),
        "dragcar99Markov": lambda: TrackedEnv(ForceMarkov(DragCar(gym.make("MountainCar-v0"), drag=.99), max_steps=200), max_steps=80000, max_rew=200),
        "dragcar99ConstLen1": lambda: TrackedEnv(ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=.99), max_steps=200), bias=1), max_steps=80000, max_rew=200),
        "dragcar99ConstLen1Markov": lambda: TrackedEnv(
                ForceMarkov(
                    ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=.99), max_steps=200), bias=1)
                    ,max_steps=200)
                , max_steps=80000, max_rew=200),
        "dragcar99ConstLen2": lambda: TrackedEnv(ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=.99), max_steps=200), bias=2), max_steps=80000, max_rew=200),
        "dragcar99ConstLen2Markov": lambda: TrackedEnv(
                ForceMarkov(
                    ModifyReward(ConstantLengthTrajectory(DragCar(gym.make("MountainCar-v0"), drag=.99), max_steps=200), bias=2)
                    ,max_steps=200)
                , max_steps=80000, max_rew=200),

    }

    if "ENV" in os.environ:
        return configs[os.environ["ENV"]]

    return configs["cartpole"]

build_env = build_env()
