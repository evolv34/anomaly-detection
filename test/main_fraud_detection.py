from fraud_detect.envs import FraudDetectEnv


def get_action(observations, state):
    return 1


def run_one_step(observation, state, local_env):
    action = get_action(observation, state)
    obs, reward, done, info = local_env.step(action)

    return obs, reward, done, info


def run_one_episode(local_env, obs_iterations):
    obs = local_env.reset()
    for sample in range(obs_iterations):
        obs, reward, done, info = run_one_step(obs, None, env)
