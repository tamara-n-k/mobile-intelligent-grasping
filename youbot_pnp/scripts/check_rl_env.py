from gymnasium.utils.env_checker import check_env
from youbot_pnp.rl.env import YouBotPnPEnv

env = YouBotPnPEnv(n_cubes=1)
check_env(env.unwrapped)

for ep in range(100):
    obs, _ = env.reset(seed=ep)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
print("check_env and 100 random episodes passed")

