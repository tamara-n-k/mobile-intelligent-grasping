from gymnasium.utils.env_checker import check_env
from youbot_pnp.rl.env import YouBotPnPEnv

env = YouBotPnPEnv(n_cubes=1)
check_env(env.unwrapped)

from collections import Counter

tried, succeeded = Counter(), Counter()
returns, solved = [], 0
for ep in range(100):
    obs, _ = env.reset(seed=ep)
    done, ep_return = False, 0.0
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        ep_return += r
        tried[info["primitive"]] += 1
        succeeded[info["primitive"]] += info["success"]
        done = term or trunc
    returns.append(ep_return)
    solved += env._all_placed()

for name in tried:
    print(f"{name}: {succeeded[name]}/{tried[name]} succeeded")
print(f"solved {solved}/100, mean return {sum(returns) / 100:.2f}")
