import sys
sys.path.insert(0, "tournament/tournament")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "policy",
    "tournament/tournament/groups/Martin Jerez/policy.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

agent = mod.QLearningAgent(n_episodes=5000)
agent.mount()
print("weights =", repr(agent.weights.tolist()))
