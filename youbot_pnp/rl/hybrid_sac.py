"""MAPLE-style hybrid SAC (Nasiriany et al. 2021, Sec. III-C), ported from maple/'s PAMDPPolicy and SACHybridTrainer.

The policy is split in two:
  task policy       pi_s(a | s):    softmax over the primitives (which primitive)
  parameter policy  pi_p(x | s, a): one tanh-Gaussian head per primitive (with which parameters)

Each part has its own entropy temperature (alpha_s, alpha_p) with its own target. With only 3 primitives the
losses sum over all of them, weighted by pi_s, instead of sampling the discrete choice (MAPLE's one_hot_s mode).

The action sent to the env is [one_hot(a), x], so YouBotPnPEnv.step's argmax over action[:3] picks a.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PRIM, N_PARAM = 3, 4                 # grasp/reach/open; x, y, z, yaw
LOG_STD_MIN, LOG_STD_MAX = -20, 2
LOGITS_MAX = 10                        # as in MAPLE's CategoricalPolicy


def mlp(inp, out, hidden=(256, 256)):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, out))


class HybridPolicy(nn.Module):
    """task policy: categorical over primitives; parameter policy: one tanh-Gaussian head per primitive"""

    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        self.task = mlp(obs_dim, N_PRIM, hidden)
        self.params = nn.ModuleList(mlp(obs_dim, 2 * N_PARAM, hidden) for _ in range(N_PRIM))

    def forward(self, obs):
        logits = self.task(obs).clamp(-LOGITS_MAX, LOGITS_MAX)
        out = torch.stack([p(obs) for p in self.params], dim=1)          # (B, N_PRIM, 2*N_PARAM)
        mean, log_std = out.split(N_PARAM, dim=-1)
        return logits, mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    @staticmethod
    def sample_params(mean, std):
        """reparameterized tanh-Gaussian sample and its log-prob, for all primitives at once"""
        u = mean + std * torch.randn_like(mean)
        x = torch.tanh(u)
        log_p = torch.distributions.Normal(mean, std).log_prob(u) - torch.log(1 - x.pow(2) + 1e-6)
        return x, log_p.sum(-1)                                            # (B, N_PRIM, N_PARAM), (B, N_PRIM)


class QNet(nn.Module):
    """Q(s, [one_hot(a), x])"""

    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + N_PRIM + N_PARAM, 1, hidden)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


def all_actions(x):
    """[one_hot(a), x_a] for every primitive a: (B, N_PRIM, N_PARAM) -> (B, N_PRIM, N_PRIM + N_PARAM)"""
    eye = torch.eye(N_PRIM, device=x.device).expand(x.shape[0], N_PRIM, N_PRIM)
    return torch.cat([eye, x], dim=-1)


class ReplayBuffer:
    def __init__(self, obs_dim, size):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, N_PRIM + N_PARAM), np.float32)
        self.rew = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)  # 1 only for a true terminal state, not for a time-limit truncation
        self.size, self.n, self.i = size, 0, 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.i], self.act[self.i], self.rew[self.i] = obs, act, rew
        self.next_obs[self.i], self.done[self.i] = next_obs, done
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n, batch_size)
        return {k: torch.as_tensor(getattr(self, k)[idx]) for k in ("obs", "act", "rew", "next_obs", "done")}


class HybridSAC:
    def __init__(self, obs_dim, lr=3e-4, gamma=0.99, tau=0.005, batch_size=256, buffer_size=300_000,
                 hidden=(256, 256), high_ent_steps=50_000, one_hot_factor=0.5, reward_scale=1.0, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.obs_dim, self.gamma, self.tau, self.batch_size = obs_dim, gamma, tau, batch_size
        self.reward_scale = reward_scale
        self.hidden = hidden

        self.policy = HybridPolicy(obs_dim, hidden)
        self.q1, self.q2 = QNet(obs_dim, hidden), QNet(obs_dim, hidden)
        self.q1_targ, self.q2_targ = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for p in [*self.q1_targ.parameters(), *self.q2_targ.parameters()]:
            p.requires_grad_(False)
        self.log_alpha_s = torch.zeros(1, requires_grad=True)
        self.log_alpha_p = torch.zeros(1, requires_grad=True)

        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha_s, self.log_alpha_p], lr=lr)

        # entropy targets as in MAPLE's robosuite_launcher (high_init_ent): almost uniform primitive choice at first
        self.high_ent_steps = high_ent_steps
        self.target_s_init, self.target_p_init = 0.97 * np.log(N_PRIM), 0.0
        self.target_s_later, self.target_p_later = one_hot_factor * np.log(N_PRIM), -float(N_PARAM)

        self.buffer = ReplayBuffer(obs_dim, buffer_size)

    def entropy_targets(self, step):
        if step < self.high_ent_steps:
            return self.target_s_init, self.target_p_init
        return self.target_s_later, self.target_p_later

    @torch.no_grad()
    def predict(self, obs, deterministic=False):
        """obs (obs_dim,) or (B, obs_dim) -> env action(s) [one_hot(a), x]; same call signature as SB3"""
        single = np.ndim(obs) == 1
        obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
        logits, mean, std = self.policy(obs_t)
        if deterministic:
            a = logits.argmax(-1)
            x = torch.tanh(mean)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
            x, _ = self.policy.sample_params(mean, std)
        x_a = x[torch.arange(len(a)), a]
        action = torch.cat([F.one_hot(a, N_PRIM).float(), x_a], dim=-1).numpy()
        return (action[0] if single else action), None

    @staticmethod
    def random_action(n):
        a = np.random.randint(0, N_PRIM, n)
        return np.concatenate([np.eye(N_PRIM)[a], np.random.uniform(-1, 1, (n, N_PARAM))], axis=1).astype(np.float32)

    def _expected(self, obs, q_nets):
        """everything summed over primitives with pi_s: values (B, N_PRIM) and policy quantities"""
        logits, mean, std = self.policy(obs)
        probs, log_probs = F.softmax(logits, -1), F.log_softmax(logits, -1)
        x, log_p_x = self.policy.sample_params(mean, std)
        acts = all_actions(x)
        obs_rep = obs.unsqueeze(1).expand(-1, N_PRIM, -1)
        q = torch.min(q_nets[0](obs_rep, acts), q_nets[1](obs_rep, acts))   # (B, N_PRIM)
        return probs, log_probs, log_p_x, q

    def update(self, step):
        b = self.buffer.sample(self.batch_size)
        alpha_s, alpha_p = self.log_alpha_s.exp().detach(), self.log_alpha_p.exp().detach()

        # critic: soft Bellman target, expectation over the next primitive taken exactly
        with torch.no_grad():
            p2, lp2, lpx2, q2_next = self._expected(b["next_obs"], (self.q1_targ, self.q2_targ))
            v_next = (p2 * (q2_next - alpha_s * lp2 - alpha_p * lpx2)).sum(-1)
            y = self.reward_scale * b["rew"] + self.gamma * (1 - b["done"]) * v_next
        q1_pred, q2_pred = self.q1(b["obs"], b["act"]), self.q2(b["obs"], b["act"])
        q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # actor: both policy parts, weighted by pi_s (the gradient to the task policy flows through probs)
        probs, log_probs, log_p_x, q = self._expected(b["obs"], (self.q1, self.q2))
        pi_loss = (probs * (alpha_s * log_probs + alpha_p * log_p_x - q)).sum(-1).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # temperatures: raise alpha if the entropy is below its target, lower it otherwise
        ent_s = -(probs * log_probs).sum(-1).mean().detach()
        ent_p = -(probs * log_p_x).sum(-1).mean().detach()
        target_s, target_p = self.entropy_targets(step)
        alpha_loss = self.log_alpha_s * (ent_s - target_s) + self.log_alpha_p * (ent_p - target_p)
        self.alpha_opt.zero_grad()
        alpha_loss.sum().backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, targ in ((self.q1, self.q1_targ), (self.q2, self.q2_targ)):
                for p, pt in zip(net.parameters(), targ.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)

        return {"train/q_loss": q_loss.item(), "train/pi_loss": pi_loss.item(),
                "train/alpha_s": alpha_s.item(), "train/alpha_p": alpha_p.item(),
                "train/entropy_s": ent_s.item(), "train/entropy_p": ent_p.item(),
                "train/target_entropy_s": target_s, "train/q_mean": q1_pred.mean().item(),
                **{f"train/pi_{n}": probs[:, i].mean().item() for i, n in enumerate(("grasp", "reach", "open"))}}

    def save(self, path):
        torch.save({"obs_dim": self.obs_dim, "hidden": self.hidden,
                    "policy": self.policy.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                    "log_alpha_s": self.log_alpha_s.detach(), "log_alpha_p": self.log_alpha_p.detach()}, path)

    @classmethod
    def load(cls, path):
        """for evaluation: restores the networks and temperatures (not the replay buffer or the optimizers)"""
        ckpt = torch.load(path, weights_only=True)
        agent = cls(ckpt["obs_dim"], hidden=tuple(ckpt["hidden"]), buffer_size=1)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        agent.log_alpha_s.data.copy_(ckpt["log_alpha_s"])
        agent.log_alpha_p.data.copy_(ckpt["log_alpha_p"])
        return agent
