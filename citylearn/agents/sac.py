import logging
from collections import deque
from statistics import NormalDist
from typing import Any, List, Union
import numpy as np
import numpy.typing as npt
import zipfile
import tempfile
import json
from pathlib import Path
import re
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except (ModuleNotFoundError, ImportError) as e:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

from citylearn.agents.rbc import RBC
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from citylearn.preprocessing import Encoder, RemoveFeature
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork
from typing import Optional

class SAC(RLC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        r"""Custom soft actor-critic algorithm.

        Parameters
        ----------
        env: CityLearnEnv
            CityLearn environment.
        
        Other Parameters
        ----------------
        **kwargs : Any
            Other keyword arguments used to initialize super class.
        """

        self.automatic_entropy_tuning = bool(kwargs.pop("automatic_entropy_tuning", False))
        self.entropy_lr = kwargs.pop("entropy_lr", None)
        self.target_entropy_override = kwargs.pop("target_entropy", None)
        self.n_step_horizon = int(kwargs.pop("n_step_horizon", 1))
        self.robust_normalization = bool(kwargs.pop("robust_normalization", False))
        self.robust_normalization_low = float(kwargs.pop("robust_normalization_low", 5.0))
        self.robust_normalization_high = float(kwargs.pop("robust_normalization_high", 95.0))
        self.normalized_observation_clip = kwargs.pop("normalized_observation_clip", None)
        self.normalized_reward_clip = kwargs.pop("normalized_reward_clip", None)
        self.device_override = kwargs.pop("device", None)

        super().__init__(env, **kwargs)

        # internally defined
        self.normalized = [False for _ in self.action_space]
        self.soft_q_criterion = nn.SmoothL1Loss()
        if self.device_override is None or str(self.device_override).lower() == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device_override)
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("SAC requested CUDA, but torch.cuda.is_available() is False.")
        self.nstep_discount = self.discount ** self.n_step_horizon
        self._nstep_buffers = [deque(maxlen=self.n_step_horizon) for _ in self.action_space]
        self.replay_buffer = [ReplayBuffer(int(self.replay_buffer_capacity)) for _ in self.action_space]
        self.soft_q_net1 = [None for _ in self.action_space]
        self.soft_q_net2 = [None for _ in self.action_space]
        self.target_soft_q_net1 = [None for _ in self.action_space]
        self.target_soft_q_net2 = [None for _ in self.action_space]
        self.policy_net = [None for _ in self.action_space]
        self.soft_q_optimizer1 = [None for _ in self.action_space]
        self.soft_q_optimizer2 = [None for _ in self.action_space]
        self.policy_optimizer = [None for _ in self.action_space]
        self.target_entropy = [None for _ in self.action_space]
        self.log_alpha = [None for _ in self.action_space]
        self.alpha_optimizer = [None for _ in self.action_space]
        self.norm_mean = [None for _ in self.action_space]
        self.norm_std = [None for _ in self.action_space]
        self.r_norm_mean = [None for _ in self.action_space]
        self.r_norm_std = [None for _ in self.action_space]
        self.last_update_losses = []
        self.set_networks()

    def _robust_center_scale(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = float(np.clip(self.robust_normalization_low, 0.0, 49.0))
        high = float(np.clip(self.robust_normalization_high, low + 1.0, 100.0))
        center = np.nanmedian(values, axis=0)
        q_low = np.nanpercentile(values, low, axis=0)
        q_high = np.nanpercentile(values, high, axis=0)
        z_low = NormalDist().inv_cdf(low / 100.0)
        z_high = NormalDist().inv_cdf(high / 100.0)
        denom = max(float(z_high - z_low), 1e-6)
        scale = (q_high - q_low) / denom
        fallback = np.nanstd(values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return center, scale + 1e-5

    def _fit_normalizers(self, index: int) -> None:
        X = np.array([j[0] for j in self.replay_buffer[index].buffer], dtype=float)
        R = np.array([j[2] for j in self.replay_buffer[index].buffer], dtype=float)
        if self.robust_normalization:
            self.norm_mean[index], self.norm_std[index] = self._robust_center_scale(X)
            r_center, r_scale = self._robust_center_scale(R.reshape(-1, 1))
            self.r_norm_mean[index] = float(r_center[0])
            self.r_norm_std[index] = float(r_scale[0]) / self.reward_scaling + 1e-5
        else:
            self.norm_mean[index] = np.nanmean(X, axis=0)
            self.norm_std[index] = np.nanstd(X, axis=0) + 1e-5
            self.r_norm_mean[index] = np.nanmean(R, dtype=float)
            self.r_norm_std[index] = np.nanstd(R, dtype=float) / self.reward_scaling + 1e-5

    def standardize_replay_buffer(self, index: int) -> None:
        self._fit_normalizers(index)
        self.replay_buffer[index].buffer = [(
            np.hstack(self.get_normalized_observations(index, o).reshape(1, -1)[0]),
            a,
            self.get_normalized_reward(index, r),
            np.hstack(self.get_normalized_observations(index, n).reshape(1, -1)[0]),
            d
        ) for o, a, r, n, d in self.replay_buffer[index].buffer]
        self.normalized[index] = True

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float], next_observations: List[List[float]], terminated: bool, truncated: bool):
        r"""Update replay buffer.

        Parameters
        ----------
        observations : List[List[float]]
            Previous time step observations.
        actions : List[List[float]]
            Previous time step actions.
        reward : List[float]
            Current time step reward.
        next_observations : List[List[float]]
            Current time step observations.
        terminated : bool
            Indication that episode has ended.
        truncated : bool
            If episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
        """

        self.last_update_losses = []

        # Run once the regression model has been fitted
        # Normalize all the observations using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes observations that are not necessary (solar irradiance if there are no solar PV panels).

        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            o_enc = self.get_encoded_observations(i, o)
            n_enc = self.get_encoded_observations(i, n)

            if self.n_step_horizon > 1:
                # Accumulate encoded (pre-normalization) transitions; raw reward
                self._nstep_buffers[i].append((o_enc, a, float(r), n_enc, bool(terminated)))

                if len(self._nstep_buffers[i]) == self.n_step_horizon:
                    window = list(self._nstep_buffers[i])
                    o0, a0 = window[0][0], window[0][1]
                    R_raw = sum(self.discount ** k * window[k][2] for k in range(self.n_step_horizon))
                    n_final = window[-1][3]
                    any_done = any(step[4] for step in window)

                    if self.normalized[i]:
                        o0 = self.get_normalized_observations(i, o0)
                        n_final = self.get_normalized_observations(i, n_final)
                        R_raw = self.get_normalized_reward(i, R_raw)
                    self.replay_buffer[i].push(o0, a0, R_raw, n_final, any_done)

                if terminated:
                    self._nstep_buffers[i].clear()
            else:
                o_enc_push, n_enc_push, r_push = o_enc, n_enc, r
                if self.normalized[i]:
                    o_enc_push = self.get_normalized_observations(i, o_enc_push)
                    n_enc_push = self.get_normalized_observations(i, n_enc_push)
                    r_push = self.get_normalized_reward(i, r_push)
                self.replay_buffer[i].push(o_enc_push, a, r_push, n_enc_push, terminated)

            if self.time_step >= self.standardize_start_time_step and self.batch_size <= len(self.replay_buffer[i]):
                if not self.normalized[i]:
                    self.standardize_replay_buffer(i)
                
                else:
                    pass

                for _ in range(self.update_per_time_step):
                    o, a, r, n, d = self.replay_buffer[i].sample(self.batch_size)
                    tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                    o = tensor(o).to(self.device)
                    n = tensor(n).to(self.device)
                    a = tensor(a).to(self.device)
                    r = tensor(r).unsqueeze(1).to(self.device)
                    d = tensor(d).unsqueeze(1).to(self.device)

                    with torch.no_grad():
                        # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) observation and its associated log probability of occurrence.
                        new_next_actions, new_log_pi, _ = self.policy_net[i].sample(n)

                        # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
                        alpha = self.log_alpha[i].exp() if self.automatic_entropy_tuning else self.alpha
                        target_q_values = torch.min(
                            self.target_soft_q_net1[i](n, new_next_actions),
                            self.target_soft_q_net2[i](n, new_next_actions),
                        ) - alpha*new_log_pi
                        q_target = r + (1 - d)*self.nstep_discount*target_q_values

                    # Update Soft Q-Networks
                    q1_pred = self.soft_q_net1[i](o, a)
                    q2_pred = self.soft_q_net2[i](o, a)
                    q1_loss = self.soft_q_criterion(q1_pred, q_target)
                    q2_loss = self.soft_q_criterion(q2_pred, q_target)
                    self.soft_q_optimizer1[i].zero_grad()
                    q1_loss.backward()
                    self.soft_q_optimizer1[i].step()
                    self.soft_q_optimizer2[i].zero_grad()
                    q2_loss.backward()
                    self.soft_q_optimizer2[i].step()

                    # Update Policy
                    new_actions, log_pi, _ = self.policy_net[i].sample(o)
                    q_new_actions = torch.min(
                        self.soft_q_net1[i](o, new_actions),
                        self.soft_q_net2[i](o, new_actions)
                    )
                    alpha = self.log_alpha[i].exp().detach() if self.automatic_entropy_tuning else self.alpha
                    policy_loss = (alpha*log_pi - q_new_actions).mean()
                    self.policy_optimizer[i].zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer[i].step()

                    alpha_loss = None
                    alpha_value = float(alpha.detach().cpu().item()) if self.automatic_entropy_tuning else float(self.alpha)
                    if self.automatic_entropy_tuning:
                        alpha_loss = -(
                            self.log_alpha[i] * (log_pi + self.target_entropy[i]).detach()
                        ).mean()
                        self.alpha_optimizer[i].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[i].step()
                        alpha_value = float(self.log_alpha[i].exp().detach().cpu().item())
                        alpha_values = [
                            float(v.exp().detach().cpu().item())
                            for v in self.log_alpha
                            if v is not None
                        ]
                        if alpha_values:
                            self.alpha = float(np.mean(alpha_values))

                    self.last_update_losses.append({
                        "agent_index": int(i),
                        "soft_q1_loss": float(q1_loss.detach().cpu().item()),
                        "soft_q2_loss": float(q2_loss.detach().cpu().item()),
                        "policy_loss": float(policy_loss.detach().cpu().item()),
                        "log_pi_mean": float(log_pi.detach().mean().cpu().item()),
                        "q_new_actions_mean": float(q_new_actions.detach().mean().cpu().item()),
                        "q_target_mean": float(q_target.detach().mean().cpu().item()),
                        "reward_batch_mean": float(r.detach().mean().cpu().item()),
                        "alpha": alpha_value,
                        "alpha_loss": (
                            np.nan
                            if alpha_loss is None
                            else float(alpha_loss.detach().cpu().item())
                        ),
                        "target_entropy": (
                            np.nan
                            if self.target_entropy[i] is None
                            else float(self.target_entropy[i])
                        ),
                    })

                    # Soft Updates
                    for target_param, param in zip(self.target_soft_q_net1[i].parameters(), self.soft_q_net1[i].parameters()):
                        target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data*self.tau)

                    for target_param, param in zip(self.target_soft_q_net2[i].parameters(), self.soft_q_net2[i].parameters()):
                        target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data*self.tau)

            else:
                pass

    def predict(self, observations: List[List[float]], deterministic: bool = None):
        r"""Provide actions for current time step.

        Will return randomly sampled actions from `action_space` if :attr:`end_exploration_time_step` <= :attr:`time_step` 
        else will use policy to sample actions.

        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[float]
            Action values
        """

        deterministic = False if deterministic is None else deterministic

        if self.time_step > self.end_exploration_time_step or deterministic:
            actions = self.get_post_exploration_prediction(observations, deterministic)
            
        else:
            actions = self.get_exploration_prediction(observations)

        self.actions = actions
        self.next_time_step()
        return actions

    def get_post_exploration_prediction(self, observations: List[List[float]], deterministic: bool) -> List[List[float]]:
        """Action sampling using policy, post-exploration time step"""

        actions = []

        for i, o in enumerate(observations):
            o = self.get_encoded_observations(i, o)
            o = self.get_normalized_observations(i, o)
            o = torch.FloatTensor(o).unsqueeze(0).to(self.device)
            result = self.policy_net[i].sample(o)
            a = result[2] if deterministic else result[0]
            actions.append(a.detach().cpu().numpy()[0])

        return actions
            
    def get_exploration_prediction(self, observations: List[List[float]]) -> List[List[float]]:
        """Return randomly sampled actions from `action_space` multiplied by :attr:`action_scaling_coefficient`."""

        # random actions
        return [list(self.action_scaling_coefficient*s.sample()) for s in self.action_space]

    def get_normalized_reward(self, index: int, reward: float) -> float:
        value = (reward - self.r_norm_mean[index])/self.r_norm_std[index]
        if self.normalized_reward_clip is not None:
            clip = abs(float(self.normalized_reward_clip))
            value = float(np.clip(value, -clip, clip))
        return value

    def get_normalized_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        try:
            value = (np.array(observations, dtype = float) - self.norm_mean[index])/self.norm_std[index]
            if self.normalized_observation_clip is not None:
                clip = abs(float(self.normalized_observation_clip))
                value = np.clip(value, -clip, clip)
            return value
        except:
            # self.time_step >= self.standardize_start_time_step and self.batch_size <= len(self.replay_buffer[i])
            logging.debug('obs:',observations)
            logging.debug('mean:',self.norm_mean[index])
            logging.debug('std:',self.norm_std[index])
            logging.debug(self.time_step, self.standardize_start_time_step, self.batch_size, len(self.replay_buffer[0]))
            assert False

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(self.encoders[index]*np.array(observations, dtype=float)) if j != None], dtype = float)

    def set_networks(self, internal_observation_count: int = None):
        internal_observation_count = 0 if internal_observation_count is None else internal_observation_count

        for i in range(len(self.action_dimension)):
            observation_dimension = self.observation_dimension[i] + internal_observation_count
            # init networks
            self.soft_q_net1[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)
            self.soft_q_net2[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)
            self.target_soft_q_net1[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)
            self.target_soft_q_net2[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)

            for target_param, param in zip(self.target_soft_q_net1[i].parameters(), self.soft_q_net1[i].parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_soft_q_net2[i].parameters(), self.soft_q_net2[i].parameters()):
                target_param.data.copy_(param.data)

            # Policy
            self.policy_net[i] = PolicyNetwork(observation_dimension, self.action_dimension[i], self.action_space[i], self.action_scaling_coefficient, self.hidden_dimension).to(self.device)
            self.soft_q_optimizer1[i] = optim.Adam(self.soft_q_net1[i].parameters(), lr=self.lr)
            self.soft_q_optimizer2[i] = optim.Adam(self.soft_q_net2[i].parameters(), lr=self.lr)
            self.policy_optimizer[i] = optim.Adam(self.policy_net[i].parameters(), lr=self.lr)
            self.target_entropy[i] = (
                -float(self.action_dimension[i])
                if self.target_entropy_override is None
                else float(self.target_entropy_override)
            )
            if self.automatic_entropy_tuning:
                initial_alpha = max(float(self.alpha), 1e-8)
                self.log_alpha[i] = torch.tensor(
                    np.log(initial_alpha),
                    dtype=torch.float32,
                    requires_grad=True,
                    device=self.device,
                )
                self.alpha_optimizer[i] = optim.Adam(
                    [self.log_alpha[i]],
                    lr=float(self.lr if self.entropy_lr is None else self.entropy_lr),
                )

    def set_encoders(self) -> List[List[Encoder]]:
        encoders = super().set_encoders()

        for i, o in enumerate(self.observation_names):
            for j, n in enumerate(o):
                if n == 'net_electricity_consumption':
                    encoders[i][j] = RemoveFeature()
            
                else:
                    pass

        return encoders

    def save_models(self, zip_path: str = 'sac.zip', dtype: Optional[str] = None):
        """
        Crea SOLO uno .zip con:
        - mean_std/config.json (statistiche normalizzazione)
        - pesi .pt di tutte le reti (soft_q_net1/2, target_*, policy)
        """

        def _to_torch_dtype(d):
            if d is None: return None
            d = d.lower()
            if d in ('fp16', 'float16'):  return torch.float16
            if d in ('bf16', 'bfloat16'): return torch.bfloat16
            if d in ('fp32', 'float32'):  return torch.float32
            raise ValueError("dtype non supportato: %s" % d)

        tgt_dtype = _to_torch_dtype(dtype)
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        print('.... saving models to ZIP, mean and std ....')

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # mean/std
            ms_dir = out_dir / 'mean_std'
            ms_dir.mkdir(parents=True, exist_ok=True)
            params = {
                "mean": [np.asarray(arr).tolist() for arr in getattr(self, "norm_mean", [])],
                "std":  [np.asarray(arr).tolist() for arr in getattr(self, "norm_std", [])],
            }
            (ms_dir / 'config.json').write_text(json.dumps(params, indent=2))

            # reti
            if hasattr(self, "policy_net"):
                n = len(self.policy_net)
            elif hasattr(self, "action_dimension"):
                n = len(self.action_dimension)
            else:
                raise RuntimeError("Impossibile determinare il numero di azioni/reti da salvare.")

            def _cast(sd):
                if tgt_dtype is None: return sd
                return {k: (v.to(tgt_dtype) if torch.is_floating_point(v) else v) for k, v in sd.items()}

            for i in range(n):
                torch.save(_cast(self.soft_q_net1[i].state_dict()),        out_dir / f'soft_q_net1_{i}.pt')
                torch.save(_cast(self.soft_q_net2[i].state_dict()),        out_dir / f'soft_q_net2_{i}.pt')
                torch.save(_cast(self.target_soft_q_net1[i].state_dict()), out_dir / f'target_soft_q_net1_{i}.pt')
                torch.save(_cast(self.target_soft_q_net2[i].state_dict()), out_dir / f'target_soft_q_net2_{i}.pt')
                torch.save(_cast(self.policy_net[i].state_dict()),         out_dir / f'policy_net_{i}.pt')

            # zip finale
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for file in out_dir.rglob('*'):
                    if file.is_file():
                        zf.write(file, file.relative_to(out_dir))

        print(f'.... models saved in ZIP: {zip_path} ....')
        return str(zip_path)


    def load_models(self,
                zip_path: str,
                map_location: Optional[str] = None,
                cast_to: Optional[str] = None,
                strict: bool = True) -> None:
        """
        Carica pesi e normalizzazioni da uno ZIP creato da `save_models`.

        Parametri
        ---------
        zip_path : str
            Percorso allo ZIP (es. '.../sac.zip').
        map_location : Optional[str]
            'cpu', 'cuda', o torch.device; passato a torch.load.
        cast_to : Optional[str]
            Se 'fp16'/'bf16'/'fp32', forza il cast dei tensori floating del checkpoint
            PRIMA del load nello shape del modulo (utile se i pesi salvati sono fp16 ma il
            modello è in fp32, o viceversa).
        strict : bool
            Passato a `module.load_state_dict`. Metti False se ci sono key extra/mancanti.
        """

        def _to_torch_dtype(d):
            if d is None: return None
            d = d.lower()
            if d in ('fp16','float16'):  return torch.float16
            if d in ('bf16','bfloat16'): return torch.bfloat16
            if d in ('fp32','float32'):  return torch.float32
            raise ValueError("dtype non supportato: %s" % d)

        tgt_cast_dtype = _to_torch_dtype(cast_to)

        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP non trovato: {zip_path}")

        print(f".... loading models from ZIP: {zip_path} ....")

        # Estrai in dir temporanea
        with zipfile.ZipFile(zip_path, 'r') as zf, tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            tmp = Path(tmpdir)

            # 1) mean/std
            ms_file = tmp / 'mean_std' / 'config.json'
            if ms_file.exists():
                params = json.loads(ms_file.read_text())
                self.norm_mean = [np.asarray(x) for x in params.get('mean', [])]
                self.norm_std  = [np.asarray(x) for x in params.get('std',  [])]
            else:
                print("[warn] mean_std/config.json non trovato: salto normalizzazioni")

            # 2) deduci indici presenti nello zip (policy come riferimento)
            def _indices(pattern):
                rx = re.compile(pattern)
                idx = []
                for p in tmp.rglob('*'):
                    m = rx.fullmatch(p.name)
                    if m:
                        idx.append(int(m.group(1)))
                return sorted(set(idx))

            indices = _indices(r'policy_net_(\d+)\.pt')
            if not indices:
                # fallback: prova con soft_q_net1
                indices = _indices(r'soft_q_net1_(\d+)\.pt')
            if not indices:
                raise RuntimeError("Impossibile dedurre gli head/azioni dal contenuto dello ZIP.")

            # 3) helper di load con cast coerente
            def _load_into(module, ckpt_path: Path):
                sd = torch.load(ckpt_path, map_location=map_location)
                # se richiesto, forza il cast dei tensori floating
                if tgt_cast_dtype is not None:
                    for k, v in sd.items():
                        if torch.is_floating_point(v):
                            sd[k] = v.to(dtype=tgt_cast_dtype)
                else:
                    # altrimenti, prova ad allineare al dtype del modulo (se noto)
                    try:
                        target_dtype = next(module.parameters()).dtype  # può alzare StopIteration
                        for k, v in sd.items():
                            if torch.is_floating_point(v) and v.dtype != target_dtype:
                                sd[k] = v.to(dtype=target_dtype)
                    except StopIteration:
                        pass
                module.load_state_dict(sd, strict=strict)

            # 4) per ogni indice, carica i vari moduli se presenti nello zip e nell'istanza
            def _maybe(module_list, i, fname):
                if module_list is None or len(module_list) <= i:
                    return False
                p = tmp / fname
                if p.exists():
                    _load_into(module_list[i], p)
                    return True
                return False

            # attributi opzionali (SAC-like)
            soft_q_net1 = getattr(self, "soft_q_net1", None)
            soft_q_net2 = getattr(self, "soft_q_net2", None)
            target_soft_q_net1 = getattr(self, "target_soft_q_net1", None)
            target_soft_q_net2 = getattr(self, "target_soft_q_net2", None)
            policy_net = getattr(self, "policy_net", None)

            for i in indices:
                loaded_any = False
                loaded_any |= _maybe(soft_q_net1,        i, f"soft_q_net1_{i}.pt")
                loaded_any |= _maybe(soft_q_net2,        i, f"soft_q_net2_{i}.pt")
                loaded_any |= _maybe(target_soft_q_net1, i, f"target_soft_q_net1_{i}.pt")
                loaded_any |= _maybe(target_soft_q_net2, i, f"target_soft_q_net2_{i}.pt")
                loaded_any |= _maybe(policy_net,         i, f"policy_net_{i}.pt")
                if not loaded_any:
                    print(f"[warn] nessun file di rete trovato per indice {i}")

        print(".... models loaded ✓")

class SACRBC(SAC):
    r"""Uses :py:class:`citylearn.agents.rbc.RBC` to select actions during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    rbc: RBC
        :py:class:`citylearn.agents.rbc.RBC` or child class, used to select actions during exploration.
    
    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, rbc: Union[RBC, str] = None, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.__set_rbc(rbc, **kwargs)

    @property
    def rbc(self) -> RBC:
        """:py:class:`citylearn.agents.rbc.RBC` class child class or string path to an RBC 
        class e.g. 'citylearn.agents.rbc.RBC', used to select actions during exploration."""

        return self.__rbc
    
    def __set_rbc(self, rbc: RBC, **kwargs):
        if rbc is None:
            rbc = RBC(self.env, **kwargs)
        
        elif isinstance(rbc, RBC):
            pass

        elif isinstance(rbc, str):
            rbc = self.env.load_agent(rbc, env=self.env, **kwargs)

        else:
            rbc = rbc(self.env, **kwargs)
        
        self.__rbc = rbc

    def get_exploration_prediction(self, observations: List[float]) -> List[float]:
        """Return actions using :class:`RBC`."""

        return self.rbc.predict(observations)
