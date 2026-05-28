# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
log = logging.getLogger(__name__)
from model.flow.mlp_meanflow import MeanFlowMLP
from model.flow.ft_ppo.ppoflow import PPOFlow
import torch
import copy
from torch import Tensor as Tensor
from torch.distributions import Normal
import torch.nn.functional as F

class NoisyMeanFlowMLP(torch.nn.Module):
    """
    Noisy version of MeanFlow MLP for PPO fine-tuning.
    Adds exploration noise capability to the base MeanFlow network.
    """
    def __init__(self, 
                 policy: MeanFlowMLP,
                 denoising_steps,
                 learn_explore_noise_from,
                 inital_noise_scheduler_type,
                 min_logprob_denoising_std,
                 max_logprob_denoising_std,
                 learn_explore_time_embedding,
                 time_dim_explore,
                 use_time_independent_noise,
                 device,
                 noise_hidden_dims,
                 activation_type):
        super().__init__()
        
        self.policy = policy
        self.denoising_steps = denoising_steps
        self.learn_explore_noise_from = learn_explore_noise_from
        self.device = device
        
        # Noise parameters for stability
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(min_logprob_denoising_std**2, device=device)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(max_logprob_denoising_std**2, device=device)), requires_grad=False
        )
        
        # Time embedding for exploration noise
        if learn_explore_time_embedding:
            from model.diffusion.modules import SinusoidalPosEmb
            self.time_embedding_explore = torch.nn.Sequential(
                SinusoidalPosEmb(time_dim_explore),
                torch.nn.Linear(time_dim_explore, time_dim_explore * 2),
                torch.nn.Mish(),
                torch.nn.Linear(time_dim_explore * 2, time_dim_explore),
            ).to(device)
        else:
            self.time_embedding_explore = None
            
        # MLP for noise prediction
        self.use_time_independent_noise = use_time_independent_noise
        if use_time_independent_noise:
            input_dim = policy.act_dim_total + policy.cond_enc_dim
        else:
            input_dim = policy.act_dim_total + policy.cond_enc_dim + time_dim_explore
            
        from model.common.mlp import MLP
        self.mlp_logvar = MLP(
            [input_dim] + noise_hidden_dims + [policy.act_dim_total],
            activation_type=activation_type,
            out_activation_type="Identity",
        ).to(device)
    
    def forward(self, action, time, r, cond, learn_exploration_noise=True, step=None):
        """
        Forward pass for noisy MeanFlow.
        
        Args:
            action: (B, Ta, Da) action trajectories
            time: (B,) time parameter t
            r: (B,) time parameter r
            cond: condition dict
            learn_exploration_noise: whether to predict exploration noise
            step: current denoising step
            
        Returns:
            u: (B, Ta, Da) average velocity
            noise_std: (B, Ta*Da) exploration noise standard deviation
        """
        B, Ta, Da = action.shape
        
        # Get average velocity from base policy
        u = self.policy(action, time, r, cond)
        
        # Predict exploration noise
        if learn_exploration_noise and step is not None and step >= self.learn_explore_noise_from:
            # Prepare inputs for noise prediction
            action_flat = action.view(B, -1)
            
            # Encode full observation (including images if present)
            cond_encoded = self.policy.forward_encoder(cond)  # Use the full encoder
                
            if self.use_time_independent_noise:
                noise_input = torch.cat([action_flat, cond_encoded], dim=-1)
            else:
                # Use time embedding for exploration
                if self.time_embedding_explore is not None:
                    if isinstance(time, (int, float)):
                        time_tensor = torch.ones((B, 1), device=action.device) * time
                    else:
                        time_tensor = time.view(B, 1)
                    time_emb = self.time_embedding_explore(time_tensor).view(B, -1)
                    noise_input = torch.cat([action_flat, cond_encoded, time_emb], dim=-1)
                else:
                    noise_input = torch.cat([action_flat, cond_encoded], dim=-1)
            
            # Predict log variance
            logvar = self.mlp_logvar(noise_input)
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
            noise_std = torch.exp(0.5 * logvar)
        else:
            # Use minimum noise when not learning exploration
            noise_std = torch.exp(0.5 * self.logvar_min).expand(B, Ta * Da)
            
        return u, noise_std


class PPOMeanFlow(PPOFlow):
    """PPO fine-tuning for MeanFlow policies."""
    
    def __init__(self, 
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type
                 ):
        
        super().__init__(
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type
        )
    
    def init_actor_ft(self, policy_copy):
        """Initialize fine-tuning actor with noisy MeanFlow MLP."""
        self.actor_ft = NoisyMeanFlowMLP(
            policy=policy_copy,
            denoising_steps=self.inference_steps,
            learn_explore_noise_from=self.inference_steps - self.ft_denoising_steps,
            inital_noise_scheduler_type=self.noise_scheduler_type,
            min_logprob_denoising_std=self.min_logprob_denoising_std,
            max_logprob_denoising_std=self.max_logprob_denoising_std,
            learn_explore_time_embedding=self.learn_explore_time_embedding,
            time_dim_explore=self.time_dim_explore,
            use_time_independent_noise=self.use_time_independent_noise,
            device=self.device,
            noise_hidden_dims=self.noise_hidden_dims,
            activation_type=self.explore_net_activation_type
        )
    
    def get_logprobs(self, 
                     cond: dict, 
                     x_chain: Tensor, 
                     get_entropy=False, 
                     normalize_denoising_horizon=False, 
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        """
        Compute log probabilities for MeanFlow policy.
        
        MeanFlow uses average velocity u(x_t, t, r, s) and sampling formula:
        x_r = x_t - (t-r) * u(x_t, t, r, s) + noise
        
        The transition probability is:
        p(x_{r}|x_t, s) = N(x_r | x_t - (t-r)*u(x_t,t,r,s), σ^2)
        """
        logprob = 0.0
        joint_entropy = 0.0
        entropy_rate_est = 0.0
        logprob_steps = 0
        
        B = x_chain.shape[0]
        chains_prev = x_chain[:, :-1, :, :].flatten(-2, -1)  # [B, inference_steps, Ta*Da]
        chains_next = x_chain[:, 1:, :, :].flatten(-2, -1)   # [B, inference_steps, Ta*Da]
        chains_stds = torch.zeros_like(chains_prev, device=self.device)
        
        # Initial probability p(x_0) ~ N(0, 1)
        init_dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)  # [B]
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)  # [B]
        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1
        
        # Compute transition probabilities for MeanFlow
        chains_vel = torch.zeros_like(chains_prev, device=self.device)  # [B, inference_steps, Ta*Da]
        
        # MeanFlow time schedule: from 1.0 to 0.0
        t_vals = torch.linspace(1.0, 0.0, self.inference_steps + 1, device=self.device)
        
        for i in range(self.inference_steps):
            # Current and next time points
            t_curr = t_vals[i]
            r_next = t_vals[i + 1]
            
            # Create batch-wise time tensors
            t = torch.full((B,), t_curr, device=self.device)
            r = torch.full((B,), r_next, device=self.device)
            
            # Current state
            xt = x_chain[:, i]  # [B, Ta, Da]
            
            # Predict average velocity and noise
            # ut, nt = self.actor_ft.forward(xt, t, r, cond, True, i)  # [B, Ta, Da], [B, Ta*Da]
            was_training = self.actor_ft.policy.training
            self.actor_ft.policy.eval()
            try:
                ut, nt = self.actor_ft.forward(xt, t, r, cond, True, i)
            finally:
                self.actor_ft.policy.train(was_training)
            
            chains_vel[:, i] = ut.flatten(-2, -1)  # [B, Ta*Da]
            chains_stds[:, i] = nt  # [B, Ta*Da]
            logprob_steps += 1
        
        # MeanFlow transition: x_r = x_t - (t-r) * u(x_t, t, r, s)
        time_diffs = []
        for i in range(self.inference_steps):
            time_diff = t_vals[i] - t_vals[i + 1]  # t_curr - r_next
            time_diffs.append(time_diff)
        time_diffs = torch.tensor(time_diffs, device=self.device).view(-1, 1).expand(-1, chains_prev.shape[-1])
        time_diffs = time_diffs.unsqueeze(0).expand(B, -1, -1)  # [B, inference_steps, Ta*Da]
        
        # Expected next state: x_t - (t-r) * u(x_t, t, r, s)
        chains_mean = chains_prev - time_diffs * chains_vel  # [B, inference_steps, Ta*Da]
        
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        
        # Transition distribution
        chains_dist = Normal(chains_mean, chains_stds)
        
        # Log probability and entropy of transitions
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)  # [B, inference_steps]
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)  # [B, inference_steps]
        
        # Total log probability
        logprob += logprob_trans.sum(-1)  # [B]
        if self.logprob_debug_recalculate: 
            log.info(f"logprob_init={logprob_init.mean().item()}, logprob_trans={logprob_trans.mean().item()}")
        
        # Total entropy
        if get_entropy:
            joint_entropy += entropy_trans.sum(-1)
        
        if get_entropy:
            entropy_rate_est = joint_entropy / logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.act_dim_total
        
        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")
        
        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est, 
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob
    
    @torch.no_grad()
    def get_actions(self, 
                    cond: dict, 
                    eval_mode: bool, 
                    save_chains=False, 
                    normalize_denoising_horizon=False, 
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True
                    ):
        """
        Generate actions using MeanFlow sampling procedure.
        
        MeanFlow sampling: x_r = x_t - (t-r) * u(x_t, t, r, s) + noise
        """
        B = cond["state"].shape[0]
        
        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps + 1, self.horizon_steps, self.action_dim), device=self.device)
        if ret_logprob:
            log_prob = 0.0 
            log_prob_steps = 0
            if self.logprob_debug_sample: 
                log_prob_list = []
        
        # Sample initial point from Gaussian
        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob += log_prob_init
            log_prob_steps += 1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())
        
        if save_chains:
            x_chain[:, 0] = xt
        
        # MeanFlow time schedule: from 1.0 to 0.0
        t_vals = torch.linspace(1.0, 0.0, self.inference_steps + 1, device=self.device)
        
        for i in range(self.inference_steps):
            # Current and next time points
            t_curr = t_vals[i]
            r_next = t_vals[i + 1]
            
            # Create batch-wise time tensors
            t = torch.full((B,), t_curr, device=self.device)
            r = torch.full((B,), r_next, device=self.device)
            
            # Predict average velocity and exploration noise
            # IMPORTANT: During training, we need to learn exploration noise to match get_logprobs()
            # During eval, we use deterministic policy without learned noise
            #ut, nt = self.actor_ft.forward(xt, t, r, cond, learn_exploration_noise=not eval_mode, step=i)
            was_training = self.actor_ft.policy.training
            self.actor_ft.policy.eval()
            try:
                ut, nt = self.actor_ft.forward(xt, t, r, cond, learn_exploration_noise=not eval_mode, step=i)
            finally:
                self.actor_ft.policy.train(was_training)
            
            # MeanFlow update: x_r = x_t - (t-r) * u(x_t, t, r, s)
            time_diff = t_curr - r_next
            xt = xt - time_diff * ut
            
            if clip_intermediate_actions:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)
            
            # Add exploration noise during training
            std = nt.unsqueeze(-1).reshape(xt.shape)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)
            dist = Normal(xt, std)
            
            if not eval_mode:
                xt = dist.sample().clamp_(
                    dist.loc - self.randn_clip_value * dist.scale,
                    dist.loc + self.randn_clip_value * dist.scale
                ).to(self.device)
            
            # Final action clipping
            if i == self.inference_steps - 1:
                xt = xt.clamp_(self.act_min, self.act_max)
                
            if ret_logprob:
                # Compute transition log probability
                logprob_transition = dist.log_prob(xt).sum(dim=(-2, -1)).to(self.device)
                if self.logprob_debug_sample: 
                    log_prob_list.append(logprob_transition.mean().item())
                log_prob += logprob_transition
                log_prob_steps += 1
                
            if save_chains:
                x_chain[:, i + 1] = xt
        
        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total
            if self.logprob_debug_sample:
                transform_logprob = torch.log(1 - torch.tanh(xt) ** 2 + 1e-7).sum(dim=(-2, -1)).mean().item()
                print(f"log_prob_list={log_prob_list}, transform={transform_logprob}")
        
        if ret_logprob:
            if save_chains:
                return (xt, x_chain, log_prob)  
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt