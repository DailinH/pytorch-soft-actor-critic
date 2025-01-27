import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
from realnvp_layers import RealNVP
from torch.distributions import MultivariateNormal
from torch import distributions, nn


class DRIL(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.prior = MultivariateNormal(torch.zeros(num_inputs).to(self.device), torch.eye(num_inputs).to(self.device))
        # print(num_inputs)
        nets = lambda: nn.Sequential(nn.Linear(num_inputs, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, num_inputs), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(num_inputs, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, num_inputs))
        # self.density_model = Net(N=num_inputs*2, input_dim=num_inputs, hidden_dim=256, device=self.device).to(self.device)
        # self.density_optim = torch.optim.Adam(self.density_model.parameters(), lr=args.lr, weight_decay=5e-4)
        prior = distributions.MultivariateNormal(torch.zeros(num_inputs).to(self.device), torch.eye(num_inputs).to(self.device))
        mask_checkerboard = np.indices((1, num_inputs)).sum(axis=0)%2
        mask_checkerboard = np.append(mask_checkerboard,1 - mask_checkerboard,axis=0)
        masks = torch.from_numpy(np.array(list(mask_checkerboard)*3).astype(np.float32)).to(self.device)
        self.flow = RealNVP(nets, nett, masks, prior).to(self.device)
        self.optimizer = torch.optim.Adam([p for p in self.flow.parameters() if p.requires_grad==True], lr=1e-4)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def density_pseudocount(self, s_tp1):
        tmp_flow = self.flow
        optim = self.optimizer
        # fake update s_tp1
        tmp_flow, loss, rho = self.density_update(s_tp1, tmp_flow, optim, update=True)
        tmp_flow.eval()
        # with torch.no_grad():
        tmp_flow, loss_, rho_ = self.density_update(s_tp1, tmp_flow, optim, update=False)
        PG =  rho_ - rho
        PG[PG<=0] = 1e-7
        N = (torch.exp(PG.detach()) - 1)
        alpha = (1e-7 / N)
        return alpha.detach(), N.detach()
    
    def density_update(self, s_t, flow, optim=None, update=True):
        if optim is None:
            optim = self.optimizer
        logp = flow.log_prob(s_t)
        loss = -logp.mean()
        
        if update:
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
        return flow, loss, logp


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters_rl(self, memory, batch_size, updates, discriminator_threshold=0.2):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        self.flow, _, _ = self.density_update(state_batch, self.flow)
        self.alpha, N = self.density_pseudocount(next_state_batch)
        # self.alpha[self.alpha > discriminator_threshold] = 0
        state_batch = state_batch[torch.where(self.alpha > discriminator_threshold)]
        action_batch = action_batch[torch.where(self.alpha > discriminator_threshold)]
        next_q_value = next_q_value[torch.where(self.alpha > discriminator_threshold)]
        alpha = self.alpha[torch.where(self.alpha > discriminator_threshold)]

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha.detach() * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # if self.automatic_entropy_tuning:
        #     alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        #     self.alpha_optim.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optim.step()

        #     self.alpha = self.log_alpha.exp()
        #     alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        # else:
        #     alpha_loss = torch.tensor(0.).to(self.device)
        #     alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), torch.mean(N), torch.mean(self.alpha), len(self.alpha) - len(alpha) #, alpha_loss.item(), alpha_tlogs.item()

    def update_parameters_il(self, memory, memory_len):
        # Sample a batch from memory
        # print(memory_len)
        # print(torch.randperm(memory_len).tolist())
        memory = np.array(memory)[torch.randperm(memory_len).tolist()]
        state_batch, action_batch = list(zip(*memory))

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)

        pi, log_pi, a_t = self.policy.sample(state_batch)
        # cross entropy loss for policy loss (not yet edited)
        policy_loss = (0.5 * (a_t - action_batch)**2).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        return policy_loss.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

