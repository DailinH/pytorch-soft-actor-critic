import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from torch.distributions import MultivariateNormal
from realnvp_layers import Net, AffineCouplingLayer
import copy
from clpf.argparser import parse_arguments
from clpf.models.clpf import model_builder
from clpf.data_tools.bm_sequential_batch import BMSequenceBatch
from clpf.lib.utils import optimizer_factory, count_parameters, subsample_data

class CBSAC(object):
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

        ### Density initialization
        config = parse_arguments() ## NOTE: probably needs to be changed
        self.density_model = model_builder(config).to(self.device)
        train_loader, val_loader = get_real_dataset(config)  ## NOTE: Need to be changed here!
        data_preprocess = BMSequenceBatch.preprocess
        trainable_parameters = self.density_model.parameters()
        self.optimizer, num_params = optimizer_factory(config, trainable_parameters)
        self.density_model = torch.nn.DataParallel(self.density_model)
        



        # self.prior = MultivariateNormal(torch.zeros(num_inputs), torch.eye(num_inputs))
        # # print(num_inputs)
        # self.density_model = model_builder(args)
        # self.density_optim = torch.optim.Adam(self.density_model.parameters(), lr=args.lr, weight_decay=5e-4)

        if args.load_density_model:
            density_model_ckpt = torch.load(args.density_model_dir)
            self.density_model.load_state_dict(density_model_ckpt['net'])
            self.density_optim.load_state_dict(density_model_ckpt['optim'])
        

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
        temp_density_model = copy.deepcopy(self.density_model)
        optim = copy.deepcopy(self.density_optim)
        prior = copy.deepcopy(self.prior)

        # fake update s_tp1
        temp_density_model.train()
        z, log_det_loss = temp_density_model.forward(s_tp1)
        optim.zero_grad()
        rho = prior.log_prob(z)
        # print(z)
        loss = rho + log_det_loss
        loss = - loss.mean()
        loss.backward()
        optim.step()
        print("rho",rho[0])

        # inference s_tp1
        with torch.no_grad():
            temp_density_model.eval()
            x, _ = temp_density_model.forward(s_tp1, reverse=True)
            rho_ = prior.log_prob(x)
            print("rho_", rho_[0])
        
        PG =  rho_ - rho
        PG[PG<0] = 1e-8
        # print("PG", PG)
        N = (torch.exp(PG) - 1)
        # print("N", N)
        alpha = (1e-8 / N)
        # print("alpha", alpha)

        del temp_density_model
        del optim
        del prior
        return alpha.detach(), N.detach()
    
    def density_update(self, s_t):
        s_t = 0.05 + (1-0.05) * s_t # what is this step?
        z, log_det_loss = self.density_model.forward(s_t)
        self.density_optim.zero_grad()
        # print("density update debug")
        # print("det ", log_det_loss)
        # print(z)
        # print([self.prior.log_prob(z_) for z_ in z])
        loss = self.prior.log_prob(z) + log_det_loss
        loss = -loss.mean()
        loss.backward()
        self.density_optim.step()
        return loss

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        self.density_model.train()
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

        self.density_update(state_batch)
        self.alpha, N = self.density_pseudocount(next_state_batch)

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

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

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

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), torch.mean(N), torch.mean(self.alpha) #, alpha_loss.item(), alpha_tlogs.item()

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

