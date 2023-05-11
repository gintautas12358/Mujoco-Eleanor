from copy import deepcopy
import itertools
import torch
from torch.optim import Adam

from SpikeActorDeepCritic import SpikeActorDeepCritic


class PopsanTrainer:
    def __init__(self, observation_space, action_space, device, actor_critic=SpikeActorDeepCritic, ac_kwargs=dict(), gamma=0.99,
              polyak=0.995, popsan_lr=1e-4, q_lr=1e-3, alpha=0.2, batch_size=100, path=".") -> None:

        self.path = path
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.polyak = polyak
        
        # Create actor-critic module and target networks
        self.ac = actor_critic(observation_space, action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.ac.to(device)
        self.ac_targ.to(device)

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # List of parameters for PopSAN parameters (save this for convenience)
        self.popsan_params = itertools.chain(self.ac.popsan.encoder.parameters(),
                                        self.ac.popsan.snn.parameters(),
                                        self.ac.popsan.decoder.parameters())
        
        # Set up optimizers for policy and q-function
        self.popsan_mean_optimizer = Adam(self.popsan_params, lr=popsan_lr)
        self.pi_std_optimizer = Adam(self.ac.popsan.log_std_network.parameters(), lr=q_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Save parameters
        with open(path + "/parameter_log_trainer.txt", "w") as f:
            f.write("observation_space " + str(observation_space) + "\n")
            f.write("action_space " + str(action_space) + "\n")
            f.write("device " + str(device) + "\n")
            f.write("actor_critic " + str(actor_critic) + "\n")
            f.write("ac_kwargs " + str(ac_kwargs) + "\n")
            f.write("gamma " + str(gamma) + "\n")
            f.write("polyak " + str(polyak) + "\n")
            f.write("popsan_lr " + str(popsan_lr) + "\n")
            f.write("q_lr " + str(q_lr) + "\n")
            f.write("alpha " + str(alpha) + "\n")
            f.write("batch_size " + str(batch_size) + "\n")
            
    def setGradRequired(self, params, status):
        for p in params:
            p.requires_grad = status

    def updateQ(self, data):
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

    def updatePi(self, data):
        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        self.setGradRequired(self.q_params, False)

        self.popsan_mean_optimizer.zero_grad()
        self.pi_std_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.popsan_mean_optimizer.step()
        self.pi_std_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.setGradRequired(self.q_params, True)

    def updateTargetNetwork(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.updateQ(data)

        # Next run one gradient descent step for pi.
        self.updatePi(data)

        # Finally, update target networks by polyak averaging.
        self.updateTargetNetwork()
                

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.popsan(o, self.batch_size)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.to('cpu').detach().numpy())

        return loss_pi, pi_info
    
     # Set up function for computing Spike-SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.popsan(o2, self.batch_size)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.to('cpu').detach().numpy(),
                      Q2Vals=q2.to('cpu').detach().numpy())

        return loss_q, q_info

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32, device=self.device), 1,
                      deterministic)
    
    def save(self, epoch):
        torch.save(self.ac.popsan.state_dict(),
                        self.path + '/model_e' + str(epoch) + '.pt')
        
    def printStatistics(self):
        print("Learned Mean for encoder population: ")
        print(self.ac.popsan.encoder.mean.data)

        print("Learned STD for encoder population: ")
        print(self.ac.popsan.encoder.std.data)