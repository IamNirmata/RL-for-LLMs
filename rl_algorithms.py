import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Mock Environment and Models for Runnable Example ---

class MockEnv:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def reset(self):
        return torch.randn(self.state_dim).to(device)
    
    def step(self, action):
        next_state = torch.randn(self.state_dim).to(device)
        reward = torch.randn(1).item()
        done = np.random.rand() > 0.95
        return next_state, reward, done, {}

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, action_dim)
    
    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)
    
    def get_log_prob(self, states, actions):
        # Dummy implementation
        return torch.randn(states.shape[0]).to(device)
    
    def logprob(self, prompt, output):
        # Dummy for GRPO
        return torch.randn(1).to(device)
    
    def logprob_batch(self, prompts, responses):
        # Dummy for GRPO
        return torch.randn(len(prompts)).to(device)

    def generate(self, prompt, max_length=100, do_sample=True, temperature=0.7):
        # Dummy generation
        return torch.randn(10, 10).to(device) # Dummy token sequence

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 1)
    
    def forward(self, x):
        return self.fc(x)

class RewardModel:
    def evaluate(self, prompt, output):
        return torch.randn(1).item()
    
    def logprob_batch(self, prompts, responses):
        return torch.randn(len(prompts)).to(device)

# Helper functions
def collect_trajectories(policy, env, num_trajectories=5, horizon=100):
    trajectories = []
    for _ in range(num_trajectories):
        states, actions, rewards = [], [], []
        state = env.reset()
        for _ in range(horizon):
            # Dummy action selection
            action = torch.randn(env.action_dim).to(device)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done: break
        
        # Convert to tensors
        if states:
            trajectories.append({
                "states": torch.stack(states),
                "actions": torch.stack(actions),
                "rewards": torch.tensor(rewards).to(device)
            })
    return trajectories

def compute_discounted_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns).to(device)

def flat_concat(grads):
    return torch.cat([g.flatten() for g in grads])

def assign_params(module, new_params):
    # Simplified parameter assignment
    pass

def get_flat_params(module):
    return torch.cat([p.flatten() for p in module.parameters()])

def conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

class Batch:
    def __init__(self):
        self.states = torch.randn(64, 10).to(device)
        self.actions = torch.randn(64, 2).to(device)
        self.rewards = torch.randn(64).to(device)
        self.next_states = torch.randn(64, 10).to(device)
        self.dones = torch.zeros(64).to(device)
        self.old_log_probs = torch.randn(64).to(device)
        self.advantages = None
        self.returns = None

    def iter_minibatches(self, size):
        yield self

def collect_batch(policy, env, batch_size, max_steps):
    return Batch()

def compute_gae_advantages(rewards, dones, values, next_values, gamma, lam):
    return torch.randn(rewards.shape).to(device), torch.randn(rewards.shape).to(device)

def sample_prompts(batch_size):
    return [torch.randn(10).to(device) for _ in range(batch_size)]

# --- TRPO Implementation ---
def run_trpo():
    print("\n--- Running TRPO ---")
    state_dim = 10
    action_dim = 2
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    value_fn = ValueNetwork(state_dim).to(device)
    env = MockEnv(state_dim, action_dim)
    critic_optimizer = optim.Adam(value_fn.parameters(), lr=1e-3)

    # Hyperparameters for TRPO
    max_kl = 0.01         # KL divergence limit (trust region size)
    cg_iters = 10         # iterations for conjugate gradient
    backtrack_iters = 10  # line search iterations for step size
    backtrack_coeff = 0.8 # step size reduction factor in line search

    # Step 1: Collect trajectories using current policy
    trajectories = collect_trajectories(policy, env, num_trajectories=5, horizon=50)
    
    if not trajectories:
        print("No trajectories collected.")
        return

    # Step 2: Compute returns and advantages for each time step in each trajectory
    for traj in trajectories:
        rewards = traj["rewards"]
        states  = traj["states"]
        # Compute discounted cumulative returns G_t for each step t
        returns = compute_discounted_returns(rewards, gamma=0.99)
        # Get state value estimates from critic for baseline
        values  = value_fn(states).squeeze()
        # Advantage = return - value (could use GAE for smoothing advantages)
        advantages = returns - values.detach()
        traj["advantages"] = advantages
        traj["returns"]    = returns

    # Combine trajectory data for training
    states     = torch.cat([traj["states"] for traj in trajectories])
    actions    = torch.cat([traj["actions"] for traj in trajectories])
    advantages = torch.cat([traj["advantages"] for traj in trajectories])
    old_log_probs = policy.get_log_prob(states, actions).detach()  # log π_old(a|s)

    # Step 3: Compute policy gradient under the current policy
    # This is the gradient of the surrogate objective L = E[ (pi_theta / pi_old) * A ]
    policy_loss = -torch.mean(torch.exp(policy.get_log_prob(states, actions) - old_log_probs) * advantages)
    policy_grad = torch.autograd.grad(policy_loss, policy.parameters(), retain_graph=True) # retain_graph for demo
    policy_grad = flat_concat(policy_grad)  # flatten gradient vector

    theta_current = get_flat_params(policy)  # current policy parameters (flattened)

    # Step 4: Define a function to compute KL and Fisher-vector product for conjugate gradient
    def compute_kl_and_fisher_vec(vec):
        # Compute KL divergence between current policy and a hypothetical policy moved by vector 'vec'
        # assign_params(policy, theta_current + vec)        # temporarily shift policy by vec
        new_log_probs = policy.get_log_prob(states, actions)
        kl = torch.mean(torch.exp(old_log_probs) * (old_log_probs - new_log_probs))  # KL(π_old || π_new)
        
        # Compute gradient of (∇_theta KL * vec) which gives Fisher-vector product
        # Note: This is a dummy implementation for the mock
        kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        kl_grad = flat_concat(kl_grad)
        fisher_vec_product = torch.autograd.grad(torch.dot(kl_grad, vec), policy.parameters(), retain_graph=True)
        fisher_vec_product = flat_concat(fisher_vec_product)
        # assign_params(policy, theta_current)              # restore original parameters
        return kl.item(), fisher_vec_product

    # Conjugate Gradient to solve F * x = -g (find search direction x = F^{-1} * (-grad))
    g = -policy_grad  # we want to maximize the objective, so take negative grad for ascent
    # Mocking the CG call for the example to run without full implementation of Fisher product
    x = torch.randn_like(g) 

    # Step 5: Determine step length to satisfy KL constraint
    step_dir = x  # this is the proposed natural gradient step
    
    # Mock KL computation
    current_kl = 0.02 
    
    if current_kl > max_kl:
        # Scale step to meet KL constraint: scale = sqrt(max_kl / current_kl)
        step_dir *= np.sqrt(max_kl / (current_kl + 1e-8))

    # Step 6: Line search along step direction to ensure improvement
    step_size = 1.0
    for _ in range(backtrack_iters):
        new_params = theta_current + step_size * step_dir
        # assign_params(policy, new_params)
        # Surrogate objective at new params
        new_loss = -torch.mean(torch.exp(policy.get_log_prob(states, actions) - old_log_probs) * advantages)
        
        kl = 0.005 # Mock KL
        
        if new_loss.item() < policy_loss.item() and kl < max_kl:
            # Found acceptable update
            theta_current = new_params
            break
        step_size *= backtrack_coeff  # reduce step and try again
    # End line search

    # Step 7: Update policy to new parameters
    # assign_params(policy, theta_current)

    # Step 8: Update value network (critic) by regression to the returns
    critic_loss = torch.mean((value_fn(states).squeeze() - returns)**2)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    print("TRPO step completed.")

# --- PPO Implementation ---
def run_ppo():
    print("\n--- Running PPO ---")
    state_dim = 10
    action_dim = 2
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    value_fn = ValueNetwork(state_dim).to(device)
    env = MockEnv(state_dim, action_dim)

    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer  = optim.Adam(value_fn.parameters(), lr=1e-3)
    clip_ratio = 0.2   # PPO clip parameter epsilon
    train_epochs = 4   # how many epochs to train on one batch of trajectories
    batch_size  = 64   # minibatch size for each epoch

    # Collect trajectories (batch of experience) using current policy
    batch = collect_batch(policy, env, batch_size=2048, max_steps=200)
    # batch contains tensors: states, actions, rewards, next_states, done, old_log_probs (log π_old(a|s))

    # Compute rewards-to-go and advantages using the current value function
    with torch.no_grad():
        # Compute value estimates for all states and next_states
        values      = value_fn(batch.states)
        next_values = value_fn(batch.next_states)
    # Compute Generalized Advantage Estimation (GAE) or simple advantage
    advantages, returns = compute_gae_advantages(batch.rewards, batch.dones, values, next_values, gamma=0.99, lam=0.95)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize advantages for stability

    # Add the advantage and returns to the batch
    batch.advantages = advantages
    batch.returns    = returns

    # Record old policy probabilities (from data collection) for importance sampling
    old_log_probs = batch.old_log_probs

    # PPO policy and value network training
    for epoch in range(train_epochs):
        # Shuffle and iterate over mini-batches
        for minibatch in batch.iter_minibatches(size=batch_size):
            # Extract mini-batch data
            mb_states      = minibatch.states
            mb_actions     = minibatch.actions
            mb_advantages  = minibatch.advantages
            mb_returns     = minibatch.returns
            mb_old_logp    = minibatch.old_log_probs

            # Forward pass: get current log probabilities and entropy from policy
            # dist = policy(mb_states)               # get action distribution for current states
            # logp = dist.log_prob(mb_actions)       # log π_theta(a|s) for taken actions
            logp = torch.randn(mb_states.shape[0]).to(device) # Mock logp

            # Also get value function predictions for these states
            values_pred = value_fn(mb_states).squeeze(-1)

            # Calculate the probability ratio r(θ) = exp(log π_new - log π_old) for each action
            ratio = torch.exp(logp - mb_old_logp)
            # Compute unclipped and clipped policy advantages
            unclipped_obj = ratio * mb_advantages
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            clipped_obj   = clipped_ratio * mb_advantages
            # PPO policy loss: maximize the minimum => minimize negative of it
            policy_loss = -torch.mean(torch.min(unclipped_obj, clipped_obj))
            # Optional: add an entropy bonus to encourage exploration (not shown here for brevity)

            # PPO value function loss: mean squared error of value predictions vs returns
            value_loss = torch.mean((values_pred - mb_returns)**2)

            # Take optimization steps for policy and value networks
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
    print("PPO step completed.")

# --- GRPO Implementation ---
def run_grpo():
    print("\n--- Running GRPO ---")
    state_dim = 10
    action_dim = 2
    policy_model = PolicyNetwork(state_dim, action_dim).to(device)
    reward_model = RewardModel()
    reference_model = RewardModel() # Mock reference model
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)

    batch_prompts = sample_prompts(batch_size=32)  # e.g., 32 different query prompts

    G = 4  # number of responses to generate per prompt (group size)
    responses = []  # to store all generated responses
    log_probs = []  # to store log-probs of those responses under current policy
    rewards = []    # to store reward model outputs

    # Step 1: Generate G responses for each prompt using the current policy model
    for prompt in batch_prompts:
        prompt_responses = []
        prompt_logps = []
        for _ in range(G):
            output = policy_model.generate(prompt, max_length=100, do_sample=True, temperature=0.7)
            # `generate` returns a sequence of tokens (the model's response).
            prompt_responses.append(output)
            # Compute log probability of the generated sequence under the current policy
            logp = policy_model.logprob(prompt, output)  # sum of log probs of each token in the sequence
            prompt_logps.append(logp.detach())
            # Score the (prompt, output) with reward model (e.g., a scalar preference score)
            R = reward_model.evaluate(prompt, output)
            rewards.append(R)
        # Store the group of responses and log-probs for this prompt
        responses.append(prompt_responses)
        log_probs.append(prompt_logps)

    # Step 2: Compute advantages using group relative rewards
    advantages = []
    idx = 0
    for i, prompt in enumerate(batch_prompts):
        # rewards for the i-th prompt's G responses:
        r_group = torch.tensor(rewards[idx: idx+G], dtype=torch.float32).to(device)
        idx += G
        # Compute group mean and (optionally) std
        mean_R = torch.mean(r_group)
        std_R  = torch.std(r_group)
        for r in r_group:
            # Advantage = (reward - mean) / (std + eps)
            if std_R > 1e-6:
                adv = (r - mean_R) / (std_R + 1e-8)
            else:
                adv = r - mean_R  # if zero variance, advantage is difference from mean
            advantages.append(adv)
    advantages = torch.stack(advantages)  # shape [batch_size*G]

    # Step 3: Prepare policy loss with PPO clipping (no value loss since no value network)
    log_probs = torch.cat([torch.stack(lp) for lp in log_probs])  # flatten log_probs of all prompt responses
    # For importance sampling, we need old log-probs. 
    # In practice, we'd store the log_probs from when these responses were generated (the policy hasn't changed since generation here).
    old_log_probs = log_probs.detach()  # treat current log_probs as old since we haven't updated yet

    # Compute policy ratio r = exp(new_logp - old_logp). Here new and old are same (one update), 
    # but if this loop iterates multiple epochs, we'd recompute new_logp each time.
    new_log_probs = policy_model.logprob_batch(batch_prompts, responses)  # hypothetical function to get log-probs for all generated outputs
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped PPO objective with our advantages
    clip_eps = 0.2
    unclipped_obj = ratio * advantages
    clipped_obj   = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    ppo_objective = torch.mean(torch.min(unclipped_obj, clipped_obj))  # we want to maximize this

    # KL regularization: compare policy to reference (e.g., SFT model) on the generated outputs
    # Compute KL(old_policy || reference_policy) as an additional penalty
    ref_log_probs = reference_model.logprob_batch(batch_prompts, responses)
    kl_div = torch.mean(torch.exp(old_log_probs) * (old_log_probs - ref_log_probs))
    # (Above is an approximation: effectively average KL over tokens or sequences)

    beta = 0.1  # weight for KL penalty
    loss = -ppo_objective + beta * kl_div  # final loss to MINIMIZE (negative of objective plus KL term)

    # Step 4: Update policy model via backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("GRPO step completed.")

if __name__ == "__main__":
    run_trpo()
    run_ppo()
    run_grpo()
