module ParallelTempering
import Base.@kwdef
import Distributions.sample
using ProgressMeter

@kwdef mutable struct ParallelSampler
    """A sampler for parallel tempering MCMC."""
    log_likelihood
    log_prior
    d
    betas
    base_proposal
    n_samples
    n_temps
    chains
    loglikes
    logpriors
    scales
    swap_type
    swap_rate
    adapt
    n_adapt
    n_mh_proposals
    accs
    swap_accs
    i
    target_acc_rate
end

"""
    ParallelSampler(log_likelihood, log_prior, d, betas, base_proposal, n_samples, swap_rate=0.5; x0=zeros(d))

Construct a ParallelSampler with default settings.

# Arguments
- `log_likelihood::function`: the log-loglikelihood of the model, as a function
  of the parameters.
- `log_prior::function`: the log-prior of the model, as a function of the
  parameters.
- `d:Int`: the dimension of the model.
- `betas::Vector{Float64}`: the inverse temperatures for parallel-tempering.
- `base_proposal::Distribution`: the base proposal distribution for MH steps.
- `n_samples::Int`: the number of samples to acquire.
- `swap_rate::Float64`: the proportion of swap proposals. A swap proposal will
   be made at each step if a `rand() < swap_rate`, otherwise a MH proposal will
   be made.
- `x0::Vector{Float64}` (optiona): the initial starting parameters. Defaults to
   zeros.
"""
function ParallelSampler(log_likelihood, log_prior, d, betas, base_proposal, n_samples, swap_rate=0.5; x0=zeros(d))
    n_temps = length(betas)
    n_adapt = n_samples ÷ 4
    loglikes = -Inf * ones(n_temps, n_samples)
    ll0 = log_likelihood(x0)
    loglikes[:, 1] .= ll0
    logpriors = -Inf * ones(n_temps, n_samples)
    lp0 = log_prior(x0)
    logpriors[:, 1] .= lp0
    chains = zeros(d, n_temps, n_samples)
    for t in 1:n_temps
        chains[:, t, 1] = x0
    end

    return ParallelSampler(
        log_likelihood = log_likelihood,
        log_prior = log_prior,
        d = d,
        betas = betas,
        base_proposal = base_proposal,
        n_samples = n_samples,
        n_temps = n_temps,
        chains = chains,
        loglikes = loglikes,
        logpriors = logpriors,
        scales = 0.5 * ones(n_temps),
        swap_type = :random,
        swap_rate = swap_rate,
        adapt = true,
        n_adapt = n_adapt,
        n_mh_proposals = 0,
        accs = zeros(n_temps),
        swap_accs = 0,
        i = 1,
        target_acc_rate = 0.5,
    )
end

"""
    run_mcmc(sampler::ParallelSampler)

Run parallel-tempering MCMC.
"""
function run_mcmc(sampler::ParallelSampler)
    acc_hist = zeros(sampler.n_temps, sampler.n_samples)
    @showprogress 1 "Sampling" for i in 2:sampler.n_samples
        step!(sampler)
        acc_hist[:, i] = sampler.accs / sampler.n_mh_proposals
        yield()
    end
    @assert !any(sampler.loglikes .== -Inf)
    @assert !any(sampler.logpriors .== -Inf)
    return acc_hist
end

"""
    step!(sampler::ParallelSampler)

Make a single step of the parallel-tempering sampler.
"""
function step!(sampler::ParallelSampler)
    if rand() < sampler.swap_rate
        propose_swap!(sampler)
    else
        propose_mh!(sampler)
        if (sampler.i < sampler.n_adapt) && (sampler.n_mh_proposals > 0)
            adapt!(sampler)
        end
    end
end

"""
    propose_swap!(sampler::ParallelSampler)

Propose a swap between two chains of different temperature.
"""
function propose_swap!(sampler::ParallelSampler)
    i = sampler.i
    if sampler.swap_type == :random
        j, k = sample(1:sampler.n_temps, 2, replace=false)
    else
        error("Swap types other than :random are not implemented.")
    end
    # alpha = (p_j(t_k) * p_k(t_j)) /(p_j(t_j) * p_k(t_k))
    log_denom = sampler.loglikes[j, i] * sampler.betas[j] + sampler.logpriors[j, i]
    log_denom += sampler.loglikes[k, i] * sampler.betas[k] + sampler.logpriors[k, i]
    t_j = sampler.chains[:, j, i]
    t_k = sampler.chains[:, k, i]
    log_numer = sampler.loglikes[k, i] * sampler.betas[j] + sampler.logpriors[k, i]
    log_numer += sampler.loglikes[j, i] * sampler.betas[k] + sampler.logpriors[j, i]
    if log(rand()) < log_numer - log_denom
        # Accept
        sampler.swap_accs += 1
        # Copy previous iteration
        sampler.chains[:, :, i + 1] = sampler.chains[:, :, i]
        sampler.loglikes[:, i + 1] = sampler.loglikes[:, i]
        sampler.logpriors[:, i + 1] = sampler.logpriors[:, i]
        # Swap configuration j and k
        sampler.chains[:, j, i + 1], sampler.chains[:, k, i + 1] = sampler.chains[:, k, i + 1], sampler.chains[:, j, i + 1]
        sampler.loglikes[j, i + 1], sampler.loglikes[k, i + 1] = sampler.loglikes[k, i + 1], sampler.loglikes[j, i + 1]
        sampler.logpriors[j, i + 1], sampler.logpriors[k, i + 1] = sampler.logpriors[k, i + 1], sampler.logpriors[j, i + 1]
    else
        # Reject
        # Copy previous iteration
        sampler.chains[:, :, i + 1] = sampler.chains[:, :, i]
        sampler.loglikes[:, i + 1] = sampler.loglikes[:, i]
        sampler.logpriors[:, i + 1] = sampler.logpriors[:, i]
    end
    sampler.i += 1
end

"""
    propose_mh!(sampler::ParallelSampler)

Metropolis-Hastings move (for each chain).
"""
function propose_mh!(sampler::ParallelSampler)
    i = sampler.i
    for (t, (scale, beta)) in enumerate(zip(sampler.scales, sampler.betas))
        new_x = sampler.chains[:, t, i] + scale * rand(sampler.base_proposal)
        new_ll = sampler.log_likelihood(new_x)
        new_lp = sampler.log_prior(new_x)
        new_logprob =  new_ll * beta + new_lp
        old_logprob = sampler.loglikes[t, i] * beta + sampler.logpriors[t, i]
        alpha = log(rand())
        if alpha < new_logprob - old_logprob
            # Accept
            sampler.accs[t] += 1
            sampler.chains[:, t, i + 1] = new_x
            sampler.loglikes[t, i + 1] = new_ll
            sampler.logpriors[t, i + 1] = new_lp
        else
            # Reject
            # Copy previous iteration
            sampler.chains[:, t, i + 1] = sampler.chains[:, t, i]
            sampler.loglikes[t, i + 1] = sampler.loglikes[t, i]
            sampler.logpriors[t, i + 1] = sampler.logpriors[t, i]
        end
    end
    sampler.i += 1
    sampler.n_mh_proposals += 1
end

logit(p) = log(p) - log(1 - p)
logistic(a) = 1 / (1 + exp(-a))
"""
    adapt!(sampler::ParallelSampler; k_p=1.0, decay_power=1.0)

Adapt the scale of each proposal distribution (isotropically) to bring the
acceptance rate towards `sampler.target_acc_rate`. `k_p` controls the rate of
adaptation, `decay_power` controls the rate at which to decay the adaptation,
as a power of `current_step / n_adapt`.
"""
function adapt!(sampler::ParallelSampler; k_p=1.0, decay_power=1.0)
    acc_rate = sampler.accs / sampler.n_mh_proposals
    setpoint = sampler.target_acc_rate
    current_error = setpoint .- acc_rate
    decay = max(0.0,
        min(1.0,
            ((sampler.n_adapt - sampler.i) / sampler.n_adapt)^decay_power
       )
    )
    # Keep scales bounded by 0 and 1:
    sampler.scales = logistic.(
        logit.(sampler.scales) - current_error * k_p * decay
    )
end

end # module ParallelTempering
