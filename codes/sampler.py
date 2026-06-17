import numpy as np
from progressbar import progressbar
import pickle
import sys

class Gibbs():
    def __init__(self, inits, target, scales):
        self.target = target
        self.num_components = len(inits)

        self.samplers = []

        t0 = lambda x: self.target(x, inits[1])
        self.samplers.append( pCN( inits[0], t0, scales[0] ) )

        t1 = lambda x: self.target(inits[0], x)
        self.samplers.append( MH( inits[1], t1, scales[1] ) )

    def warm_up(self, N_outer=1, N_inner=1):
        if(N_inner > 10):
            skip_len = int(N_inner/10)
        else:
            skip_len = 1
        for i in progressbar( range(N_outer) ):
            t0 = lambda x: self.target(x, self.samplers[1].current_sample)
            self.samplers[0].set_target( t0 )
            self.samplers[0].warm_up(N_inner, skip_len=skip_len)
            self.samplers[0].clear()

            t1 = lambda x: self.target(self.samplers[0].current_sample, x)
            self.samplers[1].set_target( t1 )
            self.samplers[1].warm_up(N_inner, skip_len=skip_len)
            self.samplers[1].clear()

    def sample(self, N_outer=1, N_inner=1, checkpoint=False, batch_size=None):
        if(checkpoint):
            self.batch_size = batch_size
            self.batch_idx = 0

        for i in progressbar( range(N_outer) ):
            t0 = lambda x: self.target(x, self.samplers[1].current_sample)
            self.samplers[0].set_target( t0 )
            self.samplers[0].multi_step(N_inner)

            t1 = lambda x: self.target(self.samplers[0].current_sample, x)
            self.samplers[1].set_target( t1 )
            self.samplers[1].multi_step(N_inner)

            if( checkpoint and ((i+1)%self.batch_size==0) ):
                self.save_checkpoint( path='./__samples_cache/checkpoint_{}.npz'.format(self.batch_idx) )
                self.dump()
                self.batch_idx += 1

    def get_samples(self):
        samples = []
        for i in range(self.num_components):
            samples.append( self.samplers[i].get_samples() )

        return samples

    def save_checkpoint(self, path='checkpoint.pickle'):
        states = []
        for i in range(self.num_components):
            states.append( self.samplers[i].get_state() )

        with open(path, 'wb') as handle:
            pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path):
        with open(path, 'rb') as handle:
            states = pickle.load(handle)

        for i in range(self.num_components):
            self.samplers[i].load_state( states[i] )

    def clear(self):
        for i in range(self.num_components):
            self.samplers[i].clear()

    def dump(self, path='./__samples_cache/'):
        if( path.endswith('/') ):
            pass
        else:
            path += '/'

        batch = []
        for i in range(self.num_components):
            batch.append( self.samplers[i].get_batch(self.batch_size) )

        np.savez(path+'_{}.npz'.format(self.batch_idx), batch=np.array(batch) )

class sampler():
    def __init__(self, x0, target, scale=1):
        self.dim = len(x0)
        self.target = target

        self.current_sample = x0
        self.current_target = self.target( x0 )

        self.samples = [ x0 ]
        self.acc = [1]

        self.scale = scale

    def sample(self, N, batch_size=None):
        if(batch_size):
            batch = 0

        for i in progressbar( range(N) ):
            self.step()

            if(batch_size):
                if((i+1)%batch_size == 1):
                    self.save_checkpoint('./checkpoints/check_{}.npz'.format(batch))
                    np.savez('__sampler_cache/samples_{}.npz'.format(batch), self.samples[-1-batch_size])
                    batch += 1

    def warm_up(self, N, skip_len=None):
        if(skip_len == None):
            self.skip_len = int( N/10 )
        else:
            self.skip_len = skip_len

        update_count = 0
        for i in range(N):
            self.step()
            if(  (i+1)%self.skip_len == 0 ):
                self.tune(update_count)
                update_count += 1

    def step(self):
        acc = self.update()

        self.acc.append(acc)
        self.samples.append(self.current_sample)

    def multi_step(self, N):
        acc = 0
        for i in range(N):
            self.update()

        self.samples.append(self.current_sample)

    def tune(self, update_count):
        hat_acc = np.mean(self.acc[-1-self.skip_len:])

        # d. compute new scaling parameter
        zeta = 1/np.sqrt(update_count+1)   # ensures that the variation of lambda(i) vanishes
        scale_temp = np.exp(np.log(self.scale) + zeta*(hat_acc-0.234))

        # update parameters
        self.scale = min(scale_temp, 1)

    def set_target(self, target):
        self.target = target

    def get_samples(self):
        return np.array(self.samples)

    def get_batch(self, batch_size):
        return np.array( self.samples[-batch_size:] )

    def load_checkpoint(self, path):
        self.clear()
        checkpoint = np.load(path)
        self.current_sample = checkpoint['current_sample']
        self.current_target = checkpoint['current_target']
        self.scale = checkpoint['scale']

    def load_state(self, state):
        self.clear()
        self.current_sample = state['current_sample']
        self.current_target = state['current_target']
        self.scale = state['scale']

    def clear(self):
        self.samples.clear()
        self.acc.clear()

class pCN(sampler):
    def __init__(self, x0, target, scale=1):
        super().__init__(x0, target, scale=scale)

    def update(self):
        xi = np.random.standard_normal(self.dim)
        x_star = np.sqrt( 1 - self.scale**2 ) * self.current_sample + self.scale*xi

        target_eval_star = self.target(x_star)

        ratio = target_eval_star - self.current_target
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_sample = x_star
            self.current_target = target_eval_star
            acc = 1

        return acc

    def save_checkpoint(self, path='checkpoint.npz'):
        np.savez(path, type='pCN', current_sample=self.current_sample, current_target=self.current_target, scale=self.scale)

    def get_state(self):
        return {'type': 'pCN', 'current_sample': self.current_sample, 'current_target': self.current_target, 'scale': self.scale}

class MH(sampler):
    def __init__(self, x0, target, scale=1):
        super().__init__(x0, target, scale=scale)

    def update(self):
        xi = np.random.standard_normal(self.dim)
        x_star = self.current_sample + self.scale*xi

        target_eval_star = self.target(x_star)

        ratio = target_eval_star - self.current_target
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_sample = x_star
            self.current_target = target_eval_star
            acc = 1

        return acc

    def save_checkpoint(self, path='checkpoint.npz'):
        np.savez(path, type='MH', current_sample=self.current_sample, current_target=self.current_target, scale=self.scale)

    def get_state(self):
        return {'type': 'MH', 'current_sample': self.current_sample, 'current_target': self.current_target, 'scale': self.scale}

class AIES_sampler():
    def __init__(self, x0, log_prior, log_like, N_AIES):
        self.N_walkers = x0.shape[0]
        self.dim = x0.shape[1]
        self.dim_AIES = N_AIES
        self.log_prior = log_prior
        self.log_like = log_like

        self.current_sample = x0
        self.current_log_prior = []
        for i in range(self.N_walkers):
            self.current_log_prior.append( self.log_prior( x0[i] ) )
        self.current_log_prior = np.array(self.current_log_prior)
        self.current_log_like = []
        for i in range(self.N_walkers):
            self.current_log_like.append( self.log_like( x0[i] ) )
        self.current_log_like = np.array(self.current_log_like)

        self.samples = [ x0 ]
        self.acc = [1]

        self.a = 1.3
        self.pCN_scale = 0.004

    def sample(self, N_sample):
        for i in progressbar(range(N_sample)):
            self.update()

    def sample_Z(self):
        y = np.random.uniform(1/np.sqrt(self.a), np.sqrt(self.a))
        return y*y

    def update(self):
        # AIES part
        current_sample = np.copy( self.current_sample )
        current_log_prior = np.copy( self.current_log_prior )
        current_log_like = np.copy( self.current_log_like )
        for idx in range(self.N_walkers):
            i = np.random.randint(self.N_walkers)
            j = np.random.randint(self.N_walkers-1)
            if(j >= i):
                j += 1

            Zi = self.sample_Z()
            x_star = np.copy( current_sample[j] )
            x_star[:self.dim_AIES] = x_star[:self.dim_AIES] +  Zi*(current_sample[i][:self.dim_AIES] - current_sample[j][:self.dim_AIES] )

            log_prior_star = self.log_prior(x_star)
            log_like_star = self.log_like(x_star)
            ratio = (self.dim_AIES-1)*np.log(Zi) + (log_like_star + log_prior_star) - ( current_log_like[i] + current_log_prior[i] )
           
            if np.log(np.random.rand()) < ratio:
                current_sample[i] = x_star
                current_log_prior[i] = log_prior_star
                current_log_like[i] = log_like_star

        self.samples.append( current_sample )
        self.current_sample = current_sample
        self.current_log_prior = current_log_prior
        self.current_log_like = current_log_like

        # pCN part
        current_sample = np.copy( self.current_sample )
        current_log_prior = np.copy( self.current_log_prior )
        current_log_like = np.copy( self.current_log_like )
        for idx in range(self.N_walkers):
            xi = np.random.standard_normal(self.dim - self.dim_AIES)
            x_star = current_sample[idx]
            x_star[self.dim_AIES:] = np.sqrt( 1 - self.pCN_scale**2 ) * current_sample[idx,self.dim_AIES:] + self.pCN_scale*xi

            log_like_star = self.log_like(x_star)
            log_prior_star = self.log_prior(x_star)
            ratio = log_like_star - current_log_like[idx]

            if np.log(np.random.rand()) < ratio:
                current_sample[idx] = x_star
                current_log_like[idx] = log_like_star
                current_log_prior[idx] = log_prior_star

        self.samples.append( current_sample )
        self.current_sample = current_sample
        self.current_log_prior = current_log_prior
        self.current_log_like = current_log_like

    def get_samples(self):
        return np.array(self.samples)



class AIES_sampler_warm_up():
    def __init__(self, x0, log_prior, log_like, N_AIES):
        self.N_walkers = x0.shape[0]
        self.dim = x0.shape[1]
        self.dim_AIES = N_AIES
        self.log_prior = log_prior
        self.log_like = log_like

        self.current_sample = x0
        self.current_log_prior = []
        for i in range(self.N_walkers):
            self.current_log_prior.append( self.log_prior( x0[i] ) )
        self.current_log_prior = np.array(self.current_log_prior)
        self.current_log_like = []
        for i in range(self.N_walkers):
            self.current_log_like.append( self.log_like( x0[i] ) )
        self.current_log_like = np.array(self.current_log_like)

        self.samples = [ x0 ]
       
        self.acc_AIES_per_walker = np.ones(self.N_walkers)
        self.acc_AIES_total = self.N_walkers
        self.acc_total_attempts = self.N_walkers
        self.acc_total_attempts_per_walker = np.ones(self.N_walkers)

        self.acc_pCN = np.ones(self.N_walkers)
        self.acc_pCN_total_attempts = self.N_walkers

        #self.a = 1.75
        #self.pCN_scale = 0.1
        self.a = 1.75
        self.pCN_scale = 0.05

        self.warm_up_phase = False
        self.acc_AIES_warm_up = 0
        self.acc_AIES_total_warm_up = 0
        self.acc_pCN_warm_up = 0
        self.acc_pCN_total_warm_up = 0

    def sample(self, N_sample):
        for i in progressbar(range(N_sample)):
            self.update()

    def warm_up(self, N_sample=5, num_windows=10):
        print('in warm up')
        sys.stdout.flush()
        self.warm_up_phase = True

        a_min, a_max = 1.2, 1.8         # stretch range
        b_min, b_max = 1e-4, 0.1       # pCN step

        a_target=0.38
        b_target=0.38

        eta0 = 0.25
        max_step = np.log(1.4)

        aies_acc_hist = []

        for t in range(1, num_windows+1):
            print('in window : ', t)
            sys.stdout.flush()
            self.acc_AIES_warm_up = 0
            self.acc_AIES_total_warm_up = 0
            self.acc_pCN_warm_up = 0
            self.acc_pCN_total_warm_up = 0

            for i in range(N_sample):
                self.update()

            ratio_a = self.acc_AIES_warm_up/self.acc_AIES_total_warm_up
            ratio_b = self.acc_pCN_warm_up/self.acc_pCN_total_warm_up

            aies_acc_hist.append(ratio_a)

            loga, eta_a = self.adapt_aies_fast(np.log(self.a), ratio_a, t, target=a_target, a_min=a_min, a_max=a_max, emergency=aies_acc_hist)
            self.a = np.exp(loga)
            logb, eta_b = self.rm_update(np.log(self.pCN_scale), ratio_b, b_target, t, gamma=0.15, t0=3, lo=b_min, hi=b_max, max_step=np.log(1.4))
            self.pCN_scale = np.exp(logb)

            print(f"[win {t}] AIES: acc={ratio_a:.3f} err={ratio_a - a_target:.3f} "f"eta={eta_a:.4f} Δlog={eta_a*(ratio_a-a_target):.4g} a→{self.a:.4f}")
            print(f"pCN : acc={ratio_b:.3f} err={ratio_b -b_target:.3f} " f"eta={eta_b:.4f} Δlog={eta_b*( ratio_b -b_target ):.4g} β→{self.pCN_scale:.4f}")
            print('-----------------------')
            sys.stdout.flush()

        self.warm_up_phase = False

    def sample_Z(self):
        y = np.random.uniform(1/np.sqrt(self.a), np.sqrt(self.a))
        return y*y

    def rm_update1(self,log_param, acc_rate, acc_target=0.38, t=1, gamma=0.05, t0=10,
                  lo=None, hi=None):
        """One Robbins–Monro update on log(param) with diminishing step size."""
        eta = gamma / (t + t0)
        #eta = 0.2 / (t - 5 + 3)
        #print('eta: ', eta)
        #eta = 0.1
       
        log_param_new = log_param + eta * (acc_rate - acc_target)
        if lo is not None or hi is not None:
            log_param_new = np.clip(log_param_new,
                                    -np.inf if lo is None else np.log(lo),
                                    +np.inf if hi is None else np.log(hi))
        return log_param_new

    def adapt_aies_fast(self, loga, acc_rate, t, *,
                        phase_switch=5,         # constant-eta for first 5 windows
                        eta0=0.25,              # constant learning rate (per window)
                        gamma=0.15, t0=3,       # diminishing after phase_switch
                        target=0.45,            # push acceptance up in warm-up
                        a_min=1.05, a_max=1.8,  # allow smaller a; cap upper bound
                        max_step=np.log(1.4),   # ±40% cap per window
                        emergency=None):        # pass last few accs for backoff
        err = acc_rate - target
        if t <= phase_switch:
            delta = eta0 * err
        else:
            eta = gamma / (t + t0)
            delta = eta * err
        # emergency backoff: if acceptance is persistently low, cut a by 10%
        if emergency is not None and len(emergency) >= 2 and all(x < 0.25 for x in emergency[-2:]):
            delta = min(delta, -0.1053605)  # ln(0.9) ≈ -0.105; i.e., a *= 0.9
        # cap multiplicative change
        delta = np.clip(delta, -max_step, max_step)
        loga_new = np.clip(loga + delta, np.log(a_min), np.log(a_max))
        return loga_new, eta0

    def rm_update(self, log_param, acc_rate, acc_target, t,
                  gamma=0.15, t0=3, lo=None, hi=None, max_step=np.log(1.4)):
        eta = gamma / (t + t0)                     # diminishing step size
        delta = eta * (acc_rate - acc_target)
        # cap multiplicative change per window
        delta = np.clip(delta, -max_step, max_step)
        out = log_param + delta
        if lo is not None: out = max(out, np.log(lo))
        if hi is not None: out = min(out, np.log(hi))
        return out, eta

    def adapt_const_eta(self, log_param, acc_rate, acc_target,
                        eta0=0.25, lo=None, hi=None, max_step=np.log(1.4)):
        # one window update on log-parameter
        delta = eta0 * (acc_rate - acc_target)
        # cap per-window multiplicative change (±40%)
        delta = np.clip(delta, -max_step, max_step)
        out = log_param + delta
        if lo is not None: out = max(out, np.log(lo))
        if hi is not None: out = min(out, np.log(hi))
        return out

    def update(self):
        dim_low = self.dim_AIES
        beta = self.pCN_scale
        N = self.N_walkers

        # local working copies
        X = np.copy(self.current_sample)
        lp = np.copy(self.current_log_prior)
        ll = np.copy(self.current_log_like)

        rng = np.random

        # ---------------------------
        # AIES: two-halves stretch on low block only
        # ---------------------------
        # split indices into two complementary halves
        A_idx = np.arange(0, N, 2)
        B_idx = np.arange(1, N, 2)

        # --- update A using frozen B ---
        B_ref = np.copy(X[B_idx])  # frozen reference half
        for ii, i in enumerate(A_idx):
            j = rng.randint(len(B_ref))             # partner from B
            Z = self.sample_Z()

            # propose: keep i's high-freq part, stretch only low block toward B_ref[j]
            x_prop = np.copy(X[i])
            x_prop[:dim_low] = B_ref[j, :dim_low] + Z * (X[i, :dim_low] - B_ref[j, :dim_low])

            lp_prop = self.log_prior(x_prop)
            ll_prop = self.log_like(x_prop)

            if np.isfinite(lp_prop) and np.isfinite(ll_prop):
                ratio = (dim_low - 1) * np.log(Z) + (ll_prop + lp_prop) - (ll[i] + lp[i])
            else:
                ratio = -np.inf  # reject non-finite proposals

            if np.log(rng.rand()) < ratio:
                X[i] = x_prop
                lp[i] = lp_prop
                ll[i] = ll_prop
                if self.warm_up_phase:
                    self.acc_AIES_warm_up += 1
                else:
                    self.acc_AIES_per_walker[i] += 1
                    self.acc_AIES_total += 1

            # count attempt
            if self.warm_up_phase:
                self.acc_AIES_total_warm_up += 1
            else:
                self.acc_total_attempts += 1
                self.acc_total_attempts_per_walker[i] += 1

        # --- update B using frozen (now-updated) A ---
        A_ref = np.copy(X[A_idx])  # freeze updated A as reference
        for ii, i in enumerate(B_idx):
            j = rng.randint(len(A_ref))             # partner from A
            Z = self.sample_Z()

            x_prop = np.copy(X[i])
            x_prop[:dim_low] = A_ref[j, :dim_low] + Z * (X[i, :dim_low] - A_ref[j, :dim_low])

            lp_prop = self.log_prior(x_prop)
            ll_prop = self.log_like(x_prop)

            if np.isfinite(lp_prop) and np.isfinite(ll_prop):
                ratio = (dim_low - 1) * np.log(Z) + (ll_prop + lp_prop) - (ll[i] + lp[i])
            else:
                ratio = -np.inf

            if np.log(rng.rand()) < ratio:
                X[i] = x_prop
                lp[i] = lp_prop
                ll[i] = ll_prop
                if self.warm_up_phase:
                    self.acc_AIES_warm_up += 1
                else:
                    self.acc_AIES_per_walker[i] += 1
                    self.acc_AIES_total += 1

            if self.warm_up_phase:
                self.acc_AIES_total_warm_up += 1
            else:
                self.acc_total_attempts += 1
                self.acc_total_attempts_per_walker[i] += 1

        # ---------------------------
        # pCN: high-frequency block
        # ---------------------------

        for idx in range(N):
            xi = rng.standard_normal(self.dim - dim_low)

            x_prop = np.copy(X[idx])  # COPY to avoid in-place mutation before accept/reject
            x_prop[dim_low:] = np.sqrt(1.0 - beta**2) * X[idx, dim_low:] + beta * xi

            lp_prop = self.log_prior(x_prop)
            ll_prop = self.log_like(x_prop)

            ratio = ll_prop - ll[idx]

            if not (np.isfinite(lp_prop) and np.isfinite(ll_prop)):
                ratio = -np.inf

            if np.log(rng.rand()) < ratio:
                X[idx] = x_prop
                lp[idx] = lp_prop
                ll[idx] = ll_prop
                if self.warm_up_phase:
                    self.acc_pCN_warm_up += 1
                else:
                    self.acc_pCN[idx] += 1

            if self.warm_up_phase:
                self.acc_pCN_total_warm_up += 1
            else:
                self.acc_pCN_total_attempts += 1

        # ---------------------------
        # finalize sweep: commit and record once
        # ---------------------------
        self.current_sample = X
        self.current_log_prior = lp
        self.current_log_like = ll
        self.samples.append(np.copy(X))

    def print_summary(self):
        print('total attempts = ', self.acc_total_attempts)
        print('AIES total acceptance = ', self.acc_AIES_total)
        print('AIES average ratio = ', self.acc_AIES_total/self.acc_total_attempts)
        print('AIES average ratio per walker: ')
        print(self.acc_AIES_per_walker/self.acc_total_attempts_per_walker)

        print('total attempts = ', self.acc_pCN_total_attempts)
        print('pCN total acceptance = ', np.sum(self.acc_pCN) )
        print('pCN average ratio = ', np.sum(self.acc_pCN)/self.acc_pCN_total_attempts)
        print('pCN average ratio per walker: ')
        print(self.acc_pCN * self.N_walkers / self.acc_pCN_total_attempts )





    def get_samples(self):
        return np.array(self.samples)

