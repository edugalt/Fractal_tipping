##############################################################
## Section 0: The required packages
##############################################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress
from joblib import Parallel, delayed

##############################################################
## Section 1.1: The maps
# PUT YOUR MAPS IN THIS SECTION
##############################################################
def modified_tent(x, u, d, s2=None, **kwargs):
    """
    The piecewise-linear (modified tent) map in the paper (example 1)
    """
    if (s2 is None):
        s2 = u/2
    x1 = 1 + d + 1/3 * (s2-1)
    x2 = 1 + d + 2/3 * (s2-1)
    a1 = (x2-d)/(x2-x1)
    s1 = (a1 * (x1-d) + u)/(a1 + u)
    b1 = a1*s1 - u*(1-s1)
    a2=1/4
    b2 = b1 + s2 * (a2-a1)
    if (x-d < s1):
        return u*min(x-d, 1-(x-d)) + d
    elif (s1 <= x-d < s2):
        return a1 * (x-d) - b1 + d
    else:
        return a2 * (x-d) - b2 + d

def modified_tent_intersections(u, d, s2=None, **kwargs):
    """
    Return x^{u3} and x^* of the modified tent map
    """
    if (not s2):
        s2 = u/2
    x1 = 1 + d + 1/3 * (s2-1)
    x2 = 1 + d + 2/3 * (s2-1)
    a1 = (x2-d)/(x2-x1)
    s1 = (a1 * (x1-d) + u)/(a1 + u)
    b1 = a1*s1 - u*(1-s1)
    a2=1/4
    b2 = b1 + s2 * (a2-a1)
    attractor = (-a2*d -b2 + d)/(1-a2)
    return x2, attractor

def tent(x, u, **kwargs):
    return u * min(x, 1-x)

def logistic(x, r, **kwargs):
    return r * x * (1-x)

##############################################################
## Section 1.2: Drifting functions
# PUT YOUR DRIFTING FUNCTIONS IN THIS SECTION
##############################################################
def tanh_change(n, rate, s_ini = 0, slope = 4, **kwargs):
    """
    tanh change used in Kiers paper, the parameter must step at s=0 for some step n
    """
    n0 = int(s_ini/rate)
    return slope * np.tanh((n+n0) * rate)

def tanh_change_mine(n, rate, slope=4, s_ini = 0, **kwargs):
    """
    Tanh drift function defined by me, the parameter doesn't necessarily step at s=0
    """
    n0 = s_ini/rate
    return slope * np.tanh((n+n0) * rate)

def linear_change_inf(n, rate, **kwargs):
    """
    Infinite linear drift
    """
    return n*rate

def linear_change_finite(n, rate, cap, s0=0, **kwargs):
    """
    Finite linear drift to the parameter 'cap'.
    """
    if (s0 + n*rate <= cap):
        return s0 + n*rate
    else:
        return cap

def step_function(n, rate, amt, s_thres=1, **kwargs):
    """
    When s goes past s_thres, the function suddenly changed
    the amount of 'amt'.
    """
    if (n * rate < s_thres):
        return 0
    else:
        return amt

##############################################################
## Section 1.3: Converging conditions
# PUT YOUR CONVERGING CONDITION IN THIS SECTION
##############################################################

def converging_condition_default(x, fixed_pt, tol=1e-5, **kwargs):
    """
    Converge if it's close to the fixed point within the distance of tol.
    """
    return abs(x - fixed_pt) < tol

def converging_condition_pullback(x, fixed_pt, x_thres, tol=1e-5, **kwargs):
    """
    Convergence test in the pullback case. Target for drifting functions that changes slowly
    at first and then fast. Since we started at the fixed point, the conventional condition will
    output Track at the start. Hence, there is one more condition that x needs to cross x_thres. 
    """
    return abs(x - fixed_pt) < tol and x > x_thres

def converging_condition_step_function(x, fixed_pt, d, d0=0, smallest_d_change = 1, tol=1e-5, **kwargs):
    """
    The step function is not changing except at the point of jump. Hence, an additional condition
    is needed to test that the parameter has jumped.
    """
    return abs(x - fixed_pt) < tol and d-d0 > smallest_d_change

##############################################################
## Section 1.4: map visualisation 
##############################################################
def visualise_fx(func, plot_diag = True, plot_ffx=False, x_min = 0, x_max = 20, npts = 1000, **kwargs):
    """
    Visualise f(x). With the option of visualising f(f(x)) and y=x
    """
    x_values = np.linspace(x_min, x_max, npts)
    fx_values = [func(x, **kwargs) for x in x_values]

    plt.figure(figsize=(5, 5))
    plt.plot(x_values, fx_values, label=r'$f(x)$')

    if (plot_diag):
        plt.plot(x_values, x_values, label=r'$y = x$', linestyle='--')

    if plot_ffx:
        ffx_values = [func(fx, **kwargs) if fx is not None else None for fx in fx_values]
        plt.plot(x_values, ffx_values, label=r'$f(f(x))$')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.title(f'Visualisation of {func.__name__}')
    plt.legend()
    plt.xlim([x_min, x_max])
    plt.grid(True)
    plt.show()

##############################################################
## Section 1.5: visualisation of drifting functions
##############################################################
def plot_parameter_drift(evolving_func, evolving_param, rate, smax = 20, **kwargs):
    '''
    Plot the parameter drift of one parameter versus s (time).
    '''
    param_path = []
    ss = []
    max_steps = smax/rate + 1
    s_ini = kwargs.get('s_ini', 0)
    if (evolving_func == tanh_change):
        n0 = int(s_ini/rate)
    else:
        n0 = s_ini/rate

    for i in range(int(max_steps)):
        ss.append((i + n0) * rate)
        param_path.append(evolving_func(n=i, rate=rate, **kwargs))

    plt.figure(figsize=(4, 4))
    plt.plot(ss, param_path, linestyle = '-', marker = 'o')
    plt.xlabel(r'$s=rn$')
    plt.ylabel(evolving_param)
    plt.title(f'Visualisation of the parameter drift at rate {rate}')
    plt.tight_layout()
    plt.show()

def plot_parameter_drift_2D(evolving_funcs, evolving_params, rate, smax = 20, **kwargs):
    '''
    Visualise the simutaneous drift of two parameters.
    '''
    
    if (len(evolving_funcs) != 2 or len(evolving_params) != 2):
        raise ValueError("Need two functions and two evolving parameters.")

    param1_path = []
    param2_path = []
    ss = []

    max_steps1 = smax/rate + 1
    evolving_func1 = evolving_funcs[0]
    evolving_func2 = evolving_funcs[1]

    for i in range(int(max_steps1)):
        ss.append(i * rate)
        param1_path.append(evolving_func1(n=i, rate=rate, **kwargs))
        param2_path.append(evolving_func2(n=i, rate=rate, **kwargs))

    plt.figure(figsize=(4, 4))
    plt.plot(param1_path, param2_path, linestyle = '-', marker = 'o')
    plt.xlabel(evolving_params[0])
    plt.ylabel(evolving_params[1])
    plt.title(f'Visualisation of 2 parameter drifts at rate {rate}')
    plt.tight_layout()
    plt.show()

##############################################################
## Section 2.1: Basic functions
##############################################################

def find_one_fixed_point_default(func, x0, steps = 100, tol=1e-8, **kwargs):
    """
    Find a fixed point by simple forward iteration.
    """
    x = x0
    for _ in range(steps):
        x_new = func(x, **kwargs)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None

def evolve(func, x, xmin_escape, steps=100, **kwargs):
    """
    Iterate a map forward for a fixed number of steps with an escape cutoff.
    """
    for _ in range(steps):
        x = func(x, **kwargs)
        if x is None or x < xmin_escape:
            break
    
    return x

def plot_trajectory(trajectory):
    """
    Plot an iteration trajectory against step index.
    """
    plot_fontsize = 12  # Global font size
    steps = np.arange(1, len(trajectory) + 1)
    plt.figure(figsize=(6, 5))
    plt.plot(steps, trajectory)
    plt.xlabel('$steps$', fontsize=plot_fontsize)
    plt.ylabel('$x$', fontsize=plot_fontsize)
    plt.title('Plot of x versus steps', fontsize=plot_fontsize)
    plt.xticks(fontsize=plot_fontsize)
    plt.yticks(fontsize=plot_fontsize)
    plt.show()

def evolve_with_trajectory(func, x, xmin_escape, steps, plot = False, **kwargs):
    """
    Iterate a map forward and record the full trajectory.
    """
    trajectory = [x]
        
    for _ in range(steps):
        x = func(x, **kwargs)
        trajectory.append(x)

        if x is None or x < xmin_escape:
            break

    if plot:
        plot_trajectory(trajectory)

    return x, trajectory

def get_escape_step(func, x, xmin_escape, max_steps, attractors, tol=1e-5, **kwargs):
    '''
    Get the step when the orbit escapes. Return -1 if it falls to an attractor.
    '''
    steps = 0
    while (0 <= steps < max_steps):
        steps += 1
        x = func(x, **kwargs)
        if x is None or x < xmin_escape:
            break
        
        if (attractors is not None):
            for attractor in attractors:
                if (abs(x - attractor) < tol):
                    steps = -1
                    break
    
    if (steps == max_steps):
        raise ValueError('Max steps is not sufficient')
    else:
        return steps

def evolve_tipping(
    func,
    evolving_funcs,
    evolving_params,
    rate,
    xmin_escape,
    evolving_range=10,
    fixed_pt_func=find_one_fixed_point_default,
    fixed_pt_x0=None,
    n0 = 0,
    **kwargs
):
    """
    Evolve a 1D map under drifting parameter(s).

    """
    ns = np.arange(0, int(evolving_range / rate) + 1)
    xs, ss = [], []

    if len(evolving_params) != len(evolving_funcs):
        raise ValueError("params and evolving_funcs must have the same length.")

    for n in ns:
        for param, evolving_func in zip(evolving_params, evolving_funcs):
            kwargs[param] = evolving_func(n, rate=rate, **kwargs)

        if n == 0:
            if (func == modified_tent):
                _, x = modified_tent_intersections(**kwargs)
            else:
                x = fixed_pt_func(func=func, x0=fixed_pt_x0, **kwargs)
        else:
            x = evolve(func=func, x=x, xmin_escape=xmin_escape, steps=1, **kwargs)

        ss.append((n+n0) * rate)
        xs.append(x)

        if x is None or x < xmin_escape:
            break

    return xs, ss

def evolve_tipping_until_escape(
    func,
    evolving_funcs,
    evolving_params,
    rate,
    xmin_escape,
    fixed_pt_func=find_one_fixed_point_default,
    fixed_pt_x0=None,
    max_steps=1000,
    converging_condition=converging_condition_default,
    return_trajectory=False,
    **kwargs
):
    """
    Evolve a drifting system until it either escapes (x < xmin_escape / None) or converges.

    We assume that max_steps is unreachable. If reached, the trajectory must 
    bt at an unreported fixed point. You can modify this if you don't assume so.

    Returns n >= 0 for escape time (in steps), or -1 if converged.
    Optionally returns the trajectory (xs, ss) as well.
    """
    if len(evolving_params) != len(evolving_funcs):
        raise ValueError("params and evolving_funcs must have the same length.")

    xs, ss = [], []

    for n in range(max_steps):
        for param, evolving_func in zip(evolving_params, evolving_funcs):
            kwargs[param] = evolving_func(n, rate=rate, **kwargs)

        # Initialisation
        if n == 0:
            if (func == modified_tent):
                _, x = modified_tent_intersections(**kwargs)
            else:
                x = fixed_pt_func(func=func, x0=fixed_pt_x0, **kwargs)
        else:
            # Evolving
            x = evolve(func=func, x=x, xmin_escape=xmin_escape, steps=1, **kwargs)

        ss.append(n * rate)
        xs.append(x)

        # Escape condition
        if x is None or x < xmin_escape:
            if return_trajectory:
                return xs, ss, n
            return n

        # Checking convergence
        if n > 0:
            if (func == modified_tent):
                _, fixed_pt = modified_tent_intersections(**kwargs)
            else:
                fixed_pt = fixed_pt_func(func=func, x0=fixed_pt_x0, **kwargs)

            if fixed_pt is not None and converging_condition(x, fixed_pt, **kwargs):
                # converged
                break

    n_out = -1
    
    if return_trajectory:
        return xs, ss, n_out
    return n_out

def find_critical_rate(func, rate_low_bound, rate_high_bound, iterations=10, **kwargs):
    """
    Estimate a critical drift rate separating tracking vs escape. Can only find one critical rate.
    
    Uses a binary search on ``rate`` over the interval
    ``[rate_low_bound, rate_high_bound]``. At each candidate rate, runs
    :func:`evolve_tipping_until_escape`:
    
    - if it returns ``-1`` (converged / tracking), the lower bound is increased;
    - otherwise (escape), the upper bound is decreased.
    """
    low, high = rate_low_bound, rate_high_bound
    rate = (low + high) / 2

    for _ in range(iterations):
        n_steps = evolve_tipping_until_escape(func=func, rate=rate, **kwargs)

        if (n_steps < 0):
            low = rate
        else:
            high = rate

        rate = (low + high) / 2

    return rate

##############################################################
## Section 2.2: Visualisation of tipping trajectories
##############################################################
def plot_tipping(func, rates, evolving_range = 20, plot_fixed_pt = True, show = True, tracking_rate = 0.01, **kwargs):
    """
    Plot trajectories under parameter drift for a range of rates. Optionally plot the trajectory of the fixed point.
    """
    plt.figure(figsize=(4,4))

    for rate in rates:
        path, ss = evolve_tipping(func=func, rate=rate, evolving_range=evolving_range, **kwargs)
        plt.plot(ss, path, '-o', mew=1.5, label = 'rate = {:.3f}'.format(np.float64(rate)))

    attractor_path, ss = evolve_tipping(func = func, rate = tracking_rate, evolving_range = evolving_range, **kwargs)

    if (plot_fixed_pt):
        plt.plot(ss, attractor_path, label = r'$x^*$', color = 'orange')

    fontsize = 12
    plt.xlabel(f'$s=rn$', fontsize=fontsize)
    plt.ylabel(f'$x$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend()

    if (show):
        plt.show()

def plot_tipping_modified_tent(rates, evolving_funcs, evolving_range = 20, func = modified_tent, plot_fixed_pt = True, 
                plot_reinjection = False, plot_tent = False, show = True, mode = 0, tracking_rate = 0.1, **kwargs):
    '''
    Plotting tracking and tipping trajectories specifically for the modified tent map. Reproducing figure 3.
    '''

    func = modified_tent
    plt.figure(figsize=(5,5), dpi = 300)
    s_ini = kwargs.get('s_ini', 0)

    if (mode == 0): # The default
        for rate in rates:
            if (evolving_funcs[0] == tanh_change):
                n0 = int(s_ini/rate)
            else:
                n0 = s_ini/rate
            path, ss = evolve_tipping(func=func, rate=rate, evolving_range=evolving_range, evolving_funcs=evolving_funcs, n0 = n0, **kwargs)
            plt.plot(ss, path, '-o', mew=1.5, label = 'rate = {:.3f}'.format(np.float64(rate)))
    elif (mode == 1): # fractal region
            # tracking rate
            rate = rates[0]
            
            if (evolving_funcs[0] == tanh_change):
                n0 = int(s_ini/rate)
            else:
                n0 = s_ini/rate

            path, ss = evolve_tipping(func=func, rate=rate, evolving_range=evolving_range, evolving_funcs=evolving_funcs, n0 = n0, **kwargs)
            plt.plot(ss, path, '-', color='blue', zorder=1)
            plt.scatter(ss, path, s=30, color='blue',
                        zorder=10,
                        label=rf'$x_n, r_1={rate:.2f}$')

            # escaping rate
            rate = rates[1]

            if (evolving_funcs[0] == tanh_change):
                n0 = int(s_ini/rate)
            else:
                n0 = s_ini/rate

            path, ss = evolve_tipping(func=func, rate=rate, evolving_range=evolving_range, evolving_funcs=evolving_funcs, n0 = n0, **kwargs)
            color = 'orange'
            plt.plot(ss, path, '-', color=color, zorder=1)

            plt.scatter(ss, path, s=30, facecolor=color, edgecolor=color,
                        marker='s', linewidth=1.2,
                        zorder=5,
                        label=r'$x_n, r_2 = r_1 + \varepsilon$')
    elif (mode == 2): # r < r_c1
        rate = rates[0]

        if (evolving_funcs[0] == tanh_change):
            n0 = int(s_ini/rate)
        else:
            n0 = s_ini/rate

        path, ss = evolve_tipping(func=func, rate=rate, evolving_range=evolving_range, evolving_funcs=evolving_funcs, n0 = n0,  **kwargs)
        plt.plot(ss, path, '-o', mfc='blue', mew=1, label = rf'$x_n, r = {rate:.2f} < r_{{c1}}$')
    elif (mode == 3): # r > r_c2
        color = 'orange'
        rate = rates[0]

        if (evolving_funcs[0] == tanh_change):
            n0 = int(s_ini/rate)
        else:
            n0 = s_ini/rate

        path, ss = evolve_tipping(func=func, rate=rate, evolving_range=evolving_range, evolving_funcs=evolving_funcs, n0 = n0, **kwargs)
        plt.plot(ss, path, marker = 's', color = color, mfc=color, mew=1, label = rf'$x_n, r = {rate:.2f} > r_{{c2}}$')


    evolving_func = evolving_funcs[0]
    d0 = evolving_func(n = 0, rate = 1, **kwargs)
    right_reinject, attractor = modified_tent_intersections(d=d0, **kwargs)

    if (evolving_funcs[0] == tanh_change):
        n0 = int(s_ini/tracking_rate)
    else:
        n0 = s_ini/tracking_rate

    attractor_path, ss = evolve_tipping(func = func, gx_func=func, rate = tracking_rate, evolving_range=evolving_range,
                                           evolving_funcs=evolving_funcs, n0 = n0, **kwargs)

    if (plot_fixed_pt):
        plt.plot(ss, attractor_path, label = r'$x^*$', color = 'black')

    if (plot_reinjection):
        dist = attractor - right_reinject
        plt.plot(ss, attractor_path - dist, color = 'black', linestyle = ':', label = r'$x^{u,3}$')

    if (plot_tent):
        plt.plot(ss, attractor_path - attractor + d0, color = 'black', linestyle = '--', label = r'$x^{u,1}$')

    fontsize = 12
    plt.xlabel(f'$s=rn$', fontsize=fontsize)
    plt.ylabel(f'$x$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim([-3, 6]) # can be adjusted
    plt.xlim([-12, 20]) # [s_ini, s_ini + evolving_range]
    leg = plt.legend(loc = 'upper left', fontsize=fontsize-1)
    leg.get_frame().set_alpha(0)          # 30% opacity

    if (show):
        plt.show()

##############################################################
## Section 3.1: Plot the first fractal.
##############################################################

def visualise_first_fractal_parallel(
    func,
    xmin=-0.1,
    xmax=1.5,
    npts=1000,
    attractor=True,
    inplace=False,
    n_jobs=-1,
    prefer="threads", 
    mark_symbol = False,
    **kwargs
):  
    '''
    Visualise the first fractal by parallel computation.
    '''

    grid = np.linspace(xmin, xmax, npts)

    def _keep(pt):
        val = evolve(x=pt, func=func, **kwargs)
        return (val > 0) if attractor else (val < 0)

    keep_mask = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_keep)(pt) for pt in grid
    )
    keep_mask = np.asarray(keep_mask, dtype=bool)
    pts = grid[keep_mask]

    if not inplace:
        plt.figure(figsize=(5, 1.7), dpi=600)

    ax = plt.gca()
    right_reinject, _ = modified_tent_intersections(**kwargs)
    ax.axvspan(xmin=xmin, xmax=right_reinject, facecolor="yellow")

    if pts.size > 0:
        plt.hist(pts, bins=len(pts), color="blue", label="Fractal set")
    else:
        # still draw an empty plot gracefully
        plt.hist([], bins=1, color="blue", label="Fractal set")

    plt.xlabel("$x$")
    plt.axvline(0, color='black', linestyle='--', linewidth=1)

    if (func == modified_tent and mark_symbol):
        plt.text(
            0, 1.1, r"$x^{u,1}$",
            ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )

        plt.axvline(right_reinject, color='white', linestyle='--', linewidth=1)
        plt.text(
            right_reinject, 1.1, r"$x^{u,3}$",
            ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )

    plt.xlim([xmin, xmax])
    plt.ylim([0, 1])
    plt.yticks([])
    plt.tight_layout()

    if not inplace:
        plt.show()

##############################################################
# Section 3.2: the uncertainty algorithm for fractal 1
##############################################################

def bootstrapping_slope_confid_interval(slope, log_eps, log_feps, boot_size, repeats = 2000, percent = 95):
    '''
    Compute the confidence interval using bootstrapping
    '''
    slopes = []
    indexes = np.arange(len(log_feps))
    
    for i in range(repeats):
        drawn_indexes = np.random.choice(indexes, size = boot_size)
        boot_slope, *_ = linregress(log_eps[drawn_indexes], log_feps[drawn_indexes])
        slopes.append(boot_slope)
    
    # 95% bootstrap percentile CI
    percent_cut = (100 - percent)/2
    lower = np.percentile(slopes, percent_cut)
    upper = np.percentile(slopes, 100 - percent_cut)
    confid_int = max(upper - slope, slope - lower)
    return confid_int

def loglog_linear_fit(
    epsilons,
    feps,
    feps_err=None,
    fit_truncation_order=-np.inf,
    num_eps=None,
    plot=False,
    title='',
    return_all_info=False,
):
    """
    The log-log linear fit for the uncertainty algorithms.
    """

    # Mask in log10 space
    log_eps_all = np.log10(epsilons)
    mask = log_eps_all < fit_truncation_order

    log_eps  = log_eps_all[mask]
    log_feps = np.log10(feps)[mask]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_eps, log_feps)

    if num_eps is None:
        num_eps = log_eps.size

    confid_int = bootstrapping_slope_confid_interval(
        slope=slope,
        log_eps=log_eps,
        log_feps=log_feps,
        boot_size=int(num_eps),
    )

    fit_line = 10**intercept * epsilons**slope

    # Plot
    if plot:
        plt.figure(figsize=(6, 5))

        if feps_err is not None:
            plt.errorbar(
                epsilons, feps, yerr=feps_err,
                fmt='o', capsize=3, label='Data'
            )
        else:
            plt.loglog(epsilons, feps, 'o', label='Data')

        plt.loglog(
            epsilons, fit_line,
            label=f'Fit: α = {slope:.3f} ± {confid_int:.3f}'
        )

        plt.xlabel('$\\varepsilon$')
        plt.ylabel('$f(\\varepsilon)$')
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    if return_all_info:
        return epsilons, feps, feps_err, fit_line, slope, confid_int
    else:
        return slope

def uncertainty_algorithm_parallel(
    func,
    threshold=100,
    plot=True,
    verbose=True,
    xmin=0,
    xmax=1,
    min_eps=1e-10,
    max_eps=1e-4,
    num_eps=20,
    return_all_info=False,
    attractors=None,
    fit_truncation_order=-6,
    n_jobs=-1,
    base_seed=0,
    prefer = 'processes',
    **kwargs
):
    '''
    The uncertainty algorithm in the phase space.
    '''
    epsilons = np.geomspace(min_eps, max_eps, num_eps)

    def _one_eps(idx, eps):
        rng = np.random.default_rng(base_seed + idx)

        trials = 0
        uncertains = 0

        while uncertains < threshold:
            trials += 1
            x1 = rng.uniform(xmin, xmax)
            x2 = x1 + eps if (x1 + eps) <= xmax else x1 - eps

            escape1 = get_escape_step(func=func, x=x1, attractors=attractors, **kwargs)
            escape2 = get_escape_step(func=func, x=x2, attractors=attractors, **kwargs)

            if attractors is None:
                if escape1 != escape2:
                    uncertains += 1
            else:
                if (escape1 < 0 and escape2 > 0) or (escape1 > 0 and escape2 < 0):
                    uncertains += 1

        p_hat = threshold / trials
        sigma_p = p_hat * np.sqrt(max(0.0, (1.0 - p_hat)) / threshold)
        return idx, p_hat, sigma_p

    if n_jobs == 1:
        if verbose:
            print("Running uncertainty algorithm (serial with tqdm)...")

        iterator = enumerate(epsilons)
        if verbose:
            iterator = tqdm(iterator, total=len(epsilons), desc="ε loop")

        results = [_one_eps(i, eps) for i, eps in iterator]
    else:
        if verbose:
            print(f"Running uncertainty_algorithm_in parallel (n_jobs={n_jobs}, prefer={prefer})...")

        results = Parallel(n_jobs=n_jobs, prefer = prefer)(
            delayed(_one_eps)(i, eps) for i, eps in enumerate(epsilons)
        )

    # restore order
    results.sort(key=lambda t: t[0])

    feps = np.asarray([t[1] for t in results], dtype=float)
    feps_err = np.asarray([t[2] for t in results], dtype=float)

    return loglog_linear_fit(
        epsilons,
        feps,
        feps_err=feps_err,
        fit_truncation_order=fit_truncation_order,
        plot=plot,
        title='f(ε) vs ε for the first fractal',
        return_all_info=return_all_info,
    )

##############################################################
## Section 4.1: Plot the second fractal
##############################################################

def visualise_second_fractal_parallel(
    func=modified_tent,
    d_low=0.2,
    d_high=4,
    npts=1000,
    n_jobs=-1,
    prefer="threads",
    mark_symbol = True, 
    **kwargs
):
    '''
    Visualise the second fractal by parallel computation.
    '''

    grid = np.linspace(d_low, d_high, npts)

    def _is_tracking(delta_d):
        steps = evolve_tipping_until_escape(func=func, rate=0.1, evolving_funcs=[step_function],
                                            converging_condition=converging_condition_step_function,
                                            amt = delta_d, **kwargs)

        # True = tracking, False = escaping
        return (steps == -1)

    tracking_mask = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_is_tracking)(delta_d) for delta_d in grid
    )
    tracking_mask = np.asarray(tracking_mask, dtype=bool)

    tracking_ds = grid[tracking_mask]
    escaping_ds = grid[~tracking_mask]
    
    if escaping_ds.size == 0 and tracking_ds.size > 0:
        print("No escaping points. Last tracking d =", tracking_ds[-1])
    elif tracking_ds.size == 0 and escaping_ds.size > 0:
        print("No tracking points. First escaping d =", escaping_ds[0])

    plt.figure(figsize=(6, 1), dpi=600)

    if tracking_ds.size > 0:
        plt.hist(tracking_ds, bins=len(tracking_ds), color="blue", label="Tracking")
    else:
        plt.hist([], bins=1, color="blue", label="Tracking")

    if escaping_ds.size > 0:
        plt.hist(escaping_ds, bins=len(escaping_ds), color="yellow", label="Escaping")
    else:
        plt.hist([], bins=1, color="yellow", label="Escaping")

    if (func == modified_tent and mark_symbol):
        right_reinject, attractor = modified_tent_intersections(d=0, **kwargs)

        plt.axvline(attractor, color='black', linestyle='--', linewidth=1)
        plt.text(
            attractor, 1.1, r"$\Delta \lambda_{c2}$",
            ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )

        plt.axvline(attractor - right_reinject, color='white', linestyle='--', linewidth=1)
        plt.text(
            attractor - right_reinject, 1.1, r"$\Delta \lambda_{c1}$",
            ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )

    plt.ylim([0, 1])
    plt.xlabel(r"$\Delta \lambda$")
    plt.yticks([])
    plt.xlim([d_low, d_high])
    plt.show()

##############################################################
## Section 4.2: the uncertainty algorithm for fractal 2
##############################################################

def uncertainty_algorithm_for_delta_d_parallel(
    func,
    dmin, dmax,
    threshold=100,
    plot=True,
    verbose=True,
    min_eps=1e-10,
    max_eps=1e-6,
    num_eps=20,
    return_all_info=False,
    have_attractor=True,
    n_jobs=-1,
    prefer="processes",
    base_seed=0,
    fit_truncation_order = 0,
    **kwargs
):
    """
    The uncertainty algorithm for fractal 2.
    """

    epsilons = np.geomspace(min_eps, max_eps, num_eps)

    def _one_eps(idx, eps):
        rng = np.random.default_rng(base_seed + idx)

        trials = 0
        uncertains = 0

        while uncertains < threshold:
            trials += 1
            d1 = rng.uniform(dmin, dmax)
            d2 = d1 + eps if (d1 + eps) <= dmax else d1 - eps
            
            step1 = evolve_tipping_until_escape(func = func, evolving_funcs=[step_function], rate = 1, amt = d1,
                                                converging_condition=converging_condition_step_function, **kwargs)

            step2 = evolve_tipping_until_escape(func = func, evolving_funcs=[step_function], rate = 1, amt = d2,
                                                converging_condition=converging_condition_step_function, **kwargs)

            if have_attractor:
                hit = ((step1 < 0 and step2 > 0) or (step1 > 0 and step2 < 0))
            else:
                hit = (step1 != step2)

            if hit:
                uncertains += 1

        p_hat = threshold / trials
        sigma_p = (np.sqrt(threshold * (1 - p_hat))) / trials

        return idx, p_hat, sigma_p

    # Serial (with tqdm) when n_jobs == 1
    if n_jobs == 1:
        it = enumerate(epsilons)
        if verbose:
            it = tqdm(it, total=len(epsilons), desc="Progress")
        results = [_one_eps(i, eps) for i, eps in it]
    else:
        if verbose:
            print(f"Running uncertainty_algorithm_for_delta_d in parallel (n_jobs={n_jobs}, prefer={prefer})...")
        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_one_eps)(i, eps) for i, eps in enumerate(epsilons)
        )

    results.sort(key=lambda t: t[0])
    
    feps = np.asarray([t[1] for t in results], dtype=float)
    feps_err = np.asarray([t[2] for t in results], dtype=float)

    return loglog_linear_fit(
        epsilons,
        feps,
        feps_err=feps_err,
        fit_truncation_order=fit_truncation_order,
        plot=plot,
        title='f(ε) vs ε for the second fractal',
        return_all_info=return_all_info,
    )

##############################################################
## Section 5.1: Plot the third fractal
##############################################################

def visualise_third_fractal_parallel(
    func,
    rate_lower_bound,
    rate_upper_bound,
    npts=1000,
    n_jobs=-1,
    prefer="threads",
    mark_rcs = True,
    **kwargs
):
    '''
    Visualise the third fractal by parallel computation.

    '''
    grid = np.linspace(rate_lower_bound, rate_upper_bound, npts)

    def _is_tracking(rate):
        steps = evolve_tipping_until_escape(func = func, rate=rate, **kwargs)
        return (steps == -1)  # True = tracking, False = escaping

    tracking_mask = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_is_tracking)(rate) for rate in grid
    )
    tracking_mask = np.asarray(tracking_mask, dtype=bool)

    tracking_rates = grid[tracking_mask]
    escaping_rates = grid[~tracking_mask]

    # Compute critical rates (guard empties)
    r_c1 = escaping_rates[0] if escaping_rates.size > 0 else None
    r_c2 = tracking_rates[-1] if tracking_rates.size > 0 else None

    plt.figure(figsize=(5, 1.7), dpi=600)

    if tracking_rates.size > 0:
        plt.hist(tracking_rates, bins=len(tracking_rates), color="blue", label="Tracking")
    else:
        plt.hist([], bins=1, color="blue", label="Tracking")

    if escaping_rates.size > 0:
        plt.hist(escaping_rates, bins=len(escaping_rates), color="yellow", label="Escaping")
    else:
        plt.hist([], bins=1, color="yellow", label="Escaping")

    plt.ylim([0, 1])
    plt.yticks([])
    plt.xlim([rate_lower_bound, rate_upper_bound])
    plt.xlabel("$r$")

    if (mark_rcs):
        if r_c1 is not None:
            plt.axvline(r_c1, color='white', linestyle='--', linewidth=1)
            plt.text(
                r_c1, 1.1, r"$r_{c1}$",
                ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
            )

        if r_c2 is not None:
            plt.axvline(r_c2, color='black', linestyle='--', linewidth=1)
            plt.text(
                r_c2, 1.1, r"$r_{c2}$",
                ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
            )

    plt.tight_layout()
    plt.show()

    print("r_c1 =", r_c1, "     r_c2 =", r_c2)

##############################################################
## Section 5.2: the uncertainty algorithm for fractal 3.
##############################################################

def get_rate_range(
    func,
    rate_lower_bound,
    rate_upper_bound,
    npts=1000,
    n_jobs=-1,
    prefer="threads",
    mark_rcs = True,
    **kwargs
):
    '''
    Get r_{c1} and r_{c2}
    '''

    grid = np.linspace(rate_lower_bound, rate_upper_bound, npts)

    def _is_tracking(rate):
        steps = evolve_tipping_until_escape(func = func, rate=rate, **kwargs)
        return (steps == -1)  # True = tracking, False = escaping

    tracking_mask = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_is_tracking)(rate) for rate in grid
    )
    tracking_mask = np.asarray(tracking_mask, dtype=bool)

    tracking_rates = grid[tracking_mask]
    escaping_rates = grid[~tracking_mask]

    # Compute critical rates (guard empties)
    r_c1 = escaping_rates[0] if escaping_rates.size > 0 else None
    r_c2 = tracking_rates[-1] if tracking_rates.size > 0 else None

    return r_c1, r_c2

def uncertainty_algorithm_for_rate_parallel(
    func,
    threshold=100,
    plot=True,
    verbose=True,
    min_eps=1e-10,
    max_eps=1e-6,
    num_eps=20,
    return_all_info=False,
    have_attractor=True,
    fit_truncation_order=0,
    n_jobs=-1,
    prefer="processes", 
    base_seed=0,
    **kwargs
):
    """
    The uncertainty algorithm for fractal 3.
    """

    epsilons = np.geomspace(min_eps, max_eps, num_eps)

    # keep your pop behavior
    rate_lower_bound = kwargs.pop('rate_lower_bound', 0.1)
    rate_upper_bound = kwargs.pop('rate_upper_bound', 4)

    rate_min, rate_max = get_rate_range(
        rate_lower_bound=rate_lower_bound,
        rate_upper_bound=rate_upper_bound,
        func=func,
        **kwargs
    )

    if rate_min >= rate_max:
        print('The basin boundary is not a fractal.')
        return 1

    def _one_eps(idx, eps):
        rng = np.random.default_rng(base_seed + idx)

        trials = 0
        uncertains = 0

        while uncertains < threshold:
            trials += 1
            rate1 = rng.uniform(rate_min, rate_max)
            rate2 = rate1 + eps if (rate1 + eps) <= rate_max else rate1 - eps

            step_x1 = evolve_tipping_until_escape(func=func, rate=rate1, **kwargs)
            step_x2 = evolve_tipping_until_escape(func=func, rate=rate2, **kwargs)

            if have_attractor:
                hit = ((step_x1 < 0 and step_x2 > 0) or (step_x1 > 0 and step_x2 < 0))
            else:
                hit = (step_x1 != step_x2)

            if hit:
                uncertains += 1

        p_hat = threshold / trials
        sigma_p = (np.sqrt(threshold * (1 - p_hat))) / trials
        return idx, p_hat, sigma_p

    # Serial (with tqdm) when n_jobs == 1
    if n_jobs == 1:
        it = enumerate(epsilons)
        if verbose:
            it = tqdm(it, total=len(epsilons), desc="Progress")
        results = [_one_eps(i, eps) for i, eps in it]
    else:
        if verbose:
            print(f"Running uncertainty_algorithm_for_rate in parallel (n_jobs={n_jobs}, prefer={prefer})...")
        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_one_eps)(i, eps) for i, eps in enumerate(epsilons)
        )

    results.sort(key=lambda t: t[0])

    feps = np.asarray([t[1] for t in results], dtype=float)
    feps_err = np.asarray([t[2] for t in results], dtype=float)

    return loglog_linear_fit(
        epsilons,
        feps,
        feps_err=feps_err,
        fit_truncation_order=fit_truncation_order,
        plot=plot,
        title='f(ε) vs ε for the second fractal',
        return_all_info=return_all_info,
    )