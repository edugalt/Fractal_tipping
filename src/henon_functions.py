############
# Packages #
############
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor

import warnings


#############
# functions #
#############
'''
linear path of the zero-start protocol
'''
def linear_path0(a_i, b_i, a_f, b_f, s_temp):
    a = a_i + (a_f - a_i) * np.tanh(s_temp)
    b = b_i + (b_f - b_i) * np.tanh(s_temp) 
    
    return a, b  


'''
Compute the stable fixed points of henon map period 1 directly with formula
Input:
    a: (float) parrameter a
    b: (float) parrameter b
Output:
    x: x-coordinate of the fixed point
    y: y-coordinate of the fixed point
'''
def direct_compute_stable_fixed_point(a, b):
    disc = max((1 - b)**2 + 4 * a, 0)
    if disc < 0 or a == 0:
        return None, None
    x_fp = (-(1 - b) + np.sqrt(disc)) / (2 * a)
    y_fp = b * x_fp
    return x_fp, y_fp


'''
Compute the next value of the henon map, with detection whether any orbit goes to infinity
Input:
    a: (float) parrameter a
    b: (float) parrameter b
    x_current: (float) the current value of x
    y_current: (float) the current value of y
    infinity_tipping_detected: (bool) whether it detects values go to infinity
    MAX_VAL: (float) threshold to indicate the mao is going to infinity
Output:
    x_next: (float) the next value of x
    y_next: (float) the next value of y
'''
def henon_map_next(a, b, x_current, y_current, MAX_VAL, infinity_tipping_detected): 
    x_next = 1 - a * x_current**2 + y_current
    y_next = b * x_current

    # Works with both scalar and array input
    overflow = (
        ~np.isfinite(x_next) | ~np.isfinite(y_next) |
        (np.abs(x_next) > MAX_VAL) | (np.abs(y_next) > MAX_VAL)
    )
    
    if np.any(overflow):
        x_next = np.where(overflow, np.nan, x_next)
        y_next = np.where(overflow, np.nan, y_next)
        infinity_tipping_detected = True

    return x_next, y_next, infinity_tipping_detected


'''
Input a value of r, see whether the system will tip with the given functions from (a_i, b_i) to (a_f, b_f)
Output:
    True: if the input r value inducing tipping to inifinity
    False: if not
    None: ambiguous case, tip to other orbits
'''
def r_inducing_tipping(func, a_i, b_i, a_f, b_f, r, x0, y0, 
                       N=1500, MAX_VAL=1e4, div_steps=60, diverge_threshold=1e-2,
                       converge_threshold=1e-4, converge_steps=30):
    
    min_steps = min(N-converge_steps, N-div_steps)

    x, y = x0, y0
    dist_ls = []
    escaped = False
    x_fp, y_fp = direct_compute_stable_fixed_point(a_f, b_f)
    
    # N_prev = 15    # for the infinite-start protocol, decide by a test round
    N_prev = 0 # zero-start protocol
    for n in range(-N_prev, N):
        if np.isinf(r):
            a, b = a_f, b_f
        else:
            s_temp = r * n
            a, b = func(a_i, b_i, a_f, b_f, s_temp)
            
        x, y, escaped = henon_map_next(a, b, x, y, MAX_VAL, escaped)
        if escaped or not np.isfinite(x) or not np.isfinite(y):
            return True

        # x_fp, y_fp = direct_compute_stable_fixed_point(a, b) # if using end-point check definition
        # if x_fp is None and y_fp is None:
        #     continue

        dist = np.hypot(x - x_fp, y - y_fp)
        if not np.isfinite(dist) or dist > MAX_VAL:
            return True
        
        # ignore earlier steps
        if n < min_steps:
            continue
        
        dist_ls.append(dist)
        
    # Check convergence
    if len(dist_ls) >= converge_steps:
        recent = np.array(dist_ls[-converge_steps:])
        if all(d < converge_threshold for d in recent):
            return False # strict convergence - so r does not induced tipping

        k = converge_steps // 3
        if k >= 3:
            is_converging = (
                np.mean(recent[0:k]) > np.mean(recent[k:2*k]) > np.mean(recent[2*k:])
            )
            if is_converging:
                return False
            
    # Check divergence  
    if len(dist_ls) >= div_steps:
        recent = np.array(dist_ls[-div_steps:])
        
        # just in case it moves away and then come back
        if k >= 3: # too few points to decide
            m1 = np.mean(recent[0:k])
            m2 = np.mean(recent[k:2*k])
            m3 = np.mean(recent[2*k:])
            all_large = (
                m1 > diverge_threshold and
                m2 > diverge_threshold and
                m3 > diverge_threshold
            )
            margin, factor = 1.05, 2
            strickly_increasing = (m1 * margin < m2) and (m2 * margin < m3)
            overall_increase = m3 > factor * m1
            if all_large and strickly_increasing and overall_increase:
                return True

    return None


'''
Check whether the fixed point (x_i, y_i) is inside the basin of attraction of (x_f, y_f)
Input:
    a_f, b_f: (float) the (a_f, b_f) that generates (x_f, y_f)
    x_i, y_i: the initial condition
    x_f, y_f: the end stable fixed point
Output:
    True: is inside
    False: not inside
    None: tip to other orbits
'''
def is_in_basin_of_attraction(
    a_f, b_f, x_i, y_i, x_f, y_f,
    N=1500, MAX_VAL=1e3,
    converge_threshold=1e-4, converge_steps=30,
    diverge_threshold=1e-2, diverge_steps=60
):
    
    min_steps = min(N-converge_steps, N-diverge_steps)
    x, y = x_i, y_i
    distances = []
    escaped = False
    for i in range(N):
        x, y, escaped = henon_map_next(a_f, b_f, x, y, MAX_VAL, escaped)

        if escaped or not np.isfinite(x) or not np.isfinite(y):
            return False

        dist = np.hypot(x - x_f, y - y_f)
        if not np.isfinite(dist) or dist > MAX_VAL:
            return False
        
        # ignore earlier steps
        if i < min_steps:
            continue
        
        distances.append(dist)


    # --- Check convergence ---
    if len(distances) >= converge_steps:
        recent = np.array(distances[-converge_steps:])
        if all(d < converge_threshold for d in recent):
            return True  # strict convergence

        k = converge_steps // 3
        if k >= 3:
            is_converging = (
                np.mean(recent[0:k]) > np.mean(recent[k:2*k]) > np.mean(recent[2*k:])
            )

            if is_converging:
                return True

            
    # --- Check diverge/tip to infinity ---
    if len(distances) >= diverge_steps:
        recent = np.array(distances[-diverge_steps:])
        
        # just in case it moves away and then come back
        k = diverge_steps // 3
        if k >= 3: # too few points to decide
            m1 = np.mean(recent[0:k])
            m2 = np.mean(recent[k:2*k])
            m3 = np.mean(recent[2*k:])
            all_large = (
                m1 > diverge_threshold and
                m2 > diverge_threshold and
                m3 > diverge_threshold
            )
            margin, factor = 1.05, 2
            strickly_increasing = (m1 * margin < m2) and (m2 * margin < m3)
            overall_increase = m3 > factor * m1
            if all_large and strickly_increasing and overall_increase:
                return False
    
    # Neither converge nor diverge to infinity: period3, period6, 3band, etc.
    return None

'''
Find out the label('tipping', 'converge', 'ambiguous') of the given (a_f, b_f) for the parameter space:
    - assume r is infty
    - iterate (x_i, y_i) once to get (x_i^*, y_i^*)
    - see whether (x_i^*, y_i^*) is in the basin of attraction of (a_f, b_f)
Input:
    x_i, x_i: (float) the initial fixed point from the fixed (a_i, b_i)
    x_f, y_f: (float) the fixed point from (a_f, b_f)
    a_i, b_i: (float) the initial fixed (a_i, b_i)
    a_f, b_f: (float) the chose (a_f, b_f)
Output:
    None: no label found because (x_f, y_f) contains None
    'tipping': this is a tipping situation
    'converge': a converge situation
    'ambiguous': an ambiguous situation
'''
def find_label(x_i, y_i, x_f, y_f, a_i, b_i, a_f, b_f):
    label = None 
   
    # a_star = (a_i+a_f)/2
    # b_star = (b_i+b_f)/2

    a_star = a_i
    b_star = b_i
    MAX_VAL = 1e6       
    infinity_tipping_detected = False    
    x_i_star, y_i_star, infinity_tipping_detected = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, infinity_tipping_detected)
    if infinity_tipping_detected:
        label = 'tipping'
        return label
    
    is_in_basin = is_in_basin_of_attraction(a_f, b_f, x_i_star, y_i_star, x_f, y_f)    
    if is_in_basin is False:
        label = 'tipping'  

    elif is_in_basin is True:
        label = 'converge'
    
    elif is_in_basin is None:
        label = 'ambiguous'
    
    return label


'''
Find the jacobian matix for the henon map evaluted at point (x0, y0).
Input:
    a: (float) parrameter a
    b: (float) parrameter b
    x: (float) value of x
    y: (float) value of y
    period: (int) period of the map
Output:
    jac: (np array, float) a list of x values after discarding n_transient points
'''
def henon_jacobian(x0, y0, a, b):
    jac = np.array([[-2 * a * x0, 1], # the jacobian matrix of period 1 henon map
                     [b, 0]])
    return jac


'''
Check whether a fixed point (x, y) is stable.
Input:
    a: (float) parrameter a
    b: (float) parrameter b
    x: (float) value of x
    y: (float) value of y
Output:
    True if all |lambda| < 1; otherwise False
'''
def is_stable(x, y, a, b):
    J = henon_jacobian(x, y, a, b)
        
    eigenvals = np.linalg.eigvals(J)

    margin = 1e-10
    return np.all(np.abs(eigenvals) < 1 - margin)


'''
Select certain number of (a, b) pair from the parameter space
Input:
    b_min: (float) the minimum value of b
    b_max: (float) the maximum value of b
    a_min: (float) the minimum value of a; 
            defaule need to use None, means the whole parameter a ranging based on b,
            if you want to fill the whole parameter space;
            do not feed in the a even if you have computed it
    a_max: (float) the maximum value of a
    n_b: number of b values
    n_a: number of a values. So in total, get n_b*n_a point
Output:
    ab_list: (list of tuples) a list of (a, b) pairs
'''
def select_ab_list(b_min, b_max, n_b, n_a, a_min=None, a_max=None):
    n_b = int(n_b)
    n_a = int(n_a)

    ab_list = []

    if b_min == b_max: # select a fixed b
        b_list =[b_min]
    else:
        b_list = np.linspace(b_min, b_max, num=n_b+2)[1:-1] # do not consider the end because it may be too close to bifurcation

    if a_min is not None and a_max is not None: 
        a_list = np.linspace(a_min, a_max, num=n_a+2)[1:-1]

    eps=1e-5
    # for each b, compute range of a
    for b in b_list:
        if a_min is None and a_max is None:
            a_min_local= -(1-b)**2/4 + eps
            a_max_local = 3*(1-b)**2/4 - eps

            a_list = np.linspace(a_min_local, a_max_local, num=n_a+2)[1:-1]

        for a in a_list:
            # and make sure the fixed point is stable
            x_fp, y_fp = direct_compute_stable_fixed_point(a, b)
            if x_fp is not None and y_fp is not None and is_stable(x_fp, y_fp, a, b):  
                # make sure derminant works
                des = (1-b)**2 + 4*a # descriminant of x+
                if des >=0 and a != 0:
                    ab_list.append( (a, b) )
    
    return ab_list


'''
Given a list of initial (a_i,b_i) pairs, 
figure out whether there exist other (a_f, b_f) pairs in the whole parameter space, 
such that the fixed point of (a_i, b_i) do not converge to the fixed points of (a_f, b_f) 
- and r tipping can happen from (a_i, b_i) to (a_f, b_f); 
also return those converge pairs 
(i.e. in the basin of attraction, is_in_basin() returns True), or ambiguous pairs (is_in_basin() returns None)
Input:
    ab_list: a list of (a, b) pairs [(a, b), ...]
    b_min: (float) the minimum value of b
    b_max: (float) the maximum value of b
    a_min: (float) the minimum value of a; defaule None, means the whole parameter range based on b
    a_max: (float) the maximum value of a
    n_b: number of b values
    n_a: number of a values. So in total, get n_b*n_a point
Output:
    r_tipping_ab_pairs: (dict with (a_i, b_i) as key, list of tuples [(a_f, b_f), ...] as values)
    converge_ab_pairs: (dict with (a_i, b_i) as key,
'''
def find_fp_possible_r_tipping_ab(ab_list, b_min, b_max, a_min=None, a_max=None, n_b=30, n_a=30):
    n_b = int(n_b)
    n_a = int(n_a)    
    
    tipping_infty_ab_pairs = {}
    converge_ab_pairs = {} 
    none_ab_pairs = {} 
    notSufficient_ab_pairs = {}
    
    ab_full_list = select_ab_list(b_min, b_max, n_b, n_a, a_min, a_max)

    for a_i, b_i in ab_list:
        tipping_infty_ab_pairs[(a_i, b_i)] = []
        # tipping_period3_ab_pairs[(a_i, b_i)] = []
        converge_ab_pairs[(a_i, b_i)] = []
        none_ab_pairs[(a_i, b_i)] = []
        notSufficient_ab_pairs[(a_i, b_i)] = []
        
        x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
        if x_i is None or y_i is None:
            continue

        for a_f, b_f in ab_full_list:
            if abs(a_f - a_i) < 1e-6 and abs(b_f - b_i) < 1e-6:
                continue

            x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f)
            if x_f is None or y_f is None:
                continue

            # calculate the actual initial point to find basin of attraction
            # n < 0, (a, b) = (a_i, b_i); n = 0, (a, b) = ( (a_i+a_f)/2, (b_i+b_f)/2 ); n>0, (a, b) = (a_f, b_f)
            # assume the case r = inf, the IC (x_i^*, y_i^*) is henon( (x_i, y_i),  (a_i+a_f)/2, (b_i+b_f)/2)
            # a_star = (a_i+a_f)/2 # protocal from -infty
            # b_star = (b_i+b_f)/2

            a_star = a_i # protocol from 0
            b_star = b_i

            MAX_VAL = 1e3
            infinity_tipping_detected = False

            # Z_1
            x_i_star, y_i_star, infinity_tipping_detected = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, infinity_tipping_detected) # fixed ai, bi
            if x_i_star is None or y_i_star is None or infinity_tipping_detected:
                tipping_infty_ab_pairs[(a_i, b_i)].append((a_f, b_f))
                continue
            
            if a_star >= (3*(1-b_star)**2) /4 or a_star <= (- (1-b_star)**2) / 4: # add not sufficient case
                is_in_basin = -2 # -2 not sufficient
            else:
                # Z_2 is merged in is_in_basin_of_attraction()
                # since Z_2 = H_{af, bf}(Z_1)
                is_in_basin = is_in_basin_of_attraction(a_f, b_f, x_i_star, y_i_star, x_f, y_f) #fixed a_i, b_i

            if is_in_basin is False:
                tipping_infty_ab_pairs[(a_i, b_i)].append((a_f, b_f))

            elif is_in_basin is True:
                converge_ab_pairs[(a_i, b_i)].append((a_f, b_f))

            elif is_in_basin is None:
                none_ab_pairs[(a_i, b_i)].append((a_f, b_f))
            
            elif is_in_basin == -2:
                notSufficient_ab_pairs[(a_i, b_i)].append((a_f, b_f))

    return tipping_infty_ab_pairs, converge_ab_pairs, none_ab_pairs, notSufficient_ab_pairs


#### Using parallel programming ######
def single_is_in_basin(args):
    try:
        a_f, b_f, a_i, b_i, x_i, y_i, MAX_VAL = args

        if abs(a_f - a_i) < 1e-6 and abs(b_f - b_i) < 1e-6:
            return None

        x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f)
        if x_f is None or y_f is None:
            return None

        # a_star = (a_i + a_f) / 2 # infinite-protocol
        # b_star = (b_i + b_f) / 2

        a_star = a_i # from 0
        b_star = b_i

        x_i_star, y_i_star, _ = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, False)
        if x_i_star is None or y_i_star is None:
            return (a_f, b_f, False)


        if a_star >= (3*(1-b_star)**2) /4 or a_star <= (- (1-b_star)**2) / 4: # add not sufficient case
            is_in_basin = -2 # -2 not sufficient
        else:
            # Z_2 is merged in is_in_basin_of_attraction()
            # since Z_2 = H_{af, bf}(Z_1)
            is_in_basin = is_in_basin_of_attraction(a_f, b_f, x_i_star, y_i_star, x_f, y_f)

        return (a_f, b_f, is_in_basin)

    except Exception as e:
        print(f"ERROR in process: ({a_f}, {b_f}) as {e}")
        return None
    

def find_fp_possible_r_tipping_ab_parallel(ab_list, b_min, b_max, 
                                           a_min=None, a_max=None, n_b=30, n_a=30, max_workers=4):
    n_b = int(n_b)
    n_a = int(n_a)

    tipping_infty_ab_pairs = {}
    converge_ab_pairs = {} 
    none_ab_pairs = {} 
    notSufficient_ab_pairs = {}

    ab_full_list = select_ab_list(b_min, b_max, n_b, n_a, a_min, a_max)

    MAX_VAL = 1e6

    num_cores_used = max_workers
    with ProcessPoolExecutor(max_workers=num_cores_used) as executor:
        for a_i, b_i in ab_list:
            tipping_infty_ab_pairs[(a_i, b_i)] = []
            converge_ab_pairs[(a_i, b_i)] = []
            none_ab_pairs[(a_i, b_i)] = []
            notSufficient_ab_pairs[(a_i, b_i)] = []
            
            x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
            if x_i is None or y_i is None:
                continue

            param_list = [
                (a_f, b_f, a_i, b_i, x_i, y_i, MAX_VAL)
                for a_f, b_f in ab_full_list
            ]

            raw_results = executor.map(single_is_in_basin, param_list)
            results = [res for res in raw_results if res is not None]

            for a_f, b_f, is_in_basin in results:

                if is_in_basin is False:
                    tipping_infty_ab_pairs[(a_i, b_i)].append((a_f, b_f))

                elif is_in_basin is True:
                    converge_ab_pairs[(a_i, b_i)].append((a_f, b_f))

                elif is_in_basin is None:
                    none_ab_pairs[(a_i, b_i)].append((a_f, b_f))
                
                elif is_in_basin == -2:
                    notSufficient_ab_pairs[(a_i, b_i)].append((a_f, b_f))

    return tipping_infty_ab_pairs, converge_ab_pairs, none_ab_pairs, notSufficient_ab_pairs


# a version fixed (af, bf) ranging (ai, bi)
def find_fp_possible_r_tipping_ab_reverse(ab_list, b_min, b_max, a_min=None, a_max=None, n_b=30, n_a=30):
    n_b = int(n_b)
    n_a = int(n_a)    
    
    tipping_infty_ab_pairs = {}
    converge_ab_pairs = {} 
    none_ab_pairs = {} 
    notSufficient_ab_pairs = {}
    
    ab_full_list = select_ab_list(b_min, b_max, n_b, n_a, a_min, a_max)

    for a_f, b_f in ab_list:
        tipping_infty_ab_pairs[(a_f, b_f)] = []
        converge_ab_pairs[(a_f, b_f)] = []
        none_ab_pairs[(a_f, b_f)] = []
        notSufficient_ab_pairs[(a_f, b_f)] = []
        
        x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f)
        if x_f is None or y_f is None:
            continue

        for a_i, b_i in ab_full_list:
            if abs(a_f - a_i) < 1e-6 and abs(b_f - b_i) < 1e-6:
                continue

            x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
            if x_i is None or y_i is None:
                continue

            a_star = a_i # protocol from 0 fixed af, bf
            b_star = b_i

            MAX_VAL = 1e3
            infinity_tipping_detected = False

            # Z_1
            x_i_star, y_i_star, infinity_tipping_detected = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, infinity_tipping_detected) # fixed af, bf
            if x_i_star is None or y_i_star is None or infinity_tipping_detected:
                tipping_infty_ab_pairs[(a_f, b_f)].append((a_i, b_i))
                continue
            
            if a_star >= (3*(1-b_star)**2) /4 or a_star <= (- (1-b_star)**2) / 4: # add not sufficient case
                is_in_basin = -2 # -2 not sufficient
            else:
                # Z_2 is merged in is_in_basin_of_attraction()
                # since Z_2 = H_{af, bf}(Z_1)
                is_in_basin = is_in_basin_of_attraction(a_f, b_f, x_i_star, y_i_star, x_f, y_f)

            if is_in_basin is False:
                tipping_infty_ab_pairs[(a_f, b_f)].append((a_i, b_i))

            elif is_in_basin is True:
                converge_ab_pairs[(a_f, b_f)].append((a_i, b_i))

            elif is_in_basin is None:
                none_ab_pairs[(a_f, b_f)].append((a_i, b_i))
            
            elif is_in_basin == -2: 
                notSufficient_ab_pairs[(a_f, b_f)].append((a_i, b_i))

    return tipping_infty_ab_pairs, converge_ab_pairs, none_ab_pairs, notSufficient_ab_pairs


def single_is_in_basin_reverse(args):
    try:
        a_i, b_i, a_f, b_f, x_f, y_f, MAX_VAL = args

        if abs(a_f - a_i) < 1e-6 and abs(b_f - b_i) < 1e-6:
            return None

        x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
        if x_i is None or y_i is None:
            return None

        # a_star = (a_i + a_f) / 2 # original
        # b_star = (b_i + b_f) / 2

        a_star = a_i # from 0
        b_star = b_i

        x_i_star, y_i_star, _ = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, False)
        if x_i_star is None or y_i_star is None:
            return (a_f, b_f, False)


        if a_star >= (3*(1-b_star)**2) /4 or a_star <= (- (1-b_star)**2) / 4: # add not sufficient case
            is_in_basin = -2 # -2 not sufficient
        else:
            # Z_2 is merged in is_in_basin_of_attraction()
            # since Z_2 = H_{af, bf}(Z_1)
            is_in_basin = is_in_basin_of_attraction(a_f, b_f, x_i_star, y_i_star, x_f, y_f)

        return (a_i, b_i, is_in_basin)

    except Exception as e:
        print(f"ERROR in process: ({a_f}, {b_f}) → {e}")
        return None
    

def find_fp_possible_r_tipping_ab_reverse_parallel(ab_list, b_min, b_max, 
                                                   a_min=None, a_max=None, n_b=30, n_a=30, max_workers=4):
    n_b = int(n_b)
    n_a = int(n_a)

    tipping_infty_ab_pairs = {}
    converge_ab_pairs = {} 
    none_ab_pairs = {} 
    notSufficient_ab_pairs = {}

    ab_full_list = select_ab_list(b_min, b_max, n_b, n_a, a_min, a_max)

    MAX_VAL = 1e6

    num_cores_used = max_workers
    with ProcessPoolExecutor(max_workers=num_cores_used) as executor:
        for a_f, b_f in ab_list:
            tipping_infty_ab_pairs[(a_f, b_f)] = []
            converge_ab_pairs[(a_f, b_f)] = []
            none_ab_pairs[(a_f, b_f)] = []
            notSufficient_ab_pairs[(a_f, b_f)] = []
            
            x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f)
            if x_f is None or y_f is None:
                continue

            param_list = [
                (a_i, b_i, a_f, b_f, x_f, y_f, MAX_VAL)
                for a_i, b_i in ab_full_list
            ]

            raw_results = executor.map(single_is_in_basin_reverse, param_list)
            results = [res for res in raw_results if res is not None]

            for a_i, b_i, is_in_basin in results:
                if is_in_basin is False:
                    tipping_infty_ab_pairs[(a_f, b_f)].append((a_i, b_i))

                elif is_in_basin is True:
                    converge_ab_pairs[(a_f, b_f)].append((a_i, b_i))

                elif is_in_basin is None:
                    none_ab_pairs[(a_f, b_f)].append((a_i, b_i))
                
                elif is_in_basin == -2:
                    notSufficient_ab_pairs[(a_f, b_f)].append((a_i, b_i))

    return tipping_infty_ab_pairs, converge_ab_pairs, none_ab_pairs, notSufficient_ab_pairs

'''
Find out the label('tipping', 'converge', 'ambiguous') of the given (a_f, b_f) for the parameter space:
    - assume r is infty
    - iterate (x_i, y_i) once to get (x_i^*, y_i^*)
    - see whether (x_i^*, y_i^*) is in the basin of attraction of (a_f, b_f)
Input:
    x_i, x_i: (float) the initial fixed point from the fixed (a_i, b_i)
    x_f, y_f: (float) the fixed point from (a_f, b_f)
    a_i, b_i: (float) the initial fixed (a_i, b_i)
    a_f, b_f: (float) the chose (a_f, b_f)
Output:
    None: no label found because (x_f, y_f) contains None
    0: this is a tipping situation to the infinity
    1: a converge situation
    -1: an ambiguous situation
'''
def find_label_numeric_para(x_i, y_i, x_f, y_f, a_i, b_i, a_f, b_f):
    label = None 
   
    # a_star = (a_i+a_f)/2 # protocol -infty
    # b_star = (b_i+b_f)/2 

    a_star = a_i # protocol 0
    b_star = b_i

    MAX_VAL = 1e6       
    infinity_tipping_detected = False    
    x_i_star, y_i_star, infinity_tipping_detected = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, infinity_tipping_detected)
    if infinity_tipping_detected:
        label = 0
        return label
    
    r = float('inf')
    x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
    induce_tipping = r_inducing_tipping(linear_path0, a_i, b_i, a_f, b_f, r, x_i, y_i)

    if induce_tipping is True:
        label = 0  

    elif induce_tipping is False:
        label = 1
    
    elif induce_tipping is None:
        label = -1
    
    return label

'''
Figure out the track/tip numerical labels of the values in the given val_space for plotting
Input:
    val_space: a list of values of in the free parameter space
    val_fixed: the fixed parameter
    x_i, y_i: input (x_i, y_i) from the selected (a_i, b_i)
    a_i, b_i: input selected (a_i, b_i)
    a_f, b_f: input selected (a_f, b_f)
output:
    labels: a list of label which contains
        - 0: tipping
        - 1: track/converge to given fixed point
        - -1: ambiguous
'''
def find_numeric_label_ls(val_space:list, val_fixed: float, 
                          x_i:float, y_i:float, 
                          a_i:float, b_i:float, a_f:float, b_f:float,
                          g_type:str):

    labels = []
    for val in val_space:
        
        if g_type == 'para':
            x_f, y_f = direct_compute_stable_fixed_point(val, val_fixed) 
            if None in (x_f, y_f):
                    continue
            label = find_label_numeric_para(x_i, y_i, x_f, y_f, a_i, b_i, val, val_fixed)
                   
        elif g_type == 'basin':
            x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f) 

            r = float('inf')
            x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
            induce_tipping = r_inducing_tipping(linear_path0, a_i, b_i, a_f, b_f, r, val_fixed, val)
            if induce_tipping is True:
                label = 0
            elif induce_tipping is False:
                label = 1
            elif induce_tipping is None:
                label = -1
        
        elif g_type == 'r':
            induce_tipping = r_inducing_tipping(linear_path0, a_i, b_i, a_f, b_f, val, x_i, y_i)  
            if induce_tipping is True:
                label = 0   
            elif induce_tipping is False:
                label = 1   
            elif induce_tipping is None:
                label = -1

        labels.append(label)

    return labels

'''
Plot panels of fractal branch cut with input data labels. 0 - tipping to infinity (green); 1 - converge (pink)
Input:
    labels: (list or array) The label values. 0 - converge; 1 - tip to infinity
    val_min, val_max: (float) the range limit of the free parameter
    ax: (matplotlib.axes.Axes) Optional. If provided, plot into this axes. Otherwise a new figure is created.
Output:
    fig, ax : Figure and Axes objects
'''
def plot_frac_bar(labels, val_min, val_max, ax=None):
    cmap = ListedColormap(['orange', 'yellow', 'blue'])
    # cmap = ListedColormap(['whitesmoke', 'whitesmoke', 'whitesmoke'])

    ax.imshow([labels],
              aspect="auto",
              interpolation="nearest",
              vmin=-1, vmax=1, # 0 maps to the yellow (tipping/not converge), and 1 map to blue (track)
              cmap=cmap,
              extent=[val_min, val_max, 0, 1])

    ax.set_yticks([])

    return ax


'''
compute the uncertainty ratio of the basin of atttraction for the given epsilon
Input:
    eps_ls: (array of float) the given epsilon list
    rng_ls: (array of random generator) a list of random generator
    TARGET_UNCERTAIN_POINTS: (int) target number of uncertain points
    b_fixed: (float) the fixed b values
    x_i, y_i: (float) the fixed point of (a_i, b_i)
    a_i, b_i: (float) the initial (a_i, b_i)
Output:
    (eps, f_eps, n_trials):
        eps:  used eps
        f_eps: the ratio of uncertain points
        n_trials: total number of trials
'''
def uncertainty_ratio_basin(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, 
                      y_min, y_max, x_fixed,
                      x_f, y_f, a_f, b_f):
    f_eps_ls = []
    used_eps = []
    for i, eps in enumerate(eps_ls):

        # figure out the actual bound
        feasible_low = y_min + eps
        feasible_up = y_max - eps
        if feasible_low >= feasible_up: # impossible to draw, skip this epsilon
            continue

        rng = rng_ls[i]
        n_trials = 0
        uncertain_counter = 0
  
        while uncertain_counter < TARGET_UNCERTAIN_POINTS:
            # calculate a_f_plus and a_f_minus and check whether they are within global bound
            y0 = rng.uniform(feasible_low, feasible_up)       
            y0_plus = y0 + eps
            y0_minus = y0 - eps        

            # figure out the labels
            y0_label  = is_in_basin_of_attraction(a_f, b_f, x_fixed, y0,       x_f, y_f)
            y0p_label = is_in_basin_of_attraction(a_f, b_f, x_fixed, y0_plus,  x_f, y_f)
            y0m_label = is_in_basin_of_attraction(a_f, b_f, x_fixed, y0_minus, x_f, y_f)

            if None in (y0_label, y0p_label, y0m_label):
                continue
    
            n_trials += 1

            if not (y0_label == y0p_label == y0m_label):
                uncertain_counter += 1

        # only keep this epsilon if we had at least one valid trial
        if n_trials == 0:
            continue
        
        if n_trials < TARGET_UNCERTAIN_POINTS:
            print("!!!INCREASE THE RESOLUTION!!!")

        f_eps = uncertain_counter / n_trials

        if f_eps > 0:
            used_eps.append(eps)
            f_eps_ls.append(f_eps)
        
    return f_eps_ls, used_eps

'''
a parallel version for the uncertainty ratio of the basin of attraction
'''
# ---- worker (must be top-level for multiprocessing) ----
def uncertainty_ratio_basin_for_single_eps(args):

    (i, eps, rng, TARGET_UNCERTAIN_POINTS,
     y_min, y_max, x_fixed,
     x_f, y_f, a_f, b_f) = args

    feasible_low = y_min + eps
    feasible_up = y_max - eps
    if feasible_low >= feasible_up: # impossible to draw, skip this epsilon
        return (i, eps, None, 0)

    n_trials = 0
    uncertain_counter = 0

    while uncertain_counter < TARGET_UNCERTAIN_POINTS:

        # calculate y0_plus and y0_minus and check whether they are within global bound
        y0 = rng.uniform(feasible_low, feasible_up)       
        y0_plus = y0 + eps
        y0_minus = y0 - eps

        # figure out the labels
        y0_label  = is_in_basin_of_attraction(a_f, b_f, x_fixed, y0,       x_f, y_f)
        y0p_label = is_in_basin_of_attraction(a_f, b_f, x_fixed, y0_plus,  x_f, y_f)
        y0m_label = is_in_basin_of_attraction(a_f, b_f, x_fixed, y0_minus, x_f, y_f)
        
        if None in (y0_label, y0p_label, y0m_label):
            continue

        n_trials += 1
        if not (y0_label == y0p_label == y0m_label):
            uncertain_counter += 1

    if n_trials == 0:
        return (i, eps, None, 0)

    f_eps = uncertain_counter / n_trials
    return (i, eps, f_eps, n_trials)


def uncertainty_ratio_basin_parallel(
    eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS,
    y_min, y_max, x_fixed,
    x_f, y_f, a_f, b_f, max_workers=None
):

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        param_list = [
            (i, eps, rng_ls[i], TARGET_UNCERTAIN_POINTS,
            y_min, y_max, x_fixed,
            x_f, y_f, a_f, b_f)
            for i, eps in enumerate(eps_ls) ]
        # map preserves input order -> results already sorted by i
        raw_results = ex.map(uncertainty_ratio_basin_for_single_eps, param_list)
        results = list(raw_results)

    f_eps_ls = []
    used_eps = []
    for _, eps, f_eps, n_trials in results:
        if n_trials == 0:
            continue

        if n_trials < TARGET_UNCERTAIN_POINTS:
            print("!!!INCREASE THE RESOLUTION!!!")

        if f_eps is not None and f_eps > 0:
            used_eps.append(eps)
            f_eps_ls.append(f_eps)

    return f_eps_ls, used_eps

'''
compute the uncertainty ratio of the parameter space for the given epsilon
Input:
    eps_ls: (array of float) the given epsilon list
    rng_ls: (array of random generator) a list of random generator
    TARGET_UNCERTAIN_POINTS: (int) target number of uncertain points
    b_fixed: (float) the fixed b values
    x_i, y_i: (float) the fixed point of (a_i, b_i)
    a_i, b_i: (float) the initial (a_i, b_i)
Output:
    (eps, f_eps, n_trials):
        eps:  used eps
        f_eps: the ratio of uncertain points
        n_trials: total number of trials
'''
def uncertainty_ratio_para(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, 
                      a_min, a_max, b_fixed,
                      x_i, y_i, a_i, b_i):
    f_eps_ls = []
    used_eps = []
    for i, eps in enumerate(eps_ls):

        # figure out the actual bound
        feasible_low = a_min + eps
        feasible_up = a_max - eps
        if feasible_low >= feasible_up: # impossible to draw, skip this epsilon
            continue

        rng = rng_ls[i]
        n_trials = 0
        uncertain_counter = 0
  
        while uncertain_counter < TARGET_UNCERTAIN_POINTS:
            # calculate a_f_plus and a_f_minus and check whether they are within global bound
            a_f = rng.uniform(feasible_low, feasible_up)
            a_f_plus = a_f + eps
            a_f_minus = a_f - eps        

            # figure put the fixed points of 
            x_f,  y_f = direct_compute_stable_fixed_point(a_f, b_fixed)
            x_fp, y_fp = direct_compute_stable_fixed_point(a_f_plus,  b_fixed)
            x_fm, y_fm = direct_compute_stable_fixed_point(a_f_minus, b_fixed)

            if None in (x_f, y_f, x_fp, y_fp, x_fm, y_fm):
                continue

            # figure out the labels
            label_af = find_label(x_i, y_i, x_f, y_f, a_i, b_i, a_f, b_fixed)
            label_af_plus  = find_label(x_i, y_i, x_fp, y_fp, a_i, b_i, a_f_plus, b_fixed)
            label_af_minus = find_label(x_i, y_i, x_fm, y_fm, a_i, b_i, a_f_minus, b_fixed)

            if None in (label_af, label_af_plus, label_af_minus):
                continue

            if 'ambiguous' in (label_af, label_af_plus, label_af_minus):
                continue
    
            n_trials += 1

            if not (label_af == label_af_plus == label_af_minus):
                uncertain_counter += 1

        # only keep this epsilon if we had at least one valid trial
        if n_trials == 0:
            continue
        
        if n_trials < TARGET_UNCERTAIN_POINTS:
            print("!!!INCREASE THE RESOLUTION!!!")

        f_eps = uncertain_counter / n_trials

        if f_eps > 0:
            used_eps.append(eps)
            f_eps_ls.append(f_eps)
 
    return f_eps_ls, used_eps


'''
a parallel version for the uncertainty ratio of the parameter space
'''
# ---- worker (must be top-level for multiprocessing) ----
def uncertainty_ratio_para_for_single_eps(args):

    (i, eps, rng, TARGET_UNCERTAIN_POINTS,
     a_min, a_max, b_fixed,
     x_i, y_i, a_i, b_i) = args

    feasible_low = a_min + eps
    feasible_up = a_max - eps
    if feasible_low >= feasible_up: # impossible to draw, skip this epsilon
        return (i, eps, None, 0)

    n_trials = 0
    uncertain_counter = 0

    while uncertain_counter < TARGET_UNCERTAIN_POINTS:

        a_f = rng.uniform(feasible_low, feasible_up)
        a_f_plus  = a_f + eps
        a_f_minus = a_f - eps

        # fixed points
        x_f,  y_f  = direct_compute_stable_fixed_point(a_f,       b_fixed)
        x_fp, y_fp = direct_compute_stable_fixed_point(a_f_plus,  b_fixed)
        x_fm, y_fm = direct_compute_stable_fixed_point(a_f_minus, b_fixed)
        if None in (x_f, y_f, x_fp, y_fp, x_fm, y_fm):
            continue

        # labels
        label_af       = find_label(x_i, y_i, x_f,  y_f,  a_i, b_i, a_f,       b_fixed)
        label_af_plus  = find_label(x_i, y_i, x_fp, y_fp, a_i, b_i, a_f_plus,  b_fixed)
        label_af_minus = find_label(x_i, y_i, x_fm, y_fm, a_i, b_i, a_f_minus, b_fixed)
        if None in (label_af, label_af_plus, label_af_minus):
            continue
        if 'ambiguous' in (label_af, label_af_plus, label_af_minus):
            continue

        n_trials += 1
        if not (label_af == label_af_plus == label_af_minus):
            uncertain_counter += 1

    if n_trials == 0:
        return (i, eps, None, 0)

    f_eps = uncertain_counter / n_trials
    return (i, eps, f_eps, n_trials)


def uncertainty_ratio_para_parallel(
    eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS,
    a_min, a_max, b_fixed,
    x_i, y_i, a_i, b_i, max_workers=None
):
    
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        param_list = [
            (i, eps, rng_ls[i], TARGET_UNCERTAIN_POINTS,
            a_min, a_max, b_fixed,
            x_i, y_i, a_i, b_i)
            for i, eps in enumerate(eps_ls) ]
        # map preserves input order -> results already sorted by i
        raw_results = ex.map(uncertainty_ratio_para_for_single_eps, param_list)
        results = list(raw_results)

    f_eps_ls = []
    used_eps = []
    for _, eps, f_eps, n_trials in results:
        if n_trials == 0:
            continue

        if n_trials < TARGET_UNCERTAIN_POINTS:
            print("!!!INCREASE THE RESOLUTION!!!")

        if f_eps is not None and f_eps > 0:
            used_eps.append(eps)
            f_eps_ls.append(f_eps)

    return f_eps_ls, used_eps


'''
compute the uncertainty ratio of the r value for the given epsilon
Input:
    eps_ls: (array of float) the given epsilon list
    rng_ls: (array of random generator) a list of random generator
    TARGET_UNCERTAIN_POINTS: (int) target number of uncertain points
    b_fixed: (float) the fixed b values
    x_i, y_i: (float) the fixed point of (a_i, b_i)
    a_i, b_i: (float) the initial (a_i, b_i)
    x0, y0: (float) the fixed point corresponding to (a_i, b_i)
Output:
    (eps, f_eps, n_trials):
        eps:  used eps
        f_eps: the ratio of uncertain points
        n_trials: total number of trials
'''
def uncertainty_ratio_r(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, 
                      r_min, r_max, func, a_i, b_i, a_f, b_f, x0, y0):
    f_eps_ls = []
    used_eps = []
    for i, eps in enumerate(eps_ls):

        # figure out the actual bound
        feasible_low = r_min + eps
        feasible_up = r_max - eps
        if feasible_low >= feasible_up: # impossible to draw, skip this epsilon
            continue

        rng = rng_ls[i]
        n_trials = 0
        uncertain_counter = 0
  
        while uncertain_counter < TARGET_UNCERTAIN_POINTS:
            # calculate a_f_plus and a_f_minus and check whether they are within global bound
            r0 = rng.uniform(feasible_low, feasible_up)       
            r0_plus = r0 + eps
            r0_minus = r0 - eps        

            # figure out the labels
            r0_label  = r_inducing_tipping(func, a_i, b_i, a_f, b_f, r0, x0, y0)
            r0p_label = r_inducing_tipping(func, a_i, b_i, a_f, b_f, r0_plus, x0, y0)
            r0m_label = r_inducing_tipping(func, a_i, b_i, a_f, b_f, r0_minus, x0, y0)
    
            n_trials += 1

            if not (r0_label == r0p_label == r0m_label):
                uncertain_counter += 1

        # only keep this epsilon if we had at least one valid trial
        if n_trials == 0:
            continue
        
        if n_trials < TARGET_UNCERTAIN_POINTS:
            print("!!!INCREASE THE RESOLUTION!!!")

        f_eps = uncertain_counter / n_trials

        if f_eps > 0:
            used_eps.append(eps)
            f_eps_ls.append(f_eps)
        
    return f_eps_ls, used_eps



'''
Compute model fit and confidence intervals of both analytical and bootstrap
Input:
    a, b: given (a, b). 
        If it's about parameter space boundary, it is (a_i, b_i); 
        if it is about basin boundary, it is (a_f, b_f)
    ab_f: 
        given (a_f, b_f) in the r plot case, 
        when both (a_i, b_i) and (a_f, b_f) are needed.
        In parameter and basin case, this is set to None
    s_type: given type of the data
        'para': boundary of the r-region in the parameter space
        'basin': boundary of the basin of attraction
Output:
    result: (dict)
        {
        'a' = the input a_i or a_f
        'b' = the input b_i or b_f
        'a_f' = the input a_f in the r case
        'b_f' = the input b_f in the r case
        'type' = the input type; either 'para' or 'basin' or 'r'
        'used_eps': (np array) of a list of epsilons
        'f_eps_ls': (np array) of corresponding f values
        'alpha': the slope of the linear regression; i.e. the uncertainty exponent
        'beta': the intersept of the linear regression
        'box_dimension': the box counting dimension
        'RMSE': RMSE of the model fit
        'R2': R-square value of the model fit
        'CI95_analytical_tSE': the t_val*SE part of the 95% confidence interval of analytical result
        'CI99_analytical_tSE': the t_val*SE part of the 99% confidence interval of analytical result
        'CI95_analytical': (tuple) the 95% confidence interval of analytical result 
                                   (alpha - t_val*SE, alpha + t_val*SE)
        'CI99_analytical': (tuple) the 95% confidence interval of analytical result
        'CI95_boot': (tuple) the 95% confidence interval of bootstrap result 
        'CI99_boot': (tuple) the 99% confidence interval of bootstrap result
        }
'''
def compute_CI(f_eps_ls:list[float], used_eps:list[float], a:float, b:float, s_type:str, ab_f:tuple):

    if ab_f is not None:
        a_f, b_f = ab_f[0], ab_f[1]
   
    result = {}

    result['a'] = a
    result['b'] = b

    result['a_f'] = a_f # only for r case; other case this is None
    result['b_f'] = b_f

    result['type'] = s_type

    # fetch the values
    used_eps = np.asarray(used_eps)
    f_eps_ls = np.asarray(f_eps_ls)

    mask = np.log10(used_eps) < -4 # rule out x = log_10(eps) >= -2 or -4 for fractal
    used_eps = used_eps[mask]
    f_eps_ls = f_eps_ls[mask]

    result['used_eps'] = used_eps
    result['f_eps_ls'] = f_eps_ls

    # transform to log space
    log_eps = np.log10(np.array(used_eps, dtype=float))
    log_f = np.log10(np.array(f_eps_ls, dtype=float))

    # -- fit linear regrssion --
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning) # ignore warning for lower resolution test
        para, cov = np.polyfit(log_eps, log_f, 1, cov=True)
    D = 1 # dimension of the phase space; 1 because is a 1D slice
    slope, intersept = para
    alpha = slope
    box_dimension = D - alpha

    result['alpha'] = alpha
    result['beta'] = intersept
    result['box_dimension'] = box_dimension

    se_slope = np.sqrt(cov[0,0])   # standard error of slope etimator
    se_intercept  = np.sqrt(cov[1,1])   # standard error of intercept etimator

    # -- model fit --
    # MSE
    fit = np.poly1d(para)
    y_hat = fit(log_eps)
    residuals = log_f - y_hat
    mse_resid = np.mean(residuals**2)    
    rmse = np.sqrt(mse_resid)   

    # R^2
    R2 = 1 - np.var(residuals, ddof=1) / np.var(log_f, ddof=1) 

    result['RMSE'] = rmse 
    result['R2']  = R2
   
    # -- theoretical CI --
    n = len(f_eps_ls)
    # 95% CI
    t95 = stats.t.ppf(0.975, df=n-2)   # 95% CI
    CI95_analytical = (alpha - t95*se_slope, alpha + t95*se_slope ) # CI = estimate +- t_{1-0.05/2}^{n-2}*SE

    # 99% CI
    t99 = stats.t.ppf(0.995, df=n-2)   # 99% CI
    CI99_analytical = (alpha - t99*se_slope, alpha + t99*se_slope )


    result['CI95_analytical_tSE'] = t95*se_slope
    result['CI99_analytical_tSE'] = t99*se_slope
    result['CI95_analytical'] = CI95_analytical
    result['CI99_analytical'] = CI99_analytical

    # -- Bootstrapping --
    rng = np.random.default_rng(2025)
    n_boot = 5000
    boot_alphas = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning) # ignore warning for lower resolution test
            a_i, _ = np.polyfit(log_eps[idx], log_f[idx], 1)
        boot_alphas[i] = a_i

    # Percentile CIs
    CI95_boot = (np.percentile(boot_alphas, 2.5),  np.percentile(boot_alphas, 97.5))
    CI99_boot = (np.percentile(boot_alphas, 0.5),  np.percentile(boot_alphas, 99.5))

    result['CI95_boot'] = CI95_boot
    result['CI99_boot'] = CI99_boot

    # -- print summary --   
    if s_type == 'basin':
        a_label = 'a_f'
        b_label = 'b_f'

    elif s_type == 'para' or 'r':
        a_label = 'a_i'
        b_label = 'b_i'
    

    print(f'''
Summary:
        ({a_label}, {b_label}) = ({a}, {b})
        type: {s_type}
        estimated alpha: {alpha}
        RMSE (analytic): {rmse} 
        R-Square: {R2}
        95% CI analytical: [{alpha} - {t95*se_slope}, {alpha} + {t95*se_slope}] \
          = [{alpha - t95*se_slope}, {alpha + t95*se_slope}]
        95% CI bootstrap: [{CI95_boot[0]}, {CI95_boot[1]}]
        99% CI analytical: [{alpha} - {t99*se_slope}, {alpha} + {t99*se_slope}] \
          = [{CI99_analytical[0]}, {CI99_analytical[1]}]
        99% CI bootstrap: [{CI99_boot[0]}, {CI99_boot[1]}]
          ''')
    
    return result


'''
Plot and fit the linear regression of the uncertainty exponent
'''
def plot_fit_uncertainty_exponent(s_type, TARGET_UNCERTAIN_POINTS, f_eps_ls, used_eps):
    if s_type == 'basin':
        name = 'Fractal 1'
    elif s_type == 'para':
        name = 'Fractal 2'
    elif s_type == 'r':
        name = 'Fractal 3'
        
    # fetch the values
    used_eps = np.asarray(used_eps)
    f_eps_ls = np.asarray(f_eps_ls)

    # filter out large epsilon
    mask = np.log10(used_eps) < -4 # keep only log10(eps) < -4
    used_eps = used_eps[mask]
    f_eps_ls = f_eps_ls[mask]

    # find error depending on negative binomial dist
    r = TARGET_UNCERTAIN_POINTS # the number of targeted uncertainty trials (based on the head comment in the .csv files)
    err = np.sqrt(f_eps_ls**2 * (1-f_eps_ls)/r) # negative binomial sd: p^2(1-p)/r

    # log space
    log_eps = np.log10(np.array(used_eps, dtype=float))
    log_f = np.log10(np.array(f_eps_ls, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning) # ignore warning for lower resolution test
        slope, intercept = np.polyfit(log_eps, log_f, 1)

    # plot
    xline = np.linspace(used_eps.min(), used_eps.max(), 200)
    yline = (xline**slope) * (10**intercept) # f = eps^{slope} * 10^{intercept} 

    plt.errorbar(used_eps, f_eps_ls, yerr=err, fmt= 'o', capsize=3, markersize=5,
                label=fr"{name}: $\alpha \approx {slope:.3f}$"
                )
    plt.plot(xline, yline)  
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$f(\epsilon)$')

    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, alpha=0.2, linewidth=0.8)
    plt.tight_layout()
    plt.show()



'''
Compute uncertainty exponent of the fractal set
Input:
    s_type: (str) type of fractal boundary computing
        - basin
        - para
        - r
    TARGET_UNCERTAIN_POINTS: (int) target number of uncertainty points (default=150 used in the paper)
    K: (int )scale of epsilon (default=40 used in the paper)
    seed: for random generator (default=2025 used in the paper)
    ab_i: (tuple) the initial/starting parameter pair (a_i, b_i)
    ab_f: (tuple) the final/end parameter pair (a_f, b_f)
    val_fixed: (float) the fixed value
    val_range: (tuple) the ranging value (val_min, val_max)
Output:
    f_eps_ls: values of f(epsilon)
    used_eps: valid eps
'''
def uncertainty_exponent(s_type, TARGET_UNCERTAIN_POINTS=150, K=40,
                         ab_i=None, ab_f=None, val_fixed=None, val_range=None, seed=2025):
    if ab_i is not None:
        a_i, b_i = ab_i[0], ab_i[1]
        x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)

    if ab_f is not None:
        a_f, b_f = ab_f[0], ab_f[1]
        x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f)
    
    if val_range is not None:
        val_min, val_max = val_range[0], val_range[1]
    else:
        print("input a range!")

    eps_ls = [2**(-k) for k in range(0, K)]
    # Create random generator for each epsilon
    ss = np.random.SeedSequence(seed) # a base RG
    child_rngs = ss.spawn(len(eps_ls))
    rng_ls = [np.random.default_rng(s) for s in child_rngs]

    if s_type == 'basin':
        max_workers = 4
        f_eps_ls, used_eps = uncertainty_ratio_basin_parallel(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, 
                        val_min, val_max, val_fixed,
                        x_f, y_f, a_f, b_f, max_workers)
        # f_eps_ls, used_eps = uncertainty_ratio_basin(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, 
        #               val_min, val_max, val_fixed,
        #               x_f, y_f, a_f, b_f)

    elif s_type == 'para':
        max_workers = 4
        f_eps_ls, used_eps = uncertainty_ratio_para_parallel(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, 
                          val_min, val_max, val_fixed,
                          x_i, y_i, a_i, b_i, max_workers)
        # f_eps_ls, used_eps = uncertainty_ratio_para(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, # reverse
        #             val_min, val_max, val_fixed,
        #             x_i, y_i, a_i, b_i) 
        
    elif s_type == 'r':
        f_eps_ls, used_eps = uncertainty_ratio_r(eps_ls, rng_ls, TARGET_UNCERTAIN_POINTS, 
                    val_min, val_max, linear_path0, a_i, b_i, a_f, b_f, x_i, y_i)

    # plot the graph
    plot_fit_uncertainty_exponent(s_type, TARGET_UNCERTAIN_POINTS, f_eps_ls, used_eps)

    # # save data as csv
    # header = (
    #     f"# (a_i, b_i) = ({a_i:.3f}, {b_i:.3f})\n"
    #     f"# (a_f, b_f) = ({a_f:.3f}, {b_f:.3f})\n"
    #     f"# r = ({r_min}, {r_max})\n"
    #     f"# parameter r, the changing rate\n"
    #     f"# K={K}, TARGET_UNCERTAIN_POINTS={TARGET_UNCERTAIN_POINTS}\n"
    #     "Short iteration = 350 for r_induced_tipping(); the other is long iter = 1500"
    #     "used_eps,f_eps"
    # )

    # np.savetxt(
    #     f"fracDimPreciseC1_r_ai{a_i:.3f}bi{b_i:.3f}_af{a_f:.3f}bf{b_f:.3f}.csv", 
    #     np.column_stack([used_eps, f_eps_ls]), 
    #     delimiter=",", 
    #     header=header, 
    #     comments=""
    # )
    
    return f_eps_ls, used_eps


'''
Given fixed point (x_f, y_f) and its basin catefory, plot basin of attraction and the fixed point itself
Input:
    basin: (data frame) basin labels from the .csv file
    x_f, y_f: (float) fixed point corresponding to (a_f, b_f)
    x_initial_ls, y_initial_ls: (array) initial conditions in the space
    highlight_region: (tuple of tuples) used to draw a highlighted frame to indicate zoom-in section if needed; default None
    xlim, ylim: (tuple of two floats) the region to focus
    save_path: (str) path to save the figure
'''
def plot_fixed_point_basin(basin, x_f, y_f, x_initial_ls, y_initial_ls, 
                           highlight_region=None, xlim=(0, 0), ylim=(0, 0), save_path=None):

    # Define custom colors
    cmap = ListedColormap(['yellow', 'blue'])  # 0 -> yellow, 1 -> blue (inside basin)

    plt.figure(figsize=(6, 6))
    plt.imshow(basin, cmap=cmap, origin='lower',
               extent=[x_initial_ls.min(), x_initial_ls.max(), 
                       y_initial_ls.min(), y_initial_ls.max()])

    
    # fixed point
    plt.plot([x_f], [y_f],
         marker='s',
         linestyle='None',
         color='red',
         markersize=15,
         markeredgewidth=4,
         label='Fixed point')
    
    # # (OPTIONAL) zoom in box
    # if highlight_region is not None:
    #     (x_min, x_max), (y_min, y_max) = highlight_region
    #     rect = Rectangle(
    #         (x_min, y_min),
    #         x_max - x_min,
    #         y_max - y_min,
    #         linewidth=0.8,
    #         edgecolor='red',
    #         facecolor='none',
    #         linestyle='-'
    #     )
    #     plt.gca().add_patch(rect)
     
    # xlim, ylim
    if not (xlim[0] == 0 and xlim[1] == 0):
        plt.xlim(xlim)
    if not (ylim[0] == 0 and ylim[1] == 0):
        plt.ylim(ylim)

    # legend
    # legend_handles = [
    #     Patch(facecolor='blue', edgecolor='grey', label='Basin of attraction'),
    #     Line2D([0], [0],
    #                marker='x',
    #                linestyle='None',
    #                color='red',
    #                markersize=5.5,
    #                markeredgewidth=1,
    #                label='Fixed point')
    # ]
    
    plt.xlabel(r'$x_0$', fontsize=22)
    plt.ylabel(r'$y_0$', fontsize=22)
    plt.gca().tick_params(labelsize=20)

    # plt.title('Basin of Attraction')
    # leg = plt.legend(handles=legend_handles, loc='upper left', frameon=True)
    # frame = leg.get_frame()
    # frame.set_facecolor("whitesmoke")   # same background
    # frame.set_edgecolor("grey")         # same border color
    # frame.set_linewidth(0.8)            # same border thickness
    # frame.set_alpha(1.0)                # fully opaque   
    # plt.grid(True)
    plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')