from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import math
import seaborn as sns
import pandas as pd
from sympy import *
from sympy import Symbol, solveset, S, erf, log, sqrt
init_printing(use_unicode=True)

import multiprocessing as mp
from functools import partial
from pathlib import Path

import scipy.optimize as optimize

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "simulation_outputs"

def U_D(gamma_1,gamma_0,C_1,r,delta):
    return (1-delta)*np.dot(r,gamma_1) - np.dot(np.dot(np.subtract(gamma_1,gamma_0),C_1),np.subtract(gamma_1,gamma_0))

#U_D = term0 * alpha_0 + termA*beta_0^2 + termB&beta_0 + termC
'''
def get_UD_zero_levelset_terms(C_1,r,delta,theta):
    term0 = (1-delta)*r[0]
    termC = ((1-delta)**2/(4*C_1[0,0]) 
             + (1-delta)*r[1]*theta 
             - (C_1[0,1]*(1-delta)*r[0]*theta/C_1[0,0]) 
             + C_1[0,1]**2 * theta**2 / C_1[0,0]
             - C_1[1,1]*theta**2
            )
    termB = (C_1[0,1]*(1-delta)*r[0] / C_1[0,0]
             - 2*C_1[0,1]**2 * theta / C_1[0,0]
             + 2*C_1[1,1]*theta
            )
    termA = (C_1[0,1]**2 / C_1[0,0]
             - C_1[1,1]
            )
    return term0,termA,termB,termC
'''

def get_UD_zero_levelset_terms(C_1, r, delta, theta):
    """
    Returns the coefficients for the U_D level set equation in the form:
    alpha_0_term * alpha_0 + beta_0_squared_term * beta_0^2 + beta_0_term * beta_0 + constant_term = 0
    """
    # Alpha_0 term coefficient
    alpha_0_term = (1 - delta) * r[0]
    
    # Beta_0^2 term coefficient
    beta_0_squared_term = (C_1[0,1]**2 / C_1[0,0]) - C_1[1,1]
    
    # Beta_0 term coefficient
    beta_0_term = (C_1[0,1] * (1 - delta) * r[0] / C_1[0,0] 
                   - 2 * C_1[0,1]**2 * theta / C_1[0,0] 
                   + 2 * C_1[1,1] * theta)
    
    # Constant term
    constant_term = ((1 - delta)**2 * r[0]**2 / (4 * C_1[0,0]) 
                     + (1 - delta) * r[1] * theta 
                     - (C_1[0,1] * (1 - delta) * r[0] * theta / C_1[0,0]) 
                     + (C_1[0,1]**2 * theta**2 / C_1[0,0]) 
                     - C_1[1,1] * theta**2)
    
    return alpha_0_term, beta_0_squared_term, beta_0_term, constant_term

'''
def get_UD_zero_levelset_terms(C_1, r, delta, theta):
    """
    Returns the updated coefficients for the U_D level set equation in the form:
    alpha_0_term * alpha_0 + beta_0_squared_term * beta_0^2 + beta_0_term * beta_0 + constant_term = 0
    """
    # Alpha_0 term coefficient
    alpha_0_term = (1 - delta) * r[0]
    
    # Beta_0 term coefficient
    beta_0_term = ((1 - delta) * r[1] 
                   + (1 - delta) * (C_1[0,1] / C_1[0,0]) * r[0] 
                   - 2 * (C_1[1,1] - (C_1[0,1]**2 / C_1[0,0])) * theta)
    
    # Beta_0^2 term coefficient
    beta_0_squared_term = (C_1[0,1]**2 / C_1[0,0]) - C_1[1,1]
    
    # Constant term
    constant_term = ((1 - delta) * (-(C_1[0,1] / C_1[0,0]) * r[0] + r[1]) * theta
                     - (C_1[1,1] - (C_1[0,1]**2 / C_1[0,0])) * theta**2
                     + ((1 - delta)**2 * r[0]**2) / (4 * C_1[0,0]))
    
    return alpha_0_term, beta_0_squared_term, beta_0_term, constant_term
'''

def D_strategy(gamma_0,C_1,r,delta,theta):
    if sum(np.equal(gamma_0,[-.1,-.1]))==2:
        return np.array([-.1,-.1]),-.1,"abstain"
    #Abstain condition
    alpha_0 = gamma_0[0]
    beta_0 = gamma_0[1]
    #This is the version without cross-terms
    #abstain_quadratic_A = -C_1[1,1]
    #abstain_quadratic_B = (C_1[0,1]*(1-delta)*r[0]/C_1[0,0]) + 2*theta*C_1[1,1]
    #abstain_quadratic_C = ((1-delta)**2 * r[0]**2 / (4*C_1[0,0])) + (1-delta)*r[1]*theta - (C_1[0,1]*(1-delta)*r[0]*theta/C_1[0,0])-C_1[1,1]*theta**2
    #if 0 > ((1-delta)*r[0])*alpha_0 + abstain_quadratic_A*beta_0**2 + abstain_quadratic_B*beta_0 + abstain_quadratic_C:
    #    return [-.1,-.1],"abstain",-0.1
    #term0,termA,termB,termC = get_UD_zero_levelset_terms(C_1,r,delta,theta)
    #print(term0*alpha_0 + termA*beta_0**2 + termB*beta_0 + termC)
    #if 0 > term0*alpha_0 + termA*beta_0**2 + termB*beta_0 + termC:
    #    return [-.1,-.1],-0.1,"abstain"
    
    #Closed-form solution
    gamma_candidate_1 = gamma_0 + 0.5 * (1-delta) * np.dot(np.linalg.inv(C_1),r)
    gamma_candidate_2 = np.array([alpha_0,np.max([beta_0+(1-delta)*r[1]/(2*C_1[1,1]),theta])]) #can likely remove the np.max() and theta option from this
    #Making a slight change here to account for cross-terms:
    gamma_candidate_3 = np.array([np.max([alpha_0,
                                          alpha_0+(1-delta)*r[0]/(2*C_1[0,0]) - (C_1[0,1]/C_1[0,0])*(theta-beta_0)]),
                                  np.max([beta_0,theta])
                                 ])
    
    #Omit unconstrained candidate if it is infeasible
    if gamma_candidate_1[0]<alpha_0 or gamma_candidate_1[1]<np.max([beta_0,theta]):
        candidates = [gamma_candidate_2,gamma_candidate_3]
        strategy_list = ['constrained $\\alpha_1=\\alpha_0$','constrained $\\beta_1=max(\\beta_0,\\theta)$']
    else:
        candidates = [gamma_candidate_1,gamma_candidate_2,gamma_candidate_3]
        strategy_list = ['unconstrained', 'constrained $\\alpha_1=\\alpha_0$','constrained $\\beta_1=max(\\beta_0,\\theta)$']
        
    #print(candidates)
    #Strategy list for explaining remaining candidates
    strategies = [strategy_list[i] for i in range(len(candidates))]
    
    #Evaluate utilities for remaining candidates
    utilities = [U_D(candidates[i],gamma_0,C_1,r,delta) for i in range(len(candidates))]
    index_choice = np.argmax(utilities)

    if max(utilities)<0:
        return np.array([-.1,-.1]),-.1,"abstain"
    
    return candidates[index_choice], utilities[index_choice], strategies[index_choice]
    
def U_G(gamma_0,C_0,C_1,r,delta,theta):
    return delta*np.dot(r,D_strategy(gamma_0,C_1,r,delta,theta)[0]) - np.dot(np.dot(gamma_0,C_0),gamma_0)

def get_candidates(C_0,C_1,delta,r,theta,thetaG,epsilon=0.000001):
    a, b, l = symbols('a b l',real=True)
    C_0_rat = np.array([[Rational(C_0[0,0]),Rational(C_0[0,1])],[Rational(C_0[1,0]),Rational(C_0[1,1])]])
    C_1_rat = np.array([[Rational(C_1[0,0]),Rational(C_1[0,1])],[Rational(C_1[1,0]),Rational(C_1[1,1])]])
    delta_rat = Rational(delta)#0.5
    theta_rat = Rational(theta)
    thetaG_rat = Rational(thetaG)
    r_rat = np.array([Rational(r[0]),Rational(r[1])])
    equation_1 = delta_rat*r_rat[0] - 2*C_0_rat[0,0]*a - 2*C_0_rat[0,1]*b - l*(1-delta_rat)*r_rat[0]
    equation_2 = (delta_rat*C_1_rat[0,1]*r_rat[0]/C_1_rat[0,0]) - 2*C_0_rat[1,1]*b - 2*C_0_rat[0,1]*a - l*(C_1_rat[0,1]*(1-delta_rat)*r_rat[0]/C_1_rat[0,0] - 2*C_1_rat[0,1]**2 * theta_rat/C_1_rat[0,0] + 2*C_1_rat[1,1]*theta_rat + 2*(C_1_rat[0,1]/C_1_rat[0,0]-C_1_rat[1,1])*b)
    term0,termA,termB,termC = get_UD_zero_levelset_terms(C_1_rat,r_rat,delta_rat,theta_rat)
    equation_3 = term0*a + termA*b**2 + termB*b + termC
    raw_results = solve([equation_1,equation_2,equation_3],[a,b,l],domain=S.Reals)#nonlinsolve([equation_1,equation_2,equation_3],[a,b,l],domain=S.Reals)
    raw_results = list(raw_results)
    #print(Float(raw_results,10))
    #print(np.array(raw_results).shape)
    #print(raw_results[0][0])
    results = []
    for i in range(len(raw_results)):
        if isinstance(raw_results[i][0], complex) or isinstance(raw_results[i][1], complex):
            continue
        if ("I" in str(raw_results[i][0]))or("I" in str(raw_results[i][1])):
            continue
        #if N(raw_results[i][0])>1000 or N(raw_results[i][1])>1000:
        #    print("Very large number.")
        #    continue
        results = results + [np.array([N(raw_results[i][j])+epsilon for j in range(2)])]#[np.add(raw_results[i][:2],epsilon)]#[np.array([N(raw_results[i][j])+epsilon for j in range(2)])]
    #results = [np.add(raw_results[i][:2],epsilon) for i in range(len(raw_results))]
    return results

def G_strategy(C_0,C_1,r,delta,theta,thetaG,epsilon=0.000001):
    candidate_1 = 0.5*delta*np.dot(np.linalg.inv(C_0),r)
    candidate_2 = np.array([0,np.max([delta*r[1]/(2*C_0[1,1]),thetaG])])
    candidate_3 = np.array([np.max([0,delta*r[0]/(2*C_0[0,0]) - (C_0[0,1]/C_0[0,0])*thetaG]),thetaG])
    
    term0,termA,termB,termC = get_UD_zero_levelset_terms(C_1,r,delta,theta)
    candidate_5 = [(-1/term0)*(termA*thetaG**2 + termB*thetaG + termC)+epsilon,thetaG+epsilon]
    candidate_6_beta_options = np.real(np.roots([termA,termB,termC]))
    candidate_6_1 = np.array([0+epsilon,candidate_6_beta_options[0]+epsilon])
    candidate_6_2 = np.array([0+epsilon,candidate_6_beta_options[1]+epsilon])
    candidate_7 = np.array([0,thetaG])

    candidate_4_candidates = get_candidates(C_0,C_1,delta,r,theta,thetaG)
    
    candidates_all = [candidate_1, candidate_2,candidate_3,candidate_5,candidate_6_1,candidate_6_2,candidate_7]+candidate_4_candidates
    strategies_all = ['unconstrained','$\\alpha_0=0$','$\\beta_0=\\theta_G$','$U_D=0, \\beta_0=\\theta_G$','$U_D=0, \\alpha_0=0$','$U_D=0, \\alpha_0=0$','$\\alpha_0=0,\\beta_0=\\theta_G$']+["$U_D=0$"]*len(candidate_4_candidates)
    utilities_all = [U_G(g0,C_0,C_1,r,delta,theta) for g0 in candidates_all]

    #Filter infeasible options going back to front
    i = len(candidates_all)-1
    while i>=0:
        #print(strategies_all)
        #for index in np.linspace(len(candidates_all),0):
        #i = int(index)
        #print(i)
        #-0.0001 > term0*alpha_0 + termA*beta_0**2 + termB*beta_0 + termC: 
        if U_D(D_strategy(candidates_all[i],C_1,r,delta,theta)[0],candidates_all[i],C_1,r,delta)< 0:
            candidates_all.pop(i)
            strategies_all.pop(i)
            utilities_all.pop(i)
            #continue
        elif U_G(candidates_all[i],C_0,C_1,r,delta,theta)< 0:
            candidates_all.pop(i)
            strategies_all.pop(i)
            utilities_all.pop(i)
            #continue
        elif (candidates_all[i][0]<0)|(candidates_all[i][1]<thetaG):
            candidates_all.pop(i)
            strategies_all.pop(i)
            utilities_all.pop(i)
            #continue
        i = i-1
    
    if len(candidates_all)<=0:
        return np.array([-.1,-.1]),-.1,"abstain"
    
    choice = np.argmax(utilities_all)
    
    if candidates_all[choice][0]<=0:
        print("Strange case: ")
        print(C_0,C_1,r,delta,theta,thetaG)
        print("Choice: ",candidates_all[choice], utilities_all[choice], strategies_all[choice])
        print("All candidates: ",candidates_all)
        print("All utilities: ",utilities_all)
        print("All strategies: ",strategies_all)
        print("D response and utility: ",D_strategy(candidates_all[i],C_1,r,delta,theta))
    
    return candidates_all[choice], utilities_all[choice], strategies_all[choice]

def process_theta_pair(args):
    """Process a single (thetaG, thetaD) pair"""
    gt, t, C_0, C_1, r, delta = args
    
    if t < gt:
        return None
    
    gamma_0, UG, strategyG = G_strategy(C_0, C_1, r, delta, t, gt)
    gamma_1, UD, strategyD = D_strategy(gamma_0, C_1, r, delta, t)
    alpha_0 = gamma_0[0]
    beta_0 = gamma_0[1]
    alpha_1 = gamma_1[0]
    beta_1 = gamma_1[1]
    
    # Calculate backfiring flag
    if t == gt:
        beta_1_anarchy = beta_1
    else:
        # For non-anarchy cases, we need to calculate anarchy case separately
        gamma_0_anarchy, _, _ = G_strategy(C_0, C_1, r, delta, gt, gt)
        gamma_1_anarchy, _, _ = D_strategy(gamma_0_anarchy, C_1, r, delta, gt)
        beta_1_anarchy = gamma_1_anarchy[1]
    
    backfiring_flag = 1 if beta_1 < beta_1_anarchy else 0
    
    return {
        'C0': str(C_0),
        'C1': str(C_1),
        'r': str(r),
        'delta': delta,
        'theta': t,
        'thetaG': gt,
        'alpha_0': float(alpha_0),
        'beta_0': float(beta_0),
        'alpha_1': float(alpha_1),
        'beta_1': float(beta_1),
        'U_G': float(UG),
        'U_D': float(UD),
        'Gstrategy': strategyG,
        'Dstrategy': strategyD,
        'backfiring_flag': backfiring_flag
    }

def create_results_df(C_0, C_1, r, delta,thetaGs,thetaDs, output_file_name=None):
    G_strategy_results = pd.DataFrame(columns=['C0','C1','r','delta','theta','thetaG','alpha_0','alpha_1','beta_1','U_G','U_D','Gstrategy','Dstrategy','backfiring_flag'])
    G_strategy_results['C0']=[None]*len(thetaGs)*len(thetaDs)

    # Create list of all parameter combinations
    param_combinations = []
    for gt in thetaGs:
        for t in thetaDs:
            param_combinations.append((gt, t, C_0, C_1, r, delta))

    # Set up parallel processing
    n_processes = min(96, mp.cpu_count())
    print(f"Using {n_processes} processes")

    # Process in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_theta_pair, param_combinations)

    # Filter out None results and convert to DataFrame
    valid_results = [result for result in results if result is not None]
    G_strategy_results_parallel = pd.DataFrame(valid_results)

    # Save to CSV
    if output_file_name is not None:
        output_path = Path(output_file_name)
        if not output_path.is_absolute():
            output_path = OUTPUT_DIR / output_path
    else:
        output_path = OUTPUT_DIR / 'backup_nobargaining_results_parallel_june5.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    G_strategy_results_parallel.to_csv(output_path, index=False)
    
    print(f"Parallel processing complete. Generated {len(G_strategy_results_parallel)} results.")

if __name__ == '__main__':
    #freeze_support()
    crossterms = [-.1,0,.1]

    for ct in crossterms:
        C_0 = np.array([[1,ct],[ct,1]])
        C_1 = np.array([[1,ct],[ct,1]])
        delta = 0.5
        r = np.array([1,1])
        thetaGs = np.linspace(0,1.5,301)#76)
        thetaDs = np.linspace(0,2.5,501)#126)
        create_results_df(C_0, C_1, r, delta,thetaGs,thetaDs, output_file_name=f'nobargaining_results_parallel_june6_crossterm_{ct}.csv')
