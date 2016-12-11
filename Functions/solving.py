from __future__ import division
import numpy as np
import scipy as scipy
import scipy.sparse as ss
#import scipy.sparse.linalg as ssl
import scipy.linalg as sl
import ctypes
import os.path
import cvxpy
import cvxopt

def expit(x):
    return np.exp(x) / (1 + np.exp(x))

## hypothesis     : "null" or "alternative"
## treatment      : a 0/1 input vector identifying the treatment group for each observation in x. 
## y       : an input vector showing the value of the continuous response variable for each observation in x. 
def setup(hypothesis, y, x, treatment=None, lam_control = None, lam_treat = None, lam_alt = None, scalar_frobenius=0.005, max_it=50000):
    # store dimensions of design matrix and ordering of features
    n = np.shape(x)[0]
    o = np.argsort(x, axis=0)
    if x.ndim == 1:
        #list_permutations = [permutation_matrix(x, sparse=sparse)]
        p = 1
        o = o.reshape(n, 1)
    elif x.ndim > 1:
        #list_permutations = list_permutation_matrices(x, sparse=sparse)
        p = np.shape(x)[1]
    # build first-differences matrix 
    e = np.mat(np.ones((1, n)))
    D1_1 = ss.spdiags(np.vstack((e, -e)), range(2), n-1, n)
    D_coo = D1_1.tocoo()
    D_cvxpy = cvxopt.spmatrix(D_coo.data, D_coo.row.tolist(), D_coo.col.tolist())
    # build permutation matrix and fused lasso penalty (permutations matrix multipled by first-differences matrix) that will be applied to y vector
    list_permutation_matrices = []
    list_ordered_fused_penalty = []
    for j in range(p):
        ind_ord = o[:, j]
        permutation_mat = np.zeros((n,n))
        for i in range(n):
            permutation_mat[i, o[i, j]] = 1
        #if np.array_equal(np.dot(permutation_mat, X[:, j]), np.sort(X[:, j])) == False:
            #raise ValueError("permutation matrix is not correct")
        list_permutation_matrices.append(cvxopt.matrix(permutation_mat))
        list_ordered_fused_penalty.append(D_cvxpy * cvxopt.matrix(permutation_mat))
    # return objects specific to whether we fitting under the null (non-crossing) or alternative
    if hypothesis == 'null':
        ind_control = np.flatnonzero(treatment == 0)
        ind_treat = np.flatnonzero(treatment == 1)
        ind_ordered_control = np.argsort(x[ind_control], axis=0)
        ind_ordered_treat = np.argsort(x[ind_treat], axis=0)
        I_treat = ss.spdiags(treatment, 0, n, n)
        I_control = ss.spdiags((1 - treatment), 0, n, n)
        X_control = I_control.dot(x)
        X_treat = I_treat.dot(x)
        Y_control = I_control.dot(y)
        Y_treat = I_treat.dot(y)
        return{'scalar_frobenius':scalar_frobenius, 'max_it':max_it, "ind_ordered":o, "ind_ordered_control":ind_ordered_control, "ind_ordered_treat":ind_ordered_treat, 'list_ordered_fused_penalty': list_ordered_fused_penalty, 'n': n, 'p' : p, 'lam_control': lam_control, 'lam_treat' : lam_treat, 'ind_control' : ind_control, 'ind_treat' : ind_treat, 'n_control' : ind_control.size, 'n_treat' : ind_treat.size, 'I_control' : I_control, 'I_treat' : I_treat, 'X_control' : X_control, 'X_treat' : X_treat,'Y_control' : Y_control, 'Y_treat' : Y_treat, 'y' : y}
    elif hypothesis == 'alternative':
        return{'scalar_frobenius':scalar_frobenius, 'max_it':max_it, "ind_ordered":o, 'list_ordered_fused_penalty': list_ordered_fused_penalty, 'p' : p, 'lam_alt' : lam_alt, 'n': n, 'y' : y}
    else:
        raise NameError('invalid hypothesis specified')

def fused_lasso(resp, fit, acting_lam, n):
    #so_abs_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'Fit.so' 
    so_abs_path = os.getcwd()+os.path.sep+'Fit.so' ### Comment out this line and uncomment the previous one if run as a script
    univFit = ctypes.CDLL(so_abs_path)
    c_fit = fit.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    univFit.tf_dp_R(ctypes.pointer(ctypes.c_int(n)),
                    resp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.pointer(ctypes.c_double(acting_lam)),
                    c_fit)


# FUNCTION: Set up optimzation problem under the null in cvxpy and return optimal fitted values in control and treatment groups
def fitted_null_cvxpy(response_type, attrib, thresh, use_frobenius=False, scalar_frobenius=0.005, verbose=False):
    list_loss, list_theta_treat, list_theta_control, list_theta_matrix_control, list_theta_matrix_treat, list_intercept_control, list_intercept_treat, list_fitted_control, list_fitted_treat = [], [], [], [], [], [], [], [], []
    for constraint_type in ["control_above", "treat_above"]:
        theta_matrix_control = cvxpy.Variable(attrib['n'], attrib['p'])
        theta_matrix_treat = cvxpy.Variable(attrib['n'], attrib['p'])
        ones_p = cvxopt.matrix(np.ones(attrib['p']))
        ones_n = cvxopt.matrix(np.ones(attrib['n']))
        intercept_control = cvxpy.Variable()
        intercept_treat = cvxpy.Variable()
        ones_control = cvxopt.matrix(np.ones((attrib['n_control'], 1)))
        ones_treat = cvxopt.matrix(np.ones((attrib['n_treat'], 1)))
        I_control = cvxopt.matrix(np.identity(attrib['n_control']))
        I_treat = cvxopt.matrix(np.identity(attrib['n_treat']))
        select_control = cvxopt.matrix(cvxopt.spmatrix(1, range(attrib['n_control']), attrib['ind_control'], size=(attrib['n_control'], attrib['n'])))
        select_treat = cvxopt.matrix(cvxopt.spmatrix(1, range(attrib['n_treat']), attrib['ind_treat'], size=(attrib['n_treat'], attrib['n'])))
        theta_matrix_control_only = select_control * theta_matrix_control
        theta_matrix_treat_only = select_treat * theta_matrix_treat
        Y_cvxopt = cvxopt.matrix(attrib['y'].reshape(attrib['n'], 1))
        Y_cvxopt_control_only = select_control * Y_cvxopt
        Y_cvxopt_treat_only = select_treat * Y_cvxopt
        # define loss functions
        if response_type == "continuous":
            loss_control = 0.5 * cvxpy.quad_form(Y_cvxopt_control_only - (theta_matrix_control_only * ones_p + intercept_control * ones_control), I_control)
            loss_treat = 0.5 * cvxpy.quad_form(Y_cvxopt_treat_only - (theta_matrix_treat_only * ones_p + intercept_treat * ones_treat), I_treat)
        elif response_type == "binary":
            loss_components_control = [cvxpy.log_sum_exp(cvxpy.vstack(0, cvxpy.sum_entries(theta_matrix_control_only[i, :]) + intercept_control)) - Y_cvxopt_control_only[i, 0] * (cvxpy.sum_entries(theta_matrix_control_only[i, :]) + intercept_control) for i in range(attrib['n_control'])]
            loss_components_treat = [cvxpy.log_sum_exp(cvxpy.vstack(0, cvxpy.sum_entries(theta_matrix_treat_only[i, :]) + intercept_treat)) - Y_cvxopt_treat_only[i, 0] * (cvxpy.sum_entries(theta_matrix_treat_only[i, :]) + intercept_treat) for i in range(attrib['n_treat'])]
            loss_control = sum(loss_components_control)
            loss_treat = sum(loss_components_treat)
        # define penalty terms
        penalty_components_control = [cvxpy.norm(attrib["list_ordered_fused_penalty"][col] * theta_matrix_control[:, col], 1)  for col in range(attrib['p'])]
        penalty_components_treat = [cvxpy.norm(attrib["list_ordered_fused_penalty"][col] * theta_matrix_treat[:, col], 1)  for col in range(attrib['p'])]
        penalty = attrib['lam_control'] * sum(penalty_components_control) + attrib['lam_treat'] * sum(penalty_components_treat)
        if use_frobenius == True:
            # define objective and solve problem
            objective = cvxpy.Minimize(loss_control + loss_treat + penalty + scalar_frobenius * cvxpy.sum_squares(theta_matrix_control) + scalar_frobenius * cvxpy.sum_squares(theta_matrix_treat))
            prob = cvxpy.Problem(objective)
        else:
            # define objective, constraints, and solve problem
            objective = cvxpy.Minimize(loss_control + loss_treat + penalty)
            if constraint_type == "control_above":
                constraints = [theta_matrix_control * ones_p + intercept_control * ones_n >= theta_matrix_treat * ones_p + intercept_treat * ones_n]
            elif constraint_type == "treat_above":
                constraints = [theta_matrix_control * ones_p + intercept_control * ones_n < theta_matrix_treat * ones_p + intercept_treat * ones_n]
            for col in range(attrib['p']):
                constraints.append(cvxpy.sum_entries(theta_matrix_control[:, col]) == 0)
                constraints.append(cvxpy.sum_entries(theta_matrix_treat[:, col]) == 0)
            prob = cvxpy.Problem(objective, constraints)
        result = prob.solve(verbose=verbose, solver="SCS", max_iters=attrib['max_it'], eps=thresh)
        # store results
        one_fitted_control = np.dot(np.array(theta_matrix_control.value), np.array(ones_p)) + intercept_control.value * ones_n
        one_fitted_treat = np.dot(np.array(theta_matrix_treat.value), np.array(ones_p)) + intercept_treat.value * ones_n
        list_loss.append(loss_control.value + loss_treat.value)
        list_fitted_control.append(np.array(one_fitted_control))
        list_fitted_treat.append(np.array(one_fitted_treat))
        list_theta_matrix_control.append(theta_matrix_control.value)
        list_theta_matrix_treat.append(theta_matrix_treat.value)
        list_intercept_control.append(intercept_control.value)
        list_intercept_treat.append(intercept_treat.value)
    if list_loss[0] <= list_loss[1]:
        theta_matrix_control = np.array(list_theta_matrix_control[0]).reshape(attrib['n'], attrib['p'])
        theta_matrix_treat = np.array(list_theta_matrix_treat[0]).reshape(attrib['n'], attrib['p'])
        intercept_control = np.array(list_intercept_control[0])
        intercept_treat = np.array(list_intercept_treat[0])
        fitted_control = np.array(list_fitted_control[0]).reshape(attrib['n'], 1)
        fitted_treat = np.array(list_fitted_treat[0]).reshape(attrib['n'], 1)
    else:
        theta_matrix_control = np.array(list_theta_matrix_control[1]).reshape(attrib['n'], attrib['p'])
        theta_matrix_treat = np.array(list_theta_matrix_treat[1]).reshape(attrib['n'], attrib['p'])
        intercept_control = np.array(list_intercept_control[1])
        intercept_treat = np.array(list_intercept_treat[1])
        fitted_control = np.array(list_fitted_control[1]).reshape(attrib['n'], 1)
        fitted_treat = np.array(list_fitted_treat[1]).reshape(attrib['n'], 1)        
    if response_type == "binary":
        fitted_control = expit(fitted_control)
        fitted_treat = expit(fitted_treat)
    return{"theta_matrix_control":theta_matrix_control, "intercept_control":intercept_control, "fitted_control":fitted_control,
             "theta_matrix_treat":theta_matrix_treat, "intercept_treat":intercept_treat, "fitted_treat":fitted_treat}

# FUNCTION: Set up optimzation problem *under the alternative* in cvxpy and return optimal fitted values for the specified group
def fitted_alt_cvxpy(response_type, attrib, thresh, use_frobenius=False, scalar_frobenius=0.005, verbose=False):
    # get loss function
    theta_matrix = cvxpy.Variable(attrib['n'], attrib['p'])
    intercept = cvxpy.Variable()
    ones_p = cvxopt.matrix(np.ones(attrib['p']))
    ones_n = cvxopt.matrix(np.ones(attrib['n']))
    #I_n = cvxopt.spmatrix(1,  range(attrib['n']), range(attrib['n']))
    I_n = np.identity(attrib['n'])
    Y_cvxopt = cvxopt.matrix(attrib['y'].reshape(attrib['n'], 1))
    # get loss function
    if response_type == "continuous":
        loss = 0.5 * cvxpy.quad_form(Y_cvxopt - (theta_matrix * ones_p + intercept * ones_n), I_n)
    elif response_type == "binary":
        loss_components = [cvxpy.log_sum_exp(cvxpy.vstack(0, cvxpy.sum_entries(theta_matrix[i, :]) + intercept)) - Y_cvxopt[i, 0] * (cvxpy.sum_entries(theta_matrix[i, :]) + intercept) for i in range(attrib['n'])]
        loss = sum(loss_components)
    penalty_components = [cvxpy.norm(attrib["list_ordered_fused_penalty"][col] * theta_matrix[:, col], 1)  for col in range(attrib['p'])]
    penalty = attrib['lam_alt'] * sum(penalty_components)
    # define objective function and solve optimization problem
    if use_frobenius == True:
        objective = cvxpy.Minimize(loss + penalty + scalar_frobenius * cvxpy.sum_squares(theta_matrix))
        prob = cvxpy.Problem(objective)
    else:
        objective = cvxpy.Minimize(loss + penalty)
        constraints = [cvxpy.sum_entries(theta_matrix[:, col]) == 0 for col in range(attrib['p'])]
        prob = cvxpy.Problem(objective, constraints)
    result = prob.solve(verbose=verbose, solver="SCS", max_iters=50000, eps=thresh)  
    # save and return converged results
    theta_matrix = np.array(theta_matrix.value).reshape(attrib['n'], attrib['p'])
    intercept = np.array(intercept.value)
    if response_type == "continuous":
        fitted = np.dot(theta_matrix, np.array(ones_p)) + intercept * ones_n
    elif response_type == "binary":
        fitted = expit(np.dot(theta_matrix, np.array(ones_p)) + intercept * ones_n)
    return{"theta_matrix":theta_matrix, "intercept":intercept, "fitted":fitted}

def fitted_alt_ggd_fused(Y, lam, ggd_thresh, ggd_theta_init=None, ggd_L=0.25):
    n = Y.size
    if ggd_theta_init is None:
        theta = np.array(np.zeros(n))
    else:
        theta = ggd_theta_init
    theta_old = theta + 1  # just so convergence criteria not satisfied on first iteration
    it = 0
    while sl.norm(theta - theta_old, 2) > ggd_thresh:
        theta_old = np.copy(theta) 
        prob = expit(theta)
        theta_resp = theta + (1 / ggd_L) * (Y - prob)
        fused_lasso(resp=theta_resp, fit=theta, acting_lam=lam / ggd_L, n=n)
        it = it + 1
    return expit(theta)


## produce n_group x n matrix to pick out entries corresponding to group (needed for solving fused under null with cvxpy)
def select_matrix(indices, n):
    total_group = np.shape(indices)[0]
    my_I = np.zeros(shape=(total_group, n))
    for i in range(total_group):
            my_I[i, indices[i]] = 1
    return my_I

# function to get sparse permutation matrix
def permutation_matrix(x, sparse=True):
    n = np.size(x)
    ordered_idx_x = np.argsort(x)
    permutation_mat = np.zeros((n, n))
    for i in range(n):
        permutation_mat[i, ordered_idx_x[i]] = 1
    if sum(np.dot(permutation_mat, x) == np.sort(x)) != n:
        raise ValueError("permutation matrix was incorrect")
    if sparse == True:
        permutation_mat_csr = ss.csr_matrix(permutation_mat)
        permutation_mat_coo = permutation_mat_csr.tocoo()
        permutation_cvxpy = cvxopt.spmatrix(permutation_mat_coo.data, permutation_mat_coo.row.tolist(), permutation_mat_coo.col.tolist())
        return permutation_cvxpy
    else:
        return permutation_mat

# FUNCTION: Return list of permutation matrices, one for each feature in deisgn matrix
def list_permutation_matrices(X, sparse=True):
    my_list = []
    if X.ndim == 1:
        my_list.append(permutation_matrix(X, sparse=sparse))
    elif X.ndim == 2:
        for col in range(np.shape(X)[1]):
            my_list.append(permutation_matrix(X[:, col], sparse=sparse))
    return my_list

# FUNCTION: Return a list of sparse matrices that will return a vector of ordered first differences when applied to each feature
def list_ordered_first_differences(D_cvxpy, list_permutations):
    my_list = []
    for permutation_mat in list_permutations:
        my_list.append(D_cvxpy * permutation_mat)
    return my_list

# FUNCTION: Return thle loss function for additive modeling of either a continuous or binary response
def get_loss_function(response_type, y, p, n_overall, n_group, I_group, theta_matrix_group, ind_group, select_group, intercept_group):
    if response_type == "continuous":
        ones_p = cvxopt.matrix(np.ones(p))
        ones_n = cvxopt.matrix(np.ones(n_overall))
        loss_function = 0.5 * cvxpy.quad_form(select_group * (y - theta_matrix_group * ones_p - intercept_group * ones_n), I_group)
    elif response_type == "binary":
        entries = [cvxpy.log_sum_exp(cvxpy.vstack(0, cvxpy.sum_entries(theta_matrix_group[i, :]) + intercept_group)) - y[i] * (cvxpy.sum_entries(theta_matrix_group[i, :]) + intercept_group) for i in ind_group]
        loss_function = sum(entries)
    else:
        raise NameError("invalid response_type specified")
    return loss_function


