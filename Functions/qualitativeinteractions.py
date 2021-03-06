from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxopt as cvxopt
import cvxpy
import math
import pandas as pd
import statsmodels.api as sm ## requires pandas and patsy to be installed
import scipy
import solving as solving ## our solvers


## FUNCTION: Return maximum lambda value for specified penalized regression method
## INPUT:
## OUTPUT:
def max_lambda(Y, X, form):
    n = np.shape(Y)[0]
    if form == "continuous":
        if X.ndim == 1:
            order = np.argsort(X)
            max_lam = np.max(np.abs(np.cumsum(Y[order] - np.mean(Y))))
        elif X.ndim == 2:
            p = np.shape(X)[1]
            max_lam_list = []
            for col in range(p):
                one_order = np.argsort(X[:, col])
                max_lam_list.append(np.max(np.abs(np.cumsum(Y[one_order] - np.mean(Y)))))
            max_lam = np.max(max_lam_list)
    elif form == "categorical":
        max_lam = np.linalg.norm(X, ord='fro') ** 2
    return max_lam

## FUNCTION fitted_values(): Solve the optimization problem, either under the null (no qualitative interaction allowed) or the alternative
def fitted_values(hypothesis, response_type, X_continuous, Y, ggd_thresh, thresh,
                       X_categorical=None,
                       treatment=None, lam_alt_continuous=None, lam_alt_categorical=None, lam_continuous_control=None, lam_continuous_treat=None, 
                       lam_categorical_control=None, lam_categorical_treat=None,
                       max_it=50000, use_frobenius=False, scalar_frobenius=0.001, verbose=False, ggd_theta_init=None, ggd_L=0.25):
    if X_categorical is not None:
        if X_categorical.ndim == 1:
            X_categorical = np.array(pd.get_dummies(X_categorical))
        else:
            X_categorical = np.concatenate([pd.get_dummies(np.array(X_categorical[:, j])) for j in range(np.shape(X_categorical)[1])], axis=1)
    if hypothesis == "null":
        attributes_null = solving.setup(hypothesis="null", y=Y, treatment=treatment, 
                                        x_continuous=X_continuous, lam_continuous_control=lam_continuous_control, lam_continuous_treat=lam_continuous_treat, 
                                        x_categorical=X_categorical, lam_categorical_control=lam_categorical_control, lam_categorical_treat=lam_categorical_treat, 
                                        scalar_frobenius=scalar_frobenius, max_it=max_it)
        result_null = solving.fitted_null_cvxpy(response_type=response_type, attrib=attributes_null, verbose=verbose, thresh=thresh, use_frobenius=use_frobenius, scalar_frobenius=attributes_null['scalar_frobenius'])
        if attributes_null['p_categorical'] > 0:
            return{"theta_result_control" : result_null["fitted_control"], "theta_matrix_control":result_null["theta_matrix_control"], "intercept_control":np.reshape(result_null["intercept_control"], (1, )),
                    "theta_result_treat" : result_null["fitted_treat"], "theta_matrix_treat":result_null["theta_matrix_treat"], "intercept_treat":np.reshape(result_null["intercept_treat"], (1, )),
                    "beta_vec_control" : result_null["beta_vec_control"], "beta_vec_treat" : result_null["beta_vec_treat"], "p_categorical":attributes_null['p_categorical']}
        else:
            return{"theta_result_control" : result_null["fitted_control"], "theta_matrix_control":result_null["theta_matrix_control"], "intercept_control":np.reshape(result_null["intercept_control"], (1, )),
                    "theta_result_treat" : result_null["fitted_treat"], "theta_matrix_treat":result_null["theta_matrix_treat"], "intercept_treat":np.reshape(result_null["intercept_treat"], (1, ))}
    elif hypothesis == "alternative":
        attributes_alt = solving.setup(hypothesis="alternative", y=Y, x_continuous=X_continuous, x_categorical=X_categorical, treatment=None,
                                       lam_alt_continuous=lam_alt_continuous, lam_alt_categorical=lam_alt_categorical)
        if attributes_alt['p'] == 1 and response_type == "continuous":
            ## solved with fused_lasso()
            ind_ordered_X = attributes_alt["ind_ordered"].reshape(attributes_alt['n'], )
            ordered_theta = np.zeros(attributes_alt['n'])
            solving.fused_lasso(resp=Y[ind_ordered_X], fit=ordered_theta, acting_lam=lam_alt_continuous, n=attributes_alt['n'])
            theta_result = np.zeros(attributes_alt['n'])
            theta_result[ind_ordered_X] = ordered_theta ## returns estimated theta *with original ordering*
            return{"ind_ordered_X":ind_ordered_X, "theta_result":theta_result, "theta_matrix":theta_result, "intercept":np.reshape(0, (1, ))}
        #elif X.ndim == 1 and response_type == "binary":
        elif attributes_alt['p'] == 1 and response_type == "binary":
            ## solve with generalized gradient descent
            ind_ordered_X = attributes_alt["ind_ordered"].reshape(attributes_alt['n'], )
            Y = Y.reshape(attributes_alt['n'], )
            ordered_theta = solving.fitted_alt_ggd_fused(Y[ind_ordered_X], lam=lam_alt_continuous, ggd_theta_init=ggd_theta_init, ggd_L=ggd_L, ggd_thresh=ggd_thresh)
            theta_result = np.zeros(attributes_alt['n'])
            theta_result[ind_ordered_X] = ordered_theta ## returns estimated theta *with original ordering*
            return{"ind_ordered_X":ind_ordered_X, "theta_result":theta_result, "theta_matrix":theta_result, "intercept":np.reshape(0, (1, ))}
        #elif X.ndim == 2:
        elif attributes_alt['p'] > 1:
            result = solving.fitted_alt_cvxpy(response_type=response_type, attrib=attributes_alt, thresh=thresh, use_frobenius=use_frobenius, scalar_frobenius=attributes_alt['scalar_frobenius'], verbose=verbose)
            # *fitted value* returned on probability scale for binary response, but *not* theta matrix
            if attributes_alt['p_categorical'] > 1:
                return{"beta_vec":result["beta_vec"], "theta_result":result["fitted"], "theta_matrix":result["theta_matrix"], "intercept":np.reshape(result["intercept"], (1, ))}
            else:
                return{"theta_result":result["fitted"], "theta_matrix":result["theta_matrix"], "intercept":np.reshape(result["intercept"], (1, ))}
        else:
            raise ValueError("Invalid response type specified")

## FUNCTION fitted_values_cv(): Solve the optimization problem in the setting of cross-validation
def fitted_values_cv(response_type, Y, n, p, ind_training, ind_test,  ggd_thresh, thresh, 
                            X_continuous, lam_continuous,
                            X_categorical=None, lam_categorical=None, 
                            max_it=50000, verbose=False, use_frobenius=False, scalar_frobenius=0.001,
                            ggd_theta_init=None, ggd_L=0.25):
    if p == 1:
        X_continuous = np.reshape(X_continuous, (n, ))
        fitted_training = fitted_values(hypothesis="alternative", response_type=response_type, X_continuous=X_continuous[ind_training], Y=Y[ind_training], lam_alt_continuous=lam_continuous, verbose=verbose,
        use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius,
        ggd_theta_init=ggd_theta_init, ggd_L=ggd_L, ggd_thresh=ggd_thresh, thresh=thresh)
        fitted_test = np.interp(X_continuous[ind_test], X_continuous[ind_training][fitted_training["ind_ordered_X"]], fitted_training["theta_result"][fitted_training['ind_ordered_X']])
        fitted_test = fitted_test.reshape(fitted_test.size, 1)
        resid_test = get_resid(response_type=response_type, actual=Y[ind_test], predicted=fitted_test)
        return{'theta_matrix_test':fitted_test, 'predicted_test':fitted_test, "resid_test":resid_test, "intercept":0}
    #elif X.ndim == 2:
    elif p > 1:
        ## now the binary treatment indicator should be replaced with a binary indicator of being in the training set
        test_indicator = np.zeros(np.shape(X_continuous)[0])
        test_indicator[ind_test] = 1
        attrib_cv = solving.setup(hypothesis="null", y=Y, treatment=test_indicator,
        x_continuous=X_continuous, x_categorical=X_categorical)
        ## minimize (residuals in the training set *only* + penalty function for *all* observations), to use the fitted values in the test set to predict
        ## I'm using most of the same code as in fitted_null_cvxpy(), with "control" standing in for "training" and "treat" standing in for "test"
        theta_matrix_complete = cvxpy.Variable(attrib_cv['n'], attrib_cv['p_continuous'])
        intercept_complete = cvxpy.Variable()
        ones_p_continuous = cvxopt.matrix(np.ones(attrib_cv['p_continuous']))
        ones_n = cvxopt.matrix(np.ones(attrib_cv['n']))
        ones_training = cvxopt.matrix(np.ones((attrib_cv['n_control'], 1)))
        I_training = cvxopt.matrix(np.identity(attrib_cv['n_control']))
        select_training = cvxopt.matrix(cvxopt.spmatrix(1, range(attrib_cv['n_control']), attrib_cv['ind_control'], size=(attrib_cv['n_control'], attrib_cv['n'])))
        theta_matrix_training_only = select_training * theta_matrix_complete
        Y_cvxopt = cvxopt.matrix(attrib_cv['y'].reshape(attrib_cv['n'], 1))
        Y_cvxopt_training_only = select_training * Y_cvxopt
        
        # initializing cvxpy objects for categorical features
        if attrib_cv['p_categorical'] > 0:
            beta_vec_training = cvxpy.Variable(attrib_cv['p_categorical'], 1)
            X_categorical_cvxopt = cvxopt.matrix(attrib_cv['X_categorical'].reshape(attrib_cv['n'], attrib_cv['p_categorical']))
            X_categorical_cvxopt_training_only = select_training * X_categorical_cvxopt

        if response_type == "continuous":
            if attrib_cv['p_categorical'] > 0:
                loss_training = 0.5 * cvxpy.quad_form(Y_cvxopt_training_only - (theta_matrix_training_only * ones_p_continuous + intercept_complete * ones_training +
                                                                     X_categorical_cvxopt_training_only * beta_vec_training), I_training)
            else:
                loss_training = 0.5 * cvxpy.quad_form(Y_cvxopt_training_only - (theta_matrix_training_only * ones_p_continuous + intercept_complete * ones_training), I_training)
        elif response_type == "binary":
            if attrib_cv['p_categorical'] > 0:
                loss_components_training = [cvxpy.log_sum_exp(cvxpy.vstack(0, cvxpy.sum_entries(theta_matrix_training_only[i, :]) + intercept_complete + 
                (X_categorical_cvxopt_training_only * beta_vec_training)[i, :])) - 
                Y_cvxopt_training_only[i, 0] * (cvxpy.sum_entries(theta_matrix_training_only[i, :]) + intercept_complete + 
                (X_categorical_cvxopt_training_only * beta_vec_training)[i, :]) for i in range(attrib_cv['n_control'])]
            else:
                loss_components_training = [cvxpy.log_sum_exp(cvxpy.vstack(0, cvxpy.sum_entries(theta_matrix_training_only[i, :]) + intercept_complete)) - Y_cvxopt_training_only[i, 0] * (cvxpy.sum_entries(theta_matrix_training_only[i, :]) + intercept_complete) for i in range(attrib_cv['n_control'])]
            loss_training = sum(loss_components_training)
        penalty_components = [cvxpy.norm(attrib_cv["list_ordered_fused_penalty"][col] * theta_matrix_complete[:, col], 1)  for col in range(attrib_cv['p_continuous'])]
        penalty_continuous = lam_continuous * sum(penalty_components)
        if attrib_cv['p_categorical'] > 0:
            penalty_categorical = lam_categorical * cvxpy.sum_squares(beta_vec_training)
            penalty = penalty_continuous + penalty_categorical
        else:
            penalty = penalty_continuous
        if use_frobenius == True:
            objective = cvxpy.Minimize(loss_training + penalty + scalar_frobenius * cvxpy.sum_squares(theta_matrix_complete))
            prob = cvxpy.Problem(objective)
        else:
            objective = cvxpy.Minimize(loss_training + penalty)
            constraints = [cvxpy.sum_entries(theta_matrix_complete[:, col]) == 0 for col in range(attrib_cv['p_continuous'])]
            prob = cvxpy.Problem(objective, constraints)
        result = prob.solve(verbose=verbose, solver="SCS", max_iters=attrib_cv['max_it'], eps=thresh)  
        # save converged results and compute test error
        if response_type == "continuous":
            if attrib_cv['p_categorical'] > 0:
                fitted_complete = np.dot(np.array(theta_matrix_complete.value), np.array(ones_p_continuous)) + intercept_complete.value * ones_n + np.array(X_categorical_cvxopt * beta_vec_training.value)
            else:
                fitted_complete = np.dot(np.array(theta_matrix_complete.value), np.array(ones_p_continuous)) + intercept_complete.value * ones_n
        elif response_type == "binary":
            if attrib_cv['p_categorical'] > 0:
                fitted_complete = solving.expit(np.dot(np.array(theta_matrix_complete.value), np.array(ones_p_continuous)) + intercept_complete.value * ones_n + np.array(X_categorical_cvxopt * beta_vec_training.value))
            else:
                fitted_complete = solving.expit(np.dot(np.array(theta_matrix_complete.value), np.array(ones_p_continuous)) + intercept_complete.value * ones_n)
        fitted_test = fitted_complete[ind_test]
        resid_test = get_resid(response_type=response_type, actual=Y[ind_test], predicted=fitted_test)
        theta_matrix_complete = np.array(theta_matrix_complete.value)
        theta_matrix_test = theta_matrix_complete[ind_test]
        if response_type == "binary" and np.any(np.array(fitted_test) < 0):
            raise ValueError("IN FITTED_VALUES_CV: a predicted probability is less than 0! i must be forgetting an expit() somewhere...")
        return{'theta_marix_complete':theta_matrix_complete, 'intercept':np.array(intercept_complete.value), 
        "theta_matrix_test":theta_matrix_test, 'predicted_test':fitted_test, "resid_test":resid_test}

###################
###  K-Fold Cross-Validation ###
###################

## FUNCTION partition_k_fold(): Randomly partition indicies into k roughly evenly sized folds
def partition_k_folds(indices, k=10):
    count = np.size(indices)
    fold_size = np.int(np.floor(count / k))
    shuffled = np.random.permutation(indices)
    permutation = np.array_split(shuffled, k)  ## already deals with remainder!
    return permutation

## FUNCTION loop_k_fold_cv(): Run k-fold cross-validation and return list of predicted negative log-resids corresponding to given lambda values
    # get permuted folds (only want to run once, before looping over lambda grid)
def loop_k_fold_cv(response_type, k, Y, ggd_thresh, thresh, 
                           X_continuous, grid_lam_continuous, 
                           X_categorical=None, grid_lam_categorical=None,
                           max_it=50000, use_frobenius=False, scalar_frobenius=0.001,
                           ggd_theta_init=None, ggd_L=0.25, verbose=False):
    n_obs = np.shape(X_continuous)[0]
    ind_complete = range(n_obs)
    list_ind_test = partition_k_folds(indices=range(n_obs), k=k)
    # get all k training sets, so we don't re-compute them for each lambda value
    list_ind_training = [np.setdiff1d(ind_complete, list_ind_test[j]) for j in range(k)]
    # Across the grid of lambda vales, call k_fold_cv and return predicted resid
    quick_attributes = solving.setup(hypothesis="null", y=Y, x_continuous=X_continuous, x_categorical=X_categorical, quick=True)
    if quick_attributes['p_categorical'] > 0:
        mat_lam = np.array([(x, y) for x in grid_lam_continuous for y in grid_lam_categorical])
    else:
        mat_lam = grid_lam_continuous
    grid_resid, grid_theta_matrix, grid_fitted_test, grid_intercept_vec  = [], [], [], []
    for mat_lam_entry in mat_lam:
        if quick_attributes['p_categorical'] > 0:
            lam_val_continuous, lam_val_categorical = mat_lam_entry[0], mat_lam_entry[1]
        else:
            lam_val_continuous = mat_lam_entry
            lam_val_categorical = None
        resid_test_components = []
        fitted_test = np.zeros([n_obs, ])
        intercept_vec_test = np.zeros([n_obs, ])
        if quick_attributes['p_continuous'] == 1:
            theta_matrix_test = np.zeros([n_obs, ])
        elif quick_attributes['p_continuous'] > 1:
            theta_matrix_test = np.zeros([n_obs, quick_attributes['p_continuous']])
        # if quick_attributes['p_categorical'] > 0:
        #     beta_mat = np.zeros([n_obs, quick_attributes['p_categorical']])
        # if X.ndim == 1:
        #     theta_matrix_test = np.zeros([n_obs, ])
        #     p = 1
        # elif X.ndim == 2:
        #     p = np.shape(X)[1]
        #     theta_matrix_test = np.zeros([n_obs, np.shape(X)[1]])
        for j in range(k):
            one_fit = fitted_values_cv(response_type=response_type, Y=Y, n=n_obs, p=quick_attributes['p'], 
                                                        X_continuous=X_continuous, lam_continuous=lam_val_continuous, 
                                                        X_categorical=X_categorical, lam_categorical=lam_val_categorical,
                                                        ind_training=list_ind_training[j], 
                                                        ind_test=list_ind_test[j],
                                                        ggd_theta_init=ggd_theta_init, ggd_L=ggd_L, ggd_thresh=ggd_thresh, thresh=thresh,
                                                        max_it=max_it, verbose=verbose, scalar_frobenius=scalar_frobenius, use_frobenius=use_frobenius)
            resid_test_components.append(one_fit["resid_test"])
            fitted_test[list_ind_test[j]] = one_fit["predicted_test"]
            theta_matrix_test[list_ind_test[j]] = one_fit["theta_matrix_test"]
            intercept_vec_test[list_ind_test[j]] = one_fit["intercept"]
        grid_resid.append(np.sum(resid_test_components))
        grid_theta_matrix.append(theta_matrix_test)
        grid_fitted_test.append(fitted_test)
        grid_intercept_vec.append(intercept_vec_test)
        # return resid grid, lambda grid, and "optimal" lambda
    optimal_resid = grid_resid[np.argmin(grid_resid)]
    if quick_attributes['p_categorical'] > 0:
        optimal_lam_continuous, optimal_lam_categorical = mat_lam[np.argmin(grid_resid)]
        return{"grid_theta_matrix":grid_theta_matrix, "grid_fitted_test":grid_fitted_test, "grid_intercept_vec":grid_intercept_vec,
        "optimal_lam_continuous":optimal_lam_continuous, "optimal_lam_categorical":optimal_lam_categorical, 
        "optimal_resid":optimal_resid, "grid_resid":grid_resid, "grid_lam_continuous":grid_lam_continuous, "grid_lam_categorical":grid_lam_categorical, "mat_lam":mat_lam}
    else:
        optimal_lam_continuous = grid_lam_continuous[np.argmin(grid_resid)]
        return{"grid_theta_matrix":grid_theta_matrix, "grid_fitted_test":grid_fitted_test, "grid_intercept_vec":grid_intercept_vec,
        "optimal_lam_continuous":optimal_lam_continuous, "optimal_resid":optimal_resid, "grid_resid":grid_resid, "grid_lam_continuous":grid_lam_continuous, "mat_lam":mat_lam}

def do_one(response_type, Y, treatment, 
                X_continuous, grid_lam_continuous_control=None, grid_lam_continuous_treat=None,
                X_categorical=None, grid_lam_control_categorical=None, grid_lam_treat_categorical=None,
                max_it=50000, verbose=False,
                use_frobenius=False, scalar_frobenius=0.001,
                ggd_thresh=1e-03, thresh=1e-03, 
                grid_lam_control=None, grid_lam_treat=None, lambda_min_ratio=0.01, n_lambda=10, k=10):
    # if X.ndim == 2 and np.shape(X)[1] == 1:
    #     X = X.reshape(X.size, )
    attrib_null = solving.setup(hypothesis="null", y=Y, treatment=treatment,
                               x_continuous=X_continuous, x_categorical=X_categorical) 
    if grid_lam_continuous_control is None:
        max_lam_continuous_control = max_lambda(Y=Y[attrib_null["ind_control"]], X=X_continuous[attrib_null["ind_control"]], form="continuous")
        grid_lam_continuous_control = np.exp(np.linspace(np.log(max_lam_continuous_control), np.log(lambda_min_ratio * max_lam_continuous_control), n_lambda))
    if grid_lam_continuous_treat is None:
        max_lam_continuous_treat = max_lambda(Y=Y[attrib_null["ind_treat"]], X=X_continuous[attrib_null["ind_treat"]], form="continuous")
        grid_lam_continuous_treat = np.exp(np.linspace(np.log(max_lam_continuous_treat), np.log(lambda_min_ratio * max_lam_continuous_treat), n_lambda))
    if attrib_null['p_categorical'] > 0:
        if X_categorical.ndim == 1:
            X_categorical = np.array(pd.get_dummies(X_categorical))
        else:
            X_categorical = np.concatenate([pd.get_dummies(np.array(X_categorical[:, j])) for j in range(np.shape(X_categorical)[1])], axis=1)
        if grid_lam_control_categorical is None:
            max_lam_control_categorical = max_lambda(Y=Y[attrib_null["ind_control"]], X=X_categorical[attrib_null["ind_control"]], form="categorical")
            grid_lam_control_categorical = np.exp(np.linspace(np.log(max_lam_control_categorical), np.log(lambda_min_ratio * max_lam_control_categorical), n_lambda))
        if grid_lam_treat_categorical is None:
            max_lam_treat_categorical = max_lambda(Y=Y[attrib_null["ind_treat"]], X=X_categorical[attrib_null["ind_treat"]], form="categorical")
            grid_lam_treat_categorical = np.exp(np.linspace(np.log(max_lam_treat_categorical), np.log(lambda_min_ratio * max_lam_treat_categorical), n_lambda))
    # run k-fold CV *under the alternative* to determine optimal lambdas in the control and treatment groups (along with those optimal fits)
    if attrib_null['p_categorical'] > 0:
        cv_result_control = loop_k_fold_cv(response_type=response_type, k=k, Y=Y[attrib_null["ind_control"]], 
                                                     X_continuous=X_continuous[attrib_null["ind_control"]], 
                                                     grid_lam_continuous=grid_lam_continuous_control, 
                                                     X_categorical=X_categorical[attrib_null["ind_control"]], 
                                                     grid_lam_categorical=grid_lam_control_categorical, 
                                                     ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose, use_frobenius=use_frobenius)
        cv_result_treat = loop_k_fold_cv(response_type=response_type, k=k, Y=Y[attrib_null["ind_treat"]], 
                                                     X_continuous=X_continuous[attrib_null["ind_treat"]], 
                                                     grid_lam_continuous=grid_lam_continuous_treat, 
                                                     X_categorical=X_categorical[attrib_null["ind_treat"]], 
                                                     grid_lam_categorical=grid_lam_treat_categorical, 
                                                     ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose, use_frobenius=use_frobenius)
    else:
        cv_result_control = loop_k_fold_cv(response_type=response_type, k=k, Y=Y[attrib_null["ind_control"]], 
                                                     X_continuous=X_continuous[attrib_null["ind_control"]], 
                                                     grid_lam_continuous=grid_lam_continuous_control, 
                                                     ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)
        cv_result_treat = loop_k_fold_cv(response_type=response_type, k=k, Y=Y[attrib_null["ind_treat"]], 
                                                     X_continuous=X_continuous[attrib_null["ind_treat"]], 
                                                     grid_lam_continuous=grid_lam_continuous_treat, 
                                                     ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)

    # fit under null using the optimal lambdas in the control and treatment groups
    if attrib_null['p_categorical'] > 0:
        fitted_null_optimal = fitted_values(hypothesis="null", response_type=response_type, Y=Y, treatment=treatment, 
                                                      X_continuous=X_continuous,
                                                      lam_continuous_control=cv_result_control["optimal_lam_continuous"], lam_continuous_treat=cv_result_treat["optimal_lam_continuous"],
                                                      X_categorical=X_categorical,
                                                      lam_categorical_control=cv_result_control["optimal_lam_categorical"], lam_categorical_treat=cv_result_treat["optimal_lam_categorical"],
                                                      use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius, ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)
    else:
        fitted_null_optimal = fitted_values(hypothesis="null", response_type=response_type,  Y=Y, treatment=treatment, 
                                                      X_continuous=X_continuous,
                                                      lam_continuous_control=cv_result_control["optimal_lam_continuous"], lam_continuous_treat=cv_result_treat["optimal_lam_continuous"],
                                                      use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius, ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)
    fitted_null_control_optimal = fitted_null_optimal["theta_result_control"][attrib_null["ind_control"]]
    fitted_null_treat_optimal = fitted_null_optimal["theta_result_treat"][attrib_null["ind_treat"]]

    if attrib_null['p_categorical'] > 0:
        result_alt_control_optimal = fitted_values(hypothesis="alternative", response_type=response_type, Y=Y[attrib_null["ind_control"]],
        X_continuous=X_continuous[attrib_null["ind_control"]], lam_alt_continuous=cv_result_control["optimal_lam_continuous"], 
        X_categorical=X_categorical[attrib_null["ind_control"]], lam_alt_categorical=cv_result_control["optimal_lam_categorical"], 
        use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius, ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)
        result_alt_treat_optimal = fitted_values(hypothesis="alternative", response_type=response_type, Y=Y[attrib_null["ind_treat"]],
        X_continuous=X_continuous[attrib_null["ind_treat"]], lam_alt_continuous=cv_result_treat["optimal_lam_continuous"], 
        X_categorical=X_categorical[attrib_null["ind_treat"]], lam_alt_categorical=cv_result_treat["optimal_lam_categorical"], 
        use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius, ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)
    else:
        result_alt_control_optimal = fitted_values(hypothesis="alternative", response_type=response_type, Y=Y[attrib_null["ind_control"]],
        X_continuous=X_continuous[attrib_null["ind_control"]], lam_alt_continuous=cv_result_control["optimal_lam_continuous"], 
        use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius, ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)
        result_alt_treat_optimal = fitted_values(hypothesis="alternative", response_type=response_type, Y=Y[attrib_null["ind_treat"]],
        X_continuous=X_continuous[attrib_null["ind_treat"]], lam_alt_continuous=cv_result_treat["optimal_lam_continuous"], 
        use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius, ggd_thresh=ggd_thresh, thresh=thresh, verbose=verbose)

    fitted_alt_control_optimal = result_alt_control_optimal["theta_result"].reshape(attrib_null["n_control"], 1)
    fitted_alt_treat_optimal = result_alt_treat_optimal["theta_result"].reshape(attrib_null["n_treat"], 1)
    # compute residuals under null and alternative
    resid_null_control = get_resid(response_type=response_type, actual=Y[attrib_null["ind_control"]], predicted=fitted_null_control_optimal)
    resid_null_treat = get_resid(response_type=response_type, actual=Y[attrib_null["ind_treat"]], predicted=fitted_null_treat_optimal)
    resid_alt_control = get_resid(response_type=response_type, actual=Y[attrib_null["ind_control"]], predicted=fitted_alt_control_optimal)
    resid_alt_treat = get_resid(response_type=response_type, actual=Y[attrib_null["ind_treat"]], predicted=fitted_alt_treat_optimal)
    observed_stat = ((resid_null_control + resid_null_treat) - (resid_alt_control + resid_alt_treat)) /  (resid_alt_control + resid_alt_treat)

    if attrib_null['p_categorical'] > 0:
        return{"fitted_null_control_optimal":fitted_null_control_optimal, "fitted_null_treat_optimal":fitted_null_treat_optimal,
                "fitted_alt_control_optimal":fitted_alt_control_optimal, "fitted_alt_treat_optimal":fitted_alt_treat_optimal, 
                "theta_matrix_null_control_all": fitted_null_optimal["theta_matrix_control"], "theta_matrix_null_treat_all": fitted_null_optimal["theta_matrix_treat"],
                "theta_matrix_alt_control":result_alt_control_optimal["theta_matrix"], "theta_matrix_alt_treat":result_alt_treat_optimal["theta_matrix"],
                "fitted_null_control_all":fitted_null_optimal["theta_result_control"], "fitted_null_treat_all":fitted_null_optimal["theta_result_treat"],
                "intercept_null_control":fitted_null_optimal["intercept_control"], "intercept_null_treat":fitted_null_optimal["intercept_treat"],
                "intercept_alt_control":result_alt_control_optimal["intercept"], "intercept_alt_treat":result_alt_treat_optimal["intercept"],
                "null_beta_vec_control":fitted_null_optimal["beta_vec_control"], "null_beta_vec_treat":fitted_null_optimal["beta_vec_treat"],
                "alt_beta_vec_control":result_alt_control_optimal["beta_vec"], "alt_beta_vec_treat":result_alt_treat_optimal["beta_vec"],
                "resid_null_control":resid_null_control,"resid_null_treat":resid_null_treat,
                "resid_alt_control":cv_result_control["optimal_resid"], "resid_alt_treat":cv_result_treat["optimal_resid"],
                "optimal_lam_continuous_control":cv_result_control["optimal_lam_continuous"], "optimal_lam_continuous_treat":cv_result_treat["optimal_lam_continuous"], 
                "grid_lam_continuous_control":cv_result_control["grid_lam_continuous"], "grid_lam_continuous_treat":cv_result_treat["grid_lam_continuous"], 
                "optimal_lam_control_categorical":cv_result_control["optimal_lam_categorical"], "optimal_lam_treat_categorical":cv_result_treat["optimal_lam_categorical"], 
                "grid_lam_control_categorical":cv_result_control["grid_lam_categorical"], "grid_lam_treat_categorical":cv_result_treat["grid_lam_categorical"], 
                "observed_stat":observed_stat}
    else:
        return{"fitted_null_control_optimal":fitted_null_control_optimal, "fitted_null_treat_optimal":fitted_null_treat_optimal,
        "fitted_alt_control_optimal":fitted_alt_control_optimal, "fitted_alt_treat_optimal":fitted_alt_treat_optimal, 
        "theta_matrix_null_control_all": fitted_null_optimal["theta_matrix_control"], "theta_matrix_null_treat_all": fitted_null_optimal["theta_matrix_treat"],
        "theta_matrix_alt_control":result_alt_control_optimal["theta_matrix"], "theta_matrix_alt_treat":result_alt_treat_optimal["theta_matrix"],
        "fitted_null_control_all":fitted_null_optimal["theta_result_control"], "fitted_null_treat_all":fitted_null_optimal["theta_result_treat"],
        "intercept_null_control":fitted_null_optimal["intercept_control"], "intercept_null_treat":fitted_null_optimal["intercept_treat"],
        "intercept_alt_control":result_alt_control_optimal["intercept"], "intercept_alt_treat":result_alt_treat_optimal["intercept"],
        "resid_null_control":resid_null_control,"resid_null_treat":resid_null_treat,
        "resid_alt_control":cv_result_control["optimal_resid"], "resid_alt_treat":cv_result_treat["optimal_resid"],
        "optimal_lam_continuous_control":cv_result_control["optimal_lam_continuous"], "optimal_lam_continuous_treat":cv_result_treat["optimal_lam_continuous"], 
        "grid_lam_continuous_control":cv_result_control["grid_lam_continuous"], "grid_lam_continuous_treat":cv_result_treat["grid_lam_continuous"], 
        "observed_stat":observed_stat}

def do_permutations(response_type, Y, treatment, 
                             ggd_thresh, thresh, 
                             X_continuous, grid_lam_continuous_control=None, grid_lam_continuous_treat=None,
                             X_categorical=None, grid_lam_control_categorical=None, grid_lam_treat_categorical=None,
                             use_frobenius=False, scalar_frobenius=0.001,
                             max_it=50000, verbose=False,
                             grid_lam_control=None, grid_lam_treat=None, lambda_min_ratio=0.01, n_lambda=10, k=10,
                             n_permutations=10):
    # make sure X, Y, and treatment have the shapes we are expecting
    if X_continuous.ndim == 2 and np.shape(X_continuous)[1] == 1:
        X_continuous = X_continuous.reshape(X_continuous.size, )
    if X_categorical is not None:
        if X_categorical.ndim == 2 and np.shape(X_categorical)[1] == 1:
            X_categorical = X_categorical.reshape(X_categorical.size, )
    if Y.ndim == 1:
        Y = Y.reshape(Y.size, 1)
    if treatment.ndim == 2:
        treatment = treatment.reshape(treatment.size, )
    observed_result = do_one(response_type=response_type, Y=Y, treatment=treatment, max_it=max_it, verbose=verbose, ggd_thresh=ggd_thresh, thresh=thresh,
                                        X_continuous=X_continuous, grid_lam_continuous_control=grid_lam_continuous_control, grid_lam_continuous_treat=grid_lam_continuous_treat,
                                        X_categorical=X_categorical, grid_lam_control_categorical=grid_lam_control_categorical, grid_lam_treat_categorical=grid_lam_treat_categorical,
                                        lambda_min_ratio=lambda_min_ratio, n_lambda=n_lambda, k=k,
                                        use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius)
    observed_stat = observed_result["observed_stat"]
    permuted_stats = []
    for j in range(n_permutations):
        shuffled_treatment = np.random.permutation(treatment) 
        permuted_result = do_one(response_type=response_type, Y=Y, treatment=shuffled_treatment, max_it=max_it, verbose=verbose, ggd_thresh=ggd_thresh, thresh=thresh,
                                        X_continuous=X_continuous, grid_lam_continuous_control=grid_lam_continuous_control, grid_lam_continuous_treat=grid_lam_continuous_treat,
                                        X_categorical=X_categorical, grid_lam_control_categorical=grid_lam_control_categorical, grid_lam_treat_categorical=grid_lam_treat_categorical,
                                        lambda_min_ratio=lambda_min_ratio, n_lambda=n_lambda, k=k,
                                        use_frobenius=use_frobenius, scalar_frobenius=scalar_frobenius)
        permuted_stats.append(permuted_result["observed_stat"])
    permuted_pval = np.mean(np.array(observed_stat) <= np.array(permuted_stats))
    if X_categorical is None:
        return{"fitted_null_control":observed_result["fitted_null_control_optimal"], "fitted_null_treat":observed_result["fitted_null_treat_optimal"], 
                "fitted_alt_control":observed_result["fitted_alt_control_optimal"],  "fitted_alt_treat":observed_result["fitted_alt_treat_optimal"],
                "fitted_null_control_all":observed_result["fitted_null_control_all"], "fitted_null_treat_all":observed_result["fitted_null_treat_all"],
                "observed_optimal_lam_continuous_control": observed_result["optimal_lam_continuous_control"],
                "observed_optimal_lam_continuous_treat": observed_result["optimal_lam_continuous_treat"],
                "p_value":permuted_pval, "permuted_stats":np.array(permuted_stats), "observed_stat":observed_stat}
    else:
        return{"fitted_null_control":observed_result["fitted_null_control_optimal"], "fitted_null_treat":observed_result["fitted_null_treat_optimal"], 
                 "fitted_alt_control":observed_result["fitted_alt_control_optimal"],  "fitted_alt_treat":observed_result["fitted_alt_treat_optimal"],
                 "fitted_null_control_all":observed_result["fitted_null_control_all"], "fitted_null_treat_all":observed_result["fitted_null_treat_all"],
                 "observed_optimal_lam_continuous_control": observed_result["optimal_lam_continuous_control"],
                 "observed_optimal_lam_continuous_treat": observed_result["optimal_lam_continuous_treat"],
                 "observed_optimal_lam_control_categorical": observed_result["optimal_lam_control_categorical"],
                 "observed_optimal_lam_treat_categorical": observed_result["optimal_lam_treat_categorical"],
                 "p_value":permuted_pval, "permuted_stats":np.array(permuted_stats), "observed_stat":observed_stat}


#############
###  Simulations ###
#############
## FUNCTION: Return response variable (Y) that adds appropriate Gaussian noise for specified functional form, SNR value, and "truth" type
## INPUT:
## OUTPUT:
def noisy_response(response_type, SNR, form, X, delta, truth, ind_control, ind_treat, treat_above=True):
    n = np.shape(X)[0]
    n_control = np.shape(ind_control)[0]
    n_treat = np.shape(ind_treat)[0]
    if SNR <= 0:
        raise ValueError("need to specify SNR > 0")
    if truth == "crossing":
        ## compute signal
        f_control = make_transform(scenario="shift", form=form)
        f_treat = make_transform(scenario="crossing", form=form)
        if X.ndim == 1:
            p = 1
            #fitted_control = f_control(X)
            #fitted_treat = f_treat(X)
        elif X.ndim == 2:
            p = np.shape(X)[1]
        fitted_control_mat = np.zeros((n, p))
        fitted_treat_mat = np.zeros((n, p))
        for col in range(p):
            fitted_control_mat[:, col] = f_control(X[:, col])
            fitted_treat_mat[:, col] = f_treat(X[:, col])
        fitted_control = np.sum(fitted_control_mat, axis=1)
        fitted_treat = np.sum(fitted_treat_mat, axis=1)
        if treat_above == True:
            diff = fitted_control - fitted_treat
        else:
            diff = fitted_treat - fitted_control
        ind_pos = np.flatnonzero(diff >= 0)
        mean_signal = (np.sum(diff[ind_pos] ** 2)) / n 
        ## compute sd for added noise term
        noise_sd = np.sqrt(mean_signal / SNR)
        ## generate and return response with added gaussian noise 
        transform_X = np.zeros(n)
        transform_X[ind_control] = fitted_control[ind_control]
        transform_X[ind_treat] = fitted_treat[ind_treat]
        Y = (transform_X + np.random.normal(loc=0, scale=noise_sd, size=n)).reshape((n, 1))
        transform_X_matrix = np.zeros((n, p))
        transform_X_matrix[ind_control] = fitted_control_mat[ind_control]
        transform_X_matrix[ind_treat] = fitted_treat_mat[ind_treat]
    elif truth in ["shift", "noncrossing"]:
        n_control = ind_control.size
        n_treat = ind_treat.size
        ## compute signal
        if truth == "shift":
            f = make_transform(scenario="shift", form=form)
            if X.ndim == 1:
                p = 1
                # fitted_control = f(X[ind_control])
                # fitted_treat = f(X[ind_treat]) - delta
            elif X.ndim == 2:
                p = np.shape(X)[1]
            fitted_control_mat = np.zeros((n_control, p))
            fitted_treat_mat = np.zeros((n_treat, p))
            for col in range(p):
                fitted_control_mat[:, col] = f(X[ind_control, col])
                fitted_treat_mat[:, col] = f(X[ind_treat, col]) - delta
            fitted_control = np.sum(fitted_control_mat, axis=1)
            fitted_treat = np.sum(fitted_treat_mat, axis=1)
        elif truth == "noncrossing":
            f_control = make_transform(scenario="shift", form=form)
            f_treat = make_transform(scenario="noncrossing", form=form)
            if X.ndim == 1:
                p = 1
                #fitted_control = f_control(X[ind_control])
                #fitted_treat = f_treat(X[ind_treat])
            elif X.ndim == 2:
                p = np.shape(X)[1]
            fitted_control_mat = np.zeros((n_control, p))
            fitted_treat_mat = np.zeros((n_treat, p))
            for col in range(p):
                fitted_control_mat[:, col] = f_control(X[ind_control, col])
                fitted_treat_mat[:, col] = f_treat(X[ind_treat, col])
            fitted_control = np.sum(fitted_control_mat, axis=1)
            fitted_treat = np.sum(fitted_treat_mat, axis=1)
            #fitted_control = np.sum(f_control(X[ind_control]), axis=1)
            #fitted_treat = np.sum(f_treat(X[ind_treat]), axis=1)
        mean_signal_control = np.sum((fitted_control - np.mean(fitted_control)) ** 2) / n_control
        mean_signal_treat = np.sum((fitted_treat - np.mean(fitted_treat)) ** 2) / n_treat
        ## compute sd for added noise term
        noise_sd_control = np.sqrt(mean_signal_control / SNR)
        noise_sd_treat = np.sqrt(mean_signal_treat / SNR)
        ## generate and return response with added gaussian noise
        transform_X = np.zeros(n)
        transform_X[ind_control] = fitted_control
        transform_X[ind_treat] = fitted_treat
        ## generate new response
        Y = np.zeros(n)
        Y[ind_control] = transform_X[ind_control] + np.random.normal(loc=0, scale=noise_sd_control, size=n_control)
        Y[ind_treat] = transform_X[ind_treat] + np.random.normal(loc=0, scale=noise_sd_treat, size=n_treat)
        Y = Y.reshape((n, 1))
        transform_X_matrix = np.zeros((n, p))
        transform_X_matrix[ind_control] = fitted_control_mat
        transform_X_matrix[ind_treat] = fitted_treat_mat
    transform_X = np.reshape(transform_X, (n, 1))
    if response_type == "binary":
        pr = solving.expit(transform_X)
        pr_matrix = solving.expit(transform_X_matrix)
        pr_noisy = solving.expit(Y)
        Y = (np.random.binomial(n=1, p=pr)).reshape(n, 1)
        if X.ndim == 1:
            return{"Y":Y, "P":pr}
        elif X.ndim == 2:
            return{"Y":Y, "P":pr, "P_matrix":pr_matrix}
    elif response_type == "continuous":
        if X.ndim == 1:
            return{"Y":Y, "P":transform_X}
        elif X.ndim == 2:
            return{"Y":Y,"P":transform_X, "P_matrix":transform_X_matrix}
    else:
        raise NameError("invalid response type specified")

def make_transform(scenario, form):
    if scenario == "shift":
        linear_int, linear_slope = -0.5, 1
        sin_A, sin_B, sin_C, sin_shift, sin_scale = 5.4375, 10, 0, -1, 0.2
        constant_break_1, constant_break_2, constant_break_3, constant_break_4 = 0.2, 0.4, 0.6, 0.8
        #val_1, val_2, val_3, val_4, val_5 = -0.5, 0.5, 0, 1, -1
        val_1, val_2, val_3, val_4, val_5 = -0.5, 0.5, 0, 1, 0.6
        linear_break_1, linear_break_2, linear_break_3, linear_break_4, linear_break_5, linear_break_6 = 0.1, 0.3, 0.4, 0.5, 0.625, 0.8
        #pw_linear_int, slope_1, slope_2, slope_3, slope_4, slope_5, slope_6, slope_7 = 0, 10, -10, 10, 0, 3, -3, 3
        pw_linear_int, slope_1, slope_2, slope_3, slope_4, slope_5, slope_6, slope_7 = -0.2, 10, -10, 10, 0, 3, -3, 10
    elif scenario == "crossing": 
        linear_int, linear_slope = 0, -0.5
        sin_A, sin_B, sin_C, sin_shift, sin_scale = 1, 10, np.pi + 0.7, 0, 1
        constant_break_1, constant_break_2, constant_break_3, constant_break_4 = 0.15, 0.35, 0.7, 0.9
        #val_1, val_2, val_3, val_4, val_5 = 0, -0.7, 0.2, 0.7, -0.5
        val_1, val_2, val_3, val_4, val_5 = 0, -0.7, 0.4, 0.9, -0.5
        linear_break_1, linear_break_2, linear_break_3, linear_break_4, linear_break_5, linear_break_6 = 0.1, 0.3, 0.4, 0.5, 0.625, 0.875
        #pw_linear_int, slope_1, slope_2, slope_3, slope_4, slope_5, slope_6, slope_7 = 0, -10, 7, -10, -3, 13, -5, -2
        pw_linear_int, slope_1, slope_2, slope_3, slope_4, slope_5, slope_6, slope_7 = -0.2, -10, 9, -10, -3, 13, -5, -10
    elif scenario == "noncrossing":
        linear_int, linear_slope = -0.6, -0.2
        #sin_A, sin_B, sin_C, sin_shift, sin_scale = 5.4375, 20, 1, -12, 0.2
        sin_A, sin_B, sin_C, sin_shift, sin_scale = 5.4375, 17, 0.7, -12, 0.2
        constant_break_1, constant_break_2, constant_break_3, constant_break_4 = 0.25, 0.35, 0.65, 0.78
        val_1, val_2, val_3, val_4, val_5 = -0.6, -1, -0.1, -0.5 , -1.3
        linear_break_1, linear_break_2, linear_break_3, linear_break_4, linear_break_5, linear_break_6 = 0.1, 0.3, 0.4, 0.5, 0.625, 0.875
        pw_linear_int, slope_1, slope_2, slope_3, slope_4, slope_5, slope_6, slope_7 = -1, -13, 3.5, -10, 5, 14, -12, 5
    else:
        raise NameError("invalid scenario specified")
    if form == "linear":
        def transform(X):
            return(linear_int + X* linear_slope)
    elif form == "sinusoid":
        def transform(X):
            return(sin_scale * (sin_A * np.sin(sin_B * X + sin_C) + sin_shift))
    elif form == "piecewise_constant":
        if (constant_break_1 >= constant_break_2 or constant_break_2 >= constant_break_3 or constant_break_3 >= constant_break_4 or constant_break_4 >= 1):
            raise ValueError("Invalid breaks specified")
        def transform(X):
            pw_constant_result = (X < constant_break_1) * val_1 + ((X >= constant_break_1) & (X < constant_break_2)) * val_2 + ((X >= constant_break_2) & (X < constant_break_3)) * val_3 + ((X >= constant_break_3) & (X < constant_break_4)) * val_4 + (X >= constant_break_4) * val_5
            return pw_constant_result
    elif form == "piecewise_linear":
        if (linear_break_1 >= linear_break_2 or linear_break_2 >= linear_break_3 or linear_break_3 >= linear_break_4 or linear_break_4 >= 1):
            raise ValueError("Invalid breaks specified")
        def transform(X):
            n_obs = X.size
            pw_linear_result = np.zeros(n_obs)
            pw_linear_result[X < linear_break_1] = pw_linear_int + slope_1 * X[X < linear_break_1]
            pw_linear_result[(X >= linear_break_1) & (X < linear_break_2)] =  slope_2 * (X[(X >= linear_break_1) & (X < linear_break_2)] - linear_break_1) + (pw_linear_int + slope_1 * linear_break_1) 
            pw_linear_result[(X >= linear_break_2) & (X < linear_break_3)] =  slope_3 * (X[(X >= linear_break_2) & (X < linear_break_3)] - linear_break_2) + ( (pw_linear_int + slope_1 * linear_break_1) + (slope_2 * (linear_break_2 - linear_break_1)) )
            pw_linear_result[(X >= linear_break_3) & (X < linear_break_4)] = slope_4 * (X[(X >= linear_break_3) & (X < linear_break_4)] - linear_break_3)  + ( (pw_linear_int +  slope_1 * linear_break_1) + (slope_2 * (linear_break_2 - linear_break_1)) + (slope_3 * (linear_break_3 - linear_break_2)))
            pw_linear_result[(X >= linear_break_4)] =  slope_5 * (X[(X > linear_break_4)] - linear_break_4)  + ( (pw_linear_int + slope_1 * linear_break_1) + (slope_2 * (linear_break_2 - linear_break_1)) +  (slope_3 * (linear_break_3 - linear_break_2)) + (slope_4 * (linear_break_4 - linear_break_3)))
            pw_linear_result[(X >= linear_break_5)] =  slope_6 * (X[(X > linear_break_5)] - linear_break_5)  + ( (pw_linear_int + slope_1 * linear_break_1) + (slope_2 * (linear_break_2 - linear_break_1)) +  (slope_3 * (linear_break_3 - linear_break_2)) + (slope_4 * (linear_break_4 - linear_break_3)) + (slope_5 * (linear_break_5 - linear_break_4)))
            pw_linear_result[(X >= linear_break_6)] =  slope_7 * (X[(X > linear_break_6)] - linear_break_6)  + ( (pw_linear_int + slope_1 * linear_break_1) + (slope_2 * (linear_break_2 - linear_break_1)) +  (slope_3 * (linear_break_3 - linear_break_2)) + (slope_4 * (linear_break_4 - linear_break_3)) + (slope_5 * (linear_break_5 - linear_break_4)) + (slope_6 * (linear_break_6 - linear_break_5)))
            return pw_linear_result
    else:
        raise NameError("Invalid functional form given")
    return transform

def simulate_mixed_features(truth, noise_sd, delta_shift=0.4, n=200):
    delta_2_0, delta_2_1 = delta_shift, 0 # same in both crossing and shift scenarios
    if truth == "crossing":
        delta_1_0, delta_1_1, delta_1_2 = delta_shift, 0, -delta_shift
        f_pw_linear_control = make_transform(scenario="shift", form="piecewise_linear")
        f_pw_constant_control = make_transform(scenario="shift", form="piecewise_constant")
        f_pw_linear_treat = make_transform(scenario="crossing", form="piecewise_linear") 
        #f_pw_constant_treat = make_transform(scenario="crossing", form="piecewise_constant") 
    elif truth == "shift":
        delta_1_0, delta_1_1, delta_1_2 = 1, 0, 1
        f_pw_linear_control = make_transform(scenario="shift", form="piecewise_linear")
        f_pw_constant_control = make_transform(scenario="shift", form="piecewise_constant")
        #f_pw_linear_treat = make_transform(scenario="crossing", form="piecewise_linear")
        #f_pw_constant_treat = make_transform(scenario="crossing", form="piecewise_constant")
    treatment = np.random.binomial(n=1, p=np.repeat(0.5, n))
    ind_control = np.flatnonzero(treatment==0)
    ind_treat = np.flatnonzero(treatment==1)
    # simulate features
    X_continuous = np.zeros((n, 2))
    for col in range(2):
        X_continuous[:, col] = np.array(np.random.uniform(0, 1, n))
    X_categorical_unformatted = np.zeros((n, 2))
    X_categorical_unformatted[:, 0] = np.array([np.random.choice([0, 1, 2], size=1, p=[1.0/3, 1.0/3, 1.0/3]) for i in range(n)]).reshape(n, )
    X_categorical_unformatted[:, 1] = np.array([np.random.choice([0, 1], size=1, p=[1.0/2, 1.0/2]) for i in range(n)]).reshape(n, )

    # simulate response
    Y_mat = np.zeros((n, 4))
    Y_mat[X_categorical_unformatted[:, 0] == 0, 0] = delta_1_0 * treatment[X_categorical_unformatted[:, 0] == 0]
    Y_mat[X_categorical_unformatted[:, 0] == 1, 0] = delta_1_1 * treatment[X_categorical_unformatted[:, 0] == 1]
    Y_mat[X_categorical_unformatted[:, 0] == 2, 0] = delta_1_2 * treatment[X_categorical_unformatted[:, 0] == 2]

    Y_mat[X_categorical_unformatted[:, 1] == 0, 1] = delta_2_0 * treatment[X_categorical_unformatted[:, 1] == 0]
    Y_mat[X_categorical_unformatted[:, 1] == 1, 1] = delta_2_1 * treatment[X_categorical_unformatted[:, 1] == 1]

    Y_mat[ind_control, 2] = f_pw_linear_control(X_continuous[ind_control, 0]) 
    Y_mat[ind_control, 3] = f_pw_constant_control(X_continuous[ind_control, 1]) 
    Y_mat[ind_treat, 3] = f_pw_constant_control(X_continuous[ind_treat, 1]) - delta_shift # same in both crossing and shift scenarios
    if truth == "crossing":
        Y_mat[ind_treat, 2] = f_pw_linear_treat(X_continuous[ind_treat, 0]) 
    elif truth == "shift": 
        Y_mat[ind_treat, 2] = f_pw_linear_control(X_continuous[ind_treat, 0]) - delta_shift

    Y_truth = np.sum(Y_mat, axis=1)
    response = Y_truth + np.random.normal(loc=0, scale=noise_sd, size=n)

    X_categorical = np.concatenate((np.array(pd.get_dummies(X_categorical_unformatted[:, 1])), np.array(pd.get_dummies(X_categorical_unformatted[:, 0]))), axis=1)
    return{"treatment":treatment, "ind_control":ind_control, "ind_treat":ind_treat,
              "X_continuous":X_continuous, "X_categorical_unformatted":X_categorical_unformatted, 'X_categorical':X_categorical,
              "Y_truth":np.array(Y_truth).reshape(n, 1), "Y_mat":Y_mat, "response":np.array(response).reshape(n, 1)}

def simulation_scenario(response_type, form, truth, seed, SNR, p, n=200, delta=0.5, X_dist="uniform", X_start=0.01, X_stop=1):
# generate X, P(Y), and Y
    np.random.seed(seed)
    SNR = float(SNR)
    X = np.zeros((n, p))
    if X_dist == "uniform":
        for col in range(p):
            X[:, col] = np.array(np.random.uniform(X_start, X_stop, n))
    elif X_dist == "evenly_spaced":
        for col in range(p):
            X[:, col] = np.linspace(start=X_start, stop=X_stop, num=n)
    elif X_dist == "AE_example":
        epsilon = (X_stop - X_start) / 100
        for col in range(p):
            X[range(int(np.floor(n/2))), col] = np.array(np.random.uniform(X_start - epsilon, X_start + epsilon,  int(np.floor(n/2))))
            X[range(int(np.floor(n/2)), n), col] = np.array(np.random.uniform(X_stop - epsilon, X_stop + epsilon,  int(np.ceil(n/2))))
    treatment = np.random.binomial(n=1, p=np.repeat(0.5, n))
    ind_treat = np.flatnonzero(treatment==1)
    ind_control = np.flatnonzero(treatment==0)
    response = noisy_response(response_type=response_type, SNR=SNR, X=X, delta=delta, truth=truth, form=form, ind_control=ind_control, ind_treat=ind_treat)
    Y = response["Y"]
    P = response["P"]
    return{"X":X, "P":P, "P_matrix":response["P_matrix"], "Y":Y, "treatment":treatment, "ind_control":ind_control, "ind_treat":ind_treat}

## FUNCTION: Test for qualitative interaction using the procedure based on linear regression
def comparison_OLS(Y, X, treatment, min_support_X, max_support_X, alpha=0.05):
    # form design matrix and get coefficient estimates from OLS
    n = np.shape(treatment)[0]
    treatment = treatment.reshape(n, 1)
    design = sm.add_constant(np.column_stack((X, treatment, X * treatment)))
    beta_hat = np.dot(np.linalg.inv(np.dot(np.transpose(design), design)), np.dot(np.transpose(design), Y))
    predicted_control = beta_hat[0] + beta_hat[1] * X 
    predicted_treat = beta_hat[0] + beta_hat[1] * X + beta_hat[2] * 1 + beta_hat[3] * X * 1
    # compute sandwich estimates
    resid = np.array(Y - (np.dot(design, beta_hat))).reshape(n, )
    bread = np.linalg.inv(np.dot(np.transpose(design), design))
    filling = np.dot(np.dot(np.transpose(design), np.diag(resid ** 2)), design)
    sandwich_cov = np.dot(np.dot(bread, filling), bread)
    sandwich_se = np.sqrt(np.diag(sandwich_cov))
    # compute critical value and form CI for beta_3
    crit_val = scipy.stats.t.ppf(q=1 - alpha/2, df=np.size(X) - 4)
    ci_beta3 = np.array([beta_hat[3] - crit_val * sandwich_se[3],
                                  beta_hat[3] + crit_val * sandwich_se[3]])
    ## reject H_0 if 0 is contained in ci_beta3
    if (np.min(ci_beta3) <= 0 and np.max(ci_beta3) >= 0):
        reject = False
        return{"predicted_control":predicted_control, "predicted_treat":predicted_treat, 
        "ratio_estimate":-beta_hat[2] / beta_hat[3], "reject":reject}
    else: 
        # compute se for (-beta_2 / beta_3), which we got using the delta method
        ratio_var = (sandwich_cov[1, 1] / beta_hat[3] ** 2) + (beta_hat[2] ** 2 * sandwich_cov[2, 2]) / (beta_hat[3] ** 4) - (2 * beta_hat[2] * sandwich_cov[1, 2]) / (beta_hat[3] ** 3)
        ratio_se = np.sqrt(ratio_var)
        ci_ratio = np.array([-(beta_hat[2] / beta_hat[3]) - crit_val * ratio_se, 
                                -(beta_hat[2] / beta_hat[3]) + crit_val * ratio_se])
        # reject H_0 IFF ci_ratio is contained in the support of X
        if (np.min(ci_ratio) >= min_support_X and np.max(ci_ratio) <= max_support_X):
            reject = True
        else:
            reject = False
        return{"predicted_control":predicted_control, "predicted_treat":predicted_treat, 
                 "ratio_estimate":-beta_hat[2] / beta_hat[3], "ci_ratio": ci_ratio, "reject":reject}

def sum_squared_errors(predicted, observed):
    s = predicted.size
    resid = np.linalg.norm(observed - predicted, 2) ** 2
    return resid

def bin_means(Y, X, treatment, n_bins):
    n = np.shape(X)[0]
    idx_ordered = np.argsort(X, 0)[:, 0]
    X = X[idx_ordered]
    Y = Y[idx_ordered]
    treatment = treatment[idx_ordered]
    # divide into bins (assumes X is ordered!)
    if np.sum(np.array(sorted(X)) == X) != n:
        raise ValueError('x must be ordered')
    ind_bins = np.array_split(range(n), n_bins)
    # compute RSS under H_A and under H_0 and RSS from fitting the mean jointly for the two groups within each bin (H_0)
    RSS_alt = 0 
    RSS_null_control_above = 0
    RSS_null_treat_above = 0
    for i in range(n_bins):
        ind = ind_bins[i]
        bin_treatment = treatment[ind]
        bin_Y = Y[ind]
        mean_control = np.mean(bin_Y[bin_treatment==0])
        mean_treat = np.mean(bin_Y[bin_treatment==1])
        # H_A: RSS from fitting the mean separately for each group within each bin 
        one_RSS_alt = sum_squared_errors(predicted=mean_control, observed=bin_Y[bin_treatment==0]) + sum_squared_errors(predicted=mean_treat, observed=bin_Y[bin_treatment==1])
        RSS_alt = RSS_alt + one_RSS_alt
        # H_0: If constraint satisfied, then use same estimate as under H_A, if constraint not satisfied then estimate mean jointly for the two groups within each bin (H_0)
        one_RSS_joint = sum_squared_errors(predicted=np.mean(bin_Y), observed = bin_Y)
        if mean_control > mean_treat:
            RSS_null_control_above = RSS_null_control_above + one_RSS_alt
            RSS_null_treat_above = RSS_null_treat_above + one_RSS_joint
        elif mean_treat > mean_control: 
            RSS_null_control_above = RSS_null_control_above + one_RSS_joint
            RSS_null_treat_above = RSS_null_treat_above + one_RSS_alt
        # if only one group has observations in a region, it makes sense to use the same "joint" mean estimate for both constraints, I think
        elif np.isnan(mean_control) == True or np.isnan(mean_treat) == True:
            RSS_null_control_above = RSS_null_control_above + one_RSS_joint
            RSS_null_treat_above = RSS_null_treat_above + one_RSS_joint
        else: 
            raise ValueError("mean_control is equal to mean_treat, or at least one group is missing observations")
        if RSS_null_control_above < RSS_null_treat_above:
            RSS_null = RSS_null_control_above
        else:
            RSS_null = RSS_null_treat_above
        if RSS_null < RSS_alt:
            raise ValueError("shouldn't have RSS_null < RSS_alt")
    stat = (RSS_null - RSS_alt) / RSS_alt
    return np.array([stat])

def comparison_prespecified(n_shuffles, Y, X, treatment, n_bins):
    # get observed test statistic
    observed_stat = bin_means(Y=Y, X=X, treatment=treatment, n_bins=n_bins)
    # get permuted test statistic
    seq_permuted_stat = []
    for i in range(n_shuffles):
        shuffled_labels = np.random.permutation(treatment) 
        seq_permuted_stat.append(bin_means(Y=Y, X=X, treatment=shuffled_labels, n_bins=n_bins))
    # compute p-value
    p_value = np.mean(seq_permuted_stat >= observed_stat)
    return{"seq_permuted_stat":seq_permuted_stat, "observed_stat":observed_stat, "p_value":p_value}


## FUNCTION: return SSR/negative likelihood of fitted values
def get_resid(response_type, actual, predicted):
    if np.shape(actual) != np.shape(predicted):
        raise ValueError("The dimensions of the predicted values do not match the dimensions of the actual values!")
    if response_type == "binary":
        if np.any(np.array(predicted) < 0):
            raise ValueError("IN GET_RESID: a predicted probability is less than 0! i must be forgetting an expit() somewhere...")
        result = -(np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))
        if math.isnan(float(result)):
            raise ValueError("taking logs of negative numbers!!")
    elif response_type == "continuous":
        #result = np.linalg.norm(actual - predicted, 2) ** 2
        result =  np.sum((actual - predicted) ** 2 )
    return result

# ##################
# ###  Troubleshooting CV  ####
# ##################
## For each lambda in the grid, want to compare the fitted value to the truth
def cv_oracle_error(response_type, fitted_values_matrix, truth, grid_intercept_vec=None):
    n_fits = np.shape(fitted_values_matrix)[0]
    if grid_intercept_vec is None:
        #grid_oracle_error = [np.sum(get_resid(response_type=response_type, actual=truth, predicted=fitted_values_matrix[j])) for j in range(n_fits)]
        grid_oracle_error = [np.sum( (fitted_values_matrix[j] - truth) ** 2) for j in range(n_fits)]
    else:
        #grid_oracle_error = [np.sum(get_resid(response_type=response_type, actual=truth, predicted=(fitted_values_matrix[j] - grid_intercept_vec[j][:, np.newaxis]))) for j in range(n_fits)]
        grid_oracle_error = [np.sum( ( (fitted_values_matrix[j] - grid_intercept_vec[j][:, np.newaxis]) - truth) ** 2) for j in range(n_fits)]
    return grid_oracle_error
        
def plot_fitted_values(hypothesis, X, Y, 
    fitted_control, fitted_treat, plot_name,
    ind_control, ind_treat,
    Y_truth=None, X_truth=None, 
    plots_dir="Plots/", plot_title="", feature_name=None, response_name=None, labels_fontsize=16,
    ind_control_truth=None, ind_treat_truth=None, include_truth=False, include_fitted=True, include_legend=False, 
    save=True):
    if X.ndim == 1:
        p = 1
    elif X.ndim == 2:
        p = np.shape(X)[1]
    if p > 1:
        raise ValueError("the design matrix should include only one feature")
    n = np.shape(X)[0]
    X = np.reshape(X, (n, 1))
    Y = np.reshape(Y, (n, 1))
    if include_truth == True:
        n_truth = X_truth.size
        X_truth = np.reshape(X_truth, (n_truth, 1))
    if hypothesis == "alternative":
        X_control = X[ind_control]
        X_treat = X[ind_treat]
    elif hypothesis == "null":
        X_control = X
        X_treat = X
    ord_control = np.argsort(X_control, axis=0)[:, 0]
    ord_treat = np.argsort(X_treat, axis=0)[:, 0]
    plt.clf()
    if include_truth == True:
        plt.scatter(X_truth[ind_control_truth], Y_truth[ind_control_truth], color="DarkOrange", s=3)
        plt.scatter(X_truth[ind_treat_truth], Y_truth[ind_treat_truth], color="blue", s=3)
    plt.scatter(X[ind_control, 0], Y[ind_control, 0], color="DarkOrange", s=3)
    plt.scatter(X[ind_treat, 0], Y[ind_treat, 0], color="blue", s=3)
    if include_fitted == True:
        plt.plot(X_control[ord_control], fitted_control[ord_control], label="treatment = 0", color="DarkOrange", linewidth=4)
        plt.plot(X_treat[ord_treat], fitted_treat[ord_treat], label="treatment = 1", color="blue", linewidth=4)
    plt.title(plot_title)
    if response_name is not None:
        plt.ylabel(response_name, fontsize=labels_fontsize)
    if feature_name is not None:
        plt.xlabel(feature_name, fontsize=labels_fontsize)
    if include_legend == True:
        plt.legend()
    if save == True:
        plt.savefig(plots_dir + plot_name + ".png")
    else: 
        plt.savefig(plots_dir + plot_name +  ".png")

def plot_fitted_surface(response_type, hypothesis, X, 
#    fitted_control, fitted_treat, plot_name,
    fitted_control_matrix, fitted_treat_matrix, 
    intercept_control, intercept_treat, 
    ind_control, ind_treat,
    plot_name, plots_dir="Plots/", plot_title="", labels_fontsize=16,
    zlim3d=(0, 1), alpha_control=0.6, alpha_treat=0.6, view_init=None,
    feature_names=None, response_name=None, groups_to_show="both",
    save=True):
    if np.shape(X)[1] != 2:
        raise ValueError("X needs to have two features as its columns")
    n = np.shape(X)[0]
    # format data and plot surfaces for different hypothesis types
    plt.clf()
    x_unif = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), num=n)
    y_unif = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), num=n)
    x_grid, y_grid = np.meshgrid(x_unif, y_unif)
    if hypothesis == "null":
        ## fitted values
        Z_control_X = np.interp(x_unif, X[np.argsort(X[:, 0]), 0], fitted_control_matrix[np.argsort(X[:, 0]), 0])
        Z_control_Y = np.interp(x_unif, X[np.argsort(X[:, 1]), 1], fitted_control_matrix[np.argsort(X[:, 1]), 1])
        Z_treat_X = np.interp(x_unif, X[np.argsort(X[:, 0]), 0], fitted_treat_matrix[np.argsort(X[:, 0]), 0])
        Z_treat_Y = np.interp(x_unif, X[np.argsort(X[:, 1]), 1], fitted_treat_matrix[np.argsort(X[:, 1]), 1])
    elif hypothesis == "alternative":
        Z_control_X = np.interp(x_unif, X[ind_control, 0][np.argsort(X[ind_control, 0])], fitted_control_matrix[np.argsort(X[ind_control, 0]), 0])
        Z_control_Y = np.interp(y_unif, X[ind_control, 1][np.argsort(X[ind_control, 1])], fitted_control_matrix[np.argsort(X[ind_control, 1]), 1])
        Z_treat_X = np.interp(x_unif, X[ind_treat, 0][np.argsort(X[ind_treat, 0])], fitted_treat_matrix[np.argsort(X[ind_treat, 0]), 0])
        Z_treat_Y = np.interp(y_unif, X[ind_treat, 1][np.argsort(X[ind_treat, 1])], fitted_treat_matrix[np.argsort(X[ind_treat, 1]), 1])
    else:
        raise NameError("need to specify either the null or alternative hypothesis")
    # finish storing interpolated values
    Z_control_fitted = np.zeros((n, n))
    Z_treat_fitted = np.zeros((n, n))
    if response_type == "continuous":
        for i in range(n):
            for j in range(n):
                Z_control_fitted[i, j] = Z_control_X[i] + Z_control_Y[j] + intercept_control
                Z_treat_fitted[i, j] = Z_treat_X[i] + Z_treat_Y[j] + intercept_treat
    elif response_type == "binary":
        for i in range(n):
            for j in range(n):
                Z_control_fitted[i, j] = solving.expit(Z_control_X[i] + Z_control_Y[j] + intercept_control)
                Z_treat_fitted[i, j] = solving.expit(Z_treat_X[i] + Z_treat_Y[j] + intercept_treat)
    # make plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if groups_to_show == "both":
        ax.plot_surface(X=x_grid, Y=y_grid, Z=Z_control_fitted, color="orange", alpha=alpha_control)
        ax.plot_surface(X=x_grid, Y=y_grid, Z=Z_treat_fitted, color="blue", alpha=alpha_treat)
    elif groups_to_show == "control_only":
        ax.plot_surface(X=x_grid, Y=y_grid, Z=Z_control_fitted, color="orange", alpha=alpha_control)
    elif groups_to_show == "treat_only":
        ax.plot_surface(X=x_grid, Y=y_grid, Z=Z_treat_fitted, color="blue", alpha=alpha_treat)
    else:
        raise NameError("need to show either both groups, control only, or treat only")
    if feature_names is None:
        ax.set_xlabel("Feature 1", fontsize=labels_fontsize)
        ax.set_ylabel("Feature 2", fontsize=labels_fontsize)
    else:
        ax.set_xlabel(feature_names[0], fontsize=labels_fontsize)
        ax.set_ylabel(feature_names[1], fontsize=labels_fontsize)
    if response_name is None:
        ax.set_zlabel('Response', fontsize=labels_fontsize)
    else:
        ax.set_zlabel(response_name, fontsize=labels_fontsize)        
    plt.title(plot_title)
    ax.set_zlim3d(zlim3d)
    ax.view_init(30, view_init)
    if save == True:
        plt.savefig(plots_dir + plot_name)
    else:
        plt.show()

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def plot_mixed_simulation_data(mixed_simulation_data, scenario, save_dir):
    plt.clf()
    # 3-level categorical variable
    plt.scatter(rand_jitter(mixed_simulation_data['X_categorical_unformatted'][mixed_simulation_data['ind_control'], 0]), 
                  rand_jitter(mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_control'], 0]), color="blue")
    plt.scatter(rand_jitter(mixed_simulation_data['X_categorical_unformatted'][mixed_simulation_data['ind_treat'], 0]), 
                rand_jitter(mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_treat'], 0]), color="orange")
    plt.xlabel("Candidate Biomarker", fontsize=16)
    plt.ylabel("Response", fontsize=16)
    plt.savefig(save_dir + scenario + "_mixed_simulation_categorical_1.png")
    # 2-level categorical variable
    plt.clf()
    plt.scatter(rand_jitter(mixed_simulation_data['X_categorical_unformatted'][mixed_simulation_data['ind_treat'], 1]), rand_jitter(mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_treat'], 1]), color="orange")
    plt.scatter(rand_jitter(mixed_simulation_data['X_categorical_unformatted'][mixed_simulation_data['ind_control'], 1]), rand_jitter(mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_control'], 1]), color="blue")
    plt.xlabel("Candidate Biomarker", fontsize=16)
    plt.ylabel("Response", fontsize=16)
    plt.savefig(save_dir + scenario + "_mixed_simulation_categorical_2.png")
    # continuous variable 1
    plt.clf()
    plt.scatter(mixed_simulation_data['X_continuous'][mixed_simulation_data['ind_control'], 0], mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_control'], 2], color="blue")
    plt.scatter(mixed_simulation_data['X_continuous'][mixed_simulation_data['ind_treat'], 0], mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_treat'], 2], color="orange")
    plt.xlabel("Candidate Biomarker", fontsize=16)
    plt.ylabel("Response", fontsize=16)
    plt.savefig(save_dir + scenario + "_mixed_simulation_continuous_1.png")
    # continuous variable 2
    plt.clf()
    plt.scatter(mixed_simulation_data['X_continuous'][mixed_simulation_data['ind_control'], 1], mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_control'], 3], color="blue")
    plt.scatter(mixed_simulation_data['X_continuous'][mixed_simulation_data['ind_treat'], 1], mixed_simulation_data['Y_mat'][mixed_simulation_data['ind_treat'], 3], color="orange")
    plt.xlabel("Candidate Biomarker", fontsize=16)
    plt.ylabel("Response", fontsize=16)
    plt.savefig(save_dir + scenario + "_mixed_simulation_continuous_2.png")
