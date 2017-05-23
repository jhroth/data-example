from Functions import qualitativeinteractions as qi
import numpy as np

# load data and set general parameters
gene_expressions_HER2_positive_enriched = np.loadtxt(fname='Data/gse_50948_9_probes_HER2_positive_enriched.csv', delimiter=',', skiprows=1)
n = np.shape(gene_expressions_HER2_positive_enriched)[0]
p = np.shape(gene_expressions_HER2_positive_enriched)[1]
pcr = np.loadtxt(fname='Data/pcr_HER2_positive.csv', delimiter=',', skiprows=1).reshape(n, 1)
tumor_grade = np.loadtxt(fname='Data/tumor_grade_HER2_positive.csv', delimiter=',', skiprows=1)
treatment = np.loadtxt(fname='Data/treatment_HER2_positive.csv', delimiter=',', skiprows=1)
ind_control = np.flatnonzero(treatment == 0)
ind_treat = np.flatnonzero(treatment == 1)
names_probes =  np.loadtxt(fname='Data/gse_50948_9_probes_HER2_positive_enriched.csv', delimiter=',', skiprows=0, dtype='str')[0, ]
#names_probes = np.loadtxt('Data/probes_HER2_enriched.csv', dtype='str', delimiter=',', skiprows=1)[:, 0]
my_thresh = 1e-03
my_ggd_thresh = 1e-03

# load data for 1-feature example
observed_stats_vec_1d = np.loadtxt("Results/100_permutations/one_probe_observed_stats_vec.csv", delimiter=',', skiprows=1)
permuted_stats_mat_1d = np.loadtxt("Results/100_permutations/one_probe_permuted_stats_mat.csv", delimiter=',', skiprows=1)
optimal_lam_control_1d = np.loadtxt("Results/100_permutations/one_probes_optimal_lambda_control.csv", delimiter=',', skiprows=1)
optimal_lam_treat_1d = np.loadtxt("Results/100_permutations/one_probes_optimal_lambda_treat.csv", delimiter=',', skiprows=1)
FDR_1d = [np.mean(observed_stats_vec_1d[j] <= permuted_stats_mat_1d) for j in range(p)]
idx_best_1d = np.argmax(observed_stats_vec_1d)
best_probe_1d = gene_expressions_HER2_positive_enriched[:, idx_best_1d].reshape(n, 1)
name_best_probe_1d = np.reshape(names_probes[idx_best_1d], (1, ))

# load data for 2-feature example
combinations_2d = np.loadtxt("Data/combinations_2_probes.csv", delimiter=',', skiprows=1).astype(int)
observed_stats_vec_2d = np.loadtxt("Results/100_permutations/two_probes_observed_stats_vec.csv", delimiter=',', skiprows=1)
permuted_stats_mat_2d = np.loadtxt("Results/100_permutations/two_probes_permuted_stats_mat.csv", delimiter=',', skiprows=1)
optimal_lam_control_2d = np.loadtxt("Results/100_permutations/two_probes_optimal_lambda_control.csv", delimiter=',', skiprows=1)
optimal_lam_treat_2d = np.loadtxt("Results/100_permutations/two_probes_optimal_lambda_treat.csv", delimiter=',', skiprows=1)
idx_best_2d = np.argmax(observed_stats_vec_2d)
best_combination_2d = (combinations_2d[idx_best_2d])
best_probes_2d = gene_expressions_HER2_positive_enriched[:, best_combination_2d].reshape(n, 2)
FDR_2d = [np.mean(observed_stats_vec_2d[j] <= permuted_stats_mat_2d) for j in range(np.shape(combinations_2d)[0])]
name_best_probes_2d = np.reshape(names_probes[best_combination_2d ], (2, ))

# load data for categorical variable
optimal_lam_control_2d_categorical = np.loadtxt("Results/two_probes_optimal_lambda_control_categorical.csv", delimiter=',', skiprows=1)
optimal_lam_treat_2d_categorical = np.loadtxt("Results/two_probes_optimal_lambda_control_categorical.csv", delimiter=',', skiprows=1)


# Fit example with "best" probes and lambda values
## 1 continuous feature
result_one_probe_null = qi.fitted_values(hypothesis="null", response_type="binary", X_continuous=best_probe_1d, Y=pcr, treatment=treatment, lam_continuous_control=optimal_lam_control_1d, lam_continuous_treat=optimal_lam_treat_1d, ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)

result_one_probe_alt_control = qi.fitted_values(hypothesis="alternative", response_type="binary", X_continuous=best_probe_1d[ind_control], Y=pcr[ind_control], lam_alt_continuous=optimal_lam_control_1d, ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)
result_one_probe_alt_treat = qi.fitted_values(hypothesis="alternative", response_type="binary", X_continuous=best_probe_1d[ind_treat], Y=pcr[ind_treat], lam_alt_continuous=optimal_lam_treat_1d, ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)


## 2 continuous features
result_two_probes_null = qi.fitted_values(hypothesis="null", response_type="binary", X_continuous=best_probes_2d, Y=pcr, treatment=treatment, lam_continuous_control=optimal_lam_control_2d, lam_continuous_treat=optimal_lam_treat_2d, ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)

result_two_probes_alt_control = qi.fitted_values(hypothesis="alternative", response_type="binary", X_continuous=best_probes_2d[ind_control], Y=pcr[ind_control], lam_alt_continuous=optimal_lam_control_2d, ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)
result_two_probes_alt_treat = qi.fitted_values(hypothesis="alternative", response_type="binary", X_continuous=best_probes_2d[ind_treat], Y=pcr[ind_treat], lam_alt_continuous=optimal_lam_treat_2d, ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)


# Plot results
## 1 continuous feature
plot_name_null_1d = "fitted_values_null_one_probe_" + str(names_probes[idx_best_1d])
fitted_plot_null_1d = qi.plot_fitted_values(hypothesis="null", X=best_probe_1d, Y=pcr,  
                                                            fitted_control=result_one_probe_null["theta_result_control"], fitted_treat=result_one_probe_null["theta_result_treat"],
                                                            ind_control=ind_control, ind_treat=ind_treat, labels_fontsize=20,
                                                            feature_name=name_best_probe_1d[0], response_name="Probability of 3-year pCR", plots_dir="Plots/", plot_name=plot_name_null_1d)
plot_name_alt_1d = "fitted_values_alt_one_probe_" + str(names_probes[idx_best_1d])
fitted_plot_alt_1d = qi.plot_fitted_values(hypothesis="alternative", X=best_probe_1d, Y=pcr,  
                                                           fitted_control=result_one_probe_alt_control["theta_result"], fitted_treat=result_one_probe_alt_treat["theta_result"],
                                                            ind_control=ind_control, ind_treat=ind_treat, labels_fontsize=20,
                                                            feature_name=name_best_probe_1d[0], response_name="Probability of 3-year pCR", plots_dir="Plots/", plot_name=plot_name_alt_1d)

## 2 continuous features
plot_name_null_2d = "fitted_surface_null_two_probes_" + str(names_probes[best_combination_2d[0]]) + "_and_" + str(names_probes[best_combination_2d[1]])
surface_plot_null = qi.plot_fitted_surface(response_type="binary", hypothesis="null", X=best_probes_2d, 
                                    fitted_control_matrix=result_two_probes_null["theta_matrix_control"],
                                    fitted_treat_matrix=result_two_probes_null["theta_matrix_treat"],
                                    intercept_control=result_two_probes_null["intercept_control"],
                                    intercept_treat=result_two_probes_null["intercept_treat"],
                                    ind_control=ind_control, ind_treat=ind_treat,
                                    plot_name=plot_name_null_2d, plots_dir="Plots/", zlim3d=(0, 1), 
                                    alpha_control=0.70, alpha_treat=0.65, view_init=10, labels_fontsize=20,
                                    feature_names=names_probes[best_combination_2d],
                                    response_name="Probability of 3-year pCR")
### Under alternative
plot_name_alt_2d = "fitted_surface_alt_two_probes_" + str(names_probes[best_combination_2d[0]]) + "_and_" + str(names_probes[best_combination_2d[1]])
surface_plot_alt = qi.plot_fitted_surface(response_type="binary", hypothesis="alternative", X=best_probes_2d, 
                                fitted_control_matrix=result_two_probes_alt_control["theta_matrix"],
                                fitted_treat_matrix=result_two_probes_alt_treat["theta_matrix"],
                                intercept_control=result_two_probes_alt_control["intercept"],
                                intercept_treat=result_two_probes_alt_treat["intercept"],
                                ind_control=ind_control, ind_treat=ind_treat,
                                plot_name=plot_name_alt_2d, plots_dir="Plots/",
                                    #plot_title="Estimated FDR=" + str(np.round(FDR_2d[idx_best_2d], 2)), 
                                zlim3d=(0, 1), 
                                alpha_control=0.70, alpha_treat=0.65, view_init=10, labels_fontsize=20,
                                feature_names=names_probes[best_combination_2d],
                                response_name="Probability of 3-year pCR")



# 2 continuous features and one categorical feature
## under null
result_mixed_two_probes_null = qi.fitted_values(hypothesis="null", response_type="binary", X_continuous=best_probes_2d, X_categorical=tumor_grade, Y=pcr, treatment=treatment, lam_continuous_control=optimal_lam_control_2d, lam_continuous_treat=optimal_lam_treat_2d, 
lam_categorical_control=optimal_lam_control_2d_categorical, lam_categorical_treat=optimal_lam_treat_2d_categorical,
ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)
## under alternative
result_mixed_two_probes_alt_control = qi.fitted_values(hypothesis="alternative", response_type="binary", 
X_continuous=best_probes_2d[ind_control], X_categorical=tumor_grade[ind_control],Y=pcr[ind_control], 
lam_alt_continuous=optimal_lam_control_2d, lam_alt_categorical=optimal_lam_control_2d_categorical, 
ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)
result_mixed_two_probes_alt_treat = qi.fitted_values(hypothesis="alternative", response_type="binary", 
X_continuous=best_probes_2d[ind_treat], X_categorical=tumor_grade[ind_treat],Y=pcr[ind_treat], 
lam_alt_continuous=optimal_lam_treat_2d, lam_alt_categorical=optimal_lam_treat_2d_categorical, 
ggd_thresh=my_ggd_thresh, thresh=my_thresh, verbose=True)

print "Under the null, beta coefficients for tumor grade indicator in control group: " + str(result_mixed_two_probes_null["beta_vec_control"])
print "Under the null, beta coefficients for tumor grade indicator in treatment group: " + str(result_mixed_two_probes_null["beta_vec_treat"])
print "Under the alternative, beta coefficients for tumor grade indicator in control group: " + str(result_mixed_two_probes_alt_control["beta_vec"])
print "Under the alternative, beta coefficients for tumor grade indicator in treatment group: " + str(result_mixed_two_probes_alt_treat["beta_vec"])
