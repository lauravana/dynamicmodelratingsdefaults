functions {
   real induced_dirichlet_lpdf(vector c, vector alpha, real phi) {
    int K = num_elements(c) + 1;
    vector[K - 1] sigma = inv_logit(phi - c);
    vector[K] p;
    matrix[K, K] J = rep_matrix(0, K, K);
    
    // Induced ordinal probabilities
    p[1] = 1 - sigma[1];
    for (k in 2:(K - 1))
      p[k] = sigma[k - 1] - sigma[k];
    p[K] = sigma[K - 1];
    
    // Baseline column of Jacobian
    for (k in 1:K) J[k, 1] = 1;
    
    // Diagonal entries of Jacobian
    for (k in 2:K) {
      real rho = sigma[k - 1] * (1 - sigma[k - 1]);
      J[k, k] = - rho;
      J[k - 1, k] = rho;
    }
    
    return   dirichlet_lpdf(p | alpha)
           + log_determinant(J);
  }

}

data {
  int<lower=0> N;         // total number of subjects
  int<lower=0> N1;        // total number of subjects in train sample
  int<lower=0> N2;        // total number of subjects in test sample
  int<lower=0> NR1_train; // number of subjects rated by R1 in training sample
  int<lower=0> NR2_train; // number of subjects rated by R2 in training sample
  int<lower=0> NR3_train; // number of subjects rated by R3 in training sample
  int<lower=0> NR1_test;  // number of subjects rated by R1 in test sample
  int<lower=0> NR2_test;  // number of subjects rated by R2 in test sample
  int<lower=0> NR3_test;  // number of subjects rated by R3 in test sample
  int<lower=1> P;   // number of covariates
  int<lower=1> J;   // number of raters
  int<lower=1> I;   // number of firms
  int<lower=1> T;   // number of years in training sample
  int firm_id[N];   // firm index 
  int year_id[N];   // year index 
  int id_test[N2];  // index of test sample N1 + seq_len(N2)
  int begin_eps[I]; // id of firm beginning row
  int end_eps[I];   // id of firm last row
  int firm_id_train[N1];    // firm index training data
  int year_id_train[N1];    // time index training data
  int firm_id_test[N2];     // firm index test data
  int year_id_test[N2];     // time index test data
  int id_obs_R1_train[NR1_train]; // index of the observed ratings from R1
  int id_obs_R2_train[NR2_train]; // index of the observed ratings from R2
  int id_obs_R3_train[NR3_train]; // index of the observed ratings from R3
  int id_obs_R1_test[NR1_test]; // index of the observed ratings from R1
  int id_obs_R2_test[NR2_test]; // index of the observed ratings from R2
  int id_obs_R3_test[NR3_test]; // index of the observed ratings from R3
  int<lower=2> K1; // number of classes 1st outcome
  int<lower=2> K2; // number of classes 2nd outcome
  int<lower=2> K3; // number of classes 3rd outcome
  int<lower=1, upper=K1> y1_train[NR1_train]; // ordinal outcome rater 1 training
  int<lower=1, upper=K2> y2_train[NR2_train]; // ordinal outcome rater 2 training
  int<lower=1, upper=K3> y3_train[NR3_train]; // ordinal outcome rater 3 training
  int<lower=1, upper=K1> y1_test[NR1_test];   // ordinal outcome rater 1 test 
  int<lower=1, upper=K2> y2_test[NR2_test];   // ordinal outcome rater 2 test 
  int<lower=1, upper=K3> y3_test[NR3_test];   // ordinal outcome rater 3 test 
  int<lower=0, upper=1> D_train[N1]; // default indicator train
  int<lower=0, upper=1> D_test[N2];  // default indicator test
  matrix[N1, P] x_train;             // matrix of covariates training set
  matrix[N2, P] x_test;              // matrix of covariates test set
}
parameters {
  // ** Regression coefficients ** 
  vector[P] beta;
  vector[P] gamma1;
  vector[P] gamma2;
  vector[P] gamma3;
  real beta0;
  // ** Cutpoints ** 
  ordered[K1 - 1] theta1; 
  ordered[K2 - 1] theta2; 
  ordered[K3 - 1] theta3; 


  // ** Standard deviation parameters **
  real<lower=0> psi; 
  // ** Raw random effects (iid) **
  vector[N] e_raw;
}

transformed parameters {
  // Train set quantities
  vector[N1] S;
  vector[N1] u_train;
  vector[N1] SR1;
  vector[N1] SR2;
  vector[N1] SR3;
  // Test set quantities
  vector[N2] S_test;
  vector[N2] u_test;
  vector[N2] SR1_test;
  vector[N2] SR2_test;
  vector[N2] SR3_test;
  // Random effects
  vector[N] epsilon;

  // idiosyncratic normal effects;
  epsilon = psi * e_raw;

  // ** Linear predictor **
  u_train = epsilon[1:N1];
  u_test  = epsilon[id_test];
  
  S   = beta0 + x_train * beta + u_train; 
  SR1 = S + x_train * gamma1; 
  SR2 = S + x_train * gamma2; 
  SR3 = S + x_train * gamma3;  
  if (N2 != 0) {
    S_test   = beta0 + x_test * beta + u_test;
    SR1_test = S_test + x_test * gamma1;
    SR2_test = S_test + x_test * gamma2;
    SR3_test = S_test + x_test * gamma3;
  }
}

model {
  // *** Priors *** 
  target += student_t_lpdf(beta | 4, 0, 1);
  target += student_t_lpdf(gamma1 | 4, 0, 1);
  target += student_t_lpdf(gamma2 | 4, 0, 1);
  target += student_t_lpdf(gamma3 | 4, 0, 1);
  target += student_t_lpdf(beta0| 4, 0, 1);
  
  target += induced_dirichlet_lpdf(theta1 | rep_vector(1, K1), 0);
  target += induced_dirichlet_lpdf(theta2 | rep_vector(1, K2), 0);
  target += induced_dirichlet_lpdf(theta3 | rep_vector(1, K3), 0);

  // *** Random effects ***  
  // 1) idiosyncratic effects
  target += std_normal_lpdf(e_raw); 
  
  // *** Likelihood ***
  // * 1) Default indicator *
  target += bernoulli_logit_lpmf(D_train| S);
  // * 2) Ordinal outcome rater R1 * 
  for (i in 1:NR1_train) {
    target += ordered_logistic_lpmf(y1_train[i] | SR1[id_obs_R1_train[i]], theta1);
  }   
  // * 3) Ordinal outcome rater R2 * 
  for (i in 1:NR2_train) {
    target += ordered_logistic_lpmf(y2_train[i] | SR2[id_obs_R2_train[i]], theta2);
  } 
  // * 4) Ordinal outcome rater R3 * 
  for (i in 1:NR3_train) {
    target += ordered_logistic_lpmf(y3_train[i] | SR3[id_obs_R3_train[i]], theta3);
  } 
}

generated quantities {
  vector[N1] log_lik_all;
  vector[N1] log_lik_D;
  vector[N1] log_lik_R1_with_NA;
  vector[N1] log_lik_R2_with_NA;
  vector[N1] log_lik_R3_with_NA;
  vector[NR1_train] log_lik_R1;
  vector[NR2_train] log_lik_R2;
  vector[NR3_train] log_lik_R3;
  vector[N2] log_lik_all_test;
  vector[N2] log_lik_D_test;
  vector[N2] log_lik_R1_with_NA_test;
  vector[N2] log_lik_R2_with_NA_test;
  vector[N2] log_lik_R3_with_NA_test;
  vector[NR1_test] log_lik_R1_test;
  vector[NR2_test] log_lik_R2_test;
  vector[NR3_test] log_lik_R3_test;
  

  log_lik_R1_with_NA = rep_vector(0, N1);
  log_lik_R2_with_NA = rep_vector(0, N1);
  log_lik_R3_with_NA = rep_vector(0, N1);
  for (n in 1:NR1_train) {
    log_lik_R1[n] = ordered_logistic_lpmf(y1_train[n] | SR1[id_obs_R1_train[n]], theta1);
    log_lik_R1_with_NA[id_obs_R1_train[n]] = log_lik_R1[n];
  } 
  for (n in 1:NR2_train) {
    log_lik_R2[n] = ordered_logistic_lpmf(y2_train[n] | SR2[id_obs_R2_train[n]], theta2);
    log_lik_R2_with_NA[id_obs_R2_train[n]] = log_lik_R2[n];
  } 
  for (n in 1:NR3_train) {
    log_lik_R3[n] = ordered_logistic_lpmf(y3_train[n] | SR3[id_obs_R3_train[n]], theta3);
    log_lik_R3_with_NA[id_obs_R3_train[n]] = log_lik_R3[n];
  } 
  for (n in 1:N1){
    log_lik_D[n]  = bernoulli_logit_lpmf(D_train[n] | S[n]);
  }
  log_lik_all = log_lik_D + log_lik_R1_with_NA + log_lik_R2_with_NA + log_lik_R3_with_NA;


  if (N2 != 0) {
     log_lik_R1_with_NA_test = rep_vector(0, N2);
     log_lik_R2_with_NA_test = rep_vector(0, N2);
     log_lik_R3_with_NA_test = rep_vector(0, N2);
     for (n in 1:N2){
        log_lik_D_test[n]  = bernoulli_logit_lpmf(D_test[n]  | S_test[n]);
     }
     for (n in 1:NR1_test){
        log_lik_R1_test[n]  = ordered_logistic_lpmf(y1_test[n] | SR1_test[id_obs_R1_test[n]], theta1);
        log_lik_R1_with_NA_test[id_obs_R1_test[n]] = log_lik_R1_test[n];
     }  
     for (n in 1:NR2_test){
        log_lik_R2_test[n]  = ordered_logistic_lpmf(y2_test[n] | SR2_test[id_obs_R2_test[n]], theta2);
        log_lik_R2_with_NA_test[id_obs_R2_test[n]] = log_lik_R2_test[n];
     }
     for (n in 1:NR3_test){
        log_lik_R3_test[n]  = ordered_logistic_lpmf(y3_test[n] | SR3_test[id_obs_R3_test[n]], theta3);
        log_lik_R3_with_NA_test[id_obs_R3_test[n]] = log_lik_R3_test[n];
     }
     log_lik_all_test = log_lik_D_test + log_lik_R1_with_NA_test + log_lik_R2_with_NA_test + log_lik_R3_with_NA_test;
  }
}
