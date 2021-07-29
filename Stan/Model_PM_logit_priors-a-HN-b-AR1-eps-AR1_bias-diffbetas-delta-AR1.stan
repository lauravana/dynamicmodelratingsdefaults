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
  /* compute the cholesky factor of an AR1 correlation matrix
   * Args: 
   *   ar: AR1 autocorrelation 
   *   nrows: number of rows of the covariance matrix 
   * Returns: 
   *   A nrows x nrows matrix 
   */ 
   matrix cholesky_cor_ar1(real ar,   int nrows) { 
     matrix[nrows, nrows] mat; 
     vector[nrows - 1] gamma; 
     mat = diag_matrix(rep_vector(1, nrows)); 
     for (i in 2:nrows) { 
       gamma[i - 1] = pow(ar, i - 1); 
       for (j in 1:(i - 1)) { 
         mat[i, j] = gamma[i - j]; 
         mat[j, i] = gamma[i - j]; 
       } 
     } 
     return cholesky_decompose(1 / (1 - ar^2) * mat); 
   }

  /* scale and correlate idiosyncratic effects (epsilon_it) - common rho
   * Args: 
   *   zerr: standardized and independent idiosyncratic effects
   *   sderr: standard deviation of the idiosyncratic effects
   *   chol_cor: cholesky factor of the correlation matrix
   *   begin: the first observation in each group 
   *   end: the last observation in each group    
   *   yid: an integer indication the id of the year for each obs
   * Returns: 
   *   vector of scaled and correlated idiosyncratic effects
   */ 
   vector scale_cov_eps(vector zerr, real sderr, matrix chol_cor, 
                        int[] begin, int[] end, int[] yid) { 
     vector[rows(zerr)] err; 
     for (i in 1:size(begin)) { 
       err[begin[i]:end[i]] = 
         sderr * chol_cor[yid[begin[i]:end[i]], yid[begin[i]:end[i]]] * zerr[begin[i]:end[i]];
     }                        
     return err; 
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
  vector<lower=0>[K1] prior_counts_R1; // number of observations in each class
  vector<lower=0>[K2] prior_counts_R2; // number of observations in each class
  vector<lower=0>[K3] prior_counts_R3; // number of observations in each class
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
  real beta0tilde;

  // ** Cutpoints ** 
  ordered[K1 - 1] theta1; // thresholds R1
  ordered[K2 - 1] theta2; // thresholds R2
  ordered[K3 - 1] theta3; // thresholds R3

  // ** Standard deviation parameters **
  vector[I] tau; // shrinkage parameter
  real<lower=0> q2;
  //vector<lower=0>[I] lambda_sq;
  //vector<lower=0>[I] om;
  //real <lower=0> tau ; // global shrinkage parameter
  //vector<lower=0>[I] lambda ; // local shrinkage parameter
  //real<lower=0> caux ;
  //real<lower=0> tau_sq;
  //real<lower=0> eta;
  //real<lower=0> tau;
  real<lower=0> omega; 
  real<lower=0> psi; 
  // ** Raw rater factor (iid) ** 
  vector[T] delta;
  vector<lower=0>[J-1] lambda;
  // ** Persistence of AR(1) **
  real<lower = 0, upper = 1> rhostar;
  real<lower = 0, upper = 1> phibstar;
  real<lower = 0, upper = 1> phideltastar;
  // ** Raw random effects **
  vector[I] a_raw;
  vector[T] btilde;
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
  vector[I] a;  // original firm effects
  matrix[T, T] chol_cor_b;
  real<lower = -1, upper = 1> phib;
  matrix[T, T] chol_cor_eps;
  vector[N] epsilon;  // original idiosyncratic effects
  real<lower = -1, upper = 1> rho;
  matrix[T, T] chol_cor_delta;
  real<lower = -1, upper = 1> phidelta;

  //vector<lower=0>[I] lambda_tilde ; // ’ truncated ’ local shrinkage parameter
  //real <lower =0 > c; // slab scale

  // ** Firm effects **
  //c = sqrt(caux);
  //lambda_tilde = sqrt ( c ^2 * square ( lambda ) ./ (c ^2 + tau ^2* square ( lambda )) );
  a =  a_raw .* tau; //lambda_tilde * tau; //sqrt(tau_sq) * sqrt(lambda_sq) .* a_raw;
  
  // ** Time AR(1) effects **
  phib = 2 * phibstar - 1;
  chol_cor_b = cholesky_cor_ar1(phib, T);
  //b = scale_cov_b(b_raw, 1, chol_cor_b);
  
  // ** Idiosyncratic AR(1) effects **
  rho = 2 * rhostar - 1;
  chol_cor_eps = cholesky_cor_ar1(rho, T);
  epsilon = scale_cov_eps(e_raw, psi, chol_cor_eps, begin_eps, end_eps, year_id);
 
  // ** Rater factor AR(1) **
  phidelta = 2 * phideltastar - 1;
  chol_cor_delta = cholesky_cor_ar1(phidelta, T);

  // ** Linear predictor **
  u_train = a[firm_id_train] - btilde[year_id_train] + epsilon[1:N1]; 
  S       = x_train * beta + u_train;
  
  SR1 = S + x_train * gamma1 +             delta[year_id_train];
  SR2 = S + x_train * gamma2 + lambda[1] * delta[year_id_train]; 
  SR3 = S + x_train * gamma3 + lambda[2] * delta[year_id_train];
  if (N2 != 0) {
    u_test  = a[firm_id_test]  - btilde[year_id_test] + epsilon[id_test]; //   
    S_test   = x_test * beta + u_test; 
    SR1_test = S_test + x_test * gamma1 +             delta[year_id_test];
    SR2_test = S_test + x_test * gamma2 + lambda[1] * delta[year_id_test]; 
    SR3_test = S_test + x_test * gamma3 + lambda[2] * delta[year_id_test];
  }
}

model {
  // *** Priors *** 
  target += student_t_lpdf(beta | 4, 0, 1);
  target += student_t_lpdf(gamma1 | 4, 0, 1);
  target += student_t_lpdf(gamma2 | 4, 0, 1);
  target += student_t_lpdf(gamma3 | 4, 0, 1);
  target += student_t_lpdf(beta0tilde | 4, 0, 1);

  target += induced_dirichlet_lpdf(theta1 | rep_vector(1, K1), 0);
  target += induced_dirichlet_lpdf(theta2 | rep_vector(1, K2), 0);
  target += induced_dirichlet_lpdf(theta3 | rep_vector(1, K3), 0);

  // *** Random effects ***  
  // 1) firm effects
  //om ~ inv_gamma(0.5, 1);
  //lambda_sq ~ inv_gamma(0.5, 1 ./ om);
//tau ~ cauchy(0, 1) is equivalent with;;
  //eta ~ inv_gamma(0.5, 1);
  //tau_sq ~ inv_gamma(0.5, 1 / eta);

  target += std_normal_lpdf(a_raw); 
  //target += cauchy_lpdf(lambda |0, 1);
  //target += cauchy_lpdf(tau |0 , 10^-4);
  //target += inv_gamma_lpdf(caux|0.5 * 2, 0.5*2);
  //target += cauchy_lpdf(tau | 0,1); 
  target += inv_gamma_lpdf(q2 | 0.5, 0.2275); 
  //target += gamma_lpdf(tau2 | 0.5, 1/(2 * q2)); 
  target += normal_lpdf(tau | 0, sqrt(q2)); 
  //target += double_exponential_lpdf(a| 0, sqrt(q2));

  // 2) time effects
  target += beta_lpdf(phibstar | 20, 2.5);  
  target += multi_normal_cholesky_lpdf(btilde | rep_vector(beta0tilde, T), omega * chol_cor_b);
  target += cauchy_lpdf(omega|0, 2); 
  
  // 3) idiosyncratic effects
  target += beta_lpdf(rhostar | 20, 5);
  target += std_normal_lpdf(e_raw); 
  target += cauchy_lpdf(psi | 0, 2); 
  
  // 4) time effects - rater
  target += beta_lpdf(phideltastar | 1, 1);
  target += multi_normal_cholesky_lpdf(delta | rep_vector(0, T), chol_cor_delta);
  target += cauchy_lpdf(lambda | 0, 2); //std_normal_lpdf(lambda);


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
  // Train set quantities
  vector[N1] log_lik_all;
  vector[N1] log_lik_D;
  vector[N1] log_lik_R1_with_NA;
  vector[N1] log_lik_R2_with_NA;
  vector[N1] log_lik_R3_with_NA;
  vector[NR1_train] log_lik_R1;
  vector[NR2_train] log_lik_R2;
  vector[NR3_train] log_lik_R3;
  // Test set quantities
  vector[N2] log_lik_all_test;
  vector[N2] log_lik_D_test;
  vector[N2] log_lik_R1_with_NA_test;
  vector[N2] log_lik_R2_with_NA_test;
  vector[N2] log_lik_R3_with_NA_test;
  vector[NR1_test] log_lik_R1_test;
  vector[NR2_test] log_lik_R2_test;
  vector[NR3_test] log_lik_R3_test;
  // Other quantities
  vector[T] b;
  real beta0;
  b = btilde - beta0tilde;
  beta0 = -beta0tilde;

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