functions {
  /* compute the cholesky factor of an AR1 correlation matrix
   * Args: 
   *   ar: AR1 autocorrelation 
   *   nrows: number of rows of the covariance matrix 
   * Returns: 
   *   A nrows x nrows matrix 
   */ 
   matrix cholesky_cor_ar1(real ar, int nrows) { 
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

  /* scale and correlate time random effects (b_t)
   * Args: 
   *   b_raw: standardized and independent random effects
   *   sd_b: standard deviation of the random effects
   *   chol_cor: cholesky factor of the correlation matrix
   * Returns: 
   *   vector of scaled and correlated random effects
   */ 
   vector scale_cov_b(vector b_raw, real sd_b,
		      matrix chol_cor) { 
     vector[rows(b_raw)] b; 
     b = sd_b * chol_cor * b_raw;                 
     return b; 
   }
  /*  compute the cutpoints from probabilities
   * Args: 
   *   probabilities: vector of probabilities
   */ 
   vector make_cutpoints(vector probabilities) {
     int C = rows(probabilities) - 1; 
     vector[C] cutpoints;
     real running_sum = 0;
     for(c in 1:C) {
       running_sum += probabilities[c];
       cutpoints[c] = logit(running_sum);
     }
     return  cutpoints;
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
  vector[P] beta1;
  vector[P] beta2;
  vector[P] beta3;
  real beta0;
  // ** Cutpoints ** 
  simplex[K1] pi1;
  simplex[K2] pi2;
  simplex[K3] pi3;
  // ** Standard deviation parameters **
  vector[I] tau; // local shrinkage parameter
  real<lower=0> q; 
  real<lower=0> omega; 
  real<lower=0> psi; 
  // ** Persistence of AR(1) **
  real<lower = 0, upper = 1> rhostar;
  real<lower = 0, upper = 1> phistar;
  // ** Raw random effects (iid) **
  vector[I] a_raw;
  vector[T] b_raw;
  vector[N] e_raw;
}

transformed parameters {
  vector[N1] S;
  vector[N1] S1;
  vector[N1] S2;
  vector[N1] S3;
  vector[N1] u_train;
  vector[N2] S_test;
  vector[N2] S1_test;
  vector[N2] S2_test;
  vector[N2] S3_test;
  vector[N2] u_test;
  vector[K1 - 1] theta1; // thresholds R1 
  vector[K2 - 1] theta2; // thresholds R2
  vector[K3 - 1] theta3; // thresholds R3
  
  vector[I] a;  // original firm effects
  vector[T] b;  // original market factor
  matrix[T, T] chol_cor_b;
  real<lower = -1, upper = 1> phi;
  matrix[T, T] chol_cor_eps;
  vector[N] epsilon;  // original idiosyncratic effects
  real<lower = -1, upper = 1> rho;

  // ** Cutpoints **
  theta1 = make_cutpoints(pi1);
  theta2 = make_cutpoints(pi2);
  theta3 = make_cutpoints(pi3);

  // ** Firm effects **
  a = tau .* a_raw;
  
  // ** Time AR(1) effects **
  phi = 2 * phistar - 1;
  chol_cor_b = cholesky_cor_ar1(phi, T);
  b = scale_cov_b(b_raw, 1, chol_cor_b);
  
  // ** Idiosyncratic AR(1) effects **
  rho = 2 * rhostar - 1;
  chol_cor_eps = cholesky_cor_ar1(rho, T);
  epsilon = scale_cov_eps(e_raw, psi, chol_cor_eps, begin_eps, end_eps, year_id);

  // ** Linear predictor **
  u_train = a[firm_id_train] - omega * b[year_id_train] + epsilon[1:N1];
  u_test  = a[firm_id_test]  - omega * b[year_id_test]  + epsilon[id_test];
  S =  beta0 + x_train * beta + u_train;
  // next beta0 will not be included in the equation but "absorbed" by the threshold params
  S1 =  x_train * beta1 + u_train;
  S2 =  x_train * beta2 + u_train;
  S3 =  x_train * beta3 + u_train;

  if (N2 != 0) {
    S_test  = beta0 + x_test * beta + u_test;
    S1_test =         x_test * beta + u_test;
    S2_test =         x_test * beta + u_test;
    S3_test =         x_test * beta + u_test;
  }
}

model {
  // *** Priors *** 
  target += student_t_lpdf(beta | 4, 0, 1);
  target += student_t_lpdf(beta1| 4, 0, 1);
  target += student_t_lpdf(beta2| 4, 0, 1);
  target += student_t_lpdf(beta3| 4, 0, 1);
  target += student_t_lpdf(beta0| 4, 0, 2);
  target += dirichlet_lpdf(pi1 | prior_counts_R1);
  target += dirichlet_lpdf(pi2 | prior_counts_R2);
  target += dirichlet_lpdf(pi3 | prior_counts_R3);

  // *** Random effects ***  
  // 1) firm effects
  target += inv_gamma_lpdf(q | 0.5, 0.5); 
  target += std_normal_lpdf(a_raw); 
  target += normal_lpdf(tau | 0, q); 

  // 2) time effects
  target += beta_lpdf(phistar | 20, 1.5);  
  target += std_normal_lpdf(b_raw); 

  // 3) idiosyncratic effects
  target += beta_lpdf(rhostar | 20, 5);
  target += std_normal_lpdf(e_raw); 
  
  // *** Likelihood ***
  // * 1) Default indicator *
  target += bernoulli_logit_lpmf(D_train| S);
  // * 2) Ordinal outcome rater R1 * 
  for (i in 1:NR1_train) {
    target += ordered_logistic_lpmf(y1_train[i] | S1[id_obs_R1_train[i]], theta1);
  }   
  // * 3) Ordinal outcome rater R2 * 
  for (i in 1:NR2_train) {
    target += ordered_logistic_lpmf(y2_train[i] | S2[id_obs_R2_train[i]], theta2);
  } 
  // * 4) Ordinal outcome rater R3 * 
  for (i in 1:NR3_train) {
    target += ordered_logistic_lpmf(y3_train[i] | S3[id_obs_R3_train[i]], theta3);
  } 
}

generated quantities {
  vector[N2] log_lik_D_test;
  vector[NR1_test] log_lik_R1_test;
  vector[NR2_test] log_lik_R2_test;
  vector[NR3_test] log_lik_R3_test;
  if (N2 != 0) {
     for (n in 1:N2){
        log_lik_D_test[n]  = bernoulli_logit_lpmf(D_test[n]  | S_test[n]);
     }
     for (n in 1:NR1_test){
        log_lik_R1_test[n]  = ordered_logistic_lpmf(y1_test[n] | SR_test[id_obs_R1_test[n]], theta1);
     }  
     for (n in 1:NR2_test){
        log_lik_R2_test[n]  = ordered_logistic_lpmf(y2_test[n] | SR_test[id_obs_R2_test[n]], theta2);
     }
     for (n in 1:NR3_test){
        log_lik_R3_test[n]  = ordered_logistic_lpmf(y3_test[n] | SR_test[id_obs_R3_test[n]], theta3);
     }
  }
}