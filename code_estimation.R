########################################################
########################################################
### CODE FOR ESTIMATION WITH rstan ON SYNTHETIC DATA ###
########################################################
########################################################
library(rstan)
library(coda)

## ----load_data-------------------------------------------------------------------
load("synthetic_dat.rda")
dat <- synthetic_dat

take.subsample <- TRUE

if (take.subsample) {
  set.seed(123)
  firm_ids_small <- sample(unique(dat$firm_id), 100)
  dat <- subset(dat, firm_id %in% firm_ids_small)
}

## ---------------- stan_models ----------------------
models <- list(
  list(## Model S1
    model = "Stan/Model_S1_logit_priors-eps-N_bias-0.stan",
    pars  = c("beta", "beta0",
              "theta1", "theta2", "theta3",
              "psi", 
              "log_lik_all", 
              "log_lik_D_test", "log_lik_R1_test", "log_lik_R2_test", "log_lik_R3_test")),
  list(## Model S2
    model = "Stan/Model_S2_logit_priors-eps-N_bias-diffgammas.stan",
    pars  = c("beta", "gamma1", "gamma2", "gamma3",  "beta0", 
              "theta1", "theta2", "theta3",
              "psi",
              "log_lik_all", 
              "log_lik_D_test", "log_lik_R1_test", "log_lik_R2_test", "log_lik_R3_test")),
  list(## Model D1
    model = "Stan/Model_D1_logit_priors-a-HN-b-AR1-eps-AR1_bias-0.stan",
    pars  = c("beta", "beta0", 
              "theta1", "theta2", "theta3",
              "q2",
              "b", "omega", "phi",
              "rho", "psi",
              "log_lik_all", 
              "log_lik_D_test", "log_lik_R1_test", "log_lik_R2_test", "log_lik_R3_test"
    )),
  list(## Model D2
    model = "Stan/Model_D2_logit_priors-a-HN-b-AR1-eps-AR1_bias-diffgammas.stan",
    pars  = c("beta", "gamma1", "gamma2", "gamma3",  "beta0",
              "theta1", "theta2", "theta3",
              "q2", 
              "b", "omega", "phi", 
              "rho", "psi",
              "log_lik_all", 
              "log_lik_D_test", "log_lik_R1_test", "log_lik_R2_test", "log_lik_R3_test"
    )),
  list(## Model PM
    model = "Stan/Model_PM_logit_priors-a-HN-b-AR1-eps-AR1_bias-diffbetas-delta-AR1.stan",
    pars  = c("beta", "gamma1", "gamma2", "gamma3", "beta0",
              "theta1", "theta2", "theta3",
              "q2", 
              "b", "omega", "phi",
              "rho", "psi",
              "delta", "lambda", "phidelta",
              "log_lik_all", 
              "log_lik_D_test", "log_lik_R1_test", "log_lik_R2_test", "log_lik_R3_test")))


##############################
### OUT OF SAMPLE ANALYSIS ###
##############################
# for the one step ahead out of sample exercise, we train the model on the
# data containing years 1 : t and check the log predictive 
# likelihoods for the following year t + 1

end_of_training_samples <- 19
number_of_models        <- length(models)
for (t in end_of_training_samples) {
  for (s in seq_len(number_of_models)){
    # select only observations in years 1:(t + 1)
    dat_tmp <- subset(dat, year_id %in% seq_len(t + 1))
    dat_tmp$firm_id <- as.numeric(factor(dat_tmp$firm_id))
    
    id_train <- which(dat_tmp$year_id <= t)
    id_test  <- which(dat_tmp$year_id ==  t + 1)
    
    N1 <- length(id_train)
    N2 <- length(id_test)
    
    # matrix of ratings in training sample
    y_train <- dat_tmp[id_train, c("R1", "R2", "R3")]
    # matrix of ratings in test sample
    y_test  <- dat_tmp[id_test,  c("R1", "R2", "R3")]
    
    begin_eps <- seq_len(N1 + N2)[!duplicated(dat_tmp$firm_id)]
    end_eps   <- c(begin_eps[-1] - 1, N1 + N2)
    
    DATA <- list(
      N             = N1 + N2, 
      N1            = N1, 
      N2            = N2,
      NR1_train     = sum(!is.na(y_train[, 1])),
      NR2_train     = sum(!is.na(y_train[, 2])),
      NR3_train     = sum(!is.na(y_train[, 3])),
      NR1_test      = sum(!is.na(y_test[, 1])),
      NR2_test      = sum(!is.na(y_test[, 2])),
      NR3_test      = sum(!is.na(y_test[, 3])),
      P             = 7,  
      J             = 3,
      I             = length(unique(dat_tmp$firm_id)),
      T             = max(dat_tmp$year_id),
      id_test       = N1 + seq_len(N2),
      begin_eps     = begin_eps,
      end_eps       = end_eps,
      firm_id       = dat_tmp$firm_id,
      firm_id_train = dat_tmp$firm_id[id_train],
      firm_id_test  = dat_tmp$firm_id[id_test],
      year_id       = dat_tmp$year_id,
      year_id_train = dat_tmp$year_id[id_train],
      year_id_test  = dat_tmp$year_id[id_test],
      id_obs_R1_train = which(!is.na(y_train[, 1])),
      id_obs_R2_train = which(!is.na(y_train[, 2])),
      id_obs_R3_train = which(!is.na(y_train[, 3])),
      id_obs_R1_test  = which(!is.na(y_test[, 1])),
      id_obs_R2_test  = which(!is.na(y_test[, 2])),
      id_obs_R3_test  = which(!is.na(y_test[, 3])),
      K1 = max(y_train[, 1], na.rm = TRUE),
      K2 = max(y_train[, 2], na.rm = TRUE),
      K3 = max(y_train[, 3], na.rm = TRUE),
      prior_counts_R1 = table(y_train[!is.na(y_train[, 1]), 1]),
      prior_counts_R2 = table(y_train[!is.na(y_train[, 2]), 2]),
      prior_counts_R3 = table(y_train[!is.na(y_train[, 3]), 3]),
      y1_train = y_train[!is.na(y_train[, 1]), 1],
      y2_train = y_train[!is.na(y_train[, 2]), 2],
      y3_train = y_train[!is.na(y_train[, 3]), 3],
      y1_test = y_test[!is.na(y_test[, 1]), 1],
      y2_test = y_test[!is.na(y_test[, 2]), 2],
      y3_test = y_test[!is.na(y_test[, 3]), 3],
      D_train = dat_tmp$D[id_train], 
      D_test  = dat_tmp$D[id_test],
      x_train = dat_tmp[id_train, grep("X", colnames(dat_tmp))], 
      x_test  = dat_tmp[id_test , grep("X", colnames(dat_tmp))]
    )
    # ------------- prepare files and folders ---------------       
    model_name <- gsub("Stan/", "", gsub("\\..*", "", models[[s]][["model"]]))
    res_folder <- paste0("Results_Synthetic_Data/", model_name, collapse = "/")
    pars <-  models[[s]][["pars"]]
    if (!dir.exists(res_folder)) dir.create(res_folder, recursive = TRUE)
    if (t == max(dat$year_id)) {
      file_out <- paste0(res_folder, "/", sprintf("results_%s_fullsample.rda", model_name))
      pars <- gsub("_test", "", pars)
    } else {
      file_out <- paste0(res_folder, "/", sprintf("OOS_results_%s_testperiod_%i.rda", model_name, t + 1))
    }
    # ------------- analysis -------------
    fit <- stan(file = models[[s]][["model"]],
                pars = models[[s]][["pars"]],
                data = DATA, 
                chains = 5, 
                iter = 100L, # 2000L,
                save_warmup = FALSE, 
                control=list(adapt_delta = 0.99))
    
    save(fit, file = file_out)
  }
}


