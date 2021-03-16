# Dynamic model of corporate credit ratings and defaults

This repository contains code and a synthetic data set for reproducing the modeling framework in Vana and Hornik (2020).

## Data set

The synthetic data set `synthetic_dat.rda` has been generated using 
the **synthpop** R package (Nowok et. al, 2016) by replacing all observations
with values simulated from probability distributions specified to 
preserve key features of the actual observed data.

The data set contains 19952 rows and 13 columns:

* `firm_id`: integer containing the firm id, maximum is 2528

* `year_id`: integer containing the year id, maximum is 19

* `R1`: integer corresponding to the rating classes assigned by rater 1

* `R2`: integer corresponding to the rating classes assigned by rater 2

* `R3`: integer corresponding to the rating classes assigned by rater 3

* `D` : binary indicator for default

* `X1` to `X7`: standardized covariates.

## Models

The Stan codes for the five models introduced in the papers can be found in the `Stan` folder.

* `Model_S1_logit_priors-eps-N_bias-0.stan`
* `Model_S2_logit_priors-eps-N_bias-diffgammas.stan`
* `Model_D1_logit_priors-a-HN-b-AR1-eps-AR1_bias-0.stan`
* `Model_D2_logit_priors-a-HN-b-AR1-eps-AR1_bias-diffgammas.stan`
* `Model_PM_logit_priors-a-HN-b-AR1-eps-AR1_bias-diffbetas-delta-AR1.stan`

## Estimation of the models using RStan

The file `code_estimation.R` contains the code for reproducing the analysis in that it performs the out of sample analysis by repeatedly estimating the five models presented in the paper (`PM`, `S1`, `S2`, `D1`, `D2`) using the **RStan** package (Stan Development Team, 2020) for different training vs. test samples from `synthetic_dat.rda`. In the one-step-ahead prediction exercise  we train the model on data containing years 1 to $t$ and then evaluate the log predictive likelihoods for the following year $t + 1$. We illustrate the approach for $t=13,\ldots, 19$.

The code creates a folder `results` which contains subfolders with the model names which in turn contain the  `.rda` files for each test period.

## References
  Beata Nowok, Gillian M. Raab, Chris Dibben (2016). synthpop: Bespoke
  Creation of Synthetic Data in R. Journal of Statistical Software,
  74(11), 1-26. doi:10.18637/jss.v074.i11

  Stan Development Team (2020). RStan: the R interface to Stan. R
  package version 2.19.3. http://mc-stan.org/.

