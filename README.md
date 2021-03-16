# Dynamic model of corporate credit ratings and defaults

This repository contains code and a synthetic data set for reproducing the modeling framework in Vana and Hornik (2020).

## Data set

The synthetic data set `synthetic_dat.rda` has been generated using 
the **synthpop** R package (Nowok et. al, 2016) by replacing all observations
with values simulated from probability distributions specified to 
preserve key features of the actual observed data.

The data set contains 19952 rows and 13 columns:

* `firm_id`: integer containing the firm id

* `year_id`: integer containing the year id

* `R1`: integer corresponding to the rating classes assigned by rater 1

* `R2`: integer corresponding to the rating classes assigned by rater 2

* `R3`: integer corresponding to the rating classes assigned by rater 3

* `D` : binary indicator for default

* `X1` to `X7`: standardized covariates.



## References
  Beata Nowok, Gillian M. Raab, Chris Dibben (2016). synthpop: Bespoke
  Creation of Synthetic Data in R. Journal of Statistical Software,
  74(11), 1-26. doi:10.18637/jss.v074.i11
