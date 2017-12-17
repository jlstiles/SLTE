# Simulations
Read the [Blip Variance Paper, full version](https://github.com/jlstiles/Simulations/blob/master/blip-variance-article.pdf) 

To install gentmle and the other tools necessary to obtain results in in the
Blip Variance paper:

devtools::install_github("jlstiles/Simulations")

The following knitr file has instructions for how to generate the results, including plots and charts, in the paper.  I will note to the reader in the code comments, how to proceed with parallelizing so the process does not take so much time as well as issues with memory one should consider when running ensemble methods as applied here.

[SET UP: run this first](#setup)

[section 4.1.2 manufactured noise simulations](#section4.1.2)

[section 4.1.3 manufactured noise simulations](#section4.1.3)

[section 4.3 well specified precaution on skewing](#section4.3)

[section 4.4.1 narrow model failures](#section4.4.1)

[section 4.4.2 narrow model failures](#section4.4.2)

well-spec pscore, mispec. outcome model

[section 4.4.3 case 2a](#section4.4.3)

[section 4.4.4 case2b SuperLearner lib 1](#section4.4.42bSL1)

[section 4.4.4 case2b SuperLearner lib 2 with CV-TMLE](#section4.4.42bCVSL2)

[section 4.4.4 case2b SuperLearner lib 2 with TMLE, overfit hurts without CV](#section4.4.42bSL2)

[section 4.4.4 combining cases as in paper](#section4.4.42bcombo)

mispecified treatment and outcome models

[section 4.5.1](#section4.5.1)

[section 4.5.2](#section4.5.2)

[section 5 real data example](#section7)

<a name="setup"></a>
```R
case = "setup"
resultsGotten = TRUE
source_file = "~/Dropbox/Jonathan/Simulations/source_paper.R"
source(source_file)

devtools::install_github("jlstiles/Simulations")
library(Simulations)
source("~/Dropbox/Jonathan/Simulations/WrappersVblip1.R")
# source('/home/jlstiles/R/WrappersVblip1.R')
```

<a name="section4.4.1"></a>
```R
# section 4.4.1
case = "LRcase2a"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
# head of compiled simulation results, all is computed from this data.frame 
results_LRcase2a 

# figure in the paper
gg_LRcase2a

```

```R

case = "LRcase2b"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
# compiled simulation results, all is computed from this data.frame 
results_LRcase2b

# figure in the paper
gg_LRcase2b

```

<a name="section4.4.2"></a>
```R
case = "HALcase2a"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
results_HALcase2a = results
gg_HALcase2a
coverage_HALcase2a
performance.sig_HALcase2a

```

```R
case = "HALcase2b"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
results_HALcase2b = results
gg_HALcase2b
coverage_HALcase2b
performance.sig_HALcase2b
```

<a name="section4.4.3"></a>
```R
# section 4.4.3
case = "case2a"
resultsGotten = TRUE
B=n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
gg_ATEcase2a
gg_BVcase2a
performance.ate_case2a
performance.sig_case2a
coverage_case2a
SL_results_case2a
```

<a name="section4.4.42bSL1"></a>
```R
# section 4.4.4 using SuperLearner Library 1
case = "case2bSL1"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
results_case2bSL1 = results
gg_ATEcase2bSL1
gg_BVcase2bSL1
performance.ate_case2bSL1
performance.sig_case2bSL1
coverage_case2bSL1
SL_results_case2bSL1
```

<a name="section4.4.42bCVSL2"></a>
```R
# section 4.4.4 using SuperLearner Library 2 but with CV-TMLE

# RUN ten times with B  = 100 on 24 cores each
g0 = g0_linear
Q0 = Q0_trig
testdata=gendata(1000000, g0=g0, Q0 = Q0)
blip_true = with(testdata,Q0(1,W1,W2,W3,W4)-Q0(0,W1,W2,W3,W4))
ATE0 = mean(blip_true)
var0 = var(blip_true)

SL.library = SL.library2
SL.libraryG = list("SL.glm")

cl = makeCluster(detectCores(), type = "SOCK")
registerDoSNOW(cl)
clusterExport(cl,cl_export)

B=100
n=1000
ALL=foreach(i=1:B,.packages=c("gentmle2","mvtnorm","hal","Simulations","SuperLearner"),
            .errorhandling = "remove")%dopar%
            {sim_cv(n, g0 = g0, Q0 = Q0, SL.library=SL.library, 
                    SL.libraryG=SL.libraryG,method = "method.NNloglik",cv=FALSE
            )}

results = data.matrix(data.frame(do.call(rbind, ALL)))

# Once you have compiled 10 of these results via rbind, into one results 
# data.frame do the following for later combining of results

# save the following objects
# rbind all 10 results data.frames into one data.frame named results
case = "case2bCVSL2"
source(source_file)

# save the following objects
results_case2bCVSL2
gg_ATEcase2bCVSL2
gg_BVcase2bCVSL2
performance.ate_case2bCVSL2
performance.sig_case2bCVSL2
coverage_case2bCVSL2
SL_results_case2bCVSL2
  ```

<a name="section4.4.42bSL2"></a> 
```R
# section 4.4.4 using SuperLearner Library 2 without CV-tmle so 
# it has a skewed sampling dist

case = "case2bSL2"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects 
results_case2bSL2
gg_ATEcase2bSL2
gg_BVcase2bSL2
performance.ate_case2bSL2
performance.sig_case2bSL2
coverage_case2bSL2
SL_results_case2bSL2

results[order(results_case2bSL2[,"Qcoef.rangerFull_screen.Main"]),1]

df = data.frame(x=results_case2bSL2[,"Qcoef.xgbFull_All"], 
                y=results_case2bSL2[,1])
assholeForest = ggplot(df, aes(x=x,y=y))+geom_point()+
  labs(x="Random Forest Superlearner Coeff",y="blip variance estimate")+
  geom_vline(xintercept = var0, color = "green")
caps = paste0("Truth is green line\n",
              "The bigger the random Forest coefficient the worse outliers due to\n",
              "overfitting the outcome regression")

assholeForest=ggdraw(add_sub(assholeForest,caps, 
                       x= 0, y = 0.5, hjust = 0, vjust = 0.5,
                       vpadding = grid::unit(1, "lines"), 
                       fontfamily = "", fontface = "plain",
                       colour = "black", size = 10, angle = 0, lineheight = 0.9))
assholeForest
```

<a name="section4.4.42bcombo"></a>   
```R
# section 4.4.4
# Have the following variables named as follows 
# results_case2bSL1 = results for case 2b with SL.library1
# results_case2bSL2 = results for case 2b with SL.library2, the overfitting lib.
# results_case2bCVSL2 = results for case 2b with SL.library2, but using CV-TMLE

case = "combo_case2b"
source(source_file)

# save the following objects
gg_cvadvert
MSE_cov
```       

<a name="section4.5.1"></a>  
```R
# section 4.5.1
##### case 3
case = "case3"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
results_case3
gg_ATEcase3
gg_BVcase3
performance.ate_case3
performance.sig_case3
coverage_case3
SL_results_case3
```

<a name="section4.5.2"></a>  
```R
# section 4.5.2
##### case 4 
case = "case4"
resultsGotten = FALSE
B=1000
n=1000
no.cores = detectCores()
source(source_file)

# save the following objects
results_case4 = results
gg_BVcase4
MSE_cov_case4
SL_results_case4
```

<a name="section4.3"></a>  
```R
##### section 4.3
##### wells
case = "wells"
source(source_file)

# to print the three arranged plots, each with four plots
nums = c(1,2,3,4)

# sample size 250
ml250=marrangeGrob(gg_wells[c(nums)],ncol=2,nrow=2, widths = c(3.5,3.5),
                heights=rep(c(1,1)))

# sample size 500
ml500=marrangeGrob(gg_wells[c(nums+16)],ncol=2,nrow=2, widths = c(3.5,3.5),
                heights=rep(c(1,1)))

# sample size 1000
ml1000=marrangeGrob(gg_wells[c(nums+32)],ncol=2,nrow=2, widths = c(3.5,3.5),
                heights=rep(c(1,1)))

# propensity score density
gg_pscoresWell
# see sim_well_small.R and writeup_wells.Rnw

# save the following objects specifying file path as shown
save(gg_pscoresWell, ml250, ml500, ml1000, gg_wells,
     file = "~/Dropbox/Jonathan/Simulations/results/wells.RData")
```

<a name="section4.1.2"></a>  
```R
# noise simulations section 4.1.2
# set case = "noise" or case = "noise_neg" for negative bias
case = "noise"
# case = "noise_neg"
no.cores  = 6
# suggest to test time for B = 1 to gauge time for B=1000
B=1000
source(source_file)

plotdf  = data.frame(n = sizes,coverage = res_noise[,5])
gg_coverage = ggplot(plotdf, aes(x = n, y = coverage)) + geom_point() +
  geom_hline(yintercept = .95, color = "green")
caption = paste0("coverage slowly becomes nominal as expected")
gg_coverage=ggdraw(add_sub(gg_coverage,caption,x= 0, y = 0.5, hjust = 0, vjust = 0.5,
                      vpadding = grid::unit(1, "lines"), fontfamily = "", fontface = "plain",
                      colour = "black", size = 10, angle = 0, lineheight = 0.9))
gg_coverage

gg_bias = ggplot(plotdf_biasmse, aes(x = n, y = bias, color = type)) + 
  geom_line() + geom_hline(yintercept = 0, color = "green")
caption = paste0("TMLE does debias in this case")
gg_bias=ggdraw(add_sub(gg_bias,caption,x= 0, y = 0.5, hjust = 0, vjust = 0.5,
                      vpadding = grid::unit(1, "lines"), fontfamily = "", fontface = "plain",
                      colour = "black", size = 10, angle = 0, lineheight = 0.9))

gg_bias
gg_mse = ggplot(plotdf_biasmse, aes(x = n, y = mse, color = type)) + 
  geom_line() + geom_hline(yintercept = 0, color = "green")
caption = paste0("TMLE raised MSE due to 2nd order remainder but",
                 "\nasymptotically behaves as expected.")
gg_mse=ggdraw(add_sub(gg_mse,caption, x= 0, y = 0.5, hjust = 0, vjust = 0.5,
                          vpadding = grid::unit(1, "lines"), fontfamily = "", 
                          fontface = "plain",colour = "black", size = 10, angle = 0, 
                          lineheight = 0.9))

gg_mse

# plots of sampling dists
ml

# plots of blip dists true overlayed with blip dists tmle
ml1

# save the following objects specifying file path as shown
save(ml, ml1, gg_bias, gg_mse, gg_coverage,L,
     file = "~/Dropbox/Jonathan/Simulations/results/noise.RData")

```

<a name="section4.1.3"></a>  
```R
# noise simulations section 4.1.3
# set case = "noise" or case = "noise_neg" for negative bias
# case = "noise"
case = "noise_neg"
no.cores  = 6
# suggest to test time for B = 1 to gauge time for B=1000
B=1000
source(source_file)

plotdf  = data.frame(n = sizes,coverage = res_noise[,5])
gg_coverage = ggplot(plotdf, aes(x = n, y = coverage)) + geom_point() +
  geom_hline(yintercept = .95, color = "green")

gg_bias = ggplot(plotdf_biasmse, aes(x = n, y = bias, color = type)) + 
  geom_line() + geom_hline(yintercept = 0, color = "green")

gg_mse = ggplot(plotdf_biasmse, aes(x = n, y = mse, color = type)) + 
  geom_line() + geom_hline(yintercept = 0, color = "green")

# plots of sampling dists
ml

# plots of blip dists true overlayed with blip dists tmle
ml1

# save the following objects specifying file path as shown
save(ml, ml1, gg_bias, gg_mse, gg_coverage,L,
     file = "~/Dropbox/Jonathan/Simulations/results/noiseneg.RData")
```

```R
# To obtain CIs via delta method for logistic regression plug-in estimates
# step 1:choose case
LRcases = list("CI_LRcase2a", "CI_LRcase2b", "CI_LRcase3", "CI_LRcase4")
case = "LRcase2a"
# number of simulations
B=1000
# sample size of each simulation
n=1000
source(source_file)
coverageLR
# you may rename it according to case
assign(paste0("coverage",case), coverageLR)
```
<a name="section7"></a>  
```R
# section 7
case = "example"
library(foreign)
library(Simulations)

source('/home/jlstiles/R/wrappers_ex.R')
wcgs = read.dta("/home/jlstiles/R/wcgs.dta")

# This will set the SL libraries.  We transform the design matrix
# to include squares, main terms and interactions and screen from there
source(source_file)

# This will take a couple of days as below. We limited the cores to 2 
# due to RAM issues but youc an specify the number of cores in the function below
stack = SL.stack(Y = Y, X = X, A = A, W = W, newdata = newdata, 
                 method = "method.NNloglik",
                 SL.library = SL.library, 
                 SL.libraryG = SL.libraryG, V=10, mc.cores = 2)

# perform the targeting step.  See package examples for more info
info = gentmle(stack$initdata, params = list(param_ATE,param_sigmaATE),
               approach = "full", max_iter = 100, simultaneous.inference = TRUE)
info$tmleests
info$initests
results = rbind(info$initests, info$tmleests)

# getting estimate for standard dev of blip using delta method
psi_sd = info$tmleests[2]^(.5)
IC_sd = .5*psi_sd^(-1)*info$Dstar[,2]

# log scaling blip variance due to left bound of CI in neg range
psi_log = log(info$tmleests[2])
IC_log = 1/info$tmleests[2]*info$Dstar[,2]

# correlation matrix the same whether using log scale, stand dev of blip
# or blip variance since all of these ICs are constants times each other
IC_ate = info$Dstar[,1]
IC = data.frame(IC_sd = IC_sd, IC_ate = IC_ate)
Sig = cor(IC)

# getting simultaneous zscore
library(mvtnorm)
z = rmvnorm(1000000, c(0,0), Sig)
n = length(IC[,2])
zscore = quantile(apply(z,1,FUN = function(x) max(abs(x))),.95)
zscore

# This ci for blip variance automatically generated by the gentmle
ci = ci_gentmle(info)

# other cis
ci_simul_sd = c(psi_sd, psi_sd-zscore*sd(IC_sd)*sqrt(n-1)/n, 
                psi_sd+zscore*sd(IC_sd)*sqrt(n-1)/n)

ci_simul_log = exp(c(psi_log,psi_log - zscore*sd(IC_log)*sqrt(n-1)/n,
                   psi_log + zscore*sd(IC_log)*sqrt(n-1)/n))

# compiling cis
ci_ate_sd_log = rbind(ci[1,c(2,4,5)], ci_simul_sd, ci_simul_log, ci[2,c(2,4,5)])
rownames(ci_ate_sd_log) = c("ate", "blip st. dev","blip var log-scaled","blip var")

# steps to convergence
info$steps
info$converge

# superlearner coefficients 
stack$Qcoef
stack$Gcoef

```