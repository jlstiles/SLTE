# Generate data
library(Simulations)
library(xgboost)
source("~/Dropbox/Jonathan/Simulations/WrappersVblip1.R")

Simulations::gendata

Q0=function(A,W1,W2,W3,W4) return(plogis(.2*(A+2*A*W4+3*W1+1*W2^2+.5*W3*W4+.25*W4)))
Q0=function(A,W1,W2,W3,W4) return(plogis(.2*(A+5*A*W4+.63*W1+1*W2^3+A*5*cos(W3)+.25*W4)))

# Q0=function(A,W1,W2,W3,W4) return(A+2*A*W4+3*W1+1*W2^2+.5*W3*W4+.25*W4)
# Q0=function(A,W1,W2,W3,W4) return(A+2*A*W4+.63*W1+1*W2^3+A*.5*cos(W3)+.25*W4)

g0=function(W1,W2,W3,W4) {plogis(-.28*W1+1*W2+.08*W3-.12*W4-1)}

g0 = g0_linear
Q0 = Q0_trig1

# 
# gendata.ols = function (n, g0, Q0) 
# {
#   W1 = runif(n, -3, 3)
#   W2 = rnorm(n)
#   W3 = runif(n)
#   W4 = rnorm(n)
#   A = rbinom(n, 1, g0(W1, W2, W3, W4))
#   Y = rnorm(n, Q0(A, W1, W2, W3, W4), 1)
#   data.frame(A, W1, W2, W3, W4, Y)
# }
# big = gendata.ols(1000000, g0, Q0)

big = gendata(1000000, g0, Q0)
mean((big$Y-min(big$Y))/(max(big$Y)-min(big$Y)))
mean(big$Y)

blips = with(big, Q0(1,W1,W2,W3,W4) - Q0(0,W1,W2,W3,W4))
Q0_1 = with(big, Q0(1,W1,W2,W3,W4))
Q0_0 = with(big, Q0(0,W1,W2,W3,W4))
g0_1 = with(big, g0(W1,W2,W3,W4))

hist(g0_1,200)
hist(Q0_1, 200)
hist(Q0_0, 200)
hist(blips, 200)
hist(big$Y, 200)

var0 = var(blips)
ate0 = mean(blips)
var0
ate0

# define custom logistic reg objective (log lik loss)
logregobj <- function(preds, dtrain) { 
  labels <- getinfo(dtrain, "label") 
  
  # interestingly, the model constraint by the logistic function not present here
  # if I don't transform and then try log-lik loss it is less stable due to the
  # xgboost starting point, perhaps, gives similar results to logistic
  
  # grad <- -labels/preds-(labels-1)/(1-preds)
  # hess <- labels/preds^2-(labels-1)/(1-preds)^2
  
  preds <- 1/(1+exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess)) }

# eval error can be any good error function to compare models
# here I did MSE of course
evalerror <- function(preds, dtrain) { 
  labels <- getinfo(dtrain, "label") 
  err <- sqrt(mean((preds-labels)^2))
  return(list(metric = "MSE", value = err)) }

# set up their matrix object, notice scaled continuous Y
# m=as.matrix(gendata.ols(1000, g0, Q0))

SL.xgbB = function (Y, X, newX, family, obsWeights, id, ntrees = 1000, 
                    max_depth = 2, shrinkage = 0.01, minobspernode = 10, params = list(), 
                    nthread = 1, verbose = 0, save_period = NULL, ...) 
{
  # get number of cols and treatment col logical
  p = ncol(X)
  txcol = 1:p==grep("A", colnames(X))
  
  # convert to matrices for xgboost
  X = as.matrix(X)
  newX = as.matrix(newX)

  # set the treatment row logical
  tx = X[,txcol]==1
  tx_newX = newX[,txcol]==1

  # set the parameters for xgboost
  param_logistic <- list(max_depth = max_depth, eta =  shrinkage, silent = 1,
                          objective = "binary:logistic", min_child_weight = minobspernode)
  
  # Fit barQ(0,W) on all W's with A = 0 '
  W0  = xgb.DMatrix(X[!tx,!txcol], label = Y[!tx])
  fit_W0 <- xgb.train(param_logistic, data=W0, nrounds = ntrees, maximize  =FALSE)
  
  # predict on data for A = 1
  W1 <- xgb.DMatrix(X[tx,!txcol], label = Y[tx])
  Q0k = predict(fit_W0, W1, outputmargin = TRUE)
  
  # on all of newW to be replaced for A = 1
  newW <- xgb.DMatrix(newX[,!txcol])
  Q0k_newW = predict(fit_W0, newW)

  # special blip fit on treated (A=1)
  W1 = xgb.DMatrix(X[tx,!txcol], label = Y[tx], base_margin = Q0k)
  fit_W1 <- xgb.train(param_logistic, data=W1, nrounds = ntrees, maximize  =FALSE)
  
  # create predictions, keeping Q0k on those with A = 0
  yhatB = Q0k_newW
  newW1 <- xgb.DMatrix(newX[tx_newX,!txcol], base_margin = qlogis(Q0k_newW[tx_newX]))
  yhatB[tx_newX] = plogis(predict(fit_W1, newW1,outputmargin = TRUE))
  
  pred = yhatB
  fit = list(object = fit_W1)
  class(fit) = c("SL.xgboost")
  out = list(pred = pred, fit = fit)
  return(out)
}

environment(SL.xgbB) <- asNamespace("SuperLearner")


sim.xgb = function(n, ate0, var0, g0, Q0)
{
  # n=10000
  data = gendata(n, g0, Q0)
  X = data
  X$Y = NULL
  X1 = X0 = X
  X1$A = 1
  X0$A = 0
  newX = rbind(X, X1, X0)
  Y = data$Y
  m=as.matrix(data)
  
  param_logistic <- list(max_depth = 2, eta = .01, silent = 1, objective = "binary:logistic")
  dtest <- xgb.DMatrix(m[,1:ncol(X)], label = m[,"Y"])
  bst <- xgb.train(param_logistic, data=dtest, nrounds = 1000, maximize  =FALSE,watchlist=list())
  yhat= predict(bst, dtest)
  
  m1 = m
  m1[,1] = 1
  dtest1 = xgb.DMatrix(m1[,1:ncol(X)], label = m1[,"Y"])
  yhat1 = predict(bst, dtest1)
  
  m0 = m
  m0[,1] = 0
  dtest0 = xgb.DMatrix(m0[,1:ncol(X)], label = m0[,"Y"])
  yhat0 = predict(bst, dtest0)
  
  resB = SL.xgbB(Y, X, newX, family, obsWeights, id, ntrees = 1000, 
                      max_depth = 2, shrinkage = 0.01, minobspernode = 10, params = list(), 
                      nthread = 1, verbose = 0, save_period = NULL) 

  yhatB1 = resB$pred[n + 1:n]
  yhatB0 = resB$pred[2*n + 1:n]
  yhatB = resB$pred[1:n]
  
  meanQAW = mean(with(big, Q0(A,W1,W2,W3,W4)))
  blip0 = with(data.frame(m[,1:ncol(X)]), Q0(1,W1,W2,W3,W4) - Q0(0,W1,W2,W3,W4))
  QAW = with(data.frame(m[,1:ncol(X)]), Q0(A,W1,W2,W3,W4))
  blipB = yhatB1 - yhatB0
  blip = yhat1 - yhat0
  
  param_bias = c(mean(blipB) - ate0, 
                 mean(blip) - ate0,
                 var(blipB) - var0,
                 var(blip) - var0,
                 mean(yhatB) - meanQAW, mean(yhat) - meanQAW)
  
  names(param_bias) = c("ateB", "ate", "varB", "var", "meanYB", "meanY")
  
  mse = c(mean((yhatB - QAW)^2), mean((yhat - QAW)^2)) 
  names(mse) = c("mseB", "mse")
  
  
  bmse = c(mean((blipB - blip0)^2), mean((blip - blip0)^2))
  names(bmse) = c("bmseB", "bmse")
  return(list(c(param_bias, mse, bmse), data.frame(blipB = blipB, blip = blip, blip0 = blip0)))
}


sim.xgb1 = function(n, ate0, var0, g0, Q0, V = 10)
{
  # n=10000
  data = gendata(n, g0, Q0)
  X = data
  X$Y = NULL
  X1 = X0 = X
  X1$A = 1
  X0$A = 0
  newX = rbind(X, X1, X0)
  Y = data$Y
  m=as.matrix(data)
  
  param_logistic <- list(max_depth = 2, eta = .01, silent = 1, objective = "binary:logistic")
  dtest <- xgb.DMatrix(m[,1:ncol(X)], label = m[,"Y"])
  bst <- xgb.train(param_logistic, data=dtest, nrounds = 1000, maximize  =FALSE,watchlist=list())
  yhat= predict(bst, dtest)
  
  m1 = m
  m1[,1] = 1
  dtest1 = xgb.DMatrix(m1[,1:ncol(X)], label = m1[,"Y"])
  yhat1 = predict(bst, dtest1)
  
  m0 = m
  m0[,1] = 0
  dtest0 = xgb.DMatrix(m0[,1:ncol(X)], label = m0[,"Y"])
  yhat0 = predict(bst, dtest0)
  
  folds = make_folds(n, V=V)
  cv_pred = lapply(folds, FUN = function(fold) {
    # fold = folds[[1]]
    if (V == 1) fold$training_set = fold$validation_set
    X = data[fold$training_set,]
    Y = X[,grep("Y", colnames(data))]
    X = as.data.frame(X[, -grep("Y", colnames(data))])
    txcol = grep("A", colnames(X))
    newX = data[fold$validation_set,-grep("Y", colnames(data))]
    X1 = X0 = newX
    X1[,txcol] = 1
    X0[,txcol] = 0
    n = nrow(X1)
    newX = as.data.frame(rbind(newX, X1, X0))
    resB = SL.xgbB(Y, X, newX, family, obsWeights, id, ntrees = 1000, 
                   max_depth = 2, shrinkage = 0.01, minobspernode = 10, params = list(), 
                   nthread = 1, verbose = 0, save_period = NULL) 
    yhatB1 = resB$pred[n + 1:n]
    yhatB0 = resB$pred[2*n + 1:n]
    yhatB = resB$pred[1:n]
    return(data.frame(yhatB1 = yhatB1, yhatB0 = yhatB0, yhatB = yhatB, 
                inds = fold$validation_set))
  })
  
  preds = do.call(rbind, cv_pred)
  preds = preds[order(preds$inds),]
  
  meanQAW = mean(with(big, Q0(A,W1,W2,W3,W4)))
  blip0 = with(data.frame(m[,1:ncol(X)]), Q0(1,W1,W2,W3,W4) - Q0(0,W1,W2,W3,W4))
  QAW = with(data.frame(m[,1:ncol(X)]), Q0(A,W1,W2,W3,W4))
  blipB = preds$yhatB1 - preds$yhatB0
  yhatB = preds$yhatB
  blip = yhat1 - yhat0
  
  param_bias = c(mean(blipB) - ate0, 
                 mean(blip) - ate0,
                 var(blipB) - var0,
                 var(blip) - var0,
                 mean(yhatB) - meanQAW, mean(yhat) - meanQAW)
  
  names(param_bias) = c("ateB", "ate", "varB", "var", "meanYB", "meanY")
  
  mse = c(mean((yhatB - QAW)^2), mean((yhat - QAW)^2)) 
  names(mse) = c("mseB", "mse")
  
  
  bmse = c(mean((blipB - blip0)^2), mean((blip - blip0)^2))
  names(bmse) = c("bmseB", "bmse")
  return(list(c(param_bias, mse, bmse), data.frame(blipB = blipB, blip = blip, blip0 = blip0)))
}

Q0=function(A,W1,W2,W3,W4) return(plogis(.2*(A+2*A*W4+3*W1+1*W2^2+.5*W3*W4+.25*W4)))
Q0=function(A,W1,W2,W3,W4) return(plogis(.2*(A+5*A*W4+.63*W1+1*W2^3+A*5*cos(W3)+.25*W4)))


g0 = g0_linear
Q0 = Q0_trig

big = gendata(1000000, g0, Q0)
blips = with(big, Q0(1,W1,W2,W3,W4) - Q0(0,W1,W2,W3,W4))

var0 = var(blips)
ate0 = mean(blips)
var0
ate0

cl = makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
n=1000
B=4

ALL =foreach(i=1:B,.packages=c("xgboost","Simulations"))%dopar%{sim.xgb1(n, ate0, var0, g0, Q0)}

res = lapply(ALL, FUN = function(x) x[[1]])
results = do.call(rbind, res)
colMeans(results)

hist(ALL[[1]][[2]][,1], 100)
hist(ALL[[1]][[2]][,2], 100)
hist(ALL[[1]][[2]][,3], 100)


library(xgboost)

## ----Customized loss function--------------------------------------------


logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  # base_margin = getinfo(dtrain, "base_margin")
  preds <- 1/(1 + exp(-gg-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}


logregobj1 <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  base_margin = getinfo(dtrain, "base_margin")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

logregobj2 <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  base_margin = getinfo(dtrain, "base_margin")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}


gg = rep(qlogis(.2), nrow(agaricus.train$data))
data(agaricus.train, package='xgboost')
train <- agaricus.train
dtrain_no <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtrain1 <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label,
                      base_margin = rep(qlogis(.2), nrow(train$data)))
dtrain2 <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label,
                      base_margin = rep(qlogis(.4), nrow(train$data)))



param <- list(max_depth = 2, eta = 0.01, silent = 1,objective = logregobj, maximize = FALSE)
param1 <- list(max_depth = 2, eta = 0.01, silent = 1,objective = logregobj1, maximize = FALSE)
# param <- list(max_depth = 2, eta = 0.01, silent = 1,objective = logregobj1, maximize = FALSE)
# param <- list(max_depth = 2, eta = 0.01, silent = 1,objective = logregobj2, maximize = FALSE)
paramLR <- list(max_depth = 2, eta = 0.01, silent = 1,objective = "binary:logistic", maximize = FALSE)


bst_handLR <- xgb.train(param1, dtrain_no, nrounds = 0, maximize = FALSE)
bst_LR <- xgb.train(paramLR, dtrain_no, nrounds = 0, maximize = FALSE)
bst_handLR1 <- xgb.train(param1, dtrain1, nrounds = 500, maximize = FALSE)
bst_LR1 <- xgb.train(paramLR, dtrain1, nrounds = 500, maximize = FALSE)
bst_handLR2 <- xgb.train(param1, dtrain2, nrounds = 1, maximize = FALSE)
bst_LR2 <- xgb.train(paramLR, dtrain2, nrounds = 1, maximize = FALSE)


predict(bst_handLR, dtrain1)[1:10]
predict(bst_LR, dtrain1, outputmargin = TRUE)[1:10]

predict(bst_handLR1, dtrain1)[1:10]
predict(bst_LR1, dtrain1, outputmargin = TRUE)[1:10]
predict(bst_handLR1, dtrain2)[1:10]
predict(bst_LR1, dtrain2, outputmargin = TRUE)[1:10]
qlogis(.2)

predict(bst_handLR2, dtrain2)[1:10]
predict(bst_LR2, dtrain2, outputmargin = TRUE)[1:10]

plogis(predict(bst_handLR1, dtrain1)[1:10])
plogis(predict(bst_LR1, dtrain1, outputmargin = TRUE)[1:10])

plogis(predict(bst_handLR2, dtrain1)[1:10])
plogis(predict(bst_handLR2, dtrain1, outputmargin = TRUE)[1:10])

