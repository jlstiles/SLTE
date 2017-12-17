#' @export
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


#' @export
SL.blip = function (Y, X, newX, family, obsWeights, model = TRUE, ...) 
{
  if (is.matrix(X)) {
    X = as.data.frame(X)
  }
  
  Wnames = names(X)[2:ncol(X)]
  Q0kform =  paste(Wnames, "", collapse = "+")
  
  X0 = X1 = X
  X0$A = 0
  X1$A = 1
  Y0 = Y[X$A == 0]
  # fit and predict Q0k
  Q0kfit = glm(formula(paste0("Y0~", Q0kform)), family = binomial, data = X[X$A == 0,Wnames])
  Q0k = predict(Q0kfit, newdata = X0[,2:ncol(X)], type = 'response')
  
  # keeping Bk regression functional form the same as for Q0k here
  Bkform = Q0kform
  
  # fit the model
  X$Q0k = Q0k
  Qkfit = glm(formula(paste0("Y~ A + A:(", Bkform, ")", "+ offset(qlogis(Q0k)) - 1")), 
              family = binomial, data = X)
  
  #get the predictions for the new method and blip, Bk
  newdata = newX
  newdata$Q0k = predict(Q0kfit, newdata = newX, type = 'response')
  pred = predict(Qkfit, newdata = newdata, type = 'response')
  # pred[1:1000+2000] - Q0k
  # pred = plogis(qlogis(predict(Qkfit, newdata = newdata, type = 'response')) - (1 - newX$A)*adj)
  
  fit <- list(object = Qkfit)
  class(fit) <- "SL.glm"
  out <- list(pred = pred, fit = fit)
  return(out)
}

environment(SL.blip) <- asNamespace("SuperLearner")
