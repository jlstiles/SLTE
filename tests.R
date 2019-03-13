library(Simulations)
library(SLblip)
library(origami)

glm.mainint = function (Y, X, newX, family, obsWeights, model = TRUE, ...)
{
  if (is.matrix(X)) {
    X = as.data.frame(X)
  }
  mainform = paste0(paste(colnames(X)[2:ncol(X)],"",collapse="+"))
  form = formula(paste0("Y ~", paste0(colnames(X)[1],"*(",mainform,")")))
  
  fit.glm <- glm(form, data = X, family = family, weights = obsWeights,
                 model = model)
  if (is.matrix(newX)) {
    newX = as.data.frame(newX)
  }
  pred <- predict(fit.glm, newdata = newX, type = "response")
  fit <- list(object = fit.glm)
  class(fit) <- "SL.glm"
  out <- list(pred = pred, fit = fit)
  return(out)
}
environment(glm.mainint) <- asNamespace("SuperLearner")

Q0=function(A,W1,W2,W3,W4) return(plogis(.2*(A+5*A*W4+.63*W1+1*W2^3+A*5*cos(W3)+.25*W4)))
g0=function(W1,W2,W3,W4) {plogis(-.28*W1+1*W2+.08*W3-.12*W4-1)}

big = gendata(1000000, g0, Q0)
blips = with(big, Q0(1,W1,W2,W3,W4) - Q0(0,W1,W2,W3,W4))

var0 = var(blips)
ate0 = mean(blips)
var0
ate0

n=1000
data=gendata(n, g0, Q0)
X = data
Y = data$Y
X$Y = NULL
X1 = X0 = X
X1$A = 1
X0$A = 0
newX = rbind(X,X1,X0)

test = SuperLearner(Y, X, newX = newX, SL.library = c("SL.xgbB", "SL.blipLR","glm.mainint"), 
                    family = binomial(), method = "method.NNloglik")


test$coef 
test$cvRisk
test$library.predict[1:10,1]
test$Z[1:10,1]

mean(test$library.predict[1:n+n,1]-test$library.predict[1:n+2*n,1])
var(test$library.predict[1:n+n,1]-test$library.predict[1:n+2*n,1])
mean(test$library.predict[1:n+n,2]-test$library.predict[1:n+2*n,2])
var(test$library.predict[1:n+n,2]-test$library.predict[1:n+2*n,2])
mean(test$library.predict[1:n+n,3]-test$library.predict[1:n+2*n,3])
var(test$library.predict[1:n+n,3]-test$library.predict[1:n+2*n,3])
