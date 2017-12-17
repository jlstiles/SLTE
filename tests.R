library(Simulations)
library(xgboost)
source("~/Dropbox/Jonathan/Simulations/WrappersVblip1.R")

Q0=function(A,W1,W2,W3,W4) return(plogis(.2*(A+2*A*W4+3*W1+1*W2^2+.5*W3*W4+.25*W4)))
Q0=function(A,W1,W2,W3,W4) return(plogis(.2*(A+5*A*W4+.63*W1+1*W2^3+A*5*cos(W3)+.25*W4)))

# Q0=function(A,W1,W2,W3,W4) return(A+2*A*W4+3*W1+1*W2^2+.5*W3*W4+.25*W4)
# Q0=function(A,W1,W2,W3,W4) return(A+2*A*W4+.63*W1+1*W2^3+A*.5*cos(W3)+.25*W4)

g0=function(W1,W2,W3,W4) {plogis(-.28*W1+1*W2+.08*W3-.12*W4-1)}

g0 = g0_linear
Q0 = Q0_trig1


big = gendata(1000000, g0, Q0)
blips = with(big, Q0(1,W1,W2,W3,W4) - Q0(0,W1,W2,W3,W4))

var0 = var(blips)
ate0 = mean(blips)
var0
ate0

n=1000
folds = make_folds(n, V=10)
folds[[1]]
data=as.matrix(gendata(n, g0, Q0))
X = data[folds[[1]]$training_set,]
Y = X[,grep("Y", colnames(data))]
X = as.data.frame(X[, -grep("Y", colnames(data))])
txcol = grep("A", colnames(X))
newX = data[folds[[1]]$validation_set,-grep("Y", colnames(data))]
X1 = X0 = newX
X1[,txcol] = 1
X0[,txcol] = 0
newX = as.data.frame(rbind(newX, X1, X0))

test_SL.blip = SL.blip(Y, X, newX, family = 'binomial', obsWeights = NULL, model = TRUE)
test_SL.xgbB = SL.xgbB(Y, X, newX, family, obsWeights, id, ntrees = 1000, 
                       max_depth = 2, shrinkage = 0.01, minobspernode = 10, params = list(), 
                       nthread = 1, verbose = 0, save_period = NULL)

test_SL.blip$pred
test_SL.xgbB$pred

n=1000
data=gendata(n, g0, Q0)
X = data
Y = data$Y
X$Y = NULL
X1 = X0 = X
X1$A = 1
X0$A = 0
newX = rbind(X,X1,X0)

test = SuperLearner(Y, X, newX = newX, SL.library = c("SL.xgbB", "glm.mainint"), 
                    family = binomial(), method = "method.NNloglik")


test$coef 
test$cvRisk
test$library.predict[1:10,1]
test$Z[1:10,1]

mean(test$library.predict[1:n+n,1]-test$library.predict[1:n+2*n,1])
var(test$library.predict[1:n+n,1]-test$library.predict[1:n+2*n,1])
mean(test$library.predict[1:n+n,2]-test$library.predict[1:n+2*n,2])
var(test$library.predict[1:n+n,2]-test$library.predict[1:n+2*n,2])

