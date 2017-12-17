# for xgboost, use the setinfo after the Dmatrix is made
setinfo(xgtrain, "base_margin", log(d$exposure))