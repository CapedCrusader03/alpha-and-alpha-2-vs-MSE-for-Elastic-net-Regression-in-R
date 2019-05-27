library(glmnet)
set.seed(123)

#create random 5000 predictors for a model and 1000 random observations

n <- 1000
p <- 5000
real_p <- 15

# The x matrix contains 1000 rows, 5000 columns, and the values in the matrix come from a std. normal
#distribution with mu = 0, sigma =1

x <- matrix(rnorm(n*p), nrow = n, ncol = p)

#Now we create a vector of values, called y, that we will try to predict with data in x.
#The apply function will return a vector of 1000 values that are the sums of the first 15 columns
#in x, since x has 1000 rows

y <- apply(x[,1:real_p], 1, sum) + rnorm(n)#the rnorm() is used to add random noise 

#training and testing dataset


train_rows <- sample(1:n, .66*n)
x.train <- x[train_rows, ]
x.test <- x[-train_rows, ]

y.train <- y[train_rows]
y.test <- y[-train_rows]

# RIDGE REGRESSION

alpha0.fit <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 0, family = "gaussian")

alpha0.predicted <- 
  + predict(alpha0.fit, s = alpha0.fit$lambda.1se, newx = x.test )

ridge_mse <- mean((y.test - alpha0.predicted)^2)

# LASSO REGRESSION

alpha1.fit <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 1, family = "gaussian")

alpha1.predicted <- predict(alpha1.fit, s= alpha1.fit$lambda.1se, newx =  x.test)

lasso_mse <- mean((y.test - alpha1.predicted)^2)

# ELASTIC-NET REGRESSION

alpha0.5.fit <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 0.5, family = "gaussian")

alpha0.5.predicted <- predict(alpha0.5.fit, s = alpha0.5.fit$lambda.1se, newx = x.test)

elastic_mse <- mean((y.test - alpha0.5.predicted)^2)

#Let's try the result for different alphas

#training part
list.of.fits <- list()
for(i in 0:10){
  fit.name <- paste0("alpha", i/10)
  
  list.of.fits[[fit.name]] <-
    cv.glmnet(x.train, y.train, type.measure = "mse", alpha = i/10,family = "gaussian")
}

#prediction part
results = data.frame()
ms2 = numeric()
al <-  numeric()
al2 = numeric()
for(i in 0:10){
  fit.name <- paste0("alpha", i/10)
  al[i] <- c(i/10)
  al2[i] <- c((i*i)/(10*10))
  predicted <- predict(list.of.fits[[fit.name]], s = list.of.fits[[fit.name]]$lambda.1se, newx = x.test)
  
  mse <- mean((y.test - predicted)^2)
  
  temp <- data.frame(alpha = i/10, mse = mse, fit.name = fit.name)
  ms2[i] <- mse
  results <- rbind(results,temp)
}
results
plot(al,ms2, type = 'l', col="red", xlab = "alpha", ylab = "error", main = "MSE vs Alpha")
plot(al2,ms2, type = 'l', col="red", xlab = "alpha^2", ylab = "error", main = "MSE vs Alpha^2")
