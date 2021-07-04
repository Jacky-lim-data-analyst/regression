# White wine quality prediction
# Reference: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

data=read.csv('winequality-white.csv',sep = ";")
str(data)
# statistical summary of each attribute
summary(data)

# distribution of quality of wine (response variable)
tabulate(as.factor(data$quality))
# bar graph to visualize
library(ggplot2)
ggplot(data=data,aes(x=quality)) +
  geom_bar(fill="steelblue") + theme_classic() +
  ggtitle('distribution of response variable (quality of white wine)')

# stratified sampling
library(rsample)
set.seed(2)
split=initial_split(data,prop=0.8,strata = "quality")
train=training(split)
test=testing(split)

ggplot(data=train,aes(x=quality)) +
  geom_bar(fill="steelblue") + theme_classic() +
  ggtitle('distribution of response variable in training dataset')

# Correlation matrix
vars=setdiff(colnames(train),'quality')
corr=cor(train[,vars])
library(reshape2)
corr[upper.tri(corr)]=NA
melted_cormat=melt(corr,na.rm = TRUE)
# heatmap
ggplot(data=melted_cormat,aes(Var1,Var2,fill=value)) +
  geom_tile(color='white') +
  scale_fill_gradient2(low = "blue", high="red",mid="white",midpoint = 0,
                       limit=c(-1,1),space = "Lab",name="Correlation") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45,vjust = 1,size = 12,hjust = 1)) +
  coord_fixed() +
  geom_text(aes(Var1,Var2,label=round(value,2)),color="black",size=3) +
  xlab("") + ylab("")

# linear regression and regularization ------------------------------------

# Train linear regression model
model_lm=lm(quality~.,data = train)
summary(model_lm)

# Check the variance inflation factor (multicollinearity)
library(knitr)
library(car)
kable(data.frame(vif=vif(model_lm)))

# Drop variable density
drops='density'
model_lm=lm(quality~.,data=train[,!(colnames(train) %in% drops)])
summary(model_lm)
kable(data.frame(vif=vif(model_lm)))

# Train ridge regression model
# First standardize the predictors
X_stand=scale(train[,vars],center = FALSE, scale = TRUE)
train_stand=data.frame(X_stand,quality=train$quality)
X1_stand=scale(test[,vars],center = FALSE, scale = attributes(X_stand)$'scaled:scale')
test_stand=data.frame(X1_stand,quality=test$quality)

# Train ridge regression model
library(glmnet)
library(glmnetUtils)
model_ridge=cv.glmnet(quality~.,data=train_stand,alpha = 0)
# visualize coefficients
coef_ridge=coef(model_ridge)
df_coef_ridge=data.frame(coef_name=rownames(coef_ridge)[-1], coef=coef_ridge[-1,1])
ggplot(data = df_coef_ridge,aes(x=coef_name,y=coef)) +
  geom_bar(stat = 'identity', width=0.5) +
  coord_flip()

# visualized cross-validation error with repect to different lambda
roundoff=function(x) sprintf("%.3f",x)
df_cvm_ridge=data.frame(lambda=model_ridge$lambda,cvm=model_ridge$cvm,
                        cvm_up=model_ridge$cvup,cvm_down=model_ridge$cvlo)
ggplot(data=df_cvm_ridge,aes(x=lambda,y=cvm,ymin=cvm_down,ymax=cvm_up)) +
  geom_line() + geom_point() + geom_errorbar() + scale_x_continuous(trans = "log2",labels = roundoff) +
  geom_vline(xintercept = model_ridge$lambda.min, linetype=2, color="red") +
  geom_vline(xintercept = model_ridge$lambda.1se, linetype=3, color="blue") +
  ggtitle('ridge regression')

# Lasso

model_lasso=cv.glmnet(quality~.,data=train_stand,alpha = 1)
# visualize coefficients
coef_lasso=coef(model_lasso)
df_coef_lasso=data.frame(coef_name=rownames(coef_lasso)[-1], coef=coef_lasso[-1,1])
ggplot(data = df_coef_lasso,aes(x=coef_name,y=coef)) +
  geom_bar(stat = 'identity', width=0.5) +
  coord_flip()

# visualized cross-validation error with repect to different lambda
# plus the number of nonzero coefficients
df_cvm_lasso=data.frame(lambda=model_lasso$lambda,cvm=model_lasso$cvm,
                        cvm_up=model_lasso$cvup,cvm_down=model_lasso$cvlo)
p1=ggplot(data=df_cvm_lasso,aes(x=lambda,y=cvm,ymin=cvm_down,ymax=cvm_up)) +
  geom_line() + geom_point() + geom_errorbar() + scale_x_continuous(trans = "log2",labels = roundoff) +
  geom_vline(xintercept = model_lasso$lambda.min, linetype=2, color="red") +
  geom_vline(xintercept = model_lasso$lambda.1se, linetype=3, color="blue") +
  ggtitle('Lasso')

df_nzero_lasso=data.frame(lambda=model_lasso$lambda,nzero=model_lasso$nzero)
p2=ggplot(data=df_nzero_lasso,aes(x=lambda,y=nzero)) +
  geom_line(color="steelblue") + scale_x_continuous(trans = "log2",labels = roundoff) + 
  ylab('Number of non-zero coefficients')

library(gridExtra)
grid.arrange(p1,p2,ncol=2)

# elastic net
elastic_net=cva.glmnet(quality~.,train_stand)

# function to get cvm of the model$lambda.1se in each alpha
get_cvm=function(model) {
  index <- match(model$lambda.1se, model$lambda)
  model$cvm[index]
}
# data frame that contains alpha and its corresponding cross-validation error
enet_performance=data.frame(alpha=elastic_net$alpha)
models=elastic_net$modlist
enet_performance$cvm=vapply(models,get_cvm,numeric(1))
# get the best alpha and train the model
best_alpha=enet_performance[which.min(enet_performance$cvm),'alpha']
model_enet=cv.glmnet(quality~.,train_stand,alpha = best_alpha)

# visualize coefficients
coef_enet=coef(model_enet)
df_coef_enet=data.frame(coef_name=rownames(coef_enet)[-1], coef=coef_enet[-1,1])
ggplot(data = df_coef_enet,aes(x=coef_name,y=coef)) +
  geom_bar(stat = 'identity', width=0.5) +
  coord_flip()

# 
df_cvm_enet=data.frame(lambda=model_enet$lambda,cvm=model_enet$cvm,
                        cvm_up=model_enet$cvup,cvm_down=model_enet$cvlo)
ggplot(data=df_cvm_enet,aes(x=lambda,y=cvm,ymin=cvm_down,ymax=cvm_up)) +
  geom_line() + geom_point() + geom_errorbar() + scale_x_continuous(trans = "log2",labels = roundoff) +
  geom_vline(xintercept = model_enet$lambda.min, linetype=2, color="red") +
  geom_vline(xintercept = model_enet$lambda.1se, linetype=3, color="blue") +
  ggtitle('Elastic net')

# grouped bar plots
df=rbind(df_coef_ridge,df_coef_lasso,df_coef_enet)
L=nrow(df_coef_ridge)
id_vec=c(rep('ridge',L),rep('lasso',L),rep('elastic_net',L))
df=cbind(df,types=id_vec)

ggplot(data=df) +
  geom_bar(aes(x = coef_name, y = coef, fill = types),
           stat = "identity", position = "dodge") +
  scale_y_continuous("Coefficient estimates") +
  scale_x_discrete("Predictors") +
  # remove grey theme
  theme_classic(base_size = 15) +
  # rotate x-axis text and remove superfluous axis elements
  theme(axis.text.x = element_text(angle = 90,hjust = 1, vjust=0),
        axis.title.x=element_text(size=13),
        axis.title.y=element_text(size=15),
        axis.line = element_blank()) 

# performance of linear regression and regularized regression ---------------------------------
# function to evaluate models' performance
performance=function(truth,pred){
  # Root mean squre error
  sse=mean((truth-pred)^2)
  rmse=sqrt(sse)
  # mean absolute error
  mae=sum(abs(truth-pred))/length(truth)
  # coefficient of determination R2
  denom=sum((truth-mean(truth))^2)
  numer=sum((truth-pred)^2)
  R_squared=1-numer/denom
  c(rmse,mae,R_squared)
}

perf_lm=performance(test$quality,predict(model_lm,newdata = test))
perf_ridge=performance(test_stand$quality,predict(model_ridge,newdata = test_stand))
perf_lasso=performance(test_stand$quality,predict(model_lasso,newdata = test_stand))
perf_enet=performance(test_stand$quality,predict(model_enet,newdata = test_stand))
df_perf=data.frame(rbind(perf_lm,perf_ridge,perf_lasso,perf_enet))
colnames(df_perf)=c("RMSE","MAE","R2")
rownames(df_perf)=c("linear regression","ridge regression","lasso","elastic net")
library(knitr)
kable(df_perf)

# repeated k-fold cv ------------------------------------------------------

set.seed(321)
k=5
library(dplyr)
data=mutate(data,folds=sample(1:k,size = nrow(data),replace=TRUE))

# function
kfold_cv=function(kfold,data,method="lm"){
  train=subset(data,folds!=kfold)
  validate=subset(data,folds==kfold)
  train_1=scale(train[,vars],center = FALSE, scale=TRUE)
  train=data.frame(train_1,quality=train$quality)
  validate_1=scale(validate[,vars],center = FALSE, scale = attributes(train_1)$'scaled:scale')
  validate=data.frame(validate_1,quality=validate$quality)
  if (method=="lm") {
    model=lm(quality~.,data = train[,!(colnames(train) %in% drops)])
  }
  if (method=="ridge"){
    model=cv.glmnet(quality~.,train,alpha=0)
  }
  if (method=="lasso"){
    model=cv.glmnet(quality~.,train,alpha=1)
  }
  if (method=="enet"){
    elastic_net=cva.glmnet(quality~.,train)
    # data frame that contains alpha and its corresponding cross-validation error
    enet_performance=data.frame(alpha=elastic_net$alpha)
    models=elastic_net$modlist
    enet_performance$cvm=vapply(models,get_cvm,numeric(1))
    # get the best alpha and train the model
    best_alpha=enet_performance[which.min(enet_performance$cvm),'alpha']
    model=cv.glmnet(quality~.,train,alpha = best_alpha)
  }
  if (!(method %in% c("lm","ridge","lasso","enet"))) {
    stop('Methods not found')
  }
  pred=predict(model,newdata=validate)
  performance(validate$quality,pred)
}

n_repeat=20
mat_perf=matrix(,nrow=n_repeat,ncol=3)
a=numeric(0)

for (i in 1:n_repeat){
  set.seed(i+100)
  cv.data=mutate(data,folds=sample(1:k,size = nrow(data),replace=TRUE))
  cv.perf=sapply(c(1:k),FUN = kfold_cv,data=cv.data,method="enet")
  a=cbind(a,cv.perf)
  #perf=apply(cv.perf,1,mean)
  #mat_perf[i,]=perf
}
rownames(a)=c("RMSE","MAE","R2")
(mean_perf=apply(a,1,mean))
# t-test confidence interval
tcrit=qt(0.025,df=ncol(a)-1,lower.tail = FALSE)
sd_perf=apply(a, 1, sd)
(ci=tcrit*sd_perf/sqrt(ncol(a)))
