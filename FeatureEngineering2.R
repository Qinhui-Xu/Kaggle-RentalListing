library(lubridate)
library(dplyr)
library(jsonlite)
library(caret)
library(purrr)
library(xgboost)
library(MLmetrics)
library(tidytext)
library(reshape2)
library(syuzhet)
library(DT)
library(data.table)
library(h2o)
#####Sentiment Analysis based on Description
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

data <- fromJSON("~/Desktop/kaggle/RentalListing/train.json")
vars <- setdiff(names(data), c("photos", "features"))
train_df <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)
train_df$num_features <- lengths(train_df$features)
train_df$id<-seq(1:length(train_df$building_id)) #numerical ids!
sentiment <- get_nrc_sentiment(train_df$description)
datatable(head(sentiment))
sentiment$id<-seq(1:nrow(sentiment))
sentiment <- subset(sentiment, select = c(negative,positive,id))
sent_df<-merge(train_df,sentiment, by.x="id", by.y="id", all.x=T, all.y=T)

#####Deal with Feature Variable
train <- fromJSON("/Users/apple1/desktop/kaggle/RentalListing/train.json")
frq_features = table(unlist(train$features))
top_features = names(frq_features[frq_features>8888])
extraction_feature = t(sapply(sent_df$features,
                              function(x) {
                                as.numeric(top_features %in% x)
                              }))
colnames(extraction_feature) <- top_features
sent_df = cbind(sent_df, extraction_feature)
sent_df <- subset(sent_df, select = -c(features) )
head(sent_df)
#library(jsonlite)
#x <- toJSON(sent_df)
#write(x, "train_feature_sent.json")

#####Log the price
sent_df$log_price = log(sent_df$price)
head(sent_df)

#####Rate by Level for Manager_id
row_by_manager_id = aggregate(rep(1, nrow(sent_df)),by=list(sent_df$manager_id), sum)
names(row_by_manager_id) <- c("manager_id","count")
row_by_manager_id$freq = row_by_manager_id$count/49352
row_by_manager_id$manager_odds = log(row_by_manager_id$freq/(1-row_by_manager_id$freq))
sent_df = merge(sent_df,row_by_manager_id,by="manager_id")
sent_df <- subset(sent_df, select = -c(manager_id,count,freq))

#####Rate by Level for Building_id
row_by_building_id = aggregate(rep(1, nrow(sent_df)),by=list(sent_df$building_id), sum)
names(row_by_building_id) <- c("building_id","count")
row_by_building_id$freq = row_by_building_id$count/49352
row_by_building_id$building_odds = log(row_by_building_id$freq/(1-row_by_building_id$freq))
sent_df = merge(sent_df,row_by_building_id,by="building_id")
sent_df <- subset(sent_df, select = -c(building_id,count,freq))

#####Rate by Level for Listing_id
row_by_listing_id = aggregate(rep(1, nrow(sent_df)),by=list(sent_df$listing_id), sum)
names(row_by_listing_id) <- c("listing_id","count")
row_by_listing_id$freq = row_by_listing_id$count/49352
row_by_listing_id$listing_odds = log(row_by_listing_id$freq/(1-row_by_listing_id$freq))
sent_df = merge(sent_df,row_by_listing_id,by="listing_id")
sent_df <- subset(sent_df, select = -c(listing_id,count,freq))

#####h2o building models
h2o.init(nthreads = -1, max_mem_size="6g")
new_train <- data.table(bathrooms=unlist(sent_df$bathrooms)
                 ,bedrooms=unlist(sent_df$bedrooms)
                 #,building_id=as.factor(unlist(sent_df$building_id))
                 ,building_odds=as.factor(unlist(sent_df$building_odds))
                 ,created=as.POSIXct(unlist(sent_df$created))
                 # ,description=unlist(t1$description) # parse errors
                 # ,display_address=unlist(t1$display_address) # parse errors
                 ,latitude=unlist(sent_df$latitude)
                 ,longitude=unlist(sent_df$longitude)
                 #,listing_id=unlist(sent_df$listing_id)
                 ,listing_odds=unlist(sent_df$listing_odds)
                 #,manager_id=as.factor(unlist(sent_df$manager_id))
                 ,manager_odds=as.factor(unlist(sent_df$manager_odds))
                 ,photo_num=as.numeric(lengths(sent_df$photos))
                 ,price=unlist(sent_df$price)
                 ,log_price=unlist(sent_df$log_price)
                 ,interest_level=as.factor(unlist(sent_df$interest_level))
                 ,negative=unlist(sent_df$negative)
                 ,positive=unlist(sent_df$positive)
                 ,feature_num=as.numeric(sent_df$num_features)
                 ,CatsAllowed=unlist(sent_df$`Cats Allowed`)
                 ,Dishwasher=unlist(sent_df$Dishwasher)
                 ,DogsAllowed=unlist(sent_df$`Dogs Allowed`)
                 ,Doorman=unlist(sent_df$Doorman)
                 ,Elevator=unlist(sent_df$Elevator)
                 ,Fitness=unlist(sent_df$`Fitness Center`)
                 ,Hardwood=unlist(sent_df$`Hardwood Floors`)
                 ,Laundry=unlist(sent_df$`Laundry in Building`)
                 ,NoFee=unlist(sent_df$`No Fee`)
                 ,PreWar=unlist(sent_df$`Pre-War`)
                 # ,street_adress=unlist(t1$street_address) # parse errors
)
new_train[,":="(yday=yday(created)
          #,month=month(created) #errors
          ,mday=mday(created)
          ,wday=wday(created))]
          #,hour=hour(created))]

new_train <-subset(new_train,select=-(created))
train <- as.h2o(new_train, destination_frame = "train.hex")
vars <- setdiff(colnames(train), "interest_level")
split = h2o.splitFrame(train, ratios = c(0.7))
traindata <- split[[1]]
validationdata <- split[[2]]
gbm1 <- h2o.gbm(x = vars
                ,y = "interest_level"
                ,training_frame = traindata
                ,validation_frame = validationdata
                ,distribution = "multinomial"
                ,model_id = "gbm1"
                #,nfolds = 5
                ,ntrees = 800
                ,learn_rate = 0.05
                ,max_depth = 7
                ,min_rows = 20
                ,sample_rate = 0.7
                ,col_sample_rate = 0.7
                #   ,stopping_rounds = 5
                #   ,stopping_metric = "logloss"
                #   ,stopping_tolerance = 0
                ,seed=321
)


####XGBoost Building Model
seed = 1985
set.seed
new_train_num <- subset(new_train, select = -c(Fitness,DogsAllowed,PreWar,CatsAllowed))
new_train_num$bathrooms=as.numeric(new_train_num$bathrooms)
new_train_num$bedrooms=as.numeric(new_train_num$bedrooms)
new_train_num$listing_odds=as.numeric(new_train_num$listing_odds)
new_train_num$manager_odds=as.numeric(new_train_num$manager_odds)
new_train_num$building_odds=as.numeric(new_train_num$building_odds)
new_train_num$latitude=as.numeric(new_train_num$latitude)
new_train_num$longitude=as.numeric(new_train_num$longitude)
new_train_num$photo_num=as.numeric(new_train_num$photo_num)
new_train_num$price=as.numeric(new_train_num$price)
new_train_num$log_price=as.numeric(new_train_num$log_price)
new_train_num$interest_level=as.numeric(new_train_num$interest_level)
new_train_num$negative=as.numeric(new_train_num$negative)
new_train_num$positive=as.numeric(new_train_num$positive)
new_train_num$feature_num=as.numeric(new_train_num$feature_num)
#new_train_num$CatsAllowed=as.numeric(new_train_num$CatsAllowed)
new_train_num$Dishwasher=as.numeric(new_train_num$Dishwasher)
#new_train_num$DogsAllowed=as.numeric(new_train_num$DogsAllowed)
new_train_num$Doorman=as.numeric(new_train_num$Doorman)
new_train_num$Elevator=as.numeric(new_train_num$Elevator)
#new_train_num$Fitness=as.numeric(new_train_num$Fitness)
new_train_num$Hardwood=as.numeric(new_train_num$Hardwood)
new_train_num$Laundry=as.numeric(new_train_num$Laundry)
new_train_num$NoFee=as.numeric(new_train_num$NoFee)
new_train_num$yday=as.numeric(new_train_num$yday)
new_train_num$mday=as.numeric(new_train_num$mday)
new_train_num$wday=as.numeric(new_train_num$wday)
names(new_train_num)
#new_train_num$PreWar=as.numeric(new_train_num$PreWar)
##################
#Convert labels to integers
new_train_num$interest_level<-as.integer(factor(new_train_num$interest_level))
y <- new_train_num$interest_level
y = y - 1
new_train_num$interest_level = NULL

#Parameters for XGB
xgb_params = list(
  colsample_bytree= 0.7,
  subsample = 0.7,
  eta = 0.1,
  objective= 'multi:softprob',
  max_depth= 4,
  min_child_weight= 1,
  eval_metric= "mlogloss",
  num_class = 3,
  seed = seed
)

#convert xgbmatrix
dtrain <- xgb.DMatrix(data.matrix(new_train_num))
#create folds
kfolds<- 10
folds<-createFolds(y, k = kfolds, list = TRUE, returnTrain = FALSE)
fold <- as.numeric(unlist(folds[1]))
x_train<-new_train_num[-fold,] #Train set
x_val<-new_train_num[fold,] #Out of fold validation set
y_train<-y[-fold]
y_val<-y[fold]
#convert to xgbmatrix
dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dval = xgb.DMatrix(as.matrix(x_val), label=y_val)
#perform training
gbdt = xgb.train(params = xgb_params,
                 data = dtrain,
                 nrounds =888,
                 watchlist = list(train = dtrain, val=dval),
                 print_every_n = 25,
                 early_stopping_rounds=50)

