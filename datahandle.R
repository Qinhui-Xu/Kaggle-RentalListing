#library(lubridate)
library(dplyr)
library(jsonlite)
library(caret)
library(purrr)
library(xgboost)
library(MLmetrics)
library(tidytext)
library(reshape2)
library(data.table)
seed = 1985
set.seed(seed)

train <- fromJSON("/Users/apple1/desktop/kaggle/RentalListing/train.json")
test <- fromJSON("/Users/apple1/desktop/kaggle/RentalListing/test.json")

t1 <- train
t2 <- data.table(bathrooms=unlist(t1$bathrooms)
                 ,bedrooms=unlist(t1$bedrooms)
                 ,building_id=as.factor(unlist(t1$building_id))
                 ,created=as.POSIXct(unlist(t1$created))
                 ,n_photos = as.numeric(sapply(t1$photos, length))
                 ,n_description = as.numeric(sapply(t1$description, nchar))
                 ,n_features = as.numeric(sapply(t1$features, length))
                 #,description=unlist(t1$description) # parse errors
                # ,display_address=unlist(t1$display_address) # parse errors
                 ,latitude=unlist(t1$latitude)
                 ,longitude=unlist(t1$longitude)
                 ,listing_id=unlist(t1$listing_id)
                 ,manager_id=as.factor(unlist(t1$manager_id))
                 ,price=unlist(t1$price)
                 ,interest_level=as.factor(unlist(t1$interest_level))
                 #,street_adress=unlist(t1$street_address) # parse errors
)

t2[,":="(yday=yday(created)
         ,month=month(created)
         ,mday=mday(created)
         ,wday=wday(created)
         ,hour=hour(created))]

# expand features
frq_features = table(unlist(t1$features))
top_features = names(frq_features[frq_features>1000])  ## can't set too small due to the limit of run time

t2_exp_feat = t(sapply(t1$features,
                       function(x) {
                         as.numeric(top_features %in% x)
                       }))
t2 = cbind(t2, t2_exp_feat)
write.csv(t2, file = "train_datahandle.csv",row.names=FALSE, na="")

s1 <- test
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
s2 <- data.table(bathrooms=unlist(s1$bathrooms)
                 ,bedrooms=unlist(s1$bedrooms)
                 ,building_id=as.factor(unlist(s1$building_id))
                 ,created=as.factor(unlist(s1$created))
                 ,n_photos = as.numeric(sapply(s1$photos, length))
                 ,n_description = as.numeric(sapply(s1$description, nchar))
                 ,n_features = as.numeric(sapply(s1$features, length))
                  ,description=unlist(s1$description) # parse errors
                  ,display_address=unlist(s1$display_address) # parse errors
                 ,latitude=unlist(s1$latitude)
                 ,longitude=unlist(s1$longitude)
                 ,listing_id=unlist(s1$listing_id)
                 ,manager_id=as.factor(unlist(s1$manager_id))
                 ,price=unlist(s1$price)
                  ,street_adress=unlist(s1$street_address) # parse errors
)
s2[,":="(yday=yday(created)
         ,month=month(created)
         ,mday=mday(created)
         ,wday=wday(created)
         ,hour=hour(created))]

s2_exp_feat = t(sapply(s1$features,
                       function(x) {
                         as.numeric(top_features %in% x)
                       }))
s2 = cbind(s2, s2_exp_feat)

write.csv(s2, file = "test.csv",row.names=FALSE, na="")