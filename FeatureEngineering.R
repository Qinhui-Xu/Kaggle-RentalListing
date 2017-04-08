library(data.table)
df_train <- read.csv(file="~/Desktop/kaggle/RentalListing/Kaggle-RentalListing/df_train.csv")
data.table(head(df_train))
names(df_train)

####### Sentiment Analysis based on description
library(syuzhet)
library(DT)
df_train$id<-seq(1:length(df_train$building_id))
df_train$description <- as.character(df_train$description)
sentiment <- get_nrc_sentiment(df_train$description)
sentiment$id<-seq(1:nrow(sentiment))
df_train_sent<-merge(df_train,sentiment, by.x="id", by.y="id", all.x=T, all.y=T)
data.table(head(df_train_sent))

write.csv(df_train_sent, file = "train_sent.csv",row.names=FALSE)

###### Deal with feature variable
###### Don't know how to seperate them into one by one in the features area
train <- fromJSON("/Users/apple1/desktop/kaggle/RentalListing/train.json")
frq_features = table(unlist(train$features))
top_features = names(frq_features[frq_features>1000])
extraction_feature = t(sapply(df_train_sent$features,
                       function(x) {
                         as.numeric(top_features %in% x)
                       }))
colnames(extraction_feature) <- top_features
df_train_sent = cbind(df_train_sent, extraction_feature)
