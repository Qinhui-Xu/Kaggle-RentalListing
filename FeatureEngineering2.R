#####Sentiment Analysis based on Description
library(syuzhet)
library(DT)
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

data <- fromJSON("~/Desktop/kaggle/RentalListing/train.json")
vars <- setdiff(names(data), c("photos", "features"))
train_df <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)
train_df$id<-seq(1:length(train_df$building_id)) #numerical ids!
library(syuzhet)
library(DT)
sentiment <- get_nrc_sentiment(train_df$description)
datatable(head(sentiment))
sentiment$id<-seq(1:nrow(sentiment))
sentiment <- subset(sentiment, select = c(negative,positive,id))
sent_df<-merge(train_df,sentiment, by.x="id", by.y="id", all.x=T, all.y=T)

#####Deal with Feature Variable
library(jsonlite)
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
#x <- toJSON(sent_df)
#write(x, "train_feature_sent.json")

#####Log the price
