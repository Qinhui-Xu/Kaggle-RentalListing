
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

train <- fromJSON("/Users/apple1/desktop/kaggle/RentalListing/train.json")

vars <- setdiff(names(train), c("photos", "features"))
train_df <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)
train_df$id<-seq(1:length(train_df$building_id)) #numerical ids!

library(syuzhet)
library(DT)
sentiment <- get_nrc_sentiment(train_df$description)
senti <- datatable(sentiment)
sentiment$id<-seq(1:nrow(sentiment))
sent_df<-merge(train_df,sentiment, by.x="id", by.y="id", all.x=T, all.y=T)
datatable(head(sent_df))
train_sent <- apply(sent_df,2,as.character)
train_sent2 <- as.data.table(train_sent)

write.csv(train_sent2, file = "train_sent.csv",row.names=FALSE)

