library(h2o)
h2o.init(nthreads = -1, max_mem_size="6g")
library(data.table)
library(jsonlite)
library(h2o)
library(lubridate)
h2o.init(nthreads = -1, max_mem_size="8g")

# Load data
t1 <- fromJSON("../input/train.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
t2 <- data.table(bathrooms=unlist(t1$bathrooms)
                 ,bedrooms=unlist(t1$bedrooms)
                 ,building_id=as.factor(unlist(t1$building_id))
                 ,created=as.POSIXct(unlist(t1$created))
                 # ,description=unlist(t1$description) # parse errors
                 # ,display_address=unlist(t1$display_address) # parse errors
                 ,latitude=unlist(t1$latitude)
                 ,longitude=unlist(t1$longitude)
                 ,listing_id=unlist(t1$listing_id)
                 ,manager_id=as.factor(unlist(t1$manager_id))
                 ,price=unlist(t1$price)
                 ,interest_level=as.factor(unlist(t1$interest_level))
                 # ,street_adress=unlist(t1$street_address) # parse errors
)
t2[,":="(yday=yday(created)
         ,month=month(created)
         ,mday=mday(created)
         ,wday=wday(created)
         ,hour=hour(created))]

train <- as.h2o(t2[,-"created"], destination_frame = "train.hex")
train = read.csv("/Users/apple1/Desktop/kaggle/RentalListing/train_sent.csv")
train <- as.h2o(train, destination_frame = "train.hex")
varnames <- setdiff(colnames(train), "interest_level")
split = h2o.splitFrame(train, ratios = c(0.7))
traindata <- split[[1]]
validationdata <- split[[2]]

gbm1 <- h2o.gbm(x = varnames
                ,y = "interest_level"
                ,training_frame = traindata
                ,validation_frame = validationdata
                ,distribution = "multinomial"
                ,model_id = "gbm1"
                #,nfolds = 5
                ,ntrees = 300
                ,learn_rate = 0.01
                ,max_depth = 7
                ,min_rows = 20
                ,sample_rate = 0.8
                ,col_sample_rate = 0.7
                ,stopping_rounds = 5
                ,stopping_metric = "logloss"
                ,stopping_tolerance = 0
                ,seed=321
)
h2o.download_mojo(gbm1) 

