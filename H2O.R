library(h2o)
h2o.init(nthreads = -1, max_mem_size="6g")
train = read.csv("/Users/apple1/Desktop/kaggle/RentalListing/train.csv")
train <- as.h2o(train, destination_frame = "train.hex")
varnames <- setdiff(colnames(train), "interest_level")
#ratios add up to less than 1?
#how to parse data using R in h2o?
split = h2o.splitFrame(train, ratios = c(0.7, 0.29))
traindata <- split[[1]]
validationdata <- split[[2]]

gbm1 <- h2o.gbm(x = varnames
                ,y = "interest_level"
                ,training_frame = traindata
                ,validation_frame = validationdata
                ,distribution = "multinomial"
                ,model_id = "gbm1"
                #,nfolds = 5
                ,ntrees = 200
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

