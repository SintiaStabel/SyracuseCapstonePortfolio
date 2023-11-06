# Melissa Mosier
# IST719
# Credit Card Fraud

##### Read in Data #####--------------------------------

my.dir <- "C:\\Users\\mmosi\\OneDrive\\Documents\\SyracuseFiles\\IST718_Big Data\\Final Project\\"
ccData <- read.csv(file=paste0(my.dir, "ccfraud1.csv"), 
                   header = TRUE, 
                   stringsAsFactors = FALSE)
View(ccData)
# unnamed* | transdatetime | ccnum | merchant | category | amt | first | last
# gender | street | city | state | zip | lat | long | citypop | job | dob
# trans_num | unixtime* | merch_lat | merch_long | is_fraud

str(ccData)
summary(ccData)
(ncol(ccData)*4)*(nrow(ccData)/100)   # 9200
# What does the data set represent?
# Transactions made legitimately or fraudulently, and all metadata associated
# with those transactions, like when and where they occurred, for what purpose,
# and the account information, etc. 

# FIRST VISUAL 
table(ccData$is_fraud)               # 9945:55
pie(table(ccData$is_fraud), main = "Cases of Credit Card Fraud")
# Exported graph edited in Illustrator 

ccData2 <- ccData

##### Cleaning #####--------------------------------

# How did I clean in Python?
##Unnamed: 0 - remove. It doesn't do anything.
#unix_time - remove. I don't know what this is. 
ccData2 <- ccData2[,-c(1,20)]

#trans_date_trans_time - convert to datetime
#cc_num	merchant - remove "fraud_"
ccData2$merchant <- gsub("fraud_", "", ccData$merchant)

#dob - convert to age
# install.packages("eeptools")
library(eeptools)
library(lubridate)
today <- Sys.Date()

which(is.na(ccData2$dob))
ccData2$dob <- dmy(ccData2$dob)

ccData2$dob <- as.Date(ccData2$dob)
age <- age_calc(ccData2$dob, today, units="years")
ccData2$age <- floor(age)

#dummy variable for gender
# install.packages("fastDummies")
library(fastDummies)
ccData2$genDummy <- dummy_cols(ccData2$gender)

# These are all fine
# category  /  amt  /  first  /  last
# street  /  city  /  state  /  zip  /  lat  /  long
# city_pop  /  job  /  trans_num  /  merch_lat
# merch_long  /  is_fraud 

View(ccData2)
summary(ccData2)

##### Exploratory Distributions #####--------------------------------
# Ex: Histograms, box plots, density plots; frequencies in bar chart or pie chart

par(mfrow = c(2,1))

fraudSubset <- subset(ccData2, ccData2$is_fraud==1)

barplot(table(ccData2$gender), main="Total Transactions by gender")            # gen
barplot(table(fraudSubset$gender), main="Fraudulent Transactions by gender")   # fraud
# t.test( ccData2$gender[ccData2$genDummy$.data_M==1], ccFraud$gender[ccFraud$genDummy$.data_M==1])  # fix this too

hist(log(ccData2$city_pop), main="Total Transactions by city population")            # gen pop
hist(log(fraudSubset$city_pop), main="Fraudulent Transactions by city population")   # frauded pop

#amt, take log
hist(log(ccData2$amt), xlim=c(0,10), ylim=c(0,2500), main="Total Transactions by transaction amount")
hist(log(fraudSubset$amt), xlim=c(0,10), ylim=c(0,25), main="Total Transactions by transaction amount")
# look into this difference!

# Exported each set of graphs as PDFs


##### Multidimensional Graph #####--------------------------------

plot(ccData2$city_pop, ccData2$age)

library(RColorBrewer)
col.vec <- rep(rgb(30,144,255,maxColorValue=255), nrow(ccData2))
col.vec[ccData2$is_fraud==1] <- rgb(255,64,64, maxColorValue = 255)
plot(ccData2$city_pop, ccData2$age, pch=16, cex=1, col=col.vec)

# Exported graph edited in Illustrator 


