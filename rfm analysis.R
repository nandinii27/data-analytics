df_data <- read.csv(file.choose())

library(data.table)
library(dplyr)
library(ggplot2)
#library(stringr)
#library(DT)
library(tidyr)
library(knitr)
library(rmarkdown)
glimpse(df_data)

df_data <- df_data %>% 
  mutate(Quantity = replace(Quantity, Quantity<=0, NA),
         UnitPrice = replace(UnitPrice, UnitPrice<=0, NA))

df_data <- df_data %>%
  drop_na()

df_data <- df_data %>% 
  mutate(InvoiceNo=as.factor(InvoiceNo), StockCode=as.factor(StockCode), 
         InvoiceDate=as.Date(InvoiceDate, '%m/%d/%Y %H:%M'), CustomerID=as.factor(CustomerID), 
         Country=as.factor(Country))

df_data <- df_data %>% 
  mutate(total_dolar = Quantity*UnitPrice)

glimpse(df_data)

df_RFM <- df_data %>% 
  group_by(CustomerID) %>% 
  summarise(recency=as.numeric(as.Date("2012-01-01")-max(InvoiceDate)),
            frequenci=n_distinct(InvoiceNo), monitery= sum(total_dolar)/n_distinct(InvoiceNo)) 

summary(df_RFM)

kable(head(df_RFM))
hist(df_RFM$recency)

df_RFM2 <- df_RFM
row.names(df_RFM2) <- df_RFM2$CustomerID

df_RFM2$CustomerID <- NULL

df_RFM2 <- scale(df_RFM2)
summary(df_RFM2)

d <- dist(df_RFM2)
c <- hclust(d, method = 'ward.D2')

plot(c)
members <- cutree(c,k = 8)

members[1:5]

table(members)
aggregate(df_RFM[,2:4], by=list(members), mean)
