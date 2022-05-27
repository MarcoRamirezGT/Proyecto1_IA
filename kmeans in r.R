getwd()
setwd("D:/UVG/2022/Semestre 1 2022/inteligencia artificial/proyecto 2/local/Proyecto1_IA")

porcentaje<-0.7

df <- read.csv("Fraud.csv")
df <- na.omit(df)
summary(df$oldbalanceOrg)
summary(df$newbalanceDest)
summary(df$amount)
summary(df$type)
summary(df$nameOrig)


str(df$type)
table(df$type)
table(df['nameOrig'])

library(ggplot2)
library (dplyr)
library(factoextra)
library(cluster)

library(caret)
library(e1071)

km.res <- kmeans(df, 4, nstart = 25)
