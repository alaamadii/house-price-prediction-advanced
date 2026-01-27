from data_loader import load_data
from preprocessing import preprocess_data
from model import train_models


df = load_data("data/train.csv")
x_train, x_test,y_train,y_test = preprocess_data(df)
train_models(x_train,x_test,y_train,y_test)
