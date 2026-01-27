from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.impute import SimpleImputer
def preprocess_data(df):
    """
    clean data , encode categorical variables , split dataset
    """
    print("Columns: ", df.columns)
    print("Initial shape: " , df.shape)

    target = "SalePrice"

    # handle missing values
    df = df.dropna(subset=[target])

    #convert categorical columns to numerical
    df = pd.get_dummies(df, drop_first= True)

    #split features and target variable
    x = df.drop(target , axis=1)
    y = df[target]

    print(" x shape befor impute: ", x.shape)

    imputer = SimpleImputer(strategy="mean")
    x = imputer.fit_transform(x)



    #Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    return x_train, x_test,y_train, y_test