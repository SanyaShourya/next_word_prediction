from preprocessing import Preprocessing
from model import Model
from training import Training
from predict import Predict
import pandas as pd
if __name__=="__main__":
    df=pd.read_csv("Data/quotes.csv")
    preprocessing_obj=Preprocessing(df)
    X,y=preprocessing_obj.start_preprocessing()
    model_obj=Model(X)
    model=model_obj.make_model()
    training_obj=Training(model=model,X=X,y=y)
    training_obj.start_training()
    user_input=input()
    predict_obj=Predict(preprocessing_obj)
    output=predict_obj.predict_next_words(model,user_input,10)
    print(output)
    
    