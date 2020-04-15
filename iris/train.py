import os
import joblib
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"log.csv")
pickles_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"pickles")

#check if pickles_path exists
if not os.path.exists(pickles_path):
    os.mkdir(pickles_path)

def train():
    #step1 load x,y as data
    x,y= load_iris(return_X_y=True)
    #step 2 split the data into test and train data
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.2)
    #STEP 3 TRAIN THE MODEL
    clf= LogisticRegression()
    scaler = StandardScaler()
    pipe= Pipeline([("scaling",scaler),("classification",clf)])
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    #step4 to mesure its performance
    f1_current = f1_score(y_test,y_pred,average="macro")
    # current time stamp
    timestamp = datetime.now().isoformat()
    # TO GET THE CLASSIFIER NAME
    clf_name = clf.__class__.__name__
     
    with open (log_path,"a+") as file:
        line="{},{},{}\n".format(clf_name,f1_current,timestamp)
        file.write(line)
    # export the model
    clf_path=os.path.join(pickles_path,"clf.pkl")
    joblib.dump(pipe,clf_path,compress=True)


def load_clf():
    clf_path=os.path.join(pickles_path,"clf.pkl")
    if  os.path.exists(clf_path):
        return joblib.load(clf_path)
    else:
        return None
    

if __name__ == "__main__":
    train()
    clf=load_clf()
    print(type(clf))



    