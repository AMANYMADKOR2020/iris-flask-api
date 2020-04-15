from flask import Flask,request,jsonify
from iris.train import load_clf

app = Flask(__name__)
global classifier
classifier = load_clf()

# check endpoint
@app.route("/")
def index():
    return "all is working"

@app.route("/clf",methods=['post'])
def classify():
    try:
        s1 = request.json['s1']
        s2 = request.json['s2']
        s3 = request.json['s3']
        s4 = request.json['s4']
        global classifier
        sample=[s1,s2,s3,s4]
        predict = int(classifier.predict([sample])[0])
        probs=classifier.predict_proba([sample])[0]
        classes = classifier.classes_
        classes_prob = {int(x):round(y,3) for x,y in zip (classes,probs)}
        return jsonify (
            { "status":"success",
             "response":" Data loaded successfully",
             "prediction":predict,
             "confedence":classes_prob})
    except Exception as e:
        print(e)
        return jsonify ({"status":"failed",
                         "response":"Check input json",
                           "error": str(e) })


@app.route("/clf/agg",methods=['post'])
def classify_agg():
    try:
        data = request.json['data']
        if not isinstance(data,list):
            raise Exception("Data has to be list else use clf endpoint")
        #preprocissing for input

        global classifier
        predict = classifier.predict(data)
        probs=classifier.predict_proba(data) 
        predict = map (lambda x:int(x),predict)
        probs = map (lambda x:list(x),probs)
        return jsonify (
            { "status":"success",
             "response":" Data loaded successfully",
             "prediction":list(predict),
             "confedence":list(probs)})
    except Exception as e:
        print(e)
        return jsonify ({"status":"failed",
                         "response":"Check input json",
                           "error": str(e) })
    