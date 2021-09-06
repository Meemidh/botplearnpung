# Import Library
import json
import os
import datetime
from re import M
import numpy as np
import os.path
from flask import Flask,request,make_response
import dill
# ----Additional from previous file----
from random import randint
import firebase_admin
from firebase_admin import credentials,firestore
import pandas as pd
import string 
import pickle
from pythainlp.corpus import thai_stopwords
from pythainlp import word_tokenize, Tokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import svm
from My_function import createSVMModel,TestingProcessSVM,text_process

cred = credentials.Certificate("plenpung-firebase-adminsdk-3kbnb-3d11f00d77.json")
firebase_admin.initialize_app(cred)

#!โหลด model 
sentiment_Model = pickle.load(open("sentiment_model.model", 'rb'))
cvec_model = dill.load(open("cvec_model.model",'rb'))
tfidf_transformer_model = pickle.load(open("tfidf_transformer_model.model",'rb'))


app = Flask(__name__) # Using post as a method
@app.route("/",methods=['POST'])

def mainfunction():
    # รับ intent จาก Dailogflow
    question_from_dailogflow_raw = request.get_json(silent=True, force=True)
    # เรียกใช้ฟังก์ชัน generating_answer เพื่อแยกส่วนของคำถาม
    answer_from_bot = generating_answer(question_from_dailogflow_raw,cvec_model,sentiment_Model,tfidf_transformer_model)
    # ตอบกลับไปที่ Dialogflow
    r = make_response(answer_from_bot)
    r.headers['Content-Type'] = 'application/json'  # Setting Content Type
    
    return r

def generating_answer(question_from_dailogflow_dict,cvec_model,sentiment_Model,tfidf_transformer_model):

    # Print intent that recived from dialogflow.
    print(json.dumps(question_from_dailogflow_dict, indent=4, ensure_ascii=False))
    # Getting intent name form intent that recived from dialogflow.
    intent_group_question_str = question_from_dailogflow_dict["queryResult"]["intent"]["displayName"]
    # Select function for answering question
    print('######'+ intent_group_question_str)
    # ref from line
    message = question_from_dailogflow_dict["queryResult"]["queryText"]
    user_id = question_from_dailogflow_dict["originalDetectIntentRequest"]["payload"]["data"]["source"]["userId"]
    # time
    time = question_from_dailogflow_dict["originalDetectIntentRequest"]["payload"]["data"]["timestamp"]
    date_time = datetime.datetime.fromtimestamp(int(time) / 1000)  # using the local timezone
    
    
    #!--------------------------------------------------------------
    # #!เอา message จากไลน์มาเข้าจากทำนาย sentiment 
    message2 = pd.Series({'message': message})
    message2 = message2.apply(text_process)
    
    
    my_predictions = TestingProcessSVM(message2,sentiment_Model,cvec_model,tfidf_transformer_model)
    
    result_sentiment = my_predictions

    if result_sentiment == 1:
        sentiment_str = '(≧ᴗ≦)'
    elif result_sentiment == -1:
        sentiment_str = '(╥﹏╥)'
    elif result_sentiment == 0:
        sentiment_str = '(๑・_・๑)'
    
    print(result_sentiment)
    
    result = result_sentiment.item() 
    # print(result)
    #!-----------------------------------------------------------

    db_ref = firestore.client().collection('UserSentimentRecord')
    newchat =  date_time.strftime("%Y-%m-%d %H:%M:%S") + "_" + user_id
    db_ref.document(newchat).set({
        'text': message,
        'userID': user_id,
        'datetime': date_time.strftime("%Y-%m-%d %H:%M:%S"),
        'intent': intent_group_question_str,
        'sentiment': sentiment_str,
        'result': result
    })
    
    ## save เป็น ไฟล์ เก็บ เวลา userid  message intent  ##
    filesave = "HistoryChats/HistChat" + str(date_time.strftime("%Y-%m-%d-%H")) + ".txt"
    textsave = str(date_time.strftime("%Y-%m-%d %H:%M:%S")) + ", " + user_id + ", " + message + "\n"
    print(textsave)
    #save these items to Firebase including message, time, sentiment_str , UserID
    
    
    if intent_group_question_str == 'stupid':
        answer_str = stupid(sentiment_str)
    if intent_group_question_str == 'congrat':
        answer_str = congrat(sentiment_str)
    if intent_group_question_str == 'fault':
        answer_str = fault_fn(sentiment_str)
    if intent_group_question_str == 'branch':
        answer_str = branch_fn(sentiment_str)
    if intent_group_question_str == 'fault_delivery':
        answer_str = fault_delivery(sentiment_str)
    if intent_group_question_str == 'want':
        answer_str = want_fn(sentiment_str)
    if intent_group_question_str == 'pay':
        answer_str = pay_fn(sentiment_str)
    if intent_group_question_str == 'blame':
        answer_str = blame_fn(sentiment_str)
    if intent_group_question_str == 'thankyou':
        answer_str = thankyou_fn(sentiment_str)
    if intent_group_question_str == 'so_far':
        answer_str = so_far(sentiment_str)
    if intent_group_question_str == 'ask_delivery':
        answer_str = ask_delivery(sentiment_str)
    if intent_group_question_str == 'Hungry':
        answer_str = hungry_fn(sentiment_str)
    if intent_group_question_str == 'haha':
        answer_str = haha(sentiment_str)
    if intent_group_question_str == 'shout':
        answer_str = shout(sentiment_str)
    if intent_group_question_str == 'no_area':
        answer_str = no_area(sentiment_str)
    if intent_group_question_str == 'cancel':
        answer_str = cancel(sentiment_str)
    if intent_group_question_str == 'covid':
        answer_str = covid(sentiment_str)
    if intent_group_question_str == 'whatdo':
        answer_str = whatdo(sentiment_str)
    if intent_group_question_str == 'curse':
        answer_str = curse(sentiment_str)
    if intent_group_question_str == 'praise':
        answer_str = praise(sentiment_str)
    if intent_group_question_str == 'Plenpung_bot':
        answer_str = Plenpung_bot(sentiment_str)
    if intent_group_question_str == 'Opening_hours':
        answer_str = Opening_hours(sentiment_str)
    if intent_group_question_str == 'food_recommend':
        answer_str = food_recommend(sentiment_str)
    if intent_group_question_str == 'drink_recommend':
        answer_str = drink_recommend(sentiment_str)
    if intent_group_question_str == 'dessert_recommend':
        answer_str = dessert_recommend(sentiment_str) 
    if intent_group_question_str == 'basic_greeting':
        answer_str = basic_greeting(sentiment_str)
    if intent_group_question_str == 'table':
        answer_str = table(sentiment_str)
    if intent_group_question_str == 'contact':
        answer_str = contact(sentiment_str)
    if intent_group_question_str == 'GoodMorning':
        answer_str = goodmoring(sentiment_str)
    if intent_group_question_str == 'GoodAfternoon':
        answer_str = goodafternoon(sentiment_str)
    if intent_group_question_str == 'GoodEvening':
        answer_str = goodevening(sentiment_str)
    if intent_group_question_str == 'Goodnight':
        answer_str = goodnight(sentiment_str)
    if intent_group_question_str == 'GoodBye':
        answer_str = goodbye(sentiment_str)
    if intent_group_question_str == 'store':
        answer_str = location(sentiment_str)
    if intent_group_question_str == 'order':
        answer_str = order(sentiment_str)
    if intent_group_question_str == 'food':
        answer_str = food(sentiment_str)
    if intent_group_question_str == 'drink':
        answer_str = drink(sentiment_str)
    if intent_group_question_str == 'dessert':
        answer_str = dessert(sentiment_str)
    if intent_group_question_str == 'Fallbackbot':
        answer_str = fallback_fn(sentiment_str)
    # Build answer dict
    answer_from_bot = {"fulfillmentText": answer_str}
    # Convert dict to JSON
    answer_from_bot = json.dumps(answer_from_bot, indent=4)
    return answer_from_bot

#!-----------------------------------------------------------------------
def stupid(sentiment_str):
    database_ref = firestore.client().document('talk/stupid')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def congrat(sentiment_str):
    database_ref = firestore.client().document('talk/congrat')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def fault_fn(sentiment_str):
    database_ref = firestore.client().document('talk/fault')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function
def branch_fn(sentiment_str):
    database_ref = firestore.client().document('talk/branch')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function
def fault_delivery(sentiment_str):
    database_ref = firestore.client().document('talk/fault_delivery')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function
def want_fn(sentiment_str):
    database_ref = firestore.client().document('talk/want')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function
def pay_fn(sentiment_str):
    database_ref = firestore.client().document('talk/pay')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function
def blame_fn(sentiment_str):
    database_ref = firestore.client().document('talk/blame')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function
def thankyou_fn(sentiment_str):
    database_ref = firestore.client().document('talk/thankyou')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def so_far(sentiment_str):  
    database_ref = firestore.client().document('talk/so_far')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def ask_delivery(sentiment_str):  
    database_ref = firestore.client().document('talk/delivery')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def hungry_fn(sentiment_str):  
    database_ref = firestore.client().document('Hungry/Qhungry')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def haha(sentiment_str):  
    database_ref = firestore.client().document('talk/haha')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def no_area(sentiment_str):  
    database_ref = firestore.client().document('talk/no_area')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def shout(sentiment_str):  
    database_ref = firestore.client().document('talk/shout')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def cancel(sentiment_str):  
    database_ref = firestore.client().document('talk/cancel')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def covid(sentiment_str):  
    database_ref = firestore.client().document('talk/covid')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def whatdo(sentiment_str):  
    database_ref = firestore.client().document('talk/whatdo')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function


def curse(sentiment_str):  
    database_ref = firestore.client().document('talk/curse')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def praise(sentiment_str):  
    database_ref = firestore.client().document('talk/praise')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def Plenpung_bot(sentiment_str):  
    database_ref = firestore.client().document('Location/bot')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def Opening_hours(sentiment_str):  
    database_ref = firestore.client().document('Location/hours')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =   aws + ' ' + sentiment_str
    return answer_function

def food_recommend(sentiment_str):  
    database_ref = firestore.client().document('Menu/recom_food')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  'แนะนำเมนู' + aws + 'เลยครับ' + ' ' + sentiment_str
    return answer_function

def drink_recommend(sentiment_str):  
    database_ref = firestore.client().document('Menu/recom_drink')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  'แนะนำเมนู' + aws + 'เลยครับ' + ' ' + sentiment_str
    return answer_function

def dessert_recommend(sentiment_str):  
    database_ref = firestore.client().document('Menu/remon_desseret')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  'แนะนำเมนู' + aws + 'เลยครับ' + ' ' + sentiment_str
    return answer_function

def basic_greeting(sentiment_str):  
    database_ref = firestore.client().document('Greeting/Basic')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def table(sentiment_str):  
    database_ref = firestore.client().document('Location/table')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def contact(sentiment_str):  
    database_ref = firestore.client().document('Location/contact')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def goodmoring(sentiment_str):  
    database_ref = firestore.client().document('Greeting/GoodMorning')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def goodafternoon(sentiment_str): 
    database_ref = firestore.client().document('Greeting/GoodAfternoon')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
 
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def goodevening(sentiment_str):  #
    database_ref = firestore.client().document('Greeting/GoodEvening')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def goodnight(sentiment_str):  
    database_ref = firestore.client().document('Greeting/GoodNight')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def goodbye(sentiment_str):  
    database_ref = firestore.client().document('Greeting/GoodBye')
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def location(sentiment_str):  
    database_ref = firestore.client().document('Location/store')  
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function = aws + ' ' + sentiment_str
    return answer_function


def order(sentiment_str):
    database_ref = firestore.client().document('Order/Question')  # ปรับ มา ใหม่
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def food(sentiment_str):
    database_ref = firestore.client().document('Menu/Food')  # ปรับ มา ใหม่
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def drink(sentiment_str):
    database_ref = firestore.client().document('Menu/drink')  # ปรับ มา ใหม่
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws + ' ' + sentiment_str
    return answer_function

def dessert(sentiment_str):
    database_ref = firestore.client().document('Menu/Dessert')  # ปรับ มา ใหม่
    database_dict = database_ref.get().to_dict()
    database_list = list(database_dict.values())
    ran_menu = randint(0, len(database_list) - 1)
    aws = database_list[ran_menu]
    # -------------------------------------
    answer_function =  aws +  ' ' + sentiment_str
    return answer_function


def fallback_fn(sentiment_str):
    if sentiment_str == '(≧ᴗ≦)':
        aws = 'แฮร่ บอทยังไม่รู้จักคำนี้ค่ะ' 
    elif sentiment_str == '(๑・_・๑)':
        aws = 'น้องบอทไม่เข้าใจค่ะ กรุณาถามใหม่'
    elif sentiment_str == '(╥﹏╥)':
        aws = 'ขออภัยค่ะ น้องบอทไม่เข้าใจ ขอเวลาให้บอทเรียนรู้หน่อยค่ะ'
    # -------------------------------------
    answer_function = aws 
    return answer_function

#!------------------------------------------------------------------
# Flask
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=True, port=port, host='0.0.0.0', threaded=True)


