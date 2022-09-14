import datetime

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import math
import pandas as pd
import numpy as np
from NewAlgo import *
from datetime import date
from dateutil.relativedelta import relativedelta
import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import load_model
#import weka.core.serialization as serialization
#from weka.classifiers import Classifier

app = Flask(__name__, template_folder="./templates", static_url_path="/static")
model = load_model("mymodel.h5")
#model = Classifier.load("multilayerperceptron.model")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('homepage'))

        return render_template("login.html", form=form)
    return render_template("login.html", form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")
    return render_template('signup.html', form=form)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/diagnose', methods=['POST'])
def diagnose():

    # retrieving values from form
    init_features = [int(x) for x in request.form.values()]
    '''smoking = float(request.form.get("smoking"))
    yellowFingers = 0.02 * float(request.form.get("yellow_fingers"))
    age = float(request.form.get("age"))
    gender = str(request.form["gender"])
    axiety = 0.04 * float(request.form.get("axiety"))
    peerPressure = float(request.form.get("peer_pressure"))
    chronicDisease = float(request.form.get("chronic_disease"))
    fatigue = 0.04 * int(request.form["fatigue"])
    alcoholConsuming = str(request.form["alcohol_consuming"])
    wheezing = 0.2 * float(request.form.get("wheezing"))
    coughing = 0.21 * float(request.form.get("coughing"))
    shortness = 0.29 * int(request.form["shortness_of_breath"])
    allergy = str(request.form.get("allergy"))
    chestPain = 0.13 * int(str(request.form.get("chest_pain")))
    swallowing = 0.06 * int(str(request.form["swallowing_difficulty"]))
    #final_features = (np.array[(smoking, yellowFingers, age, gender, axiety, peerPressure,
                               chronicDisease,fatigue, alcoholConsuming, wheezing,
                               coughing, shortness, allergy, chestPain, swallowing )]).reshape(1,15)
    features_name = ["Gender", "Age", "Smoking", "Yellow Fingers", "Axiety", "Peer Pressure", "Chronic Disease",
                     "Fatigue", "Allergy", "Wheezing", "Alcohol Consuming", "Coughing", "Shortness Of Breath",
                     "Swallowing Difficulty", "Chest Pain"]

    df = pd.DataFrame(final_features, columns=features_name)
    output = model.predict(final_features)'''

    final_features = np.array(init_features).reshape(1, 15)
    prediction = model.predict(final_features)  # making prediction

    return render_template('index.html',
                               prediction_text= 'Probability of lung cancer presence is {}.'.format(prediction[0]))

@app.route('/hybrid', methods=['GET','POST'])
def hybrid():
    return render_template('func2.html')

@app.route('/hybridpredict', methods=['POST'])
def hybrid_predict():
    #inal_features = []
    now = datetime.datetime.now()
    #username = session.username
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features).reshape(1,15)

    features_name = ["Gender", "Age", "Smoking", "Yellow Fingers", "Axiety", "Peer Pressure", "Chronic Disease",
                      "Fatigue", "Allergy", "Wheezing",  "Alcohol Consuming", "Coughing", "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(features_value)

    if output == 1:
        res_val = "a high risk of Lung Disease"
    else:
        res_val = "a low risk of Lung Disease"
    '''if (
        request.method == "POST"
        and request.form["action"] == "⚖ Calculate"
        and "weight" in request.form
        and "height" in request.form
        and "age" in request.form
        and "gender" in request.form
        and "waist" in request.form
        and "neck" in request.form
        and "hip" in request.form
        and "activity-level" in request.form
        and "swallowing" in request.form
        and "wheezing" in request.form
        and "coughing" in request.form
        and "body-type" in request.form
        and "allergy" in request.form
        and "shortness" in request.form
    ):
        weight = float(request.form.get("weight"))
        height = float(request.form.get("height"))
        age = float(request.form.get("age"))
        gender = str(request.form["gender"])
        hip = float(request.form.get("hip"))
        neck = float(request.form.get("neck"))
        waist = float(request.form.get("waist"))
        activity_level = request.form["activity-level"]
        bodyType = str(request.form["body-type"])
        wheezing = float(request.form.get("wheezing"))
        coughing = float(request.form.get("coughing"))
        shortness = request.form["shortness"]
        allergy = str(request.form.get("allergy"))


        if gender == "0":
            final_features = [0]
        else:
            final_features = [1]

        if weight == 0:
            final_features = final_features.append(0)
        else:
            final_features = final_features.append(1)

        prediction = model.predict(final_features)

    #return (final_features)'''

    return render_template('result.html', prediction_text='Probability of lung cancer presence is 0.4337.', now=now)

@app.route('/homepage', methods=['GET','POST'])
def homepage():
    data = [(1, 69, 0,	1,	1,	0,	0,	1,	0,	1,	1,	1,	1,	1,	1,	1),
            (1, 74, 1,	0,	0,	0,	1,	1,	1,	0,	0,	0,	1,	1,	1,	1),
    (0	,59	,0,	0,	0,	1,	0,	1,	0,	1,	0,	1,	1,	0,	1,	0),
    (1	,63	,1,	1,	1,	0,	0,	0,	0,	0,	1,	0,	0,	1,	1,	0),
    (0	,63	,0,	1,	0,	0,	0,	0,	0,	1,	0,	1,	1,	0,	0,	0)]
    labels = [row[-1] for row in data]
    values = [row[:14] for row in data]

    return render_template('index.html', values=values, labels=labels)

@app.route("/predict_api", methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    :return:
    '''
    init_features = [int(x) for x in request.form.values()]
    final_features = np.array(init_features).reshape(1, 15)

    prediction = model.predict(final_features)
    output=prediction[0]
    return jsonify(output)

@app.route("/", methods=["POST", "GET"])
def Calculate():
    ######## BASIC INFORMATION ########
    today = date.today()
    bmr = ""
    bodyFat = ""
    tdde = ""
    calo_total = 1
    bmi = ""
    bmi_comment = ""
    ######## SET GOAL ####################
    goal = ""
    weight_change_rate = ""
    date_goal = ""
    weight_goal = ""

    ############## BODY BUILDING ############
    bodyType = ""
    #### MACRONUTRIENTS ####
    cabohydrate = 1
    protein = 1
    fats = 1

    ######## SUGGESTION #########
    comment = ""
    sorry = ""
    x = 0
    suggestion = ""
    comment_calo = ""
    ideal_fat = 1
    date_goal_print = ""
    title = ""
    calories = ""
    pro = ""
    carbs = ""
    fat = ""
    sum_calo = 0
    sum_pro = 0
    sum_carbs = 0
    sum_fat = 0
    data = []
    labels = []
    values = []
    percentPro = 0
    percentCarbs = 0
    percentFat = 0
    num_meal = 0
    sum_Cholesterol = 0
    sum_fiber = 0
    sum_sugar = 0
    sum_VinD = 0
    sum_iron = 0
    sum_calcium = 0
    sum_potassium = 0
    sum_sodium = 0
    dishType = ""
    firstMeal = ""
    secondMeal = ""
    thirdMeal = ""
    fourthMeal = ""
    fifthMeal = ""

    firstMeal_name = ""
    firstMeal_serving = ""
    firstMeal_calo = ""
    firstMeal_pro = ""
    firstMeal_Carb = ""
    firstMeal_fat = ""
    firstMeal_ingredients = ""

    secondMeal_name = ""
    secondMeal_serving = ""
    secondMeal_calo = ""
    secondMeal_pro = ""
    secondMeal_Carb = ""
    secondMeal_fat = ""
    secondMeal_ingredients = ""

    thirdMeal_name = ""
    thirdMeal_serving = ""
    thirdMeal_calo = ""
    thirdMeal_pro = ""
    thirdMeal_Carb = ""
    thirdMeal_fat = ""
    thirdMeal_ingredients = ""

    fourthMeal_name = ""
    fourthMeal_serving = ""
    fourthMeal_calo = ""
    fourthMeal_pro = ""
    fourthMeal_Carb = ""
    fourthMeal_fat = ""
    fourthMeal_ingredients = ""

    fifthMeal_name = ""
    fifthMeal_serving = ""
    fifthMeal_calo = ""
    fifthMeal_pro = ""
    fifthMeal_Carb = ""
    fifthMeal_fat = ""
    fifthMeal_ingredients = ""
    ##### TIME TO LOSS/GAIN WEIGHT ####
    num_of_weeks_to_gain = ""
    num_of_weeks_to_loss = ""
    df = pd.read_csv(
        "FatFixed.csv"
    )

    if (
        request.method == "POST"
        and request.form["action"] == "⚖ Calculate"
        and "weight" in request.form
        and "height" in request.form
        and "age" in request.form
        and "gender" in request.form
        and "waist" in request.form
        and "neck" in request.form
        and "hip" in request.form
        and "activity-level" in request.form
        and "goal-weight" in request.form
        and "weight-rate" in request.form
        and "goal" in request.form
        and "body-type" in request.form
        and "allergy" in request.form
    ):
        weight = float(request.form.get("weight"))
        height = float(request.form.get("height"))
        age = float(request.form.get("age"))
        gender = str(request.form["gender"])
        hip = float(request.form.get("hip"))
        neck = float(request.form.get("neck"))
        waist = float(request.form.get("waist"))
        activity_level = request.form["activity-level"]
        bodyType = str(request.form["body-type"])
        weight_goal = float(request.form.get("goal-weight"))
        weight_change_rate = float(request.form.get("weight-rate"))
        goal_radio = request.form["goal"]
        allergy = str(request.form.get("allergy"))
        print(waist)
        print(gender)
        print(hip)
        print(height)
        print(goal_radio)
        print(age)
        print(neck)
        print(activity_level)
        print(weight_goal)
        print(weight_change_rate)
        print(bodyType)
        print(allergy)

        if gender == "Male":

            bmi = weight / ((height * 0.01) * (height * 0.01))

            if bmi < 18.5:
                bmi_comment = "Underweight"
            elif bmi >= 18.5 and bmi < 24.9:
                bmi_comment = "Normal"
            elif bmi >= 25 and bmi < 29.9:
                bmi_comment = "Overweight"
            else:
                bmi_comment = "Very overweight"

            bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) + 5

            tdde = bmr * float(activity_level)

            bodyFat = (
                495
                / (
                    1.0324
                    - 0.19077 * math.log10(float(waist) - float(neck))
                    + 0.15456 * math.log10(float(height))
                )
                - 450
            )
            if bodyFat > 2 and bodyFat <= 5:
                comment = "Essential fat"
            elif bodyFat > 5 and bodyFat <= 13:
                comment = "Athletes"
            elif bodyFat > 13 and bodyFat <= 17:
                comment = "Fitness"
            elif bodyFat > 17 and bodyFat <= 24:
                comment = "Average"
            elif bodyFat > 24:
                comment = "Obese"

            ################# ====> GAIN WEIGHT <======= ###############
            if goal_radio == "G":
                weight_add = np.abs(weight - weight_goal)
                calories_gain_each_day = (float(weight_change_rate) / 7) * 7716.17
                num_of_weeks_to_gain = round(
                    float(weight_add) / float(weight_change_rate)
                )
                calo_total = tdde + calories_gain_each_day
                date_goal = date.today() + relativedelta(weeks=+num_of_weeks_to_gain)
                date_goal = date_goal.strftime("%d/%m/%Y")

                date_goal_print = (
                    "You should reach your goal in {} to gain {} kg !".format(
                        date_goal, weight_goal
                    )
                )

                cabohydrate = ((calo_total) * 0.4) / 4

                protein = ((calo_total) * 0.3) / 4

                fats = ((calo_total) * 0.3) / 9

                # suggestion = hill_climbing(protein, cabohydrate, fats)

                # suggestion = pd.DataFrame(suggestion).to_html(header=False, index=False)

            ################# ====> LOSE WEIGHT <======= ###############
            if goal_radio == "L":
                weight_loss = np.abs(weight - weight_goal)

                # weight_change_rate = 1.6  # (kg with man)

                calories_loss_each_day = (float(weight_change_rate) / 7) * 7716.17

                num_of_weeks_to_loss = float(weight_loss) / float(weight_change_rate)
                calo_total = tdde - calories_loss_each_day
                date_goal = date.today() + relativedelta(weeks=+num_of_weeks_to_loss)
                date_goal = date_goal.strftime("%d/%m/%Y")

                if calo_total < 1800:
                    comment_calo = "Based on your entries, our equations came up with a target under 1800 Calories. Eating under 1800 Calories per day is generally considered extreme, and you should consult a health professional before doing so."
                else:
                    date_goal_print = (
                        "You should reach your goal in {} to gain {} kg !".format(
                            date_goal, weight_goal
                        )
                    )

                    cabohydrate = (calo_total * 0.4) / 4

                    protein = (calo_total * 0.4) / 4

                    fats = (calo_total * 0.2) / 9

                    # suggestion = hill_climbing(protein, cabohydrate, fats)

                    # suggestion = pd.DataFrame(suggestion).to_html(
                    # header=False, index=False
                    # )

            ############# ====> MAINTAIN <======= ###############
            if goal_radio == "M":
                calo_total = tdde
                protein = weight * float(activity_level)
                fats = weight * 1.0
                cabohydrate = (tdde * 0.55) / 4

                # suggestion = hill_climbing(protein, cabohydrate, fats)

                # suggestion = pd.DataFrame(suggestion).to_html(header=False, index=False)

            ############# ====> BUILD MUSCLE <======= ###############
            if goal_radio == "B":

                # ```You’re here to pack on size, so you’ll need to
                #    increase the number of calories
                #    you eat each day by 15% more than your TDEE```

                Calories_for_musclebuilding = tdde * 0.15

                calo_total = tdde + Calories_for_musclebuilding

                if bodyType == "ENDOMORPH BODY":

                    cabohydrate = (calo_total * 0.25) / 4

                    protein = (calo_total * 0.40) / 4

                    fats = (calo_total * 0.35) / 9

                    # suggestion = hill_climbing(protein, cabohydrate, fats)

                    # suggestion = pd.DataFrame(suggestion).to_html(
                    #     header=False, index=False
                    # )

                if bodyType == "ECTOMORPH BODY":

                    cabohydrate = (calo_total * 0.40) / 4

                    protein = (calo_total * 0.30) / 4

                    fats = (calo_total * 0.30) / 9

                    # suggestion = hill_climbing(protein, cabohydrate, fats)

                    # suggestion = pd.DataFrame(suggestion).to_html(
                    #     header=False, index=False
                    # )

                if bodyType == "MESOMORPH BODY TYPE":

                    cabohydrate = (calo_total * 0.40) / 4

                    protein = (calo_total * 0.35) / 4

                    fats = (calo_total * 0.25) / 9

                    # suggestion = hill_climbing(protein, cabohydrate, fats)

                    # suggestion = pd.DataFrame(suggestion).to_html(
                    #     header=False, index=False
                    # )

        #########3======> ENDING (BUILD MUSCLE FOR MALE) <===========###########

        if gender == "Female":

            bmi = weight / ((height * 0.01) * (height * 0.01))
            if bmi < 18.5:
                bmi_comment = "Underweight"
            elif bmi >= 18.5 and bmi < 24.9:
                bmi_comment = "Normal"
            elif bmi >= 25 and bmi < 29.9:
                bmi_comment = "Overweight"
            else:
                bmi_comment = "Very overweight"

            bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) - 161

            tdde = bmr * float(activity_level)

            bodyFat = (
                495
                / (
                    1.29579
                    - 0.35004 * math.log10(float(waist) + float(hip) - float(neck))
                    + 0.22100 * math.log10(float(height))
                )
                - 450
            )

            ############# ADVICE FOR USER ###############
            if bodyFat > 10 and bodyFat <= 13:
                comment = "Essential fat"
            elif bodyFat > 13 and bodyFat <= 20:
                comment = "Athletes"
            elif bodyFat > 20 and bodyFat <= 24:
                comment = "Fitness"
            elif bodyFat > 24 and bodyFat <= 31:
                comment = "Average"
            elif bodyFat > 31:
                comment = "Obese"
            ##################### - GAIN WEIGHT -#################

            if goal_radio == "G":
                weight_add = np.abs(weight - weight_goal)
                calories_gain_each_day = (float(weight_change_rate) / 7) * 7716.17
                num_of_weeks_to_gain = round(
                    float(weight_add) / float(weight_change_rate)
                )
                calo_total = tdde + calories_gain_each_day
                date_goal = date.today() + relativedelta(weeks=+num_of_weeks_to_gain)
                date_goal = date_goal.strftime("%d/%m/%Y")

                date_goal_print = (
                    "You should reach your goal in {} to gain {} kg !".format(
                        date_goal, weight_goal
                    )
                )

                cabohydrate = ((calo_total) * 0.4) / 4

                protein = ((calo_total) * 0.3) / 4

                fats = ((calo_total) * 0.3) / 9

            ################# ====> LOSE WEIGHT <======= ###############
            if goal_radio == "L":
                weight_loss = np.abs(weight - weight_goal)

                # weight_change_rate = 1.6  # (kg/week with man but for women is about 1kg/week)

                calories_loss_each_day = (float(weight_change_rate) / 7) * 7716.17

                num_of_weeks_to_loss = round(
                    float(weight_loss) / float(weight_change_rate)
                )
                date_goal = date.today() + relativedelta(weeks=+num_of_weeks_to_loss)
                date_goal = date_goal.strftime("%d/%m/%Y")
                calo_total = tdde - calories_loss_each_day

                if calo_total < 1200:
                    comment_calo = "Based on your entries, our equations came up with a target under 1200 Calories. Eating under 1200 Calories per day is generally considered extreme, and you should consult a health professional before doing so."
                else:
                    date_goal_print = (
                        "You should reach your goal in {} to gain {} kg !".format(
                            date_goal, weight_goal
                        )
                    )

                    cabohydrate = (calo_total * 0.4) / 4

                    protein = (calo_total * 0.4) / 4

                    fats = (calo_total * 0.2) / 9

            ############# ====> MAINTAIN <======= ###############
            if goal_radio == "M":
                calo_total = tdde
                protein = weight * float(activity_level)
                fats = weight * 1.0
                cabohydrate = (tdde * 0.55) / 4

            ############# ====> BUILD MUSCLE <======= ###############
            if goal_radio == "B":

                # ```You’re here to pack on size, so you’ll need to
                #    increase the number of calories
                #    you eat each day by 15% more than your TDEE```

                Calories_for_musclebuilding = tdde * 0.15

                calo_total = tdde + Calories_for_musclebuilding

                if bodyType == "ENDOMORPH BODY":

                    cabohydrate = (calo_total * 0.25) / 4

                    protein = (calo_total * 0.40) / 4

                    fats = (calo_total * 0.35) / 9

                    # suggestion = hill_climbing(protein, cabohydrate, fats)

                    # suggestion = pd.DataFrame(suggestion).to_html(
                    #     header=False, index=False
                    # )

                if bodyType == "ECTOMORPH BODY":

                    cabohydrate = (calo_total * 0.40) / 4

                    protein = (calo_total * 0.30) / 4

                    fats = (calo_total * 0.30) / 9

                    # suggestion = hill_climbing(protein, cabohydrate, fats)

                    # suggestion = pd.DataFrame(suggestion).to_html(
                    #     header=False, index=False
                    # )

                if bodyType == "MESOMORPH BODY TYPE":

                    cabohydrate = (calo_total * 0.40) / 4

                    protein = (calo_total * 0.35) / 4

                    fats = (calo_total * 0.25) / 9

                    # suggestion = hill_climbing(protein, cabohydrate, fats)

                    # suggestion = pd.DataFrame(suggestion).to_html(
                    #     header=False, index=False
                    # )

        #########3======> ENDING (BUILD MUSCLE FOR FEMALE) <===========###########

        # @app.route("/", methods=["GET", "POST"])
        # def data():

        #     suggestion = ""
        #     if (
        #         request.method == "POST"
        #         and "fat" in request.form
        #         and "protein" in request.form
        #         and "carbon" in request.form
        #     ):
        #         fat = float(request.form.get("fat"))
        #         protein = float(request.form.get("protein"))
        #         carbon = float(request.form.get("carbon"))
        #         suggestion = hill_climbing(protein, carbon, fat)
        #         suggestion = pd.DataFrame(suggestion).to_html(header=False, index=False)
        # type(protein)
        import csv

        with open("nutri.csv", mode="w") as nutri_file:
            nutri_writer = csv.writer(
                nutri_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            nutri_writer.writerow(
                [
                    bmi,
                    bodyFat,
                    calo_total,
                    protein,
                    cabohydrate,
                    fats,
                    allergy,
                    bmi_comment,
                    comment,
                ]
            )

        df1 = pd.read_csv("nutri.csv")
        # print(df1[0])

    if request.method == "POST" and request.form["action"] == "✔ Apply these settings":
        num_meal = int(request.form["num-meals-selector"])
        df1 = pd.read_csv("nutri.csv")
        bmi = df1.columns.values[0]
        bodyFat = df1.columns.values[1]
        calo_total = df1.columns.values[2]
        protein = float(df1.columns.values[3])
        # cabohydrate = float("".join(df1.columns.values[4].rsplit(".", 1)))
        fats = float(df1.columns.values[5])
        allergy = df1.columns.values[6]
        bmi_comment = df1.columns.values[7]
        comment = df1.columns.values[8]
        # print(cabohydrate)
        if df1.columns.values[2].count(".") > 1:
            calo_total = float("".join(df1.columns.values[2].rsplit(".", 1)))
        else:
            calo_total = float(df1.columns.values[2])

        if df1.columns.values[3].count(".") > 1:
            protein = float("".join(df1.columns.values[3].rsplit(".", 1)))
        else:
            protein = float(df1.columns.values[3])

        if df1.columns.values[4].count(".") > 1:
            cabohydrate = float("".join(df1.columns.values[4].rsplit(".", 1)))
        else:
            cabohydrate = float(df1.columns.values[4])

        if df1.columns.values[5].count(".") > 1:
            fats = float("".join(df1.columns.values[5].rsplit(".", 1)))
        else:
            fats = float(df1.columns.values[5])

        if (protein > 1) and (cabohydrate > 1) and (fats > 1):
            suggestion = hill_climbing(num_meal, protein, cabohydrate, fats, allergy)
            print(suggestion)
            suggestion = pd.DataFrame(suggestion)
            print(suggestion)
            print(protein)
            print(cabohydrate)
            print(fats)
            suggestion = suggestion.rename(
                {1: "Protein", 2: "Carbon", 3: "Fat"}, axis=1
            )
            print(suggestion)
            suggestion = pd.merge(
                suggestion, df, how="inner", on=["Protein", "Carbon", "Fat"]
            ).drop_duplicates(subset=["Protein", "Carbon", "Fat"])
            # suggestion = suggestion.to_html(header=True, index=False)
            # suggestion = suggestion["title"]
            # # print(suggestion)
            # suggestion = suggestion.sort_values(by=['calories'])
            # .to_html(header=True, index=True)
            # dishType = suggestion['dishTypes']
            dishType = suggestion[
                suggestion["dishTypes"] == "lunch,main course,main dish,dinner"
            ]
            firstMeal = dishType.iloc[0]
            secondMeal = dishType.iloc[1]
            print(suggestion)
            print(dishType)
            print(dishType.iloc[0:2, :])
            firstMeal_name = firstMeal.get(key="title")
            firstMeal_serving = firstMeal.get(key="weightPerServing")
            firstMeal_calo = firstMeal.get(key="calories")
            firstMeal_pro = firstMeal.get(key="Protein")
            firstMeal_Carb = firstMeal.get(key="Carbon")
            firstMeal_fat = firstMeal.get(key="Fat")
            firstMeal_ingredients = firstMeal.get(key="ingredients")
            secondMeal_name = secondMeal.get(key="title")
            secondMeal_serving = secondMeal.get(key="weightPerServing")
            secondMeal_calo = secondMeal.get(key="calories")
            secondMeal_pro = secondMeal.get(key="Protein")
            secondMeal_Carb = secondMeal.get(key="Carbon")
            secondMeal_fat = secondMeal.get(key="Fat")
            secondMeal_ingredients = secondMeal.get(key="ingredients")
            others = suggestion[~suggestion.isin(dishType.iloc[0:2, :])].dropna(
                subset=["Protein"]
            )
            print(others)
            print(suggestion[~suggestion.isin(dishType.iloc[0:2, :])])
            x = len(suggestion)
            if num_meal >= 2 and x >= 3:
                thirdMeal = others.iloc[0]
                thirdMeal_name = thirdMeal.get(key="title")
                thirdMeal_serving = thirdMeal.get(key="weightPerServing")
                thirdMeal_calo = thirdMeal.get(key="calories")
                thirdMeal_pro = thirdMeal.get(key="Protein")
                thirdMeal_Carb = thirdMeal.get(key="Carbon")
                thirdMeal_fat = thirdMeal.get(key="Fat")
                thirdMeal_ingredients = thirdMeal.get(key="ingredients")

            if num_meal >= 3 and x >= 4:
                fourthMeal = others.iloc[1]
                fourthMeal_name = fourthMeal.get(key="title")
                fourthMeal_serving = fourthMeal.get(key="weightPerServing")
                fourthMeal_calo = fourthMeal.get(key="calories")
                fourthMeal_pro = fourthMeal.get(key="Protein")
                fourthMeal_Carb = fourthMeal.get(key="Carbon")
                fourthMeal_fat = fourthMeal.get(key="Fat")
                fourthMeal_ingredients = fourthMeal.get(key="ingredients")
            if num_meal >= 4 and x >= 5:
                fifthMeal = others.iloc[2]
                fifthMeal_name = fifthMeal.get(key="title")
                fifthMeal_serving = fifthMeal.get(key="weightPerServing")
                fifthMeal_calo = fifthMeal.get(key="calories")
                fifthMeal_pro = fifthMeal.get(key="Protein")
                fifthMeal_Carb = fifthMeal.get(key="Carbon")
                fifthMeal_fat = fifthMeal.get(key="Fat")
                fifthMeal_ingredients = fifthMeal.get(key="ingredients")
            if num_meal >= 2 and x < 3:
                sorry = "Sorry! Out of food "
            if num_meal >= 3 and x < 4:
                sorry = "Sorry! Out of food "
            if num_meal >= 4 and x < 5:
                sorry = "Sorry! Out of food "

            sum_calo = round(suggestion["calories"].sum(), 2)
            sum_carbs = round(suggestion["Carbon"].sum(), 2)
            sum_fat = round(suggestion["Fat"].sum(), 2)
            sum_pro = round(suggestion["Protein"].sum(), 2)
            sum_Cholesterol = round(suggestion["Cholesterol/mg"].sum(), 2)
            sum_fiber = round(suggestion["Fiber/g"].sum(), 2)
            sum_sugar = round(suggestion["Sugar/g"].sum(), 2)
            sum_VinD = round(suggestion["Vitamin D/?g"].sum(), 2)
            sum_iron = round(suggestion["Iron/mg"].sum(), 2)
            sum_calcium = round(suggestion["Calcium/mg"].sum(), 2)
            sum_potassium = round(suggestion["Potassium/mg"].sum(), 2)
            sum_sodium = round(suggestion["Sodium/mg"].sum(), 2)
            percentCarbs = sum_carbs / (sum_carbs + sum_pro + sum_fat)
            percentFat = sum_fat / (sum_carbs + sum_pro + sum_fat)
            percentPro = sum_pro / (sum_carbs + sum_pro + sum_fat)
            data = [
                ("carbs", percentCarbs),
                ("protein", percentPro),
                ("fat", percentFat),
            ]
            labels = [row[0] for row in data]
            values = [row[1] for row in data]

    return render_template(
        "index.html",
        suggestion=suggestion,
        bmi=bmi,
        bmi_comment=bmi_comment,
        bodyFat=bodyFat,
        tdde=tdde,
        goal=goal,
        bodyType=bodyType,
        calo_total=calo_total,
        cabohydrate=cabohydrate,
        protein=protein,
        fats=fats,
        comment=comment,
        comment_calo=comment_calo,
        ideal_fat=ideal_fat,
        date_goal=date_goal,
        date_goal_print=date_goal_print,
        sum_calo=sum_calo,
        sum_carbs=sum_carbs,
        sum_fat=sum_fat,
        sum_pro=sum_pro,
        labels=labels,
        values=values,
        num_meal=num_meal,
        sum_Cholesterol=sum_Cholesterol,
        sum_fiber=sum_fiber,
        sum_sugar=sum_sugar,
        sum_VinD=sum_VinD,
        sum_iron=sum_iron,
        sum_calcium=sum_calcium,
        sum_potassium=sum_potassium,
        sum_sodium=sum_sodium,
        dishType=dishType,
        firstMeal=firstMeal,
        secondMeal=secondMeal,
        thirdMeal=thirdMeal,
        fourthMeal=fourthMeal,
        fifthMeal=fifthMeal,
        firstMeal_calo=firstMeal_calo,
        firstMeal_Carb=firstMeal_Carb,
        firstMeal_fat=firstMeal_fat,
        firstMeal_pro=firstMeal_pro,
        secondMeal_calo=secondMeal_calo,
        secondMeal_Carb=secondMeal_Carb,
        secondMeal_fat=secondMeal_fat,
        secondMeal_pro=secondMeal_pro,
        thirdMeal_calo=thirdMeal_calo,
        thirdMeal_Carb=thirdMeal_Carb,
        thirdMeal_fat=thirdMeal_fat,
        thirdMeal_pro=thirdMeal_pro,
        fourthMeal_calo=fourthMeal_calo,
        fourthMeal_Carb=fourthMeal_Carb,
        fourthMeal_fat=fourthMeal_fat,
        fourthMeal_pro=fourthMeal_pro,
        fifthMeal_calo=fifthMeal_calo,
        fifthMeal_Carb=fifthMeal_Carb,
        fifthMeal_fat=fifthMeal_fat,
        fifthMeal_pro=fifthMeal_pro,
        firstMeal_ingredients=firstMeal_ingredients,
        secondMeal_ingredients=secondMeal_ingredients,
        thirdMeal_ingredients=thirdMeal_ingredients,
        fourthMeal_ingredients=fourthMeal_ingredients,
        fifthMeal_ingredients=fifthMeal_ingredients,
        firstMeal_name=firstMeal_name,
        firstMeal_serving=firstMeal_serving,
        secondMeal_name=secondMeal_name,
        thirdMeal_name=thirdMeal_name,
        fourthMeal_name=fourthMeal_name,
        fifthMeal_name=fifthMeal_name,
        secondMeal_serving=secondMeal_serving,
        thirdMeal_serving=thirdMeal_serving,
        fourthMeal_serving=fourthMeal_serving,
        fifthMeal_serving=fifthMeal_serving,
        sorry=sorry,
        x=x,
    )

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True, port=4000)
