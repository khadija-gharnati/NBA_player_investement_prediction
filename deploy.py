from flask import Flask, render_template, request
import joblib
import pandas as pd

model = joblib.load('Prediction_model')
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return (render_template('home.html', result= ' ' ))

@app.route('/result', methods=['POST', 'GET'])
def result():

    # récupérer les valeurs inserées
    Name = request.form['Player_name']
    GP = request.form['GP']
    MIN = request.form['MIN']
    PTS = request.form['PTS']
    FTM = request.form['FTM']
    OREB = request.form['OREB']
    DREB = request.form['DREB']
    TOV = request.form['TOV']
    STL = request.form['STL']
    FG = request.form['FG%']
    BLK = request.form['BLK']

    # transformer les params
    player_params = pd.DataFrame([[GP, MIN, PTS, FTM, OREB, DREB, TOV, STL, FG, BLK]],
                                   columns=['GP', 'MIN', 'PTS', 'FTM', 'OREB', 'DREB', 'TOV', 'STL', 'FG%', 'BLK'],
                                   dtype=float)
    if Name == '':
        Name = "This player"

    # prédire le résultat
    prediction = model.predict(player_params.values)

    # Définition de messages à afficher
    positif_result = Name + " is worth investing in"
    negatif_result = Name + " is not worth investing in"

    # afficher un message en fonction du résultat
    if int(prediction) == 1:
        return render_template('home.html',
                                     result=positif_result)
    elif int(prediction) == 0:
        return render_template('home.html',
                                     result=negatif_result)





if __name__ == '__main__':
    app.run()