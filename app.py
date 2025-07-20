from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import numpy as np
import io
from xhtml2pdf import pisa
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session

# Load models and encoders
reg_model = joblib.load('score_model.pkl')
clf_model = joblib.load('risk_model.pkl')
risk_encoder = joblib.load('risk_encoder.pkl')
label_encoders = joblib.load('label_encoders.pkl')


# ✅ Haircare Tips Generator
def generate_tips(stress_level, sleep, water, issues):
    tips = []

    if stress_level == 'High':
        tips.append("Try relaxation techniques like yoga or meditation to reduce stress.")
    if sleep < 6:
        tips.append("Aim for at least 7-8 hours of sleep to support hair health.")
    if water < 2:
        tips.append("Drink more water to keep your scalp hydrated.")
    if "Hair Fall" in issues:
        tips.append("Include more protein in your diet to reduce hair fall.")
    if "Dandruff" in issues:
        tips.append("Use anti-dandruff shampoo with natural ingredients.")
    if "Dryness" in issues:
        tips.append("Apply oil regularly and use hydrating hair masks.")
    if "Oily Scalp" in issues:
        tips.append("Avoid over-washing; use mild, sulfate-free shampoos.")
    if not tips:
        tips.append("Your hair health seems good! Maintain your current routine.")

    return tips


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        stress = request.form['stress']
        sleep = float(request.form['sleep'])
        water = float(request.form['water'])
        pollution = request.form['pollution']
        coloring = request.form['coloring']
        issues = request.form.getlist('issues')
        budget = request.form['budget']
        genetics = request.form['genetics']

        stress_encoded = label_encoders['Stress Level'].transform([stress])[0]
        pollution_encoded = label_encoders['Pollution Exposure'].transform([pollution])[0]
        coloring_encoded = label_encoders['Hair Coloring Frequency'].transform([coloring])[0]
        budget_encoded = label_encoders['Hair Care Budget'].transform([budget])[0]
        genetics_encoded = label_encoders['Genetic/Hormonal Wellness'].transform([genetics])[0]

        issue_count = len(issues)

        input_data = np.array([[stress_encoded, sleep, water, pollution_encoded,
                                coloring_encoded, issue_count, budget_encoded, genetics_encoded]])

        score = reg_model.predict(input_data)[0]
        risk = clf_model.predict(input_data)[0]
        risk_label = risk_encoder.inverse_transform([risk])[0]

        result_class = 'good' if score > 40 else 'bad'

        # ✅ Store in session for later PDF download
        tips = generate_tips(stress, sleep, water, issues)
        session['tips'] = tips
        session['score'] = round(score)
        session['risk'] = risk_label

        return redirect(url_for('result', score=round(score), risk=risk_label, result_class=result_class))

    except Exception as e:
        return f"Error: {e}"


@app.route('/result')
def result():
    score = request.args.get('score')
    risk = request.args.get('risk')
    result_class = request.args.get('result_class')
    tips = session.get('tips', [])
    return render_template('result.html', score=score, risk=risk, result_class=result_class, tips=tips)


# ✅ PDF Report Generator
@app.route('/download-pdf')
def download_pdf():
    score = session.get('score')
    risk = session.get('risk')
    tips = session.get('tips', [])

    if score is None or risk is None:
        return "No result data to generate PDF."

    html = render_template('pdf_template.html', score=score, risk=risk, tips=tips)

    pdf = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf)
    pdf.seek(0)

    if pisa_status.err:
        return "Error generating PDF"
    return send_file(pdf, mimetype='application/pdf', as_attachment=True, download_name='HairHealthReport.pdf')


# ✅ Render-compatible server start
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
