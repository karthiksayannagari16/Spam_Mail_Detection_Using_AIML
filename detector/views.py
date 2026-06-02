from django.shortcuts import render
import pickle
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model = pickle.load(
    open(
        os.path.join(BASE_DIR,'spam_model.pkl'),
        'rb'
    )
)

vectorizer = pickle.load(
    open(
        os.path.join(BASE_DIR,'vectorizer.pkl'),
        'rb'
    )
)

def home(request):

    result = ""

    if request.method == "POST":

        email = request.POST.get("email")

        transformed_email = vectorizer.transform(
            [email]
        )

        prediction = model.predict(
            transformed_email
        )

        if prediction[0] == 1:
            result = "✅ HAM MAIL (Safe Email)  It is Safe to Open this Email "
        else:
            result = "⚠️ SPAM MAIL (Spam Email) It is Not Safe to Open this Email"

    return render(
        request,
        'index.html',
        {
            'prediction_text': result
        }
    )