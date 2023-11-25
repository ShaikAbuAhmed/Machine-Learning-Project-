from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

# from users.Algorithm.Algorithm import Algorithms
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

# algo = Algorithms()


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)

            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account is not yet activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHome.html', {})

def Viewdata(request):
    import os
    import pandas as pd
    from django.conf import settings
    path = os.path.join(settings.MEDIA_ROOT,'HotelReviews.csv')
    df = pd.read_csv(path)
    print(df)
    df = df.to_html()
    return render(request, 'users/Viewdata.html', {'data': df})

def Prediction(request):
    if request.method=="POST":
        review = request.POST.get('review')
        print("Reviews is ", review)
        from .utility import predictions
        sentiment = predictions.get_tweet_sentiment(review)
        return render(request, "users/predict_form.html", {"review": review, "sentiment": sentiment})
    else:
        return render(request, "users/predict_form.html",{})

def Classification(request):
    from .utility import classification
    accuracy, precession, recall, f1_score = classification.build_naive_model()
    lg_accuracy, lg_precession, lg_recall, lg_f1_score = classification.build_logistic_model()
    rf_accuracy, rf_precession, rf_recall, rf_f1_score = classification.build_random_forest_model()
    svm_accuracy, svm_precession, svm_recall, svm_f1_score = classification.build_svm_model()
    dt_accuracy, dt_precision, dt_recall, dt_f1_score = classification.build_decision_tree_model()
    nn_accuracy, nn_precision, nn_recall, nn_f1_score = classification.build_neural_network_model()

    nb = {'accuracy': accuracy, 'precession': precession, "recall": recall,"f1_score": f1_score}
    lg = {'lg_accuracy': lg_accuracy, 'lg_precession': lg_precession, "lg_recall": lg_recall,"lg_f1_score":lg_f1_score}
    rf = {'rf_accuracy': rf_accuracy, 'rf_precession': rf_precession, "rf_recall": rf_recall,"rf_f1_score": rf_f1_score}
    svm = {'svm_accuracy': svm_accuracy, 'svm_precession': svm_precession, "svm_recall": svm_recall,"svm_f1_score": svm_f1_score}
    dt = {'dt_accuracy':dt_accuracy, 'dt_precision':dt_precision, 'dt_recall':dt_recall, 'dt_f1_score':dt_f1_score}
    nn = {'nn_accuracy':nn_accuracy, 'nn_precision':nn_precision, 'nn_recall':nn_recall, 'nn_f1_score':nn_f1_score}
    return render(request, "users/Classification.html", {"nb": nb, "lg": lg,"rf": rf, "svm": svm, "dt":dt, "nn":nn})
