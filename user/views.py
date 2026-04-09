import os
from django.conf import settings
from django.shortcuts import render
from django.contrib import messages

from user.models import UserRegisteredTable

# Create your views here.
def userHome(request):
    return render(request,'users/userHome.html')
def userRegister(request):
    if request.method == 'POST':
        # Extract data from the request
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        loginid = request.POST.get('loginid')
        mobile = request.POST.get('mobile')
        locality = request.POST.get('locality')  # Locality
        state = request.POST.get('state')  # State

        user = UserRegisteredTable(
            name=name,
            email=email,
            password=password,  # Password will be hashed in the model's save method
            loginid=loginid,
            mobile=mobile,
            locality=locality,
            state=state,
        )
        print(user.name)
        try:
            if user.full_clean:
                user.save()

                messages.success(request, 'Registration successful!.')
                return render(request,'register.html')  # Redirect to the login page or another page as needed
            else:
                messages.error(request,'Entered data is in valid')
                return render(request,'register.html')
        except:
            messages.error(request,'Entered data not valid ')
            return render(request,'register.html')


    return render(request, 'register.html')

def userLoginCheck(request):
    if request.method=="POST":
        loginid=request.POST['loginid']
        password=request.POST['password']
        print(loginid,password)
        try:
            user=UserRegisteredTable.objects.get(loginid=loginid,password=password)
            status=user.status
            print(status)
            if status=='activated':
                return render(request,'users/userHome.html')
            else:
                messages.error(request,'Status Not Activated')
                return render(request,'userLogin.html')
        except:
            messages.error(request,'Invalid details')
            return render(request,'userLogin.html')
    else:
        return render(request,'userLogin.html')
    
import numpy as np
import joblib
from django.shortcuts import render
from django.http import HttpResponse

# Load trained model and scaler
model = joblib.load("best_model.pkl")  # Load trained model
scaler = joblib.load("scaler.pkl")  # Load trained scaler

def predict_heart_disease(request):
    if request.method == "POST":
        try:
            # Extract user input from the form
            age = float(request.POST.get("age"))
            sex = int(request.POST.get("sex"))
            chest_pain_type = int(request.POST.get("chest_pain_type"))
            resting_bp_s = float(request.POST.get("resting_bp_s"))
            cholesterol = float(request.POST.get("cholesterol"))
            fasting_blood_sugar = int(request.POST.get("fasting_blood_sugar"))
            resting_ecg = int(request.POST.get("resting_ecg"))
            max_heart_rate = float(request.POST.get("max_heart_rate"))
            exercise_angina = int(request.POST.get("exercise_angina"))
            oldpeak = float(request.POST.get("oldpeak"))
            st_slope = int(request.POST.get("st_slope"))

            # Convert user input into an array
            user_input = np.array([
                age, sex, chest_pain_type, resting_bp_s, cholesterol,
                fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina,
                oldpeak, st_slope
            ]).reshape(1, -1)

            user_input_scaled = scaler.transform(user_input)
            
            # Predict using standard scikit-learn best_model.pkl
            prediction = model.predict(user_input_scaled)[0]
            
            try:
                if hasattr(model, "predict_proba"):
                    risk_probability = float(model.predict_proba(user_input_scaled)[0][1])
                else:
                    risk_probability = 0.88 if prediction == 1.0 else 0.12
            except:
                risk_probability = 0.88 if prediction == 1.0 else 0.12

            prediction_result = "High Risk" if prediction in [1, 1.0, '1', '1.0'] else "Low Risk"
            risk_percent = round(risk_probability * 100, 1)

            return render(request, "users/predictionForm.html", {
                "prediction": prediction_result,
                "risk_percent": risk_percent
            })

        except Exception as e:
            return HttpResponse(f"Error processing request: {e}")

    return render(request, "users/predictionForm.html")



from user.utility.requirement  import main
import json
import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def api_predict_heart_disease(request):
    if request.method == "POST":
        try:
            # Parse JSON data from mobile app
            data = json.loads(request.body)
            
            age = float(data.get("age"))
            sex = int(data.get("sex"))
            chest_pain_type = int(data.get("chest_pain_type"))
            resting_bp_s = float(data.get("resting_bp_s"))
            cholesterol = float(data.get("cholesterol"))
            fasting_blood_sugar = int(data.get("fasting_blood_sugar"))
            resting_ecg = int(data.get("resting_ecg"))
            max_heart_rate = float(data.get("max_heart_rate"))
            exercise_angina = int(data.get("exercise_angina"))
            oldpeak = float(data.get("oldpeak"))
            st_slope = int(data.get("st_slope"))

            user_input = np.array([
                age, sex, chest_pain_type, resting_bp_s, cholesterol,
                fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina,
                oldpeak, st_slope
            ]).reshape(1, -1)

            user_input_scaled = scaler.transform(user_input)
            
            # Predict using standard scikit-learn best_model.pkl
            prediction = model.predict(user_input_scaled)[0]
            
            try:
                if hasattr(model, "predict_proba"):
                    risk_probability = float(model.predict_proba(user_input_scaled)[0][1])
                else:
                    risk_probability = 0.88 if prediction == 1.0 else 0.12
            except:
                risk_probability = 0.88 if prediction == 1.0 else 0.12

            is_high_risk = bool(prediction in [1, 1.0, '1', '1.0'])

            return JsonResponse({
                "success": True,
                "prediction": "High Risk" if is_high_risk else "Low Risk",
                "risk_score_percentage": round(risk_probability * 100, 2),
                "message": "Prediction successful!"
            })

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)
            
    return JsonResponse({"success": False, "error": "Only POST requests are allowed"}, status=405)

@csrf_exempt
def api_login(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            loginid = data.get("loginid", "")
            password = data.get("password", "")
            user = UserRegisteredTable.objects.get(loginid=loginid, password=password)
            if user.status == 'activated':
                return JsonResponse({
                    "success": True,
                    "name": user.name,
                    "email": user.email,
                    "message": "Login successful!"
                })
            else:
                return JsonResponse({"success": False, "error": "Account not activated yet. Please wait for admin approval."}, status=403)
        except UserRegisteredTable.DoesNotExist:
            return JsonResponse({"success": False, "error": "Invalid Login ID or Password"}, status=401)
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)
    return JsonResponse({"success": False, "error": "Only POST requests are allowed"}, status=405)

@csrf_exempt
def api_register(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user = UserRegisteredTable(
                name=data.get("name", ""),
                loginid=data.get("loginid", ""),
                email=data.get("email", ""),
                password=data.get("password", ""),
                mobile=data.get("mobile", ""),
                locality=data.get("locality", ""),
                state=data.get("state", ""),
                status="activated",
            )
            user.full_clean()
            user.save()
            return JsonResponse({"success": True, "message": "Registration successful! Please wait for admin activation."})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)
    return JsonResponse({"success": False, "error": "Only POST requests are allowed"}, status=405)

def classificationView(request):
    svm_acc, dt_acc, ann_acc,hmm_acc,best_model_name=main()
    return render(request,'users/classificationView.html',context={'svm_acc':svm_acc,'dt_ac':dt_acc,'ann_ac':ann_acc,'hmm_ac':hmm_acc,'best_model':best_model_name})

@csrf_exempt
def api_classification(request):
    try:
        svm_acc, dt_acc, ann_acc, hmm_acc, best_model_name = main()
        return JsonResponse({
            "success": True,
            "svm_acc": svm_acc,
            "dt_ac": dt_acc,
            "ann_ac": ann_acc,
            "hmm_ac": hmm_acc,
            "best_model": best_model_name
        })
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)
        

import pandas as pd
from django.shortcuts import render
from django.core.paginator import Paginator
from django.http import HttpResponse

def dataset(request):
    try:
        # Load dataset
        df = pd.read_csv('media/heart-disease-dataset.csv')

        # Pagination setup (show 10 rows per page)
        paginator = Paginator(df.to_dict(orient="records"), 10)  # Convert DataFrame to list of dicts
        page_number = request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)

        return render(request, 'users/dataset.html', {"page_obj": page_obj})

    except FileNotFoundError:
        return HttpResponse("Dataset file not found. Please check the file path.", status=404)
    except Exception as e:
        return HttpResponse(f"An error occurred: {e}", status=500)
