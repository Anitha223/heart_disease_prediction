from django.shortcuts import render
from django.contrib import messages

from user.models import UserRegisteredTable

# Create your views here.

def adminLoginCheck(request):
    if request.method=="POST":
        login_id=request.POST['loginid']
        password=request.POST['password']

        if login_id=='admin' and password=='admin':
            return render(request,'admin/adminHome.html')
        else:
            messages.error(request,'Invalid details')
            return render(request,'adminLogin.html')
    else:
        return render(request,'adminLogin.html')
        
def adminHome(request):
    return render(request,'admin/adminHome.html')

def userDetails(request):
    user=UserRegisteredTable.objects.all()
    return render(request,'admin/userDetails.html',{'user':user})

def activateUser(request):
    loginid=request.GET['loginid']
    user=UserRegisteredTable.objects.get(loginid=loginid)
    user.status='activated'
    user.save()
    userr=UserRegisteredTable.objects.all()
    return render(request,'admin/userDetails.html',{'user':userr})

def deactivateUser(request):
    loginid=request.GET['loginid']
    user=UserRegisteredTable.objects.get(loginid=loginid)
    user.status='Waiting'
    user.save()
    userr=UserRegisteredTable.objects.all()
    return render(request,'admin/userDetails.html',{'user':userr})


from user.utility.requirement  import main
def adminclassificationView(request):
    svm_acc, dt_acc, ann_acc, hmm_acc, hybrid_acc, best_model_name = main()
    return render(request, 'admin/adminClassificationView.html', context={
        'svm_acc': round(svm_acc * 100, 2),
        'dt_ac': round(dt_acc * 100, 2),
        'ann_ac': round(ann_acc * 100, 2),
        'hmm_ac': round(hmm_acc * 100, 2),
        'hybrid_acc': round(hybrid_acc * 100, 2),
        'best_model': best_model_name
    })


            

