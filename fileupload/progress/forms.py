from django import forms
from django import forms
from django.contrib.auth.models import User
from django import forms
from django.contrib.auth.models import User
from django import forms
from progress.models import UserProfile



class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())
    
    sex = forms.CharField()

    class Meta:
        model = User
        fields = ('username', 'email', 'password')


class UserImageUpdate(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ('image', 'website')
