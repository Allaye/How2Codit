from django import forms
from django.contrib.auth.models import User
from progress.models import Profile



class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())
    sex = forms.CharField()

    class Meta:
        model = User
        fields = ('username', 'email', 'password')


class UserDetailsUpdate(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ('image', 'website', 'sex')

class UserLogin(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput())
