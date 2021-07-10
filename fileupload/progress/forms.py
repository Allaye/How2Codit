from django import forms
from django.forms import fields 
from progress.models import User



class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['name', 'email',]

