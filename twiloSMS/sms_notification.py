import os 
from twilio.rest import Client


def send_sms_notification(message: str) -> None:
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    client = Client(account_sid, auth_token)
    message = client.messages.create(to=os.environ.get('MY_PHONE_NUMBER'), from_=os.environ.get('TWILIO_PHONE_NUMBER'), body=message)
    return None


if __name__ == "main":

    message = "please check your daily sms commitment"
    send_sms_notification(message)

