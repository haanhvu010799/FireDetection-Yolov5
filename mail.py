import paho.mqtt.client as mqtt
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
def guiemail():
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "kltnhaanh@gmail.com"  # Enter your address
    receiver_email = "haanhvu010799@gmail.com"  # Enter receiver address
    password = "Vuhaanh1999"
    message = """\
    Subject: Canh Bao hoa Hoan

    Co hoa hoan dang xay ra, vui long kiem tra camera."""

    context = ssl.create_default_context()
    print("Gui email thanh cong")
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("topic/haanh")

def on_message(client, userdata, msg):
    if msg.payload.decode() == "F":
        guiemail()
        client.disconnect()
    
client = mqtt.Client()
client.connect("192.168.1.7",1883,60)

client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()


