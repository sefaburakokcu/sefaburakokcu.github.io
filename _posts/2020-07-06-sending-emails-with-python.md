---
title: "Sending E-mails with Python"
date: 2020-06-07 18:25:00 +0800
categories: [Posts, Python]
tags: [python]
---

## Introduction

Python has several libraries that enable us to design extraordinary projects without too much effort. "email" and "smtplib" are two of them. In this post, I will share my solution for sending e-mails using Python 3.7. 

In my work, I am training different Deep Learning networks and training sometimes takes several days. When I am away from the work, I want to be informed about the training process. Therefore, 
I created this project in order to get accuracy, loss values and output plots. In the next part, I will share the source code for the project and then I will show how to use it in your code with an example.

## Source Code

The project has EmailSender class that supports sending text, images and documents such as pdf and word on e-mail. The source code can be found below and at [github](https://github.com/sefaburakokcu/python-email-sender).

```python
import os
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication


class EmailSender():
    """ 
    This is a class for automatic e-mail sending. 
      
    Parameters
    ----------
    user_email : str
        The user(sender) e-mail adress.
    password : str
        The user(sender)'s pasword for e-mail authentication.
        
    """
    
    def __init__(self, user_email, password):
        """
        The constructor for EmailSender class.
        Parameters
        ----------
        user_email : str
            The user(sender) e-mail adress.
        password : str
            The user(sender)'s pasword for e-mail authentication.
        Returns
        -------
        None.
        """

        self.sender_email = user_email
        self.__password = password
        
        self.server_address = self.get_email_server()
        
    def get_email_server(self):
        """
        The function to obtain e-mail server's address.
        Returns
        -------
        server_address : str
            The e-mail provider's server address.
        """
        provider = (self.sender_email.split("@")[-1]).split(".")[0]
        
        if provider == "outlook":
            server_address = "smtp-mail.outlook.com"
        elif provider == "gmail":
            server_address = "smtp.gmail.com"
        elif provider == "yahoo":
            server_address = "smtp.mail.yahoo.com"
        else:
            print("%s is not supported. Please use outlook, gmail or yahoo e-mail.")
        return server_address
                
        
    def get_msg(self, receiver_email, subject="", message_body="", attachments=[]):
        """
        The funtion to prepare e-mail messages.
        Parameters
        ----------
        receiver_email : str
            The e-mail address of receiver.
        subject : str, optional
            The e-mail subject. The default is "".
        message_body : str, optional
            The body of e-mail message. The default is "".
        attachments : list, optional
            The attachment files. The default is [].
        Returns
        -------
        msg : MIMEMultipart object
            The e-mail message that will be sent.
        """
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        
        msg.attach(MIMEText(message_body, "plain"))
        
        if len(attachments) != 0:
            image_extensions = ["jpg","png","jpeg","JPG"]
            doc_extensions = ["doc","docx","odt","pdf"]
            for attachment in attachments:
                ext = attachment.split(".")[-1]
                if ext in image_extensions:
                    image_data = open(attachment, 'rb').read()
                    msg.attach(MIMEImage(image_data, name=os.path.basename(attachment)))
                elif ext in doc_extensions:    
                     with open(attachment, 'rb') as f:
                         file = MIMEApplication(f.read(),name=os.path.basename(attachment))
                     file['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment)}"'
                     msg.attach(file)
                else:
                    print("Sending %s file is not supported! Please use files with one of the extension in %s." %(ext,(image_extensions+doc_extensions)))
                    
        return msg
    
    def send_email(self, receiver_email, subject="", message_body="", attachments=[]):
        """
        The function to send an e-mail message
        Parameters
        ----------
        receiver_email : str
            The e-mail address of receiver.
        subject : str, optional
            The e-mail subject. The default is "".
        message_body : str, optional
            The body of e-mail message. The default is "".
        attachments : list, optional
            The attachment files. The default is [].
        Returns
        -------
        None.
        """
        msg = self.get_msg(receiver_email, subject, message_body, attachments)
        text = msg.as_string()
        
        try:
            server = smtplib.SMTP_SSL(self.server_address,465)
            server.ehlo()
            server.login(self.sender_email, self.__password)
            server.sendmail(self.sender_email, receiver_email, text)
            server.quit()
            print("Email is sent successfully.")
        except:
            print("An error occured.")

```

## Usage

In order to send an e-mail, first creat email_sender instance from EmailSender object by passing your e-mail address and password. In addition, if you are using your gmail account, you need to turn [less secure apps](https://myaccount.google.com/lesssecureapps) on. Later, pass the receiver email address, the subject and the body text of the e-mail and attachments to send_email method. When the e-mail is sent successfully, it wil print "Email is sent successfully." message on terminal. 

```python

from email_sender import EmailSender
receiver_email = "jane.doe@example.com"
subject = "The first e-mail"
message_body = "This is a body of the message."
attachments = ["image.png", "output.pdf"]

email_sender = EmailSender("john.doe@example.com", "Pswrd!0001")
email_sender.send_email(receiver_email, subject, message_body, attachments)

```

## Sources

* https://towardsdatascience.com/automate-email-with-python-1e755d9c6276
* https://medium.com/@bakiiii/automatic-e-mail-sending-with-python-eb41855119e1
* https://realpython.com/python-send-email/

