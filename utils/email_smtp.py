import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import yaml

# 读取config.yaml文件
with open('config.yaml', 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 邮件配置
email = config['email']
sender_email = email['sender_email']
sender_password = email['sender_password']
receiver_email = email['receiver_email']
smtp_server = email['smtp_server']
port = email['port']

def send_email(image, subject, message):
    # 创建邮件对象
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # 添加邮件正文
    body = message
    msg.attach(MIMEText(body, "plain"))

    # 添加人脸图像作为附件
    image_attachment = MIMEImage(image)
    image_attachment.add_header("Content-Disposition", "attachment", filename="face.jpg")
    msg.attach(image_attachment)

    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)

import os

def send_video(video_data, video_path, subject, message):
    # 检查视频文件大小
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # 计算文件大小，单位为 MB
    if file_size > 100:
        subject = "录像发送失败"
        message = "录像发送失败，请在原设备中查看。"
        
        # 创建邮件对象
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject

        # 添加邮件正文
        body = message
        msg.attach(MIMEText(body, "plain"))
    else:
        # 创建邮件对象
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject

        # 添加邮件正文
        body = message
        msg.attach(MIMEText(body, "plain"))

        # 添加视频文件作为附件
        video_attachment = MIMEApplication(video_data, _subtype="octet-stream")
        video_attachment.add_header("Content-Disposition", "attachment", filename="video.avi")
        msg.attach(video_attachment)

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        print(f"邮件发送失败: {e}")