#!/usr/bin/env python3

from flask import Flask, render_template, Response, request
import cv2
import time
import os
import numpy as np
import uuid
import logging
import yaml
from utils.email_smtp import send_email, send_video
import threading
from utils.auth import check_auth, authenticate, requires_auth

# 读取config.yaml文件
with open('config.yaml', 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)

video_config = config['video']
fps = float(video_config['fps'])
motion_threshold = int(video_config['motion_threshold'])
output_port = int(video_config['port'])

# 错误记录
logging.basicConfig(filename='running_log.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0) 
        self.classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.color = (0, 255, 0)
        self.detected_faces = set()  # 用于记录已检测到的人脸
        self.last_email_time = 0  # 上次发送邮件的时间戳
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False) # 背景减除器
        self.motion_threshold = motion_threshold # 动作识别阈值，数值越大，需要激活人脸识别的动作就越大
        self.current_frame = None
        self.running = True
        # 以下为录像功能代码
        self.recording = False
        self.output_file = None
        self.recordings_dir = "recordings"
        self.photos_dir = "photos"
        # create folder to save photos and recordings
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        if not os.path.exists(self.photos_dir):
            os.makedirs(self.photos_dir)

    def __del__(self):
        self.video.release()

    def start(self):
        threading.Thread(target=self._update_frame, args=()).start()

    def stop(self):
        self.running = False

    def save_photo(self, image):
        photo_name = f"{uuid.uuid4()}.jpg"
        photo_path = os.path.join(self.photos_dir, photo_name)
        cv2.imwrite(photo_path, image)
        return photo_path

    def start_recording(self):
        if not self.recording:
            self.recording = True
            current_time = int(time.time())
            self.file_path = os.path.join(self.recordings_dir, f"{current_time}.avi")  # 将文件路径存储为类属性
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.output_file = cv2.VideoWriter(self.file_path, fourcc, 20.0, (640, 480))
            except Exception as e:
                logging.error(f"Video saving error: {e}")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.output_file is not None:
                if self.current_frame is not None:
                    try:
                        last_frame = cv2.imdecode(np.frombuffer(self.current_frame, np.uint8), cv2.IMREAD_COLOR)
                        self.output_file.write(last_frame)  # 写入最后一帧
                    except Exception as e:
                        logging.error(f"Video saving error: {e}")
                
                video_path = self.file_path
                self.output_file.release()
                self.output_file = None

                try:
                    # 读取录像文件内容
                    with open(video_path, "rb") as video_file:
                        video_data = video_file.read()

                    # 发送录像
                    subject = "录像视频"
                    message = "手动录像视频内容"
                    send_video(video_data, video_path, subject, message)  # 直接调用send_video函数，传递视频文件的二进制内容
                except Exception as e:
                    logging.error(f"Open video error: {e}")

    def _update_frame(self):
        while self.running:
            self.check_video_stream()
            self.current_frame = self.get_frame()
            if self.recording and self.output_file is not None:
                self.output_file.write(cv2.imdecode(np.frombuffer(self.current_frame, np.uint8), cv2.IMREAD_COLOR))

    def _capture_frame(self):
        success, image = self.video.read()
        if success:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return None
        
    def check_video_stream(self):
        if not self.video.isOpened():
            print("Video stream is not opened, trying to reopen...")
            self.video.release()
            self.video = cv2.VideoCapture(0)
            if self.video.isOpened():
                print("Video stream reopened successfully.")
            else:
                print("Failed to reopen video stream.")

    def get_frame(self):
        success, image = self.video.read()

        num = 0
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
        # 动作检测
        fg_mask = self.bg_subtractor.apply(grey) # 前景掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数（只有在检测到物体运动后进行）
        if np.sum(fg_mask) > self.motion_threshold:
            faceRects = self.classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:  # 大于0则检测到人脸
                for faceRect in faceRects:  # 单独框出每一张人脸
                    num += 1
                    x, y, w, h = faceRect

                    # 画出矩形框
                    origin_image = image.copy()
                    cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), self.color, 2)

                    # 显示当前捕捉到了多少人脸图片了
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, 'num:%d' % num, (x + 30, y - 30), font, 1, (255, 0, 255), 4)

                    # 检查是否是新的人脸
                    current_time = time.time()
                    time_diff = current_time - self.last_email_time
                    if time_diff >= 300:  # 5分钟（300秒）
                        if (x, y, w, h) not in self.detected_faces:
                            # 添加保存照片
                            # face_image = origin_image[y-10:y+h+10, x-10:x+w+10]  # 提取人脸区域，如果想只保存人脸，则把以下图片保存和邮件发送中的origin_image替换为face_image
                            try:
                                photo_path = self.save_photo(origin_image)  # 保存人脸照片
                                # 发送邮件通知
                                send_email(image=cv2.imencode(".jpg", origin_image)[1].tobytes(),
                                           subject="人脸检测邮件通知",
                                           message=f"检测到新的人脸，照片保存在 {photo_path}")
                            except Exception as e:
                                logging.error(f"Image saving error: {e}")

                            # 记录已检测到的人脸
                            self.detected_faces.add((x, y, w, h))
                        # 更新上次发送邮件的时间戳
                        self.last_email_time = current_time

        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

app = Flask(__name__)

camera = VideoCamera()

@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在template.html文件中
    return render_template('template.html')

def gen(camera):
    while True:
        try:
            frame = camera.get_frame()
            time.sleep(np.round(1/fps, 3)) # 这里设置休眠时间，设置合适时长，确保视频流不间断

            # 使用generator函数输出视频流，每次请求输出的content类型是image/jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            logging.error(f"Error while generating video stream frame: {e}")
            time.sleep(1)  # 休眠一段时间，尝试恢复视频流

@app.route("/test_auth")
@requires_auth
def test_auth():
    return "Authenticated"

@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame') 

@app.route('/capture', methods=['POST'])
def capture():
    frame = camera._capture_frame()
    if frame is not None:
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        try:
            photo_path = camera.save_photo(image)  # 保存拍摄的照片
            send_email(image=frame,
                       subject="拍摄照片",
                       message=f"手动拍摄照片内容，照片保存在 {photo_path}")
            return Response(status=200)
        except Exception as e:
            logging.error(f"Photo saving error: {e}")
    else:
        return Response(status=500)
    
@app.route('/start_recording', methods=['POST'])
def start_recording():
    camera.start_recording()
    return Response(status=200)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    camera.stop_recording()
    return Response(status=200)

if __name__ == '__main__':
    camera.start()
    try:
        app.run(host='0.0.0.0', debug=False, port=output_port)
    except KeyboardInterrupt:
        camera.stop()
