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
from datetime import datetime


# 导入线程池并初始化线程池
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=3)  # 最大线程数量，可根据实际需求调整
def send_email_async(image, subject, message):
    """
    异步发送邮件的包装函数
    """
    executor.submit(send_email, image=image, subject=subject, message=message)

try:
    with open('config.yaml', 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logging.error(f"Failed to load config.yaml: {e}")
    raise

def add_timestamp_to_frame(frame):
    """
    在视频帧的左上角添加系统时间。
    """
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 设置字体和位置
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    font_scale = 0.8  # 字体大小
    font_color = (0, 255, 0)  # 字体颜色 (绿色)
    thickness = 1  # 线条厚度
    position = (10, 30)  # 文字的位置 (左上角)

    # 在帧上绘制时间
    cv2.putText(frame, current_time, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    return frame

video_config = config.get('video', {})
fps = float(video_config.get('fps', 30))  # 默认 FPS 为 30
keypoints = int(video_config.get('keypoints', 3))
padding = int(video_config.get('padding', 10))
motion_threshold = int(video_config.get('motion_threshold', 500))
output_port = int(video_config.get('port', 5001))

logging.basicConfig(
    filename='running_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class VideoCamera(object):
    def __init__(self):
        self.lock = threading.Lock()
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置宽度为 640
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度为 480 
        self.classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.color = (0, 255, 0)
        self.last_email_time = 0  # 上次发送邮件的时间戳
        self.last_motion_time = 0 # 上次移动时间戳
        self.motion_detect_interval = 2  # 移动检测间隔时间（单位：秒）
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
        self.stop()
        if self.video.isOpened():
            self.video.release()
        if self.output_file is not None:
            self.output_file.release()

    def start(self):
        threading.Thread(target=self._update_frame, args=()).start()

    def stop(self):
        self.running = False

    def save_photo(self, image):
        photo_name = f"{uuid.uuid4()}.jpg"
        photo_path = os.path.join(self.photos_dir, photo_name)
        try:
            cv2.imwrite(photo_path, image)
        except Exception as e:
            logging.error(f"Failed to save photo: {e}")
            return None
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
        """更新当前帧并处理异常"""
        last_check_time = time.time()  # 初始化最后检查时间
        while self.running:
            try:
                current_check_time = time.time()
                if current_check_time - last_check_time > 5:  # 每隔 5 秒检查一次
                    self.check_video_stream()
                    last_check_time = current_check_time
                frame = self.get_frame()
                if frame is not None:
                    with self.lock:# 加锁
                        self.current_frame = frame
                    if self.recording and self.output_file is not None:
                        try:
                            self.output_file.write(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR))
                        except Exception as e:
                            logging.error(f"Failed to write frame to video: {e}")
            except Exception as e:
                logging.error(f"Error in _update_frame: {e}")

    def _capture_frame(self):
        success, image = self.video.read()
        if success:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return None
        
    def check_video_stream(self):
        max_retries = 3  # 设置最大重试次数
        retry_count = 0
        while not self.video.isOpened() and retry_count < max_retries:
            logging.warning("Video stream is not opened, trying to reopen...")
            self.video.release()
            self.video = cv2.VideoCapture(0)
            retry_count += 1
            time.sleep(1)  # 等待一秒再重试
        if not self.video.isOpened():
            logging.error("Failed to reopen video stream after retries.")
            # 记录错误，但不抛出异常，让程序继续运行
        # 读取一帧测试
        ret, frame = self.video.read()
        if not ret or frame is None:
            logging.error("Failed to capture a frame from the video stream.")
            raise RuntimeError("Failed to capture a frame from the video stream.")
        else:
            logging.debug("Camera stream is working correctly.")  # 使用 DEBUG 级别

    def get_frame(self):
        with self.lock:
            success, image = self.video.read()
            if not success or image is None:
                logging.error("Failed to capture frame from video stream.")
                return None

        num = 0
        try:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
        except cv2.error as e:
            logging.error(f"Error converting frame to grayscale: {e}")
            return None

        # 动作检测
        fg_mask = self.bg_subtractor.apply(grey) # 前景掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数（只有在检测到物体运动后进行）
        if np.sum(fg_mask) > self.motion_threshold:
            # logging.info("Motion detected.") # 纪录移动检测
            current_time = time.time()
            motion_time_diff = current_time - self.last_motion_time
            if motion_time_diff >= self.motion_detect_interval:
                self.last_motion_time = current_time  # 更新最后的运动检测时间
                faceRects = self.classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=keypoints, minSize=(32, 32))
                if len(faceRects) > 0:  # 大于0则检测到人脸
                    logging.info("Face detected.") # 纪录人脸检测
                    origin_image = image.copy()
                    for faceRect in faceRects:  # 单独框出每一张人脸
                        num += 1
                        x, y, w, h = faceRect

                        # 画出矩形框
                        # origin_image = image.copy()
                        # cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), self.color, 2)

                        # 显示当前捕捉到了多少人脸图片了
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(image, 'person:%d' % num, (x + 30, y - 30), font, 1, (255, 0, 255), 4)

                        # 检查是否是新的人脸
                        current_time = time.time()
                        time_diff = current_time - self.last_email_time
                        if time_diff >= 300:  # 5分钟（300秒）
                            # 添加保存照片
                            face_image = origin_image[y-padding:y+h+padding, x-padding:x+w+padding]  # 提取人脸区域，如果想只保存人脸，则把以下图片保存和邮件发送中的origin_image替换为face_image
                            try:
                                photo_path = self.save_photo(origin_image)  # 保存人脸/原始照片
                                # 发送邮件通知
                                send_email_async(image=cv2.imencode(".jpg", origin_image)[1].tobytes(),
                                                 subject="人脸检测邮件通知",
                                                 message=f"检测到新的人脸，照片保存在 {photo_path}")
                            except Exception as e:
                                logging.error(f"Image saving error: {e}")
                            # 更新上次发送邮件的时间戳
                            self.last_email_time = current_time

        # 加时间戳
        add_timestamp_to_frame(image)
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            logging.error("Failed to encode image to JPEG format.")
            return None
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
            if frame is None:
                logging.warning("Frame is None, skipping...")
                time.sleep(1)
                continue
            time.sleep(np.round(1 / fps, 3))  # 这里设置休眠时间，确保视频流不间断
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
    logging.info("Endpoint '/capture' called. Initiating manual photo capture.")

    try:
        camera.check_video_stream()  # 确保视频流正常
        frame = camera._capture_frame()
        if frame is None:
            logging.error("Failed to capture frame. Camera returned None.")
            return Response("Failed to capture frame", status=500)
        
        # 解码图像并保存
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        try:
            photo_path = camera.save_photo(image)  # 保存拍摄的照片
            logging.info(f"Photo saved at {photo_path}. Sending email notification.")

            # 发送邮件
            try:
                send_email(
                    image=frame,
                    subject="拍摄照片",
                    message=f"手动拍摄照片内容，照片保存在 {photo_path}"
                )
                logging.info("Email sent successfully.")
            except Exception as email_error:
                logging.error(f"Email sending error: {email_error}")
                return Response(f"Photo saved, but email sending failed: {email_error}", status=500)

            return Response("Photo captured and email sent successfully.", status=200)
        
        except Exception as save_error:
            logging.error(f"Photo saving error: {save_error}")
            return Response(f"Photo saving error: {save_error}", status=500)

    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        return Response(f"Internal server error: {e}", status=500)
    
@app.route('/start_recording', methods=['POST'])
def start_recording():
    logging.info("Endpoint '/start_recording' called. Starting video recording.")
    camera.check_video_stream()  # 确保视频流正常
    camera.start_recording()
    return Response(status=200)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    logging.info("Endpoint '/stop_recording' called. Stopping video recording.")
    camera.stop_recording()
    return Response(status=200)

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {e}")
    return "An error occurred. Please try again later.", 500

if __name__ == '__main__':
    camera.start()
    try:
        app.run(host='0.0.0.0', debug=False, port=output_port)
    except KeyboardInterrupt:
        camera.stop()
