# Mini Surveillance: 基于Flask的简易视频监控及人脸检测网页应用端 (Web Application for Simple Video Surveillance and Face Detection based on Flask)

## 功能介绍
该项目基于Flask实现了一个具有视频监控、人脸检测、自动邮件提醒、录像及邮件提醒等功能的监控器，可以部署在树莓派等开发板上，但是对于硬件性能有一定的要求，也可以通过调整相关参数使其适配您的机器。

## 安装
- 首先确保安装好了python3
- 克隆项目后通过以下命令安装依赖：
```cmd
git clone https://github.com/Avenlee-github/Mini-Surveillance.git
cd your_path/MiniSurveillance
pip install -r requirements.txt
```

## 准备工作
- 配置文件：首先准备好配置文件(这是必备操作，否则项目无法正常运行)，删除根目录下的<font color=#FFFF00>***"config.yaml.example"***</font>的.example后缀，改为<font color=#FFFF00>***"config.yaml"***</font>文件；
- 实现用户认证登录：该项目支持用户认证登录后查看监控视频，通过设置<font color=#FFFF00>***"config.yaml"***</font>配置文件中authentication模块下的username和password参数即可达成。如果不设置以上参数则默认用户名为<font color=#FFFF00>***"your_username"***</font>，默认密码为<font color=#FFFF00>***"your_password"***</font>；
- 实现邮件发送功能：通过设置<font color=#FFFF00>***"config.yaml"***</font>配置文件中email模块的sender_email, sender_password, receiver_email, smtp_server和port参数即可实现邮件提醒功能。注意如果你使用的运营商提供的邮箱服务，其中的sender_password不是你设置的登录密码，而是SMTP授权密码，需要到运营商邮箱官方网站申请开通。如果不设置相关参数，对应照片和视频仍然可以保存到本地路径中，但是不会发送邮件提醒；
- 确保人脸检测模型文件（<font color=#FFFF00>***"haarcascade_frontalface_alt.xml"***</font>）在项目根目录下；

## 运行
- 安装好依赖并设置好参数后运行以下命令打开运行项目，项目运行成功后会在本地生成一个<font color=#FFFF00>***"running_log.log"***</font>文件用于保存运行日志，方便后期开发：
```cmd
python main.py
```
- 客户端：运行后通过访问<font color=#FFFF00>http://127.0.0.1:5000</font>或者你运行设备IP的5000端口就可以访问项目客户端，输入用户名和密码后就可以查看监控视频流，客户端界面如下：

  <img src=".\figure\GUI.png" alt="stable" style="zoom:40%;" />

- 人脸检测：项目在检测到人脸后会自动拍照，存储在本地的<font color=#FFFF00>***"photos"***</font>文件夹中，并同时向设置好的邮箱发送提醒邮件；
- 手动拍照：点击<font color=#FFFF00>***"Capture"***</font>按钮可以手动拍照并存储在<font color=#FFFF00>***"photos"***</font>文件夹中，并同时向设置好的邮箱发送提醒邮件；
- 手动录像：点击1次<font color=#FFFF00>***"Record"***</font>按钮会开始录像，同时按钮变为红色，再点击一次按钮则结束录像，将录像保存到<font color=#FFFF00>***"recordings"***</font>文件夹中，并同时向设置好的邮箱发送提醒邮件（注意send_video函数设置了发送邮件最大文件大小不超过100MB，如果需要自己修改请到<font color=#FFFF00>***"email_smtp.py"***</font>文件中自行修改）；
- 中止运行：在运行终端通过Ctrl + C即可关闭运行。

## 开发
- 项目后端：代码都在<font color=#FFFF00>***"main.py"***</font>和<font color=#FFFF00>***"utils/"***</font>文件夹中的文件中，可以根据自己需要修改；
- 项目前端：HTML客户端在<font color=#FFFF00>***"templates/"***</font>文件夹中的<font color=#FFFF00>***"template.html"***</font>模板文件中，请根据自己需要修改；

## 声明/Declaration
- 本项目代码仅供学习、个人测试和非商业用途使用。禁止将本项目代码用于有损他人隐私安全或违法行为。任何他人在使用本项目代码时造成的不良后果与本项目作者无关。使用者需自行承担使用本项目代码所产生的风险和责任。本项目作者不对使用本项目代码所造成的任何直接或间接损失负责。使用本项目代码即表示您同意遵守以上声明。
- The code in this project is intended for educational, personal testing, and non-commercial purposes only. It must not be used for compromising the privacy and security of others or engaging in any illegal activities. The author of this project bears no responsibility for any adverse consequences resulting from the use of this code by others. Users are solely responsible for the risks and liabilities associated with the use of this code. The author of this project shall not be held liable for any direct or indirect damages caused by the use of this code. By using this code, you agree to comply with the aforementioned disclaimer.