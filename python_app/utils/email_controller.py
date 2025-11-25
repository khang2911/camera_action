import json
import os
import smtplib
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText

# ================= CONFIG =================
# SMTP_SERVER = "proxy-fptcom.fpt.net"
# SMTP_PORT = 587
# USERNAME = "aicads@fpt.com"
# PASSWORD = "Rx8S4TEcy5egTl7YpOxVd1g8CUCI68"

SMTP_SERVER = "proxy-fptcom.fpt.net"
SMTP_PORT   = 587
USERNAME    = "aicads@fpt.com"
PASSWORD    = "BsU5TI5KY8CnsEqQ7M2QOpoRaeH5LKYy"

TO = ["PhuongVTH@fpt.com","phunghx2@fpt.com", "khangtd6@fpt.com", "Duytp3@fpt.com", "hand6@fpt.com"]
# TO = ["hand6@fpt.com"]
CC = [
    "Hieunv15@fpt.com",
    "Daonta10@fpt.com",
    "Thanhnt130@fpt.com",
    "FTEL.FLI.SUPPORT@fpt.com",
    "nuongpt@fpt.com",
    "duynm14@fpt.com",
    "huynd78@fpt.com",
    "tuandt27@fpt.com",
    "nhilta2@fpt.com",
    "dungnh41@fpt.com",
    "tuannva4@fpt.com"
]

# TO = ["huynd78@fpt.com"]
# CC = ["dungnh41@fpt.com"]
# ===========================================

def send_email(login, password, message):
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(login, password)
        server.send_message(message)


def get_filed(line):
    data = json.loads(line)

    try:
        error_data = data
    except:
        return json.dumps(data)
    return json.dumps(error_data)

if __name__ == "__main__":
    # lấy ngày hôm qua
    # now_vn = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    # yesterday = (now_vn - timedelta(days=1)).date()
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    date_str = yesterday.strftime("%Y-%m-%d")

    # file nguồn và file copy
    src_file = f"/age_gender/detection/logs_prod/monitor_invalid_format/{date_str}.log"
    dst_file = f"/age_gender/detection/logs_prod/email/{date_str}.log"

    if not os.path.exists(src_file):
        print(f"[ERROR] File log không tồn tại: {src_file}")
        exit(1)

    # đảm bảo thư mục đích tồn tại
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    # copy từng dòng sang file mới
    lines = []
    with open(src_file, "r", encoding="utf-8") as src:
        lines = src.readlines()

    if not lines:
        print(f"[INFO] File {src_file} rỗng, không gửi email.")
        exit(0)

    # copy từng dòng sang file mới
    with open(dst_file, "w", encoding="utf-8") as dst:
        for line in lines:
            dst.write(get_filed(line))
            dst.write("\n")

    # tạo email
    msg = MIMEMultipart()
    msg['Subject'] = f"[AICAMERA - FLI - ACTION RECOGNITION ALERT] ERROR MESSAGE {yesterday.strftime('%d/%m/%Y')}"
    msg['From'] = USERNAME
    msg['To'] = ", ".join(TO)
    msg['Cc'] = ", ".join(CC)

    # body chuẩn
    body = f"""Dear Anh/Chị,

Hệ thống phát hiện các phiếu tiêm bị lỗi ngày {yesterday.strftime("%d/%m/%Y")}. Anh/Chị vui lòng check file đính kèm. 

Trân trọng,
AI-CADS Camera Monitoring System
"""
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # attach file log
    part = MIMEBase('application', "octet-stream")
    with open(dst_file, "rb") as f:
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(dst_file))
    msg.attach(part)

    # gửi
    send_email(USERNAME, PASSWORD, msg)
    print(f"[INFO] Email sent with log {dst_file}")
