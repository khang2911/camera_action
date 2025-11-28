import json
import subprocess
import requests
import copy
from datetime import datetime

import pytz
hanoi_tz = pytz.timezone("Asia/Ho_Chi_Minh")


def getConfig(camID, orgName = "FLI_Action", staging = False):
    
    if staging:
        url = f"https://staging-aicam.cads.live/api/internal/cameras/{camID}/configs?service=action-recognition&orgName={orgName}"
    else:
        url = f"https://inside-camera.cads.live/api/internal/cameras/{camID}/configs?service=action-recognition&orgName={orgName}"

    try:
        proxies = {
            'http': None,
            'https': None
        }
        response = requests.get(url, proxies=proxies, timeout=2)
        config   = response.json()
        return dict(config['config'])
    
    except:
        return {}

def unscale_box(bbox_n, image_w, image_h):
    """
    Chuyển bbox normalized -> pixel
    Args:
        bbox_n: tuple (x1_n, y1_n, x2_n, y2_n)
        image_w, image_h: kích thước ảnh muốn vẽ lên
    Returns:
        tuple (x1, y1, x2, y2)
    """
    x1_n, y1_n, x2_n, y2_n = bbox_n
    return (
        int(x1_n * image_w),
        int(y1_n * image_h),
        int(x2_n * image_w),
        int(y2_n * image_h),
    )

def setConfig(serial, config_data, staging = False):

    if staging:
        access_token = "WNeUR4ir99N4s3DXooF6YyZztitE9Ew6"
        url = "https://staging-aicam.cads.live/api/external/ai/v2/action-recognition/config"
    else:
        access_token = "aKmRCOdp94mxrv64kpmQWYyvwSkikDvU"
        url = "https://aicads.fpt.com/api/external/ai/v2/action-recognition/config"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Access-token": f"{access_token}",
    }
    payload = {
                "serial": serial,
                "service_type": "2perday",   # cố định theo yêu cầu
                "config": config_data,
            }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"[{serial}] Status: {response.status_code}")
        if response.status_code != 200:
            print("Response:", response.text)
        return response.status_code
    except Exception as e:
        print(f"[{serial}] Error: {e}")
        return None

def download_videos(url, video_path):
    # stream download để tránh load cả file vào RAM
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except:
        return False
    return True


def fix_isoformat(dt_str: str) -> str:
    if "." in dt_str:
        # bỏ phần microseconds, giữ lại timezone (nếu có)
        date_part, rest = dt_str.split(".", 1)
        if "+" in rest:
            _, tz = rest.split("+", 1)
            return f"{date_part}+{tz}"
        elif "-" in rest:
            _, tz = rest.split("-", 1)
            return f"{date_part}-{tz}"
        else:
            return date_part
    return dt_str

def get_timestamp(start: str, end: str):
    start = fix_isoformat(start)
    end   = fix_isoformat(end)

    start_time = datetime.fromisoformat(start).astimezone(hanoi_tz)
    end_time   = datetime.fromisoformat(end).astimezone(hanoi_tz)

    return start_time.timestamp(), end_time.timestamp()

def get_single_timstamp(datetime_):
    end   = fix_isoformat(datetime_)
    end_time   = datetime.fromisoformat(end).astimezone(hanoi_tz)
    return end_time.timestamp()

def get_single_datetime(timestamp):
    return datetime.fromtimestamp(timestamp, tz=hanoi_tz).isoformat(timespec="seconds")

def get_expand_timestamp(start: str, end: str):
    start = fix_isoformat(start)
    end   = fix_isoformat(end)
    start_time = datetime.fromisoformat(start).astimezone(hanoi_tz)
    end_time   = datetime.fromisoformat(end).astimezone(hanoi_tz)

    return start_time.timestamp() - 30 , end_time.timestamp() + 5

def get_fps_ffprobe(video_path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    for stream in info["streams"]:
        try:
            if stream["codec_type"] == "video":
                nb_frame = int(stream.get("nb_frames"))
                duration = float(stream.get('duration'))
                
                fps = nb_frame/duration
                return fps
        except:
            return 0
    return 0

if __name__ == "__main__":
    import os
    import json
    import requests


    # batch_upload()
    # cam_config = getConfig('c03o24090007877')
    # print(cam_config)
    # print([cam_config['box'][0][0], cam_config['box'][0][1], cam_config['box'][2][0], cam_config['box'][2][1]])
    

