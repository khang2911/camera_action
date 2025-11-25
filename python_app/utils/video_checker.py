import cv2
import json
import subprocess
from datetime import datetime, timedelta
import pytz
import cv2
from datetime import datetime
from decord import VideoReader, cpu
import tqdm


hanoi_tz = pytz.timezone("Asia/Ho_Chi_Minh")


# start_time = 
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

def get_expand_timestamp(start: str, end: str):
    start = fix_isoformat(start)
    end   = fix_isoformat(end)

    start_time = datetime.fromisoformat(start).astimezone(hanoi_tz)
    end_time   = datetime.fromisoformat(end).astimezone(hanoi_tz)

    return start_time.timestamp() - 5, end_time.timestamp() + 5

def get_cutoff_timestamp(video_end_time: str, offset_sec: int = 30) -> float:
    """Trả về timestamp cutoff (video_end_time - offset_sec)"""
    end = fix_isoformat(video_end_time)
    end_time = datetime.fromisoformat(end).astimezone(hanoi_tz)
    cutoff_time = end_time + timedelta(seconds=offset_sec)
    return cutoff_time.timestamp()

def ffprobe_info(video_path):
    """Lấy metadata video bằng ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,duration,nb_frames",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)["streams"][0]

        # duration
        duration = float(info.get("duration", 0))

        # nb_frames có thể không có trong một số container
        nb_frames = info.get("nb_frames", None)
        nb_frames = int(nb_frames) if nb_frames and nb_frames.isdigit() else None

        # fps: ưu tiên từ avg_frame_rate, fallback từ nb_frames/duration
        avg_frame_rate = info.get("avg_frame_rate", "0/0")
        num, den = avg_frame_rate.split("/")
        fps = float(num) / float(den) if den != "0" else 0
        if fps <= 0 and nb_frames and duration > 0:
            fps = nb_frames / duration
            # print(nb_frames)
            # print(duration)
            # print(nb_frames/duration)

        return {"fps": fps, "duration": duration, "nb_frames": nb_frames}
    
    except Exception as e:
        return {"fps": 0, "duration": 0, "nb_frames": None}


def is_video_valid(data, time = 2):
    raw_alarm = data['alarm']['raw_alarm']
    moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
    video_paths = [rec['file'] for rec in raw_alarm['record_list']]
    durations = [rec['duration'] for rec in raw_alarm['record_list']]

    for video_path, moment_time, duration in zip(video_paths, moment_times, durations):
        # cap = cv2.VideoCapture(video_path)
        # if not cap.isOpened():
        #     raise RuntimeError(f"Cannot open video: {video_path}")
        
        ff_info = ffprobe_info(video_path)

        if float(duration) - ff_info["duration"] >= time:
            return True

    return False

def check_h264_errors(data) -> bool:
    raw_alarm = data['alarm']['raw_alarm']
    moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
    video_paths = [rec['file'] for rec in raw_alarm['record_list']]
    durations = [rec['duration'] for rec in raw_alarm['record_list']]
    print("-- validating video ...")
    for video_path, moment_time, duration in zip(video_paths, moment_times, durations):
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", video_path,
            "-f", "null", "-"
        ]

        # Mở process, đọc stderr line by line
        with subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1) as proc:
            for line in proc.stderr:
                # Các pattern lỗi cần check
                if any(p in line for p in [
                    "reference picture missing during reorder",
                    "Missing reference picture",
                    "mmco: unref short failure",
                    "illegal short term buffer state detected",
                ]):
                    proc.kill()  # dừng ffmpeg ngay lập tức
                    return True
    return False

def extract_time(start_frame, end_frame):
    # from ocr.check import 
    pass

def check_missing_frame_end(data, max_gap=3):
    raw_alarm = data['alarm']['raw_alarm']
    moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
    # video_paths = [rec['file'] for rec in raw_alarm['record_list']]
    video_end_time_str = raw_alarm['video_end_time']
    video_paths = data['playback_location']
    print(" -- validating video ...")
    cutoff_ts = get_cutoff_timestamp(video_end_time_str, offset_sec=30)

    for video_index, video_path in enumerate(video_paths):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frame = len(vr)
        
        # start_time tính từ moment_time (epoch)
        start_time = datetime.fromtimestamp(moment_times[video_index], tz=hanoi_tz)

        prev_ts = None
        for i in range(num_frame):
            try:
                _ = vr[i]  # force decode
                ts = vr.get_frame_timestamp(i)[0]
                frame_time = start_time + timedelta(seconds=float(ts))
                # nếu vượt cutoff thì dừng
                if frame_time.timestamp() > cutoff_ts:
                    break

                if prev_ts is not None:
                    gap = ts - prev_ts
                    if gap >= max_gap:
                        return True
                prev_ts = ts
            except Exception:
                continue

    return False


def check_missing_frame(data, max_gap = 3):
    raw_alarm = data['alarm']['raw_alarm']
    moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
    video_paths = [rec['file'] for rec in raw_alarm['record_list']]
    durations = [rec['duration'] for rec in raw_alarm['record_list']]
    video_end_time = raw_alarm['video_end_time']
    print(" -- validating video ...")
    # fps = float(vr.get_avg_fps())
    
    for video_index, video_path in enumerate(video_paths):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frame = len(vr)
        prev_ts = None
        total_gap = 0
        for i in range(num_frame):
            # try:
                _ = vr[i]  # force decode this frame
                ts = vr.get_frame_timestamp(i)[0]
                if prev_ts is not None:
                    gap = ts - prev_ts
                    # total_gap += gap
                    # if total_gap >= 5:
                    #     return True
                    
                    if gap >= max_gap:
                        return True
                prev_ts = ts
            # except Exception as e:
            #     continue
        
    return False
        # start_frame, end_frame = None
        # while True:
        #     ind = 0
        #     if ind > 26:
        #         break
        #     try:
        #         start_frame = vr[ind]
        #         break
        #     except:
        #         ind+=1

        # while True:
        #     ind = -1
        #     if ind < 26:
        #         break
        #     try:
        #         end_frame = vr[ind]
        #         break
        #     except:
        #         ind-=1
        
        # if start_frame is not None and end_frame is not None:



def check_missing_frame_video(video_paths, max_gap = 3):
    # raw_alarm = data['alarm']['raw_alarm']
    # moment_times = [rec['moment_time'] for rec in raw_alarm['record_list']]
    # video_paths = [rec['file'] for rec in raw_alarm['record_list']]
    # durations = [rec['duration'] for rec in raw_alarm['record_list']]
    # video_end_time = raw_alarm['video_end_time']
    print(" -- validating video ...")
    # fps = float(vr.get_avg_fps())
    
    for video_index, video_path in enumerate(video_paths):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frame = len(vr)
        prev_ts = None
        total_gap = 0
        for i in range(num_frame):
            try:
                _ = vr[i]  # force decode this frame
                ts = vr.get_frame_timestamp(i)[0]
                if prev_ts is not None:
                    gap = ts - prev_ts
                    # total_gap += gap
                    # if total_gap >= 5:
                    #     return True
                    
                    if gap >= max_gap:
                        return True
                prev_ts = ts
            except Exception as e:
                continue
        
    return False


        
if __name__ == '__main__':
    import json





    data = '''{"alarm":{"serial":"c03o24120001908","record_url":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7eeab849ad59f6b7e79fd372780a3d87f.mp4","record_list":[{"file":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7eeab849ad59f6b7e79fd372780a3d87f.mp4","record_start_time":"2025-10-13T11:40:05"},{"file":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7f48d148bedee29a7264875094386077c.mp4","record_start_time":"2025-10-13T11:55:05"},{"file":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7044a34cc8ec6bc0e26d36ce44346b55f.mp4","record_start_time":"2025-10-13T12:10:06"}],"record_start_time":"2025-10-13T11:40:05","alarm_time":"2025-10-13T13:41:21","receive_time":"2025-10-13T13:41:28.707212","record_id":"1620ff33-03e1-48d3-87cd-0b169ce7ef7a","raw_alarm":{"alarm":"trigger_alarm","serial":"c03o24120001908","record_id":"1620ff33-03e1-48d3-87cd-0b169ce7ef7a","home_id":"2100d114-bb00-49a1-a3ba-d53e8695acbd","send_at":"20251013134121","video_start_time":"2025-10-13T12:04:10.3110991+07:00","video_end_time":"2025-10-13T12:08:03.8089944+07:00","record_list":[{"duration":"900","file":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7eeab849ad59f6b7e79fd372780a3d87f.mp4","image":"https://live-10-hcm.fcam.vn/thumbnails/c03o24120001908/1760330405411.jpg","recordtimestamp":"2025-10-13T11:40:05+07:00","moment_time":1760330405},{"duration":"900","file":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7f48d148bedee29a7264875094386077c.mp4","image":"https://live-10-hcm.fcam.vn/thumbnails/c03o24120001908/1760331305649.jpg","recordtimestamp":"2025-10-13T11:55:05+07:00","moment_time":1760331305},{"duration":"900","file":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7044a34cc8ec6bc0e26d36ce44346b55f.mp4","image":"https://live-10-hcm.fcam.vn/thumbnails/c03o24120001908/1760332205888.jpg","recordtimestamp":"2025-10-13T12:10:06+07:00","moment_time":1760332206}],"vaccine_info":{"Lcvid":"LP500011760328460659","CameraId":"2220-I03S-R18-C03O24120001908","ShopCode":"58037","ShopName":"VX HNI 216 Thái Hà, P. Đống Đa","Indication":[{"Sku":"00038256","Odac":"TgPiR2CS0OmzNG3g","Taxonomies":"DẠI","VaccineName":"VERORAB VẮC XIN DẠI"}],"PersonName":"Phạm Trường Giang","TicketCode":"TK58037567811760328560571","TrackingTime":"2025-10-13T12:08:03.8089944+07:00","InjectingTime":"2025-10-13T12:04:10.3110991+07:00","InjectingNursingCode":"64584","InjectingNursingName":"Hà Trần Huyền Khanh"},"target_action":[{"action_id":"1","action_name":"Lắc vaccin","type":"time","value":[0]},{"action_id":"2","action_name":"Sát khuẩn tay","type":"count","value":3}]},"tracking_key":"c03o24120001908#20251013114005#1620ff33-03e1-48d3-87cd-0b169ce7ef7a"},"download_info":[{"local_filepath":"/shared_storage/action_videos/c03o24120001908/20251013114005_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","minio_object_name":"videos/c03o24120001908/20251013114005_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","record_url":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7eeab849ad59f6b7e79fd372780a3d87f.mp4"},{"local_filepath":"/shared_storage/action_videos/c03o24120001908/20251013115505_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","minio_object_name":"videos/c03o24120001908/20251013115505_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","record_url":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7f48d148bedee29a7264875094386077c.mp4"},{"local_filepath":"/shared_storage/action_videos/c03o24120001908/20251013121006_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","minio_object_name":"videos/c03o24120001908/20251013121006_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","record_url":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7044a34cc8ec6bc0e26d36ce44346b55f.mp4"}],"playback_location":["/shared_storage/action_videos/c03o24120001908/20251013114005_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","/shared_storage/action_videos/c03o24120001908/20251013115505_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4","/shared_storage/action_videos/c03o24120001908/20251013121006_1620ff33-03e1-48d3-87cd-0b169ce7ef7a.mp4"],"config":{"service_id":"eb305b04-a231-4ac2-a925-6cea95b27d9a","service_name":"Action Recognition","camera_id":"d3c653ed-4509-4d64-8ee4-59293ef2dc46","serial":"c03o24120001908","camera_name":"c03o24120001908","location_id":"2100d114-bb00-49a1-a3ba-d53e8695acbd","notify_result":true,"zone":1,"external_data":{"actions":[{"id":"1","name":"Sát khuẩn tay","config":{"time":5}},{"id":"2","name":"Lắc vaccine","config":{"time":30}}],"roi_config":[549,651,1369,1249]}},"prioritize":false,"task_type":"2perday","result_queue":"aicam#979219ff-b1d8-40ee-bfdc-c05f968b8972#results#h7UU0Trq85","video_resolution":{"width":2304,"height":1296},"run_only_office_hours":false,"ml_results":{"alarm":"trigger_alarm","record_id":"1620ff33-03e1-48d3-87cd-0b169ce7ef7a","serial":"c03o24120001908","service_id":"eb305b04-a231-4ac2-a925-6cea95b27d9a","home_id":"2100d114-bb00-49a1-a3ba-d53e8695acbd","url_video":"http://pb-01-hcm.fcam.vn/iwanttoplay/9dd373af15433a8df12a60634e31533e27fdd4a26f1a34b294d8d18a96c6e8bc84b945cb9e9dff18e95c5d1f91e185cccf3a91d71722ae5c52167b9ae841f59bbfbf1b0c92c8009465a2125e00c9515dfa1729ea9d48f667b4d1f42ce8132cb7eeab849ad59f6b7e79fd372780a3d87f.mp4","timestamp":1760337867,"config":{},"actions":[{"action_id":"2","action_name":"Sát khuẩn tay","start_time":"2025-10-13T12:04:01+07:00","duration":2.6923076923076925},{"action_id":"2","action_name":"Sát khuẩn tay","start_time":"2025-10-13T12:05:09+07:00","duration":2.6923076923076925},{"action_id":"2","action_name":"Sát khuẩn tay","start_time":"2025-10-13T12:07:09+07:00","duration":3.3846153846153846},{"action_id":"2","action_name":"Sát khuẩn tay","start_time":"2025-10-13T12:07:44+07:00","duration":3.3846153846153846}],"extra_info":[{"action_id":"2","action_name":"Sát khuẩn tay","error_id":["0"]},{"action_id":"1","action_name":"Lắc vaccine","error_id":["0"]}]},"root_fields":{"config":{"service_id":"eb305b04-a231-4ac2-a925-6cea95b27d9a","service_name":"Action Recognition","camera_id":"d3c653ed-4509-4d64-8ee4-59293ef2dc46","serial":"c03o24120001908","camera_name":"c03o24120001908","location_id":"2100d114-bb00-49a1-a3ba-d53e8695acbd","notify_result":true,"zone":1,"external_data":{"actions":[{"id":"1","name":"Sát khuẩn tay","config":{"time":5}},{"id":"2","name":"Lắc vaccine","config":{"time":30}}],"roi_config":[549,651,1369,1249]}}}}'''
    # video_path = ["/age_gender/detection/ocr/c03o24120001908_1759981189.mp4"]

    data = json.loads(data)

    missing = check_missing_frame_end(data)
    # missing = check_missing_frame_video(video_path)
    print(missing)