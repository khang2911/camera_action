import random
import redis
import json
import time
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import pytz

from python_app.utils.utils import get_fps_ffprobe, getConfig, unscale_box

hanoi_tz = pytz.timezone("Asia/Ho_Chi_Minh")

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


def is_cam_complete(cam_id):
    with open('../cams_complete.json', 'r') as f:
        cameras = dict(json.load(f))

    if cam_id not in cameras.keys():
        return False
    
    cam_info = cameras[cam_id]
    if cam_info['total'] < 10:
        return False
    
    total_error = cam_info['shaking'] + cam_info['washing']
    if total_error/ cam_info['total'] < (5/100):
    # if total_error == 0:
        return True
    return False
    

class RedisQueueConsumer:
    def __init__(self):
        """Khởi tạo Redis connection"""
        # Thông tin kết nối Redis
        self.input_config = {
            'host': '172.24.178.105',
            'port': 6379,
            'db': 0,
            'decode_responses': True
        }

        self.output_config = {
            'host': '172.24.178.105',
            'port': 6379,
            'db': 0,
        }

        self.redis_host = '172.24.178.105'
        self.redis_port = 6379
        self.db = 0
        
        # Queue names
        self.action_queue = "ml#vaccine_monitor_prod"
        self.detection_queue = "det#vaccine_monitor_prod_in"
        self.output_queue = "action-ml#results-prod"

        # Khởi tạo Redis connection
        self.redis_client = None
        self.output_redis = redis.Redis(**self.output_config)
        # Tạo thư mục logs/prod nếu chưa tồn tại
        os.makedirs('logs/prod', exist_ok=True)

        self.debug = False
        self.connect_redis()
    
    def connect_redis(self):
        try:
            self.redis_client = redis.Redis(**self.input_config)
            # Test connection
            self.redis_client.ping()
            print("Connected Redis server")
        except Exception as e:
            print(f"Error connecting Redis: {e}")
            raise
    
    def pop_message_from_queue(self, queue_name: str) -> Optional[Tuple[str, float]]:
        try:
            # Sử dụng rpop để lấy element có score thấp nhất
            member = self.redis_client.rpop(queue_name)
            
            if member:
                return member, 0
            return None
            
        except Exception as e:
            # print(f"Lỗi khi pop message từ queue {queue_name}: {e}")
            return None
    
    def log_to_file(self, message_data, file, prin = False):
        os.makedirs(f"logs/prod/{file}", exist_ok=True)
        try:
            if prin:
                print(f"--> INFO: Log to {file} ...")
            if 'message' in message_data:
                dt_utc = datetime.strptime(message_data['message']['alarm']['raw_alarm']['send_at'], '%Y%m%d%H%M%S')
            else:
                dt_utc = datetime.strptime(message_data['alarm']['raw_alarm']['send_at'], '%Y%m%d%H%M%S')
                
            # dt_hcm = dt_utc.astimezone(timezone(timedelta(hours=7)))
            dt_string = dt_utc.strftime('%Y-%m-%d')
            
            with open(f'logs/prod/{file}/{dt_string}.log', 'a+', encoding='utf-8') as f:
                f.write(json.dumps(message_data, ensure_ascii=False) + '\n')
        
        except Exception as e:
            print(f"--> INFO: Log to error_message ...")
            with open(f'logs/prod/error_message.log', 'a+', encoding='utf-8') as f:
                f.write(f"{message_data}" + '\n')

    def push_to_detection(self, data):
        self.redis_client.lpush(self.detection_queue, json.dumps(data))
        print("===>>> Pushed message to forward queue:", self.detection_queue)

    def push_to_queue(self, data):
        self.output_redis.lpush(self.output_queue, json.dumps(data))
        print("===>>> Pushed message to forward queue:", self.output_queue)

    def get_result(self, raw_msg):
        data = {}
        data['alarm'] = raw_msg['alarm']['raw_alarm']['alarm']
        data['record_id'] = raw_msg['alarm']['raw_alarm']['record_id']
        data['serial'] = raw_msg['alarm']['raw_alarm']['serial']
        data['service_id'] = raw_msg['config']['service_id']
        data['home_id'] = raw_msg['alarm']['raw_alarm']['home_id']
        data['url_video'] = raw_msg['alarm']['record_url']

        data['timestamp'] = int(time.time())
        data['config'] = {}
        data['actions'] = []
        data['extra_info'] = [
                                {
                                    "action_id": "2",
                                    "action_name": "Sát khuẩn tay",
                                    "error_id": [
                                    "0"
                                    ]
                                },
                                {
                                    "action_id": "1",
                                    "action_name": "Lắc vaccine",
                                    "error_id": [
                                    "0"
                                    ]
                                }
                            ]

        return data

    def process_action_queue(self):
        """Xử lý action queue"""
        result = self.pop_message_from_queue(self.action_queue)
        if result:
            message, score = result
            # Parse message thành JSON nếu có thể
            try:
                parsed_message = json.loads(message)
            except:
                self.log_to_file(
                        {"error": "invalid json format", "message": parsed_message},
                        "monitor_invalid_format", True)
                return False
            
            # print(parsed_message)
            print(f"\n## GET message: {parsed_message['alarm']['serial']}, timestamp: {parsed_message['alarm']['raw_alarm']['record_list'][0]['recordtimestamp']}")

            # Log full origin message dạng JSON
            self.log_to_file(parsed_message, "monitor_msg")

            record_info = parsed_message['alarm']['raw_alarm']['record_list'][0]
            raw_alarm = parsed_message['alarm']['raw_alarm']
            serial = raw_alarm['serial']
            home_id = raw_alarm['home_id']
            
            # check start and end time
            if "video_start_time" not in raw_alarm and "video_end_time" not in raw_alarm:
                message_data = {
                    "error": "Missing start/end time",
                    "message": parsed_message,
                }
                self.log_to_file(message_data, "monitor_invalid_format", True)
                return False

            recordtimestamp = record_info['recordtimestamp']

            # khoảng thời gian sự kiện
            event_start, event_end = get_timestamp(
                raw_alarm['video_start_time'],
                raw_alarm['video_end_time']
            )

            # check missing segment haha
            # records = raw_alarm.get("record_list", [])
            # for rec, next_rec in zip(records, records[1:]):
            #     if int(next_rec["moment_time"]) - (int(rec["moment_time"]) + float(rec.get("duration", 0))) >= 3:
            #         self.log_to_file(
            #             {"error": "missing video segment", "message": parsed_message},
            #             "monitor_invalid_format", True)
                    # return False

            # lấystart và end bao phủ tất cả record
            record_start = min(rec['moment_time'] for rec in raw_alarm['record_list'])
            record_end   = max(rec['moment_time'] + float(rec['duration']) for rec in raw_alarm['record_list'])

            # kiểm tra sự kiện có nằm trong video range không
            if not (record_start <= event_start and record_end + 3 >= event_end):
                self.log_to_file(
                    {"error": "start/end time is invalid", "message": parsed_message},
                    "monitor_invalid_format", True)
                return False
            
            try:
                target_action = raw_alarm['target_action']
                vaccine_info = raw_alarm['vaccine_info']
            except:
                self.log_to_file(
                    {"error": "missing vaccine/target info", "message": parsed_message},
                    "monitor_invalid_format", True)
                return False

            # if is_cam_complete(serial):
            #     ml_results = self.get_result(parsed_message)
            #     result_msg = parsed_message.copy()

            #     result_msg['ml_results'] = ml_results
            #     result_msg['root_fields'] = {'config': result_msg['config']}
            #     # self.log_to_file(result_msg, "temp")
            #     self.log_to_file(result_msg, "monitor_all_result")
            #     self.push_to_queue(result_msg)
            #     return False

            image_w, image_h = parsed_message['video_resolution']['width'], parsed_message['video_resolution']['height']
            if image_w < 1920 or image_h < 1080:
                self.log_to_file(
                    {"error": "video resolution is invalid", "message": parsed_message},
                    "monitor_invalid_format", True)
                return False

            video_path = parsed_message['playback_location'][0]
            fps_video = get_fps_ffprobe(video_path=video_path)

            if fps_video > 13.5 or fps_video < 12.5:
                self.log_to_file(
                    {"error": "video fps is invalid", "message": parsed_message},
                    "monitor_invalid_format", True)
                return False
            
            parsed_message['fps'] = fps_video

            ### check frame problem
            # #=====================================
            # is_time_valid = is_video_valid(parsed_message, 3)
            # if is_time_valid:
            #     self.log_to_file(
            #         {"error": "time in video is invalid", "message": parsed_message},
            #         "monitor_invalid_format", True)
            #     return False
            
            # is_frame_invalid = check_missing_frame_end(parsed_message, max_gap=3)
            # if is_frame_invalid:
            #     self.log_to_file(
            #         {"error": "missing frame", "message": parsed_message},
            #         "monitor_invalid_format", True)
            #     return False
            # #======================================

            # get roi config
            cam_config = getConfig(serial)
            if "box" not in cam_config:
                self.log_to_file(
                    {"error": "camera is not configured", "message": parsed_message},
                    "monitor_missing_config", True)
                return False
            
            try:
                bbox_config = [cam_config['box'][0][0], cam_config['box'][0][1], cam_config['box'][2][0], cam_config['box'][2][1]]
                process_roi = unscale_box(bbox_config, image_w, image_h)
            except:
                self.log_to_file(
                    {"error": "camera is not configured", "message": parsed_message},
                    "monitor_missing_config", True)
                return False
            
            ## process
            # ======================================================
            self.push_to_detection(parsed_message)
            
            return True
        return False
    
    def run_consumer(self, check_interval: float = 0.1):
        """
        Chạy consumer để liên tục pop messages từ cả hai queues
        
        Args:
            check_interval: Khoảng thời gian giữa các lần check (seconds)
        """
        print("Started process vaccine monitoring...")
        
        # try:
        while True:
            # Xử lý cả hai queues
            action_processed = self.process_action_queue()
            
            # Nếu không có message nào, chờ một chút
            if not action_processed:
                time.sleep(check_interval)


def main():
    """Hàm main"""
    # try:
        # Khởi tạo và chạy consumer
    consumer = RedisQueueConsumer()
    consumer.run_consumer()
        
    # except Exception as e:
    #     print(f"Lỗi: {e}")
    #     return 1
    
    return 0

if __name__ == "__main__":
    exit(main())