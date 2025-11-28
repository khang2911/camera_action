import random
import redis
import json
import time
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
from python_app.models.vaccine_processor import Runner
import pytz

from python_app.utils.analyster import ExtraInfoProcessor
from python_app.utils.utils import get_single_datetime, get_single_timstamp
hanoi_tz = pytz.timezone("Asia/Ho_Chi_Minh")


def fake_actions(actions):
    new_actions = []
    for act in actions:
        start_time = act['start_time']
        start_timetamp_fake = get_single_timstamp(start_time) + random.randint(-20, 20)

        fake_act = act.copy()
        fake_act['start_time'] = get_single_datetime(start_timetamp_fake)
        
        new_actions.append(fake_act)
    
    return new_actions

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
        
        # Queue names
        self.detection_queue = "det#vaccine_monitor_prod_out"
        self.output_queue = "action-ml#results-prod"
        # Khởi tạo Redis connection
        self.redis_client = None
        self.output_redis = redis.Redis(**self.output_config)

        os.makedirs('logs/prod', exist_ok=True)
        
        self.extra_processor = ExtraInfoProcessor()
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
    
    def push_to_queue(self, data):
        self.output_redis.lpush(self.output_queue, json.dumps(data))
        print("===>>> Pushed message to forward queue:", self.output_queue)

    def push_to_queue_forward(self, data):
        forward_data = data['ml_results']
        self.output_redis.lpush(self.output_queue, json.dumps(forward_data))
        print("===>>> Pushed message to forward queue:", self.output_queue)
    
    def get_result(self, raw_msg, moment_time):
        data = {}
        data['alarm'] = raw_msg['alarm']['raw_alarm']['alarm']
        data['record_id'] = raw_msg['alarm']['raw_alarm']['record_id']
        data['serial'] = raw_msg['alarm']['raw_alarm']['serial']
        data['service_id'] = raw_msg['config']['service_id']
        data['home_id'] = raw_msg['alarm']['raw_alarm']['home_id']
        data['url_video'] = raw_msg['alarm']['record_url']

        h = random.randint(2, 12)
        dt_utc = datetime.strptime(raw_msg['alarm']['raw_alarm']['send_at'], '%Y%m%d%H%M%S') + timedelta(hours=h)

        data['timestamp'] = int(dt_utc.timestamp())
        
        data['config'] = {}

        data['actions'] = []
        data['extra_info'] = []

        return data

    def _cut_video(self, video_path, start_time, end_time):
        pass

    def process_action_queue(self):
        """Xử lý action queue"""
        result = self.pop_message_from_queue(self.detection_queue)
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
                        
            print(f"\n\n## GET message: {parsed_message['alarm']['serial']}, record_id: {parsed_message['alarm']['raw_alarm']['record_id']}")

            # Log full origin message dạng JSON
            self.log_to_file(parsed_message, "monitor_detection")

            record_info = parsed_message['alarm']['raw_alarm']['record_list'][0]
            raw_alarm = parsed_message['alarm']['raw_alarm']
            serial = raw_alarm['serial']
            moment_time = record_info['moment_time']
            recordtimestamp = record_info['recordtimestamp']

            video_start_time = raw_alarm["video_start_time"]
            video_end_time = raw_alarm["video_end_time"]
            alcohol_path = parsed_message['alcohol']
            hand_path = parsed_message['hand']
            pose_path = parsed_message['pose']
            
            try:
                fps = parsed_message['fps']
            except:
                fps = 13
            

            try:
                target_action = raw_alarm['target_action']
                vaccine_info = raw_alarm['vaccine_info']
            except:
                self.log_to_file(
                    {"error": "missing vaccine/target info", "message": parsed_message},
                    "monitor_invalid_format", True)
                return False
            
            runner = Runner(serial, alcohol_paths=alcohol_path,
                            hand_paths=hand_path,
                            pose_paths=pose_path, fps = fps)
            
            print(f"==>> Start processing message: {serial}, video_start_time: {video_start_time}, video_end_time: {video_end_time}")
            
            result_msg = parsed_message.copy()
            try:
                result_msg['ml_results'] = self.get_result(parsed_message, moment_time)
                result_msg['root_fields'] = {'config': parsed_message['config']}
            except:
                self.log_to_file({"error": "can't get field result", "message": parsed_message},
                                 "monitor_invalid_format", True)
                return False
            
            # no target, no vaccine
            if len(target_action) == 0 or len(vaccine_info) == 0:
                self.log_to_file(result_msg, "no_target_vaccine")
                self.push_to_queue(result_msg)
                return False
            
            # # process
            # ======================================================
            actions = runner.run(video_start_time, video_end_time)
            print(raw_alarm['serial'], raw_alarm['send_at'], json.dumps(actions, ensure_ascii=False, indent=2).encode('utf8').decode())
            
            result_msg['ml_results']['actions'] = actions
            self.log_to_file(result_msg, "monitor_wo_extra")

            extra_info = self.extra_processor.process(actions, target_action, vaccine_info)
            result_msg['ml_results']['extra_info'] = extra_info

            keep = False
            if self.debug:
                for act in target_action:
                    if act['action_id'] == "1":
                        if 0 in act['value']:
                            keep = True

                if len(actions) > 0:
                    for extra in extra_info:
                        if "0" not in extra['error_id']:
                            keep = True
                
                if len(actions) == 0:
                    keep = True
            
            keep = False

            if keep:
                self.log_to_file(result_msg, "monitor_keep_result")

            if len(actions) == 0:
                self.log_to_file(result_msg, "monitor_not_result")
                if not keep:
                    self.push_to_queue(result_msg)
                    self.log_to_file(result_msg, "monitor_all_result")

            else:
                self.log_to_file(result_msg, "monitor_result")
                if not keep:
                    self.push_to_queue(result_msg)
                    self.log_to_file(result_msg, "monitor_all_result")
            
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
            action_processed = self.process_action_queue()
            
            if not action_processed:
                time.sleep(check_interval)
                

def main():
    """Hàm main"""
    consumer = RedisQueueConsumer()
    consumer.run_consumer()
    return 0

if __name__ == "__main__":
    exit(main())