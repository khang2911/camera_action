from datetime import datetime, timedelta, timezone
import os
import redis
import json
import time

rc = redis.Redis("172.24.178.105", 6379, db = 0)

def log_to_file(message_data, file):
    os.makedirs(f"logs_prod/{file}", exist_ok=True)
    try:
        if 'message' in message_data:
            dt_utc = datetime.strptime(message_data['message']['member']['alarm']['raw_alarm']['send_at'], '%Y%m%d%H%M%S')
        else:
            dt_utc = datetime.strptime(message_data['member']['alarm']['raw_alarm']['send_at'], '%Y%m%d%H%M%S')
            
        # dt_hcm = dt_utc.astimezone(timezone(timedelta(hours=7)))
        dt_string = dt_utc.strftime('%Y-%m-%d')
        print(f"--> INFO: Log to {file} ...")
        with open(f'logs_prod/{file}/{dt_string}.log', 'a+', encoding='utf-8') as f:
            f.write(json.dumps(message_data, ensure_ascii=False) + '\n')
    
    except Exception as e:
        print(f"INFO: Log to error_message ...")
        with open(f'logs_prod/error_message.log', 'a+', encoding='utf-8') as f:
            f.write(f"{message_data}" + '\n')

while True:
    message = rc.zpopmin("ml#action_recognition#fli_action_prod_new")
    if message:
        member, score = message[0]
        
        data = json.loads(member)
        record_info = data['alarm']['raw_alarm']['record_list'][0]
        raw_alarm = data['alarm']['raw_alarm']

        print(f"## GET message: {data['alarm']['serial']}, Score: {score}")
        # Giả sử message_data là một dict chứa dữ liệu cần log
        message_data = {
            'member': data,
            'score': score,
        }

        log_to_file(message_data, "origin_msg")

        
        config = data['config']
        actions_config = config['external_data']['actions']
        actions = []
        for action in actions_config:
            actions.append(action['id'])
        
        # Process message based on actions
        if "1" in actions or "2" in actions:
            rc.lpush('ml#vaccine_monitor_prod', json.dumps(data))
            log_to_file(message_data, "monitor_alarm_msg")

        elif any(action in actions for action in ["3", "4", "5", "6"]):
            rc.lpush('ml#vaccine_protection_prod', json.dumps(data))
            log_to_file(message_data, "protection_alarm_msg")
        else:
            log_to_file(message_data, "unknown_alarm_msg")

        # break
    else:
        time.sleep(1)  # Nếu không có message, đợi 1 giây trước khi kiểm tra lại