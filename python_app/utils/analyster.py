import json


class ExtraInfoProcessor:
    ERROR_MAP = {
        "0": "Không vi phạm",
        "1": "Lắc vắc xin không được lắc",
        "2": "Không lắc vắc xin",
        "3": "Lắc vắc xin không đủ thời gian (không kỹ)",
        "4": "Không sát khuẩn tay",
        "5": "Sát khuẩn tay thiếu (nhiều)",
        "6": "Sát khuẩn tay thiếu (ít)",
        "7": "Luân chuyển vắc xin giữa các thiết bị quá thời gian quy định",
        "8": "Mở phích lạnh quá thời gian quy định",
        "9": "Mở tủ đứng quá thời gian quy định",
        "10": "Mở tủ ngang quá thời gian quy định",
    }

    def __init__(self):
        pass

    def _get_sat_khuan_tay_error(self, num_vaccine, num_washing, target_count):
        # print(num_vaccine, num_washing, target_count)
        if num_washing >= target_count:
            return ['0']
        
        if num_washing == 0:
            return ['4']
        
        if num_vaccine == 1:
            if num_washing == 1:
                return ['5']
            elif num_washing >= 2:
                return ['6']
            
        elif num_vaccine == 2:
            if num_washing <= 2:
                return ['5']
            elif num_washing >=3:
                return ['6']
            
        elif num_vaccine == 3:
            if num_washing <= 3:
                return ['5']
            elif num_washing >= 4:
                return ['6']
        
        else:
            return ['0']
    
    # ---------- Rule cụ thể ----------
    def _check_lac_vaccine(self, shaking_actions_predict, shaking_time_targets, num_vaccine):
        # Nếu số lần dự đoán ít hơn số target → lỗi 2
        if len(shaking_actions_predict) < len(shaking_time_targets):
            # print(shaking_actions_predict, shaking_time_targets)
            return ['2']
        

        
        # # Nếu số lần dự đoán nhiều hơn số target → lỗi 1: lac vaccine khogn duoc lac
        # if len(shaking_actions_predict) > len(shaking_time_targets):
        #     return ['1']

        # Nếu số lượng bằng nhau thì check thời gian
        if len(shaking_actions_predict) == len(shaking_time_targets):
            for i, shaking_predict in enumerate(shaking_actions_predict):
                if shaking_predict['duration'] < shaking_time_targets[i]:
                    return ['3']

        # Nếu không có lỗi nào
        return ['0']

    def process(self, actions, target_action=None, vaccine_info = None):
        """
        Gom lỗi theo action_id, loại bỏ "0" nếu có lỗi khác.
        target_action: list chứa target yêu cầu (vd: số lần sát khuẩn tay).
        """
        extra_data = []
        if len(actions) == 0:
            return []
        
        vaccine_list = vaccine_info['Indication']
        num_vaccine = len(vaccine_list)

        shaking_target = None
        washing_count = None

        for t in target_action:
            if t.get("action_id") == "2" and t.get("type") == "count":
                washing_count = t.get("value")
            
            if t.get("action_id") == "1" and t.get("type") == "time":
                shaking_target = t.get("value")
        
        if washing_count is not None:
            washing_error = self._check_sat_khuan(actions, washing_count, num_vaccine)
            extra_data.append({"action_id": '2',
                        "action_name": 'Sát khuẩn tay',
                        "error_id": washing_error})
        
        if shaking_target is not None:
            shaking_actions_predict = []
            shaking_time_targets = []
            for act in actions:
                if act['action_id'] == '1':
                    shaking_actions_predict.append(act)
            
            for act in shaking_target:
                if act != 0:
                    shaking_time_targets.append(act)

            shaking_actions_predict_filtered = self.filter_shaking(shaking_actions_predict, len(shaking_time_targets))
            shaking_errors = self._check_lac_vaccine(shaking_actions_predict_filtered, shaking_time_targets, num_vaccine)
            
            # if len(shaking_errors) > 0:
            extra_data.append({"action_id": '1',
                        "action_name": 'Lắc vaccine',
                        "error_id": shaking_errors})
    
        return extra_data

    def filter_shaking(self, shaking_actions, num_vaccine):
        if len(shaking_actions) <= num_vaccine:
            return shaking_actions
        
        sorted_actions = sorted(shaking_actions, key=lambda x: x["duration"], reverse=True)
        # print(sorted_actions)
        cut_sorted_actions = sorted_actions[:num_vaccine]
        return cut_sorted_actions

    def _check_sat_khuan(self, actions, washing_count, num_vaccine):
        # đếm số lần sát khuẩn tay trong actions
        num_washing = sum(1 for a in actions if a.get("action_id") == "2")
        error_list = self._get_sat_khuan_tay_error(num_vaccine, num_washing, washing_count)
        return error_list


if __name__ == "__main__":
    # with open("logs/msg.log", "r") as f:
    #     lines = f.readlines()

    # for line in lines:
        # data = json.loads(line)
        # if not line:
        #     break

        # actions = data['ml_results']['actions']
        # print(data['alarm']['raw_alarm']['record_id'])

        actions = [{
                        "action_id": "1",
                        "action_name": "Sát khuẩn tay",
                        "start_time": "2025-11-16T13:36:05+07:00",
                        "duration": 15.8461538461538463
                    },
                    {
                        "action_id": "1",
                        "action_name": "Sát khuẩn tay",
                        "start_time": "2025-11-16T13:36:59+07:00",
                        "duration": 14
                    }]
                   
        target_action =  [ 
            {"action_id": "1", "action_name": "Lắc vaccin", "type": "time", "value": [15, 15, 15, 0]}, 
            {"action_id": "2", "action_name": "Sát khuẩn tay", "type": "count", "value": 3 } ]
        vaccine_info = {
        "Indication": [
            {
            "Sku": "00038255",
            "Odac": "N7JlW6VG87qNVARA",
            "Taxonomies": "CÚM",
            "VaccineName": "VAXIGRIP TETRA"
            },
            # {
            #   "Sku": "00038226",
            #   "Odac": "N7JlW6VG87qNVCRC",
            #   "Taxonomies": " ROTARIX",
            #   "VaccineName": "ROTARIX VIAL 1.5ML 1'S "
            # }
        ]}
        a = ExtraInfoProcessor()

        r = a.process(actions, target_action, vaccine_info=vaccine_info)
        print(r)
        # break