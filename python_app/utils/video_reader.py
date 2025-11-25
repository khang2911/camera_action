import cv2
import time
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import numpy as np

@dataclass
class VideoInfo:
    """Thông tin video"""
    path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration: float  # seconds

class VideoReader:
    """
    Video Reader với khả năng đọc nhiều video liên tiếp và filter theo timestamp
    """
    def __init__(self, video_paths: List[str], moment_times: List[float], durations: List[float], serial):
        """
        Args:
            video_paths: List đường dẫn video
            moment_times: List timestamp bắt đầu của mỗi video
            durations: List thời lượng thực tế của mỗi video (seconds)
            serial: Serial ID
        """
        if len(video_paths) != len(moment_times):
            raise ValueError("video_paths và moment_times phải có cùng độ dài")
        
        self.video_paths = video_paths
        self.moment_times = moment_times
        self.serial = serial
        self.durations = durations
        self.current_video_idx = 0
        self.current_cap = None
        self.current_frame_idx = 0

        self.vaccine_frame_index = 0

        self.avg_fps = 0
        
        # Load info của tất cả videos
        self.video_infos = self._load_video_infos()
    
    def _load_video_infos(self) -> List[VideoInfo]:
        """Load thông tin của tất cả videos"""
        infos = []
        for i, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(total_frames / self.durations[i])  # Tính FPS từ total_frames / duration
            
            infos.append(VideoInfo(
                path=path,
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
                duration=self.durations[i]
            ))
            cap.release()

            self.avg_fps += fps
        self.avg_fps/=len(self.video_paths)

        return infos
    
    def _open_next_video(self) -> bool:
        """Mở video tiếp theo"""
        if self.current_cap:
            self.current_cap.release()
        
        if self.current_video_idx >= len(self.video_paths):
            return False
        
        path = self.video_paths[self.current_video_idx]
        self.current_cap = cv2.VideoCapture(path)
        self.current_frame_idx = 0
        
        if not self.current_cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        
        # print(f"Opened video: {path}")
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[any], int, float]:
        """
        Đọc frame tiếp theo
        
        Returns:
            (ret, frame, frame_index, timestamp)
        """
        # Mở video đầu tiên nếu chưa mở
        if self.current_cap is None:
            if not self._open_next_video():
                return False, None, 0, 0.0
        
        while True:
            ret, frame = self.current_cap.read()
            frame_index = self.current_frame_idx  # Lưu frame index TRƯỚC khi tăng
            self.current_frame_idx += 1  # Tăng ngay sau khi đọc
            
            if not ret:
                # Frame lỗi hoặc hết video
                info = self.video_infos[self.current_video_idx]
                
                if frame_index >= info.total_frames:
                    # Hết video hiện tại, chuyển sang video tiếp theo
                    self.current_video_idx += 1
                    if not self._open_next_video():
                        return False, None, 0, 0.0
                    continue
                else:
                    # Frame lỗi, thử tiếp
                    print(f"Skipped bad frame: {frame_index}")
                    continue
            
            # Tính timestamp hiện tại (dùng frame_index đã lưu)
            info = self.video_infos[self.current_video_idx]
            moment_time = self.moment_times[self.current_video_idx]
            current_timestamp = moment_time + frame_index / info.fps
            
            return True, frame, frame_index, current_timestamp
    
    def _seek_to_timestamp(self, target_timestamp: float) -> bool:
        """
        Seek đến timestamp gần nhất với target
        
        Args:
            target_timestamp: Timestamp cần seek tới
            
        Returns:
            True nếu seek thành công
        """
        # Tìm video chứa timestamp này
        for idx, (path, moment_time) in enumerate(zip(self.video_paths, self.moment_times)):
            info = self.video_infos[idx]
            video_end_time = moment_time + info.duration
            
            if moment_time <= target_timestamp <= video_end_time:
                # Đây là video cần mở
                if self.current_video_idx != idx or self.current_cap is None:
                    # Đóng video cũ nếu có
                    if self.current_cap:
                        self.current_cap.release()
                    
                    # Mở video mới
                    self.current_video_idx = idx
                    self.current_cap = cv2.VideoCapture(self.video_paths[idx])
                    self.current_frame_idx = 0
                    
                    if not self.current_cap.isOpened():
                        print(f"Cannot open video: {self.video_paths[idx]}")
                        return False
                    
                    # print(f"Opened video: {self.video_paths[idx]}")
                
                # Tính frame cần seek tới
                time_in_video = target_timestamp - moment_time
                # print(info.fps)
                target_frame = int(time_in_video * info.fps)
                
                # Seek đến frame đó
                self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.current_frame_idx = target_frame
                
                print(f"Seeked to video {idx}, frame {target_frame}, timestamp {target_timestamp:.2f}s")
                return True
        
        print(f"Timestamp {target_timestamp} not found in any video")
        return False

    def preprocess_tensorrt(self, image, input_size):
        """Tiền xử lý ảnh đơn (giữ lại cho tương thích)"""
        input_h, input_w = input_size[1], input_size[0]
        img_h, img_w = image.shape[:2]
        
        scale = min(input_h / img_h, input_w / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        resized_img = cv2.resize(image, (new_w, new_h))
        
        pad_h = (input_h - new_h) // 2
        pad_w = (input_w - new_w) // 2
        
        padded_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_img
        
        processed_img = padded_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)
        
        return processed_img, scale, (pad_h, pad_w)

    def read_range(self, start_timestamp: float, end_timestamp: float) -> Tuple[bool, Optional[any], int, float]:
        """
        Đọc frame tiếp theo trong khoảng timestamp
        Gọi liên tục trong while True cho đến khi ret = False
        
        Args:
            start_timestamp: Timestamp bắt đầu (chỉ dùng lần đầu để seek)
            end_timestamp: Timestamp kết thúc
        
        Returns:
            (ret, frame, frame_index, timestamp)
        """
        # Seek lần đầu tiên
        if not hasattr(self, '_seeked'):
            if not self._seek_to_timestamp(start_timestamp):
                print(f"Cannot seek to timestamp {start_timestamp}")
                return False, None, 0, 0.0
            self._seeked = True
        
        ret, frame, frame_index, timestamp = self.read_frame()
        
        if not ret:
            return False, None, 0, 0.0
        
        # Dừng nếu vượt quá end_timestamp
        if timestamp > end_timestamp:
            return False, None, frame_index, timestamp
        
        self.vaccine_frame_index +=1
        return True, frame, self.vaccine_frame_index, timestamp
    
    def get_info(self) -> List[VideoInfo]:
        """Lấy thông tin tất cả videos"""
        return self.video_infos
    
    def reset(self):
        """Reset reader về đầu"""
        if self.current_cap:
            self.current_cap.release()
        self.current_cap = None
        self.current_video_idx = 0
        self.current_frame_idx = 0
        # Reset seek flag
        if hasattr(self, '_seeked'):
            delattr(self, '_seeked')
    
    def release(self):
        """Giải phóng resources"""
        if self.current_cap:
            self.current_cap.release()
        self.current_cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ============ Cách sử dụng ============

# Ví dụ 1: Đọc tất cả frames đơn giản
def example1():
    video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
    moment_times = [0.0, 900.0, 1800.0]
    durations = [900.0, 900.0, 900.0]
    serial = "camera_01"
    
    with VideoReader(video_paths, moment_times, durations, serial) as reader:
        while True:
            ret, frame, frame_idx, timestamp = reader.read_frame()
            if not ret:
                break
            
            print(f"Frame {frame_idx} at {timestamp:.2f}s")
            # Xử lý frame...


# Ví dụ 2: Đọc trong khoảng timestamp (KHUYẾN NGHỊ)
def example2():
    video_paths = ["video1.mp4", "video2.mp4"]
    moment_times = [0.0, 900.0]
    durations = [900.0, 900.0]
    serial = "camera_01"
    start_timestamp = 100.0
    end_timestamp = 1200.0
    
    with VideoReader(video_paths, moment_times, durations, serial) as reader:
        while True:
            ret, frame, frame_idx, timestamp = reader.read_range(start_timestamp, end_timestamp)
            if not ret:
                break
            
            print(f"Frame {frame_idx} at {timestamp:.2f}s")
            # Xử lý frame...
