
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def postprocess(self, output: np.ndarray, scale: float, pad: Tuple[int, int]):
        """Hậu xử lý output cho 1 ảnh"""
        predictions = output[0].T
        boxes = predictions[:, :4]
        confidences = predictions[:, 4]
        
        valid_indices = confidences > self.conf_threshold
        if not np.any(valid_indices):
            return []
        
        boxes = boxes[valid_indices]
        confidences = confidences[valid_indices]
        
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        boxes = np.column_stack([x1, y1, x2, y2])
        keep_indices = self.nms(boxes, confidences, self.iou_threshold)
        
        if not keep_indices:
            return []
        
        detections = []
        pad_h, pad_w = pad
        
        for i in keep_indices:
            x1, y1, x2, y2 = boxes[i]
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            detection = DetectionsOuput(bbox=[int(x1), int(y1), int(x2), int(y2)],
                                        conf=confidences[i],
                                        class_id=0)
            detections.append(detection)
            # detections.append({
            #     'bbox': [int(x1), int(y1), int(x2), int(y2)],
            #     'confidence': float(confidences[i]),
            #     'class_id': 0  # Chỉ có 1 class
            # })
        
        return detections

    def postprocess_pose(self, output, scale=1.0, pad=(0, 0)):
        """Hậu xử lý cho pose estimation (giữ nguyên logic cũ)"""
        if len(output.shape) == 3:
            output = output[0]
        
        if output.shape[0] == 56:
            output = output.T
        
        boxes = output[:, :4]
        scores = output[:, 4]
        keypoints = output[:, 5:]
        
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        keypoints = keypoints[mask]
        
        if len(boxes) == 0:
            return []
        
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        keep_indices = self.nms(boxes_xyxy, scores, self.iou_threshold)
        
        boxes_xyxy = boxes_xyxy[keep_indices]
        scores = scores[keep_indices]
        keypoints = keypoints[keep_indices]
        
        num_detections = len(boxes_xyxy)
        keypoints = keypoints.reshape(num_detections, 17, 3)
        
        pad_h, pad_w = pad
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
        
        keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_w) / scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_h) / scale
        
        results = []
        for i in range(num_detections):
            person_kpts = Keypoints([], [])
            for kpt in keypoints[i]:
                person_kpts.xy.append((int(kpt[0]), int(kpt[1])))
                person_kpts.conf.append(kpt[2])


            pose = PoseOutput(bbox=boxes_xyxy[i].tolist(),
                              conf=float(scores[i]),
                              keypoints=person_kpts,
                              class_id=0) # shape (17, 3)
            
            results.append(pose)
        
        return results

    def predict_batch(self, images: List[np.ndarray], preprocess = True, scale = None, pad_w = None, pad_h = None):
        """
        Inference batch ảnh với CUDA context management
        
        Args:
            images: List các ảnh (BGR format), tối đa max_batch_size ảnh
            
        Returns:
            List kết quả detection/pose cho từng ảnh
        """
        batch_size = len(images)
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} vượt quá max_batch_size {self.max_batch_size}")
        
        self.ctx.push()
        try:
            # Preprocessing
            t1 = time.time()
            if not preprocess:
                scales, pads = scale, (pad_w, pad_h)
                batch_processed = images
            else:
                # batch_processed, scales, pads = self.preprocess_batch(images)
                # batch_processed, scales, pads = self.preprocess_batch_vectorized(images)
                batch_processed, scales, pads = self.preprocess_batch_parallel(images)
            t2 = time.time()

            # Pad batch nếu cần (TensorRT yêu cầu đúng batch size)
            if batch_size < self.max_batch_size:
                pad_size = self.max_batch_size - batch_size
                padding = np.zeros((pad_size, *batch_processed.shape[1:]), dtype=np.float32)
                batch_processed = np.vstack([batch_processed, padding])

            # Copy to GPU
            np.copyto(self.inputs[0]['host'], batch_processed.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Set batch size cho context
            self.context.set_binding_shape(0, batch_processed.shape)
            
            # Run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Copy output từ GPU
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            
            # Reshape output
            output_shape = list(self.output_shape)
            output_shape[0] = self.max_batch_size
            output = self.outputs[0]['host'].reshape(output_shape)
            
            t3 = time.time()
            # Postprocess từng ảnh trong batch
            results = []
            for i in range(batch_size):
                if self.is_pose:
                    result = self.postprocess_pose(output[i:i+1], scales[i], pads[i])
                else:
                    result = self.postprocess(output[i:i+1], scales[i], pads[i])
                results.append(result)
            t4 = time.time()

            # print(f"preprocess-time {round(t2 - t1, 3)}")
            # print(f"infer-time {round(t3 - t2, 3)}")
            # print(f"postprocess cpu-time {round(t4 - t3, 3)}")
            return results
        
            
        finally:
            self.ctx.pop()