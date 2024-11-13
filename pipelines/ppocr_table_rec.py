import os
import cv2
import time
import yaml
import copy
import numpy as np
import onnxruntime as ort
from collections import namedtuple

from pipelines.utils.utility import expand
from pipelines.utils.matcher import TableMatch

from .utils.text_system import TextDetector, TextRecognizer, sorted_boxes
from .utils.table_system import TableSystem


class PPOCRTableRec(object):
    """PaddleOCR Table recognition pipeline."""

    def __init__(self, model_config) -> None:
        # Load config
        if isinstance(model_config, str):
            if not os.path.isfile(model_config):
                raise FileNotFoundError(f"[ERROR] Config file not found: {model_config}")
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            self.config = model_config
        else:
            raise ValueError(f"[ERROR] Unknown config type: {type}")
        self.ppocr_configs = self.config['ppocr_models']
        self.table_configs = self.config['table_model']
        self.use_gpu = self.config['use_gpu']

        self.det_net = self.load_model(self.ppocr_configs["det_model"]["det_model_path"])
        self.rec_net = self.load_model(self.ppocr_configs["rec_model"]["rec_model_path"])
        self.table_net = self.load_model(self.table_configs["table_model_path"])
        self.match = TableMatch(filter_ocr_result=True)

        ppocr_args = self.parse_ppocr_args()
        table_args = self.parse_table_args()
        self.text_detector = TextDetector(ppocr_args)
        self.text_recognizer = TextRecognizer(ppocr_args)
        self.table_sys = TableSystem(table_args)

    def load_model(self, model_path):
        if model_path is None and not os.path.exists(model_path):
            raise ValueError(
                "not find layout model file path {}".format(model_path)
            )

        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )
        if self.use_gpu:
            self.providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
        else:
            self.providers=["CPUExecutionProvider"]

        net = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )
        return net

    def parse_ppocr_args(self):
        args = {}
        for key, value in self.ppocr_configs.items():
            if isinstance(value, dict):
                args.update(value)
            else:
                args[key] = value
        args['det_model'] = self.det_net
        args['rec_model'] = self.rec_net
        return namedtuple('Args', args.keys())(**args)

    def parse_table_args(self):
        args = {}
        for key, value in self.table_configs.items():
            args[key] = value
        args['table_model'] = self.table_net
        return namedtuple('Args', args.keys())(**args)
    
    def text_system(self, img):
        h, w = img.shape[:2]
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0) : int(y1), int(x0) : int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        return dt_boxes, rec_res, det_elapse, rec_elapse

    def predict(self, image_path, return_ocr_result_in_table=False, is_visualize=True, save_result_dir=None):
        """
        Predict shapes from image
        """
        result = dict()
        time_dict = {"det": 0, "rec": 0, "table": 0, "all": 0, "match": 0}
        img = cv2.imread(image_path)

        start = time.time()

        structure_res, elapse = self.table_sys(copy.deepcopy(img))
        result["cell_bbox"] = structure_res[1].tolist()
        time_dict["table"] = elapse

        dt_boxes, rec_res, det_elapse, rec_elapse = self.text_system(copy.deepcopy(img))
        time_dict["det"] = det_elapse
        time_dict["rec"] = rec_elapse

        if return_ocr_result_in_table:
            result["boxes"] = [x.tolist() for x in dt_boxes]
            result["rec_res"] = rec_res

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict["match"] = toc - tic
        result["html"] = pred_html
        end = time.time()
        time_dict["all"] = end - start
        return result, time_dict
