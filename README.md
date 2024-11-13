# PaddleOCR_TableRec_ONNX_Inference
基于PaddleOCR的SLANet表格识别算法，实现表格识别onnx模型推理
## 环境配置
- numpy
- openpyxl
- onnxruntime
- opencv-python
- bs4
- pillow
- shapely
- pyclipper
- html
- six
## 代码说明
- main.py为表格识别主程序
- pipelines/ppocr_table_rec.py为表格识别pipeline
## 推理命令
python main.py --image_path ./images/table3.jpg --visualize