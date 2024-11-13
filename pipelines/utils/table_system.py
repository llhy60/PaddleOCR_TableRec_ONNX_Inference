import time
import numpy as np

from pipelines.utils.operators import create_operators, create_predictor
from pipelines.utils.table_postprocess import TableLabelDecode


class TableSystem(object):
    def __init__(self, args):
        pre_process_list = args.preprocess_params
        post_process_dict = args.postprocess_params

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = TableLabelDecode(**post_process_dict)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = create_predictor(args, "table") 

    def pre_process(self, data, ops=None):
        """transform"""
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def __call__(self, img):
        data = {"image": img}
        data = self.pre_process(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()

        starttime = time.time()
        input_dict, preds = {}, {}
        input_dict[self.input_tensor.name] = img
        outputs = self.predictor.run(self.output_tensors, input_dict)

        preds["structure_probs"] = outputs[1]
        preds["loc_preds"] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result["structure_batch_list"][0]
        bbox_list = post_result["bbox_batch_list"][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = (
            ["<html>", "<body>", "<table>"]
            + structure_str_list
            + ["</table>", "</body>", "</html>"]
        )
        elapse = time.time() - starttime
        return (structure_str_list, bbox_list), elapse
