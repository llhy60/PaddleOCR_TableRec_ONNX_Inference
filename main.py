import os
import argparse
from pipelines.ppocr_table_rec import PPOCRTableRec
from pipelines.utils.utility import handle_html2excel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='./configs/ch_ppocr_table_rec.yaml')
    parser.add_argument('--image_path', type=str, default='images/table.jpg')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize output')
    parser.add_argument('--save_dir', type=str, default='./result')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.save_dir
    image_path = args.image_path
    model_config = args.model_config
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(image_path)
    shufix_name, _ = os.path.splitext(file_name)
    result_filename = f"{save_dir}/{shufix_name}.xlsx"

    ppocr = PPOCRTableRec(model_config)
    result, time_dict = ppocr.predict(image_path)
    excel_data = handle_html2excel(result)

    with open(result_filename, 'wb') as f:
        f.write(excel_data)
    print(f"[INFO] Result saved to {result_filename}")
    print(f"[INFO] Time cost: {time_dict}") 
    
    
        