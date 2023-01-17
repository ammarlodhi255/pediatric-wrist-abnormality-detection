import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import shutil
from tqdm import tqdm
import argparse
import yaml
import matplotlib.pyplot as plt
from xml.etree import ElementTree as et
from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    
def write_results(path, text):
    file = open(path, "w")
    file.write(text)
    file.close()
    
def parse_ground_truth(name):
    annot_file_path = f'data/ppe/dataset/labels/test/{name}.xml'
    tree = et.parse(annot_file_path)
    root = tree.getroot()
    text = ''
    for member in root.findall('object'):
        classText = (member.find('name').text.replace('\n', '')).replace(' ', '')
        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)    
        
        text += f'{classText} {xmin} {ymin} {xmax} {ymax}\n'
    return text
def delete_files():
    path = f'mAP_utils/input/detection-results'
    if os.path.exists(path) and len(os.listdir(path)) != 0:
        shutil.rmtree(path)	
    os.mkdir(path)
    
    path = f'mAP_utils/input/ground-truth'
    if os.path.exists(path) and len(os.listdir(path)) != 0:
        shutil.rmtree(path)	
    os.mkdir(path)
    
    path = f'mAP_utils/input/images-optional'
    if os.path.exists(path) and len(os.listdir(path)) != 0:
        shutil.rmtree(path)	
    os.mkdir(path)
    print('Index Updated...')
        
def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        default='data/ppe/dataset/images/test',
        help='folder path to input input image (one image or a folder path)'
    )
    parser.add_argument(
        '-c', '--config', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default=None,
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', default='outputs/training/fasterrcnn_resnet50_fpn_ppe/best_model.pth',
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.5, type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show-image', dest='show_image', action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', dest='mpl_show', action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    # np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['config'] is not None:
        with open(args['config']) as file:
            data_configs = yaml.safe_load(file)
    NUM_CLASSES = 10
    CLASSES = ['boneanomaly','bonelesion','foreignbody','fracture','metal','periostealreaction','pronatorsign', 'softtissue','text']

    DEVICE = args['device']
    # OUT_DIR = set_infer_dir()
    
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        try:
            print('Building from model name arguments...')
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    delete_files()
    
    DIR_TEST = args['input']
    test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    for i in tqdm(range(len(test_images))):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold].astype(np.float32)
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        text = ''
        for j, box in enumerate(draw_boxes):
            text += f'{pred_classes[j]} {round(float(scores[j]), 6)} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n'
            
        write_results(f'mAP_utils/input/detection-results/{image_name}.txt', text)
        text = parse_ground_truth(image_name)
        write_results(f'mAP_utils/input/ground-truth/{image_name}.txt', text)
        cv2.imwrite(f'mAP_utils/input/images-optional/{image_name}.png', orig_image)
    
if __name__ == '__main__':
    args = parse_opt()
    main(args)
    print('')
    os.system('python mAP_utils/main.py')