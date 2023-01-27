import cv2
import numpy as np

class_names = ["boneanomaly", "bonelesion", "foreignbody", "fracture",
               "metal", "periostealreaction", "pronatorsign", "softtissue", "text"]
colors = [[255, 0, 0], [0, 255, 0], [255, 178, 29], [255, 178, 29], [
    207, 210, 49], [71, 249, 10], [255, 128, 0], [26, 147, 52], [26, 147, 52]]


def draw_boxes(image_path, txt_path, output_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Read the bounding box information from the txt file
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # Draw the bounding boxes on the image
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x = float(data[1])
        y = float(data[2])
        w = float(data[3])
        h = float(data[4])

        xmin = int((x - w/2) * image.shape[1])
        ymin = int((y - h/2) * image.shape[0])

        xmax = int((x + w/2) * image.shape[1])
        ymax = int((y + h/2) * image.shape[0])


        color = tuple(colors[class_id])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        class_name = class_names[class_id]
    # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    # Draw the text rectangle
        cv2.rectangle(image, (xmin, ymin-text_height-20),
                  (xmin+text_width+20, ymin), color, -1)
        cv2.putText(image, class_name, (xmin+10, ymin-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
    cv2.imwrite(output_path, image)
    print("Output image saved to", output_path)


draw_boxes(r"D:\Downloads\test images\3799_0949300875_03_WRI-L1_M010.png",
           r"D:\Downloads\test\3799_0949300875_03_WRI-L1_M010.txt", r"D:\Downloads\out.png")
