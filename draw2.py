import cv2
import numpy as np

class_names = ["boneanomaly", "bonelesion", "foreignbody", "fracture",
               "metal", "periostealreaction", "pronatorsign", "softtissue", "text"]
colors = np.random.uniform(0, 255, size=(len(class_names), 3))


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
        cv2.putText(image, class_name, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (xmin, ymin-30), (xmin+80, ymin), color, -1)
        cv2.putText(image, class_name, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the output image
    cv2.imwrite(output_path, image)
    print("Output image saved to", output_path)


draw_boxes(r"D:\Downloads\test images\3799_0949300875_03_WRI-L1_M010.png",
           r"D:\Downloads\test\3799_0949300875_03_WRI-L1_M010.txt", r"D:\Downloads\out.png")
