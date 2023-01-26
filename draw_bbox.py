import cv2


def draw_bounding_boxes(image_path, txt_path):

    # read image
    image = cv2.imread(image_path)
    # read txt file
    with open(txt_path, "r") as f:
        lines = f.readlines()

    class_names = ["boneanomaly", "bonelesion", "foreignbody", "fracture",
                "metal", "periostealreaction", "pronatorsign", "softtissue", "text"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255,
                                                                    0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0), (255, 128, 0)]
    # parse bounding box information from txt file
    for line in lines:
        parts = line.strip().split(" ")
        class_id = parts[0]
        class_name = class_names[int(class_id)]
        color = colors[int(class_id)]
        x, y, w, h = map(float, parts[1:5])
        x1 = int((x - w / 2) * image.shape[1])
        y1 = int((y - h / 2) * image.shape[0])
        x2 = int((x + w / 2) * image.shape[1])
        y2 = int((y + h / 2) * image.shape[0])

        # draw bounding box on image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_name, (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    # save the output image
    cv2.imwrite("output2.jpg", image)

image_path = r"D:\Downloads\test images\3799_0949300875_03_WRI-L1_M010.png"
txt_path = r"D:\Downloads\test\3799_0949300875_03_WRI-L1_M010.txt"
draw_bounding_boxes(image_path, txt_path)
