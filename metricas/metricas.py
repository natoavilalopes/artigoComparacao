import cv2
from ultralytics import YOLO
import numpy as np
import random
import os


# - - - - -  logs

nome = "teste"

txt = f'logs/{nome}.txt'
with open(txt, 'w') as f:
    f.write('     bboxInfer          classeInfer           bboxTrue            classTrue            IoU      \n')

file_object = open(txt, 'a')




path = "/home/renato/PycharmProjects/acuracia/"
interesse = ["car", "truck", "person"]

def calculate_iou(box1, box2):
    # Box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-9)
    return iou

def loadClass():
    classes = []
    rede = "dnn_model/classes.txt"
    with open(path + rede, "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)
    return classes

def class_colors(names):
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}

# def calculate_metrics(true_positives, false_negatives, true_negatives):
#     precision = true_positives / (true_positives + false_negatives + 1e-9)
#     recall = true_positives / (true_positives + true_negatives + 1e-9)
#     f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
#     return precision, recall, f1_score

def calculate_metrics(true_positives, false_negatives, true_negatives):
    precision = true_positives / (true_positives + false_negatives + 1e-9)
    recall = true_positives / (true_positives + true_negatives + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    return precision, recall, f1_score


names = loadClass()
#print(names)
color = class_colors(names)

#folder_path = "testeAcuracia/train/"
folder_path = "imageTest"
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(image_files)

model = YOLO("dnn_model/yolov8m.pt")

true_positives = 0
false_positives = 0
false_negatives = 0

tp = fp = fn = 0


imageAcc = 0

for image_file in image_files:

    imageAcc += 1

    frame = cv2.imread(image_file)
    height, width, _ = frame.shape  # Obter altura e largura da imagem

    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu())




    num_score = 0


    for score, class_id, bbox in zip(scores, classes, bboxes):
        className = names[class_id]
        if className in interesse:
            num_score += 1

    # Read corresponding txt file
    txt_file = os.path.splitext(image_file)[0] + ".txt"
    print(txt_file)
    with open(txt_file, "r") as txt:
        lines = txt.readlines()

    num_lines = len(lines)

    for line in lines:

        matched = False

        parts = line.strip().split()
        box = list(map(float, parts[1:5]))

        for score, class_id, bbox in zip(scores, classes, bboxes):
            if score >= 0.5:

                className = names[class_id]

                (x, y, x2, y2) = bbox

                text = className + " " + str(round(score, 2))
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color[className], 2)
                cv2.rectangle(frame, (x, y), (x2, y2), color[className], 2)

                if class_id == 0:
                    class_id = 1
                if class_id == 2:
                    class_id = 0
                if class_id == 7:
                    class_id = 2

                w = x2 - x
                h = y2 - y
                # formato yolov4 é centerx, centery, w, h
                x += int(w / 2)
                y += int(h / 2)

                bbox_norm = [x / width, y / height, w / width, h / height]


                iou = calculate_iou(bbox_norm, box)
                if iou >= 0.75:
                    matched = True
                    if parts[0] == str(class_id):
                        true_positives += 1
                        tp = 1
                    else:
                        false_positives += 1
                        fp = 1
                    break



        if not matched:
            false_negatives += 1
            fn = 1





        # - - - - - -  logs

        file_object.write(str(bbox_norm) + "  " +
                        str(parts[0]) + "  " +
                        str(box) + "  " +
                        str(class_id) + "  " +
                        str(round(iou, 2)) + "  " +
                        str("TP" if tp else "FP" if fp else "FN" if fn else None) + "  " +
                        str(' ')
                        + "\n")

        tp = fp = fn = 0


    print("Score Linhas")
    print(num_score, num_lines)

    



print()
print(true_positives, false_positives, false_negatives)

precision, recall, f1_score = calculate_metrics(true_positives, false_positives, false_negatives)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print("Quantidade imagens: " + str(imageAcc))







file_object.close()

cv2.destroyAllWindows()
