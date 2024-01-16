import cv2
import random
import time


def run(source):
    t1 = time.time()

    # Opencv DNN

    def nivelAlerta(res0, res1, w, h):
        bboxArea = w * h
        # max = math.exp(10)          # 22026.46
        potencia = 4.0
        max = 100 ** potencia
        res = res0 * res1
        coef = max / res
        areaBBN = bboxArea * coef
        return areaBBN ** (1.0 / potencia)

    net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1 / 255)

    # gpu
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    classes = []
    with open("dnn_model/classes.txt", "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)

    def class_colors(names):
        return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}

    color = class_colors(classes)

    cap = cv2.VideoCapture(source)

    width = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`
    resolution = (width, height)

    cv2.namedWindow("Frame")

    # logs

    nome = source[10:]

    txt = f'logs/{nome}.txt'
    with open(txt, 'w') as f:
        f.write('frame      fps       prob          classe\n')
    # f.write('frame      fps      x        y        w       h       prob         classe\n')

    file_object = open(txt, 'a')

    prev_frame_time = 0

    idFrame = 0

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    nome = nome[:-4]

    result = cv2.VideoWriter(f'out/{nome}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 32, size)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)

        prev_frame_time = new_frame_time

        cv2.putText(frame, str(round(fps, 2)), (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 100), 3, cv2.LINE_AA)

        # Object Detection

        (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)

        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox

            className = classes[class_id]
            text = className + " " + str(round(score, 2))

            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color[className], 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color[className], 2)

            msg = "Alerta Nivel " + str(int(nivelAlerta(resolution[0], resolution[1], w, h)))

            file_object.write(str(idFrame) + "          " +
                              str(round(fps, 2)) + "       " +
                              # str(x) + "      " +
                              # str(y) + "      " +
                              # str(w) + "      " +
                              # str(h) + "      " +
                              # str(centerPoints.numpy()) + "  " +
                              str(round(score, 2)) + "          " +
                              str(className) + "          " +
                              (msg if className == 'person' else "")
                              + "\n")

        cv2.imshow("Frame", frame)
        result.write(frame)

        keyCode = cv2.waitKey(1) & 0xff

        if keyCode == 27 or keyCode == ord('q'):
            break

        idFrame += 1

    t2 = time.time()

    file_object.write("Tempo = " + str(t2 - t1))

    file_object.close()

    print(t2 - t1)

    cap.release()
    result.release()

    cv2.destroyAllWindows()
