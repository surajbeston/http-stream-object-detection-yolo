import numpy as np
import cv2
from django.conf import settings
import string
import random
import os 


classes = None
with open(os.path.join(settings.BASE_DIR,  "coco.names"), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(os.path.join(settings.BASE_DIR,  "yolov3.weights"), os.path.join(settings.BASE_DIR, "yolov3.cfg")) 

def detect_object_in_image(file):

    image_array = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)


    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:] 
            class_id = np.argmax(scores) 
            confidence = scores[class_id] 
            if confidence > 0.5:
                center_x = int(detection[0] * Width) 
                center_y = int(detection[1] * Height)  
                w = int(detection[2] * Width) 
                h = int(detection[3] * Height) 
                x = center_x - w / 2 
                y = center_y - h / 2 
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


    labels = []

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        labels.append(classes[class_ids[i]])
        
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)

    # display output image    
    # cv2.imshow("object detection", image)

    # # wait until any key is pressed
    # cv2.waitKey()

    _, encoded_image = cv2.imencode('.jpg', image)
    print(encoded_image)
    print(_)

    image_bytes = encoded_image.tobytes()

    return image_bytes, labels

        
    # save output image to disk
    # filename = ''.join(random.choices(string.ascii_letters, k = 10)) + ".jpg"
    # cv2.imwrite(os.path.join(settings.BASE_DIR, filename) , image)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
