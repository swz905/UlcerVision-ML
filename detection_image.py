import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import pandas as pd

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder', action='store_true')
parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)', action='store_false')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu
save_results = args.save_results  # Defaults to False
show_results = args.noshow_results  # Defaults to True

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del labels[0]

# Load the Tensorflow Lite model.
interpreter = None
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
    interpreter = Interpreter(
        model_path=PATH_TO_CKPT,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')] if use_TPU else [])
else:
    from tensorflow.lite.python.interpreter import Interpreter
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Prompt the user for the image or folder path
input_path = input("Enter the path to the image or folder you want to perform detection on: ")

# Check if the provided path exists
if not os.path.exists(input_path):
    print(f"Error: The specified path '{input_path}' does not exist.")
    sys.exit()

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Image Name', 'Detection', 'Accuracy'])

# Function to perform object detection on a single image and update the results DataFrame
def perform_detection(image_path):
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        # Initialize variables to store the detection result and accuracy
        detection_result = "Non-Bleeding"
        accuracy = 0

        # Loop over all detections and find the highest confidence detection
        max_confidence = 0
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                if scores[i] > max_confidence:
                    max_confidence = scores[i]
                    detection_result = labels[int(classes[i])]
                    accuracy = scores[i]

        # Update the results DataFrame
        results_df.loc[len(results_df)] = [os.path.basename(image_path), detection_result, accuracy]

        # Loop over all detections and draw detection box if confidence is above the minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Show the result image
        cv2.imshow('Object detector', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the labeled image to results folder if desired
        if save_results:
            base_fn, ext = os.path.splitext(image_path)
            image_savepath = f"{base_fn}_result{ext}"
            cv2.imwrite(image_savepath, image)

    except Exception as e:
        print(f"Error: An exception occurred while processing the image: {str(e)}")

# Check if the input path is a directory
if os.path.isdir(input_path):
    # If it's a directory, iterate through all image files in the directory
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_path, filename)
            print(f"Processing image: {image_path}")
            perform_detection(image_path)
else:
    # If it's a single image file, perform detection on that image
    image_path = input_path
    print(f"Processing image: {image_path}")
    perform_detection(image_path)

# Save the results DataFrame to an Excel file in the same folder
results_filename = os.path.join(MODEL_NAME, 'detection_results.xlsx')
results_df.to_excel(results_filename, index=False)

print(f"Detection results saved to {results_filename}")
