import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append("..")

# Name of the directory where the inference graph is stored.
MODEL_NAME = 'model'

# Get the current working directory.
CWD_PATH = os.getcwd()

# Get the path to the frozen inference graph.
PATH_TO_INFERENCE_GRAPH = os.path.join(
    CWD_PATH,
    'model', 
    'frozen_inference_graph.pb')

# Get the path to the label map.
PATH_TO_LABELMAP = os.path.join(
    CWD_PATH, 
    'model', 
    'labelmap.pbtxt')

PATH_TO_ORIGINAL_IMAGES = os.path.join(
    CWD_PATH,
    'original_images')

PATH_TO_CLASSIFIED_IMAGES = os.path.join(
    CWD_PATH,
    'classified_images')

# Define the number of classes the model is trained for.
NUM_CLASSES = 12

def classify():

    # Each category is extracted from the label map. 
    # The number of categories matches NUM_CLASSES.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELMAP)
    
    categories = label_map_util.convert_label_map_to_categories(
		label_map, 
		max_num_classes=NUM_CLASSES, 
		use_display_name=True)
    
    category_index = label_map_util.create_category_index(categories)

    # The inference graph must be loaded outside of the directory loop. This
    # limits the application to only opening one graph session at a time.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_INFERENCE_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Load each tensor from the detection graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # For each file(image) in the given directory, the inference graph is
    # applied. The resulting image is saved in the same directory.
    #
    # TO-DO
    # UPDATE SAVE LOCATION WHEN THE APPLICATION STRUCTURE IS FINALIZED
    for filename in os.listdir('original_images'):

        image_path = os.path.join(PATH_TO_ORIGINAL_IMAGES,filename)

        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, 
            detection_scores, 
            detection_classes, 
            num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the bounding boxes on the given image
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6,
            min_score_thresh=0.55)
        
        output_name = 'classified_' + filename
        output_path = os.path.join(PATH_TO_CLASSIFIED_IMAGES, output_name)

        cv2.imwrite(output_path, image)

classify()