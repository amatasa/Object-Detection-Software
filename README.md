# Object Detection Software
A bundled python application used to classify objects for a directory of images. This software MUST be run on Ubuntu 18.04.X. 
The object detection utilizes TensorFlow 1.12 and does not require users to install any packages.

# How to use
Users can run the classify_directory executable located within the dist/classify_directory folder. The application looks in
the folder original_images and labels each image. The labeled images are then saved with there corresponding bounding boxes
in the folder classified_images. 

Within the folder dist/classify_directory/model there are two files: A frozen inference graph and a label map. If you would like to use your own trained model, you should replace these files with your corresponding inference graph and label map. 

# Future work
This application is in beta stages so there are many updates that need to be made. A list of future updates are:

1. Utilize a GUI to allow users to select their own directory
2. Display labeled images within GUI
