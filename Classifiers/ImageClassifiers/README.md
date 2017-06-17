Image Classifier
================

This uses the Google's Deep Learning Model Inception for Image Classification.
 
Training
--------
* Getting Images - 
'''curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz'''

* Getting retarin.py script (Already in repo) - 
'''curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py'''

* Training on Images dataset - 
'''python retrain.py --bottleneck_dir=bottlenecks --model_dir=inception --summaries_dir=training_summaries/basic --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=flower_photos'''

* Script to test on own image - 
'''curl -L https://goo.gl/3lTKZs > label_image.py'''

* Testing on own image - 
'''python label_image.py path/to/own/image.jpg '''
