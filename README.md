# coco-ssd-tf-object-detection
object detection api based on tensorflow object API 2.0 and flask

the model can be find in tensorflow object model zoo 2.0 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
we use the model SSD ResNet50 V1 FPN 640x640 (RetinaNet50)


the project has two api:

http://127.0.0.1/upload
which can let you upload a test picture

the other one is:
http://127.0.0.1/detect?url=tmp/tmp.jpg
it can return the bounding box position and category prediction, like the below string:
{
"object": [
{
  "left_up_x": 122.89736938476562, 
  "left_up_y": 124.1726131439209, 
  "right_down_x": 569.6692199707031, 
  "right_down_y": 452.6667251586914, 
  "class": "2"
}, 
{ 
  "left_up_x": 130.05878448486328, 
  "left_up_y": 207.01769828796387, 
  "right_down_x": 319.4461441040039, 
  "right_down_y": 544.6496200561523, 
  "class": "18"}
]
}

