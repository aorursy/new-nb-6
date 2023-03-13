import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

print(os.listdir("../input/test/")[:1])

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow_hub as hub

import tensorflow as tf
#https://github.com/WillKoehrsen/Data-Analysis/blob/master/widgets/Widgets-Overview.ipynb

#https://towardsdatascience.com/interactive-controls-for-jupyter-notebooks-f5c94829aee6

from IPython.display import Image, display, HTML

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

fdir = '../input/test/'



@interact

def show_images(file=os.listdir(fdir)[-10:]):

    display(Image(fdir+file))
df_sample = pd.read_csv("../input/sample_submission.csv")

df_sample.info()
df_sample.head()
df_sample.iloc[0,1]
#https://github.com/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb
def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):

  """Overlay labeled boxes on an image with formatted scores and label names."""

  colors = list(ImageColor.colormap.values())



#   try:

#     font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",

#                               25)

#   except IOError:

#     print("Font not found, using default font.")

  font = ImageFont.load_default()



  for i in range(min(boxes.shape[0], max_boxes)):

    if scores[i] >= min_score:

      ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())

      display_str = "{}: {}%".format(class_names[i].decode("ascii"),

                                     int(100 * scores[i]))

      color = colors[hash(class_names[i]) % len(colors)]

      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

      draw_bounding_box_on_image(

          image_pil,

          ymin,

          xmin,

          ymax,

          xmax,

          color,

          font,

          display_str_list=[display_str])

      np.copyto(image, np.array(image_pil))

  return image



def draw_bounding_box_on_image(image,

                               ymin,

                               xmin,

                               ymax,

                               xmax,

                               color,

                               font,

                               thickness=4,

                               display_str_list=()):

  """Adds a bounding box to an image."""

  draw = ImageDraw.Draw(image)

  im_width, im_height = image.size

  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,

                                ymin * im_height, ymax * im_height)

  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),

             (left, top)],

            width=thickness,

            fill=color)



  # If the total height of the display strings added to the top of the bounding

  # box exceeds the top of the image, stack the strings below the bounding box

  # instead of above.

  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

  # Each display_str has a top and bottom margin of 0.05x.

  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)



  if top > total_display_str_height:

    text_bottom = top

  else:

    text_bottom = bottom + total_display_str_height

  # Reverse list and print from bottom to top.

  for display_str in display_str_list[::-1]:

    text_width, text_height = font.getsize(display_str)

    margin = np.ceil(0.05 * text_height)

    draw.rectangle([(left, text_bottom - text_height - 2 * margin),

                    (left + text_width, text_bottom)],

                   fill=color)

    draw.text((left + margin, text_bottom - text_height - margin),

              display_str,

              fill="black",

              font=font)

    text_bottom -= text_height - 2 * margin
from PIL import ImageColor

from PIL import ImageFont

from PIL import Image

from PIL import ImageDraw
#https://www.kaggle.com/cttsai/inference-script-with-pretrained-models-from-tfhub
with tf.device('/device:GPU:0'):

    with tf.Graph().as_default():

      detector = hub.Module("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

      image_string_placeholder = tf.placeholder(tf.string)

      decoded_image = tf.image.decode_jpeg(image_string_placeholder)

      # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch

      # of size 1 and type tf.float32.

      decoded_image_float = tf.image.convert_image_dtype(

          image=decoded_image, dtype=tf.float32)

      module_input = tf.expand_dims(decoded_image_float, 0)

      result = detector(module_input, as_dict=True)

      init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]



      session = tf.Session()

      session.run(init_ops)



      # Load the downloaded and resized image and feed into the graph.

      with tf.gfile.Open("../input/test/da4245f6018ea0a2.jpg", "rb") as binfile:

        image_string = binfile.read()



      result_out, image_out = session.run(

          [result, decoded_image],

          feed_dict={image_string_placeholder: image_string})

      print("Found %d objects." % len(result_out["detection_scores"]))

      #session.close()



image_with_boxes = draw_boxes(

    np.array(image_out), result_out["detection_boxes"],

    result_out["detection_class_entities"], result_out["detection_scores"])



#display_image(image_with_boxes)

fig = plt.figure(figsize=(20, 15))

plt.grid(False)

_=plt.imshow(image_with_boxes)



#https://www.kaggle.com/vikramtiwari/baseline-predictions-using-inception-resnet-v2
df_train = pd.DataFrame(os.listdir("../input/test/"), columns=['file'])

df_train.info()
df_train.iloc[:10, 0].values
df_train.iloc[:, 0] = "../input/test/" + df_train.iloc[:, 0]
#https://www.kaggle.com/vikramtiwari/baseline-predictions-using-inception-resnet-v2

#image_paths = get_images(10)

images = df_train.iloc[:1, 0].values

predictions = []

ImageID = []

PredictionString = []

with tf.device('/device:GPU:0'):

    for image in images:

        print(image)

        with tf.gfile.Open(image, "rb") as binfile:

            image_string = binfile.read()

        

        result_out, image_out = session.run(

        [result, decoded_image],

        feed_dict={image_string_placeholder: image_string})

        print("Found %d objects." % len(result_out["detection_scores"]))

        print(result_out["detection_scores"].shape)

        print(result_out["detection_scores"][0])

        print("{:0.1f}".format(result_out["detection_scores"][0]))

        print(result_out["detection_class_entities"][0])

        print(result_out.keys())

        print(result_out["detection_class_labels"][0])

        print(result_out["detection_class_names"][0])

        print(result_out["detection_boxes"][0])

        print(["{:0.1f}".format(dbc) for dbc in result_out["detection_boxes"][0]])

        print(' '.join([(result_out["detection_class_names"][0]).decode("utf-8") , "{:0.1f}".format(result_out["detection_scores"][0])]))

        ls = [(result_out["detection_class_names"][0]).decode("utf-8") , "{:0.1f}".format(result_out["detection_scores"][0])]

        ls.extend(["{:0.1f}".format(dbc) for dbc in result_out["detection_boxes"][0]])

        print(' '.join(ls))

        

        for i in range(1,5):

            print(result_out["detection_scores"][i])

            print(result_out["detection_class_entities"][i])



        print(np.histogram(result_out["detection_scores"]))

        
sns.set(rc={'figure.figsize':(13,13)})

_ = sns.swarmplot(result_out["detection_scores"],result_out["detection_class_entities"], orient='h')
from collections import Counter
# images = df_train.iloc[:, 0].values

# predictions = []

# ImageID = []

# entities = []

# with tf.device('/device:GPU:0'):

#     for i, image in enumerate(images):

#         #print(image)

#         with tf.gfile.Open(image, "rb") as binfile:

#             image_string = binfile.read()

        

#         result_out, image_out = session.run(

#         [result, decoded_image],

#         feed_dict={image_string_placeholder: image_string})

#         #print(result_out["detection_class_entities"])

        

#         if i == 0:

#             ctr = Counter(result_out["detection_class_entities"])

#         else:

#             ctr.update(result_out["detection_class_entities"])

        #entities.extend(result_out["detection_class_entities"])

        

#print(len(entities))

images = df_train.iloc[:7000, 0].values

predictions = []

ImageID = []

entities = []

with tf.device('/device:GPU:0'):

    for i, image in enumerate(images):

        #print(image)

        with tf.gfile.Open(image, "rb") as binfile:

            image_string = binfile.read()

        

        result_out, image_out = session.run(

        [result, decoded_image],

        feed_dict={image_string_placeholder: image_string})

        #print(result_out["detection_class_entities"])

       # print(result_out["detection_scores"]>0.5)

       # print(result_out["detection_class_entities"][result_out["detection_scores"]>0.5])

        

        #print(type(result_out["detection_scores"]))

        if i == 0:

            ctr = Counter(result_out["detection_class_entities"][result_out["detection_scores"]>0.5])

        else:

            ctr.update(result_out["detection_class_entities"][result_out["detection_scores"]>0.5])

        #entities.extend(result_out["detection_class_entities"])
# from collections import Counter

# ctr = Counter(entities).most_common(10)

df_counter = pd.DataFrame(ctr.most_common(10))

#print(Counter(entities).most_common(10))
df_counter.columns=["Entity", "Count"]
#http://www.image-net.org/about-stats
_ = df_counter.plot(x="Entity", y="Count", kind="barh")
import requests



jsons = requests.get('https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-label500-hierarchy.json').json()
# for k in jsons.keys():

#     print(k)

#     print(jsons.get(k))

    

# for v in jsons.get("Subcategory"):

#     print(v)

    
jsons.get("LabelName")
#https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url

import io

s=requests.get("https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-detection-human-imagelabels.csv").content

train_imagelabels=pd.read_csv(io.StringIO(s.decode('utf-8')))
train_imagelabels.info()
train_imagelabels.sort_values(by="ImageID").head()
train_imagelabels.Confidence.value_counts()
train_imagelabels.ImageID.nunique()
train_imagelabels[train_imagelabels.Confidence==1].ImageID.nunique()
s=requests.get("https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-detection-human-imagelabels.csv").content

validation_imagelabels=pd.read_csv(io.StringIO(s.decode('utf-8')))
validation_imagelabels.info()
validation_imagelabels.sort_values(by="ImageID").head()
validation_imagelabels.Confidence.value_counts()
validation_imagelabels.ImageID.nunique()
validation_imagelabels[validation_imagelabels.Confidence==1].ImageID.nunique()
s=requests.get("https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-500.csv").content

classes_desc=pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)
classes_desc.columns=['LabelName', 'Desc']
classes_desc.info()
classes_desc.head()
classes_desc[classes_desc.LabelName.str.contains('/m/0cmf2')]
classes_desc[classes_desc.LabelName.str.contains('/m/019dx1')]