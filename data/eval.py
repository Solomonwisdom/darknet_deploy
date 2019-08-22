import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
from voc_eval import voc_eval
import os

# ./darknet detector valid /home/miracle/Documents/detect/dataSet/cfg/js_detect.data /home/miracle/Documents/detect/dataSet/cfg/js_detct_valid.cfg /home/miracle/Documents/detect/dataSet/backup/js_detect_10000.weights -out " " -thresh .5


model = ['person', 'car', 'truck', 'tank', 'off-road_vehicle', 'armored_car', 'airplane']
# model = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

for ele in model:
    if os.path.exists(BASE_DIR + '/annots.pkl'):
        os.remove(BASE_DIR + '/annots.pkl')
    prec, tp, total = voc_eval(ROOT_DIR + '/results/ ' + ele + '.txt',
                           ROOT_DIR + '/data/Annotations/{}.xml',
                           ROOT_DIR + '/data/val.txt',
                           ele,
                           ROOT_DIR)

    print(ele)
    # print('rec',rec[-1])
    print('prec %f %d / %d' % (prec[-1], tp, total))
    # print('ap',ap)
    # break
