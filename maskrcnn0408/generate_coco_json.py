from __future__ import division
from __future__ import print_function

import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi
from viz import draw_annotation_bbox_mask
from common import segmentation_to_mask

class DAVIS2016():
    """
        DAVIS 2016 class to convert annotations to COCO Json format
    """
    def __init__(self, datapath, imageres="480p"):
        self.info = {"year" : 2016,
                     "version" : "1.0",
                     "description" : "A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation (DAVIS)",
                     "contributor" : "F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, A. Sorkine-Hornung ",
                     "url" : "http://davischallenge.org/",
                     "date_created" : "2016"
                    }
        self.licenses = [{"id": 1,
                          "name": "Attribution-NonCommercial",
                          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                         }]
        self.type = "instances"
        self.datapath = datapath
        self.seqs = yaml.load(open(os.path.join(self.datapath, "Annotations", "db_info.yml"),
                                   "r")
                             )["sequences"]

        self.categories = [{"id": seqId+1, "name": seq["name"], "supercategory": seq["name"]}
                              for seqId, seq in enumerate(self.seqs)]
        self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}

        # for i in range(10):
          # print('folder--', i)
          # for s in [str(i)+"_train", str(i)+"_val", str(i)+'_test']:
        # txt_path = os.path.join(self.datapath, "ImageSets", imageres, '3_folder')
        txt_path = os.path.join(self.datapath, "ImageSets", imageres, '10_fold_txt_elastic')
        # txt_path = os.path.join(self.datapath, "ImageSets", imageres, '10_folder')
        # txt_path = os.path.join(self.datapath, "ImageSets", imageres, 'test')
        for f_name in os.listdir(txt_path):
            imlist = np.loadtxt(os.path.join(txt_path, f_name), dtype=str)
            print(f_name, len(imlist))
            images, annotations = self.__get_image_annotation_pairs__(imlist)
            json_data = {"info" : self.info,
                         "images" : images,
                         "licenses" : self.licenses,
                         "type" : self.type,
                         "annotations" : annotations,
                         "categories" : self.categories}

            with open(os.path.join(self.datapath, "Annotations", "instances_" +
                                   f_name[:-4]+".json"), "w") as jsonfile:
                json.dump(json_data, jsonfile, sort_keys=True, indent=4)

    def __get_image_annotation_pairs__(self, image_set):
        images = []
        annotations = []
        for imId, paths in enumerate(image_set):
            # print(paths)
            impath, annotpath = paths[0], paths[1]
            # print (impath)
            # name = impath.split("/")[3]
            name = 'tumor'
            img = np.array(Image.open(os.path.join(self.datapath + impath)).convert('RGB'))
            mask = np.array(Image.open(os.path.join(self.datapath + annotpath)).convert('L'))
            # mask = cv2.imread(self.datapath + annotpath, 0)
            if np.all(mask == 0):
              print('!!!wrong!!!mask == 0')
              continue
            # print(mask.shape)
            # print(mask)
            segmentation, bbox, area, mask_viz = self.__get_annotation__(mask, img)

            new_box = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]]
            # print(impath, annotpath)
            # print(img.shape, mask.shape)
            viz_im = draw_annotation_bbox_mask(img, new_box, [0], mask_viz)
            # cv2.imwrite("{}.png".format(imId), viz_im)
            # cv2.imwrite('{}_mask.bmp'.format(imId), mask)

            images.append({"date_captured" : "2016",
                           "file_name" : impath[:], # remove "/"
                           "id" : imId+1,
                           "license" : 1,
                           "url" : "",
                           "height" : mask.shape[0],
                           "width" : mask.shape[1]})
            annotations.append({"segmentation" : segmentation,
                                "area" : np.float(area),
                                "iscrowd" : 0,
                                "image_id" : imId+1,
                                "bbox" : bbox,
                                "category_id" : self.cat2id[name],
                                "id": imId+1})
        return images, annotations

    def __get_annotation__(self, mask, image=None):

        # print('mask.shape', mask.shape)
        # print(mask.dtype)
        # print(type(mask))
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
                # print(contour.flatten().tolist())

        RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        RLE = cocomask.merge(RLEs)
        mask_viz = cocomask.decode(RLE)
        # RLE = cocomask.encode(np.asfortranarray(mask))
        area = cocomask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(mask)

        # if image is not None:
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     cv2.drawContours(image, contours, -1, (0,255,0), 1)
        #     cv2.rectangle(image,(x,y),(x+w,y+h), (255,0,0), 2)
        #     cv2.imshow("", image)
        #     cv2.waitKey(1)

        return segmentation, [x, y, w, h], area, [mask_viz]

if __name__ == "__main__":
    # datapath = '/media/sdc/xzhao/code/small_dataset/'
    datapath = '/media/sdc/xzhao/code/wavelet_small_dataset/'
    DAVIS2016(datapath)

    # test
    from PIL import Image
    from pycocotools.coco import COCO 
    coco = COCO(datapath+'Annotations/instances_cv1_train_all.json')
    # coco = COCO(datapath+'Annotations/instances_0_train.json')
    im = Image.fromarray(coco.annToMask(coco.loadAnns([1])[0])*255)
    # im.save("out.jpg")
    categories = coco.getCatIds()
    print(categories)
    cats = coco.loadCats(categories)
    print(cats)
    nms = [cat['name'] for cat in cats]
    print(nms)
