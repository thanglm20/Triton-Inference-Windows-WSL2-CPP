
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import operator

class ColorClassifier:
    def __init__(self, hsv_table_path="ColorDef.xml"):
        self.load_hsv_color_map(hsv_table_path)
        self.upper_mask_ids = [5,6,7]
        self.lower_mask_ids = [9, 10, 12]

    def load_hsv_color_map(self, hsv_table_path):
        self.color_map =  self.get_color_map_from_file(hsv_table_path)
        
    def get_color_map_from_file(self, hsv_table_path):
        color_map = []
        tree = ET.parse(hsv_table_path)
        root = tree.getroot()
        for color in root.findall("./Lists/Color"):
            color_name  = str(color.findtext("Name"))
            # read rgb
            r = color.findtext("./R")
            g = color.findtext("./G")
            b = color.findtext("./B")
            bgr = [int(b), int(g), int(r)]
            for lower, upper in zip(color.findall("./LowerHSV"), color.findall("./UpperHSV")):
                # read lower
                lh = lower.findtext("H")
                ls = lower.findtext("S")
                lv = lower.findtext("V")
                lower_hsv = [int(lh), int(ls), int(lv)]
                # read upper
                uh = upper.findtext("H")
                us = upper.findtext("S")
                uv = upper.findtext("V")
                upper_hsv = [int(uh), int(us), int(uv)]
                color_map.append([color_name,  lower_hsv, upper_hsv,  bgr])
        return color_map
    
    def find_dominant_hsv_color(self, hsv_color_map, hsv_img, mask):
        w = hsv_img.shape[1]
        h = hsv_img.shape[0]
        mask = np.where(mask > 0, 255, 0)
        # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        # hsv_img = hsv_img & mask
        locations = cv2.findNonZero(mask)
        print("location: ", locations.shape)
        filtered_hsv = hsv_img[(mask>0).any(axis=2)]
        new_h = filtered_hsv.shape[0] // 2
        if filtered_hsv.shape[0] == 0:
            return {"unknown": 0.0}
        elif filtered_hsv.shape[0] % 2 == 0:
            filtered_hsv1 = filtered_hsv.reshape(new_h, 2, 3)
        else:
            filtered_hsv1 = filtered_hsv[:-1].reshape(new_h, 2, 3)
        max_nonzero = 0
        color_name = ''
        output_table  = {}
        points_in_region = filtered_hsv.shape[0]
        for color, lower, upper, _ in hsv_color_map:
            mask_hsv = cv2.inRange(filtered_hsv1, np.array(lower), np.array(upper))
            nonzero = cv2.countNonZero(mask_hsv)
            # check if color is more than a half of region
            # if nonzero >= new_h:
            #     return color
            # or find max of color
            if(nonzero > max_nonzero):
                max_nonzero = nonzero
                color_name = color
            # calc percentage
            percent =  round(nonzero / points_in_region * 100, 2)
            if percent < 10.0:
                continue
            output_table[color] = percent
        output_table = dict(sorted(output_table.items(), key=lambda x:x[1], reverse=True))
        # print(output_table)
        return output_table

    def get_upper_color(self, hsv_img, bgr_img, pred):
        # upper_mask = np.where( pred in self.upper_mask_ids, 255, 0)
        upper_mask = np.isin(pred, self.upper_mask_ids)
        upper_mask = upper_mask.reshape(upper_mask.shape[:2]).astype(np.uint8)
        hsv_outputs = self.find_dominant_hsv_color(self.color_map, hsv_img, upper_mask)
        return hsv_outputs, upper_mask

    def get_lower_color(self, hsv_img, bgr_img, pred):
        # lower_mask = np.where(pred in self.lower_mask_ids, 255, 0)
        lower_mask = np.isin(pred, self.lower_mask_ids)
        lower_mask = lower_mask.reshape(lower_mask.shape[:2]).astype(np.uint8)
        hsv_outputs = self.find_dominant_hsv_color(self.color_map, hsv_img, lower_mask)
        return hsv_outputs, lower_mask