from pathlib import Path
import json
import math
import shutil
import random

import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import cv2
import imgviz


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get('Orientation', None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


class SegmentGenerator:
    def labelme_to_segment(self,
                           class_names_file_path,
                           image_dir_path,
                           json_dir_path,
                           output_dir_path,
                           train_data_percent):
        class_names_file_path = Path(class_names_file_path)
        image_dir_path  = Path(image_dir_path)
        json_dir_path   = Path(json_dir_path)
        output_dir_path = Path(output_dir_path)

        if output_dir_path.exists():
            shutil.rmtree(output_dir_path.as_posix())

        output_dir_path.mkdir(parents=True)

        class_names = []

        with open(class_names_file_path.as_posix(), 'r') as file_stream:
            for line in file_stream:
                line = line.strip()

                if line:
                    class_names.append(line)

        print('class_names:', class_names)

        class_name_dict = {class_name: class_index for class_index, class_name in enumerate(class_names)}

        print('class_name_dict:', class_name_dict)

        self.convert(image_dir_path,
                     json_dir_path,
                     output_dir_path,
                     class_name_dict)

        self.slplit_train_val(output_dir_path, train_data_percent)

    def convert(self,
                image_dir_path,
                json_dir_path,
                output_dir_path,
                class_name_dict):
        output_image_dir_path = output_dir_path / 'images'
        output_image_dir_path.mkdir(parents=True)
        output_mask_dir_path = output_dir_path / 'masks'
        output_mask_dir_path.mkdir(parents=True)

        json_file_paths = list(json_dir_path.glob('*.json'))

        colormap = imgviz.label_colormap()
        file_path_pairs = []
        output_file_index = 0

        for json_file_path in json_file_paths:
            # if json_file_path.stem != '045_sozai_l':
            #     continue

            if not json_file_path.suffix in ['.json']:
                warn_format = r'[Warn] Ignore json_file_path {} because it is not json file!'
                print(warn_format.format(json_file_path.as_posix()))
                continue

            with open(json_file_path.as_posix(), 'r') as file_stream:
                labelme_data = json.load(file_stream)

            image_file_path = image_dir_path / labelme_data['imagePath']

            if not image_file_path.exists():
                warn_format = r'[Warn] Ignore json_file_path {} because image_file_path {} not exist!'
                print(warn_format.format(json_file_path.as_posix(), image_file_path.as_posix()))
                continue

            if image_file_path.stem != json_file_path.stem:
                warn_format = r'[Warn] Name different between json_file_path {} and image_file_path {}!'
                print(warn_format.format(json_file_path.as_posix(), image_file_path.as_posix()))

            labelme_image_height = labelme_data['imageHeight']
            labelme_image_width  = labelme_data['imageWidth']

            origin_image = PIL.Image.open(image_file_path.as_posix())
            image_pil = apply_exif_orientation(origin_image)
            image_arr = np.array(image_pil)
            image_height, image_width = image_arr.shape[:2]

            if not (labelme_image_height == image_height and labelme_image_width == image_width):
                warn_format = r'Image height, size different between labelme {} and image {}!'
                print(warn_format.format((labelme_image_height, labelme_image_width),
                                         (image_height, image_width)))
                continue

            output_image_file_name = r'{:05}.jpg'.format(output_file_index)
            # output_image_file_name = image_file_path.with_suffix('.jpg').name
            output_image_file_path = output_image_dir_path / output_image_file_name
            image_pil.save(output_image_file_path.as_posix())
            image_pil.close()

            mask_arr = self.shapes_to_mask(image_arr.shape,
                                           labelme_data['shapes'],
                                           class_name_dict)

            # mask_pil = PIL.Image.fromarray(mask_arr.astype(np.uint8), mode='P')
            output_mask_file_path = output_mask_dir_path / output_image_file_path.with_suffix('.png').name

            # class_names = [shape['label'] for shape in labelme_data['shapes']]
            # print(set(class_names))
            # print(np.unique(mask_arr))

            # mask_pil.putpalette(colormap.flatten())
            # mask_pil.save(output_mask_file_path.as_posix())
            # mask_pil.close()
            cv2.imwrite(output_mask_file_path.as_posix(), mask_arr.astype(np.uint8))
            file_path_pair = {'mask_file_path': output_mask_file_path.as_posix(),
                              'json_file_path': json_file_path.as_posix()}

            file_path_pairs.append(file_path_pair)

            output_file_index += 1

        file_pair_path = output_dir_path / 'file_pairs.json'

        with open(file_pair_path.as_posix(), 'w') as file_stream:
            json.dump(file_path_pairs, file_stream, indent=4)


    def shapes_to_mask(self,
                        image_shape,
                        shapes,
                        class_name_dict):
        class_mask = np.zeros(image_shape[:2], dtype=np.int32)

        shapes.sort(key=lambda shape:class_name_dict[shape['label']])

        for shape in shapes:
            class_name = shape['label']
            points = shape['points']
            shape_type = shape.get('shape_type', 'polygon')

            mask = self.shape_to_mask(image_shape[:2], points, shape_type)
            class_id = class_name_dict[class_name]
            class_mask[mask] = class_id

            # print('Draw:', class_name, class_id, points)

        return class_mask

    def shape_to_mask(self,
                      image_shape,
                      points,
                      shape_type=None,
                      line_width=10,
                      point_size=5):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)
        xy = [tuple(point) for point in points]

        if shape_type == 'circle':
            assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        elif shape_type == 'rectangle':
            assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
            draw.rectangle(xy, outline=1, fill=1)
        elif shape_type == 'line':
            assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == 'linestrip':
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == 'point':
            assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
            cx, cy = xy[0]
            r = point_size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
        else:
            assert len(xy) > 2, 'Polygon must have points more than 2'
            draw.polygon(xy=xy, outline=1, fill=1)

        mask = np.array(mask, dtype=bool)
        return mask

    def slplit_train_val(self, output_dir_path, train_data_percent):
        image_dir_path = output_dir_path / 'images'
        mask_dir_path = output_dir_path / 'masks'

        image_file_paths = list(image_dir_path.glob('*.jpg'))

        random.seed(7)
        random.shuffle(image_file_paths)

        train_samples_count = round(len(image_file_paths) * train_data_percent)
        output_image_dir_path = output_dir_path / 'img_dir'
        output_ann_dir_path = output_dir_path / 'ann_dir'

        for image_file_index, image_file_path in enumerate(image_file_paths):
            data_type = 'train' if image_file_index < train_samples_count else 'val'

            output_image_file_path = output_image_dir_path / data_type / image_file_path.name
            output_image_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_file_path.as_posix(), output_image_file_path.as_posix())

            ann_file_path = mask_dir_path / image_file_path.with_suffix('.png').name
            output_ann_file_path = output_ann_dir_path / data_type / ann_file_path.name
            output_ann_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ann_file_path.as_posix(), output_ann_file_path.as_posix())



def main():
    labelme_dir_path = 'data/Watermelon87_Semantic_Seg_Labelme'

    class_names_file_path = 'data/watermelon_class_names.txt'

    labelme_dir_path = Path(labelme_dir_path)
    image_dir_path = labelme_dir_path / 'images'
    json_dir_path = labelme_dir_path / 'labelme_jsons'

    output_dir_path = 'data/watermelon87_database'
    train_data_percent = 0.8

    generater = SegmentGenerator()

    generater.labelme_to_segment(class_names_file_path,
                                 image_dir_path,
                                 json_dir_path,
                                 output_dir_path,
                                 train_data_percent)


if __name__ == '__main__':
    main()
