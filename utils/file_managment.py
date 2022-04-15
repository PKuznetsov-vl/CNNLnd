import os
import xml.etree.ElementTree as ET

import dlib


class FileOp:

    def __init__(self, directory):
        self.directory = directory

    def get_files_from_dir(self):
        files = os.listdir(self.directory)
        images = list(filter(lambda x: x.endswith('.jpg'), files))
        pts = list(filter(lambda x: x.endswith('.pts'), files))
        located_pts = []
        for file in pts:
            located_pts.append(os.path.join(self.directory, file))

        return images, pts, located_pts


class PathOp(FileOp):

    def __init__(self, files_path, directory):
        super().__init__(directory)
        self._files_path = files_path

    def get_pts(self):
        points_lst = []

        i = 0
        for f in self._files_path:
            corr_path = f.replace('.jpg', '.pts')
            print("Processing file: {}".format(corr_path))
            with open(corr_path) as file:
                rows = [rows.strip() for rows in file]
                if '68' in rows[1]:
                    """Use the curly braces to find the start and end of the point data"""
                    head = rows.index('{') + 1
                    tail = rows.index('}')

                    """Select the point data split into coordinates"""
                    raw_points = rows[head:tail]

                    coords_set = [point.split() for point in raw_points]

                    """Convert entries from lists of strings to tuples of floats"""
                    point = [tuple([float(point) for point in coords]) for coords in coords_set]
                    points_lst.append(point)
                    i += 1
                else:
                    print('not 68 points')

        return points_lst


class Detector(FileOp):

    def __init__(self, images_path, directory):
        super().__init__(directory)
        self._images_path = images_path

    def det(self):
        detector = dlib.get_frontal_face_detector()
        # win = dlib.image_window()
        value_list = []
        new_img_list = []
        for f in self._images_path:
            print("Processing file: {}".format(f))
            try:
                img = dlib.load_rgb_image(f)
                dets = detector(img, 1)
                if len(dets) == 1:
                    print("Number of faces detected: {}".format(len(dets)))
                    for i, d in enumerate(dets):  # <box top='78' left='74' width='138' height='140'>
                        value = [d.top(), d.left(), d.width(), d.bottom() - d.top()]
                        #if d.left()*d.right()>2800:

                        value_list.append(value)
                        new_img_list.append(f)
                        # else:
                        #     print("Low Quality image")
                else:
                    print("Pass")
            except:
                print("Unable to open image")



        return value_list, new_img_list


class Checker():

    def __init__(self, images_path, ):
        self._images_path = images_path

    def check_pts(self):

        path_lst = []
        for f in self._images_path:
            # full_path = os.path.join(self._images_path, f)
            print("Processing file: {}".format(f))

            write_path = f.replace('.pts', '.jpg')
            with open(f) as file:
                rows = [rows.strip() for rows in file]
                if '68' in rows[1]:
                    path_lst.append(write_path)
                else:
                    print('Have not 68 points')

        return path_lst


class CreateXML():
    def __init__(self, images_list, bb_list, points_list,name):
        self._images_list = images_list
        self._bb_list = bb_list
        self._points_list = points_list
        self._name=name

    def create_xml(self):
        # we make root element
        i = 0
        root = ET.Element("dataset")

        # create sub element
        images = ET.Element("images")

        for img in range(len(self._images_list)):
            image = ET.Element('image', file=f'{self._images_list[img]}')
            # insert list element into sub elements
            box = ET.Element('box', top=str(self._bb_list[i][0]), left=str(self._bb_list[i][1]),
                             width=str(self._bb_list[i][2]), height=str(self._bb_list[i][3]))
            images.append(image)
            image.append(box)
            counter = 0

            for pnt in self._points_list[i]:
                if counter > 9:
                    name = f"part name='{counter}'"
                else:
                    name = f"part name='0{counter}'"

                x = str(pnt[0])
                y = str(pnt[1])
                point = ET.SubElement(box, str(name + f" x='{x[:x.index('.')]}' y='{y[:y.index('.')]}'"))
                counter += 1

            i += 1
        root.append(images)
        tree = ET.ElementTree(root)

        # write the tree into an XML file
        # with open(self._name, 'w') as file:
        #
        #     file.write(doc)
        # file.close()

        tree.write(f"{self._name}", encoding='ISO-8859-1', xml_declaration=True)