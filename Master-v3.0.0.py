import os
import threading

import cv2
import dlib
import numpy as np

from tkinter import *
from tkinter import font, filedialog
from PIL     import ImageTk, Image

def extract_index_nparray(nparray):
    for num in nparray[0]: return num
    return None

def extract_index_subdiv(subdiv, points):
    indexes = list()
    for t in np.array(subdiv.getTriangleList(), dtype=np.int32):
        index_pt1 = extract_index_nparray(np.where((points == tuple(t[:2])).all(axis = 1)))
        index_pt2 = extract_index_nparray(np.where((points == tuple(t[2:4])).all(axis = 1)))
        index_pt3 = extract_index_nparray(np.where((points == tuple(t[4: ])).all(axis = 1)))
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            indexes.append([index_pt1, index_pt2, index_pt3])
    return indexes

class Helper:
    def get_exportable(src): return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)))
    def get_fix(src, cpn): return Helper.get_exportable(Helper.resize(src, cpn['width'], cpn['height']))

    def resize(image, width, height):
        img_width, img_height = image.shape[:2]
        ratio = width / img_width
        if img_height > img_width: ratio = height / img_height
        return cv2.resize(image, None, fx=ratio, fy=ratio)



class CaptureEngine:
    def __init__(self) -> None:
        self.cap_device = cv2.VideoCapture(0)
        self.last_frame = None

    def read(self):
        _, self.last_frame = self.cap_device.read()
        return self.last_frame



class dVMSEngine:
    def __init__ (self):
        self.source_image                 = None
        self.source_landmarks             = None
        self.img_gray                     = None
        self.face_image_1                 = None
        self.indexes_triangles            = None
        self.indexes_triangles_convexhull = None

        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("68_landmarks.model")


    def set_source(self, image):
        self.source_image = image
        grayscale = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)

        for face in self.detector(grayscale):
            landmarks = self.predictor(grayscale, face)
            self.source_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            points = np.array(self.source_landmarks, np.int32)
            convex = cv2.convexHull(points)

            rect   = cv2.boundingRect(convex)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(self.source_landmarks)
            self.indexes_triangles = extract_index_subdiv(subdiv, points)

            rect   = cv2.boundingRect(convex)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(convex)
            for lanmark_index in [27, 30, 36, 39, 42, 45]:
                subdiv.insert(self.source_landmarks[lanmark_index])
            self.indexes_triangles_convexhull = extract_index_subdiv(subdiv, points)

    def swapping(self, target_image):
        grayscale = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        def swapface(indexes_triangles, clone_mode):
            img2_new_face = np.zeros_like(target_image)

            for face in self.detector(grayscale):
                landmarks = self.predictor(grayscale, face)
                landmarks_points2 = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

                points2 = np.array(landmarks_points2, np.int32)
                convexhull2 = cv2.convexHull(points2)

                for triangle_index in indexes_triangles:
                    tr1_pt1 = self.source_landmarks[triangle_index[0]]
                    tr1_pt2 = self.source_landmarks[triangle_index[1]]
                    tr1_pt3 = self.source_landmarks[triangle_index[2]]
                    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                    (x, y, w, h) = cv2.boundingRect(triangle1)
                    cropped_triangle = self.source_image[y: y + h, x: x + w]
                    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)


                    tr2_pt1 = landmarks_points2[triangle_index[0]]
                    tr2_pt2 = landmarks_points2[triangle_index[1]]
                    tr2_pt3 = landmarks_points2[triangle_index[2]]

                    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
                    (x, y, w, h) =  cv2.boundingRect(triangle2)

                    cropped_tr2_mask = np.zeros((h, w), np.uint8)
                    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
                    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                    points  = np.float32(points)
                    points2 = np.float32(points2)

                    M = cv2.getAffineTransform(points, points2)
                    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask = cropped_tr2_mask)

                    img2_new_face_rect_area      = img2_new_face[y: y + h, x: x + w]
                    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                    _, mask_triangles_designed   = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                    warped_triangle              = cv2.bitwise_and(warped_triangle, warped_triangle, mask = mask_triangles_designed)

                    img2_new_face_rect_area           = cv2.add(img2_new_face_rect_area, warped_triangle)
                    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area


            img2_face_mask = np.zeros_like(grayscale)
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv2.bitwise_not(img2_head_mask)

            img2_head_noface = cv2.bitwise_and(target_image, target_image, mask = img2_face_mask)
            result           = cv2.add(img2_head_noface, img2_new_face)

            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center = ((x + x + w) // 2, (y + y + h) // 2)
            return cv2.seamlessClone(result, target_image, img2_head_mask, center, clone_mode)

        return (swapface(self.indexes_triangles, cv2.MIXED_CLONE),
                swapface(self.indexes_triangles_convexhull, cv2.NORMAL_CLONE))



class RenderEngine:
    WND_WIDTH, WND_HEIGHT = (1200, 800)
    SQR_WIDTH, SQR_HEIGHT = (384 , 384)

    def __init__(self) -> None:
        self.app = Tk()
        self.app.title('FACE dVMS')
        self.app.geometry('{}x{}'.format(self.WND_WIDTH, self.WND_HEIGHT))
        font.nametofont('TkDefaultFont').configure(family='Source Code Pro for Powerline', size=12)
        self.local_path = os.path.abspath(os.path.dirname(__file__))

        _sb = Frame(self.app, bd=0, relief=FLAT)
        _sb.pack(side=BOTTOM, fill=X)
        _sb_label = Label(_sb, text=' dVMS ', bd=0, bg='#16825d', fg='white')
        _sb_label.pack(side=LEFT)
        self.sb_text = Label(_sb, text='...', bd=0, bg='#007acc', fg='white')
        self.sb_text.pack(side=LEFT, fill=X, expand=TRUE)
        self.load_components()

    def select_capture(self, event=None):
        self.if_photo.image = self.if_capture.image
        self.dvms_engine.set_source(self.capture_engine.last_frame)

        self.if_photo.config(image=self.if_photo.image)
        self.sb_text.config(text='LOAD::SUCCESS::CAPTURE_DEVICE')
        pass

    def select_source(self, event=None):
        filepath = filedialog.askopenfilename(
            initialdir=self.local_path, title="Select An Image",
            filetypes=(("JPEG", "*.jpg"), ("GIF", "*.gif*"), ("PNG", "*.png")))

        if len(filepath):
            self.sb_text.config(text='LOAD::{}'.format(filepath))
            image = cv2.imread(filepath)
            if image is None: self.sb_text.config(text='LOAD::FAIL::{}'.format(filepath))
            else:
                self.source_image = image
                self.dvms_engine.set_source(image)
                self.if_photo.image = Helper.get_fix(self.source_image, self.if_photo)
                self.if_photo.config(image=self.if_photo.image)
                self.sb_text.config(text='LOAD::SUCCESS::{}'.format(filepath))

    def load_components(self) -> None:
        im_noimage = cv2.imread('Photos/NO_IMAGE.PNG')

        # INPUT_FRAME
        _if = Frame(self.app, bg='#fff')
        _if.pack(side=TOP)
        self.if_capture = Label(_if, width=self.SQR_WIDTH, height=self.SQR_HEIGHT)
        self.if_capture.image = Helper.get_fix(im_noimage, self.if_capture)
        self.if_capture.config(image=self.if_capture.image)
        self.if_capture.bind('<Button-1>', self.select_capture)
        self.if_photo = Label(_if, width=self.SQR_WIDTH, height=self.SQR_HEIGHT)
        self.if_photo.image = Helper.get_fix(im_noimage, self.if_photo)
        self.if_photo.config(image=self.if_photo.image)
        self.if_photo.bind('<Button-1>', self.select_source)

        Label(_if,  text=' +/device0 ').grid(row=0, column=0)
        Label(_if,  text=' +/input0 ').grid(row=0, column=1)
        self.if_capture.grid(row=1, column=0, padx=5, pady=5)
        self.if_photo.grid(row=1, column=1, padx=5, pady=5)

        # OUTPUT_FRAME
        _of = Frame(self.app, bg='#fff')
        _of.pack(side=TOP)
        self.of_feature0 = Label(_of, width=self.SQR_WIDTH, height=self.SQR_HEIGHT)
        self.of_feature0.image = Helper.get_fix(im_noimage, self.of_feature0)
        self.of_feature0.config(image=self.of_feature0.image)
        self.of_feature1 = Label(_of, width=self.SQR_WIDTH, height=self.SQR_HEIGHT)
        self.of_feature1.image = Helper.get_fix(im_noimage, self.of_feature1)
        self.of_feature1.config(image=self.of_feature1.image)

        Label(_of,  text=' -/dev/feature0 ').grid(row=0, column=0)
        Label(_of,  text=' -/dev/feature1 ').grid(row=0, column=1)
        self.of_feature0.grid(row=1, column=0, padx=5, pady=5)
        self.of_feature1.grid(row=1, column=1, padx=5, pady=5)

    def load_engine(self) -> None:
        self.sb_text.config(text='LOAD::dVMS_ENGINE')
        self.dvms_engine = dVMSEngine()

        self.sb_text.config(text='LOAD::CAPTURE_ENGINE')
        self.capture_engine = CaptureEngine()

        self.sb_text.config(text='LOAD::SUCCESS')
        self.capture_handle()

    def capture_handle(self):
        capture_frame = self.capture_engine.read()
        self.if_capture.image = Helper.get_fix(capture_frame, self.if_capture)
        self.if_capture.config(image=self.if_capture.image)

        try:
            a, b = self.dvms_engine.swapping(capture_frame)
            self.of_feature0.image = Helper.get_fix(a, self.of_feature0)
            self.of_feature1.image = Helper.get_fix(b, self.of_feature1)
        except:
            self.of_feature0.image = Helper.get_fix(capture_frame, self.if_capture)
            self.of_feature1.image = Helper.get_fix(capture_frame, self.if_capture)

        self.of_feature0.config(image=self.of_feature0.image)
        self.of_feature1.config(image=self.of_feature1.image)
        self.app.after(10, self.capture_handle)

    def start(self) -> None:
        threading.Thread(target=self.load_engine).start()
        self.app.mainloop()

if __name__ == '__main__':
    render_ = RenderEngine()
    render_.start()
