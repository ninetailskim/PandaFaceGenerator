import paddlehub as hub
import cv2
import numpy as np
import math
import os
import copy
import imageio
from PIL import Image, ImageDraw, ImageFont
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

debug = False

class segUtils():
    def __init__(self):
        super().__init__()
        self.module = hub.Module(name="ace2p")

    def predict(self, frame):
        result = self.module.segmentation(images=[frame], use_gpu=True)
        result = result[0]['data']
        result[result != 13] = 0
        result[result == 13] = 1
        return result

class detUtils():
    def __init__(self):
        super(detUtils, self).__init__()
        self.module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
        self.last = None

    def distance(self, a, b):
        return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

    def iou(self, bbox1, bbox2):

        b1left = bbox1['left']
        b1right = bbox1['right']
        b1top = bbox1['top']
        b1bottom = bbox1['bottom']

        b2left = bbox2['left']
        b2right = bbox2['right']
        b2top = bbox2['top']
        b2bottom = bbox2['bottom']

        area1 = (b1bottom - b1top) * (b1right - b1left)
        area2 = (b2bottom - b2top) * (b2right - b2left)

        w = min(b1right, b2right) - max(b1left, b2left)
        h = min(b1bottom, b2bottom) - max(b1top, b2top)

        dis = self.distance([(b1left+b1right)/2, (b1bottom+b1top)/2],[(b2left+b2right)/2, (b2bottom+b2top)/2])

        if w <= 0 or h <= 0:
            return 0, dis
        
        iou = w * h / (area1 + area2 - w * h)
        return iou, dis

    def predict(self, frame):
        res = self.module.face_detection(images=[frame], use_gpu=True)
        reslist = res[0]['data']
        if len(reslist) == 0:
            if self.last is not None:
                return self.last
            else:
                return None
        elif len(reslist) == 1:
            self.last = reslist[0]
            return reslist[0]
        else:
            maxiou = -float('inf')
            maxi = 0
            mind = float('inf')
            mini = 0
            for index in range(len(reslist)):
                tiou, td = self.iou(self.last, reslist[index])
                if tiou > maxiou:
                    maxi = index
                    maxiou = tiou
                if td < mind:
                    mind = td
                    mini = index  
            if tiou == 0:
                self.last = reslist[mini]
                return reslist[mini]
            else:
                self.last = reslist[maxi]
                return reslist[maxi]

class LandmarkUtils():
    def __init__(self):
        super().__init__()
        self.module = hub.Module(name="face_landmark_localization")

    def predict(self, frame):
        result = self.module.keypoint_detection(images=[frame], use_gpu=True)
        if result is not None:
            return result[0]['data']
        else:
            return None

class FaceCut():
    def __init__(self):
        super().__init__()
        self.SU = segUtils()
        self.DU = detUtils()
        self.LU = LandmarkUtils()

    def getFace(self, frame):
        #dres = self.DU.predict(frame)
        sres = self.SU.predict(frame)
        sres3 = np.repeat(sres[:,:,np.newaxis], 3, axis=2)
        return sres3

    def getFaceByLandmark(self, frame):
        result = self.LU.predict(frame)
        mask = None
        if result is None:
            mask = None
        else:
            result = result[0]
            mask = np.zeros_like(frame).astype(np.uint8)
            pts = []
            order = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,27,26,25,20,19,18]
            
            h,w = frame.shape[:2]
            top = h
            bottom = 0
            left = w
            right = 0

            for o in order:
                tx = int(result[o-1][0])
                ty = int(result[o-1][1])
                pts.append([tx, ty])
            mask = cv2.fillPoly(mask, [np.array(pts)], (255, 255, 255)).astype(np.uint8)

        mask2 = self.getFace(frame)
        if mask is None:
            if mask2 is None:
                return None, None, None, None, None, None
            else:
                mask = mask2
        else:
            if mask2 is None:
                mask = mask
            else:
                mask = mask * mask2
        tmask = mask[:,:,0] * mask2[:,:,0]
        xaxis = np.where(np.sum(tmask, axis=0))
        yaxis = np.where(np.sum(tmask, axis=1))

        top = np.min(yaxis)
        bottom = np.max(yaxis)
        left = np.min(xaxis)
        right = np.max(xaxis)

        return mask[top:bottom, left:right], frame[top:bottom, left:right], top, bottom, left, right

def histMatch(hista, tpl):
    j = 1
    res = np.zeros_like(hista)
    for i in range(256):
        #print(i,hista[i][0],"---",j,tpl[j][0])
        while j < 255 and hista[i][0] > tpl[j][0]:
            j += 1 
        if abs(hista[i][0] - tpl[j][0]) < abs(hista[i][0] - tpl[j - 1][0]):
            res[i][0] = j
        else:
            res[i][0] = j - 1
    res = np.reshape(res, [256]).astype(np.uint8)
    return res

class myHistogram():
    def __init__(self, img):
        super().__init__()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.h, self.w = img.shape
        self.hist = self.calhistogram(img) / (self.h * self.w)
        self.p = self.hist2p()

    def calhistogram(self, image):
        return cv2.calcHist([image],[0],None, [256],[0,256])

    def hist2p(self):
        p = np.zeros_like(self.hist)
        sum = 0
        for i in range(256):
            t = self.hist[i][0]
            sum += t
            p[i][0] = sum
        return p

class PandaFace():
    def __init__(self):
        super().__init__()
        self.FC = FaceCut()
        self.top = 0
        self.bottom = 0
        self.right = 0
        self.left = 0

    def memecut(self, meme):
        mask = self.FC.getFace(meme)
        if debug:
            cv2.imshow("mask", mask * 255)
            cv2.waitKey(0)

        kernel = np.ones((3,3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        ones255 = np.ones_like(mask) * 255
        meme = (ones255 * mask + meme * (1 - mask)).astype(np.uint8)
        return meme

    def constrast_img(self, img1, con=2.2, bri=3, name=None):
        rows, cols, channels = img1.shape
        blank = np.zeros([rows, cols, channels], img1.dtype)
        dst = cv2.addWeighted(img1, con, blank, 1-con, bri)
        grey = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        #O = cv2.equalizeHist(grey)
        #img = np.concatenate([img1, dst], axis=1)
        #img2 = np.concatenate([grey, O], axis=1)
        #return img , img2 
        #grey = cv2.equalizeHist(grey)
        grey =  cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        if name is not None:
            cv2.imwrite(name, grey)
        return grey 

    def mf(self, mask, face, name=None):
        back = np.ones_like(face) * 255
        res = (mask / 255) * face + (1-(mask / 255)) * back
        res = res.astype(np.uint8)
        if name is not None:
            cv2.imwrite(name, res)
        return res

    def facecut(self, face):
        mask, face, _, _, _, _ = self.FC.getFaceByLandmark(face)
        if mask is None:
            return None, None
        res = self.mf(mask, face)
        image = self.constrast_img(res)
        if debug:
            cv2.imshow("imagelut", image)
            cv2.waitKey(0)
        return image, mask

    def compose(self, meme, face):
        memem, memef, top, bottom, left, right = self.FC.getFaceByLandmark(meme)
        neme = cv2.rectangle(copy.deepcopy(meme), (left, top), (right, bottom), (0, 255, 0), 2)
        if debug:
            cv2.imshow("newme", neme)
            cv2.waitKey(0)
        meme = self.memecut(meme)
        face,mask = self.facecut(face)
        h,w = face.shape[:2]
        mh = bottom - top - 10
        mw = right - left - 10
        # print(mh, mw)
        cx = int((right + left) / 2)
        cy = int((top + bottom) / 2)
        neww = mw
        newh = mh
        if h/w < mh/mw:
            #w
            neww = mw
            newh = int(neww / w * h)
            face = cv2.resize(face, (neww, newh))
            mask = cv2.resize(mask, (neww, newh))
        else:
            #h
            newh = mh
            neww = int(newh / h * w)
            face = cv2.resize(face, (neww, newh))
            mask = cv2.resize(mask, (neww, newh))
        # print(newh, neww)
        mres = self.mf(memem, memef)
        memehist = myHistogram(mres)
        facehist = myHistogram(face)
        lut = histMatch(facehist.p, memehist.p)
        image = cv2.LUT(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), lut)
        if debug:
            cv2.imshow("tmplut", mres)
            cv2.waitKey(0)
            cv2.imshow("facelut", image)
            cv2.waitKey(0)

        cx -= int(neww / 2)
        cy -= int(newh / 2)
        if debug:
            cv2.imshow("meme", meme.astype(np.uint8))
            cv2.waitKey(0)
        meme[cy:cy+newh, cx:cx+neww] = face * (mask / 255) + meme[cy:cy+newh, cx:cx+neww] * (1 - mask / 255)
        meme.astype(np.uint8)
        return meme

    def dmeme(self, memepath):
        # print(memepath)
        meme = cv2.imread(memepath)
        memem, memef, self.top, self.bottom, self.left, self.right = self.FC.getFaceByLandmark(meme)
        # debug
        neme = cv2.rectangle(copy.deepcopy(meme), (self.left, self.top), (self.right, self.left), (0, 255, 0), 2)
        if debug:
            cv2.imshow("newme", neme)
            cv2.waitKey(0)

        meme = self.memecut(meme)
        return meme, memem, memef

    def pickmeme(self):
        memepathlist = glob.glob("origin/*.png")
        lenpl = len(memepathlist)
        while True:
            index = np.random.randint(0, lenpl)
            yield memepathlist[index]
    
    def dfaceAC(self, face, meme, memem, memef, textarea=None, ismatch=False):
        face, mask = self.facecut(face)
        if face is None:
            return None
        h, w = face.shape[:2]
        mh = self.bottom - self.top - 10
        mw = self.right - self.left - 10
        cx = int((self.right + self.left) / 2)
        cy = int((self.top + self.bottom) / 2)
        
        neww = mw
        newh = mh

        if h / w < mh / mw:
            neww = mw
            newh = int(neww / w * h)
            face = cv2.resize(face, (neww, newh))
            mask = cv2.resize(mask, (neww, newh))
        else:
            newh = mh
            neww = int(newh / h * w)
            face = cv2.resize(face, (neww, newh))
            mask = cv2.resize(mask, (neww, newh))
        if ismatch:
            mres = self.mf(memem, memef)
            memehist = myHistogram(mres)
            facehist = myHistogram(face)
            lut = histMatch(facehist.p, memehist.p)
            image = cv2.LUT(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), lut)
            if debug:
                cv2.imshow("tmplut", mres)
                cv2.waitKey(0)
                cv2.imshow("facelut", image)
                cv2.waitKey(0)
            face = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cx -= int(neww / 2)
        cy -= int(newh / 2)
        # debug
        if debug:
            cv2.imshow("meme", meme.astype(np.uint8))
            cv2.waitKey(0)

        meme[cy:cy+newh, cx:cx+neww] = face * (mask / 255) + meme[cy:cy+newh, cx:cx+neww] * (1 - mask / 255)
        meme.astype(np.uint8)

        h, w = meme.shape[:2]
        fw = 500
        fh = int(fw / w * h)

        meme = cv2.resize(meme, (fw, fh))
        
        if textarea is not None:
            meme = np.concatenate([meme, textarea], axis=0)

        return meme

    def addText(self, width, text, textColor=(0, 0, 0), textSize=50):

        lt = len(text)
        line = math.ceil(lt / 8)

        area = np.ones([50 * line, width], dtype=np.uint8) * 255
        area = Image.fromarray(cv2.cvtColor(area, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(area)

        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")

        fline = lt % 8
        if fline == 0:
            fline = 8
        
        if fline + 8 <= 10:
            lastword = text[-fline-8:]
            text = text[0:-fline-8]
            texts = []
            for i in range(0,len(text), 8):
                texts.append(text[i:i+8])
            texts.append(lastword)
        else:
            texts  = [text[:fline]]
            for i in range(fline, lt, 8):
                texts.append(text[i:i+8])
        
        for row in range(len(texts)):
            sx = (width - len(texts[row]) * 50 ) / 2
            sy = row * 50
            draw.text((sx, sy), texts[row], textColor, font=fontStyle)

        return cv2.cvtColor(np.asarray(area), cv2.COLOR_RGB2BGR)


    def resize(self, frame, ml):
        h, w = frame.shape[:2]
        while max(w, h) > ml:
            w /= 2
            h /= 2
        frame = cv2.resize(frame, (int(w), int(h)))
        return frame


    def finalCompose(self, res, text=None):
        #pick out meme
        memepath = next(self.pickmeme())

        #deal with meme
        meme, memem, memef = self.dmeme(memepath)

        textarea = None
        if text is not None:
            textarea = self.addText(500, text)


        #deal with user's res
        filename, ext = os.path.splitext(res)

        resource = "Q:\\LAB\\Django\\wechaty-getting-started\\"+res
        if ext == ".mp4":
            gif = []
            cap = cv2.VideoCapture(resource)
            index = 0
            ret = True
            while cap.isOpened() and ret:
                ret, frame = cap.read()
                frame = self.resize(frame,800)
                timg = self.dfaceAC(frame, meme, memem, memef, textarea)
                if timg is None:
                    cut = 8
                    while cut > 0:
                        ret, frame = cap.read()
                        cut -= 1
                    continue
                gif.append(cv2.cvtColor(timg,cv2.COLOR_BGR2RGB))
                # print(index)
                index += 1
                cut = 8
                while cut > 0:
                    ret, frame = cap.read()
                    cut -= 1
            # print(len(gif))
            if len(gif) == 0:
                return False
            imageio.mimsave("res"+filename+".gif", gif, fps=5)
            return "res"+filename+".gif"

        else:
            image = self.resize(cv2.imread(resource), 800)
            image = self.dfaceAC(image, meme, memem, memef, textarea)
            cv2.imwrite("res"+res, image)
            if debug:
                cv2.imshow("res", image)
                cv2.waitKey(0)
            return "res"+res
        return False

PF = PandaFace()

def main():

    PF.finalCompose("testvideo.mp4", "我爱你,亲爱的姑娘")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()