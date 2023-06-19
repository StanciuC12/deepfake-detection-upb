import cv2
import torch
from torchvision import transforms


class Preprocessor:

    def __init__(self, transform=None):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.scaleFactor = 1.15
        self.minNeighbors = 8
        self.img_dim = (299, 299)
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])

    def preprocess(self, img):

        if type(img) == str:
            img = cv2.imread(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors)
        if len(faces) == 0:
            return None
        elif len(faces) > 1:
            print("More than one face detected")
            return None
        else:
            for (x, y, w, h) in faces:
                faces = img[y:y + h, x:x + w]
                resized = cv2.resize(faces, self.img_dim)
                # cv2.imshow("face", resized)
                # cv2.waitKey()

            if self.transform:
                resized = self.transform(resized)

            return resized

    def preprocess_video(self, video_adr):

        print('Preprocessing video: ', video_adr)
        cap = cv2.VideoCapture(video_adr)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = self.preprocess(frame)
            if frame is not None:
                frames.append(frame)

        cap.release()

        frames = torch.stack(frames)
        frames = frames.reshape(-1, 3, 299, 299)

        print('Preprocessing finished')

        return frames


if __name__ == '__main__':

    img_adr = r'C:\Users\Cristi\Desktop\Doctorat\Deepfakes\138_142_blending_artifact.png'
    img_preprocess = Preprocessor()
    img = img_preprocess.preprocess(img_adr)

    video_adr = r'C:\Users\Cristi\Desktop\Doctorat\Deepfakes\038_125_deepfake.mp4'
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    frames = img_preprocess.preprocess_video(video_adr)
    print(frames.shape)