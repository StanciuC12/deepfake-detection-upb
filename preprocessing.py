import cv2
import torch
from torchvision import transforms


class Preprocessor:

    def __init__(self, scaleFactor=1.15, minNeighbors=6, img_dim=(299, 299)):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.img_dim = img_dim
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])

    def preprocess(self, img):

        if type(img) is str:
            img = cv2.imread(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors)
        if len(faces) == 0:
            return None
        elif len(faces) > 1:
            print("More than one face detected in this frame")
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
        failed = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = self.preprocess(frame)
            if frame is not None:
                frames.append(frame)
            else:
                failed += 1


        cap.release()

        frames = torch.stack(frames)
        frames = frames.reshape(-1, 3, 299, 299)

        print(f'Preprocessing finished, failed frames: {failed}')

        return frames


if __name__ == '__main__':

    video_path = 'demo/038_125_deepfake.mp4'
    image_path = 'demo/038_125_deepfake.PNG'
    img_preprocess = Preprocessor()
    img = img_preprocess.preprocess(image_path)


    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    frames = img_preprocess.preprocess_video(video_path)
    print(frames.shape)