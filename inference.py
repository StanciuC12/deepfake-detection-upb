import torch
from model import CNN
from preprocessing import Preprocessor

class DeepfakeDetector:

    def __init__(self, model_path='models/Xception_upb_fullface_epoch_15_param_FF++_186_2230.pkl', batch_size: int=16, max_frames: int =None):


        self.model_path = model_path
        self.model_type = 'Xception'
        self.model = CNN(pretrained=True, architecture=self.model_type)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.preprocessor = Preprocessor(scaleFactor=1.3, minNeighbors=10, img_dim=(299, 299))
        if batch_size is not None:
            self.batch_size = batch_size
        if max_frames is not None:
            self.max_frames = max_frames

    def inference(self, video_path=None, img_path=None):

        print('Inference started...')
        print('Video path: ', video_path)

        if video_path is not None:
            video = self.preprocessor.preprocess_video(video_path)
        else:
            video = self.preprocessor.preprocess(img_path)

        if video is None:
            print('Problem with detecting face in the file/ reading the file!!!')
            return None

        print('Video shape: ', video.shape)

        if len(video.shape) == 3: # for one frame
            video = video.unsqueeze(0)
            print('Video shape after unsqueeze: ', video.shape)
            with torch.no_grad():
                output = self.model(video)

            output = output.to('cpu').detach().numpy()[0][0]

            return output

        else:  # for a video
            if self.max_frames is not None:
                video = video[:self.max_frames]
            outputs = []
            for batch in range(video.shape[0]//self.batch_size):
                print(f'Batch: {batch}/{video.shape[0]//self.batch_size}')
                batch = video[batch*self.batch_size:(batch+1)*self.batch_size]
                with torch.no_grad():
                    output = self.model(batch)

                outputs.append(output.to('cpu').detach())

            mean_output = float(torch.mean(torch.cat(outputs)).numpy())

        return mean_output



if __name__ == "__main__":


    video_path = 'demo/038_125_deepfake.mp4'
    image_path = 'demo/038_125_deepfake.PNG'
    image_path = 'demo/check_foto_left.jpg'
    detector = DeepfakeDetector(max_frames=100)
    out = detector.inference(video_path=None, img_path=image_path)

    print('Output: ', out) # 1=deepfake, 0=real
    print('Prediction: ', 'FAKE' if out > 0.5 else 'REAL')

