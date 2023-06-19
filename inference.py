import torch
from model import CNN
from preprocessing import Preprocessor

class DeepfakeDetector:

    def __init__(self):

        self.model_path = 'models/Xception_upb_fullface_epoch_25_param_FF++_186_2346.pkl'
        self.model_type = 'Xception'
        self.model = CNN(pretrained=True, architecture=self.model_type)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.preprocessor = Preprocessor()
        self.batch_size = 16

    def inference(self, video_path=None, img_path=None):

        print('Inference started...')
        print('Video path: ', video_path)

        if video_path is not None:
            video = self.preprocessor.preprocess_video(video_path)
        else:
            video = self.preprocessor.preprocess(img_path)
        print('Video shape: ', video.shape)
        print('Video dtype: ', video.dtype)

        if len(video.shape) == 3: # for one frame
            video = video.unsqueeze(0)
            print('Video shape after unsqueeze: ', video.shape)
            with torch.no_grad():
                output = self.model(video)

            return output.to('cpu')

        else:  # for a video
            outputs = []
            for batch in range(video.shape[0]//self.batch_size):
                print(f'Batch: {batch}/{video.shape[0]//self.batch_size}')
                batch = video[batch*self.batch_size:(batch+1)*self.batch_size]
                with torch.no_grad():
                    output = self.model(batch)

                outputs.append(output.to('cpu'))

            mean_output = torch.mean(torch.cat(outputs))

        return mean_output



if __name__ == "__main__":


    video_path = 'demo/038_125_deepfake.mp4'
    video_path = 'demo/038_125_deepfake.PNG'
    detector = DeepfakeDetector()
    out = detector.inference(video_path=None, img_path=video_path)

    print('Output: ', out) # 1=deepfake, 0=real

