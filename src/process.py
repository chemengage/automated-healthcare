from model import detection

class Mitosisdetection:
    def __init__(self):
        #self.input_image = input_image
        #super().__init__()

        # network parameters
        self.size = 512
        self.patch_size = 250
        self.batchsize = 10

        # relative paths to model files
        # local directory structure same as docker
        path_model_od = './models/CODAEL_OD_v1_weights.pth'
        path_model_cp = './models/patch_classifier_CODAEL_v0_weights.pth'
        path_model_ve = './models/vision_encoder'
        path_model_te = './models/text_encoder'
        path_text = './models/unique_texts.csv'

        self.d = detection(path_model_od, path_model_cp, path_model_ve, path_model_te, path_text, self.size, self.patch_size, self.batchsize)

        vision_encoder = self.d.load_model_ve()
        text_encoder = self.d.load_model_te()
        model_od = self.d.load_model_od()
        model_cp = self.d.load_model_cp()

    def predict(self, input_image):
        result = self.d.patch_classifier(input_image)

        superimposed_img = self.d.heatmap(input_image, result)

        return result, superimposed_img