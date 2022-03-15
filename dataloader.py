# Image pair dataloader
class ImagePairDataset(data.Dataset):
    def __init__(self, ImgADir, ImgBDir, MatchAFile, MatchBFile, NonMatchAFile, NonMatchBFile, Augmentation):
        # Image path
        self.imgADir = ImgADir
        self.imgBDir = ImgBDir
        self.matchA = MatchAFile
        self.matchB = MatchBFile
        self.nonMatchA = NonMatchAFile
        self.nonMatchB = NonMatchBFile
        self.augmentation = Augmentation
        self.rotate = False
        self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.colorJitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    # Overloaded len method
    def __len__(self):
        return len(os.listdir(self.imgADir))
    # Overloaded getter method
    def __getitem__(self, idx):
        # reinit random rotation flag
        self.rotate = False
        # Load specific image with index and convert to color tensor with shape [B,C,H,W]
        imgA = read_image(self.imgADir+"/"+str(idx)+".png")
        imgB = read_image(self.imgBDir+"/"+str(idx)+".png")
        H = imgA.size()[2]
        W = imgA.size()[3]
        # Data augmentation for the image B (match and training)
        if self.augmentation == True:
            imgB = self.colorJitter(imgB)
            if random.random() > 0.6:
                self.rotate = True
                imgB = K.geometry.transform.rot180(imgB)
        # get the current match and non-match from the mega-list
        listMA =
        listMB =
        listNMA =
        listNMB =
        # linearize keypoints and respect rotation
        MA = 0
        NMA = 0
        if self.rotate == True:
            MB = 0
            NMB = 0
        else:
            MB = 0
            NMB = 0
        # Normalize image for training
        imgA = self.normalization(imgA/255.)
        imgB = self.normalization(imgB/255.)
        # create a dictionnary for access
        pair = {'image A': imgA, 'image B': imgB, 'MatchA':MA, 'MatchB':MB, 'NonMatchA':NMA, 'NonMatchB':NMB,}
        return pair
