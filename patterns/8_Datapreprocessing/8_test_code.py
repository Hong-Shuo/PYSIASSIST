
TRANSFORM_IMG = transforms.Compose([
                                    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                    transforms.RandomRotation(degrees=15),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(size=224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                   ])