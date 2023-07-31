Pytorch_Data_preprocessing_API = ["Resize(", "v2.Resize(", "v2.ScaleJitter(", "v2.RandomShortestSize(", "v2.RandomResize(",
"RandomCrop(", "v2.RandomCrop(", "RandomResizedCrop(", "v2.RandomResizedCrop(", "v2.RandomIoUCrop(",
"CenterCrop(", "v2.CenterCrop(", "FiveCrop(", "v2.FiveCrop(", "TenCrop(", "v2.TenCrop(", "Pad(",
"v2.Pad(", "v2.RandomZoomOut(", "RandomRotation(", "v2.RandomRotation(", "RandomAffine(",
"v2.RandomAffine(", "RandomPerspective(", "v2.RandomPerspective(", "ElasticTransform(",
"v2.ElasticTransform(", "RandomHorizontalFlip(", "v2.RandomHorizontalFlip(", "RandomVerticalFlip(",
"v2.RandomVerticalFlip(", "ColorJitter(", "v2.ColorJitter(", "v2.RandomPhotometricDistort(",
"Grayscale(", "v2.Grayscale(", "RandomGrayscale(", "v2.RandomGrayscale(", "GaussianBlur(",
"v2.GaussianBlur(", "RandomInvert(", "v2.RandomInvert(", "RandomPosterize(", "v2.RandomPosterize(",
"RandomSolarize(", "v2.RandomSolarize(", "RandomAdjustSharpness(", "v2.RandomAdjustSharpness(",
"RandomAutocontrast(", "v2.RandomAutocontrast(", "RandomEqualize(", "v2.RandomEqualize(",
"Compose(", "v2.Compose(", "RandomApply(", "v2.RandomApply(", "RandomChoice(", "v2.RandomChoice(",
"RandomOrder(", "v2.RandomOrder(", "LinearTransformation(", "v2.LinearTransformation(", "Normalize(",
"v2.Normalize(", "RandomErasing(", "v2.RandomErasing(", "Lambda(", "v2.Lambda(", "v2.SanitizeBoundingBox(",
"v2.ClampBoundingBox(", "v2.UniformTemporalSubsample(", "ToPILImage(", "v2.ToImagePIL(", "ToTensor(",
"v2.ToTensor(", "PILToTensor(", "v2.PILToTensor(", "v2.ToImageTensor(", "ConvertImageDtype(",
"v2.ConvertDtype(", "v2.ToDtype(", "v2.ConvertBoundingBoxFormat(", "AutoAugmentPolicy(",
"AutoAugment(", "v2.AutoAugment(", "RandAugment(", "v2.RandAugment(", "TrivialAugmentWide(",
"v2.TrivialAugmentWide(", "AugMix(", "v2.AugMix(", "adjust_brightness(", "adjust_contrast(",
"adjust_gamma(", "adjust_hue(", "adjust_saturation(", "adjust_sharpness(", "affine(", "autocontrast(",
"center_crop(", "convert_image_dtype(", "crop(", "equalize(", "erase(", "five_crop(", "gaussian_blur(",
"get_dimensions(", "get_image_num_channels(", "get_image_size(", "hflip(", "invert(", "normalize(",
"pad(", "perspective(", "pil_to_tensor(", "posterize(", "resize(", "resized_crop(", "rgb_to_grayscale(",
"rotate(", "solarize(", "ten_crop(", "to_grayscale(", "to_pil_image(", "to_tensor(", "vflip(","SubsetRandomSampler(","Sampler(","SequentialSampler(","RandomSampler(",
"WeightedRandomSampler(","BatchSampler(","DistributedSampler("]

SKlearn_Data_preprocessing_API = [
"Binarizer(",
"FunctionTransformer(",
"KBinsDiscretizer(",
"KernelCenterer(",
"LabelBinarizer(",
"LabelEncoder(",
"MultiLabelBinarizer(",
"MaxAbsScaler(",
"MinMaxScaler(",
"Normalizer(",
"OneHotEncoder(",
"OrdinalEncoder(",
"PolynomialFeatures(",
"PowerTransformer(",
"QuantileTransformer(",
"RobustScaler(",
"SplineTransformer(",
"StandardScaler(",
"add_dummy_feature(",
"binarize(",
"label_binarize(",
"maxabs_scale(",
"minmax_scale(",
"normalize(",
"quantile_transform(",
"robust_scale(",
"scale(",
"power_transform("
]

Pytorch_activation_functions_API = ["threshold(","threshold_(","relu(","relu_(","hardtanh(","hardtanh_(","hardswish(","relu6(","elu(","elu_(","selu(","celu(","leaky_relu(","leaky_relu_(","prelu(","rrelu(","rrelu_(","glu(","gelu(","logsigmoid(","hardshrink(","tanhshrink(","softsign(","softplus(","softmin(","softmax(","softshrink(","gumbel_softmax(","log_softmax(","tanh(","sigmoid(","hardsigmoid(","silu(","mish(","batch_norm(","group_norm(","instance_norm(","layer_norm(","local_response_norm(","normalize("]

Pytorch_all_loss_functions_API = [" .L1Loss(", " .MSELoss(", " .CrossEntropyLoss(", " .CTCLoss(", " .NLLLoss(", " .PoissonNLLLoss(", " .GaussianNLLLoss(", " .KLDivLoss(", " .BCELoss(", " .BCEWithLogitsLoss(", " .MarginRankingLoss(", " .HingeEmbeddingLoss(", " .MultiLabelMarginLoss(", " .HuberLoss(", " .SmoothL1Loss(", " .SoftMarginLoss(", " .MultiLabelSoftMarginLoss(", " .CosineEmbeddingLoss(", " .MultiMarginLoss(", " .TripletMarginLoss(", " .TripletMarginWithDistanceLoss(", " .binary_cross_entropy_with_logits(", " .binary_cross_entropy(", " .binary_cross_entropy(", " .binary_cross_entropy_with_logits(", " .poisson_nll_loss(", " .cosine_embedding_loss(", " .cross_entropy(", " .ctc_loss(", " .gaussian_nll_loss(", " .hinge_embedding_loss(", " .kl_div(", " .l1_loss(", " .mse_loss(", " .margin_ranking_loss(", " .multilabel_margin_loss(", " .multilabel_soft_margin_loss(", " .multi_margin_loss(", " .nll_loss(", " .huber_loss(", " .smooth_l1_loss(", " .soft_margin_loss(", " .triplet_margin_loss(", " .triplet_margin_with_distance_loss("]

Pytorch_loss_functions_API_need_activation_fuctions =["NLLLoss(", "PoissonNLLLoss(", "KLDivLoss(", "BCELoss(","CTCLoss(","MultiLabelSoftMarginLoss(","KLDivLoss(","binary_cross_entropy("]

Pytorch_loss_functions_API_have_activation_fuctions =["CrossEntropyLoss(","BCEWithLogitsLoss(","binary_cross_entropy_with_logits("]