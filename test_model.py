# USAGE
# python test_model.py --images test_images --codebook output/vocab.cpickle --model output/model.cpickle

# import the necessary packages
from __future__ import print_function
from descriptors.detectanddescribe import DetectAndDescribe
from ir.bagofvisualwords import BagOfVisualWords
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
from result_montage import ResultsMontage
import argparse
import pickle
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="Path to input images directory")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the codebook")
ap.add_argument("-m", "--model", required=True,
	help="Path to the classifier")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the classifier
model = pickle.loads(open(args["model"], "rb").read())

num_tests = len(list(paths.list_images(args["images"])))
montage = ResultsMontage((240, 320), 3, num_tests)

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
	# load the image and prepare it from description
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = imutils.resize(gray, width=min(320, image.shape[1]))

	# describe the image and classify it
	(kps, descs) = dad.describe(gray)
	hist = bovw.describe(descs)
	hist /= hist.sum()
	prediction = model.predict(hist)[0]

	# show the prediction
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[PREDICTION] {}: {}".format(filename, prediction))
	montage.addResult(image, text="#{}".format(prediction))

# show the output image of results
cv2.imshow("Results", imutils.resize(montage.montage, height=400))
cv2.waitKey(0)