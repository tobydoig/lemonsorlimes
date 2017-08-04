"""
Download classes of images from imagenet and then prepare them into a directory structure that Keras is expecting
so we can play with image classification.

Doing the fast.ai course and lesson 1 trains a vgg classifier using some dogs and cats pictures from a previous
kaggle competition. to practise this on a different dataset i thought i'd do a lemon, lime or both classifier.
that required downloading images (thanks imagenet) and putting them (randomly) into a particular directory structure
that Keras is expecting, so this seemed the perfect little project to use to learn python.

"""
import os
import urllib2
from string import split
from ssl import CertificateError
from random import shuffle
from shutil import copyfile, rmtree


def loadurl(url):
    try:
        response = urllib2.urlopen(url=url, timeout=5)
        data = response.read()
        if response.getcode() == 200 and response.geturl() == url: return data
    except (urllib2.URLError, CertificateError, IOError):
        print "Error: " + url
    return None


def writetofile(path, data):
    f = open(path, "wb")
    f.write(bytearray(data))
    f.close()


def downloadimage(path, url):
    """
    you give me a url of an image and i'll save it to path. if the url does a redirect then i'll skip the image
    as i'm treating it as moved/deleted.
    :param path: where to save the image (must end with /)
    :param url: the url of the image
    :return: the image data or None if an error occurred
    """
    data = loadurl(url)
    if data is not None:
        name = url[url.rfind("/") + 1:]
        try:
            writetofile(path + name, data)
            return data
        except IOError:
            return None


def downloadimages(path, urls):
    """
    given an array of image urls i'll download them, save them in path and return you the number i saved
    :param path: where to store the images (must end with /)
    :param urls: the array of image urls
    :return: the number successfully saved
    """
    c = 0
    for url in urls[:]:
        print url
        if downloadimage(path, url) is not None: c+= 1
    return c


def downloadfromimagenet(path, url):
    """
    you give me a url, i'll download it and assume the result is a list of image urls which i'll then download and
    save in path which i'll also create if necessary
    :param path: where to store the images (must end with /)
    :param url: where to get the list of image urls
    :return:
    """
    makedirsquietly(path)
    images = loadurl(url)
    if images is not None:
        c = downloadimages(path, split(images))
        print "Got " + str(c) + " images"


def makedirsquietly(path):
    try:
        os.makedirs(path)
    except os.error:
        pass


def makekerasdirectories(path, classes):
    """
    make the keras directory structure. we make path/test, path/sample/train/*, path/sample/valid/*, path/train/*
    and path/valid/* where * is replaced with each class name.
    :param path: the directory into which the keras directories are made (must end with /)
    :param classes: an array of class names for your images
    :return:
    """
    makedirsquietly(path + "test")
    makedirsquietly(path + "sample/train")
    makedirsquietly(path + "sample/valid")
    for c in classes[:]:
        makedirsquietly(path + "train/" + c)
        makedirsquietly(path + "valid/" + c)
        makedirsquietly(path + "sample/train/" + c)
        makedirsquietly(path + "sample/valid/" + c)


def distributeimagesintodirectories(path, imagesdir, samplenum=100, validpc=0.1, trainpc=0.6):
    """
    make the directory structure keras is expecting (including the image class subdirs), then move the images from
    imagesdir into the directory structure that Keras is expecting. some images are *copied* into the sample
    directory, some are *moved* into the valid directory (for fine tuning), some are *moved* into the train
    directory for training, and the rest are moved into the test directory for final testing your network. the images
    copied/moved around are chosen randomly from each class.
    :param path: the root directory where the keras directories will be created and populated (must end with /)
    :param imagesdir: the folder containing images split into class folders
    :param samplenum: the number of images to use for sample (quick dev/testing)
    :param validpc: the %age of images from each class to use for validation (fine tuning)
    :param trainpc: the %age of images from each class to use for training
    :return:
    """
    classes = os.listdir(imagesdir)
    makekerasdirectories(path, classes)
    for c in classes[:]:
        classdir = imagesdir + "/" + c
        images = os.listdir(classdir)
        shuffle(images)
        numimages = len(images)
        print str(numimages) + " in " + c
        numsamplevalid = int(samplenum * validpc)
        print "  0:" + str(numsamplevalid) + " for sample/valid"
        for img in images[:numsamplevalid]:
            copyfile(classdir + "/" + img, path + "sample/valid/" + c + "/" + img)
        print "  " + str(numsamplevalid) + ":" + str(samplenum) + " for sample/train"
        for img in images[numsamplevalid:samplenum]:
            copyfile(classdir + "/" + img, path + "sample/train/" + c + "/" + img)
        numvalid = int(numimages * validpc)
        print "  0:" + str(numvalid) + " for /valid"
        for img in images[:numvalid]:
            os.rename(classdir + "/" + img, path + "valid/" + c + "/" + img)
        numtrain = int(numimages * trainpc)
        print "  " + str(numvalid) + ":" + str(numtrain + numvalid) + " for /train"
        for img in images[numvalid:numtrain + numvalid]:
            os.rename(classdir + "/" + img, path + "train/" + c + "/" + img)
        print "  " + str(numtrain + numvalid) + ": for /test"
        for img in images[numtrain + numvalid:]:
            try:
                os.rename(classdir + "/" + img, path + "test/" + img)
            except OSError: # exists - which is ok to ignore as some images may appear in multiple classes
                pass


# base directory into which we'll be working
base = "data/lemonsorlimes/"

# each imagenet synset is a url that returns a text document with a list of urls
# so here we're downloading classes of images
downloadfromimagenet(base + "raw/lemons/", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07749582")
downloadfromimagenet(base + "raw/limes/", "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07749731")

# and now we setup the directory structure Keras wants and we distribute our classes of images across those dirs
distributeimagesintodirectories(base, base + "raw")

# and now delete our raw directory as the images have been moved
rmtree(base + "raw")
