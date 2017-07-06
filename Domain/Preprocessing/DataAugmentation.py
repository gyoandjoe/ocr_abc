from random import randint, uniform

from skimage import transform, filters, exposure
import numpy as np

from skimage.io import imshow

from skimage import transform, filters

PIXELS = 64
imageSize = PIXELS * PIXELS
num_features = imageSize

# much faster than the standard skimage.transform.warp method
def fast_warp(img, tf, output_shape, mode='constant'):
    return transform._warps_cy._warp_fast(img, tf.params,
                                          output_shape=output_shape, mode=mode)

def batch_iterator(data, y, batchsize): #, model
    '''
    Data augmentation batch iterator for feeding images into CNN.
    rotate all images in a given batch between -10 and 10 degrees
    random translations between -10 and 10 pixels in all directions.
    random zooms between 1 and 1.3.
    random shearing between -25 and 25 degrees.
    randomly applies sobel edge detector to 1/4th of the images in each batch.
    randomly inverts 1/2 of the images in each batch.
    '''

    n_samples = data.shape[0]
    loss = []
    for i in range((n_samples + batchsize -1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.empty(shape = (X_batch.shape[0], 1, PIXELS, PIXELS),
                               dtype = 'float32')

        # random rotations betweein -10 and 10 degrees
        dorotate = randint(-10,10)

        # random translations
        trans_1 = randint(-10,10)
        trans_2 = randint(-10,10)

        # random zooms
        zoom = uniform(1, 1.3)

        # shearing
        shear_deg = uniform(-25, 25)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (trans_1, trans_2))

        tform = tform_center + tform_aug + tform_uncenter

        # images in the batch do the augmentation
        for j in range(X_batch.shape[0]):

            X_batch_aug[j][0] = fast_warp(X_batch[j][0], tform,
                                          output_shape = (PIXELS, PIXELS))

        # use sobel edge detector filter on one quarter of the images
        indices_sobel = np.random.choice(X_batch_aug.shape[0],
                                         X_batch_aug.shape[0] // 4, replace = False)
        for k in indices_sobel:
            img = X_batch_aug[k][0]
            X_batch_aug[k][0] = filters.sobel(img)

        # invert half of the images
        indices_invert = np.random.choice(X_batch_aug.shape[0],
                                          X_batch_aug.shape[0] // 2, replace = False)
        for l in indices_invert:
            img = X_batch_aug[l][0]
            X_batch_aug[l][0] = np.absolute(img - np.amax(img))

        # fit model on each batch
        loss.append((X_batch_aug, y_batch))
        #loss.append(model.train_on_batch(X_batch_aug, y_batch))

    #return np.mean(loss)
    return loss