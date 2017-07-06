
import time

import numpy as np

import theano



def time_call(fn):

    theano.sandbox.cuda.synchronize()

    t1 = time.time()

    fn()

    theano.sandbox.cuda.synchronize()

    return time.time() - t1



def benchmark(fn):

    return min(time_call(fn) for _ in range(5))



def main():

    try:

        if not theano.sandbox.cuda.dnn.dnn_available():

            print ("cuDNN not available. We got:")

            print (theano.sandbox.cuda.dnn.dnn_available.msg)

            return

    except NameError:

        print ("This requires the latest Theano version from github.")

        return



    # full convolution, forward pass

    image_shape = (64, 32, 108, 75)

    kernel_shape = (1, 32, 8, 6)

    output_shape = (64, 1, 115, 80)

    image = theano.shared(np.random.randn(*image_shape).astype(np.float32))

    kernel = theano.shared(np.random.randn(*kernel_shape).astype(np.float32))

    fwd_conv = theano.sandbox.cuda.dnn.dnn_conv(image, kernel,

                                                'full', (1 ,1), 'conv')



    # same result computed as backward pass of valid convolution

    output = theano.shared(np.zeros(output_shape, dtype=np.float32))

    bwd_conv = theano.grad(None, wrt=output,

                           known_grads={theano.sandbox.cuda.dnn.dnn_conv(output,

                                                                         kernel.dimshuffle(1, 0, 2, 3), 'valid', (1 ,1), 'cross'): image})



    # compile both

    mode = theano.compile.get_default_mode().including('gpu')

    fn_fwd_conv = theano.function([], fwd_conv, mode=mode)

    fn_bwd_conv = theano.function([], bwd_conv, mode=mode)



    # compare results

    res_fwd = np.array(fn_fwd_conv())

    res_bwd = np.array(fn_bwd_conv())

    print (res_fwd[0 ,0 ,:4 ,:4])

    print (res_bwd[0 ,0 ,:4 ,:4])

    print ("Same results?", np.allclose(res_fwd, res_bwd, atol=4e-4))



    # compare execution times

    print ("fwd_conv takes %.5f sec" % benchmark(fn_fwd_conv))

    print ("bwd_conv takes %.5f sec" % benchmark(fn_bwd_conv))




main()