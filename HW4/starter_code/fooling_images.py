import pickle
import matplotlib.pyplot as plt
from softmax import *

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict

def gradient_ascent(model, target_class, init, learning_rate=1e-3):
    """
    Inputs:
    - model: Image classifier.
    - target_class: Integer, representing the target class the fooling image
      to be classified as.
    - init: Array, shape (1, Din), initial value of the fooling image.
    - learning_rate: A scalar for initial learning rate.
    
    Outputs:
    - image: Array, shape (1, Din), fooling images classified as target_class
      by model
    """
    img = init.copy().astype(float)
    img /= 255
    y = np.array([target_class])

    for i in range(100):
        dx = model.forwards_backwards(img, y, True)
        img += (dx * learning_rate)
        if model.forwards_backwards(img).argmax(axis=1) == y: 
            img *= 255
            img = img.astype(int)
            return img 
    else:
        return None

def img_reshape(flat_img):
    # Use this function to reshape a CIFAR 10 image into the shape 32x32x3, 
    # this should be done when you want to show and save your image.
    return np.moveaxis(flat_img.reshape(3,32,32),0,-1)
    
    
def main():
    # Initialize your own model
    model = SoftmaxClassifier(hidden_dim=1000, reg=0.1)
    ###########################################################################
    # TODO: load your trained model, correctly classified image and set your  #
    # hyperparameters, choose a different label as your target class          #
    ###########################################################################  
    b1 = unpickle("cifar-10-batches-py/data_batch_1")
    img = b1['data'][0].reshape([1,3072])
    target_class = b1['labels'][1]
    model.load("Model_2layer")
    f_img = gradient_ascent(model, target_class, init=img, learning_rate=1) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    

    
    ###########################################################################
    # TODO: compute the (magnified) difference of your original image and the #
    # fooling image, save all three images for your report                    #
    ###########################################################################
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Show initial picture
    plt.imshow(img_reshape(img))
    plt.savefig("initial.jpg", bbox_inches='tight')
    if f_img is not None:
        # Show fooled image
        plt.imshow(img_reshape(f_img))
        plt.savefig("fool.jpg", bbox_inches='tight')

        # Calculate, show, and save difference
        difference = (img - f_img).astype(int)
        plt.imshow(img_reshape(np.abs(difference)))
        plt.savefig("dif.jpg", bbox_inches='tight')
    else:
        print("Didn't find fooling image")
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


if __name__ == "__main__":
    main()
