#This is the code for hw4 Question 1

import matplotlib.pyplot as plt 
import numpy as np
import imageio
from sklearn import cluster, mixture

class image:
    def __init__(self, picture):
        picture_shape = picture.shape
        picture = picture/255
        n_pixel = picture_shape[0] * picture_shape[1]
        feature_vector = np.zeros((n_pixel,5))
        for i in range(picture_shape[0]):
            feature_vector[0 + i*picture_shape[1]:picture_shape[1]+i*picture_shape[1],0] = np.arange(0,1,1/picture_shape[1])
            feature_vector[0 + i*picture_shape[1]:picture_shape[1]+i*picture_shape[1],1] = (i/picture_shape[0])* np.ones(picture_shape[1])
        feature_vector[:, 2] = np.reshape(picture[:, :, 0], (n_pixel, 1)).squeeze()
        feature_vector[:, 3] = np.reshape(picture[:, :, 1], (n_pixel, 1)).squeeze()
        feature_vector[:, 4] = np.reshape(picture[:, :, 2], (n_pixel, 1)).squeeze()
        self.feature_vector = feature_vector
        self.picture = picture
    
    def k_mean_clustering(self, k):
        k_means = cluster.KMeans(n_clusters=k, n_init=4)
        k_means.fit(self.feature_vector)
        # values = k_means.cluster_centers_.squeeze()
        values = np.array([[255,0,0],[0,255,0],[0,0,255],[100,100,0],[0,100,100]])
        labels = k_means.labels_
        # r = np.choose(labels, values[:, 2])
        # g = np.choose(labels, values[:, 3])
        # b = np.choose(labels, values[:, 4])
        r = np.choose(labels, values[0:k, 0])
        g = np.choose(labels, values[0:k, 1])
        b = np.choose(labels, values[0:k, 2])
        r.shape = self.picture[:,:,0].shape
        g.shape = self.picture[:,:,0].shape
        b.shape = self.picture[:,:,0].shape
        picture_out = np.dstack((r,g,b))
        return picture_out
    
    def GMM_clustering(self,k):
        GMM = mixture.GaussianMixture(n_components=k, n_init=4)
        GMM.fit(self.feature_vector)
        # values = GMM.means_.squeeze()
        values = np.array([[255,0,0],[0,255,0],[0,0,255],[100,100,0],[0,100,100]])
        labels = GMM.predict(self.feature_vector)
        # r = np.choose(labels, values[:, 2])
        # g = np.choose(labels, values[:, 3])
        # b = np.choose(labels, values[:, 4])
        r = np.choose(labels, values[0:k, 0])
        g = np.choose(labels, values[0:k, 1])
        b = np.choose(labels, values[0:k, 2])
        r.shape = self.picture[:,:,0].shape
        g.shape = self.picture[:,:,0].shape
        b.shape = self.picture[:,:,0].shape
        picture_out = np.dstack((r,g,b))
        return picture_out
        


if __name__ == "__main__":
    # load the picture
    bird = imageio.imread('./image/Bird.jpg')
    plane = imageio.imread('./image/Plane.jpg')

    image_bird = image(bird)
    image_plane = image(plane)

    # Bird k-mean
    plt.figure('Bird K-means')
    for k in range(2,6):
        plt.subplot(2,2,k-1)
        image_bire_out = image_bird.k_mean_clustering(k)
        plt.imshow(image_bire_out)
        plt.title('K-means: K = %d'%k)

    # Bird GMM
    plt.figure('Bird GMM')
    for k in range(2,6):
        plt.subplot(2,2,k-1)
        image_bire_out = image_bird.GMM_clustering(k)
        plt.imshow(image_bire_out)
        plt.title('GMM: K = %d'%k)


    # Plane k-mean
    plt.figure('Plane K-means')
    for k in range(2,6):
        plt.subplot(2,2,k-1)
        image_plane_out = image_plane.k_mean_clustering(k)
        plt.imshow(image_plane_out)
        plt.title('K-means: K = %d'%k)
    
    # Plane GMM
    plt.figure('Plane GMM')
    for k in range(2,6):
        plt.subplot(2,2,k-1)
        image_plane_out = image_plane.GMM_clustering(k)
        plt.imshow(image_plane_out)
        plt.title('GMM: K = %d'%k)
    
    plt.show()



    # plt.imshow(bird)
    # plt.show()

