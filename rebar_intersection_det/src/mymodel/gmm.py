from cmath import nan
import numpy as np
import itertools
from scipy.stats import multivariate_normal
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from collections import deque
import torch
import time
import matplotlib.patches as patches
from src.realsense.camera import project_pixel2pcd

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"


class GMM:
    def __init__(self, n_components=10, max_iter=30):
        if n_components % 2 == 1:
            n_components += 1

        self.kc = n_components // 2  # number of constrained components
        self.kt = n_components - self.kc  # number of traditional components
        self.k = n_components
        self.max_iter = int(max_iter)
        self.kmeans = KMeans(init="k-means++", n_clusters=self.k, n_init=3)
        self.filter_flag = False
        self.merge_flag = False

    def initialize(self, X):
        self.shape = X.shape

        self.n, self.m = self.shape
        self.kmeans.fit(X)

        self.phi = np.full(shape=self.kc + self.kt, fill_value=1 / (self.kc + self.kt))
        self.weights = np.full(shape=self.shape, fill_value=1 / (self.kc + self.kt))

        # random_row = np.random.randint(low=0, high=self.n, size=self.kc + self.kt)
        # self.mu = [ X[row_index,:] for row_index in random_row ]
        self.mu = np.vstack((self.kmeans.cluster_centers_, self.kmeans.cluster_centers_))
        # self.sigma = [ np.cov(X.T) for _ in range(self.kc + self.kt) ]
        self.sigma = []
        for i in range(self.kc):
            x = X[self.kmeans.labels_ == i, :]
        self.sigma.append(np.cov(x.T))
        for i in range(self.kt):
            self.sigma.append(self.sigma[i])

    def e_step(self, X):
        self.weights = self.predict_proba(X)

        self.phi = self.weights.mean(axis=0)

    def m_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]

            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T,
                                   aweights=(weight / total_weight).flatten(),
                                   bias=True)

    def fit(self, X):
        self.initialize(X)

        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

    def predict_proba(self, X):
        likelihood = np.zeros((self.n, self.k))

        for i in range(self.k):
            if i < self.kc:  # constrained GMM
                sigma_i = self.sigma[i]
                u, s, vh = np.linalg.svd(sigma_i, full_matrices=True)
                if s[0] < s[1] * 4:
                    s[1] = s[0] / 4
                    sigma_i = (u * s) @ vh
                else:  # traditional GMM
                    sigma_i = self.sigma[i]
                    u, s, vh = np.linalg.svd(sigma_i, full_matrices=True)
                    if s[0] > s[1] * 2:
                        s[0] = s[0] / 2
                        sigma_i = (u * s) @ vh
                distribution = multivariate_normal(mean=self.mu[i], cov=sigma_i, allow_singular=True)
                likelihood[:, i] = distribution.pdf(X)

        total_likelihood = likelihood * self.phi
        weights = total_likelihood / total_likelihood.sum(axis=1)[:, np.newaxis]
        return weights

    def predict(self, X):
        weights = self.predict_proba(X)

        return np.argmax(weights, axis=1)

    def bic(self):
        nn = self.n

        kk = 2 * self.k
        sigma_e = 0
        for i in range(self.k):
            sigma_e += np.sum(self.sigma[i])
        return nn * np.log(sigma_e) + kk * np.log(nn)

    def filter_GMM(self, X, filter_ratio=4):  # filter out fat
        self.filter_flag = True

        Y_ = self.predict(X)
        covariances = self.sigma
        means = self.mu
        self.filter_list = []
        u_max = np.max(X[:, 0]);
        u_min = np.min(X[:, 0])
        v_max = np.max(X[:, 1]);
        v_min = np.min(X[:, 1])
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            if np.abs(v[1] / v[0]) < filter_ratio:
                continue
            if np.sum(Y_ == i) < 1 / self.k * len(Y_) * 0.3:
                continue
            # if (mean[0]< (u_max-u_min)*0.15+u_min) or (mean[0]> u_max - 0.15*(u_max-u_min)):
            # continue
            # if (mean[1]< (v_max-v_min)*0.1+v_min) or (mean[1]> v_max - 0.1*(v_max-v_min)):
            # continue

            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            self.filter_list.append((i, mean, covar, angle, u, v))

    def merge_GMM(self, X):
        self.merge_list = []

        Y_ = self.predict(X)
        if self.filter_flag:
            class_indices = [x[0] for x in self.filter_list]
            self.merge_list = []
            visited = set()
            for i in range(len(class_indices)):
                if i in visited:
                    continue
                class_idx_i, mean_i, covar_i, angle_i, u_i, v_i = self.filter_list[i]
                visited.add(i)
                c_i = -u_i[0] * mean_i[0] - u_i[1] * mean_i[1]
                for j in range(i + 1, len(class_indices)):
                    if j in visited:
                        continue
                    class_idx_j, mean_j, covar_j, angle_j, u_j, v_j = self.filter_list[j]
                    delta_angle = min(abs(angle_i - angle_j), 180 - abs(angle_i - angle_j))
                    dist = np.abs(u_i[0] * mean_j[0] + u_i[1] * mean_j[1] + c_i) / np.linalg.norm(u_i)
                    if (delta_angle < 10) and (dist < 10):  # and abs(u_i @ u_j)>0.8:
                        Y_[Y_ == class_idx_j] = class_idx_i
                        Xi = X[Y_ == class_idx_i, :]
                        mean_i = np.mean(Xi, axis=0)
                        covar_i = np.cov(Xi.T)
                        v_i, w_i = linalg.eigh(covar_i)
                        v_i = 2.0 * np.sqrt(2.0) * np.sqrt(v_i)
                        u_i = w_i[0] / linalg.norm(w_i[0])
                        angle_i = np.arctan(u_i[1] / u_i[0])
                        angle_i = 180.0 * angle_i / np.pi  # convert to degrees
                        visited.add(j)

                self.merge_list.append((class_idx_i, mean_i, covar_i, angle_i, u_i, v_i))

        else:
            print("filter before merging GMMs")
            raise

    def plot_results(self, X, gmm_list=None, ax=None):
        color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange",
                                      "mediumspringgreen", "plum", "orangered", "green"])
        # Y_ = self.predict(X)
        # covariances = self.sigma
        # means = self.mu
        if gmm_list == None:
            gmm_list = self.merge_list
        elif gmm_list == 'merge_list':
            gmm_list = self.merge_list
        elif gmm_list == 'filter_list':
            gmm_list = self.filter_list
        elif gmm_list == 'original':
            self.filter_GMM(X, filter_ratio=0)
            gmm_list = self.filter_list

        if ax == None:
            fig, axes = plt.subplots(1)
        else:
            axes = ax

        for i, ((class_idx, mean, covar, angle, u, v), color) in enumerate(zip(gmm_list, color_iter)):
        # plt.scatter(X[Y_==i,0],X[Y_==i,1],0.8,color=color)
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(axes.bbox)
            ell.set_alpha(1)
            axes.add_artist(ell)
            axes.axis('equal')
            axes.set_xlim(X[:, 0].min(), X[:, 0].max())
            axes.set_ylim(X[:, 1].min(), X[:, 1].max())
            axes.invert_yaxis()
            axes.set_xlim([0, 1280])
            axes.set_ylim([0, 720])


class GMM_torch:
    def __init__(self, n_components=20, max_iter=30, init_iter=10):

        # if n_components%2 ==1:
        # n_components +=1
        # self.kc = n_components//2 # number of constrained components
        # self.kt = n_components - self.kc # number of traditional components
        self.K = n_components
        self.max_iter = int(max_iter)
        self.init_iter = init_iter
        # self.filter_flag = False
        # self.merge_flag = False
        self.fail_count = 0

    def initialize(self, X, init_mu=None, init_sigma=None, init_iter=10):
        _, self.N, self.D = X.shape

        device = X.device
        if init_mu != None and init_sigma != None:
            self.phi = torch.full([self.K], fill_value=1 / (self.K)).to(device)
            self.weights = torch.full((self.K, self.N), fill_value=1 / (self.K)).to(device)
            self.mu = init_mu
            self.sigma = init_sigma
            return

        kmeans_clf, kmeans_centroids = kmeans(X, K=self.K, Niter=init_iter)
        self.kmeans_clf_save = kmeans_clf
        self.kmeans_centroids_save = kmeans_centroids

        if X.is_cuda:
            self.phi = torch.full([self.K], fill_value=1 / (self.K)).cuda()
            self.weights = torch.full((self.K, self.N), fill_value=1 / (self.K)).cuda()
            self.mu = kmeans_centroids.unsqueeze(1)
            self.sigma = torch.zeros(self.K, self.D, self.D).to(torch.double).cuda()
        else:
            self.phi = torch.full([self.K], fill_value=1 / (self.K))
            self.weights = torch.full((self.K, self.N), fill_value=1 / (self.K))
            self.mu = kmeans_centroids.unsqueeze(1)
            self.sigma = torch.zeros(self.K, self.D, self.D).to(torch.double)
        for i in range(self.K):
            x = X[0, i == kmeans_clf, :]
            self.sigma[i, :, :] = torch.cov(x.T)

    def e_step(self, X):
        self.weights = self.predict_proba(X)

        if torch.any(torch.isnan(self.weights)):
            print("found nan in weights")
            raise
        self.phi = self.weights.mean(dim=1)
        if torch.any(torch.isnan(self.phi)):
            print("found nan in phi")
            raise

    # self.phi[torch.isnan(self.phi)]=0.0
    def m_step(self, X):
        # print(torch.any(torch.isnan(self.weights)))
        self.mu = (X * self.weights.unsqueeze(2)).sum(dim=1).unsqueeze(1) / self.weights.sum(dim=1).unsqueeze(1).unsqueeze(2)  # K1D
        self.mu[torch.isnan(self.mu) | torch.isinf(self.mu)] = 0
        if torch.any(torch.isnan(self.mu)):
            print("found nan in mu")
            raise
        self.sigma = bcov(X, aweights=self.weights / self.weights.sum(0))
        self.sigma[torch.isnan(self.sigma) | torch.isinf(self.sigma)] = 0
        if torch.any(torch.isnan(self.sigma)):
            print("found nan in sigma")
            raise

    def fit(self, X, init_mu=None, init_sigma=None):
        self.initialize(X, init_mu=init_mu, init_sigma=init_sigma, init_iter=self.init_iter)
        for iteration in range(self.max_iter):
        # print(iteration)
            self.e_step(X)
            self.m_step(X)

    # print(f'iteration = {iteration}')
    def predict_proba(self, X):
        x = (X - self.mu).unsqueeze(3)
        numerator = torch.exp(-0.5 * (x * torch.transpose(x, 2, 3) * torch.linalg.pinv(self.sigma).unsqueeze(1)).sum((2, 3)))
        numerator[torch.isnan(numerator)] = 0
        self.numerator_save = numerator
        likelihood = numerator / torch.sqrt((2 * np.pi) ** self.D * torch.linalg.det(self.sigma)).unsqueeze(1)
        likelihood[torch.isinf(likelihood) | torch.isnan(likelihood)] = 0
        self.likelihood_save = likelihood
        total_likelihood = likelihood * self.phi.unsqueeze(1)
        weights = total_likelihood / total_likelihood.sum(dim=0).unsqueeze(0)
        weights[torch.isinf(weights) | torch.isnan(weights)] = 0
        return weights

    def predict(self, X):
        weights = self.predict_proba(X)
        return torch.argmax(weights, axis=0)

    def bic(self):
        raise NotImplementedError

    def filter_GMM(self, X, filter_ratio=4):  # filter out fat
        self.filter_flag = True
        Y_ = self.predict(X).cpu()
        covariances = self.sigma.cpu()
        means = self.mu.squeeze().cpu()
        self.filter_list = []

        vs, ws = torch.linalg.eigh(covariances)  # eigenvalues, eigenvectors
        vs = vs.cpu();
        ws = ws.cpu()
        # phi_max = self.phi.max()
        for i, (mean, v, w) in enumerate(zip(means, vs, ws)):
            v = 2.0 * np.sqrt(2.0) * torch.sqrt(v)
            u = w[0] / torch.linalg.norm(w[0])

            # if self.phi[i] < 0.15*phi_max:
            # continue
            if torch.abs(v[1] / v[0]) < filter_ratio or v[1] < 50:
                continue
            if torch.sum(Y_ == i) < 1 / self.K * len(Y_) * 0.3:
                continue
            angle = torch.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            self.filter_list.append((i, mean, covariances[i, :, :], angle, u, v))

    def merge_GMM(self, X):
        self.merge_list = []

        Y_ = self.predict(X).cpu()
        if self.filter_flag:
            class_indices = [x[0] for x in self.filter_list]
            self.merge_list = []
            visited = set()
            for i in range(len(class_indices)):
                if i in visited:
                    continue
                class_idx_i, mean_i, covar_i, angle_i, u_i, v_i = self.filter_list[i]
                visited.add(i)
                c_i = -u_i[0] * mean_i[0] - u_i[1] * mean_i[1]
                for j in range(i + 1, len(class_indices)):
                    if j in visited:
                        continue
                    class_idx_j, mean_j, covar_j, angle_j, u_j, v_j = self.filter_list[j]
                    delta_angle = min(abs(angle_i - angle_j), 180 - abs(angle_i - angle_j))
                    dist = torch.abs(u_i[0] * mean_j[0] + u_i[1] * mean_j[1] + c_i) / torch.linalg.norm(u_i)
                    if (delta_angle < 10) and (dist < 10):  # and abs(u_i @ u_j)>0.8:
                        Y_[Y_ == class_idx_j] = class_idx_i
                        X_i = X[0, Y_ == class_idx_i, :].cpu()
                        mean_i = torch.mean(X_i, axis=0)
                        covar_i = torch.cov(X_i.T)
                        v_i, w_i = torch.linalg.eigh(covar_i)
                        v_i = 2.0 * np.sqrt(2.0) * torch.sqrt(v_i)
                        u_i = w_i[0] / torch.linalg.norm(w_i[0])
                        angle_i = torch.arctan(u_i[1] / u_i[0])
                        angle_i = 180.0 * angle_i / np.pi  # convert to degrees
                        visited.add(j)

                self.merge_list.append((class_idx_i, mean_i, covar_i, angle_i, u_i, v_i))
        else:
            print("filter before merging GMMs")
            raise

    def plot_results(self, X, gmm_list=None, ax=None):
        color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange",
                                      "mediumspringgreen", "plum", "orangered", "green"])

        if gmm_list == None or gmm_list == 'merge_list':
            gmm_list = self.merge_list
        elif gmm_list == 'filter_list':
            gmm_list = self.filter_list
        elif gmm_list == 'original':
            self.filter_GMM(X, filter_ratio=0)
            gmm_list = self.filter_list
        if ax == None:
            fig, axes = plt.subplots(1)
        else:
            axes = ax

        for i, ((class_idx, mean, cov, angle, u, v), color) in enumerate(zip(gmm_list, color_iter)):
            # plt.scatter(X[Y_==i,0],X[Y_==i,1],0.8,color=color)
            ell = mpl.patches.Ellipse(mean.numpy(), v[0].numpy(), v[1].numpy(), 180.0 + angle.numpy(), color=color)
            ell.set_clip_box(axes.bbox)
            ell.set_alpha(0.3)
            axes.add_artist(ell)
            axes.axis('equal')
            # axes.set_xlim(X[:,0].min(),X[:,0].max())
            # axes.set_ylim(X[:,1].min(),X[:,1].max())
            axes.invert_yaxis()
            axes.set_xlim([0, 1280])
            axes.set_ylim([0, 720])

def kmeans(X, K=15, Niter=10):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    _, N, D = X.shape  # Number of samples, dimension of the ambient space
    x = X[0, :, :]


    c = x[(torch.rand(K) * N).to(torch.long), :]  # Simplistic initialization for the centroids
    c_max = x.max(0);
    c_min = x.min(0)
    c_mean = x.mean(0)
    if K >= 2:
        c[0, :] = c_mean + (c_max.values - c_mean) / 2
        c[1, :] = c_mean + (c_min.values - c_mean) / 2
    if K >= 4:
        c[0, :] = x[c_max.indices[0], :]
        c[1, :] = x[c_max.indices[1], :]
        c[2, :] = x[c_min.indices[0], :]
        c[3, :] = x[c_min.indices[1], :]

    for i in range(D):
        c[:,i] = c[:,i]*(c[:,i].max()-c[:,i].min())-c[:,i].min()

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
        c[torch.isinf(c) | torch.isnan(c)] = x.max() / 2
    return cl, c


def bcov(points, aweights=None):
    K, N, D = points.size()
    if aweights != None:
        K, _ = aweights.size()
    if aweights == None:
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(K * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(K, N, D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    else:
        aweights = aweights * (N - 1)
        mean = (points * aweights.unsqueeze(2)).sum(dim=1).unsqueeze(1) / aweights.sum(dim=1).unsqueeze(1).unsqueeze(2)  # K1D
        diffs = (points - mean).reshape(K * N, D)  #
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(K, N, D, D)
        aweights = aweights.unsqueeze(2).unsqueeze(3)
        bcov = (prods * aweights).sum(dim=1) / (aweights.sum(dim=1) - 1)
    return bcov  # KDD


def compute_intersections(gmm_list):
    lines = []
    for i, (class_idx, mean, covar, angle, u, v) in enumerate(gmm_list):
        p2 = mean - torch.tensor([u[1], -u[0]]) * v[1]
        p3 = mean + torch.tensor([u[1], -u[0]]) * v[1]
        lines.append((i, p2, p3))

    intersections = []
    vectors1 = []
    vectors2 = []
    rebar_id = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            id_i, (x1, y1), (x2, y2) = lines[i]
            id_j, (x3, y3), (x4, y4) = lines[j]
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            thres = 300
            m1 = torch.arctan((y1 - y2) / (x1 - x2))
            m2 = torch.arctan((y3 - y4) / (x3 - x4))
            diff_m = min(abs(m1 - m2), np.pi - abs(m1 - m2))

            if px < max(min(x1, x2), min(x3, x4)) - thres or px > min(max(x1, x2), max(x3, x4)) + thres:
                continue
            # elif diff_m < np.pi/2*0.40: # filter out the non-perpendicular pair
            # continue
            # elif px < color.shape[1] and py < color.shape[0] and int(model.predict(np.array(color[int(py), int(px), :]).reshape(1,3))) == 0: # RGB filtering
            # continue
            else:
            # 2d space
                intersections.append(np.array([px.numpy(), py.numpy()]))
                v1 = torch.tensor([x2 - x1, y2 - y1]);
                v1 = v1 / torch.linalg.norm(v1)
                v2 = torch.tensor([x4 - x3, y4 - y3]);
                v2 = v2 / torch.linalg.norm(v2)
                vectors1.append(v1.numpy());
                vectors2.append(v2.numpy());  # th_vec1_
                rebar_id.append([id_i, id_j])
    return intersections, vectors1, vectors2, rebar_id


def intersection_doubleCheck(intersections, vectors1, vectors2, rebar_id, depth_top, patchsize=(200, 200), axarr=None):
    start_double_check_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_W, patch_H = patchsize
    H, W = depth_top.shape
    intersections_checked = []
    vectors1_checked = []
    vectors2_checked = []
    rebar_id_checked = []
    tolerance_pixel = 10
    for (px, py), (v1x, v1y), (v2x, v2y), id in zip(intersections, vectors1, vectors2, rebar_id):
        if px < patch_W // 2 + tolerance_pixel or px > W - patch_W // 2 - tolerance_pixel:
            continue
        if py < patch_H // 2 + tolerance_pixel or py > H - patch_H // 2 - tolerance_pixel:
            continue
        intersections_checked.append([int(px), int(py)])
        vectors1_checked.append([v1x, v1y])
        vectors2_checked.append([v2x, v2y])
        rebar_id_checked.append(id)
        # second check
        my_gmm = GMM_torch(n_components=4, max_iter=1, init_iter=2)
        intersections_checked2 = []
        vectors1_checked2 = []
        vectors2_checked2 = []
        rebar_id_checked2 = []
        if axarr != None:
            axarr.clear()

        for (px, py), (v1x, v1y), (v2x, v2y), id in zip(intersections_checked, vectors1_checked, vectors2_checked, rebar_id_checked):
            # prepare GMM data in a patch
            depth_patch = depth_top[py - patch_H // 2:py + patch_H // 2, px - patch_W // 2:px + patch_W // 2]
            binary_depth = depth_patch > 0.001
            y_idx = np.tile(np.expand_dims(np.arange(py - patch_H // 2, py + patch_H // 2), 1), (1, patch_W))
            x_idx = np.tile(np.arange(px - patch_W // 2, px + patch_W // 2), (patch_H, 1))
            X = torch.from_numpy(np.c_[x_idx[binary_depth].reshape(-1), y_idx[binary_depth].reshape(-1)]).unsqueeze(0).to(torch.double).to(device)
            if X.shape[1] < 10:
                continue
            # fitting GMM
            my_gmm.fit(X)
            my_gmm.filter_GMM(X, filter_ratio=1.5)
            if axarr != None:
                my_gmm.plot_results(X, gmm_list='filter_list', ax=axarr)
                rect = patches.Rectangle((px - patch_W // 2, py - patch_H // 2), patch_W, patch_H, linewidth=1, edgecolor='r', facecolor='none')
                axarr.add_patch(rect)
            # compute the intersections
            intersections_check, _, _, _ = compute_intersections(my_gmm.filter_list)
            # count the valid intersections
            check_count = 0
            for px_check, py_check in intersections_check:
                if abs(px_check - px) < 40 and abs(py_check - py) < 40:
                    check_count += 1
            if check_count < 2:
                continue
            # save the valid intersections poses
            intersections_checked2.append([px, py])
            vectors1_checked2.append([v1x, v1y])
            vectors2_checked2.append([v2x, v2y])
            rebar_id_checked2.append(id)
            if axarr != None:
                axarr.imshow(depth_top / depth_top.max(), cmap='Greys')
            axarr.invert_yaxis()
            end_double_check_time = time.time()
        return intersections_checked2, vectors1_checked2, vectors2_checked2, rebar_id_checked2


def compute_3d_intersections(depth_top, intersections_checked2, vectors1_checked2, vectors2_checked2, K):
    intersection_3d = []
    vectors1_3d = []
    vectors2_3d = []
    for intersect, v1, v2 in zip(intersections_checked2, vectors1_checked2, vectors2_checked2):
        i_3d, v1_3d = compute_3d_vector(intersect, v1, depth_top, K)
        i_3d, v2_3d = compute_3d_vector(intersect, v2, depth_top, K)
        intersection_3d.append(i_3d)
        vectors1_3d.append(v1_3d)
        vectors2_3d.append(v2_3d)
    return np.array(intersection_3d), np.array(vectors1_3d), np.array(vectors2_3d)


def compute_3d_poses(rebar_id_checked2, X_cpu, D_cpu, K, my_gmm):
    vectors1_3d = []
    vectors2_3d = []
    intersections_3d = []
    rebar_itc1 = []
    rebar_itc2 = []
    # X_cpu = X.cpu().numpy()
    # X_cpu = np.squeeze(X_cpu)
    # D_cpu = depth_top[binary_depth].reshape(-1)
    for id1, id2 in rebar_id_checked2:
        _, mean1, covar1, _, _, _ = my_gmm.merge_list[id1]
        mean_rebar1, nu1 = compute_3d_vector(mean1, covar1, X_cpu, D_cpu, K)
        vectors1_3d.append(nu1[:, -1])

        _, mean2, covar2, _, _, _ = my_gmm.merge_list[id2]
        mean_rebar2, nu2 = compute_3d_vector(mean2, covar2, X_cpu, D_cpu, K)
        vectors2_3d.append(nu2[:, -1])

        # computing poses, reference: https://www.jianshu.com/p/e80a6a461a49
        t1_space = np.dot(np.cross((mean_rebar2 - mean_rebar1), nu2[:, -1]), np.cross(nu1[:, -1], nu2[:, -1]))
        t2_space = np.dot(np.cross((mean_rebar2 - mean_rebar1), nu1[:, -1]), np.cross(nu1[:, -1], nu2[:, -1]))
        rebar_itc1.append(t1_space * nu1[:, -1] + mean_rebar1)
        rebar_itc2.append(t2_space * nu2[:, -1] + mean_rebar2)
        intersections_3d.append((rebar_itc1[-1] + rebar_itc2[-1]) / 2)

    return intersections_3d, rebar_itc1, vectors1_3d, rebar_itc2, vectors2_3d


def compute_3d_vector(mean2, covar2, X_cpu, D_cpu, K):
    s, vh = np.linalg.eigh(covar2)
    p = vh * np.sqrt(0.5) * (1 / np.sqrt(s)) @ vh.T @ (X_cpu.T - np.expand_dims(mean2.cpu().numpy(), axis=1))
    prob = np.sum(p ** 2, axis=0)
    idx_fit = prob < 1
    u_fit = X_cpu[idx_fit, 0]
    v_fit = X_cpu[idx_fit, 1]
    d_fit = D_cpu[idx_fit]
    xyz = project_pixel2pcd(u_fit, v_fit, d_fit, K)
    mean_rebar2 = xyz.mean(0)
    cov_rebar2 = np.cov(xyz.T)
    lamb, nu2 = np.linalg.eigh(cov_rebar2)
    return mean_rebar2, nu2