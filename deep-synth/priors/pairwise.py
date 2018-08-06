#!/usr/bin/env python3

import argparse
import os
import utils
from math_utils import *
from sklearn import mixture

from priors.observations import RelativeObservationsDatabase, RelativeObservation, ObservationCategory


class PairwiseArrangementPrior:
    """Encapsulates arrangement priors for a specific category key (room_type, obj_category, ref_obj_category)"""
    def __init__(self, category_key):
        self._category_key = category_key
        self._records = []
        self._gmm_centroid = None  # GMM for offset to centroid of reference object
        self._gmm_closest = None  # GMM for offset to closest point on reference object
        self._hist_orientation = None  # histogram distribution for orientation of object front in reference frame

    @property
    def category_key(self):
        return self._category_key

    @property
    def centroid(self):
        return self._gmm_centroid

    @property
    def closest(self):
        return self._gmm_closest

    @property
    def orientation(self):
        return self._hist_orientation
        # return self._vmf
        # return self._vmf_orientation

    def add_observations(self, observations, parameterization_scheme):
        for o in observations:
            self._records.append(o.parameterize(scheme=parameterization_scheme))

    @property
    def num_observations(self):
        return len(self._records)

    def fit_model(self, gmm_opts=None):
        if gmm_opts is None:
            gmm_opts = {'n_components': 4, 'covariance_type': 'tied', 'n_init': 5, 'init_params': 'random'}
        X = np.stack(self._records, axis=0)
        try:
            self._gmm_centroid = mixture.BayesianGaussianMixture(**gmm_opts).fit(X[:, 0:2])
            self._gmm_closest = mixture.BayesianGaussianMixture(**gmm_opts).fit(X[:, 2:4])
            angles = X[:, 4][:, np.newaxis]
            # avoid dirac delta-like peaks that cause fit issues by adding noise
            # angles = np.clip(np.random.normal(angles, scale=math.pi/16), a_min=-np.pi, a_max=np.pi)
            low = -math.pi * 33. / 32.
            high = math.pi * 33. / 32.
            hist = np.histogram(angles, bins=np.linspace(low, high, num=17))
            h = hist[0] + 1  # add one (i.e. laplace) smoothing on angle histogram
            h = h / np.sum(h)
            self._hist_orientation = scipy.stats.rv_histogram((h, hist[1]))
            # kappa, mu, scale = vonmises.fit(angles, fscale=1)
            # self._vmf = {'kappa': kappa, 'mu': mu, 'scale': scale}
            # cos_sin_angles = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
            # self._vmf_orientation = VonMisesFisherMixture(**vmf_opts).fit(cos_sin_angles)
            # print(self._vmf_orientation.cluster_centers_, self._vmf_orientation.concentrations_)
            print(f'GMM+VMF fit obtained for {self._category_key} with {len(self._records)} observations')
        except Exception:
            print(f'Error fitting priors for {self.category_key} with {len(self._records)} samples.')

    def plot_model(self):
        X = np.stack(self._records, axis=0)
        plot_gmm_fit(X[:, 0:2], self._gmm_centroid.predict(X[:, 0:2]), self._gmm_centroid.means_,
                     self._gmm_centroid.covariances_, 0, str(self._category_key) + ':centroid')
        plot_gmm_fit(X[:, 2:4], self._gmm_closest.predict(X[:, 2:4]), self._gmm_closest.means_,
                     self._gmm_closest.covariances_, 1, str(self._category_key) + ':closest')
        ang = X[:, 4].reshape(-1, 1)
        plot_hist_fit(ang, self._hist_orientation, 2, title=str(self._category_key) + ':orientation')
        # plot_vmf_fit(ang, mu=self._vmf['mu'], kappa=self._vmf['kappa'], scale=self._vmf['scale'],
        #              index=2, title=str(self._category_key) + ':orientation')
        # samples = self.sample(n_samples=1000)
        # # print(samples)
        # # print(prior.log_prob(samples))
        # c = samples[:, 0:2]
        # plot_gmm_fit(c, prior.centroid.predict(c), prior.centroid.means_, prior.centroid.covariances_, 2,
        #              str(prior.category_key) + ':samples')
        plt.show()

    def log_prob(self, x):
        lp_centroid = self._gmm_centroid.score_samples(x[:, 0:2])[:, np.newaxis]
        lp_closest = self._gmm_closest.score_samples(x[:, 2:4])[:, np.newaxis]
        angles = x[:, 4][:, np.newaxis]
        lp_orientation = self._hist_orientation.logpdf(angles)
        # lp_vmf = vonmises.logpdf(angles, kappa=self._vmf['kappa'], loc=self._vmf['mu'], scale=self._vmf['scale'])
        # cos_sin_angles = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
        # lp_orientation = self._vmf_orientation.log_likelihood(cos_sin_angles)[0][:, np.newaxis]
        # print(lp_centroid)
        # print(lp_closest)
        # print(np.concatenate((angles, lp_orientation), axis=1))
        # print(lp_vmf)
        return lp_centroid + lp_closest + lp_orientation

    def log_prob_offset(self, x):
        if not self._gmm_centroid:
            print(f'Warning: tried log_prob_offset with no prior for {self.category_key}')
            return math.log(0.5)
        return self._gmm_centroid.score_samples(x[:, 0:2])[:, np.newaxis]

    def log_prob_closest(self, x):
        if not self._gmm_closest:
            print(f'Warning: tried log_prob_closest with no prior for {self.category_key}')
            return math.log(0.5)
        return self._gmm_closest.score_samples(x[:, 2:4])[:, np.newaxis]

    def log_prob_orientation(self, x):
        if not self._hist_orientation:
            print(f'Warning: tried log_prob_orientation with no prior for {self.category_key}')
            return math.log(0.5)
        return self._hist_orientation.logpdf(x[:, 4][:, np.newaxis])

    def sample(self, n_samples):
        # self._vmf_orientation.sample(n_samples=n_samples)
        centroid_samples = self._gmm_centroid.sample(n_samples=n_samples)[0]
        closest_samples = self._gmm_closest.sample(n_samples=n_samples)[0]
        orientation_samples = self._hist_orientation.rvs(size=(n_samples, 1))
        # orientation_samples = np.random.vonmises(self._vmf['mu'], self._vmf['kappa'], size=(n_samples, 1))
        # print(np.mean(orientation_samples))
        return np.concatenate((centroid_samples, closest_samples, orientation_samples), axis=1)

    def trim(self):
        self._records = [None] * len(self._records)  # NOTE force garbage collection of records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute arrangement priors')
    parser.add_argument('--task', type=str, required=True, help='<Required> task [fit|load]')
    parser.add_argument('--priors_dir', type=str, required=True, help='directory to save priors')
    parser.add_argument('--room_type', type=str, help='room type to filter priors')
    parser.add_argument('--min_num_observations', type=int, default=100, help='ignore priors with < this observations')
    args = parser.parse_args()

    rod = RelativeObservationsDatabase(name='suncg_priors', priors_dir=args.priors_dir, verbose=True)

    if args.task == 'fit':
        rod.load()
        priors = {}
        for obs_category in rod.grouped_observations:
            if args.room_type and obs_category.room_types != args.room_type:
                continue
            pap = PairwiseArrangementPrior(obs_category)
            observations = rod.grouped_observations[obs_category]
            if len(observations) > args.min_num_observations:
                pap.add_observations(observations, parameterization_scheme='offsets_angles')
                pap.fit_model()
                priors[obs_category] = pap
        filename = os.path.join(args.priors_dir, f'priors.pkl.gz')
        utils.pickle_dump_compressed(priors, filename)

    if args.task == 'load':
        filename = os.path.join(args.priors_dir, f'priors.pkl.gz')
        priors = utils.pickle_load_compressed(filename)
        for (cat_key, prior) in priors.items():
            print(f'Loaded PairwiseArrangementPrior for {cat_key} with {prior.num_observations} observations')
            if prior.num_observations > args.min_num_observations:
                print(f'centroid: mu={prior.centroid.means_}, w={prior.centroid.weights_}')
                print(f'closest: mu={prior.closest.means_}, w={prior.closest.weights_}')
                # print(f'orientation: mu={prior.orientation["mu"]}, kappa={prior.orientation["kappa"]}')
                # samples = prior.sample(n_samples=1000)
                # lp = prior.log_prob(samples)
                # i_max = np.argmax(lp, axis=0)
                # print(f'max sample={samples[i_max, :]} lp={lp[i_max, :]}')
                # x = np.stack(prior._records, axis=0)
                # lp_x = prior.log_prob(x)
                # i_max_x = np.argmax(lp_x, axis=0)
                # print(f'max observ={x[i_max_x, :]} lp={lp_x[i_max_x, :]}')
                if cat_key.obj_category == 'office_chair' and cat_key.ref_obj_category == 'desk':
                    prior.plot_model()

    print('DONE')
