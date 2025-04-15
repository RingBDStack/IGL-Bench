import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

tfd = tfp.distributions


def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))

class gaussianMixtureModel:

    def __init__(self, mixture_components, latent_size):
        self.mixture_components = mixture_components
        self.latent_size = latent_size

    def make_mixture_posterior(self, feats):
        return tfd.MultivariateNormalDiag(
            loc=feats,
            scale_diag=tf.nn.softplus(feats + _softplus_inverse(1.0)),
            name="unlabeled_dist"
        )

    def make_mixture_prior(self):
        """Creates the mixture of Gaussians prior distribution.
        Returns:
            A `tfd.Distribution` instance representing the prior over latent encodings.
        """
        if self.mixture_components == 1:
            # Use fixed standard Gaussian
            return tfd.MultivariateNormalDiag(
                loc=tf.zeros([self.latent_size]),
                scale_diag=tf.ones([self.latent_size]),  # <- replaced scale_identity_multiplier
                name="labeled_dist"
            )

        # Learnable mixture parameters
        loc = tf.compat.v1.get_variable(
            name="loc", shape=[self.mixture_components, self.latent_size])
        raw_scale_diag = tf.compat.v1.get_variable(
            name="raw_scale_diag", shape=[self.mixture_components, self.latent_size])
        mixture_logits = tf.compat.v1.get_variable(
            name="mixture_logits", shape=[self.mixture_components])

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=mixture_logits),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=tf.nn.softplus(raw_scale_diag)
            ),
            name="labeled_dist"
        )
        




