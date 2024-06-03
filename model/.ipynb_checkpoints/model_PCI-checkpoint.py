import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from .module import (
    CompoundEmbedding, MLP,
    Bernoulli, NegativeBinomial, ZeroInflatedNegativeBinomial
)

from ..utils.math_utils import (
    logprob_normal, kldiv_normal,
    logprob_bernoulli_logits,
    logprob_nb_positive,
    logprob_zinb_positive
)

#####################################################
#                     LOAD MODEL                    #
#####################################################

def load_VCI(args, state_dict=None):
    device = (
        "cuda:" + str(args["gpu"])
            if (not args["cpu"]) 
                and torch.cuda.is_available() 
            else 
        "cpu"
    )

    model = VCI(
        args["num_outcomes"],
        args["num_treatments"],
        args["num_covariates"],
        omega0=args["omega0"],
        omega1=args["omega1"],
        omega2=args["omega2"],
        dist_mode=args["dist_mode"],
        dist_outcomes=args["dist_outcomes"],
        patience=args["patience"],
        device=device,
        hparams=args["hparams"]
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

#####################################################
#                     MAIN MODEL                    #
#####################################################

class VCI(nn.Module):
    def __init__(
        self,
        num_outcomes,
        num_treatments,
        num_covariates,
        embed_outcomes=True,
        embed_treatments=False,
        embed_covariates=True,
        omega0=1.0,
        omega1=2.0,
        omega2=0.1,
        dist_mode="match",
        dist_outcomes="normal",
        type_treatments=None,
        type_covariates=None,
        mc_sample_size=30,
        best_score=-1e3,
        patience=5,
        device="cuda",
        hparams=""
    ):
        super(VCI, self).__init__()
        # generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        self.embed_outcomes = embed_outcomes
        self.embed_treatments = embed_treatments
        self.embed_covariates = embed_covariates
        self.dist_outcomes = dist_outcomes
        self.type_treatments = type_treatments
        self.type_covariates = type_covariates
        self.mc_sample_size = mc_sample_size
        # vci parameters
        self.omega0 = omega0
        self.omega1 = omega1
        self.omega2 = omega2
        self.dist_mode = dist_mode
        # early-stopping
        self.best_score = best_score
        self.patience = patience
        self.patience_trials = 0

        # set hyperparameters
        self._set_hparams_(hparams)

        # individual-specific model
        self._init_indiv_model()

        # covariate-specific model
        self._init_covar_model()

        self.iteration = 0

        self.history = {"epoch": [], "stats_epoch": []}

        self.to_device(device)

    def _set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "latent_dim": 128,
            "ZX_dim": 128,
            "ZT_dim":128,
            "outcome_emb_dim": 256,
            "treatment_emb_dim": 64,
            "covariate_emb_dim": 16,
            "encoder_width": 128,
            "encoder_depth": 3,
            "decoder_width": 128,
            "decoder_depth": 3,
            "discriminator_width": 64,
            "discriminator_depth": 2,
            "autoencoder_lr": 3e-4,
            "discriminator_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "discriminator_wd": 4e-7,
            "discriminator_steps": 3,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                with open(hparams) as f:
                    dictionary = json.load(f)
                self.hparams.update(dictionary)
            else:
                self.hparams.update(hparams)

        self.outcome_dim = (
            self.hparams["outcome_emb_dim"] if self.embed_outcomes else self.num_outcomes)
        self.treatment_dim = (
            self.hparams["treatment_emb_dim"] if self.embed_treatments else self.num_treatments)
        self.covariate_dim = (
            self.hparams["covariate_emb_dim"]*len(self.num_covariates) 
            if self.embed_covariates else sum(self.num_covariates)
        )

        return self.hparams

    def _init_indiv_model(self):
        params = []

        # embeddings
        if self.embed_outcomes:
            self.outcomes_embeddings = self.init_outcome_emb()
            self.outcomes_contr_embeddings = self.init_outcome_emb()
            
            params.extend(list(self.outcomes_embeddings.parameters()))
            params.extend(list(self.outcomes_contr_embeddings.parameters()))

        if self.embed_treatments:
            self.treatments_embeddings = self.init_treatment_emb()
            params.extend(list(self.treatments_embeddings.parameters()))

        if self.embed_covariates:
            self.covariates_embeddings = nn.Sequential(*self.init_covariates_emb())
            for emb in self.covariates_embeddings:
                params.extend(list(emb.parameters()))

        # models
        self.encoder_ZX = self.init_encoder_X()
        self.encoder_ZT = self.init_encoder_T()
        self.control_encoder = self.init_encoder_control()
        params.extend(list(self.encoder_ZX.parameters()))
        params.extend(list(self.encoder_ZT.parameters()))
        params.extend(list(self.control_encoder.parameters()))
        
        ## initialize the prior encoders
        self.encoder_ZX_prior = self.init_encoder_X_prior()
        self.encoder_ZT_prior = self.init_encoder_T_prior()
        self.control_prior = self.init_control_prior()
        params.extend(list(self.encoder_ZX_prior.parameters()))
        params.extend(list(self.encoder_ZT_prior.parameters()))
        params.extend(list(self.control_prior.parameters()))

        ## eval models
        self.encoder_ZX_eval = copy.deepcopy(self.encoder_ZX)
        self.encoder_ZT_eval = copy.deepcopy(self.encoder_ZT)
        self.encoder_ZX_prior_eval = copy.deepcopy(self.encoder_ZX_prior)
        self.encoder_ZT_prior_eval = copy.deepcopy(self.encoder_ZT_prior)
        self.control_encoder_eval = copy.deepcopy(self.control_encoder)
        self.control_prior_eval = copy.deepcopy(self.control_prior)

        # change the decoder to control decoder and decoder of experiments
        # self.decoder = self.init_decoder()
        self.decoder = self.init_decoder_experiments()
        self.control_decoder = self.init_decoder()
        params.extend(list(self.decoder.parameters()))
        params.extend(list(self.control_decoder.parameters()))
        
        ## covariate decoder
        self.cov_decoder = self.init_decoder_cov()
        params.extend(list(self.cov_decoder.parameters()))
                
        ##intervention decoder
        self.interv_decoder = self.init_decoder_interv()
        params.extend(list(self.interv_decoder.parameters()))

        # optimizer
        self.optimizer_autoencoder = torch.optim.Adam(
            params,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        return self.encoder_ZX, self.encoder_ZT,self.decoder, self.control_encoder, self.control_decoder, self.cov_decoder, self.interv_decoder

    def _init_covar_model(self):

        if self.dist_mode == "discriminate":
            params = []

            # embeddings
            if self.embed_outcomes:
                self.adv_outcomes_emb = self.init_outcome_emb()
                params.extend(list(self.adv_outcomes_emb.parameters()))

            if self.embed_treatments:
                self.adv_treatments_emb = self.init_treatment_emb()
                params.extend(list(self.adv_treatments_emb.parameters()))

            if self.embed_covariates:
                self.adv_covariates_emb = nn.Sequential(*self.init_covariates_emb())
                for emb in self.adv_covariates_emb:
                    params.extend(list(emb.parameters()))

            # model
            self.discriminator = self.init_discriminator()
            self.loss_discriminator = nn.BCEWithLogitsLoss()
            params.extend(list(self.discriminator.parameters()))

            self.optimizer_discriminator = torch.optim.Adam(
                params,
                lr=self.hparams["discriminator_lr"],
                weight_decay=self.hparams["discriminator_wd"],
            )
            self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_discriminator, step_size=self.hparams["step_size_lr"]
            )

            return self.discriminator

        elif self.dist_mode == "fit":
            raise NotImplementedError(
                'TODO: implement dist_mode "fit" for distribution loss')

        elif self.dist_mode == "match":
            return None

        else:
            raise ValueError("dist_mode not recognized")

    def encode(self, outcomes, treatments, covariates, eval=False):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)

        if eval:
            return self.encoder_eval(inputs)
        else:
            return self.encoder(inputs)
    
    def encode_control(self, outcomes, covariates, eval=False):
        
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
            
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]

        covariates = torch.cat(covariates, -1)
        inputs = torch.cat([outcomes, covariates], -1)

        if eval:
            return self.control_encoder_eval(inputs)
        else:
            return self.control_encoder(inputs)
        
        
    def encode_ZX(self, outcomes, covariates, eval=False):
        
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_covariates:
            # print("covariates[0] {}".format(covariates[0]))
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
        covariates = torch.cat(covariates, -1)

        inputs = torch.cat([outcomes, covariates], -1)

        if eval:
            return self.encoder_ZX_eval(inputs)
        else:
            return self.encoder_ZX(inputs)
        
    def encode_ZT(self, outcomes, treatments, eval=False):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)
    
        inputs = torch.cat([outcomes, treatments], -1)

        if eval:
            return self.encoder_ZT_eval(inputs)
        else:
            return self.encoder_ZT(inputs)
        
    def encode_ZT_prior(self, treatments, eval=False):
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)
    
        inputs = treatments

        if eval:
            return self.encoder_ZT_prior_eval(inputs)
        else:
            return self.encoder_ZT_prior(inputs)    
        
    
    def encode_ZX_prior(self, covariates, eval=False):
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
        covariates = torch.cat(covariates, -1)
        inputs = covariates

        if eval:
            return self.encoder_ZX_prior_eval(inputs)
        else:
            return self.encoder_ZX_prior(inputs)    
     
    
    def encode_control_prior(self, covariates, eval=False):
        
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
        covariates = torch.cat(covariates, -1)
        inputs = covariates

        if eval:
            return self.control_prior_eval(inputs)
        else:
            return self.control_prior(inputs)    
        
    def decode(self, latents):
        inputs = latents
        return self.decoder(inputs)
    
    def control_decode(self, latents, treatments):
        
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)

        inputs = torch.cat([latents, treatments], -1)

        return self.control_decoder(inputs)
    
    def covariate_decode(self, ZX):
        
        # inputs = torch.cat([ZX, latent_control], -1)
        inputs = ZX
        return self.cov_decoder(inputs)
    
    
    def intervention_decode(self, ZT, latent_control):
        
        inputs = torch.cat([ZT, latent_control], -1)
        return self.interv_decoder(inputs)
        
    def discriminate(self, outcomes, treatments, covariates):
        if self.embed_outcomes:
            outcomes = self.adv_outcomes_emb(outcomes)
        if self.embed_treatments:
            treatments = self.adv_treatments_emb(treatments)
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.adv_covariates_emb)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)

        return self.discriminator(inputs).squeeze()

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param sigma: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def distributionize(self, constructions, dim=None, dist=None, eps=1e-3):
        if dim is None:
            dim = self.num_outcomes
        if dist is None:
            dist = self.dist_outcomes

        if dist == "nb":
            mus = F.softplus(constructions[..., 0]).add(eps)
            thetas = F.softplus(constructions[..., 1]).add(eps)
            dist = NegativeBinomial(
                mu=mus, theta=thetas
            )
        elif dist == "zinb":
            mus = F.softplus(constructions[..., 0]).add(eps)
            thetas = F.softplus(constructions[..., 1]).add(eps)
            zi_logits = constructions[..., 2].add(eps)
            dist = ZeroInflatedNegativeBinomial(
                mu=mus, theta=thetas, zi_logits=zi_logits
            )
        elif dist == "normal":
            locs = constructions[..., 0]
            scales = F.softplus(constructions[..., 1]).add(eps)
            dist = Normal(
                loc=locs, scale=scales
            )
        elif dist == "bernoulli":
            logits = constructions[..., 0]
            dist = Bernoulli(
                logits=logits
            )

        return dist

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor, treatments: torch.Tensor, 
            size=1) -> torch.Tensor:
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        treatments = treatments.repeat(size, 1)

        latents = self.reparameterize(mu, sigma)

        return self.decode(latents, treatments)
    
    
    def sample_latent(self,  mu: torch.Tensor, sigma: torch.Tensor, size=1):
        
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        latents = self.reparameterize(mu, sigma)
        return latents
    
    def sample_control(self, mu: torch.Tensor, sigma: torch.Tensor, treatments: torch.Tensor, 
            size=1) -> torch.Tensor:
        
        # print("mu shape {}".format(mu.shape))
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        # print("repeated mu shape {}, size= {}".format(mu.shape, size))
        latents = self.reparameterize(mu, sigma)
        treatments = treatments.repeat(size, 1)
        return self.control_decode(latents, treatments)
    
    
    def sample_expr(self, mu_x: torch.Tensor, sigma_x: torch.Tensor, mu_t: torch.Tensor, sigma_t: torch.Tensor, size=1) -> torch.Tensor:
        
        mu_x = mu_x.repeat(size, 1)
        sigma_x = sigma_x.repeat(size, 1)
        latents_x = self.reparameterize(mu_x, sigma_x)
        
        mu_t = mu_t.repeat(size, 1)
        sigma_t = sigma_t.repeat(size, 1)
        latents_t = self.reparameterize(mu_t, sigma_t)
        
        latents = torch.cat([latents_x, latents_t], -1)
        
        return self.decode(latents)


    def predict(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr = self.encode(outcomes, treatments, covariates)
            latents_dist = self.distributionize(
                latents_constr, dim=self.hparams["latent_dim"], dist="normal"
            )

            outcomes_constr = self.decode(latents_dist.mean, cf_treatments)
            outcomes_dist = self.distributionize(outcomes_constr)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean
        
        
    def predict_self(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, covariates = self.move_inputs(
            outcomes, treatments, covariates
        )
        with torch.autograd.no_grad():
            ZX_constr = self.encode_ZX(outcomes, covariates)
            ZX_dist = self.distributionize(
            ZX_constr, dim=self.hparams["ZX_dim"], dist="normal"
        )
        
            ZT_constr = self.encode_ZT(outcomes, treatments)
            ZT_dist = self.distributionize(
                ZT_constr, dim=self.hparams["ZT_dim"], dist="normal"
            )
            
            outcomes_constr = self.sample_expr(ZX_dist.mean, ZX_dist.stddev,ZT_dist.mean, ZT_dist.stddev)
            outcomes_dist = self.distributionize(outcomes_constr)
            
            # outcomes_constr = self.decode(latents_dist.mean, cf_treatments)
            # outcomes_dist = self.distributionize(outcomes_constr)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean    

    def generate(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr = self.encode(outcomes, treatments, covariates)
            latents_dist = self.distributionize(
                latents_constr, dim=self.hparams["latent_dim"], dist="normal"
            )

            outcomes_constr_samp = self.sample(
                latents_dist.mean, latents_dist.stddev, cf_treatments
            )
            outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

        if return_dist:
            return outcomes_dist_samp
        else:
            return outcomes_dist_samp.mean

    def logprob(self, outcomes, outcomes_param, dist=None):
        """
        Compute log likelihood.
        """
        if dist is None:
            dist = self.dist_outcomes

        num = len(outcomes)
        if isinstance(outcomes, list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes_param = [
                torch.repeat_interleave(out, sizes, dim=0) 
                for out in outcomes_param
            ]
            outcomes = torch.cat(outcomes, 0)
        elif isinstance(outcomes_param[0], list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes_param[0]], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes = torch.repeat_interleave(outcomes, sizes, dim=0)
            outcomes_param = [
                torch.cat(out, 0)
                for out in outcomes_param
            ]
        else:
            weights = None

        if dist == "nb":
            logprob = logprob_nb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                weight=weights
            )
        elif dist == "zinb":
            logprob = logprob_zinb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                zi_logits=outcomes_param[2],
                weight=weights
            )
        elif dist == "normal":
            logprob = logprob_normal(outcomes,
                loc=outcomes_param[0],
                scale=outcomes_param[1],
                weight=weights
            )
        elif dist == "bernoulli":
            logprob = logprob_bernoulli_logits(outcomes,
                loc=outcomes_param[0],
                weight=weights
            )

        return (logprob.sum(0)/num).mean()

    def loss(self, outcomes, outcomes_dist_samp,
            cf_outcomes, cf_outcomes_out,
            latents_dist, cf_latents_dist,
            treatments, cf_treatments,
            covariates, kde_kernel_std=1.0):
        """
        Compute losses.
        """
        # (1) individual-specific likelihood
        indiv_spec_nllh = -outcomes_dist_samp.log_prob(
            outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1))
        ).mean()

        # (2) covariate-specific likelihood
        if self.dist_mode == "discriminate":
            if self.iteration % self.hparams["discriminator_steps"]:
                self.update_discriminator(
                    outcomes, cf_outcomes_out.detach(),
                    treatments, cf_treatments, covariates
                )

            covar_spec_nllh = self.loss_discriminator(
                self.discriminate(cf_outcomes_out, cf_treatments, covariates),
                torch.ones(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
            )
        elif self.dist_mode == "fit":
            raise NotImplementedError(
                'TODO: implement dist_mode "fit" for distribution loss')
        elif self.dist_mode == "match":
            notNone = [o != None for o in cf_outcomes]
            cf_outcomes = [o for (o, n) in zip(cf_outcomes, notNone) if n]
            cf_outcomes_out = cf_outcomes_out[notNone]

            kernel_std = [kde_kernel_std * torch.ones_like(o) 
                for o in cf_outcomes]
            covar_spec_nllh = -self.logprob(
                cf_outcomes_out, (cf_outcomes, kernel_std), dist="normal"
            )

        # (3) kl divergence
        kl_divergence = kldiv_normal(
            latents_dist.mean,
            latents_dist.stddev,
            cf_latents_dist.mean,
            cf_latents_dist.stddev
        )

        return (indiv_spec_nllh, covar_spec_nllh, kl_divergence)
    
    ## control_outcomes_dist_samp, expr_outcomes_dist_samp, ZX_dist, 
    ## ZT_dist, ZX_prior_dist, ZT_prior_dist, control_latents_dist, control_prior_dist, 
    ## cov_constr, treatment_constr
    def loss_paired(self, control_outcomes, control_outcomes_dist_samp,
            expr_outcomes, expr_outcomes_dist_samp,ZX_dist, ZT_dist, control_latents_dist,
            ZX_prior_dist, ZT_prior_dist, control_prior_dist,
            cov_constr, treatment_constr, treatments,
            covariates, kde_kernel_std=1.0):
        """
        Compute losses.
        """
        # (1) individual-specific likelihood
        # indiv_spec_nllh = -outcomes_dist_samp.log_prob(
        #     outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1))
        # ).mean()
        
        ## control likelihood
        indiv_spec_nllh_control = -control_outcomes_dist_samp.log_prob(
            control_outcomes.repeat(self.mc_sample_size, *[1]*(control_outcomes.dim()-1))
        ).mean()
            
        ## experiments likelihood
        indiv_spec_nllh_experiments = -expr_outcomes_dist_samp.log_prob(
            expr_outcomes.repeat(self.mc_sample_size, *[1]*(expr_outcomes.dim()-1))
        ).mean()
            
        # covariate cross-entropy loss 
        cross_entropy_loss = nn.CrossEntropyLoss()
        cov_loss = 0.0
        for i in range(len(covariates)):
            cov_loss = cov_loss + cross_entropy_loss(cov_constr[...,i],covariates[i].squeeze())
    
        treatment_loss = cross_entropy_loss(treatment_constr.squeeze(),torch.argmax(treatments, 1))

        # (3) kl divergence
        ## ZX divergence
        kl_divergence_X = kldiv_normal(
            ZX_prior_dist.mean,
            ZX_prior_dist.stddev,
            ZX_dist.mean,
            ZX_dist.stddev,
        )
        print("kl_divergence_X type:{}, shape {}".format(type(kl_divergence_X), kl_divergence_X.shape))
        ## ZT divergence
        kl_divergence_T = kldiv_normal(
            ZT_prior_dist.mean,
            ZT_prior_dist.stddev,
            control_latents_dist.mean,
            control_latents_dist.stddev
        )
        
        ## control divergence
        kl_divergence_control = kldiv_normal(
            control_prior_dist.mean,
            control_prior_dist.stddev,
            control_latents_dist.mean,
            control_latents_dist.stddev
        )

        return (indiv_spec_nllh_control, indiv_spec_nllh_experiments, cov_loss, treatment_loss, kl_divergence_X, kl_divergence_T,kl_divergence_control)


    def forward(self, outcomes, treatments, control_outcomes, covariates,
                sample_latent=True, sample_outcome=False, detach_encode=False, detach_eval=True):
        """
        Execute the workflow.
        """

        # q(z_control | y_control, x)
        control_latent_constr = self.encode_control(control_outcomes, covariates)
        control_latents_dist = self.distributionize(
            control_latent_constr, dim=self.hparams["latent_dim"], dist="normal"
        )
        # print("control_latents_dist shape {}".format(control_latents_dist.shape))        

        ## q(z_x| y, x)
        ZX_constr = self.encode_ZX(outcomes, covariates)
        ZX_dist = self.distributionize(
            ZX_constr, dim=self.hparams["ZX_dim"], dist="normal"
        )
        
        ## q(z_t| y, T)
        ZT_constr = self.encode_ZT(outcomes, treatments)
        ZT_dist = self.distributionize(
            ZT_constr, dim=self.hparams["ZT_dim"], dist="normal"
        )

        control_prior = self.encode_control_prior(covariates)
        control_prior_dist = self.distributionize(
            control_prior, dim=self.hparams["latent_dim"], dist="normal"
        )
        
        ## p(z_x|x)
        ZX_constr_prior = self.encode_ZX_prior(covariates)
        ZX_prior_dist = self.distributionize(
            ZX_constr_prior, dim=self.hparams["ZX_dim"], dist="normal"
        )

        ZT_constr_prior = self.encode_ZT_prior(treatments)
        ZT_prior_dist = self.distributionize(
            ZT_constr_prior, dim=self.hparams["ZT_dim"], dist="normal"
        )
      
        ## define the sampling function
        ## sampling P(Z|y_control)
        control_latent_constr_samp = self.sample_latent(control_latents_dist.mean, control_latents_dist.stddev)
        control_latent_dist_samp = self.distributionize(control_latent_constr_samp)
        
        ## p(x|ZX)
        ZX_resample = self.sample_latent(ZX_dist.mean, ZX_dist.stddev)
        print("ZX_resample shape: {}".format(ZX_resample.shape))
        cov_inputs =ZX_resample
        cov_constr = self.covariate_decode(cov_inputs)
 
        ## p(t | ZT, control_latent_constr)
        ZT_resample = self.sample_latent(ZT_dist.mean, ZT_dist.stddev)
        print("ZT_resample shape: {}".format(ZT_resample.shape))

        treatment_constr = self.intervention_decode(ZT_resample, control_latent_constr_samp)
        
        control_outcomes_constr_samp = self.sample_control(control_latents_dist.mean, control_latents_dist.stddev,
            treatments, size=self.mc_sample_size
        )
        control_outcomes_dist_samp = self.distributionize(control_outcomes_constr_samp)
        expr_outcomes_constr_samp = self.sample_expr(ZX_dist.mean, ZX_dist.stddev,ZT_dist.mean, ZT_dist.stddev, size=self.mc_sample_size
        )
        expr_outcomes_dist_samp = self.distributionize(expr_outcomes_constr_samp)
        results = [control_outcomes_dist_samp, expr_outcomes_dist_samp, ZX_dist, ZT_dist, 
                   ZX_prior_dist, ZT_prior_dist, control_latents_dist, control_prior_dist, cov_constr, treatment_constr]
            
        return results

    
    def update(self, expr_outcomes, treatments, contol_outcomes, covariates):
        """
        Update model's parameters given a minibatch of outcomes, treatments, and covariates.
        """
        expr_outcomes, treatments, control_outcomes, covariates = self.move_inputs(
            expr_outcomes, treatments, contol_outcomes, covariates
        )

        control_outcomes_dist_samp, expr_outcomes_dist_samp, ZX_dist, ZT_dist, \
        ZX_prior_dist, ZT_prior_dist,  control_latents_dist, control_prior_dist, \
        cov_constr, treatment_constr = self.forward(
            expr_outcomes, treatments, control_outcomes, covariates
        )
        
        ndiv_spec_nllh_control, indiv_spec_nllh_experiments, cov_loss, \
        treatment_loss, kl_divergence_X, kl_divergence_T,kl_divergence_control =\
        self.loss_paired(control_outcomes, control_outcomes_dist_samp, expr_outcomes, expr_outcomes_dist_samp,
                       ZX_dist, ZT_dist, control_latents_dist,ZX_prior_dist, ZT_prior_dist, control_prior_dist,
                       cov_constr, treatment_constr, treatments,covariates)
    
        indiv_spec_nllh = ndiv_spec_nllh_control + indiv_spec_nllh_experiments
        covar_spec_nllh = treatment_loss + cov_loss
        kl_divergence = kl_divergence_X+kl_divergence_T+kl_divergence_control
        
        loss = (self.omega0 * indiv_spec_nllh
            + self.omega1 * covar_spec_nllh
            + self.omega2 * kl_divergence
        )

        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()
        self.iteration += 1

        return {
            "Indiv-spec NLLH": indiv_spec_nllh.item(),
            "Covar-spec NLLH": covar_spec_nllh.item(),
            "KL Divergence": kl_divergence.item()
        }

    def update_discriminator(self, outcomes, cf_outcomes_out,
                                treatments, cf_treatments, covariates):
        loss_tru = self.loss_discriminator(
            self.discriminate(outcomes, treatments, covariates),
            torch.ones(outcomes.size(0), device=outcomes.device)
        )

        loss_fls = self.loss_discriminator(
            self.discriminate(cf_outcomes_out, cf_treatments, covariates),
            torch.zeros(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
        )

        loss = (loss_tru+loss_fls)/2.
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item()

    def update_eval_encoder(self):
        for target_param, param in zip(
            self.encoder_eval.parameters(), self.encoder.parameters()
        ):
            target_param.data.copy_(param.data)
            
    def update_eval_encoder(self):
        for target_param, param in zip(
            self.encoder_ZX_eval.parameters(), self.encoder_ZX.parameters()
        ):
            target_param.data.copy_(param.data)   
            
        for target_param, param in zip(
            self.encoder_ZT_eval.parameters(), self.encoder_ZT.parameters()
        ):
            target_param.data.copy_(param.data)          

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def init_outcome_emb(self):
        return MLP(
            [self.num_outcomes, self.hparams["outcome_emb_dim"]], final_act="relu"
        )

    def init_treatment_emb(self):
        
        if self.type_treatments in ("object", "bool", "category", None):
            print(self.num_treatments, self.hparams["treatment_emb_dim"])
            return CompoundEmbedding(
                self.num_treatments, self.hparams["treatment_emb_dim"]
            )
        else:
            return MLP(
                [self.num_treatments] + [self.hparams["treatment_emb_dim"]] * 2
            )

    def init_covariates_emb(self):
        type_covariates = self.type_covariates
        if type_covariates is None or isinstance(type_covariates, str):
            type_covariates = [type_covariates] * len(self.num_covariates)

        covariates_emb = []
        for num_cov, type_cov in zip(self.num_covariates, type_covariates):
            if type_cov in ("object", "bool", "category", None):
                covariates_emb.append(CompoundEmbedding(
                        num_cov, self.hparams["covariate_emb_dim"]
                    ))
            else:
                covariates_emb.append(MLP(
                        [num_cov] + [self.hparams["covariate_emb_dim"]] * 2
                    ))
        return covariates_emb

    def init_encoder(self):
        return MLP([self.outcome_dim+self.treatment_dim+self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["latent_dim"]],
            heads=2, final_act="relu"
        )
    
    def init_encoder_control(self):
        return MLP([self.outcome_dim+self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["latent_dim"]],
            heads=2, final_act="relu"
        )
     
    def init_encoder_X(self):
        return MLP([self.outcome_dim+self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZX_dim"]],
            heads=2, final_act="relu"
        )
    
    def init_encoder_X_prior(self):
        return MLP([self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZX_dim"]],
            heads=2, final_act="relu"
        )
    def init_encoder_T(self):
        return MLP([self.outcome_dim+self.treatment_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZT_dim"]],
            heads=2, final_act="relu"
        )
    
    def init_encoder_T_prior(self):
        return MLP([self.treatment_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZT_dim"]],
            heads=2, final_act="relu"
        )
    
    def init_control_prior(self):
        return MLP([self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZT_dim"]],
            heads=2, final_act="relu"
        )


    def init_decoder(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return MLP([self.hparams["latent_dim"]+self.treatment_dim]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_outcomes],
            heads=heads
        )
    
    def init_decoder_control(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return MLP([self.hparams["latent_dim"]]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_outcomes],
            heads=heads
        )
    
    
    def init_decoder_experiments(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return MLP([self.hparams["ZX_dim"] + self.hparams["ZT_dim"]]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_outcomes],
            heads=heads
        )
    
    def init_decoder_cov(self):
        return MLP([self.hparams["ZX_dim"]]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [np.max(self.num_covariates)],
            heads=len(self.num_covariates), final_act= "softmax"
        )
    
    def init_decoder_interv(self):
        return MLP([self.hparams["latent_dim"]+self.hparams["ZT_dim"]]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_treatments],
            heads=1, final_act= "softmax"
        )
    

    def init_discriminator(self):
        return MLP([self.outcome_dim+self.treatment_dim+self.covariate_dim]
            + [self.hparams["discriminator_width"]] * (self.hparams["discriminator_depth"] - 1)
            + [1]
        )

    def move_input(self, input):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if isinstance(input, list):
            return [i.to(self.device) if i is not None else None for i in input]
        else:
            return input.to(self.device)

    def move_inputs(self, *inputs: torch.Tensor):
        """
        Move minibatch tensors to CPU/GPU.
        """
        return [self.move_input(i) if i is not None else None for i in inputs]

    def to_device(self, device):
        self.device = device
        self.to(self.device)

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for VCI
        """

        return self._set_hparams_(self, "")

    
    @torch.no_grad()
    def get_latent(self, outcomes, treatments, covariates):
        
        ZX_constr = self.encode_ZX(outcomes, covariates)
        ZX_dist = self.distributionize(
            ZX_constr, dim=self.hparams["ZX_dim"], dist="normal"
        )
        
        ## q(z_t| y, T)
        ZT_constr = self.encode_ZT(outcomes, treatments)
        ZT_dist = self.distributionize(
            ZT_constr, dim=self.hparams["ZT_dim"], dist="normal"
        )

        ZX = ZX_dist.rsample()
        ZT = ZT_dist.rsample()
        return ZX, ZT
        
    
    
    @torch.no_grad()
    def get_latent_presentation(self, outcomes, treatments, covariates):
        outcomes, treatments, covariates = self.move_inputs(
            outcomes, treatments, covariates
        )
        ZX, ZT = self.get_latent(
            outcomes, treatments, covariates
        )
        return ZX, ZT
    
    
    @torch.no_grad()
   

    def get_control_latent(self, outcomes, covariates):
        
        control_latent_constr = self.encode_control(outcomes, covariates)
        control_latents_dist = self.distributionize(
            control_latent_constr, dim=self.hparams["latent_dim"], dist="normal"
        )
        return control_latents_dist.rsample()
    
    
    @torch.no_grad()
    def get_control_representation(self, outcomes, covariates):
        outcomes, covariates = self.move_inputs(
            outcomes, covariates
        )
        latents = self.get_control_latent(
            outcomes, covariates
        )
        return latents