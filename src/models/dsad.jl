using Flux: setup, train!
using Statistics: mean

"""
    DSADDetector(encoder = Chain(),
                    decoder = Chain(),
                    batchsize = 32,
                    epochs = 1,
                    shuffle = true,
                    partial = false,
                    opt = Adam(),
                    loss = mse,
                    eta = 1,
                    eps = 1e-6,
                    callback = _ -> () -> ())

Deep Semi-Supervised Anomaly detection technique based on the distance to a hypersphere center as described in [1].

Parameters
----------
$AE_PARAMS

$AE_LOSS

    eta::Real
Weighting parameter for the labeled data; i.e. higher values of eta assign higher weight to labeled data in the svdd
loss function. For a sensitivity analysis of this parameter, see [1].

    eps::Real
Because the inverse distance used in the svdd loss can lead to division by zero, the parameters `eps` is added for
numerical stability.

    callback::Function
*Experimental parameter that might change*. A function to be called after the model parameters have been updated that
can call Flux's callback helpers, see <https://fluxml.ai/Flux.jl/stable/utilities/#Callback-Helpers-1>.

**Notice:** The parameters `batchsize`, `epochs`, `shuffle`, `partial`, `opt` and `callback` can also be tuples of size
2, specifying the corresponding values for (1) pretraining and (2) training; otherwise the same values are used for
pretraining and training.

Examples
--------
$(SCORE_SUPERVISED("DSADDetector"))

References
----------
[1] Ruff, Lukas; Vandermeulen, Robert A.; Görnitz, Nico; Binder, Alexander; Müller, Emmanuel; Müller, Klaus-Robert;
Kloft, Marius (2019): Deep Semi-Supervised Anomaly Detection.
"""
mutable struct DSADDetector <: SupervisedDetector
    encoder::Chain
    decoder::Chain
    batchsize::Tuple{Integer,Integer}
    epochs::Tuple{Integer,Integer}
    shuffle::Tuple{Bool,Bool}
    partial::Tuple{Bool,Bool}
    opt::Any
    loss::Function
    eta::Number
    eps::Number
    callback::Tuple{Function,Function}
    function DSADDetector(;
                          encoder::Chain = Chain(),
                          decoder::Chain = Chain(),
                          batchsize = 32, epochs = 1, shuffle = false,
                          partial = true, opt = Adam(), loss = mse, eta = 1, eps = 1e-6,
                          callback = (((m, x) -> ()), ((m, x, y) -> ())))

        # unify all possible tuples to tuples
        tuplify = t -> isa(t, Tuple) ? t : (t, t)
        batchsize, epochs, shuffle, partial, opt, callback =
            map(tuplify, (batchsize, epochs, shuffle, partial, opt, callback))

        new(encoder, decoder, batchsize, epochs, shuffle, partial, opt, loss, eta, eps, callback)
    end
end

struct DSADModel <: DetectorModel
    chain::Chain
    center::AbstractArray
    ndims::Integer
end

(m::DSADModel)(x) = svddScore(m.chain(x), m.center, m.ndims)

function OD.fit(detector::DSADDetector, X::Data, y::Labels; verbosity)::Fit
    makeLoader = i -> DataLoader((X, y), batchsize = detector.batchsize[i], shuffle = detector.shuffle[i],
        partial = detector.partial[i])
    loaderPretrain = makeLoader(1)
    loaderTrain = makeLoader(2)

    # Create the autoencoder
    ae_model = Chain(detector.encoder, detector.decoder)

    # pretraining (train the autoencoder based on a reconstruction loss)
    pretrain_state = setup(detector.opt[1], ae_model)
    for _ in 1:detector.epochs[1]
        train!(ae_model,
               loaderPretrain,
               pretrain_state
               ) do model, x, _
                   detector.loss(model(x), x)
               end
        detector.callback[1](ae_model, X)
    end

    # Determine the normal class
    normal_class = first(levels(y))

    # Use normal data and unlabeled data to calculate hypersphere center
    dims = ndims(X)
    nColons = X -> ntuple(_ -> :, dims - 1)
    prediction = detector.encoder(X[nColons(X)..., findall((y .=== missing) .| (y .== normal_class))])
    center = dropdims(mean(prediction, dims = dims), dims = dims)

    ad_model = DSADModel(detector.encoder, center, dims)
    # training based on the calculated hypersphere center
    train_state = setup(detector.opt[2], ad_model)
    for _ in 1:detector.epochs[2]
        train!(ad_model,
               loaderTrain,
               train_state
               ) do model, x, y
                   svddLoss(model.chain(x), model.center, y, detector.eta, detector.eps, dims, normal_class)
               end
        detector.callback[2](ad_model, X, y)
    end

    return ad_model, ad_model(X)
end

function OD.transform(_::DSADDetector, model::DSADModel, X::Data)::Scores
    model(X)
end

function svddLoss(latent, center, y, eta, eps, dims, normal_class)
    # The svdd loss function is based on the distance to the hypersphere center. The inverse distance is used if an
    # example is an outlier and labeled samples are weighted using the hyperparameter eta.
    conditional_dist(y, dist, eps_dist) = ifelse(y === missing, dist,
        eta .* (y == normal_class ? eps_dist : eps_dist .^ -1))
    dist = dropdims(sum((latent .- center) .^ 2, dims = 1:dims-1), dims = 1)
    eps_dist = dist .+ eps
    mean(conditional_dist.(y, dist, eps_dist))
end

function svddScore(latent, center, dims)
    # Element-wise mean squared distance to the hypersphere center.
    dropdims(mean((latent .- center) .^ 2, dims = 1:dims-1), dims = 1)
end
