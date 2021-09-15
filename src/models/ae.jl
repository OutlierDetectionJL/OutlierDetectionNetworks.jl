using Flux: Chain, train!, params
using Flux.Losses:mse
using Flux.Data:DataLoader
using Flux.Optimise:ADAM
using IterTools:ncycle
using Statistics:mean

"""
    AEDetector(encoder= Chain(),
               decoder = Chain(),
               batchsize= 32,
               epochs = 1,
               shuffle = false,
               partial = true,
               opt = ADAM(),
               loss = mse)

Calculate the anomaly score of an instance based on the reconstruction loss of an autoencoder, see [1] for an
explanation of auto encoders.

Parameters
----------
$AE_PARAMS

$AE_LOSS

Examples
--------
$(SCORE_UNSUPERVISED("AEDetector"))

References
----------
[1] Aggarwal, Charu C. (2017): Outlier Analysis.
"""
OD.@detector struct AEDetector <: UnsupervisedDetector
    encoder::Chain = Chain()
    decoder::Chain = Chain()
    batchsize::Integer = 32
    epochs::Integer = 1
    shuffle::Bool = false
    partial::Bool = true
    opt::Any = ADAM()
    loss::Function = mse
end

struct AEModel <: DetectorModel
    chain::Chain
end

function OD.fit(detector::AEDetector, X::Data; verbosity)::Fit
    loader = DataLoader(X, batchsize=detector.batchsize, shuffle=detector.shuffle, partial=detector.partial)

    # Create the autoencoder
    model = Chain(detector.encoder, detector.decoder)

    # train the neural network model
    train!(x -> detector.loss(model(x), x), params(model), ncycle(loader, detector.epochs), detector.opt)

    scores = detector.loss(model(X), X, agg=instance_mean)
    return AEModel(model), scores
end

function OD.transform(detector::AEDetector, model::AEModel, X::Data)::Scores
    detector.loss(model.chain(X), X, agg=instance_mean)
end
