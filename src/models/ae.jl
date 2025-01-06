using Flux: Chain, train!, setup, DataLoader, Adam, mse
using Statistics:mean

"""
    AEDetector(encoder= Chain(),
               decoder = Chain(),
               batchsize= 32,
               epochs = 1,
               shuffle = false,
               partial = true,
               opt = Adam(),
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
    opt::Any = Adam()
    loss::Function = mse
end

struct AEModel <: DetectorModel
    chain::Chain
    loss::Function
end

(m::AEModel)(x) = m.loss(m.chain(x), x, agg=instance_mean)

function OD.fit(detector::AEDetector, X::Data; verbosity)::Fit
    loader = DataLoader(X, batchsize=detector.batchsize, shuffle=detector.shuffle, partial=detector.partial)

    # Create the autoencoder network
    model = Chain(detector.encoder, detector.decoder)

    # train the neural network model
    state = setup(detector.opt, model)
    for _ in 1:detector.epochs
        train!(
            model,
            loader,
            state
        ) do m, x
            detector.loss(m(x), x)
        end
    end

    trained_model = AEModel(model, detector.loss)
    return trained_model, trained_model(X)
end

function OD.transform(_::AEDetector, model::AEModel, X::Data)::Scores
    model(X)
end
