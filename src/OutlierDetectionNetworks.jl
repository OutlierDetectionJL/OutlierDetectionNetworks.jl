module OutlierDetectionNetworks
    using OutlierDetectionInterface
    using OutlierDetectionInterface:SCORE_UNSUPERVISED, SCORE_SUPERVISED
    const OD = OutlierDetectionInterface

    include("utils.jl")
    include("models/ae.jl")
    include("models/dsad.jl")
    include("models/esad.jl")
    include("templates/templates.jl")

    const UUID = "c7f57e37-4fcb-4a0b-a36c-c2204bc839a7"
    const MODELS = [:AEDetector,
                    :DSADDetector,
                    :ESADDetector]

    for model in MODELS
        @eval begin
            OD.@default_frontend $model
            OD.@default_metadata $model $UUID
            export $model
        end
    end
end
