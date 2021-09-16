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
            OD.metadata_pkg($model, package_name=string(@__MODULE__), package_uuid=$UUID,
                            package_url="https://github.com/OutlierDetectionJL/$(@__MODULE__).jl",
                            is_pure_julia=true, package_license="MIT", is_wrapper=false)
            OD.load_path(::Type{$model}) = string($model)
            export $model
        end
    end
end
