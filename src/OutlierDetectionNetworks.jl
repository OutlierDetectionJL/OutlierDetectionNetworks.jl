module OutlierDetectionNetworks
    using OutlierDetectionInterface
    using OutlierDetectionInterface:SCORE_UNSUPERVISED, SCORE_SUPERVISED
    const OD = OutlierDetectionInterface

    export AEDetector,
           AEModel,
           DSADDetector,
           DSADModel,
           ESADDetector,
           ESADModel

    include("utils.jl")
    include("models/ae.jl")
    include("models/dsad.jl")
    include("models/esad.jl")
    include("templates/templates.jl")

    MODELS = (AEDetector,
              DSADDetector,
              ESADDetector)

    ORG = "OutlierDetectionJL"
    UUID = "51249a0a-cb36-4849-8e04-30c7f8d311bb"
    for model in MODELS
        OD.metadata_pkg(model, package_name=@__MODULE__, package_uuid=UUID,
                        package_url="https://github.com/$ORG/$(@__MODULE__).jl",
                        is_pure_julia=true, package_license="MIT", is_wrapper=false)
        OD.load_path(::Type{model}) = "$(@__MODULE__).$model"
    end
end
