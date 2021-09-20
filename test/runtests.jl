using OutlierDetectionNetworks
using OutlierDetectionNetworks.Templates
using OutlierDetectionTest

# Test the metadata of all exported detectors
test_meta.(eval.(OutlierDetectionNetworks.MODELS))

data = TestData()
run_test(detector) = test_detector(detector, data)

const encoder, decoder = MLPAutoEncoder(size(data.x_raw, 1), 5, [50,20]; bias=false);

# AE
run_test(AEDetector(encoder=encoder, decoder=decoder))

# DSAD
run_test(DSADDetector(encoder=encoder, decoder=decoder))

# ESAD
run_test(ESADDetector(encoder=encoder, decoder=decoder))
