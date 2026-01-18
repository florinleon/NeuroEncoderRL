# ------------------------------------------------------------------
# Input and detector parameters
# ------------------------------------------------------------------
InputDimensionCount = 69  # number of input dimensions (full 69-D bridge observation)
DetectorsPerDimension = 11  # number of detectors per input dimension
GaussianSigmaFactor = 1.5  # scaling factor for detector tuning width
GaussianSigma = 0.8 #GaussianSigmaFactor / float(DetectorsPerDimension)  # Gaussian width for detector tuning curves

# ------------------------------------------------------------------
# Neural sheet (encoder) parameters
# ------------------------------------------------------------------
SheetWidth = 20  # width of the neural sheet (units)
SheetHeight = 20  # height of the neural sheet (units)

InhibitionSigma = 2.0      # lateral inhibition sigma on the sheet
InhibitionStrength = 1.0   # strength of lateral inhibition on the sheet
TimeSteps = 8              # number of recurrent inhibition time steps
SparsityPercentile = 0.97  # percentile used to threshold sheet activations
FanIn = 10                 # number of detector inputs per sheet unit
LeakRate = 0.20            # leak rate in sheet dynamics

UseNoise = False  # whether to add neural noise on the sheet before sparsification
NoiseStd = 0      # standard deviation of the neural noise on the sheet
