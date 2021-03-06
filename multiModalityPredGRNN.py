# 2018/6/12~. K-step prediction

# Main call to test the GNN architectures. This simulates several graphs and
# runs several data realizations per graph.

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       of each realization on a directory named 'savedModels'
#   - It saves a pickle file with the torch random state and the numpy random
#       state for reproducibility.
#   - It saves a text file 'hyperparameters.txt' containing the specific
#       (hyper)parameters that control the run, together with the main (scalar)
#       results obtained.
#   - If desired, logs in tensorboardX the training loss and evaluation measure
#       both of the training set and the validation set. These tensorboardX logs
#       are saved in a logsTB directory.
#   - If desired, saves the vector variables of each realization (training and
#       validation loss and evaluation measure, respectively); this is saved
#       both in pickle and in Matlab(R) format. These variables are saved in a
#       trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in both pickle and
#       Matlab(R) format. These plots (and variables) are in a figs directory.

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import datetime
from scipy.io import savemat

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Utils.miscTools as misc
import Modules.architectures as archit
import Modules.model as model
import Modules.train_rnn as train

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

import ipdb
#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

thisFilename = 'RNNcomp' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

nNodes = 80 # Number of nodes
graphType = 'SBM' # Type of graph
nCommunities = 5 # Number of communities
graphOptions = {}
graphOptions['nCommunities'] = 5 
graphOptions['probIntra'] = 0.8 # Intracommunity probability
graphOptions['probInter'] = 0.2 # Intercommunity probability

nTrain = 100 # Number of training samples
nValid = int(0.25 * nTrain) # Number of validation samples
nTest = int(0.05 * nTrain) # Number of testing samples
tMax = None # Maximum number of diffusion times (A^t for t < tMax)

nDataRealizations = 1 # Number of data realizations
nGraphRealizations = 1 # Number of graph realizations

K = 1 # predict signal K steps ahead
num_timestep = 200
seqLen = num_timestep - K # sequence length
F_t = 5

sigmaSpatial = 0.1
sigmaTemporal = 0.1
rhoSpatial = 0
rhoTemporal = 0

#\\\ Save values:
writeVarValues(varsFile,
               {'nNodes': nNodes,
                'graphType': graphType,
                'nCommunities': graphOptions['nCommunities'],
                'probIntra': graphOptions['probIntra'],
                'probInter': graphOptions['probInter'],
                'nTrain': nTrain,
                'nValid': nValid,
                'nTest': nTest,
                'tMax': tMax,
                'nDataRealizations': nDataRealizations,
                'nGraphRealizations': nGraphRealizations,
                'seqLen': seqLen,
                'sigmaSpatial': sigmaSpatial,
                'sigmaTemporal': sigmaTemporal,
                'rhoSpatial': rhoSpatial,
                'rhoTemporal': rhoTemporal})

############
# TRAINING #
############

#\\\ Individual model training options
trainer = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.001 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = misc.batchTimeL1Loss # This applies a softmax before feeding
    # it into the NLL, so we don't have to apply the softmax ourselves.

#\\\ Overall training options
nEpochs = 100 # Number of epochs
batchSize = 25 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 40 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'trainer': trainer,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Now that we have accumulated quite an extensive number of architectures, we
# might not want to run all of them. So here we select which ones we want to
# run.

# Select desired architectures
doSelectionGNN = False
doRNN_MLP = False
doGCRNN_MLP = False
doGCRNN_GNN =  False
doTimeGCRNN_MLP = False
doTimeGCRNN_GNN =  True
doNodeGCRNN_MLP = False
doEdgeGCRNN_MLP = False
doTimeNodeGCRNN_MLP = False
doTimeEdgeGCRNN_MLP = False

'''
observations: 
- edge gates are worse than node gates
- TimeNode gates are worse than time alone or node alone #TODO: explain this
- Time gate alone with GNN is the best among all gating machenism (also slightly better than no gate w/ GNN)
'''
# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. That is, any new architecture in this part, needs also
# to be coded later on. This is just to be easy to change the parameters once
# the architecture is created. Do not forget to add the name of the architecture
# to modelList.

modelList = []

# Parameters to share across several architectures

# Linear layer
F1 = 5 # Number of features for the first layer
rnnStateFeat = 1 # Number of state features for the RNN
K1 = 5 # Number of filter taps for the first layer, or number of attention heads
# MLP layer
dimMLP = [1] # MLP after the last layer

#\\\\\\\\\\\\
#\\\ MODEL 1: Selection GNN ordered by Degree
#\\\\\\\\\\\\

if doSelectionGNN:

    hParamsSelGNNDeg = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsSelGNNDeg['name'] = 'Sel' # Name of the architecture

    #\\\ Architecture parameters
    hParamsSelGNNDeg['F'] = [1, 8, 1] # Features per layer
    hParamsSelGNNDeg['K'] = [10, 10] # Number of filter taps per layer
    hParamsSelGNNDeg['bias'] = True # Decide whether to include a bias term
    hParamsSelGNNDeg['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsSelGNNDeg['N'] = [nNodes, nNodes] # Number of nodes to keep at the end of
        # each layer
    hParamsSelGNNDeg['rho'] = gml.NoPool # Summarizing function
    hParamsSelGNNDeg['alpha'] = [1, 1] # alpha-hop neighborhood that
        # is affected by the summary
    hParamsSelGNNDeg['dimLayersMLP'] = [] # Dimension of the fully
        # connected layers after the GCN layers

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsSelGNNDeg)
    modelList += [hParamsSelGNNDeg['name']]  

#\\\\\\\\\\\\
#\\\ MODEL 2: GCRNN followed by localized perceptrons
#\\\\\\\\\\\\

if doGCRNN_MLP:

    hParamsGCRNN_MLP = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsGCRNN_MLP['name'] = 'GCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsGCRNN_MLP['inFeatures'] = 1 
    hParamsGCRNN_MLP['stateFeatures'] = F1 
    hParamsGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsGCRNN_MLP['dimLayersMLP'] = [1] 
    hParamsGCRNN_MLP['bias'] = True
    hParamsGCRNN_MLP['time_gating'] = False
    hParamsGCRNN_MLP['spatial_gating'] = None
    hParamsGCRNN_MLP['mlpType'] = 'multipMlp'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGCRNN_MLP)
    modelList += [hParamsGCRNN_MLP['name']]


#\\\\\\\\\\\\
#\\\ MODEL 3: RNN followed by MLP
#\\\\\\\\\\\\

if doRNN_MLP:

    hParamsRNN_MLP = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsRNN_MLP['name'] = 'RNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsRNN_MLP['inFeatures'] = 1 # discounting the number of nodes, that is,
    # the real number of input features is nNodes*inFeatures
    hParamsRNN_MLP['stateFeatures'] = rnnStateFeat # actual number of state features
    hParamsRNN_MLP['stateNonlinearity'] = 'tanh'
    hParamsRNN_MLP['dimLayersMLP'] = [1] # discounting the number of nodes
    hParamsRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsRNN_MLP['bias'] = True

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsRNN_MLP)
    modelList += [hParamsRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 4: GCRNN followed by GNN
#\\\\\\\\\\\\

if doGCRNN_GNN:

    hParamsGCRNN_GNN = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsGCRNN_GNN['name'] = 'GCRNNGNN' # Name of the architecture

    #\\\ Architecture parameters
    hParamsGCRNN_GNN['inFeatures'] = 1 
    hParamsGCRNN_GNN['stateFeatures'] = F1
    hParamsGCRNN_GNN['inputFilterTaps'] = K1 
    hParamsGCRNN_GNN['stateFilterTaps'] = K1 
    hParamsGCRNN_GNN['stateNonlinearity'] = torch.tanh
    hParamsGCRNN_GNN['outputNonlinearity'] = nn.ReLU
    hParamsGCRNN_GNN['dimLayersMLP'] = [] 
    hParamsGCRNN_GNN['bias'] = True
    hParamsGCRNN_GNN['time_gating'] = False
    hParamsGCRNN_GNN['spatial_gating'] = None
    hParamsGCRNN_GNN['mlpType'] = 'oneMlp'
    hParamsGCRNN_GNN['finalNonlinearity'] = nn.ReLU
    hParamsGCRNN_GNN['dimNodeSignals'] = [5,1]
    hParamsGCRNN_GNN['nFilterTaps'] = [5]
    hParamsGCRNN_GNN['nSelectedNodes'] = [nNodes]
    hParamsGCRNN_GNN['poolingFunction'] = gml.NoPool
    hParamsGCRNN_GNN['poolingSize=None'] = [1]

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGCRNN_GNN)
    modelList += [hParamsGCRNN_GNN['name']]


#\\\\\\\\\\\\
#\\\ MODEL 5: Time gated GCRNN followed by GNN
#\\\\\\\\\\\\

if doTimeGCRNN_GNN:

    hParamsTimeGCRNN_GNN = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsTimeGCRNN_GNN['name'] = 'TimeGCRNNGNN' # Name of the architecture

    #\\\ Architecture parameters
    hParamsTimeGCRNN_GNN['inFeatures'] = 1 
    hParamsTimeGCRNN_GNN['stateFeatures'] = F1 
    hParamsTimeGCRNN_GNN['inputFilterTaps'] = K1
    hParamsTimeGCRNN_GNN['stateFilterTaps'] = K1 
    hParamsTimeGCRNN_GNN['stateNonlinearity'] = torch.tanh
    hParamsTimeGCRNN_GNN['outputNonlinearity'] = nn.ReLU
    hParamsTimeGCRNN_GNN['dimLayersMLP'] = [] 
    hParamsTimeGCRNN_GNN['bias'] = True
    hParamsTimeGCRNN_GNN['time_gating'] = True
    hParamsTimeGCRNN_GNN['spatial_gating'] = None
    hParamsTimeGCRNN_GNN['mlpType'] = 'oneMlp'
    hParamsTimeGCRNN_GNN['finalNonlinearity'] = nn.ReLU
    hParamsTimeGCRNN_GNN['dimNodeSignals'] = [5,1]
    hParamsTimeGCRNN_GNN['nFilterTaps'] = [5]
    hParamsTimeGCRNN_GNN['nSelectedNodes'] = [nNodes]
    hParamsTimeGCRNN_GNN['poolingFunction'] = gml.NoPool
    hParamsTimeGCRNN_GNN['poolingSize=None'] = [1]
    hParamsTimeGCRNN_GNN['multiModalOutput'] ='mlp' #'avg' 'NA'
    hParamsTimeGCRNN_GNN['F_t'] = F_t

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsTimeGCRNN_GNN)
    modelList += [hParamsTimeGCRNN_GNN['name']]

#\\\\\\\\\\\\
#\\\ MODEL 6: Time gated GCRNN followed by localized perceptrons
#\\\\\\\\\\\\

if doTimeGCRNN_MLP:

    hParamsTimeGCRNN_MLP = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsTimeGCRNN_MLP['name'] = 'TimeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsTimeGCRNN_MLP['inFeatures'] = 1 
    hParamsTimeGCRNN_MLP['stateFeatures'] = F1 
    hParamsTimeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsTimeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsTimeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsTimeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsTimeGCRNN_MLP['dimLayersMLP'] = [1] 
    hParamsTimeGCRNN_MLP['bias'] = True
    hParamsTimeGCRNN_MLP['time_gating'] = True
    hParamsTimeGCRNN_MLP['spatial_gating'] = None
    hParamsTimeGCRNN_MLP['mlpType'] = 'multipMlp'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsTimeGCRNN_MLP)
    modelList += [hParamsTimeGCRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 7: Node gated GCRNN followed by localized perceptrons
#\\\\\\\\\\\\

if doNodeGCRNN_MLP:

    hParamsNodeGCRNN_MLP = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsNodeGCRNN_MLP['name'] = 'NodeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsNodeGCRNN_MLP['inFeatures'] = 1 
    hParamsNodeGCRNN_MLP['stateFeatures'] = F1 
    hParamsNodeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsNodeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsNodeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsNodeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsNodeGCRNN_MLP['dimLayersMLP'] = [1] 
    hParamsNodeGCRNN_MLP['bias'] = True
    hParamsNodeGCRNN_MLP['time_gating'] = False
    hParamsNodeGCRNN_MLP['spatial_gating'] = 'node'
    hParamsNodeGCRNN_MLP['mlpType'] = 'multipMlp'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsNodeGCRNN_MLP)
    modelList += [hParamsNodeGCRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 8: Edge gated GCRNN followed by localized perceptrons
#\\\\\\\\\\\\

if doEdgeGCRNN_MLP:

    hParamsEdgeGCRNN_MLP = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsEdgeGCRNN_MLP['name'] = 'EdgeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsEdgeGCRNN_MLP['inFeatures'] = 1 
    hParamsEdgeGCRNN_MLP['stateFeatures'] = F1 
    hParamsEdgeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsEdgeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsEdgeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsEdgeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsEdgeGCRNN_MLP['dimLayersMLP'] = [1] 
    hParamsEdgeGCRNN_MLP['bias'] = True
    hParamsEdgeGCRNN_MLP['time_gating'] = False
    hParamsEdgeGCRNN_MLP['spatial_gating'] = 'edge'
    hParamsEdgeGCRNN_MLP['mlpType'] = 'multipMlp'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsEdgeGCRNN_MLP)
    modelList += [hParamsEdgeGCRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 9: Time Node gated GCRNN followed by localized perceptrons
#\\\\\\\\\\\\

if doTimeNodeGCRNN_MLP:

    hParamsTimeNodeGCRNN_MLP = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsTimeNodeGCRNN_MLP['name'] = 'TimeNodeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsTimeNodeGCRNN_MLP['inFeatures'] = 1 
    hParamsTimeNodeGCRNN_MLP['stateFeatures'] = F1 
    hParamsTimeNodeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsTimeNodeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsTimeNodeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsTimeNodeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsTimeNodeGCRNN_MLP['dimLayersMLP'] = [1] 
    hParamsTimeNodeGCRNN_MLP['bias'] = True
    hParamsTimeNodeGCRNN_MLP['time_gating'] = True
    hParamsTimeNodeGCRNN_MLP['spatial_gating'] = 'node'
    hParamsTimeNodeGCRNN_MLP['mlpType'] = 'multipMlp'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsTimeNodeGCRNN_MLP)
    modelList += [hParamsTimeNodeGCRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 10: Time Edge gated GCRNN followed by localized perceptrons
#\\\\\\\\\\\\

if doTimeEdgeGCRNN_MLP:

    hParamsTimeEdgeGCRNN_MLP = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)

    hParamsTimeEdgeGCRNN_MLP['name'] = 'TimeEdgeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsTimeEdgeGCRNN_MLP['inFeatures'] = 1 
    hParamsTimeEdgeGCRNN_MLP['stateFeatures'] = F1 
    hParamsTimeEdgeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsTimeEdgeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsTimeEdgeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsTimeEdgeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsTimeEdgeGCRNN_MLP['dimLayersMLP'] = [1] 
    hParamsTimeEdgeGCRNN_MLP['bias'] = True
    hParamsTimeEdgeGCRNN_MLP['time_gating'] = True
    hParamsTimeEdgeGCRNN_MLP['spatial_gating'] = 'edge'
    hParamsTimeEdgeGCRNN_MLP['mlpType'] = 'multipMlp'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsTimeEdgeGCRNN_MLP)
    modelList += [hParamsTimeEdgeGCRNN_MLP['name']]
    
###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 10 # How many validation steps in between those shown,
    # same as above.

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'

# Notify:
if doPrint:
    print("Device selected: %s" % device)

#\\\ Logging options
if doLogging:
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each for the trained models.
# It basically is a dictionary, containing a list of lists. The key of the
# dictionary determines de the model, then the first list index determines
# which graph, and the second list index, determines which realization within
# that graph. Then, this will be converted to numpy to compute mean and standard
# deviation (across the graph dimension).
accBest = {} # Accuracy for the best model
accLast = {} # Accuracy for the last model
for thisModel in modelList: # Create an element for each graph realization,
    # each of these elements will later be another list for each realization.
    # That second list is created empty and just appends the results.
    accBest[thisModel] = [None] * nGraphRealizations
    accLast[thisModel] = [None] * nGraphRealizations

####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of this options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

#%%##################################################################
#                                                                   #
#                    GRAPH REALIZATION                              #
#                                                                   #
#####################################################################

# Start generating a new graph for each of the number of graph realizations that
# we previously specified.

for graph in range(nGraphRealizations):

    # The accBest and accLast variables, for each model, have a list with a
    # total number of elements equal to the number of graphs we will generate
    # Now, for each graph, we have multiple data realization, so we want, for
    # each graph, to create a list to hold each of those values
    for thisModel in modelList:
        accBest[thisModel][graph] = []
        accLast[thisModel][graph] = []

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    #########
    # GRAPH #
    #########

    # Create graph
    G = graphTools.Graph(graphType, nNodes, graphOptions)
    G.computeGFT() # Compute the eigendecomposition of the stored GSO
    a = np.arange(G.N)
    
    for realization in range(nDataRealizations):

        ############
        # DATASETS #
        ############

        #   Now that we have the list of nodes we are using as sources, then we
        #   can go ahead and generate the datasets.
        data = Utils.dataTools.MultiModalityPrediction(G, nTrain, nValid, nTest, num_timestep, 
                                                    F_t=F_t, pooltype='avg', #'avg', 'selectOne', 'weighted'
                                                    sigmaSpatial=sigmaSpatial, sigmaTemporal=sigmaTemporal,
                                                    rhoSpatial=rhoSpatial, rhoTemporal=rhoTemporal)
        data.astype(torch.float64)

        data.to(device)
    
        #%%##################################################################
        #                                                                   #
        #                    MODELS INITIALIZATION                          #
        #                                                                   #
        #####################################################################

        # This is the dictionary where we store the models (in a model.Model
        # class, that is then passed to training).
        modelsGNN = {}

        # If a new model is to be created, it should be called for here.

        #%%\\\\\\\\\\
        #\\\ MODEL 1: Selection GNN ordered by Degree
        #\\\\\\\\\\\\
        
        if doSelectionGNN:

            thisName = hParamsSelGNNDeg['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering
            S, order = graphTools.permDegree(G.S/np.max(np.diag(G.E)))
            # order is an np.array with the ordering of the nodes with respect
            # to the original GSO (the original GSO is kept in G.S).

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.SelectionGNN(# Graph filtering
                                             hParamsSelGNNDeg['F'],
                                             hParamsSelGNNDeg['K'],
                                             hParamsSelGNNDeg['bias'],
                                             # Nonlinearity
                                             hParamsSelGNNDeg['sigma'],
                                             # Pooling
                                             hParamsSelGNNDeg['N'],
                                             hParamsSelGNNDeg['rho'],
                                             hParamsSelGNNDeg['alpha'],
                                             # MLP
                                             hParamsSelGNNDeg['dimLayersMLP'],
                                             # Structure
                                             S)
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            SelGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = SelGNNDeg

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        #%%\\\\\\\\\\
        #\\\ MODEL 2: GCRNN + MLP
        #\\\\\\\\\\\\
        
        if doGCRNN_MLP:

            thisName = hParamsGCRNN_MLP['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsGCRNN_MLP['inFeatures'],
                                             hParamsGCRNN_MLP['stateFeatures'],
                                             hParamsGCRNN_MLP['inputFilterTaps'],
                                             hParamsGCRNN_MLP['stateFilterTaps'],
                                             hParamsGCRNN_MLP['stateNonlinearity'],
                                             hParamsGCRNN_MLP['outputNonlinearity'],
                                             hParamsGCRNN_MLP['dimLayersMLP'],S,
                                             hParamsGCRNN_MLP['bias'],
                                             hParamsGCRNN_MLP['time_gating'],
                                             hParamsGCRNN_MLP['spatial_gating'],
                                             hParamsGCRNN_MLP['mlpType'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            GCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = GCRNN_MLP

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

 
        #%%\\\\\\\\\\
        #\\\ MODEL 3: RNN + MLP
        #\\\\\\\\\\\\
        
        if doRNN_MLP:

            thisName = hParamsRNN_MLP['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = np.matmul(np.linalg.inv(G.D),G.L)
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.RNNforRegression(hParamsRNN_MLP['inFeatures'],
                                     hParamsRNN_MLP['stateFeatures'],
                                     hParamsRNN_MLP['stateNonlinearity'],
                                     hParamsRNN_MLP['dimLayersMLP'],
                                     hParamsRNN_MLP['outputNonlinearity'],
                                     S,
                                     hParamsRNN_MLP['bias'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            GCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = GCRNN_MLP

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})      

        #%%\\\\\\\\\\
        #\\\ MODEL 4: GCRNN + GNN
        #\\\\\\\\\\\\
        
        if doGCRNN_GNN:

            thisName = hParamsGCRNN_GNN['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsGCRNN_GNN['inFeatures'],
                                             hParamsGCRNN_GNN['stateFeatures'],
                                             hParamsGCRNN_GNN['inputFilterTaps'],
                                             hParamsGCRNN_GNN['stateFilterTaps'],
                                             hParamsGCRNN_GNN['stateNonlinearity'],
                                             hParamsGCRNN_GNN['outputNonlinearity'],
                                             hParamsGCRNN_GNN['dimLayersMLP'],S,
                                             hParamsGCRNN_GNN['bias'],
                                             hParamsGCRNN_GNN['time_gating'],
                                             hParamsGCRNN_GNN['spatial_gating'],
                                             hParamsGCRNN_GNN['mlpType'],
                                             hParamsGCRNN_GNN['finalNonlinearity'], 
                                             hParamsGCRNN_GNN['dimNodeSignals'],
                                             hParamsGCRNN_GNN['nFilterTaps'],
                                             hParamsGCRNN_GNN['nSelectedNodes'],
                                             hParamsGCRNN_GNN['poolingFunction'],
                                             hParamsGCRNN_GNN['poolingSize=None'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            GCRNN_GNN = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = GCRNN_GNN

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

         #%%\\\\\\\\\\
        #\\\ MODEL 5: Time Gated GCRNN + GNN
        #\\\\\\\\\\\\
        
        if doTimeGCRNN_GNN:

            thisName = hParamsTimeGCRNN_GNN['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsTimeGCRNN_GNN['inFeatures'],
                                             hParamsTimeGCRNN_GNN['stateFeatures'],
                                             hParamsTimeGCRNN_GNN['inputFilterTaps'],
                                             hParamsTimeGCRNN_GNN['stateFilterTaps'],
                                             hParamsTimeGCRNN_GNN['stateNonlinearity'],
                                             hParamsTimeGCRNN_GNN['outputNonlinearity'],
                                             hParamsTimeGCRNN_GNN['dimLayersMLP'],S,
                                             hParamsTimeGCRNN_GNN['bias'],
                                             hParamsTimeGCRNN_GNN['time_gating'],
                                             hParamsTimeGCRNN_GNN['spatial_gating'],
                                             hParamsTimeGCRNN_GNN['mlpType'],
                                             hParamsTimeGCRNN_GNN['finalNonlinearity'], 
                                             hParamsTimeGCRNN_GNN['dimNodeSignals'],
                                             hParamsTimeGCRNN_GNN['nFilterTaps'],
                                             hParamsTimeGCRNN_GNN['nSelectedNodes'],
                                             hParamsTimeGCRNN_GNN['poolingFunction'],
                                             hParamsTimeGCRNN_GNN['poolingSize=None'], None,
                                             hParamsTimeGCRNN_GNN['multiModalOutput'], 
                                             hParamsTimeGCRNN_GNN['F_t'], [G.assign_dict])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            TimeGCRNN_GNN = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = TimeGCRNN_GNN

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

    
        #%%\\\\\\\\\\
        #\\\ MODEL 6: Time Gated GCRNN + MLP
        #\\\\\\\\\\\\
        
        if doTimeGCRNN_MLP:

            thisName = hParamsTimeGCRNN_MLP['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsTimeGCRNN_MLP['inFeatures'],
                                             hParamsTimeGCRNN_MLP['stateFeatures'],
                                             hParamsTimeGCRNN_MLP['inputFilterTaps'],
                                             hParamsTimeGCRNN_MLP['stateFilterTaps'],
                                             hParamsTimeGCRNN_MLP['stateNonlinearity'],
                                             hParamsTimeGCRNN_MLP['outputNonlinearity'],
                                             hParamsTimeGCRNN_MLP['dimLayersMLP'],S,
                                             hParamsTimeGCRNN_MLP['bias'],
                                             hParamsTimeGCRNN_MLP['time_gating'],
                                             hParamsTimeGCRNN_MLP['spatial_gating'],
                                             hParamsTimeGCRNN_MLP['mlpType'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            TimeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = TimeGCRNN_MLP

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        #%%\\\\\\\\\\
        #\\\ MODEL 7: Node Gated GCRNN + MLP
        #\\\\\\\\\\\\
        
        if doNodeGCRNN_MLP:

            thisName = hParamsNodeGCRNN_MLP['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsNodeGCRNN_MLP['inFeatures'],
                                             hParamsNodeGCRNN_MLP['stateFeatures'],
                                             hParamsNodeGCRNN_MLP['inputFilterTaps'],
                                             hParamsNodeGCRNN_MLP['stateFilterTaps'],
                                             hParamsNodeGCRNN_MLP['stateNonlinearity'],
                                             hParamsNodeGCRNN_MLP['outputNonlinearity'],
                                             hParamsNodeGCRNN_MLP['dimLayersMLP'],S,
                                             hParamsNodeGCRNN_MLP['bias'],
                                             hParamsNodeGCRNN_MLP['time_gating'],
                                             hParamsNodeGCRNN_MLP['spatial_gating'],
                                             hParamsNodeGCRNN_MLP['mlpType'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            NodeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = NodeGCRNN_MLP

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        #%%\\\\\\\\\\
        #\\\ MODEL 8: Edge Gated GCRNN + MLP
        #\\\\\\\\\\\\
        
        if doEdgeGCRNN_MLP:

            thisName = hParamsEdgeGCRNN_MLP['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsEdgeGCRNN_MLP['inFeatures'],
                                             hParamsEdgeGCRNN_MLP['stateFeatures'],
                                             hParamsEdgeGCRNN_MLP['inputFilterTaps'],
                                             hParamsEdgeGCRNN_MLP['stateFilterTaps'],
                                             hParamsEdgeGCRNN_MLP['stateNonlinearity'],
                                             hParamsEdgeGCRNN_MLP['outputNonlinearity'],
                                             hParamsEdgeGCRNN_MLP['dimLayersMLP'],S,
                                             hParamsEdgeGCRNN_MLP['bias'],
                                             hParamsEdgeGCRNN_MLP['time_gating'],
                                             hParamsEdgeGCRNN_MLP['spatial_gating'],
                                             hParamsEdgeGCRNN_MLP['mlpType'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            EdgeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = EdgeGCRNN_MLP

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        #%%\\\\\\\\\\
        #\\\ MODEL 9: Time Node Gated GCRNN + MLP
        #\\\\\\\\\\\\
        
        if doTimeNodeGCRNN_MLP:

            thisName = hParamsTimeNodeGCRNN_MLP['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsTimeNodeGCRNN_MLP['inFeatures'],
                                             hParamsTimeNodeGCRNN_MLP['stateFeatures'],
                                             hParamsTimeNodeGCRNN_MLP['inputFilterTaps'],
                                             hParamsTimeNodeGCRNN_MLP['stateFilterTaps'],
                                             hParamsTimeNodeGCRNN_MLP['stateNonlinearity'],
                                             hParamsTimeNodeGCRNN_MLP['outputNonlinearity'],
                                             hParamsTimeNodeGCRNN_MLP['dimLayersMLP'],S,
                                             hParamsTimeNodeGCRNN_MLP['bias'],
                                             hParamsTimeNodeGCRNN_MLP['time_gating'],
                                             hParamsTimeNodeGCRNN_MLP['spatial_gating'],
                                             hParamsTimeNodeGCRNN_MLP['mlpType'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            TimeNodeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = TimeNodeGCRNN_MLP

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        #%%\\\\\\\\\\
        #\\\ MODEL 10: Time Edge Gated GCRNN + MLP
        #\\\\\\\\\\\\
        
        if doTimeEdgeGCRNN_MLP:

            thisName = hParamsTimeEdgeGCRNN_MLP['name']

            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization

            ##############
            # PARAMETERS #
            ##############

            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2

            #\\\ Ordering/GSO
            S = G.S/np.max(np.diag(G.E))
            order = np.arange(G.N)

            ################
            # ARCHITECTURE #
            ################

            thisArchit = archit.GatedGCRNNforRegression(hParamsTimeEdgeGCRNN_MLP['inFeatures'],
                                             hParamsTimeEdgeGCRNN_MLP['stateFeatures'],
                                             hParamsTimeEdgeGCRNN_MLP['inputFilterTaps'],
                                             hParamsTimeEdgeGCRNN_MLP['stateFilterTaps'],
                                             hParamsTimeEdgeGCRNN_MLP['stateNonlinearity'],
                                             hParamsTimeEdgeGCRNN_MLP['outputNonlinearity'],
                                             hParamsTimeEdgeGCRNN_MLP['dimLayersMLP'],S,
                                             hParamsTimeEdgeGCRNN_MLP['bias'],
                                             hParamsTimeEdgeGCRNN_MLP['time_gating'],
                                             hParamsTimeEdgeGCRNN_MLP['spatial_gating'],
                                             hParamsTimeEdgeGCRNN_MLP['mlpType'])
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)

            #############
            # OPTIMIZER #
            #############

            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)

            ########
            # LOSS #
            ########

            thisLossFunction = lossFunction # (if different from default, change
                                            # it here)

            #########
            # MODEL #
            #########

            TimeEdgeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)

            modelsGNN[thisName] = TimeEdgeGCRNN_MLP

            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

                
        #%%##################################################################
        #                                                                   #
        #                    TRAINING                                       #
        #                                                                   #
        #####################################################################


        ############
        # TRAINING #
        ############

        # On top of the rest of the training options, we pass the identification
        # of this specific graph/data realization.
        if nGraphRealizations > 1:
            trainingOptions['graphNo'] = graph
        if nDataRealizations > 1:
            trainingOptions['realizationNo'] = realization

        # This is the function that trains the models detailed in the dictionary
        # modelsGNN using the data, with the specified training options.
        assign_dicts = [G.assign_dict]
        train.MultipleModels(modelsGNN, data,
                             nEpochs = nEpochs, batchSize = batchSize, seqLen = seqLen, 
                             F_t = F_t, assign_dicts = assign_dicts,
                             stateFeat = F1, rnnStateFeat = rnnStateFeat, device = device,
                             **trainingOptions)

        #%%##################################################################
        #                                                                   #
        #                    EVALUATION                                     #
        #                                                                   #
        #####################################################################

        # Now that the model has been trained, we evaluate them on the test
        # samples.

        # We have two versions of each model to evaluate: the one obtained
        # at the best result of the validation step, and the last trained model.

        ########
        # DATA #
        ########
        F_len = ((seqLen + 1) // F_t) - 1
        xTest, yTest, x1Test, y1Test = data.getSamples('test')
        # expand xTest (F) along the time dimension
        xTest = xTest.view(nTest, F_len, -1)
        xTest = xTest.repeat_interleave(F_t, dim=1)
        yTest = yTest.view(nTest, F_len, -1)

        # expand x1Test (E) along from cluster signal to graph signal
        if len(assign_dicts) == 1:
            # if all the graphs share a same cluster structure
            assign_dict = assign_dicts[0]
            # here uses F_len*F_t instead of seqLen as it's smaller
            # I choose to cut overall signal into a same length (temporary, #TODO maybe different len)
            x1Test = x1Test.view(nTest, seqLen, -1)[:, :F_len*F_t, :]
            y1Test = y1Test.view(nTest, seqLen, -1)[:, :F_len*F_t, :]
            _x1Test = torch.zeros_like(xTest)
            # _y1Train = torch.zeros_like(yTrain)
            # the last dimension should be cluster number
            assert x1Test.shape[-1] == len(assign_dict) and y1Test.shape[-1] == len(assign_dict)
            for k in range(len(assign_dict)):
                _x1Test[:, :, assign_dict[k]] = x1Test[:, :, k:k+1].repeat(1, 1, len(assign_dict[k]))
            x1Test = _x1Test; del _x1Test
        else:
            #TODO: different graphs (i.e. cluster assignments) for different samples
            pass

        ##############
        # BEST MODEL #
        ##############

        if doPrint:
            print("Total testing loss (Best):", flush = True)

        for key in modelsGNN.keys():
            # Update order and adapt dimensions (this data has one input feature,
            # so we need to add that dimension; make it from B x N to B x F x N)
            xTestOrdered = xTest[:,:,modelsGNN[key].order]
            x1TestOrdered = x1Test[:,:,modelsGNN[key].order]
            if 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                                    'Rnn' in modelsGNN[key].name:
                xTestOrdered = xTestOrdered.unsqueeze(2)
                yTestModel = yTest.unsqueeze(2)
                x1TestOrdered = x1TestOrdered.unsqueeze(2)
                y1TestModel = y1Test.unsqueeze(2)
            else:
                ipdb.set_trace() #check correctness
                xTestOrdered = xTestOrdered.view(nTest*xTestOrdered.shape[1],1,-1)
                yTestModel = yTest.view(nTest*xTestOrdered.shape[1],1,-1)
                x1TestOrdered = x1TestOrdered.view(nTest*x1TestOrdered.shape[1],1,-1)
                y1TestModel = y1Test.view(nTest*x1TestOrdered.shape[1],1,-1)
            
            with torch.no_grad():
                # Process the samples
                if 'GCRNN' in modelsGNN[key].name or 'gcrnn' in modelsGNN[key].name or \
                                                    'GCRnn' in modelsGNN[key].name:
                    h0t = torch.zeros(nTest,F1,nNodes).to(device)
                    yHatTest, y1HatTest = modelsGNN[key].archit(xTestOrdered,h0t)
                    
                elif 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                                    'Rnn' in modelsGNN[key].name:
                    ipdb.set_trace() # TODO: dealing with output
                    h0t = torch.zeros(nTest,rnnStateFeat).to(device)
                    c0t = h0t
                    yHatTest = modelsGNN[key].archit(xTestOrdered,h0t,c0t)
                else:
                    ipdb.set_trace() # TODO: dealing with output
                    yHatTest = modelsGNN[key].archit(xTestOrdered)
                    yHatTest = yHatTest.unsqueeze(1)

                # We compute the accuracy
                thisAccBest = (data.evaluate(yHatTest, yTestModel) \
                            + data.evaluate(y1HatTest, y1TestModel)) / 2

            if doPrint:
                print("%s: %4.4f" % (key, thisAccBest ), flush = True)

            # Save value
            writeVarValues(varsFile,
                       {'accBest%s' % key: thisAccBest})

            # Now check which is the model being trained
            for thisModel in modelList:
                # If the name in the modelList is contained in the name with
                # the key, then that's the model, and save it
                # For example, if 'SelGNNDeg' is in thisModelList, then the
                # correct key will read something like 'SelGNNDegG01R00' so
                # that's the one to save.
                if thisModel in key:
                    accBest[thisModel][graph] += [thisAccBest.item()]
                # This is so that we can later compute a total accuracy with
                # the corresponding error.

        ##############
        # LAST MODEL #
        ##############

        # And repeat for the last model

        if doPrint:
            print("Total testing loss (Last):", flush = True)

        # Update order and adapt dimensions
        for key in modelsGNN.keys():
            modelsGNN[key].load(label = 'Last')
            xTestOrdered = xTest[:,:,modelsGNN[key].order]
            x1TestOrdered = x1Test[:,:,modelsGNN[key].order]
            if 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                                'Rnn' in modelsGNN[key].name:
                xTestOrdered = xTestOrdered.unsqueeze(2)
                yTestModel = yTest.unsqueeze(2)
                x1TestOrdered = x1TestOrdered.unsqueeze(2)
                y1TestModel = y1Test.unsqueeze(2)         
            
            else:
                ipdb.set_trace() #check correctness
                xTestOrdered = xTestOrdered.view(nTest*xTestOrdered.shape[1],1,-1)
                yTestModel = yTest.view(nTest*xTestOrdered.shape[1],1,-1)
                x1TestOrdered = x1TestOrdered.view(nTest*x1TestOrdered.shape[1],1,-1)
                y1TestModel = y1Test.view(nTest*x1TestOrdered.shape[1],1,-1)
            
            with torch.no_grad():
                # Process the samples
                if 'GCRNN' in modelsGNN[key].name or 'gcrnn' in modelsGNN[key].name or \
                                                    'GCRnn' in modelsGNN[key].name:
                    h0t = torch.zeros(nTest,F1,nNodes).to(device)
                    yHatTest, y1HatTest = modelsGNN[key].archit(xTestOrdered,h0t)
                    # plt.figure()
                    # plt.plot(yHatTest[0,:,0,0].cpu().numpy())
                    # plt.plot(yTestModel[0,:,0,0].cpu().numpy())
                    # plt.show()
                elif 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                                    'Rnn' in modelsGNN[key].name:
                    ipdb.set_trace() # TODO: dealing with output
                    h0t = torch.zeros(nTest,rnnStateFeat).to(device)
                    c0t = h0t
                    yHatTest = modelsGNN[key].archit(xTestOrdered,h0t,c0t)
                else:
                    ipdb.set_trace() # TODO: dealing with output
                    yHatTest = modelsGNN[key].archit(xTestOrdered)
                    yHatTest = yHatTest.unsqueeze(1)

                # We compute the accuracy
                thisAccLast = (data.evaluate(yHatTest, yTestModel) \
                            + data.evaluate(y1HatTest, y1TestModel)) / 2

            if doPrint:
                print("%s: %4.4f" % (key, thisAccLast), flush = True)

            # Save values:
            writeVarValues(varsFile,
                       {'accLast%s' % key: thisAccLast})
            # And repeat for the last model:
            for thisModel in modelList:
                if thisModel in key:
                    accLast[thisModel][graph] += [thisAccLast.item()]

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanAccBestPerGraph = {} # Compute the mean accuracy (best) across all
    # realizations data realizations of a graph
meanAccLastPerGraph = {} # Compute the mean accuracy (last) across all
    # realizations data realizations of a graph
meanAccBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanAccLast = {} # Mean across graphs
stdDevAccBest = {} # Standard deviation across graphs
stdDevAccLast = {} # Standard deviation across graphs

if doPrint:
    print("\nFinal evaluations (%02d graphs, %02d realizations)" % (
            nGraphRealizations, nDataRealizations))

for thisModel in modelList:
    # Convert the lists into a nGraphRealizations x nDataRealizations matrix
    accBest[thisModel] = np.array(accBest[thisModel])
    accLast[thisModel] = np.array(accLast[thisModel])

    # Compute the mean (across realizations for a given graph)
    meanAccBestPerGraph[thisModel] = np.mean(accBest[thisModel], axis = 1)
    meanAccLastPerGraph[thisModel] = np.mean(accLast[thisModel], axis = 1)

    # And now compute the statistics (across graphs)
    meanAccBest[thisModel] = np.mean(meanAccBestPerGraph[thisModel])
    meanAccLast[thisModel] = np.mean(meanAccLastPerGraph[thisModel])
    stdDevAccBest[thisModel] = np.std(meanAccBestPerGraph[thisModel])
    stdDevAccLast[thisModel] = np.std(meanAccLastPerGraph[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.4f (+-%6.4f) [Best] %6.4f (+-%6.4f) [Last]" % (
                thisModel,
                meanAccBest[thisModel],
                stdDevAccBest[thisModel],
                meanAccLast[thisModel],
                stdDevAccLast[thisModel]))

    # Save values
    writeVarValues(varsFile,
               {'meanAccBest%s' % thisModel: meanAccBest[thisModel],
                'stdDevAccBest%s' % thisModel: stdDevAccBest[thisModel],
                'meanAccLast%s' % thisModel: meanAccLast[thisModel],
                'stdDevAccLast%s' % thisModel : stdDevAccLast[thisModel]})
