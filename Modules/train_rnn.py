# 2019/01/10~
"""
train.py Training Module

Methods for training the models.

MultipleModels: Handles the training for multiple models simultaneously
"""

import torch
import numpy as np
import os
import pickle
import datetime

from scipy.io import savemat
import ipdb

def MultipleModels(modelsDict, data, nEpochs, batchSize, seqLen, 
                F_t, assign_dicts, stateFeat, rnnStateFeat, **kwargs):
    """
    Trains multiple models simultaneously

    Inputs:

        modelsDict (dict): Dictionary containing the models to be trained (see
            Modules.model.Model class)
        data (class): Data to carry out the training (see Utils.dataTools)
        nEpochs (int): number of epochs (passes over the dataset)
        batchSize (int): size of each minibatch
        seqLen (int): length of input sequence for recurrent architectures
        F_t (int): the time resolution of F signals (have 1 F measurement on how many steps)
        assign_dicts (list of dict): the cluster assignment of sample graphs
        stateFeat (int): number of state features of GCRNN architectures
        rnnStateFeat (int): number of state features of RNN architectures

        Keyword arguments:

        validationInterval (int): interval of training (number of training
            steps) without running a validation stage.

        Optional (keyword) arguments:

        learningRateDecayRate (float): float that multiplies the latest learning
            rate used.
        learningRateDecayPeriod (int): how many training steps before 
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        saveDir (string): path to the directory where to save relevant training
            variables.
        printInterval (int): how many training steps after which to print
            partial results (0 means do not print)
        graphNo (int): keep track of what graph realization this is
        realitizationNo (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model

    Observations:
    - Model parameters for best and last are saved.

    """

    ####################################
    # ARGUMENTS (Store chosen options) #
    ####################################

    # Training Options:
    if 'logger' in kwargs.keys():
        doLogging = True
        logger = kwargs['logger']
    else:
        doLogging = False

    if 'saveDir' in kwargs.keys():
        doSaveVars = True
        saveDir = os.path.join(kwargs['saveDir'],'trainVars')
    else:
        doSaveVars = False

    if 'printInterval' in kwargs.keys():
        doPrint = True
        printInterval = kwargs['printInterval']
    else:
        doPrint = False

    if 'learningRateDecayRate' in kwargs.keys() and \
        'learningRateDecayPeriod' in kwargs.keys():
        doLearningRateDecay = True
        learningRateDecayRate = kwargs['learningRateDecayRate']
        learningRateDecayPeriod = kwargs['learningRateDecayPeriod']
    else:
        doLearningRateDecay = False

    validationInterval = kwargs['validationInterval']

    if 'graphNo' in kwargs.keys():
        graphNo = kwargs['graphNo']

    if 'realizationNo' in kwargs.keys():
        realizationNo = kwargs['realizationNo']

    # No training case:
    if nEpochs == 0:
        doSaveVars = False
        doLogging = False
        # If there's no training happening, there's nothing to report about
        # training losses and stuff.


    ###########################################
    # DATA INPUT (pick up on data parameters) #
    ###########################################

    nTrain = data.nTrain # size of the training set

    # Number of batches: If the desired number of batches does not split the
    # dataset evenly, we reduce the size of the last batch (the number of
    # samples in the last batch).
    # The variable batchSize is a list of length nBatches (number of batches),
    # where each element of the list is a number indicating the size of the
    # corresponding batch.
    if nTrain < batchSize:
        nBatches = 1
        batchSize = [nTrain]
    elif nTrain % batchSize != 0:
        nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
        batchSize = [batchSize] * nBatches
        # If the sum of all batches so far is not the total number of graphs,
        # start taking away samples from the last batch (remember that we used
        # ceiling, so we are overshooting with the estimated number of batches)
        while sum(batchSize) != nTrain:
            batchSize[-1] -= 1
    # If they fit evenly, then just do so.
    else:
        nBatches = np.int(nTrain/batchSize)
        batchSize = [batchSize] * nBatches
    # batchIndex is used to determine the first and last element of each batch.
    # If batchSize is, for example [20,20,20] meaning that there are three
    # batches of size 20 each, then cumsum will give [20,40,60] which determines
    # the last index of each batch: up to 20, from 20 to 40, and from 40 to 60.
    # We add the 0 at the beginning so that batchIndex[b]:batchIndex[b+1] gives
    # the right samples for batch b.
    batchIndex = np.cumsum(batchSize).tolist()
    batchIndex = [0] + batchIndex

    ##############
    # TRAINING   #
    ##############

    # Learning rate scheduler:
    if doLearningRateDecay:
        learningRateScheduler = {}
        for key in modelsDict.keys():
            learningRateScheduler[key] = torch.optim.lr_scheduler.StepLR(
                    modelsDict[key].optim, learningRateDecayPeriod,
                    learningRateDecayRate)

    # Initialize counters (since we give the possibility of early stopping, we
    # had to drop the 'for' and use a 'while' instead):
    epoch = 0 # epoch counter

    #\\\ Save variables to be used for each model
    # Logging variables
    if doLogging:
        lossTrainTB = {}
        evalTrainTB = {}
        lossValidTB = {}
        evalValidTB = {}

    # Training variables of interest
    if doSaveVars:
        lossTrain = {}
        evalTrain = {}
        lossValid = {}
        evalValid = {}
        timeTrain = {}
        timeValid = {}
        for key in modelsDict.keys():
            lossTrain[key] = []
            evalTrain[key] = []
            lossValid[key] = []
            evalValid[key] = []
            timeTrain[key] = []
            timeValid[key] = []

    # Model tracking
    bestScore = {}
    bestEpoch = {}
    bestBatch = {}

    for epoch in range(nEpochs):

        # Randomize dataset for each epoch
        randomPermutation = np.random.permutation(nTrain)
        # Convert a numpy.array of numpy.int into a list of actual int.
        idxEpoch = [int(i) for i in randomPermutation]

        # Learning decay
        if doLearningRateDecay:
            for key in learningRateScheduler.keys():
                learningRateScheduler[key].step()

            if doPrint:
                # All the optimization have the same learning rate, so just
                # print one of them
                # TODO: Actually, they might be different, so I will need to
                # print all of them.
                print("Epoch %d, learning rate = %.8f" % (epoch+1,
                      learningRateScheduler[key].optim.param_groups[0]['lr']))

        # Initialize counter
        batch = 0 # batch counter
        F_len = ((seqLen + 1) // F_t) - 1
        for batch in range(nBatches):

            # Extract the adequate batch
            thisBatchIndices = idxEpoch[batchIndex[batch] : batchIndex[batch+1]]
            # Get the samples: F pairs, E pairs
            xTrain, yTrain, x1Train, y1Train = data.getSamples('train', thisBatchIndices)

            # expand xTrain (F) along the time dimension
            # no need to expand y
            xTrain = xTrain.view(batchSize[batch], F_len, -1)
            xTrain = xTrain.repeat_interleave(F_t, dim=1)
            yTrain = yTrain.view(batchSize[batch], F_len, -1)
            # yTrain = yTrain.repeat_interleave(F_t, dim=1)

            # expand x1Train (E) along from cluster signal to graph signal
            if len(assign_dicts) == 1:
                # if all the graphs share a same cluster structure
                assign_dict = assign_dicts[0]
                # here uses F_len*F_t instead of seqLen as it's smaller
                # I choose to cut overall signal into a same length (temporary, #TODO maybe different len)
                x1Train = x1Train.view(batchSize[batch], seqLen, -1)[:, :F_len*F_t, :]
                y1Train = y1Train.view(batchSize[batch], seqLen, -1)[:, :F_len*F_t, :]
                _x1Train = torch.zeros_like(xTrain)
                # _y1Train = torch.zeros_like(yTrain)
                # the last dimension should be cluster number
                assert x1Train.shape[-1] == len(assign_dict) and y1Train.shape[-1] == len(assign_dict)
                for k in range(len(assign_dict)):
                    _x1Train[:, :, assign_dict[k]] = x1Train[:, :, k:k+1].repeat(1, 1, len(assign_dict[k]))
                    # _y1Train[:, :, assign_dict[k]] = y1Train[:, :, k:k+1].repeat(1, 1, len(assign_dict[k]))
                x1Train = _x1Train; del _x1Train
                # y1Train = _y1Train; del _y1Train
            else:
                #TODO: different graphs (i.e. cluster assignments) for different samples
                pass

            if doPrint and printInterval > 0:
                if (epoch * nBatches + batch) % printInterval == 0:
                    trainPreamble = ''
                    if 'graphNo' in kwargs.keys():
                        trainPreamble += 'G:%02d ' % graphNo
                    if 'realizationNo' in kwargs.keys():
                        trainPreamble += 'R:%02d ' % realizationNo
                    print("[%sTRAINING - E: %2d, B: %3d]" % (
                            trainPreamble, epoch+1, batch+1))

            for key in modelsDict.keys():
                # Set the ordering #ISSUE: why no ordering for y?
                xTrainOrdered = xTrain[:,:,modelsDict[key].order] # B x F x N
                x1TrainOrdered = x1Train[:,:,modelsDict[key].order]
                # Check if it is an RNN to add sequence dimension
                if 'RNN' in modelsDict[key].name or 'rnn' in modelsDict[key].name or \
                                                    'Rnn' in modelsDict[key].name:
                    xTrainOrdered = xTrainOrdered.unsqueeze(2) # To account for just F=1 feature
                    x1TrainOrdered = x1TrainOrdered.unsqueeze(2)
                    yTrainModel = yTrain.unsqueeze(2)
                    y1TrainModel = y1Train.unsqueeze(2)

                else:
                    ipdb.set_trace() #check correctness
                    xTrainOrdered = xTrainOrdered.view(batchSize[batch]*xTrainOrdered.shape[1],1,-1)
                    yTrainModel = yTrain.view(batchSize[batch]*xTrainOrdered.shape[1],1,-1)
                    x1TrainOrdered = x1TrainOrdered.view(batchSize[batch]*x1TrainOrdered.shape[1],1,-1)
                    y1TrainModel = yT1rain.view(batchSize[batch]*x1TrainOrdered.shape[1],1,-1)
                # Start measuring time
                startTime = datetime.datetime.now()

                # Reset gradients
                modelsDict[key].archit.zero_grad()
                
                
                if 'GCRNN' in modelsDict[key].name or 'gcrnn' in modelsDict[key].name or \
                                                    'GCRnn' in modelsDict[key].name:
                    # Obtain the output of the GCRNN
                    h0 = torch.zeros(batchSize[batch],stateFeat,xTrainOrdered.shape[3])
                    signalHatTrain = modelsDict[key].archit(xTrainOrdered, h0)
                    # averaging along t
                    yHatTrain = signalHatTrain.reshape(batchSize[batch], F_len, F_t, 1, -1).mean(2)
                    # averaging output signal for each cluster
                    if len(assign_dicts) == 1:
                        # if all the graphs share a same cluster structure
                        assign_dict = assign_dicts[0]
                        y1HatTrain = []
                        for k in range(len(assign_dict)):
                            y1HatTrain.append(signalHatTrain[:,:,:,assign_dict[k]].mean(3))
                        y1HatTrain = torch.stack(y1HatTrain, dim=len(y1HatTrain[0].shape))
                    else:
                        #TODO: different graphs (i.e. cluster assignments) for different samples
                        pass
                
                elif 'RNN' in modelsDict[key].name or 'rnn' in modelsDict[key].name or \
                                            'Rnn' in modelsDict[key].name:
                    # Obtain the output of the RNN
                    ipdb.set_trace() # TODO: dealing with output
                    h0 = torch.zeros(batchSize[batch],rnnStateFeat)
                    c0 = h0
                    yHatTrain = modelsDict[key].archit(xTrainOrdered, h0, c0)
                else:
                    # Obtain the output of the GNN
                    ipdb.set_trace() # TODO: dealing with output
                    yHatTrain = modelsDict[key].archit(xTrainOrdered)
                    yHatTrain = yHatTrain.unsqueeze(1)

                # Compute loss: F_loss + E_loss (E's y need to be contiguous..)
                lossValueTrain = modelsDict[key].loss(yHatTrain, yTrainModel) \
                                + modelsDict[key].loss(y1HatTrain, y1TrainModel)

                # Compute gradients
                lossValueTrain.backward()

                # Optimize
                modelsDict[key].optim.step()
                
                # Finish measuring time
                endTime = datetime.datetime.now()
                
                timeElapsed = abs(endTime - startTime).total_seconds()

                # Compute the accuracy
                #   Note: Using yHatTrain.data creates a new tensor with the
                #   same value, but detaches it from the gradient, so that no
                #   gradient operation is taken into account here.
                #   (Alternatively, we could use a with torch.no_grad():)
                accTrain = (data.evaluate(yHatTrain.data, yTrainModel) \
                        + data.evaluate(y1HatTrain.data, y1TrainModel)) / 2

                # Logging values
                if doLogging:
                    lossTrainTB[key] = lossValueTrain.item()
                    evalTrainTB[key] = accTrain.item() 
                # Save values
                if doSaveVars:
                    lossTrain[key] += [lossValueTrain.item()]
                    evalTrain[key] += [accTrain.item()]
                    timeTrain[key] += [timeElapsed]                    

                # Print:
                if doPrint and printInterval > 0:
                    if (epoch * nBatches + batch) % printInterval == 0:
                        print("\t(%s) %6.4f / %6.4f - %6.4fs" % (
                                    key, accTrain, lossValueTrain.item()),
                                    timeElapsed)

            #\\\\\\\
            #\\\ TB LOGGING (for each batch)
            #\\\\\\\

            if doLogging:
                modeLoss = 'Loss'
                modeEval = 'Accuracy'
                if 'graphNo' in kwargs.keys():
                    modeLoss += 'G%02d' % graphNo
                    modeEval += 'G%02d' % graphNo
                if 'realizationNo' in kwargs.keys():
                    modeLoss += 'R%02d' % realizationNo
                    modeEval += 'R%02d' % realizationNo
                logger.scalar_summary(mode = 'Training' + modeLoss,
                                      epoch = epoch * nBatches + batch,
                                      **lossTrainTB)
                logger.scalar_summary(mode = 'Training' + modeEval,
                                      epoch = epoch * nBatches + batch,
                                      **evalTrainTB)

            #\\\\\\\
            #\\\ VALIDATION
            #\\\\\\\

            if (epoch * nBatches + batch) % validationInterval == 0:
                # Validation:
                nValid = data.nValid
                xValid, yValid , x1Valid, y1Valid = data.getSamples('valid')
                # expand xValid (F) along the time dimension
                xValid = xValid.view(nValid, F_len, -1)
                xValid = xValid.repeat_interleave(F_t, dim=1)
                yValid = yValid.view(nValid, F_len, -1)

                # expand x1Valid (E) along from cluster signal to graph signal
                if len(assign_dicts) == 1:
                    # if all the graphs share a same cluster structure
                    assign_dict = assign_dicts[0]
                    # here uses F_len*F_t instead of seqLen as it's smaller
                    # I choose to cut overall signal into a same length (temporary, #TODO maybe different len)
                    x1Valid = x1Valid.view(nValid, seqLen, -1)[:, :F_len*F_t, :]
                    y1Valid = y1Valid.view(nValid, seqLen, -1)[:, :F_len*F_t, :]
                    _x1Valid = torch.zeros_like(xValid)
                    # _y1Train = torch.zeros_like(yTrain)
                    # the last dimension should be cluster number
                    assert x1Valid.shape[-1] == len(assign_dict) and y1Valid.shape[-1] == len(assign_dict)
                    for k in range(len(assign_dict)):
                        _x1Valid[:, :, assign_dict[k]] = x1Valid[:, :, k:k+1].repeat(1, 1, len(assign_dict[k]))
                    x1Valid = _x1Valid; del _x1Valid
                else:
                    #TODO: different graphs (i.e. cluster assignments) for different samples
                    pass

                if doPrint:
                    validPreamble = ''
                    if 'graphNo' in kwargs.keys():
                        validPreamble += 'G:%02d ' % graphNo
                    if 'realizationNo' in kwargs.keys():
                        validPreamble += 'R:%02d ' % realizationNo
                    print("[%sVALIDATION - E: %2d, B: %3d]" % (
                            validPreamble, epoch+1, batch+1))

                for key in modelsDict.keys():
                    # Set the ordering
                    xValidOrdered = xValid[:,:,modelsDict[key].order] # BxFxN
                    x1ValidOrdered = x1Valid[:,:,modelsDict[key].order]
                    if 'RNN' in modelsDict[key].name or 'rnn' in modelsDict[key].name or \
                                                    'Rnn' in modelsDict[key].name:
                        xValidOrdered = xValidOrdered.unsqueeze(2)
                        x1ValidOrdered = x1ValidOrdered.unsqueeze(2)
                        yValidModel = yValid.unsqueeze(2)
                        y1ValidModel = y1Valid.unsqueeze(2)
                    else:
                        ipdb.set_trace() #check correctness
                        xValidOrdered = xValidOrdered.view(nValid*xValidOrdered.shape[1],1,-1)
                        yValidModel = yValid.view(nValid*xValidOrdered.shape[1],1,-1)
                        x1ValidOrdered = x1ValidOrdered.view(nValid*x1ValidOrdered.shape[1],1,-1)
                        y1ValidModel = y1Valid.view(nValid*x1ValidOrdered.shape[1],1,-1)
                    # Start measuring time
                    startTime = datetime.datetime.now()
                    
                    # Under torch.no_grad() so that the computations carried out
                    # to obtain the validation accuracy are not taken into
                    # account to update the learnable parameters.
                    with torch.no_grad():
                        # Obtain the output of the GNN
                        if 'GCRNN' in modelsDict[key].name or 'gcrnn' in modelsDict[key].name or \
                                                    'GCRnn' in modelsDict[key].name:
                            h0v = torch.zeros(nValid,stateFeat,xValidOrdered.shape[3])
                            signalHatValid = modelsDict[key].archit(xValidOrdered,h0v)
                            # averaging output signal along t
                            yHatValid = signalHatValid.reshape(nValid, F_len, F_t, 1, -1).mean(2)
                            # averaging output signal for each cluster
                            if len(assign_dicts) == 1:
                                # if all the graphs share a same cluster structure
                                assign_dict = assign_dicts[0]
                                y1HatValid = []
                                for k in range(len(assign_dict)):
                                    y1HatValid.append(signalHatValid[:,:,:,assign_dict[k]].mean(3))
                                y1HatValid = torch.stack(y1HatValid, dim=len(y1HatValid[0].shape))
                            else:
                                #TODO: different graphs (i.e. cluster assignments) for different samples
                                pass

                        elif 'RNN' in modelsDict[key].name or 'rnn' in modelsDict[key].name or \
                                                    'Rnn' in modelsDict[key].name:
                            # Obtain the output of the GCRNN
                            ipdb.set_trace() # TODO: dealing with output
                            h0v = torch.zeros(nValid,rnnStateFeat)
                            c0v = h0
                            yHatValid = modelsDict[key].archit(xValidOrdered,h0v,c0v)
                        else:
                            ipdb.set_trace() # TODO: dealing with output
                            yHatValid = modelsDict[key].archit(xValidOrdered)
                            yHatValid = yHatValid.unsqueeze(1)

                        # Compute loss: F_loss + E_loss (E's y need to be contiguous..)
                        lossValueValid = modelsDict[key].loss(yHatValid, yValidModel) \
                                        + modelsDict[key].loss(y1HatValid, y1ValidModel)

                        # Finish measuring time
                        endTime = datetime.datetime.now()
                        
                        timeElapsed = abs(endTime - startTime).total_seconds()

                        # Compute accuracy:
                        accValid = (data.evaluate(yHatValid, yValidModel) \
                                + data.evaluate(y1HatValid, y1ValidModel)) / 2

                        # Logging values
                        if doLogging:
                            lossValidTB[key] = lossValueValid.item()
                            evalValidTB[key] = accValid.item()
                        # Save values
                        if doSaveVars:
                            lossValid[key] += [lossValueValid.item()]
                            evalValid[key] += [accValid.item()]
                            timeValid[key] += [timeElapsed]

                        # Print:
                        if doPrint:
                            print("\t(%s) %6.4f / %6.4f - %6.4fs" % (
                                    key, accValid , lossValueValid.item(),
                                    timeElapsed))

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        bestScore[key] = accValid
                        bestEpoch[key], bestBatch[key] = epoch, batch
                        # Save this model as the best (so far)
                        modelsDict[key].save(label = 'Best')
                        # Store the keys of the best models when they happen
                        keyBest = []
                    else:
                        thisValidScore = accValid
                        if thisValidScore < bestScore[key]:
                            bestScore[key] = thisValidScore
                            bestEpoch[key], bestBatch[key] = epoch, batch
                            if doPrint:
                                keyBest += [key]
                            modelsDict[key].save(label = 'Best')

                if doPrint:
                    if len(keyBest) > 0:
                        for key in keyBest:
                            print("\t=> New best achieved for %s: %.4f" % \
                                              (key, bestScore[key]))
                        keyBest = []

                if doLogging:
                    logger.scalar_summary(mode = 'Validation' + modeLoss,
                                          epoch = epoch * nBatches + batch,
                                          **lossValidTB)
                    logger.scalar_summary(mode = 'Validation' + modeEval,
                                          epoch = epoch * nBatches + batch,
                                          **evalValidTB)

            #\\\\\\\
            #\\\ END OF BATCH:
            #\\\\\\\

            #\\\ Increase batch count:
            batch += 1

        #\\\\\\\
        #\\\ END OF EPOCH:
        #\\\\\\\

        #\\\ Save models:
        for key in modelsDict.keys():
            modelsDict[key].save(label = 'Last')

        #\\\ Increase epoch count:
        epoch += 1

    #################
    # TRAINING OVER #
    #################

    if doSaveVars:
        # We convert the lists into np.arrays to be handled by both Matlab(R)
        # and matplotlib
        for key in modelsDict.keys():
            lossTrain[key] = np.array(lossTrain[key])
            evalTrain[key] = np.array(evalTrain[key])
            timeTrain[key] = np.array(timeTrain[key])
            lossValid[key] = np.array(lossValid[key])
            evalValid[key] = np.array(evalValid[key])
            timeValid[key] = np.array(timeValid[key])
        # And we would like to save all the relevant information from training
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        # Dictionaries of variables to save
        varsMatlab = {}
        varsPickle = {}
        # And let's start with pickle
        varsPickle['nEpochs'] = nEpochs
        varsPickle['nBatches'] = nBatches
        varsPickle['validationInterval'] = nBatches
        varsPickle['batchSize'] = np.array(batchSize)
        varsPickle['batchIndex'] = np.array(batchIndex)
        varsPickle['lossTrain'] = lossTrain
        varsPickle['evalTrain'] = evalTrain
        varsPickle['timeTrain'] = timeTrain
        varsPickle['lossValid'] = lossValid
        varsPickle['evalValid'] = evalValid
        varsPickle['timeValid'] = timeValid
        # Create file for pickling
        varsFilename = 'trainVars'
        # And add the information if this is a specific realization run
        if 'graphNo' in kwargs.keys():
            varsFilename += 'G%02d' % graphNo
            varsPickle['graphNo'] = graphNo
            varsMatlab['graphNo'] = graphNo
        if 'realizationNo' in kwargs.keys():
            varsFilename += 'R%02d' % realizationNo
            varsPickle['realizationNo'] = realizationNo
            varsMatlab['realizationNo'] = realizationNo
        # Create the file
        pathToFile = os.path.join(saveDir, varsFilename + '.pkl')
        # Open and save it
        with open(pathToFile, 'wb') as trainVarsFile:
            pickle.dump(varsPickle, trainVarsFile)
        # And because of the SP background, why not save it in matlab too?
        pathToMat = os.path.join(saveDir, varsFilename + '.mat')
        varsMatlab['nEpochs'] = nEpochs
        varsMatlab['nBatches'] = nBatches
        varsMatlab['validationInterval'] = nBatches
        varsMatlab['batchSize'] = np.array(batchSize)
        varsMatlab['batchIndex'] = np.array(batchIndex)
        for key in modelsDict.keys():
            varsMatlab['lossTrain' + key] = lossTrain[key]
            varsMatlab['evalTrain' + key] = evalTrain[key]
            varsMatlab['timeTrain' + key] = timeTrain[key]
            varsMatlab['lossValid' + key] = lossValid[key]
            varsMatlab['evalValid' + key] = evalValid[key]
            varsMatlab['timeValid' + key] = timeValid[key]
        savemat(pathToMat, varsMatlab)

    # Now, if we didn't do any training (i.e. nEpochs = 0), then the last is
    # also the best.
    if nEpochs == 0:
        for key in modelsDict.keys():
            modelsDict[key].save(label = 'Best')
            modelsDict[key].save(label = 'Last')
        if doPrint:
            print("WARNING: No training. Best and Last models are the same.")

    # After training is done, reload best model before proceeding to evaluation:
    for key in modelsDict.keys():
        modelsDict[key].load(label = 'Best')

    #\\\ Print out best:
    if doPrint and nEpochs > 0:
        for key in modelsDict.keys():
            print("=> Best validation achieved for %s (E: %2d, B: %2d): %.4f" %(
                   key, bestEpoch[key] + 1, bestBatch[key] + 1, bestScore[key]))
