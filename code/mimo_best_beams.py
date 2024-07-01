# Script context use	: This script is for RAYMOBTIME purposes uses
# Author       		: Ailton Oliveira - Msc Degree (in progress) - PPGEE - UFPA
# Email          	: ailtonpoliveira01@gmail.com
################################################################################

'''
Read channel information (rays organized as npz files) and output the complex-valued
equivalent channels (not anymore only the indices of best pair of beams, which can
be calculated with the generated data).
'''
import numpy as np
import os

from builtins import print

from mimo_channels import getNarrowBandULAMIMOChannel, getDFTOperatedChannel
import csv
import h5py
from math import ceil
import random

random.seed(0)

#def processBeamsOutput(data_folder='.', dataset='.'):
def processBeamsOutput(csvFile, num_episodes, outputFolder, inputPath):
    print ('Generating Beams ...')
    # file generated with ak_generateInSitePlusSumoList.py:
    # need to use both LOS and NLOS here, cannot use restricted list because script does a loop over all scenes
    #insiteCSVFile = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    insiteCSVFile = csvFile
    #numEpisodes = 2086  # total number of episodes
    numEpisodes = num_episodes

    #outputFolder = '../data/beams_output/dataset_s008/beam_output_generate_rt_s008/'
    outputFolder = outputFolder
    # parameters that are typically not changed
    #inputPath = dataset + '/' + 'ray_tracing_data_s008_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    inputPath = inputPath
    #inputPath = '../data/beams_output/dataset_s008/beam_output_generate_rt_s008/ray_tracing_data_s008_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    normalizedAntDistance = 0.5
    angleWithArrayNormal = 0  # use 0 when the angles are provided by InSite

    # ULA
    number_Tx_antennas = 32
    number_Rx_antennas = 8

    if not os.path.exists (outputFolder):
        os.makedirs (outputFolder)

    # initialize variables
    numOfValidChannels = 0
    numOfInvalidChannels = 0
    numLOS = 0
    numNLOS = 0
    count = 0

    '''use dictionary taking the episode, scene and Rx number of file with rows e.g.:
    0,0,0,flow11.0,Car,753.83094753535,649.05232524135,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=0
    0,0,2,flow2.0,Car,753.8198286576,507.38595866735,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=1
    0,0,3,flow2.1,Car,749.7071175056,566.1905128583,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=1'''

    with open (insiteCSVFile, 'r') as f:
        insiteReader = csv.DictReader (f)
        insiteDictionary = {}
        numExamples = 0
        for row in insiteReader:
            isValid = row ['Val']  # V or I are the first element of the list thisLine
            if isValid == 'V':  # filter the valid channels
                numExamples += 1
                thisKey = str (row ['EpisodeID']) + ',' + str (row ['SceneID']) + ',' + str (row ['VehicleArrayID'])
                insiteDictionary [thisKey] = row
        lastEpisode = int (row ['EpisodeID'])

    allOutputs = np.nan * np.ones((numExamples, number_Rx_antennas, number_Tx_antennas), np.complex128)

    for e in range (numEpisodes):
        print ("Episode # ", e)
        b = h5py.File (inputPath + str (e) + '.hdf5', 'r')
        allEpisodeData = b.get('allEpisodeData')
        numScenes = allEpisodeData.shape[0]
        numReceivers = allEpisodeData.shape[1]
        # store the position (x,y,z), 4 angles of strongest (first) ray and LOS or not
        receiverPositions = np.nan * np.ones ((numScenes, numReceivers, 8), np.float32)
        # store two integers converted to 1
        episodeOutputs = np.nan * np.ones((numScenes, numReceivers, number_Rx_antennas, number_Tx_antennas),
                                           np.float32)

        for s in range (numScenes):
            print ("Scene # ", s)
            for r in range (numReceivers):  # 10
                insiteData = allEpisodeData [s, r, :, :]

                # if insiteData corresponds to an invalid channel, all its values will be NaN.
                # We check for that below
                numNaNsInThisChannel = sum(np.isnan(insiteData.flatten()))
                if numNaNsInThisChannel == np.prod(insiteData.shape):
                    numOfInvalidChannels += 1
                    continue  # next Tx / Rx pair
                thisKey = str(e) + ',' + str(s) + ',' + str(r)

                try:
                    thisInSiteLine = list(insiteDictionary[thisKey].items())  # recover from dic
                except KeyError:
                    print ('Could not find in dictionary the key: ', thisKey)
                    print ('Verify file', insiteCSVFile)
                    exit (-1)
                # tokens = thisInSiteLine.split(',')
                # new_data = data[~np.isnan(data)] ; new_data = data[np.isfinite(data)]
                if numNaNsInThisChannel > 0:
                    numOfValidRays = int(thisInSiteLine[8][1])  # number of rays is in 9-th position in CSV list
                    # I could simply use
                    # insiteData = insiteData[0:numOfValidRays]
                    # given the NaN are in the last rows, but to be safe given that did not check, I will go for a slower solution
                    insiteDataTemp = np.zeros ((numOfValidRays, insiteData.shape [1]))
                    numMaxRays = insiteData.shape [0]
                    validRayCounter = 0
                    for itemp in range(numMaxRays):
                        if sum (np.isnan (insiteData [itemp].flatten ())) == 1:  # if insite version 3.2, else use 0
                            insiteDataTemp [validRayCounter] = insiteData [itemp]
                            validRayCounter += 1
                    insiteData = insiteDataTemp  # replace by smaller array without NaN
                receiverPositions[s, r, 0:3] = np.array(
                    [thisInSiteLine[5][1], thisInSiteLine[6][1], thisInSiteLine[7][1]])

                numOfValidChannels += 1
                gain_in_dB = insiteData[:, 0]
                timeOfArrival = insiteData[:, 1]
                # InSite provides angles in degrees. Convert to radians
                # This conversion is being done within the channel function
                AoD_el = insiteData[:, 2]
                AoD_az = insiteData[:, 3]
                AoA_el = insiteData[:, 4]
                AoA_az = insiteData[:, 5]
                RxAngle = insiteData[:, 8] [0]
                RxAngle = RxAngle + 90.0
                if RxAngle > 360.0:
                    RxAngle = RxAngle - 360.0
                # Correct ULA with Rx orientation
                AoA_az = - RxAngle + AoA_az  # angle_new = - delta_axis + angle_wi;

                # first ray is the strongest, store its angles
                receiverPositions [s, r, 3] = AoD_el[0]
                receiverPositions [s, r, 4] = AoD_az[0]
                receiverPositions [s, r, 5] = AoA_el[0]
                receiverPositions [s, r, 6] = AoA_az[0]

                isLOSperRay = insiteData[:, 6]
                pathPhases = insiteData[:, 7]

                # in case any of the rays in LOS, then indicate that the output is 1
                isLOS = 0  # for the channel
                if np.sum (isLOSperRay) > 0:
                    isLOS = 1
                    numLOS += 1
                else:
                    numNLOS += 1
                receiverPositions [s, r, 7] = isLOS
                mimoChannel = getNarrowBandULAMIMOChannel (AoD_az, AoA_az, gain_in_dB, number_Tx_antennas,
                                                           number_Rx_antennas, normalizedAntDistance,
                                                           angleWithArrayNormal)
                equivalentChannel = getDFTOperatedChannel (mimoChannel, number_Tx_antennas, number_Rx_antennas)
                equivalentChannelMagnitude = np.abs (equivalentChannel)
                episodeOutputs [s, r] = np.abs (equivalentChannel)
                allOutputs [count] = episodeOutputs [s, r]
                count += 1

            # finished processing this episode
        # Save beam per episode
        '''
        npz_name = outputFolder + 'beams_output' + '_positions_e_' + str(e) + '.npz'
        np.savez(npz_name, episodeOutputs=episodeOutputs)
        np.savez(npz_name, receiverPositions=receiverPositions)
        print('Saved file ', npz_name) '''

    npz_name_train = outputFolder + f'beams_output_{number_Rx_antennas}x{number_Tx_antennas}' + '.npz'
    np.savez (npz_name_train, output_classification=allOutputs)
    print ('Saved file ', npz_name_train)

    print ('Sanity check (must be 0 NaN) sum of isNaN = ', np.sum (np.isnan (allOutputs [:])))
    print ('total numOfInvalidChannels = ', numOfInvalidChannels)
    print ('total numOfValidChannels = ', numOfValidChannels)
    print ('Sum = ', numOfValidChannels + numOfInvalidChannels)

    print ('total numNLOS = ', numNLOS)
    print ('total numLOS = ', numLOS)
    print ('Sum = ', numLOS + numNLOS)


#if __name__ == '__main__':
#    processBeamsOutput ()