# ----------------------------------------------------------------------
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# Copyright (C) 2019, @breznak
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np
import math

# htm.core imports
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood

from collections import deque

parameters_best = {
    'anomaly': {
        'likelihood': {
            'probationaryPct': 0.10793172183908652,
            'reestimationPeriod': 100
        }
    },
    'enc': {
        'time': {
            'timeOfDay': (21, 6.456740123240503)
        },
        'value': {
            'activeBits': 23,
            'size': 400,
            'seed': 0,
        }
    },
    'sp': {
        'boostStrength': 0.0,
        "wrapAround": True,
        'columnDimensions': 1487,
        'dutyCyclePeriod': 1017,
        'minPctOverlapDutyCycle': 0.0009087943213583929,
        'localAreaDensity': 0,
        'numActiveColumnsPerInhArea': 40,
        'potentialPct': 0.9281708146689587,
        "globalInhibition": True,
        'stimulusThreshold': 0,
        'synPermActiveInc': 0.003892649892638879,
        'synPermConnected': 0.22110323252238637,
        'synPermInactiveDec': 0.0006151856346474387,
        'seed': 0,
    },
    'spatial_tolerance': 0.04115653095415344,
    'tm': {
        'activationThreshold': 14,
        'cellsPerColumn': 32,
        'connectedPermanence': 0.43392460530288607,
        'initialPermanence': 0.2396689292225759,
        'maxNewSynapseCount': 27,
        'maxSegmentsPerCell': 161,
        'maxSynapsesPerSegment': 141,
        'minThreshold': 13,
        'permanenceDecrement': 0.008404653537413292,
        'permanenceIncrement': 0.046393736556088694,
        'predictedSegmentDecrement': 0.0009973866301803873,
        'seed': 0,
    }
}


class UnivHTMDetector(object):
    """
    This detector uses an HTM based anomaly detection technique.
    """

    def __init__(self, name, probationaryPeriod, smoothingKernelSize, htmParams=None, verbose=False):
        self.useSpatialAnomaly = True
        self.verbose = verbose
        self.name = name  # for logging

        self.probationaryPeriod = probationaryPeriod
        self.parameters = parameters_best

        self.minVal = None
        self.maxVal = None
        self.spatial_tolerance = None
        self.encTimestamp = None
        self.encValue = None
        self.sp = None
        self.tm = None
        self.anomalyLikelihood = None

        # optional debug info
        self.enc_info = None
        self.sp_info = None
        self.tm_info = None

        # for initialization
        self.init_data = []
        self.is_initialized = False
        self.iteration_ = 0

        # for smoothing with gaussian
        self.historic_raw_anomaly_scores = deque(maxlen=smoothingKernelSize)
        self.kernel = None
        self.learningPeriod = None

    def initialize(self, input_min=0, input_max=0):
        # setup spatial anomaly
        if self.useSpatialAnomaly:
            self.spatial_tolerance = self.parameters["spatial_tolerance"]

        ## setup Enc, SP, TM
        # Make the Encoders.  These will convert input data into binary representations.
        self.encTimestamp = DateEncoder(timeOfDay=self.parameters["enc"]["time"]["timeOfDay"])

        scalarEncoderParams = RDSE_Parameters()
        scalarEncoderParams.size = self.parameters["enc"]["value"]["size"]
        scalarEncoderParams.activeBits = self.parameters["enc"]["value"]["activeBits"]
        scalarEncoderParams.resolution = max(0.001, (input_max - input_min) / 130)
        scalarEncoderParams.seed = self.parameters["enc"]["value"]["seed"]

        self.encValue = RDSE(scalarEncoderParams)
        encodingWidth = (self.encTimestamp.size + self.encValue.size)
        self.enc_info = Metrics([encodingWidth], 999999999)

        # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
        # SpatialPooler
        spParams = self.parameters["sp"]
        self.sp = SpatialPooler(
            inputDimensions=(encodingWidth,),
            columnDimensions=(spParams["columnDimensions"],),
            potentialRadius=encodingWidth,
            potentialPct=spParams["potentialPct"],
            globalInhibition=spParams["globalInhibition"],
            localAreaDensity=spParams["localAreaDensity"],
            numActiveColumnsPerInhArea=spParams["numActiveColumnsPerInhArea"],
            stimulusThreshold=spParams["stimulusThreshold"],
            synPermInactiveDec=spParams["synPermInactiveDec"],
            synPermActiveInc=spParams["synPermActiveInc"],
            synPermConnected=spParams["synPermConnected"],
            boostStrength=spParams["boostStrength"],
            wrapAround=spParams["wrapAround"],
            minPctOverlapDutyCycle=spParams["minPctOverlapDutyCycle"],
            dutyCyclePeriod=spParams["dutyCyclePeriod"],
            seed=spParams["seed"],
        )
        self.sp_info = Metrics(self.sp.getColumnDimensions(), 999999999)

        # TemporalMemory
        tmParams = self.parameters["tm"]
        self.tm = TemporalMemory(
            columnDimensions=(spParams["columnDimensions"],),
            cellsPerColumn=tmParams["cellsPerColumn"],
            activationThreshold=tmParams["activationThreshold"],
            initialPermanence=tmParams["initialPermanence"],
            connectedPermanence=tmParams["connectedPermanence"],
            minThreshold=tmParams["minThreshold"],
            maxNewSynapseCount=tmParams["maxNewSynapseCount"],
            permanenceIncrement=tmParams["permanenceIncrement"],
            permanenceDecrement=tmParams["permanenceDecrement"],
            predictedSegmentDecrement=tmParams["predictedSegmentDecrement"],
            maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
            maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"],
            seed=tmParams["seed"]
        )
        self.tm_info = Metrics([self.tm.numberOfCells()], 999999999)

        anParams = self.parameters["anomaly"]["likelihood"]
        self.learningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
        self.anomalyLikelihood = AnomalyLikelihood(
            learningPeriod=self.learningPeriod,
            estimationSamples=self.probationaryPeriod - self.learningPeriod,
            reestimationPeriod=anParams["reestimationPeriod"])

        self.kernel = self._gauss_kernel(self.historic_raw_anomaly_scores.maxlen,
                                         self.historic_raw_anomaly_scores.maxlen)

    def modelRun(self, ts, val):
        """
           Run a single pass through HTM model
           @config ts - Timestamp
           @config val - float input value
           @return rawAnomalyScore computed for the `val` in this step
        """
        self.iteration_ += 1

        # 0. During the probation period, gather the data and return 0.01.
        if self.iteration_ <= self.probationaryPeriod:
            self.init_data.append((ts, val))
            return 0.01

        if self.is_initialized is False:
            if self.verbose:
                print("[{}] Initializing".format(self.name))
            temp_iteration = self.iteration_
            vals = [i[1] for i in self.init_data]
            self.initialize(input_min=min(vals), input_max=max(vals))
            self.is_initialized = True
            for ts, val in self.init_data:
                self.modelRun(ts, val)
            self.iteration_ = temp_iteration
            if self.verbose:
                print("[{}] Initialization done".format(self.name))


        ## run data through our model pipeline: enc -> SP -> TM -> Anomaly
        # 1. Encoding
        # Call the encoders to create bit representations for each value.  These are SDR objects.
        dateBits = self.encTimestamp.encode(ts)
        valueBits = self.encValue.encode(float(val))
        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR(self.encTimestamp.size + self.encValue.size).concatenate([valueBits, dateBits])
        self.enc_info.addData(encoding)

        # 2. Spatial Pooler
        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        activeColumns = SDR(self.sp.getColumnDimensions())
        # Execute Spatial Pooling algorithm over input space.
        self.sp.compute(encoding, True, activeColumns)
        self.sp_info.addData(activeColumns)

        # 3. Temporal Memory
        # Execute Temporal Memory algorithm over active mini-columns.
        self.tm.compute(activeColumns, learn=True)
        self.tm_info.addData(self.tm.getActiveCells().flatten())

        # 4. Anomaly
        # handle spatial, contextual (raw, likelihood) anomalies
        # -Spatial
        spatialAnomaly = 0.0
        if self.useSpatialAnomaly:
            # Update min/max values and check if there is a spatial anomaly
            if self.minVal != self.maxVal:
                tolerance = (self.maxVal - self.minVal) * self.spatial_tolerance
                maxExpected = self.maxVal + tolerance
                minExpected = self.minVal - tolerance
                if val > maxExpected or val < minExpected:
                    spatialAnomaly = 1.0
            if self.maxVal is None or val > self.maxVal:
                self.maxVal = val
            if self.minVal is None or val < self.minVal:
                self.minVal = val

        # -Temporal
        raw = self.tm.anomaly
        like = self.anomalyLikelihood.anomalyProbability(val, raw, ts)
        logScore = self.anomalyLikelihood.computeLogLikelihood(like)
        temporalAnomaly = logScore

        anomalyScore = max(spatialAnomaly, temporalAnomaly)  # this is the "main" anomaly, compared in NAB

        # 5. Apply smoothing
        self.historic_raw_anomaly_scores.append(anomalyScore)
        historic_scores = np.asarray(self.historic_raw_anomaly_scores)
        convolved = np.convolve(historic_scores, self.kernel, 'valid')
        anomalyScore = convolved[-1]

        return anomalyScore

    @staticmethod
    def estimateNormal(sampleData, performLowerBoundCheck=True):
        """
        :param sampleData:
        :type sampleData: Numpy array.
        :param performLowerBoundCheck:
        :type performLowerBoundCheck: bool
        :returns: A dict containing the parameters of a normal distribution based on
            the ``sampleData``.
        """
        mean = np.mean(sampleData)
        variance = np.var(sampleData)
        st_dev = 0

        if performLowerBoundCheck:
            # Handle edge case of almost no deviations and super low anomaly scores. We
            # find that such low anomaly means can happen, but then the slightest blip
            # of anomaly score can cause the likelihood to jump up to red.
            if mean < 0.03:
                mean = 0.03

            # Catch all for super low variance to handle numerical precision issues
            if variance < 0.0003:
                variance = 0.0003

        # Compute standard deviation
        if variance > 0:
            st_dev = math.sqrt(variance)

        return mean, variance, st_dev

    @staticmethod
    def _calcSkipRecords(numIngested, windowSize, learningPeriod):
        """Return the value of skipRecords for passing to estimateAnomalyLikelihoods

        If `windowSize` is very large (bigger than the amount of data) then this
        could just return `learningPeriod`. But when some values have fallen out of
        the historical sliding window of anomaly records, then we have to take those
        into account as well so we return the `learningPeriod` minus the number
        shifted out.

        :param numIngested - (int) number of data points that have been added to the
          sliding window of historical data points.
        :param windowSize - (int) size of sliding window of historical data points.
        :param learningPeriod - (int) the number of iterations required for the
          algorithm to learn the basic patterns in the dataset and for the anomaly
          score to 'settle down'.
        """
        numShiftedOut = max(0, numIngested - windowSize)
        return min(numIngested, max(0, learningPeriod - numShiftedOut))

    @staticmethod
    def _gauss_kernel(std, size):
        def _norm_pdf(x, mean, sd):
            var = float(sd) ** 2
            denom = (2 * math.pi * var) ** .5
            num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
            return num / denom

        kernel = [2 * _norm_pdf(idx, 0, std) for idx in list(range(-size + 1, 1))]
        kernel = np.array(kernel)
        kernel = np.flip(kernel)
        kernel = kernel / sum(kernel)
        return kernel
