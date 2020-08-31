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


class MultivHTMDetector(object):
    """
    This detector uses an HTM based anomaly detection technique.
    """

    def __init__(self, name, probationaryPeriod, params=None, verbose=False):
        self.verbose = verbose
        self.name = name  # for logging

        self.probationaryPeriod = probationaryPeriod

        if params is not None:
            self.parameters = params
        else:
            self.parameters = parameters_best

        self.encTimestamp = None
        self.scalar_encoders = []
        self.enc_width = None
        self.sp = None
        self.tm = None
        self.anomalyLikelihood = None

        # for initialization
        self.init_data = []
        self.init_min_max = {}
        self.is_initialized = False
        self.iteration_ = 0

        self.learningPeriod = None

    def _createRDSE(self, min_val=0, max_val=0):
        scalarEncoderParams = RDSE_Parameters()
        scalarEncoderParams.size = self.parameters["enc"]["value"]["size"]
        scalarEncoderParams.activeBits = self.parameters["enc"]["value"]["activeBits"]
        scalarEncoderParams.resolution = max(0.001, (max_val - min_val) / 130)
        scalarEncoderParams.seed = self.parameters["enc"]["value"]["seed"]
        return RDSE(scalarEncoderParams)

    def initialize(self):
        # Setup Encoders
        self.encTimestamp = DateEncoder(timeOfDay=self.parameters["enc"]["time"]["timeOfDay"])
        for idx, (key, value) in enumerate(self.init_min_max.items()):
            self.scalar_encoders.append({
                'idx': idx,
                'name': key,
                'encoder':  self._createRDSE(min_val=value['min'], max_val=value['max'])
            })

        self.enc_width = self.encTimestamp.size + sum([enc_info.get('encoder').size for enc_info in self.scalar_encoders])

        # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
        # SpatialPooler
        spParams = self.parameters["sp"]
        self.sp = SpatialPooler(
            inputDimensions=(self.enc_width,),
            columnDimensions=(spParams["columnDimensions"],),
            potentialRadius=self.enc_width,
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

        anParams = self.parameters["anomaly"]["likelihood"]
        self.learningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
        self.anomalyLikelihood = AnomalyLikelihood(
            learningPeriod=self.learningPeriod,
            estimationSamples=self.probationaryPeriod - self.learningPeriod,
            reestimationPeriod=anParams["reestimationPeriod"])

    def modelRun(self, ts, data):
        """
           Run a single pass through HTM model
           @config ts - Timestamp
           @config val - float input value
           @return rawAnomalyScore computed for the `val` in this step
        """
        self.iteration_ += 1

        # 0. During the probation period, gather the data and return 0.01.
        if self.iteration_ <= self.probationaryPeriod:
            self.init_data.append((ts, data))
            for col_name, val in data.items():
                if val is None:
                    continue
                if col_name not in self.init_min_max:
                    self.init_min_max[col_name] = {}
                if 'min' not in self.init_min_max[col_name] or val < self.init_min_max[col_name]['min']:
                    self.init_min_max[col_name]['min'] = val
                if 'max' not in self.init_min_max[col_name] or val > self.init_min_max[col_name]['max']:
                    self.init_min_max[col_name]['max'] = val
            return 0.01, 0.01

        if self.is_initialized is False:
            if self.verbose:
                print("[{}] Initializing".format(self.name))
            temp_iteration = self.iteration_
            self.initialize()
            self.is_initialized = True
            for ts, data in self.init_data:
                self.modelRun(ts, data)
            self.iteration_ = temp_iteration
            if self.verbose:
                print("[{}] Initialization done".format(self.name))

        # run data through model pipeline: enc -> SP -> TM -> Anomaly
        # 1. Encoding
        # Call the encoders to create bit representations for each value. These are SDR objects.
        dateBits = self.encTimestamp.encode(ts)
        scalarBits = []
        for enc_info in sorted(self.scalar_encoders, key=lambda i: i.get('idx')):
            name = enc_info.get('name')
            encoder = enc_info.get('encoder')
            val = data.get(name)
            if val is None:
                raise Exception('Value for {} is None. Aborting.'.format(name))
            scalarBits.append(encoder.encode(float(val)))

        encoding = SDR(self.enc_width).concatenate([dateBits] + scalarBits)

        # 2. Spatial Pooler
        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        activeColumns = SDR(self.sp.getColumnDimensions())
        # Execute Spatial Pooling algorithm over input space.
        self.sp.compute(encoding, True, activeColumns)

        # 3. Temporal Memory
        # Execute Temporal Memory algorithm over active mini-columns.
        self.tm.compute(activeColumns, learn=True)

        # 4. Anomaly
        raw = self.tm.anomaly
        like = self.anomalyLikelihood.anomalyProbability(data, raw, ts)
        logScore = self.anomalyLikelihood.computeLogLikelihood(like)
        anomalyScore = logScore

        return anomalyScore, raw
