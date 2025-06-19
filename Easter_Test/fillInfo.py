import pandas as pd
import numpy as np

from nxcals import build_spark_dataSet, to_timestamp_CET

# 'LHC.STATS:BETA_STAR_ALICE',
# 'LHC.STATS:BETA_STAR_ATLAS',
# 'LHC.STATS:BETA_STAR_CMS',
# 'LHC.STATS:BETA_STAR_LHCB',
# 'LHC.STATS:ENERGY',
                    
def _to_ts(string, round_seconds=True, unit='s'):
    t = pd.Timestamp(string, unit=unit, tz='CET')
    is_summer_time = t.dst().seconds > 0.1
    if round_seconds:
        t = t.round(freq='S', ambiguous=is_summer_time)
    return t

class Fill:
    def __init__(self, fill_number, *, ldb=None, correct_dump_time=False,
                 get_metadata=True, spark=None ):
        self._number = fill_number
        if ldb is None:
            if spark is None:
                raise ValueError("Need to pass a spark session 'spark=spark' to be "
                               + "able to connect to pytimber for fill data!")
            import pytimber
            ldb=pytimber.LoggingDB(source="nxcals", spark_session=spark)
        self._ldb = ldb
        data = ldb.get_lhc_fill_data(fill_number)
        self._t_start = _to_ts(data['startTime'])
        self._t_end = _to_ts(data['endTime'])
        bm = np.array([ 
                [_to_ts(mode['startTime']),_to_ts(mode['endTime']),mode['mode']] 
                for mode in data['beamModes'] ])
        self.modes = bm[bm[:, 0].argsort()]
        self._b1 = None
        self._b2 = None
        self._nrj = None
        self._t_dump = None
        if correct_dump_time and spark is not None:
            self.set_dump_time(spark=spark)
        if get_metadata and spark is not None:
            self.get_metadata(spark=spark)
        else:
            self.get_metadata(spark=None)


    def show(self):
        print(f"============= Fill {self._number} =============")
        print(f"From {self.t_start.strftime('%Y-%m-%d %H:%M:%S CET')} until {self.t_end.strftime('%Y-%m-%d %H:%M:%S CET')}")
        for tstart, tend, mode in self.modes:
            print(f"Beam mode {mode}: from {tstart.strftime('%Y-%m-%d %H:%M:%S CET')} until {tend.strftime('%Y-%m-%d %H:%M:%S CET')}")

    @property
    def number(self):
        return self._number

    @property
    def ldb(self):
        return self._ldb

    @property
    def beam_modes(self):
        return list(self.modes[:,2])

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_end(self):
        return self._t_end

    @property
    def t_dump(self):
        if self._t_dump is not None:
            return self._t_dump
        else:
            return np.array([mode[1] for mode in self.modes]).max()

    @property
    def t_injection(self):
        if 'INJPHYS' not in self.beam_modes:
            return None
        else:
            t_start = None
            for tstart, tend, mode in self.modes:
                if mode == 'INJPHYS':
                    t_start = tstart
                    t_end = tend              # This ensures the timestamp of the last beam mode is kept
            if t_start >= self.t_dump:
                return None
            else:
                return [t_start, min(self.t_dump, t_end)]

    @property
    def t_ramp(self):
        if 'RAMP' not in self.beam_modes:
            return None
        else:
            for t_start, t_end, mode in self.modes:
                if mode == 'RAMP':
                    if t_start >= self.t_dump:
                        return None
                    else:
                        return [t_start, min(self.t_dump, t_end)]

    @property
    def t_flat_top(self):
        if 'FLATTOP' not in self.beam_modes:
            return None
        else:
            for t_start, t_end, mode in self.modes:
                if mode == 'FLATTOP':
                    if t_start >= self.t_dump:
                        return None
                    else:
                        return [t_start, min(self.t_dump, t_end)]

    @property
    def t_squeeze(self):
        if 'SQUEEZE' not in self.beam_modes:
            return None
        else:
            for t_start, t_end, mode in self.modes:
                if mode == 'SQUEEZE':
                    if t_start >= self.t_dump:
                        return None
                    else:
                        return [t_start, min(self.t_dump, t_end)]

    @property
    def t_adjust(self):
        if 'ADJUST' not in self.beam_modes:
            return None
        else:
            for t_start, t_end, mode in self.modes:
                if mode == 'ADJUST':
                    if t_start >= self.t_dump:
                        return None
                    else:
                        return [t_start, min(self.t_dump, t_end)]

    @property
    def t_top_energy(self):
        if 'FLATTOP' not in self.beam_modes:
            return None
        else:
            for tstart, tend, mode in self.modes:
                if mode == 'FLATTOP':
                    t_start = tstart
                    break      
            if 'BEAMDUMP' in self.beam_modes:
                for tstart, tend, mode in self.modes:
                    if mode == 'BEAMDUMP':
                        t_end = tstart
                        break # do not break, to get the last stable mode (in case more would exist)
            elif 'STABLE' in self.beam_modes:
                for tstart, tend, mode in self.modes:
                    if mode == 'STABLE':
                        t_end = tend
            elif 'ADJUST' in self.beam_modes:
                for tstart, tend, mode in self.modes:
                    if mode == 'ADJUST':
                        t_end = tend
            elif 'SQUEEZE' in self.beam_modes:
                for tstart, tend, mode in self.modes:
                    if mode == 'SQUEEZE':
                        t_end = tend
            else:
                for tstart, tend, mode in self.modes:
                    if mode == 'FLATTOP':
                        t_end = tend
            if t_start >= self.t_dump:
                return None
            else:
                return [t_start, min(self.t_dump, t_end)]

    @property
    def t_stable(self):
        if 'STABLE' not in self.beam_modes:
            return None
        else:
            for t_start, t_end, mode in self.modes:
                if mode == 'STABLE':
                    if t_start >= self.t_dump:
                        return None
                    else:
                        return [t_start, min(self.t_dump, t_end)]


    def set_dump_time(self, *, spark=None):
        if 'INJPHYS' not in self.beam_modes:
            t_inj = self.t_start
        else:
            for tstart, tend, mode in self.modes:
                if mode == 'INJPHYS':
                    t_inj = tstart
        df = build_spark_dataSet(self.t_start, self.t_end, spark=spark, variable=[
                                    'LHC.BCTFR.A6R4.B1:BEAM_INTENSITY', 
                                    'LHC.BCTFR.A6R4.B2:BEAM_INTENSITY',
#                                     'HX:ENG'  # Wrong values
                                ]).orderBy('nxcals_variable_name','nxcals_timestamp').toPandas()
        df['diff'] = df['nxcals_value'].diff()
        self._b1 = df[df['nxcals_variable_name'] == 'LHC.BCTFR.A6R4.B1:BEAM_INTENSITY'].reset_index(drop=True)
        self._b2 = df[df['nxcals_variable_name'] == 'LHC.BCTFR.A6R4.B2:BEAM_INTENSITY'].reset_index(drop=True)
#         self._nrj = df[df['nxcals_variable_name'] == 'HX:ENG'].reset_index(drop=True)
        # we select the timestamp where the intensity loses at least 20% of the maximum in one second:
        # TODO: probably, to be on the safe side, we want the timestamp just before this
        int_drop_b1 = self._b1[
                        (self._b1['diff'] <= -0.2*self._b1['nxcals_value'].max()) &
                        (self._b1['diff'] < 0) &
                        [ to_timestamp_CET(nxt) > t_inj for nxt in self._b1['nxcals_timestamp'] ]
                      ]
        int_drop_b2 = self._b2[
                        (self._b2['diff'] <= -0.2*self._b2['nxcals_value'].max()) &
                        (self._b2['diff'] < 0) &
                        [ to_timestamp_CET(nxt) > t_inj for nxt in self._b2['nxcals_timestamp'] ]
                      ]
        if len(int_drop_b1)==0 and len(int_drop_b2)==0:
            self._t_dump = self.t_end
        elif len(int_drop_b1)==0:
            t_b2 = self._b1.loc[int_drop_b2.index[-1] - 1, 'nxcals_timestamp']
            self._t_dump = pd.Timestamp(t_b2, tz='CET')
        elif len(int_drop_b2)==0:
            t_b1 = self._b1.loc[int_drop_b1.index[-1] - 1, 'nxcals_timestamp']
            self._t_dump = pd.Timestamp(t_b1, tz='CET')
        else:
            t_b1 = self._b1.loc[int_drop_b1.index[0] - 1, 'nxcals_timestamp']
            t_b2 = self._b1.loc[int_drop_b2.index[0] - 1, 'nxcals_timestamp']
            self._t_dump = pd.Timestamp(min([t_b1, t_b2]), tz='CET')

    @property
    def num_bunches(self):
        return self._num_bunches

    @property
    def num_collision_IP15(self):
        return self._num_collision_IP15

    @property
    def num_collision_IP2(self):
        return self._num_collision_IP2

    @property
    def num_collision_IP8(self):
        return self._num_collision_IP8

    def get_metadata(self, *, spark=None):
        if spark is None:
            self._num_bunches        = None
            self._num_collision_IP15 = None
            self._num_collision_IP2  = None
            self._num_collision_IP8  = None
        else:
            df = build_spark_dataSet(self.t_start, self.t_end, spark=spark,
                                             variable=[
                                                       'LHC.STATS:B1_NUMBER_BUNCHES',
                                                       'LHC.STATS:B2_NUMBER_BUNCHES',
                                                       'LHC.STATS:NUMBER_COLLISIONS_IP1_5',
                                                       'LHC.STATS:NUMBER_COLLISIONS_IP2',
                                                       'LHC.STATS:NUMBER_COLLISIONS_IP8',
                                             ]).toPandas()
            b1num = df.loc[df['nxcals_variable_name'] == 'LHC.STATS:B1_NUMBER_BUNCHES','nxcals_value']
            b2num = df.loc[df['nxcals_variable_name'] == 'LHC.STATS:B2_NUMBER_BUNCHES','nxcals_value']
            self._num_bunches = [
                None if len(b1num)==0 else b1num.values[0],
                None if len(b2num)==0 else b2num.values[0]
            ]
            # HX:ENG HX:BETASTAR_IP{1,2,5,8}  HX:COLLST HX:COLLSTOP LHC.STATS:B1_PARTICLE_TYPE
            # LHC.STATS:BETA_STAR_{ALICE,ATLAS,CMS,LHCB}
            # LHC.STATS:EMITTANCE_BSRT_B{1,2}_{H,V}_{SSB,EOF}
            # LHC.STATS:FILLING_SCHEME
            ip15col = df.loc[df['nxcals_variable_name'] == 'LHC.STATS:NUMBER_COLLISIONS_IP1_5','nxcals_value']
            ip2col = df.loc[df['nxcals_variable_name'] == 'LHC.STATS:NUMBER_COLLISIONS_IP2','nxcals_value']
            ip8col = df.loc[df['nxcals_variable_name'] == 'LHC.STATS:NUMBER_COLLISIONS_IP8','nxcals_value']
            self._num_collision_IP15 = None if len(ip15col)==0 else ip15col.values[0]
            self._num_collision_IP2 = None if len(ip2col)==0 else ip2col.values[0]
            self._num_collision_IP8 = None if len(ip8col)==0 else ip8col.values[0]

                    

