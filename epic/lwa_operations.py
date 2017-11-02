import numpy as NP
import astropy
from astropy.io import fits
from collections import Counter
import progressbar as PGB
try:
    from lsl.reader import tbn, errors
    from lsl.common import stations
except ImportError:
    lsl_module_found = False
else:
    lsl_module_found = True

epic_path = epic.__path__[0]+'/'    

#################################################################################  

class LWAPol:

    """
    ----------------------------------------------------------------------------
    Class to manage polarization information of an LWA data set. 

    Attributes:

    pol:          [scalar] specifies polarization. Can be string or integer.
                  Default = 'P1'.

    frames:       [list] List of frames from LWA data set belonging to the
                  polarization specified by attribute pol. Each frame in this 
                  list is a FrameType object. Read LWA LSL for more
                  documentation.

    stands:       [list of strings] List of strings (converted from integers)
                  specifying the stands used in the data set.

    timestamps:   [list of strings] List of strings specifying the time tag for
                  the specific electric field time series.

    n_timestamps: [integer] Number of timestamps used in the measurement.

    n_stands:     [integer] Number of stands used in the measurement.
    
    antennas:     [list of LSL antenna objects] Each element is an instance of
                  LSL Antenna class

    antpos:       [numpy array] Nx3 antenna positions (in m). [East, North, Up]

    cable_delays: [dictionary] contains the following keys with following info:
                  'freq': frequencies (in Hz) for which delays are provided
                  stand_id: An integer that specifies the stand (antenna) for
                          which the delays are given as function of frequency. 
                          The size of delays is same as the number of frequencies
                          There will be one stand_id key for each stand

    data:         [dictionary] contains the following keys with following info:
                  'tod': denotes time-ordered data. It is a sub-dictionary with 
                         the following keys and information:
                         timestamp: string denoting timestamp. One key for each
                                timestamp. Each of these keys holds another
                                dictionary with the following keys and
                                information:
                                stand_id: an integer key denoting the stand 
                                      (antenna) and there is one for each of the 
                                      stands. Each of these keys holds complex
                                      electric fields for that timestamp. The 
                                      length of this vector is equal to the 
                                      number of delays and frequency channels.
                  'aod': denotes antenna-ordered data. This is a future feature
                         that needs to be developed.

    Member functions:

    __init__():   Initializes an instance of class LWAPol 

    parse():      Parses the information held by an instance of class LWAPol 

    Read the member function docstrings for details. 
    -----------------------------------------------------------------------------
    """

    def __init__(self, pol=None):

        """
        ------------------------------------------------------------------------
        Initializes an instance of class LWAPol 

        Class attributes initialized are:
        pol, frames, stands, timestamps, n_timestamps, n_stands, antennas, data,
        antpos, cable_delays

        Read docstring of class LWAPol for details on these attributes.
        ------------------------------------------------------------------------
        """

        if pol is None:
            self.pol = 'P1'
        else:
            self.pol = pol
        self.frames = []
        self.stands = []
        self.timestamps = []
        self.n_timestamps = 0
        self.data = {}
        self.data['tod'] = {}
        self.data['aod'] = {}
        self.antennas = {}
        self.antpos = []
        self.cable_delays = {}

    ############################################################################# 

    def parse(self, order='T', freq=None):

        """
        -------------------------------------------------------------------------
        Parses the information held by an instance of class LWAPol

        Keyword Input(s):

        order    [string] specifies ordering of data. Accepted values are 'T'
                 (time ordering, default) or 'A' (antenna ordering)

        freq     [list or numpy vector] frequencies (in Hz) that will be used to
                 estimate cable delays.
        ------------------------------------------------------------------------
        """

        if freq is None:
            raise ValueError('freq is necessary to compute cable delays.')

        self.cable_delays['freq'] = freq
        antpos = {}
        for digitizer in self.antennas:
            self.cable_delays[self.antennas[digitizer].stand.id] = self.antennas[digitizer].cable.delay(freq)
            antpos[self.antennas[digitizer].stand.id] = [self.antennas[digitizer].stand.x, self.antennas[digitizer].stand.y, self.antennas[digitizer].stand.z]

        # for antenna in self.antennas:
        #     self.cable_delays[antenna.stand.id] = antenna.cable.delay(freq)
        #     if self.antpos is None:
        #         self.antpos = [[antenna.stand.x, antenna.stand.y, antenna.stand.z]]
        #     else:
        #         self.antpos += [[antenna.stand.x, antenna.stand.y, antenna.stand.z]]
        # self.antpos = NP.asarray(self.antpos)

        if order == 'T':
            for frame in self.frames:
                timestamp = '{0:.5f}'.format(frame.getTime())
                stand = self.antennas[frame.header.tbnID].stand.id
                # stand = '{0:0d}'.format(stand)
                if timestamp not in self.data['tod']:
                    self.data['tod'][timestamp] = {}
                self.data['tod'][timestamp][stand] = frame.data.iq
                if timestamp not in self.timestamps:
                    self.timestamps += [timestamp]
                self.n_timestamps = len(self.timestamps)
                if stand not in self.stands:
                    self.stands += [stand]
                    self.antpos += [antpos[stand]]
                self.n_stands = len(self.stands)

#################################################################################

class LWAObs:
    
    """
    -----------------------------------------------------------------------------
    Class to hold LWA observation information. 

    Attributes:

    n_stands     [integer] number of stands used in the observation. 

    n_stations   [integer] number of LWA stations used in the observation. At the 
                 moment n_stations is always set to 1.

    n_pol        [integer] number of polarizations measured in the observation. 
                 Accepted values are 0, 1, or 2.

    frames_per_obs
                 [integer] Number of frames in the observation 

    sample_rate  [float] Sample rate of the time series (number/sec or Hz) and 
                 equal to inverse of sampling interval

    center_freq  [float] Center frequency of the observation (in Hz)

    freq         [numpy vector] Frequency channels in the observation. The values
                 are in units of Hz.

    gain         [scalar] Analog gain (not really used anywhere yet)

    antennas     [list of LSL antenna objects] Each element is an instance of
                 LSL Antenna class. 

    latitude     [float] Latitude of LWA station in degrees

    epoch        [string] Epoch of coordinate system

    P1           [instance of class LWAPol] Object holding polarization P1 data 
                 and associated operations. Read docstring of class LWAPol for 
                 more details 

    P2           [instance of class LWAPol] Object holding polarization P2 data 
                 and associated operations. Read docstring of class LWAPol for 
                 more details 

    Member functions:

    __init__():  Initialize an instance of class LWAObs

    parseTBN():  Parse a TBN data set obtained with the LWA

    writeTBN():  Write a parsed TBN data set into a FITS file to disk

    -----------------------------------------------------------------------------
    """

    def __init__(self):

        """
        -------------------------------------------------------------------------
        Initialize an instance of class LWAObs

        Class sttributes initialized are:
        n_stations, n_stands, n_pol, frames_per_obs, sample_rate, center_freq, 
        P1, P2, freq, gain, antennas, latitude, epoch

        Read docstring of class LWAPol for details on these attributes.
        -------------------------------------------------------------------------
        """

        self.n_stations = 1
        self.n_stands = 0
        self.n_pol = 0
        self.frames_per_obs = 0
        self.sample_rate = 0.0
        self.P1 = LWAPol(pol=0)
        self.P2 = LWAPol(pol=1)
        self.center_freq = None
        self.freq = None
        self.gain = None
        self.antennas = {}
        self.latitude = None
        self.epoch = ''

    #############################################################################

    def parseTBN(self, filename, append=False, order='T'):

        """
        -------------------------------------------------------------------------
        Parses the information held by an instance of class LWAObs by calling the
        parse functionality of class LWAPol.

        Input:

        filename    [string] string containing filename for the LWA data

        Keyword Input(s):

        append      [boolean] If True, append the information in the filename to
                    the currently held information. Default = False, create a 
                    new instance holding information only from the filename
                    provided

        order       [string] string specifying ordering of data to be implemented
                    by the parser. Accepted values are 'T' (default) and 'A'. 'T'
                    denotes time ordering while 'A' denotes antenna ordering.
        -------------------------------------------------------------------------
        """

        fh = open(filename, 'rb')
        if not append:
            self.P1.frames = []
            self.P2.frames = []

        lwa1 = stations.lwa1
        self.station = lwa1.name
        self.latitude = NP.degrees(lwa1.lat)
        self.epoch = str(lwa1.epoch)
        
        antennas = lwa1.getAntennas()
        for antenna in antennas:
            # self.antennas[antenna.digitizer] = antenna
            if antenna.pol == 0:
                self.P1.antennas[antenna.digitizer] = antenna
            elif antenna.pol == 1:
                self.P2.antennas[antenna.digitizer] = antenna

        self.frames_per_obs = tbn.getFramesPerObs(fh)
        self.sample_rate = tbn.getSampleRate(fh)
        while True:
            try:
                frame = tbn.readFrame(fh)
            except errors.syncError:
                continue
            except errors.eofError:
                break

            if self.center_freq is None:
                self.center_freq = frame.getCentralFreq()
            if self.gain is None:
                self.gain = frame.getGain()

            some_id, pol = frame.parseID()
            if pol == 0:
                self.P1.frames += [frame]
            elif pol == 1:
                self.P2.frames += [frame]
            else:
                raise ValueError('Invalid polarization value.')
        fh.close()

        self.freq = self.center_freq + NP.fft.fftshift(NP.fft.fftfreq(frame.data.iq.size, d=1.0/self.sample_rate))

        self.P1.parse(order=order, freq=self.freq)
        self.P2.parse(order=order, freq=self.freq)

#################################################################################

    def writeTBN(self, filename, pol=None, tabtype='BinTableHDU',
                 overwrite=False, verbose=True):

        """
        -------------------------------------------------------------------------
        Write a parsed TBN data set into a FITS file to disk

        Inputs:

        filename  [string] Prefix of the output file name to be written to disk
                  in FITS format. This will be suffixed with a string describing
                  polarization as given in the input parameter pol and a filename
                  extension '.fits'

        Keyword Inputs:
        
        pol       [scalar] integer or string denotes the polarization to be
                  written to disk. 'P1' and '0' are synonymous while 'P2' and '1'
                  are synonymous. Default = None, writes out both polarizations

        tabtype   [string] indicates table type for one of the extensions in 
                  the FITS file. Allowed values are 'BinTableHDU' and 
                  'TableHDU' for binary ascii tables respectively. Default is
                  'BinTableHDU'.
        
        overwrite [boolean] True indicates overwrite even if a file already 
                  exists. Default = False (does not overwrite)

        verbose   [boolean] If True (default), prints diagnostic and progress
                  messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        try:
            filename
        except NameError:
            raise NameError('No filename provided. Aborting LWAObs.writeTBN()')

        use_ascii = False
        if tabtype == 'TableHDU':
            use_ascii = True

        if (pol == 'P1') or (pol == 0) or (pol is None):
            outfile = filename + '.pol-{0:0d}.fits'.format(self.P1.pol)
            
            hdulist = []
            hdulist += [fits.PrimaryHDU()]
            hdulist[0].header['pol'] = self.P1.pol
            hdulist[0].header['sample_rate'] = self.sample_rate
            hdulist[0].header['center_freq'] = self.center_freq
            hdulist[0].header['nchan'] = self.freq.size
            hdulist[0].header['gain'] = self.gain
            hdulist[0].header['station'] = self.station
            hdulist[0].header['latitude'] = self.latitude
            hdulist[0].header['epoch'] = self.epoch
            hdulist[0].header['EXTNAME'] = 'PRIMARY'

            hdulist += [fits.ImageHDU(self.freq, name='freqs')]
            
            cols = []
            cols += [fits.Column(name='Antenna', format='I3', array=NP.asarray(self.P1.stands))]
            cols += [fits.Column(name='Position', format='3D', array=NP.asarray(self.P1.antpos))]
            if astropy.__version__ == '0.4':
                columns = fits.ColDefs(cols, tbtype=tabtype)
            else:
                columns = fits.ColDefs(cols, ascii=use_ascii)
                
            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'Antenna Positions')
            hdulist += [tbhdu]

            cols = []
            cols += [fits.Column(name='timestamp', format='16A', array=NP.asarray(self.P1.timestamps))]
            if astropy.__version__ == '0.4':
                columns = fits.ColDefs(cols, tbtype=tabtype)
            else:
                columns = fits.ColDefs(cols, ascii=use_ascii)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'TIMESTAMPS')
            hdulist += [tbhdu]

            cols = []
            cols += [fits.Column(name='frequency', format='D', array=self.P1.cable_delays['freq'])]
            for stand in self.P1.stands:
                cols += [fits.Column(name='{0:0d}'.format(stand), format='D', array=self.P1.cable_delays[stand])]

            if astropy.__version__ == '0.4':
                columns = fits.ColDefs(cols, tbtype=tabtype)
            else:
                columns = fits.ColDefs(cols, ascii=use_ascii)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'FREQUENCIES AND CABLE DELAYS')
            hdulist += [tbhdu]

            if verbose:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' [', right='] '), '/{0:0d} timestamps '.format(self.P1.n_timestamps), PGB.ETA()], maxval=self.P1.n_timestamps).start()
                i = 0

            for timestamp in self.P1.timestamps:
                cols = []
                for stand in self.P1.data['tod'][timestamp]:
                    cols += [fits.Column(name='{0:0d}'.format(stand), format='2D', array=NP.hstack((self.P1.data['tod'][timestamp][stand].real.reshape(-1,1), self.P1.data['tod'][timestamp][stand].imag.reshape(-1,1))))]
                if astropy.__version__ == '0.4':
                    columns = fits.ColDefs(cols, tbtype=tabtype)
                else:
                    columns = fits.ColDefs(cols, ascii=use_ascii)

                tbhdu = fits.new_table(columns)
                tbhdu.header.set('EXTNAME', timestamp)
                hdulist += [tbhdu]
                if verbose:
                    progress.update(i+1)
                    i += 1
            if verbose:
                progress.finish()
            
            hdu = fits.HDUList(hdulist)
            hdu.writeto(outfile, clobber=overwrite)

        if (pol == 'P2') or (pol == 1) or (pol is None):
            outfile = filename + '.pol-{0:0d}.fits'.format(self.P2.pol)
            
            hdulist = []
            hdulist += [fits.PrimaryHDU()]
            hdulist[0].header['pol'] = self.P2.pol
            hdulist[0].header['sample_rate'] = self.sample_rate
            hdulist[0].header['center_freq'] = self.center_freq
            hdulist[0].header['nchan'] = self.freq.size
            hdulist[0].header['gain'] = self.gain
            hdulist[0].header['EXTNAME'] = 'PRIMARY'

            hdulist += [fits.ImageHDU(self.freq, name='freqs')]
            
            cols = []
            cols += [fits.Column(name='Antenna', format='I3', array=NP.asarray(self.P2.stands))]
            cols += [fits.Column(name='Position', format='3D', array=self.P2.antpos)]
            if astropy.__version__ == '0.4':
                columns = fits.ColDefs(cols, tbtype=tabtype)
            else:
                columns = fits.ColDefs(cols, ascii=use_ascii)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'Antenna Positions')
            hdulist += [tbhdu]

            cols = []
            cols += [fits.Column(name='timestamp', format='16A', array=NP.asarray(self.P2.timestamps))]
            if astropy.__version__ == '0.4':
                columns = fits.ColDefs(cols, tbtype=tabtype)
            else:
                columns = fits.ColDefs(cols, ascii=use_ascii)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'TIMESTAMPS')
            hdulist += [tbhdu]

            cols = []
            cols += [fits.Column(name='frequency', format='D', array=self.P2.cable_delays['freq'])]
            for stand in self.P2.stands:
                cols += [fits.Column(name='{0:0d}'.format(stand), format='D', array=self.P2.cable_delays[stand])]
            if astropy.__version__ == '0.4':
                columns = fits.ColDefs(cols, tbtype=tabtype)
            else:
                columns = fits.ColDefs(cols, ascii=use_ascii)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'FREQUENCIES AND CABLE DELAYS')
            hdulist += [tbhdu]

            if verbose:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' [', right='] '), '/{0:0d} timestamps '.format(self.P2.n_timestamps), PGB.ETA()], maxval=self.P2.n_timestamps).start()
                i = 0
            for timestamp in self.P2.timestamps:
                cols = []
                for stand in self.P2.data['tod'][timestamp]:
                    cols += [fits.Column(name='{0:0d}'.format(stand), format='2D', array=NP.hstack((self.P2.data['tod'][timestamp][stand].real.reshape(-1,1), self.P2.data['tod'][timestamp][stand].imag.reshape(-1,1))))]
                if astropy.__version__ == '0.4':
                    columns = fits.ColDefs(cols, tbtype=tabtype)
                else:
                    columns = fits.ColDefs(cols, ascii=use_ascii)

                tbhdu = fits.new_table(columns)
                tbhdu.header.set('EXTNAME', timestamp)
                hdulist += [tbhdu]
                if verbose:
                    progress.update(i+1)
                    i += 1
            if verbose:
                progress.finish()
            
            hdu = fits.HDUList(hdulist)
            hdu.writeto(outfile, clobber=overwrite)

#################################################################################

def lwa_TBN_data_reformatter(infile, outfile, overwrite=False, verbose=True):

    """
    -----------------------------------------------------------------------------
    Wrapper routine to make use of LWAPol and LWAObs classes to read in a LWA
    TBN data file and write to a FITS file on disk

    Inputs:

    infile    [string] string specifying the file location of LWA data. 

    outfile   [string] Prefix of the output file name to be written to disk
              in FITS format. This will be suffixed with a string describing
              polarization as given in the input parameter pol and a filename
              extension '.fits'

    overwrite [boolean] True indicates overwrite even if a file already 
              exists. Default = False (does not overwrite)

    verbose   [boolean] If True (default), prints diagnostic and progress
              messages. If False, suppress printing such messages.
    -----------------------------------------------------------------------------
    """

    try:
        fh = open(infile, 'rb')
    except IOError:
        raise IOError('File not found. Image instance not initialized.')
    except EOFError:
        raise EOFError('EOF encountered. File cannot be read. Image instance not initialized.')
    else:
        fh.close()
        lwainfo = LWAObs()
        lwainfo.parseTBN(infile, append=False, order='T')
        lwainfo.writeTBN(outfile, pol=None, tabtype='BinTableHDU', overwrite=True, verbose=verbose)

#################################################################################

