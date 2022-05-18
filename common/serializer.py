from pytARD_2D.partition import Partition2D

from common.notification import Notification
from common.parameters import SimulationParameters

import pickle
import lzma
from datetime import date, datetime
import string

class Serializer():
    '''
    Saves and reads simulation to and from disk.
    '''
    def __init__(self, compress: bool=False):
        '''
        Creates a serializer.

        Parameters
        ----------
        compress : bool
            Determines if the data is read or written with LZMA compression.
        '''
        self.compress = compress

    def create_filename(self):
        '''
        Generates a file name using time stamps.
        '''
        # TODO: put reference to rooms, sizes etc.
        return f"pytard_{date.today()}_{datetime.now().time()}"

    def dump(self, sim_param: SimulationParameters, partitions: Partition2D, filename: str=None):
        '''
        Writes simulation state data to disk.

        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        partitions : list
            List of Partition objects. All partitions of the domain are collected here.
        filename : str
            File path to write to.
        '''
        if filename:
            file_path = filename + ".xz"
        else:
            file_path = self.create_filename() + ".xz"
        if sim_param.verbose:
            print(f"Writing state data to disk ({file_path}). Please wait...", end="")
        Notification.notify("Writing state data to disk ({file_path}). Please wait...", "pytARD: Writing state data")
        if self.compress: 
            with lzma.open(file_path, 'wb') as fh:
                pickle.dump((sim_param, partitions), fh)
                fh.close()
        else:
            with pickle.open(file_path, 'wb') as fh:
                pickle.dump((sim_param, partitions), fh)
                fh.close()
        if sim_param.verbose:
            print("Done.")
    Notification.notify("Writing state data completed", "pytARD: Writing state data")


    def read(self, file_path: string):
        '''
        Reads simulation state data from disk.

        Parameters
        ----------
        filename : str
            File path to read from.
        '''
        # TODO: Idea -> See which suffix the file has. If xz, use lzma
        if self.compress:
            raw_bytes = lzma.open(file_path, 'rb')
        else:
            raw_bytes = pickle.open(file_path, 'rb')
        return pickle.load(raw_bytes)
