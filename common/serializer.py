from pytARD_2D.partition import Partition2D

from common.notification import Notification
from common.parameters import SimulationParameters

import pickle
import lzma
from datetime import date, datetime
import string

class Serializer():
    def __init__(self, compress: bool=False):
        self.compress = compress

    def create_filename(self):
        # TODO: put reference to rooms, sizes etc.
        return f"pytard_{date.today()}_{datetime.now().time()}"

    def dump(self, sim_params: SimulationParameters, partitions: Partition2D, filename: string=None):
        if filename:
            file_path = filename + ".xz"
        else:
            file_path = self.create_filename() + ".xz"
        if sim_params.verbose:
            print(f"Writing state data to disk ({file_path}). Please wait...", end="")
        Notification.notify("Writing state data to disk ({file_path}). Please wait...", "pytARD: Writing state data")
        if self.compress: 
            with lzma.open(file_path, 'wb') as fh:
                pickle.dump((sim_params, partitions), fh)
                fh.close()
        else:
            with pickle.open(file_path, 'wb') as fh:
                pickle.dump((sim_params, partitions), fh)
                fh.close()
        if sim_params.verbose:
            print("Done.")
    Notification.notify("Writing state data completed", "pytARD: Writing state data")


    def read(self, file_path: string):
        # TODO: Idea -> See which suffix the file has. If xz, use lzma
        if self.compress:
            raw_bytes = lzma.open(file_path, 'rb')
        else:
            raw_bytes = pickle.open(file_path, 'rb')
        return pickle.load(raw_bytes)
