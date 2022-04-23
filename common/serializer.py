import pickle
import lzma
from datetime import date, datetime

class Serializer():
    def __init__(self, compress=False):
        self.compress = compress

    def create_filename(self):
        # TODO: put reference to rooms, sizes etc.
        return f"pytard_{date.today()}_{datetime.now().time()}"

    def dump(self, sim_params, partitions):
        file_path = self.create_filename() + ".xz"
        if sim_params.verbose:
            print(f"Writing state data to disk ({file_path}). Please wait...", end="")
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

    def read(self, file_path):
        # TODO: Idea -> See which suffix the file has. If xz, use lzma
        if self.compress:
            raw_bytes = lzma.open(file_path, 'rb')
        else:
            raw_bytes = pickle.open(file_path, 'rb')
        return pickle.load(raw_bytes)
