import os
import glob
import csv
import json
import time

class LogWriter(object):
    #kind of inspired form openai.baselines.bench.monitor
    #We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, monitor='monitor.csv', header='', load_model=None):
        self.keys = tuple(keys)+('t',)
        assert path is not None
        self._clean_log_dir(path)
        filename = os.path.join(path, monitor)

        self.f = open(filename, "w" if load_model is None else "a")

        is_empty = os.stat(filename).st_size == 0
        
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)
        if load_model is None or is_empty:  # Write header if clean_log_dir is True or the file is empty
            self.f.write(header)
            self.logger.writeheader()
        
        self.f.flush()

        self.tstart = time.time()

    def write_row(self, epinfo):
        if self.logger:
            t = time.time() - self.tstart
            epinfo['t'] = t
            self.logger.writerow(epinfo)
            self.f.flush()

    def _clean_log_dir(self,log_dir):
        try:
            os.makedirs(log_dir)
        except OSError:
            files = glob.glob(os.path.join(log_dir, '*.csv'))
            #for f in files:
            #    os.remove(f)
