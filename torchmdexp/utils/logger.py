import os
import glob
import csv
import json
import time

class LogWriter(object):
    #kind of inspired form openai.baselines.bench.monitor
    #We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, monitor='monitor.csv', header=''):
        self.keys = tuple(keys)+('t',)
        assert path is not None
        self._clean_log_dir(path)
        filename = os.path.join(path, monitor)

        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)
        self.logger.writeheader()
        self.f.flush()
        self.tstart = time.time()

    def write_row(self, epinfo):
        if self.logger:
            # Remove keys that should not be written
            keys = list(epinfo.keys())
            [epinfo.pop(k, None) for k in keys if k not in self.logger.fieldnames]
            
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

def init_logger(log_dir, name, file_level='info'):
    import logging
    import sys
    
    # Create file logger
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh_form = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    fh.setFormatter(fh_form)
    file_level = logging.INFO if file_level.lower().strip().startswith('i') else logging.DEBUG
    fh.setLevel(file_level)

    # Create console logger
    ch = logging.StreamHandler()
    ch_form = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    ch.setFormatter(ch_form)
    ch.setLevel(logging.INFO)
    
    # Get the root logger and add the handlers of interest
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.removeHandler(logger.handlers[0])
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Modify the behavior when getting uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    
    return logging.getLogger(name)
