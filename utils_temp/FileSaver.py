import os
from os.path import join as pjoin
from .utils_ import write_dict_to_csv
import logging

class FileSaver:

    def __init__(self, logger, save_name=None, model_dir=None, args=None):
        if save_name:
            self.save_name = save_name
        if model_dir:
            self.model_dir = model_dir
        if args:
            self.args = args
        if model_dir and save_name:
            os.makedirs(pjoin(model_dir, save_name), exist_ok=True)
        self.logger = logger

    def _init(self, save_name, model_dir, args):
        self.save_name = save_name
        self.model_dir = model_dir
        self.args = args
        os.makedirs(pjoin(model_dir, save_name), exist_ok=True)

    def save(self, save_type, save_name, value):
        """
        Save the given value based on the save_type and save_name.

        :param save_type: Type of the save operation ('setup', 'result', 'model')
        :param save_name: Name of the file to save
        :param value: Value to be saved
        """
        if save_type == 'setup':
            self._save_setup(save_name, value)
        elif save_type == 'result':
            self._save_result(save_name, value)
        elif save_type == 'model':
            self._save_model(save_name, value)
        else:
            self.logger.log(logging.INFO, 'saveType must be args, result, or model')

    def _save_setup(self, save_name, value):
        with open(pjoin(self.model_dir, self.save_name, 'setup.txt'), 'a') as f:
            message = 'Saved {}:\n{}'.format(save_name, str(value))
            self.logger.log(logging.INFO, message)
            f.write(message)
            f.write('\n\n')

    def _save_result(self, save_name, value):
        if save_name == '': save_name = self.args.id
        with open(pjoin(self.model_dir, self.save_name, 'result.txt'), 'a') as f:
            message = '{}:\n{}\n'.format(save_name, str(value))
            self.logger.log(logging.INFO, message)
            f.write(message)
            f.write('\n\n')
        write_dict_to_csv(value, pjoin(self.model_dir, self.save_name, save_name + '.csv'))
        write_dict_to_csv(value, pjoin('tempdata', save_name + '.csv'))

    def _save_model(self, save_name, value):
        if save_name == '': save_name = self.args.id
        paths = pjoin(self.model_dir, self.save_name, save_name)
        value.save_model(paths)

