#!/usr/bin/python
from configparser import ConfigParser 
import os

class Config(object):
    @staticmethod
    def config(filename='config.ini', section='DB_STATS'):
        parser = ConfigParser()
        #print filename
        parser.read(filename)

        db = {}

        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))
        return db


if __name__ == '__main__':
    current_dir = os.getcwd()
    config_path = current_dir + '/config.ini'
    print config_path
    db = Config.config(section = 'DB_STATS')
    print db
