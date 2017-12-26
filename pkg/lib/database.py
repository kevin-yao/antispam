#!/usr/bin/python
from ..config.config import Config 
import psycopg2
import os

class Database(object):
    __db_connection = None

    def __init__(self, dbType = 'DB_STATS'):
        try:
            params = Config.config(os.getcwd() + '/pkg/config/config.ini', dbType)
            self.__db_connection = psycopg2.connect(**params)
            self.__db_connection.autocommit = True
        except (Exception, psycopg2.DatabaseError) as error:
                print(error)

    def getSql(self, query):
        try:
            cur = self.__db_connection.cursor()          
            
            # execute a statement
            cur.execute(query)
            
            result = cur.fetchall()
            
            # close the communication with the PostgreSQL
            cur.close()


            
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        return result


    def insertOneRow(self, sql, value):
        try:
            cur = self.__db_connection.cursor()          
            
            # execute a statement
            cur.execute(sql, (value,))
           
            # close the communication with the PostgreSQL
            cur.close()
            
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def insertManyRows(self, sql, values):
        try:
            cur = self.__db_connection.cursor()          
            
            # execute a statement
            cur.executemany(sql, values)
           
            # close the communication with the PostgreSQL
            cur.close()
            
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def queryFunc(self, func, params = None):
        try:
            cur = self.__db_connection.cursor()
            
            cur.callproc(func, params)

            result = cur.fetchall()
            
            # close the communication with the PostgreSQL
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        return result

    def queryFuncBySize(self, func, params = None, size = 10):
        try:
            cur = self.__db_connection.cursor()
            
            cur.callproc(func, params)

            result = cur.fetchmany(size)
            
            # close the communication with the PostgreSQL
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)   

        return result

    def __del__(self):
        self.__db_connection.close()



if __name__ == '__main__':
    db = Database()
    db.getSql('SELECT * FROM stats.core_users LIMIT 1')

