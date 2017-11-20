#!/usr/bin/python
#coding=utf-8
from pkg.lib.database import Database
import sys

def output_data(output_file, data, header = None):
    fp = open(output_file, 'w+')
    fp.write('\t'.join(header) + '\n')
    for row in data:
        row_str = '\t'.join([str(column) for column in row])
        fp.write(row_str + '\n')
    fp.close()

if __name__ == '__main__':
    db = Database()
    #result = db.getSql('SELECT user_id, name, created_time FROM stats.core_users LIMIT 2')
    #print result[1]
    #sql = """INSERT INTO kevin.sample_users(user_id) VALUES(%s);"""
    #values = [("1"),("2"),("3")]
    #db.insertManyRows(sql, values)
    #result = db.queryFunc(func = 'nowcst')
    #print result
    '''
    result = db.queryFuncBySize('get_users_messages_sharded', ('98', '-1'), 10)
    for row in result:
        for column in row:
            print str(column)
    print(sys.version)
    print "哈哈"
    print sys.getdefaultencoding()
    print sys.stdin.encoding
    print sys.stdout.encoding
    '''
    query = """WITH users AS (
                              SELECT
                                      cu.user_id,
                                      d.os_name,
                                      COALESCE(d2.shared_device, 0) AS shared_device,
                                      gender,
                                      looking_for_gender,
                                      age_calculator(birthdate) AS age,
                                      search_min_age,
                                      search_max_age,
                                      LEFT(mobile_number, 7) AS mobile_prefix,
                                      cu.core_status as status
                              FROM
                                    stats.core_users cu
                              LEFT JOIN LATERAL(
                                                SELECT
                                                      cd.user_id,
                                                      CASE WHEN lower( os_name ) = 'android' THEN 'android' ELSE 'ios' END AS os_name,
                                                      cd.device_identifier
                                                FROM
                                                      stats.core_devices cd
                                                WHERE
                                                      cu.user_id = cd.user_id
                                                ORDER BY user_id, device_id ASC
                                                LIMIT 1
                              ) AS d ON TRUE 
                              LEFT JOIN LATERAL(
                                  SELECT
                                          device_identifier,
                                          COUNT(DISTINCT user_id) as shared_device
                                  FROM
                                          stats.core_devices cd 
                                  WHERE
                                          cd.device_identifier = d.device_identifier
                                      AND cd.user_id <> d.user_id
                                  GROUP BY 1 
                              ) AS d2 ON TRUE
                              WHERE
                                    cu.created_time BETWEEN '2017-08-01'::date AND '2017-08-02'::date
                          ), contact_list AS(
                              SELECT
                                      users.user_id,
                                      COALESCE(count(c.md5_hash11), 0) AS contact_list
                              FROM
                                    users
                              LEFT JOIN LATERAL(
                                      SELECT
                                              user_id,
                                              md5_hash11
                                      FROM
                                              core.user_mobile_contact_hashes m
                                      WHERE
                                              users.user_id = m.user_id
                                ) AS c ON TRUE
                              GROUP BY 1
                          ), swipes AS (
                              SELECT
                                    users.user_id,
                                    COALESCE(SUM(likes), 0) AS given_likes,
                                    COALESCE(SUM(dislikes), 0) AS given_dislikes,
                                    COALESCE(SUM(received_likes), 0) AS received_likes,
                                    COALESCE(SUM(received_dislikes), 0) AS received_dislikes
                              FROM
                                    users
                              LEFT JOIN
                                    yay.daily_swipes_by_users s ON users.user_id = s.user_id AND s.date_time >= '2017-08-01'::date
                              GROUP BY 1
                          )
                          SELECT
                                users.user_id,
                                users.os_name,
                                users.shared_device,
                                contact_list.contact_list,
                                users.gender,
                                users.looking_for_gender,
                                users.age,
                                users.search_min_age,
                                users.search_max_age,
                                users.mobile_prefix,
                                swipes.given_likes,
                                swipes.given_dislikes,
                                swipes.received_likes,
                                swipes.received_dislikes,
                                users.status
                          FROM
                                users
                          LEFT JOIN
                                contact_list USING(user_id)
                          LEFT JOIN
                                swipes USING(user_id)""";

                                
    header = ['user_id', 'os_name', 'shared_device', 'contact_list', 'gender', 'looking_for_gender', 'age', 'search_min_age',
                'search_max_age', 'mobile_prefix', 'given_likes', 'given_dislikes', 'received_likes', 'received_dislikes', 'status'];
    data = db.getSql(query);
    output_file = '/Users/kangpingyao/Documents/yaokp/antispam/data/users_20170801.txt'
    output_data(output_file, data, header)


