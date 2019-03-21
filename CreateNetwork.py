import psycopg2
import geopandas as gpd
import numpy as np
from GeoMultiGraph import GeoMultiGraph

NETWORK_LIST = [['20121102', '20121103', '20121104', '20121105', '20121106', '20121107', '20121108'],
                ['20130523', '20130524', '20130525', '20130526', '20130527', '20130528', '20130529'],
                ['20140301', '20140302', '20140303', '20140304', '20140305', '20140306', '20140307'],
                ['20150301', '20150302', '20150303', '20150304', '20150305', '20150306', '20150307'],
                ['20160301', '20160302', '20160303', '20160304', '20160305', '20160306', '20160307'],
                ['20170301', '20170302', '20170303', '20170304', '20170305', '20170306', '20170307']]

GEO_TABLE = 'zz.spatialunit_ring6_deg'


def create_network(cursor, table_name):
    cursor.execute('select distinct tazid from %s order by tazid' % GEO_TABLE)
    region_list = [row[0] for row in cursor.fetchall()]
    network = np.zeros((len(region_list), len(region_list)), dtype=np.int)
    k = 0
    for oRegion in region_list:
        for name in table_name:
            cursor.execute('SELECT d_region, count(o_region) FROM %s WHERE o_region = %d GROUP BY d_region order by d_region ' % (name, oRegion))
            for row in cursor.fetchall():
                if row[0] in region_list:
                    network[k][region_list.index(row[0])] += row[1]
        k = k + 1
        print(oRegion)
    return network


def create_mapping(con):
    sql = 'select tazid, code, geom from %s order by tazid' % GEO_TABLE
    df = gpd.GeoDataFrame.from_postgis(sql, con, geom_col='geom')
    return df


if __name__ == '__main__':
    conn = None
    try:
        conn = psycopg2.connect(database='taxidata',
                                user='lazzy',
                                password='7325891325',
                                host='192.168.61.251',
                                port='5432')
        cur = conn.cursor()
        geoMapping = create_mapping(conn)
        multiNetwork = []
        for i in NETWORK_LIST:
            print('Begin %s' % i)
            tableName = ['public.od_unit_%s' % k for k in i]
            multiNetwork.append(create_network(cur, tableName))
            print('Finish %s' % i[0][0: 4])
        gmg = GeoMultiGraph(geoMapping, np.array(multiNetwork))
        gmg.save('GeoMultiGraph_week')
    except psycopg2.Error as e:
        print('Error %s' % e)
    finally:
        if conn:
            conn.close()
