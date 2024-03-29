import pickle
import os
from pyspark import SparkConf, SparkContext
import re
from datetime import datetime
import argparse

def get_field(row, field_name):
    search_str = ' '+field_name+'=\"'
    start_point = row.find(search_str, 0)
    if start_point == -1:
        return None
    start_point += len(search_str)
    end_point = row.find('\"', start_point)
    return row[start_point:end_point].encode('utf8')


def get_datetime(date, dt_format='%Y-%m-%dT%H:%M:%S.%f'):
    return datetime.strptime(date, dt_format)


def get_tags(tags_list):
    opening_str = '&lt;'
    closing_str = '&gt;'
    opening_tag = [m.end() for m in re.finditer(opening_str, tags_list)]
    closing_tag = [m.start() for m in re.finditer(closing_str, tags_list)]

    result_list = []
    for i in range(len(opening_tag)):
        result_list.append(tags_list[opening_tag[i]:closing_tag[i]])

    return result_list

def main():
    """Creates a dataframe of tags' monthly question counts
    for the tags provided in a file and for the provided time period."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--posts', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tags_list', type=str, required=True)
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    args = parser.parse_args()

    dt_format='%Y-%m'

    input_posts_file = args.posts
    output_dir = args.output_dir
    tags_list_filename = args.tags_list
    start_date = get_datetime(args.start_date,dt_format)
    end_date = get_datetime(args.end_date,dt_format)

    tags_list = open(tags_list_filename, 'r').readlines()
    tags_list = [x.strip() for x in tags_list]
    tags_list = [x for x in tags_list if x != '']

    conf = SparkConf().set("spark.driver.maxResultSize", "10G"). \
        set("spark.hadoop.validateOutputSpecs", "false"). \
        set('spark.default.parallelism', '200')

    spark = SparkContext(conf=conf)

    tags_list_broadcast = spark.broadcast(tags_list)

    # Filtering out lines without an id (i.e. metadata lines)
    posts_rdd = spark.textFile(input_posts_file).filter(lambda x: get_field(x, 'Id') is not None and
                                                                  get_field(x, 'Tags') is not None and
                                                                  get_field(x, 'CreationDate') is not None)

    posts_rdd = posts_rdd.map(lambda x: (int(get_field(x, 'Id').decode('utf-8')),
                                         get_field(x, 'Tags').decode('utf-8'),
                                         get_datetime(get_field(x, 'CreationDate').decode('utf-8')))). \
        filter(lambda x: x[1] is not None and x[2] is not None). \
        filter(lambda x: x[2] <= end_date and x[2] >= start_date). \
        filter(lambda x: len(x[1]) > 0). \
        map(lambda x: (x[0], get_tags(x[1]), x[2])). \
        flatMap(lambda x: [(x[0], (y, x[2])) for y in x[1]]). \
        filter(lambda x: x[1][0] in tags_list_broadcast.value)

    posts_rdd = posts_rdd.map(lambda x: (x[1][0] + '_' + (str(x[1][1].year) + '-' + str(x[1][1].month)), 1))
    posts_rdd = posts_rdd.reduceByKey(lambda x,y: x+y).\
                map(lambda x: (x[0].split('_')[0], x[0].split('_')[1], x[1]))

    collected = posts_rdd.collect()
    df = {'TagName': [x[0] for x in collected],
          'Date': [x[1] for x in collected],
          'Count': [x[2] for x in collected]}

    with open(os.path.join(output_dir, 'so_tag_counts_df.pkl'), mode='wb') as f:
        pickle.dump(df, f)

if __name__ == '__main__':
    main()