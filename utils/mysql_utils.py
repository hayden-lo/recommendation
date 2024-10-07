import pymysql
import pandas as pd


class MysqlClient:
    def __init__(self):
        self.host = "localhost"
        self.user_name = "root"
        pwd = "root"
        self.conn = pymysql.connect(host=self.host, user=self.user_name, password=pwd, autocommit=True)
        self.cursor = self.conn.cursor()

    def get_data(self, sql):
        self.cursor.execute(sql)
        columns = [i[0] for i in self.cursor.description]
        return pd.DataFrame(self.cursor.fetchall(), columns=columns)

    def insert_df(self, df, table):
        insert_cols = ",".join(df.columns)
        placeholders = len(df.columns) * "%s,"
        insert_sql = f"insert into {table}({insert_cols}) values({placeholders[:-1]}) "
        self.cursor.executemany(insert_sql, df.values.tolist())
        self.conn.commit()

    def delete_data(self, table, key, ids):
        self.cursor.execute(f"delete from {table} where {key} in({ids})")
        self.conn.commit()
