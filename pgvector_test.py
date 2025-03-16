# pip install psycopg2-binary


import os
import psycopg2
import io
import pandas as pd
import numpy as np


def query(query_string: str, cursor: psycopg2.extensions.cursor) -> list:

    cursor.execute(query_string)
    conn.commit()

    rows = cursor.fetchall()

    return rows


def create_table(table_name: str, conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor) -> None:
    
    # SQL-Statement
    sql = f"""
    CREATE TABLE {table_name} (
        Company VARCHAR(255),
        Fiscal_Year VARCHAR(255),
        Document VARCHAR(255)
    );
    """

    # Tabelle erstellen
    cur.execute(sql)
    conn.commit()


def import_datatable(file_path: str, table_name: str, conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor) -> None:

    # Convert DataFrame to CSV format in memory
    csv_buffer = io.StringIO()
    df = pd.read_csv(file_path, delimiter = ',', header=0, quotechar='"', quoting=1, encoding='utf-8')
    df.to_csv(csv_buffer, index=False, header=False)
    csv_buffer.seek(0)

    print(csv_buffer)

    # Use COPY FROM to insert data into PostgreSQL
    cur.copy_from(csv_buffer, table_name, sep=",", columns=("Company", "Fiscal_Year", "Document"))

    # Commit and close connection
    conn.commit()


# Verbindung zur PostgreSQL-Datenbank
conn = psycopg2.connect(
    dbname=os.environ.get("DATABASE"),
    user=os.environ.get("USER"),
    password=os.environ.get("PASSWORD"),
    host="localhost",
    port="5432"
)

# Cursor erstellen
cur = conn.cursor()

type(cur)

#result = query("SELECT * FROM pg_database", cur)
#print(result)


file_path = "/home/aunkofer/aunkofer/SAP/SAP_ECC_old_GL_Ergebnisse/result_usr_journal_old_GL.csv"

create_table("usr_journal_old_GL", conn, cur)
import_datatable(file_path, "usr_journal_old_GL", conn, cur)


# Cursor und Verbindung schließen
cur.close()
conn.close()



