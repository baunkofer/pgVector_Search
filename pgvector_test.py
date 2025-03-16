# pip install psycopg2-binary


import os
import psycopg2
import io
import pandas as pd
import numpy as np
from psycopg2 import sql


def select_all_from_table(table_name: str, cur: psycopg2.extensions.cursor):
    # Sichere Abfrage mit sql.Identifier (verhindert SQL-Injection)
    query = sql.SQL("SELECT * FROM public.{}").format(sql.Identifier(table_name))

    cur.execute(query)
    rows = cur.fetchall()  # Alle Zeilen abrufen
    
    print(f"✅ {len(rows)} rows retrieved from {table_name}")
    for row in rows[:5]:  # Zeige die ersten 5 Zeilen
        print(row)

def query(query_string: str, cursor: psycopg2.extensions.cursor) -> list:

    cursor.execute(query_string)
    conn.commit()

    rows = cursor.fetchall()

    return rows


def create_table(table_name: str, conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor) -> None:
    
    
    cur.execute("SET search_path TO public;") 
    # Verify the table exists
    cur.execute("""
        SELECT table_name FROM information_schema.tables WHERE table_name = '{table_name}'
    """)
    table_exists = cur.fetchone()

    print("->" + str(table_exists))

    if table_exists:
        print("Table exists in the database.")
    else:
        print("Table still does not exist! Check permissions and schema.")

    # SQL-Statement
    sql = f"""
    CREATE TABLE IF NOT EXISTS public.{table_name} (
        "company" VARCHAR(255),
        "fiscal_year" VARCHAR(255),
        "document" VARCHAR(255),
        "line_item" VARCHAR(255),
        "account_number" VARCHAR(255),
        "account_name" VARCHAR(255),
        "posting_date" DATE,
        "recording_date" DATE,
        "document_date" DATE,
        "document_type" VARCHAR(255),
        "user" VARCHAR(255),
        "user_type" VARCHAR(255),
        "local_currency" VARCHAR(10),
        "document_currency" VARCHAR(10),
        "source" VARCHAR(255),
        "posting_text" VARCHAR(255),
        "debit" NUMERIC,
        "credit" NUMERIC,
        "debit_dc" NUMERIC,
        "credit_dc" NUMERIC,
        "manual_automatic_posting" VARCHAR(255),
        "intercompany_partner" VARCHAR(255)
    );
    """

    print(sql)

    # Tabelle erstellen
    cur.execute(sql)
    conn.commit()
    print(f"Table '{table_name}' has been created (or already exists).")


def import_datatable(file_path: str, table_name: str, conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor) -> None:

    # Convert DataFrame to CSV format in memory
    csv_buffer = io.StringIO()
    df = pd.read_csv(file_path, delimiter = ',', header=0, quotechar='"', quoting=1, encoding='utf-8')
    df.to_csv(csv_buffer, index=False, header=False, sep="|")
    csv_buffer.seek(0)

    print(df.columns)

    # Use COPY FROM to insert data into PostgreSQL
    cur.execute("SET search_path TO public;") # Set the schema
    cur.execute(f"TRUNCATE TABLE {table_name};") # Clear the table

    # Use COPY FROM to insert data
    try:
        cur.copy_from(csv_buffer, table_name, sep="|", columns=('company', 
                                                                'fiscal_year', 
                                                                'document', 
                                                                'line_item', 
                                                                'account_number',
                                                                'account_name', 
                                                                'posting_date', 
                                                                'recording_date', 
                                                                'document_date',
                                                                'document_type', 
                                                                'user', 
                                                                'user_type', 
                                                                'local_currency',
                                                                'document_currency', 
                                                                'source', 
                                                                'posting_text', 
                                                                'debit', 
                                                                'credit',
                                                                'debit_dc', 
                                                                'credit_dc', 
                                                                'manual_automatic_posting',
                                                                'intercompany_partner'))
 
        conn.commit()
        print("✅ Data imported successfully!")
 
    except Exception as e:
        conn.rollback()
        print(f"❌ Error during COPY FROM: {e}")

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
table_name = "usr_journal_old_gl"


create_table(table_name, conn, cur)

# Funktionsaufruf mit der gewünschten Tabelle
select_all_from_table(table_name, cur)

import_datatable(file_path, table_name, conn, cur)

# Funktionsaufruf mit der gewünschten Tabelle
select_all_from_table(table_name, cur)

# Cursor und Verbindung schließen
cur.close()
conn.close()



