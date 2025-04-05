# pip install psycopg2-binary

import os
import psycopg
import io
import pandas as pd
import numpy as np
import logging
from ollama_api import OllamaClient
from pandas import DataFrame
from psycopg import Connection
from psycopg.errors import DatabaseError
from pandas import DataFrame
from psycopg import sql
from pgvector.psycopg import register_vector


def select_all_from_table(table_name: str, conn: psycopg.connection):
    
    cur = conn.cursor()
    
    # Sichere Abfrage mit sql.Identifier (verhindert SQL-Injection)
    query = sql.SQL("SELECT * FROM public.{}").format(sql.Identifier(table_name))

    cur.execute(query)

    rows = cur.fetchall()  # Alle Zeilen abrufen
    
    print(f"✅ {len(rows)} rows retrieved from {table_name}")

    for row in rows[:5]:  # Zeige die ersten 5 Zeilen

        print(row)

def query(query_string: str, cursor: psycopg.cursor) -> list:

    cursor.execute(query_string)
    conn.commit()

    rows = cursor.fetchall()

    return rows

def delete_table(table_name: str, conn: psycopg.connection) -> None:
    
    cur = conn.cursor()

    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.commit()
    print(f"Table '{table_name}' has been deleted (or did not exist).") 



def create_table(conn: psycopg.connection, table_name: str) -> None:
    
    cur = conn.cursor()

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
        "intercompany_partner" VARCHAR(255),
        "Posting_Date_new" DATE, 
        "Recording_Date_new" DATE,
        "Document_Date_new" DATE, 
        "year_posting" INTEGER,
        "month_posting" INTEGER,
        "weekday_posting" INTEGER,
        "year_recording" INTEGER,
        "month_recording" INTEGER,
        "weekday_recording" INTEGER,
        "year_document" INTEGER,
        "month_document" INTEGER,
        "weekday_document" INTEGER

    );
    """

    print(sql)

    # Tabelle erstellen
    cur.execute(sql)
    conn.commit()
    print(f"Table '{table_name}' has been created (or already exists).")


def import_datatable(data_table: DataFrame, table_name: str, conn: Connection) -> None:
    logger = logging.getLogger(__name__)
    
    columns = [
        'Company', 'Fiscal_Year', 'Document', 'Line_Item', 'Account_Number', 
        'Account_Name', 'Posting_Date', 'Recording_Date', 'Document_Date', 
        'Document_Type', 'User_Type', 'Local_Currency', 
        'Document_Currency', 'Source', 'Posting_Text', 'Debit', 'Credit', 
        'Debit_DC', 'Credit_DC', 'Manual_Automatic_Posting', 
        'Intercompany_Partner', '"Posting_Date_new"', '"Recording_Date_new"', 
        '"Document_Date_new"', 'year_posting', 'month_posting', 'weekday_posting', 
        'year_recording', 'month_recording', 'weekday_recording', 
        'year_document', 'month_document', 'weekday_document'
    ]
    
    # Sicherheitsprüfung auf Spaltenvollständigkeit
    missing_cols = [col for col in columns if col not in data_table.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data_table: {missing_cols}")
    
    # Tabellenname prüfen – einfache Absicherung gegen Missbrauch
    if not table_name.isidentifier():
        raise ValueError(f"Invalid table name: {table_name}")
    
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    
    data = [tuple(row[col] for col in columns) for row in data_table.to_dict(orient='records')]
    
    try:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, data)
        conn.commit()
        logger.info("✅ Data inserted into table '%s' successfully.", table_name)
    
    except DatabaseError as e:
        conn.rollback()
        logger.error("❌ Database error during import: %s", e)

def create_document_vector_table(conn: psycopg.connection, table_name: str = "vector_store", amount_columns: int = 1, vector_dimensions: int = 1536):
    """
    Erstellt die Tabelle 'documents' mit pgvector-Spalte, falls sie nicht existiert.

    Args:
        conn: Eine offene psycopg2-Verbindung zu PostgreSQL.
        vector_dimensions (int): Die Länge des Embedding-Vektors (z. B. 768, 1024, 1536).
    """
    with conn.cursor() as cur:
        # Stelle sicher, dass die Extension pgvector existiert
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        create_table_query = sql.SQL(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                embedding VECTOR({amount_columns})  -- entsprechend der Anzahl Features
            );
            
        """)
        cur.execute(create_table_query)
        #cur.execute(create_table_query, [vector_dimensions])
        conn.commit()
        print("✅ Tabelle 'documents' wurde erstellt (oder existierte bereits).")


def insert_vectors_to_db(dataset: DataFrame, vectors: np.ndarray, conn: psycopg.connection, table_name: str = "vector_store", create_index = False, debug = False) -> None:
    cur = conn.cursor()

    for i, row in dataset[features].iterrows():
        # Vektor für die aktuelle Zeile abrufen
        vector = vectors[i]
        vector_str = ','.join(map(str, vector))
        
        # Das Statement vorbereiten
        sql = "INSERT INTO vector_store (id, embedding) VALUES (%s, %s)"
        values = (i, f'[{vector_str}]')

        if debug:
            # Vor dem Ausführen ausgeben
            print(cur.mogrify(sql, values).decode('utf-8'))  # Zeigt das finale SQL-Statement

        # Ausführen
        cur.execute(sql, values)

        if create_index:
            # Index erstellen
            sql = "CREATE INDEX ON vector_store USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);"

            cur.execute(sql)

    conn.commit()


def detect_anomalies(conn, table_name = "vector_store", limit = 10000) -> list:
    """
    Detect anomalies in the dataset based on the given threshold.
    Args:
        vectors (np.ndarray): The dataset of vectors.
        threshold (float): The threshold for anomaly detection.
    Returns:
        list: Indices of detected anomalies.
    """
    # Referenz: z. B. Mittelwert aller Vektoren
    mean_vector = vectors.mean(axis=0)
    mean_vector_str = ','.join(map(str, mean_vector))

    cur.execute("""
        SELECT id, embedding <-> %s AS dist
        FROM vector_store
        WHERE embedding IS NOT NULL
        ORDER BY dist DESC
        LIMIT %s
    """, (f'[{mean_vector_str}]', limit))

    anomalies = cur.fetchall()

    return anomalies

# Verbindung zur PostgreSQL-Datenbank
conn = psycopg.connect(
    dbname=os.environ.get("DATABASE"),
    user=os.environ.get("USER"),
    password=os.environ.get("PASSWORD"),
    host="localhost",
    port="5432"
)

# Cursor erstellen
cur = conn.cursor()
register_vector(conn)

type(cur)

#result = query("SELECT * FROM pg_database", cur)
#print(result)


file_path = "/home/aunkofer/aunkofer/SAP/SAP_ECC_old_GL_Ergebnisse/result_usr_journal_old_GL.csv"
table_name = "usr_journal_old_gl"


dataset = pd.read_csv(file_path, delimiter = ',', header=0, quotechar='"', quoting=1, encoding='utf-8')

# Angenommen, df['document_date'] sieht so aus: "2000-04-17"
dataset['Posting_Date_new'] = pd.to_datetime(dataset['Posting_Date'], format='%Y-%m-%d')
dataset['Recording_Date_new'] = pd.to_datetime(dataset['Recording_Date'], format='%Y-%m-%d')
dataset['Document_Date_new'] = pd.to_datetime(dataset['Document_Date'], format='%Y-%m-%d')

dataset['year_posting'] = dataset['Posting_Date_new'].dt.year
dataset['month_posting'] = dataset['Posting_Date_new'].dt.month
dataset['weekday_posting'] = dataset['Posting_Date_new'].dt.weekday  # 0 = Montag, 6 = Sonntag

dataset['year_recording'] = dataset['Recording_Date_new'].dt.year
dataset['month_recording'] = dataset['Recording_Date_new'].dt.month
dataset['weekday_recording'] = dataset['Recording_Date_new'].dt.weekday  # 0 = Montag, 6 = Sonntag

dataset['year_document'] = dataset['Document_Date_new'].dt.year
dataset['month_document'] = dataset['Document_Date_new'].dt.month
dataset['weekday_document'] = dataset['Document_Date_new'].dt.weekday  # 0 = Montag, 6 = Sonntag

dataset.rename(columns={"DocumentCurrency": "Document_Currency"}, inplace=True)

delete_table(table_name, conn)

create_table(conn, table_name)

import_datatable(dataset, table_name, conn)

# Funktionsaufruf mit der gewünschten Tabelle
select_all_from_table(table_name, conn)

# Delete table
delete_table("vector_store", conn)

columns = ['Company', 'Fiscal_Year', 'Document', 'Line_Item', 'Account_Number',
            'Account_Name', 'Posting_Date', 'Recording_Date', 'Document_Date',
            'Document_Type', 'User', 'User_Type', 'Local_Currency',
            'DocumentCurrency', 'Source', 'Posting_Text', 'Debit', 'Credit',
            'Debit_DC', 'Credit_DC', 'Manual_Automatic_Posting',
            'Intercompany_Partner']

"""
for column in dataset.columns:

    if dataset[column].dtype == np.str_:
        dataset[column] = dataset[column].astype('category').cat.codes
"""

ollama = OllamaClient(model="llama3.2")

# Spalten für Vektor auswählen
features_normal_vectors = ['Company', 'Fiscal_Year', 'Document', 'Line_Item', 'Account_Number', 'year_document', 'year_posting', 'month_posting', 'weekday_posting','Debit', 'Credit', 'Debit_DC', 'Credit_DC']
features_embeddings = [] # ['Posting_Text_embedding']
features = features_normal_vectors + features_embeddings

"""
posting_text_embedding = ollama.embed_dataframe(dataset, text_column='Posting_Text', embedding_column='Posting_Text_embedding')

dataset['Posting_Text_embedding'] = posting_text_embedding['Posting_Text_embedding']
"""

vectors = dataset[features_normal_vectors].fillna(0).to_numpy().astype(np.float64)

"""
create_document_vector_table(conn, table_name="vector_store", amount_columns=len(features), vector_dimensions = vectors)

insert_vectors_to_db(dataset, vectors, conn, table_name="vector_store", debug=False)

anomalies = detect_anomalies(conn, table_name="vector_store", limit=10)


print("Anomalies:")
for anomaly in anomalies:
    print(f"ID: {anomaly[0]}, Distanz: {anomaly[1]}")

print(dataset.columns)

# Cursor und Verbindung schließen
cur.close()
conn.close()


"""

"""
Vector Operations on pgVector:

<#> = Cosine Distance

<-> = Euclidean Distance

<=> = Inner Product

"""