# pip install psycopg2-binary

import os
import psycopg2
import io
import pandas as pd
import numpy as np
from ollama_api import OllamaClient
from pandas import DataFrame
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

def delete_table(table_name: str, conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor) -> None:
    
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.commit()
    print(f"Table '{table_name}' has been deleted (or did not exist).") 



def create_table(conn: psycopg2.extensions.connection, table_name: str, cur: psycopg2.extensions.cursor) -> None:
    
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


def import_datatable(data_table: DataFrame, table_name: str, conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor) -> None:

    # Convert DataFrame to CSV format in memory
    csv_buffer = io.StringIO()
    
    data_table.to_csv(csv_buffer, index=False, header=False, sep="|")
    csv_buffer.seek(0)

    print(data_table.columns)

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


def create_document_vector_table(conn: psycopg2.extensions.connection, table_name: str = "vector_store", amount_columns: int = 1, vector_dimensions: int = 1536):
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
        cur.execute(create_table_query, [vector_dimensions])
        conn.commit()
        print("✅ Tabelle 'documents' wurde erstellt (oder existierte bereits).")


def insert_vectors_to_db(dataset: DataFrame, vectors: np.ndarray, conn: psycopg2.extensions.connection, table_name: str = "vector_store", create_index = False, debug = False) -> None:
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


dataset = pd.read_csv(file_path, delimiter = ',', header=0, quotechar='"', quoting=1, encoding='utf-8')

#dataset = dataset.iloc[:300, :]

delete_table(table_name, conn, cur)

create_table(conn, table_name, cur)

import_datatable(dataset, table_name, conn, cur)

# Funktionsaufruf mit der gewünschten Tabelle
select_all_from_table(table_name, cur)

# Delete table
delete_table("vector_store", conn, cur)

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
features_normal_vectors = ['Company', 'Fiscal_Year', 'Document', 'Line_Item', 'Account_Number']
features_embeddings = [] # ['Posting_Text_embedding']
features = features_normal_vectors + features_embeddings

"""
posting_text_embedding = ollama.embed_dataframe(dataset, text_column='Posting_Text', embedding_column='Posting_Text_embedding')

dataset['Posting_Text_embedding'] = posting_text_embedding['Posting_Text_embedding']
"""

vectors = dataset[features_normal_vectors].fillna(0).to_numpy().astype(np.float64)

create_document_vector_table(conn, table_name="vector_store", amount_columns=len(features), vector_dimensions = vectors)

insert_vectors_to_db(dataset, vectors, conn, table_name="vector_store", debug=False)

anomalies = detect_anomalies(conn, table_name="vector_store")


print("Anomalies:")
for anomaly in anomalies:
    print(f"ID: {anomaly[0]}, Distanz: {anomaly[1]}")

print(dataset.columns)

# Cursor und Verbindung schließen
cur.close()
conn.close()


"""
Vector Operations on pgVector:

<#> = Cosine Distance

<-> = Euclidean Distance

<=> = Inner Product

"""