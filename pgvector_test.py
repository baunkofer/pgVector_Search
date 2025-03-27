# pip install psycopg2-binary

import os
import psycopg2
import io
import pandas as pd
import numpy as np
from ollama_api import OllamaClient
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


def create_document_vector_table(conn: psycopg2.extensions.connection, table_name: str = "documents", vector_dimensions: int = 1536):
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
                content TEXT,
                embedding VECTOR(%s)
            );
        """)
        cur.execute(create_table_query, [vector_dimensions])
        conn.commit()
        print("✅ Tabelle 'documents' wurde erstellt (oder existierte bereits).")


def insert_embedding_to_db(conn, data_table: str, text: str, embedding: list[float]):
    
    with conn.cursor() as cur:
        insert_query = sql.SQL(f"""
            INSERT INTO {data_table} (content, embedding)
            VALUES (%s, %s)
        """)
        cur.execute(insert_query, (text, embedding))
        conn.commit()


def search_similar_documents(conn, client, query_text: str, top_k: int = 5):
    """
    Führt eine semantische Vektor-Suche in der PostgreSQL-Datenbank durch.

    Args:
        conn: Eine psycopg2-Verbindung zu PostgreSQL.
        client: Eine Instanz von OllamaClient (mit Embedding-Funktion).
        query_text (str): Der Text, zu dem ähnliche Dokumente gesucht werden sollen.
        top_k (int): Anzahl der zurückgegebenen ähnlichen Dokumente.

    Returns:
        list[tuple]: Liste von Tupeln mit (content, distance)
    """
    embedding = client.get_embedding(query_text)
    
    if not embedding:
        print("❌ Konnte kein Embedding für den Suchtext erzeugen.")
        return []

    with conn.cursor() as cur:
        query = """
            SELECT content, embedding <-> %s AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT %s;
        """
        cur.execute(query, (embedding, top_k))
        results = cur.fetchall()

    return results


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


#create_table(table_name, conn, cur)

# Funktionsaufruf mit der gewünschten Tabelle
#select_all_from_table(table_name, cur)

#import_datatable(file_path, table_name, conn, cur)

# Funktionsaufruf mit der gewünschten Tabelle
select_all_from_table(table_name, cur)

# Delete table
delete_table("documents", conn, cur)

# Create Document Vector Database Table
create_document_vector_table(conn, table_name="documents", vector_dimensions=3072)


client = OllamaClient(model="llama3.2")  # oder anderes embeddingfähiges Modell



data_table = "documents"
text = "Data Engineering ist eine Kunst."
embedding = client.get_embedding(text)


"""
Die n im VECTOR(n) in deiner Tabelle muss exakt der Länge des Embedding-Vektors entsprechen! Prüfe das z. B. mit:
"""

print("Embedding-Length: {}".format(len(embedding)))  # z. B. 768, 1024, 1536 ...


if embedding:
    insert_embedding_to_db(conn, data_table, text, embedding)
    print("✅ Embedding gespeichert!")
else:
    print("❌ Embedding konnte nicht erzeugt werden.")


suchtext = "Wie funktioniert Data Engineering?"
resultate = search_similar_documents(conn, client, suchtext, top_k=3)

print("\n🔍 Ähnliche Dokumente:")
for i, (content, dist) in enumerate(resultate, start=1):
    print(f"\n📄 Dokument {i}")
    print(f"➡️ Ähnlichkeit (Distanz): {dist:.4f}")
    print(f"📝 Inhalt: {content}")


# Cursor und Verbindung schließen
cur.close()
conn.close()


"""
Vector Operations on pgVector:

<#> = Cosine Distance

<-> = Euclidean Distance

<=> = Inner Product

"""