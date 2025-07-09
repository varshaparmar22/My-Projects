import sqlite3
import os

DB_DIR = 'data'
DB_PATH = os.path.join(DB_DIR, 'anesthetics.db')

def init_db():
    """Initializes the database and creates the anesthetists table."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anesthetists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone_number TEXT,
            specialty TEXT,
            experience_years INTEGER,
            notes TEXT
        )
    ''')

    # Check if the table is empty before populating
    cursor.execute("SELECT COUNT(*) FROM anesthetists")
    if cursor.fetchone()[0] == 0:
        populate_dummy_data(cursor)

    conn.commit()
    conn.close()
    print("Database initialized and populated.")

def populate_dummy_data(cursor):
    """Populates the anesthetists table with dummy data."""
    dummy_data = [
        ('Dr. Alice Smith', 'alice.smith.demo@example.com', '+15550001111', 'Pediatric', 15, 'Prefers daytime shifts. Expert in neonatal care.'),
        ('Dr. Bob Johnson', 'bob.johnson.demo@example.com', '+15550002222', 'Cardiac', 20, 'Extensive experience with high-risk cardiac procedures.'),
        ('Dr. Carol White', 'carol.white.demo@example.com', '+15550003333', 'Neuro', 12, 'Skilled in neuro-anesthesia and spinal surgeries.'),
        ('Dr. David Green', 'david.green.demo@example.com', '+15550004444', 'General', 8, 'Flexible with hours, great all-rounder.'),
        ('Dr. Eve Black', 'eve.black.demo@example.com', '+15550005555', 'Obstetric', 10, 'Specializes in epidurals and C-section anesthesia.'),
        ('Dr. Frank Wright', 'frank.wright.demo@example.com', None, 'Pediatric', 5, 'Recently completed a fellowship in pediatric anesthesia.'),
        ('Dr. Grace Hall', 'grace.hall.demo@example.com', '+15550007777', 'Cardiac', 18, 'Works part-time, available on weekends.')
    ]
    cursor.executemany('''
        INSERT INTO anesthetists (name, email, phone_number, specialty, experience_years, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', dummy_data)
    print(f"{len(dummy_data)} dummy records inserted.")

def get_all_anesthetists():
    """Fetches all anesthetists from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM anesthetists")
    anesthetists = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return anesthetists

if __name__ == '__main__':
    init_db()
    print("\nFetching all anesthetists:")
    all_docs = get_all_anesthetists()
    for doc in all_docs:
        print(doc)
