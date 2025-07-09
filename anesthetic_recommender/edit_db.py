import sqlite3

conn = sqlite3.connect('data/anesthetics.db')
cursor = conn.cursor()

# Edit record
cursor.execute("UPDATE anesthetists SET phone_number = ? WHERE id = ?", ("+918238004064", "1"))

# Delete record
#cursor.execute("DELETE FROM users WHERE name = ?", ("Bob",))
print('here')
conn.commit()
conn.close()
